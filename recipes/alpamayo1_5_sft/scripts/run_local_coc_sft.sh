#!/usr/bin/env bash
# Launch Alpamayo-1.5 Stage-1 SFT on locally exported CoC labels.
#
# Required environment variables:
#   LOCAL_COC_SAMPLE_ROOT       Root containing <group>/<record>/sample data.
#   LOCAL_COC_TRAIN_VERSIONS    Comma-separated formal-label version directories.
#   LOCAL_COC_VAL_VERSIONS      Comma-separated formal-label version directories.
#   ALPAMAYO_A1_CHECKPOINT      Converted Alpamayo-1.5 A1-format checkpoint.
#
# Optional:
#   NPROC_PER_NODE               Number of GPUs (default: 8).
#   LOCAL_COC_OUTPUT_DIR         Training output (default: ./output_stage1_local_coc).
#   LOCAL_COC_MAX_STEPS          Override max_steps; -1 means epoch-based training.
#   LOCAL_COC_NUM_EPOCHS         Override num_train_epochs (default comes from config).
#   LOCAL_COC_DATALOADER_WORKERS Override dataloader_num_workers.
#
# Example:
#   LOCAL_COC_SAMPLE_ROOT=/data/vla_kl/sample \
#   LOCAL_COC_TRAIN_VERSIONS=/data/formal/v001,/data/formal/v002 \
#   LOCAL_COC_VAL_VERSIONS=/data/formal/v003 \
#   ALPAMAYO_A1_CHECKPOINT=/models/Alpamayo-1.5-10B-A1-format \
#   NPROC_PER_NODE=8 \
#   ./scripts/run_local_coc_sft.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
recipe_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${recipe_dir}"

required_vars=(
  LOCAL_COC_SAMPLE_ROOT
  LOCAL_COC_TRAIN_VERSIONS
  LOCAL_COC_VAL_VERSIONS
  ALPAMAYO_A1_CHECKPOINT
)
for variable in "${required_vars[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing required environment variable: ${variable}" >&2
    exit 2
  fi
done

for path in "${LOCAL_COC_SAMPLE_ROOT}" "${ALPAMAYO_A1_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Path does not exist: ${path}" >&2
    exit 2
  fi
done
if [[ ! -f "${ALPAMAYO_A1_CHECKPOINT}/config.json" ]]; then
  echo "Checkpoint is missing config.json (expected an A1-format directory): ${ALPAMAYO_A1_CHECKPOINT}" >&2
  exit 2
fi

IFS=',' read -r -a train_versions <<< "${LOCAL_COC_TRAIN_VERSIONS}"
IFS=',' read -r -a val_versions <<< "${LOCAL_COC_VAL_VERSIONS}"
if [[ "${#train_versions[@]}" -eq 0 || "${#val_versions[@]}" -eq 0 ]]; then
  echo "At least one train and one validation formal-label version are required." >&2
  exit 2
fi

for version in "${train_versions[@]}" "${val_versions[@]}"; do
  if [[ ! -f "${version}/manifest.json" || ! -f "${version}/annotations.jsonl" ]]; then
    echo "Formal-label version is missing manifest.json or annotations.jsonl: ${version}" >&2
    exit 2
  fi
done

hydra_list() {
  local result="["
  local path
  local first=1
  for path in "$@"; do
    if [[ "${first}" -eq 0 ]]; then
      result+=","
    fi
    result+="${path}"
    first=0
  done
  result+="]"
  printf '%s' "${result}"
}

train_override="$(hydra_list "${train_versions[@]}")"
val_override="$(hydra_list "${val_versions[@]}")"

output_dir="${LOCAL_COC_OUTPUT_DIR:-${recipe_dir}/output_stage1_local_coc}"
nproc="${NPROC_PER_NODE:-8}"
overrides=(
  "model.checkpoint_path=${ALPAMAYO_A1_CHECKPOINT}"
  "data.train_dataset.sample_root=${LOCAL_COC_SAMPLE_ROOT}"
  "data.val_dataset.sample_root=${LOCAL_COC_SAMPLE_ROOT}"
  "data.train_dataset.formal_versions=${train_override}"
  "data.val_dataset.formal_versions=${val_override}"
  "paths.output_dir=${output_dir}"
)

if [[ -n "${LOCAL_COC_MAX_STEPS:-}" ]]; then
  overrides+=("trainer.max_steps=${LOCAL_COC_MAX_STEPS}")
fi
if [[ -n "${LOCAL_COC_NUM_EPOCHS:-}" ]]; then
  overrides+=("trainer.num_train_epochs=${LOCAL_COC_NUM_EPOCHS}")
fi
if [[ -n "${LOCAL_COC_DATALOADER_WORKERS:-}" ]]; then
  overrides+=("trainer.dataloader_num_workers=${LOCAL_COC_DATALOADER_WORKERS}")
fi

echo "Launching Alpamayo-1.5 local CoC SFT"
echo "  recipe:    ${recipe_dir}"
echo "  GPUs:      ${nproc}"
echo "  samples:   ${LOCAL_COC_SAMPLE_ROOT}"
echo "  output:    ${output_dir}"
echo "  train sets: ${#train_versions[@]}"
echo "  val sets:   ${#val_versions[@]}"

command=(
  torchrun "--nproc_per_node=${nproc}"
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_local_coc \
  "${overrides[@]}"
)

if [[ "${LOCAL_COC_DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

"${command[@]}"
