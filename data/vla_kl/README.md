# VLA-KL data entry points

This directory is a stable path layout for VLA-KL Alpamayo-1.5 CoC training.
The large files stay outside Git; create the four machine-local symlinks below
on each host:

```text
sample/       -> <raw sample root>
train_labels/ -> <formal-label version used for training>
val_labels/   -> <formal-label version used for validation>
checkpoint/   -> <converted Alpamayo-1.5 A1-format checkpoint>
```

For example, on the development host:

```bash
cd /path/to/alpamayo-recipes
mkdir -p data/vla_kl
ln -s /mnt/disk3/vla_kl/sample data/vla_kl/sample
ln -s /path/to/alpamayo-coc-autolabeler/outputs/dataset_workflow/guided_runs/vla_kl/20261019_lin_007_out/20261110_181035/screening_policy/policy_v010/formal_labels/versions/v002 data/vla_kl/train_labels
ln -s /path/to/alpamayo-coc-autolabeler/outputs/dataset_workflow/guided_runs/vla_kl/20261111_BB_008_out/20261112_171400/screening_policy/policy_v001/formal_labels/versions/v001 data/vla_kl/val_labels
ln -s /mnt/disk1/models/Alpamayo-1.5-10B-A1-format data/vla_kl/checkpoint
```

The symlink targets are intentionally ignored by Git. On another machine,
recreate the same names with that machine's mount points. Check the layout with:

```bash
find -L data/vla_kl -maxdepth 2 -type d -print
test -f data/vla_kl/train_labels/manifest.json
test -f data/vla_kl/val_labels/manifest.json
test -f data/vla_kl/checkpoint/config.json
```

The training launcher can then use these stable paths:

```bash
export LOCAL_COC_SAMPLE_ROOT="$PWD/data/vla_kl/sample"
export LOCAL_COC_TRAIN_VERSIONS="$PWD/data/vla_kl/train_labels"
export LOCAL_COC_VAL_VERSIONS="$PWD/data/vla_kl/val_labels"
export ALPAMAYO_A1_CHECKPOINT="$PWD/data/vla_kl/checkpoint"
```
