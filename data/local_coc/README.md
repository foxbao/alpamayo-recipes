# Local CoC data entry points

This directory is a stable path layout for local Alpamayo-1.5 CoC training.
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
mkdir -p data/local_coc
ln -s /mnt/disk3/vla_kl/sample data/local_coc/sample
ln -s /path/to/alpamayo-coc-autolabeler/outputs/dataset_workflow/guided_runs/vla_kl/20261019_lin_007_out/20261110_181035/screening_policy/policy_v010/formal_labels/versions/v002 data/local_coc/train_labels
ln -s /path/to/alpamayo-coc-autolabeler/outputs/dataset_workflow/guided_runs/vla_kl/20261111_BB_008_out/20261112_171400/screening_policy/policy_v001/formal_labels/versions/v001 data/local_coc/val_labels
ln -s /mnt/disk1/models/Alpamayo-1.5-10B-A1-format data/local_coc/checkpoint
```

The symlink targets are intentionally ignored by Git. On another machine,
recreate the same names with that machine's mount points. Check the layout with:

```bash
find -L data/local_coc -maxdepth 2 -type d -print
test -f data/local_coc/train_labels/manifest.json
test -f data/local_coc/val_labels/manifest.json
test -f data/local_coc/checkpoint/config.json
```

The training launcher can then use these stable paths:

```bash
export LOCAL_COC_SAMPLE_ROOT="$PWD/data/local_coc/sample"
export LOCAL_COC_TRAIN_VERSIONS="$PWD/data/local_coc/train_labels"
export LOCAL_COC_VAL_VERSIONS="$PWD/data/local_coc/val_labels"
export ALPAMAYO_A1_CHECKPOINT="$PWD/data/local_coc/checkpoint"
```
