#!/bin/sh
# Full reproduction. Steps 1a/1b need the raw data (see data/README.md);
# all later steps run from the feature tables in features/.
set -e
cd "$(dirname "$0")/scripts"
# 1. feature extraction (optional; features/ already contains the outputs)
[ -d ../data/MUST/RawData ]       && python 01_extract_must.py
[ -d ../data/STU-BA/wfdb_data ]   && python 01_extract_stuba.py
# 2. within-dataset analyses (stage 4 takes several minutes)
for ds in must stuba; do
  for st in 1 2 3 4; do python 02_within_dataset.py $st all_records $ds; done
done
for st in 1 2 3 4; do python 02_within_dataset.py $st qc_pass_only stuba; done
python 02b_merge.py
# 3-6. cross-dataset, subgroups, figures, tables
python 03_cross_dataset.py
python 04_subgroup.py
for f in 2 3 4 5 6 7; do python 05_figures.py $f; done
python 06_tables.py
