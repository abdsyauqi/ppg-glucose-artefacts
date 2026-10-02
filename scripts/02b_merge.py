"""Merge stage outputs of 02_within_dataset.py into one JSON per dataset/subset."""
import os, json
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
for ds in ['must', 'stuba']:
    for tag in ['all_records', 'qc_pass_only']:
        R, ok = {}, True
        for st in '1234':
            fn = f'{ROOT}/results/{ds}_{tag}_{st}.json'
            if not os.path.exists(fn): ok = False; break
            R.update(json.load(open(fn)))
        if ok:
            json.dump(R, open(f'{ROOT}/results/{ds}_{tag}_all.json', 'w'), indent=1, default=float)
            print('merged', ds, tag)
