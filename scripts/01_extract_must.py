"""Extract the 16 features from MUST (Kermani & Esmaeili 2023, doi:10.17632/37pm7jk7jn.3).
Expects data/MUST/RawData/signal_<id>.mat and data/MUST/Labels/label_<id>.csv."""
import os, sys, csv, re
import numpy as np, pandas as pd, scipy.io as sio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ppg_features import extract_features

ROOT = os.path.join(os.path.dirname(__file__), '..')
RAW, LAB, FS = f'{ROOT}/data/MUST/RawData', f'{ROOT}/data/MUST/Labels', 2175
rows = []
for fname in sorted(os.listdir(LAB)):
    if not fname.endswith('.csv'):
        continue
    sid = re.sub(r'(?i)^label_', '', fname[:-4])   # files are named label_XX_XXXX.csv
    if not re.fullmatch(r'\d+_\d+', sid):   # skip aggregate files such as Label_Total.csv
        continue
    with open(f'{LAB}/{fname}') as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) < 4:
                continue
            h = float(row[4]) if len(row) > 4 else np.nan
            w = float(row[5]) if len(row) > 5 else np.nan
            mat = sio.loadmat(f'{RAW}/signal_{sid}.mat')
            sig = mat[[k for k in mat if not k.startswith('_')][0]].flatten().astype(float)
            feats = extract_features(sig, FS)
            if feats is None:
                continue
            feats.update(signal_id=sid, subject_id=int(row[0].strip()), bgl=float(row[3]),
                         age=float(row[2]), bmi=w / (h / 100) ** 2 if h and w else np.nan,
                         gender_enc=1 if row[1].strip().upper().startswith('M') else 0,
                         record_idx=int(sid.split('_')[1]), record_passes=True, t1dm=0)
            rows.append(feats)
df = pd.DataFrame(rows)
df.to_csv(f'{ROOT}/features/must_features.csv', index=False)
print(df.shape)
