"""Extract the 16 features from STU-BA (Berki & Mateasik 2026, doi:10.5281/zenodo.22206256).
Expects data/STU-BA/wfdb_data/ and data/STU-BA/qc_channels.csv, qc_records.csv."""
import os, sys, glob, re
import numpy as np, pandas as pd, wfdb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ppg_features import extract_features

ROOT = os.path.join(os.path.dirname(__file__), '..')
D = f'{ROOT}/data/STU-BA'
ROOT_W = f'{D}/wfdb_data'
qc_ch = pd.read_csv(f'{D}/qc_channels.csv')
qc_rec = pd.read_csv(f'{D}/qc_records.csv')
subj = pd.read_csv(f'{ROOT_W}/subjects.csv')
subj['bmi'] = subj.weight / (subj.height / 100) ** 2

rows, failed = [], []
for hea in sorted(glob.glob(f'{ROOT_W}/Subject_*/*.hea')):
    rec_path = hea[:-4]
    sname, rname = rec_path.split('/')[-2], rec_path.split('/')[-1]
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')
    m = re.search(r'([\d.]+)\s*mmol', ann.aux_note[0])
    bgl = float(m.group(1)) * 18.016
    # choose green detector with best QC
    q = qc_ch[(qc_ch.record == rname) & (qc_ch.subject == sname) & (qc_ch.channel.isin(['G1', 'G2']))]
    q = q.sort_values(['passes', 'frac_windows_pass'], ascending=False)
    ch = q.channel.iloc[0]
    ch_pass = bool(q.passes.iloc[0])
    sig = rec.p_signal[:, rec.sig_name.index(ch)]
    if np.isnan(sig).any():
        sig = pd.Series(sig).interpolate().bfill().ffill().values
    f = extract_features(sig, 100, auto_polarity=True)
    if f is None:
        failed.append(rname)
        continue
    s = subj[subj.name == sname].iloc[0]
    qr = qc_rec[(qc_rec.record == rname) & (qc_rec.subject == sname)].iloc[0]
    f.update(dict(signal_id=f'{sname}/{rname}', subject_id=int(s.id), record_idx=int(rname[-6:]),
                  bgl=bgl, age=s.age, bmi=s.bmi, gender_enc=1 if s.sex == 'M' else 0,
                  t1dm=1 if s.diabetes == 'Yes' else 0, channel=ch, channel_pass=ch_pass,
                  record_passes=bool(qr.record_passes)))
    rows.append(f)

df = pd.DataFrame(rows)
df.to_csv(f'{ROOT}/features/stuba_features.csv', index=False)
print('extracted', len(df), 'failed', failed)
print('subjects', df.subject_id.nunique(), ' records passing QC', df.record_passes.sum())
print(df[['bgl']].describe().T)
print(df.groupby('t1dm').bgl.describe())
print(df.isnull().sum()[df.isnull().sum() > 0])
print(df.hr_bpm.describe())
