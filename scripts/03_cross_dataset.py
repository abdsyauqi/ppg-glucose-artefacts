import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import json, copy
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

M = pd.read_csv(f'{ROOT}/features/must_features.csv'); M['dataset'] = 'MUST'; M['fs'] = 2175
S = pd.read_csv(f'{ROOT}/features/stuba_features.csv'); S['dataset'] = 'STU-BA'; S['fs'] = 100
# AUC features were integrated in samples -> convert to seconds so both datasets share units
for d in (M, S):
    d['auc_sys'] = d['auc_sys'] / d['fs']; d['auc_dias'] = d['auc_dias'] / d['fs']
MORPH = ['hr_bpm', 'ibi_mean', 'ibi_std', 'ibi_rmssd', 'sys_amp', 'dias_amp', 'dias_ratio', 'pulse_width',
         'rise_time', 'fall_time', 'auc_sys', 'auc_dias', 'auc_ratio', 'skewness', 'kurtosis_', 'spectral_ratio']
SETS = {'A': ['hr_bpm', 'sys_amp'], 'B': MORPH, 'C': MORPH + ['age', 'bmi', 'gender_enc']}
MODELS = {'LR': LinearRegression(), 'Ridge': Ridge(alpha=1.0),
          'RF': RandomForestRegressor(n_estimators=100, random_state=42),
          'GB': GradientBoostingRegressor(n_estimators=100, random_state=42),
          'SVR': SVR(kernel='rbf', C=10, epsilon=0.5)}
rng = np.random.default_rng(42)


def clarke_ab(ref, pred):
    ab = 0
    for r, p in zip(ref, pred):
        if (r <= 70 and p <= 70) or (0.8 * r <= p <= 1.2 * r): ab += 1; continue
        if (r >= 180 and p <= 70) or (r <= 70 and p >= 180): continue
        if ((70 <= r <= 290) and p >= r + 110) or ((130 <= r <= 180) and p <= (7 / 5) * r - 182): continue
        if (r >= 240 and 70 <= p <= 180) or (r <= 175 / 3 and 70 <= p <= 180) or ((175 / 3 <= r <= 70) and p >= (6 / 5) * r): continue
        ab += 1
    return 100 * ab / len(ref)


def subj_boot_ci(yt, yp, sid, B=1000):
    df = pd.DataFrame({'s': sid, 'e': np.abs(np.asarray(yt) - np.asarray(yp))})
    subs = df.s.unique(); g = {s: df.e[df.s == s].values for s in subs}; vals = []
    for _ in range(B):
        smp = rng.choice(subs, len(subs), replace=True)
        vals.append(np.concatenate([g[s] for s in smp]).mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summ(yt, yp, sid):
    yt, yp = np.asarray(yt), np.asarray(yp)
    e = pd.DataFrame({'s': sid, 'e': np.abs(yt - yp)})
    return dict(n=len(yt), mae=mean_absolute_error(yt, yp), mae_ci=subj_boot_ci(yt, yp, sid),
                mae_subject_weighted=float(e.groupby('s').e.mean().mean()),
                rmse=float(np.sqrt(mean_squared_error(yt, yp))), r2=r2_score(yt, yp), clarke_AB=clarke_ab(yt, yp))


OUT = {}
# ---------- cross-dataset transfer ----------
cross = {}
for tr_name, TR, te_name, TE in [('MUST', M, 'STU-BA', S), ('STU-BA', S, 'MUST', M)]:
    for sk, fs in SETS.items():
        for mk, mdl in MODELS.items():
            for norm in ['pooled_scaler', 'per_dataset_z']:
                Xtr, Xte = TR[fs].values.astype(float), TE[fs].values.astype(float)
                if norm == 'per_dataset_z':
                    Xtr = StandardScaler().fit_transform(Xtr); Xte = StandardScaler().fit_transform(Xte)
                else:
                    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
                m = copy.deepcopy(mdl).fit(Xtr, TR.bgl.values)
                cross[f'{tr_name}->{te_name}|{sk}|{mk}|{norm}'] = summ(TE.bgl, m.predict(Xte), TE.subject_id)
        cross[f'{tr_name}->{te_name}|BASELINE_train_mean'] = summ(TE.bgl, np.full(len(TE), TR.bgl.mean()), TE.subject_id)
OUT['cross'] = cross

# ---------- pooled LOSO over 47 subjects ----------
P = pd.concat([M.assign(gid='M' + M.subject_id.astype(str)), S.assign(gid='S' + S.subject_id.astype(str))],
              ignore_index=True)
P['is_stuba'] = (P.dataset == 'STU-BA').astype(int)
pooled = {}
g = P.gid.values; y = P.bgl.values
for sk, fs in SETS.items():
    fs2 = fs + ['is_stuba']
    for mk, mdl in MODELS.items():
        yt, yp, ss_ = [], [], []
        for tr, te in LeaveOneGroupOut().split(P, groups=g):
            sc = StandardScaler().fit(P.loc[tr, fs2])
            m = copy.deepcopy(mdl).fit(sc.transform(P.loc[tr, fs2]), y[tr])
            yt += list(y[te]); yp += list(m.predict(sc.transform(P.loc[te, fs2]))); ss_ += list(g[te])
        pooled[f'{sk}|{mk}'] = summ(yt, yp, ss_)
yt, yp, ss_ = [], [], []
for tr, te in LeaveOneGroupOut().split(P, groups=g):
    yt += list(y[te]); yp += [y[tr].mean()] * len(te); ss_ += list(g[te])
pooled['BASELINE_train_mean'] = summ(yt, yp, ss_)
OUT['pooled47'] = pooled
json.dump(OUT, open(f'{ROOT}/results/cross_results.json', 'w'), indent=1, default=float)

for k, v in sorted(cross.items(), key=lambda kv: kv[1]['mae'])[:6]:
    print(f"{k:45s} MAE={v['mae']:.2f} {np.round(v['mae_ci'],1)} R2={v['r2']:+.3f} AB={v['clarke_AB']:.1f}")
for k in ['MUST->STU-BA|BASELINE_train_mean', 'STU-BA->MUST|BASELINE_train_mean']:
    v = cross[k]; print(f"{k:45s} MAE={v['mae']:.2f} R2={v['r2']:+.3f} AB={v['clarke_AB']:.1f}")
for d in ['MUST->STU-BA', 'STU-BA->MUST']:
    best = min([(k, v) for k, v in cross.items() if k.startswith(d) and 'BASELINE' not in k], key=lambda kv: kv[1]['mae'])
    print('best', best[0], round(best[1]['mae'], 2), round(best[1]['r2'], 3))
print('--- pooled 47')
for k, v in sorted(pooled.items(), key=lambda kv: kv[1]['mae'])[:5]:
    print(f"{k:25s} MAE={v['mae']:.2f} {np.round(v['mae_ci'],1)} R2={v['r2']:+.3f} AB={v['clarke_AB']:.1f}")
v = pooled['BASELINE_train_mean']; print('baseline', round(v['mae'], 2), np.round(v['mae_ci'], 1), round(v['r2'], 3), round(v['clarke_AB'], 1))
