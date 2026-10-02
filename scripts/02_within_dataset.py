"""Within-dataset analyses. Usage: python 02_within_dataset.py <stage 1-4> <all_records|qc_pass_only> <must|stuba>
1: pooled vs repeated-measures correlation, OLS commonality, adjusted R2, ICC/mixed model + bootstrap CI
2: LOSO, 3 feature sets x 5 regressors + training-mean baseline
3: within-subject LOO vs calibration-only baselines; temporally held-out (subjects with >=20 sessions)
4: in-sample vs honest best |r| with permutation nulls; feature stability vs empirical null
"""
import warnings; warnings.filterwarnings('ignore')
import copy, json
import numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import pearsonr, wilcoxon
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.formula.api as smf
import pingouin as pg
import sys

OUT = {}
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DS = sys.argv[3] if len(sys.argv) > 3 else 'stuba'
df_all = pd.read_csv(f'{ROOT}/features/{DS}_features.csv')
MORPH = ['hr_bpm', 'ibi_mean', 'ibi_std', 'ibi_rmssd', 'sys_amp', 'dias_amp', 'dias_ratio', 'pulse_width',
         'rise_time', 'fall_time', 'auc_sys', 'auc_dias', 'auc_ratio', 'skewness', 'kurtosis_', 'spectral_ratio']
SET_A = ['hr_bpm', 'sys_amp']; SET_B = MORPH; SET_C = MORPH + ['age', 'bmi', 'gender_enc']
MODELS = {'LinearRegression': LinearRegression(), 'Ridge': Ridge(alpha=1.0),
          'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
          'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
          'SVR': SVR(kernel='rbf', C=10, epsilon=0.5)}
rng = np.random.default_rng(42)


def clarke_zones(ref, pred):
    z = []
    for r, p in zip(ref, pred):
        if (r <= 70 and p <= 70) or (0.8 * r <= p <= 1.2 * r):
            z.append('A')
        elif (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
            z.append('E')
        elif ((70 <= r <= 290) and p >= r + 110) or ((130 <= r <= 180) and p <= (7 / 5) * r - 182):
            z.append('C')
        elif (r >= 240 and 70 <= p <= 180) or (r <= 175 / 3 and 70 <= p <= 180) or \
                ((175 / 3 <= r <= 70) and p >= (6 / 5) * r):
            z.append('D')
        else:
            z.append('B')
    z = np.array(z)
    return {k: float((z == k).mean() * 100) for k in 'ABCDE'}


def metrics(yt, yp, sid=None):
    yt, yp = np.asarray(yt), np.asarray(yp)
    m = dict(n=len(yt), mae=mean_absolute_error(yt, yp), rmse=float(np.sqrt(mean_squared_error(yt, yp))),
             r2=r2_score(yt, yp))
    if sid is not None:
        e = pd.DataFrame({'s': sid, 'e': np.abs(yt - yp)})
        m['mae_subject_weighted'] = float(e.groupby('s').e.mean().mean())
    cz = clarke_zones(yt, yp); m['clarke_A'] = cz['A']; m['clarke_AB'] = cz['A'] + cz['B']
    m['clarke'] = cz
    return m


STAGE, TAG = sys.argv[1], sys.argv[2]
import os
df = df_all if TAG=='all_records' else df_all[df_all.record_passes]
def prep(df):
    R = {}
    d = df.dropna(subset=SET_C + ['bgl']).reset_index(drop=True)
    R['n_records'] = len(d); R['n_subjects'] = int(d.subject_id.nunique())
    R['bgl_mgdl'] = dict(min=d.bgl.min(), max=d.bgl.max(), mean=d.bgl.mean(), sd=d.bgl.std())
    R['records_per_subject'] = d.groupby('subject_id').size().to_dict()

    return R, d
R, d = prep(df)
def stage1(R, d):
    y = d.bgl.values
    # ---------- G1 pooled Pearson vs rmcorr ----------
    rows = []
    for f in MORPH:
        r0, p0 = pearsonr(d[f], d.bgl)
        rm = pg.rm_corr(data=d, x=f, y='bgl', subject='subject_id')
        rows.append(dict(feature=f, r_pooled=r0, p_pooled=p0, r_rm=rm['r'].iloc[0], p_rm=rm['pval'].iloc[0],
                         ci_rm=list(rm['CI95'].iloc[0])))
    R['corr'] = sorted(rows, key=lambda x: x['p_rm'])

    # ---------- variance decomposition OLS (as manuscript) ----------
    y = d.bgl.values; SS = np.sum((y - y.mean()) ** 2)
    Xs = StandardScaler().fit_transform(OneHotEncoder(sparse_output=False, drop='first').fit_transform(d[['subject_id']]))
    Xp = StandardScaler().fit_transform(d[MORPH].values)
    r2 = lambda X: 1 - np.sum((y - LinearRegression().fit(X, y).predict(X)) ** 2) / SS
    R2s, R2p, R2b = r2(Xs), r2(Xp), r2(np.hstack([Xs, Xp]))
    n, ks, kp = len(y), Xs.shape[1], Xp.shape[1]
    adj = lambda R2_, k: 1 - (1 - R2_) * (n - 1) / (n - k - 1)
    R['vardecomp_ols'] = dict(R2_subject=R2s, R2_ppg=R2p, R2_both=R2b, unique_subject=R2b - R2p,
                              unique_ppg=R2b - R2s, shared=R2s + R2p - R2b,
                              adjR2_subject=adj(R2s, ks), adjR2_ppg=adj(R2p, kp), adjR2_both=adj(R2b, ks + kp),
                              p_subject=ks, p_ppg=kp)

    # ---------- ICC / mixed effects ----------
    m0 = smf.mixedlm('bgl ~ 1', d, groups=d.subject_id).fit(reml=True)
    vs, ve = float(m0.cov_re.iloc[0, 0]), float(m0.scale)
    dz = d.copy(); dz[MORPH] = StandardScaler().fit_transform(dz[MORPH])
    try:
        m1 = smf.mixedlm('bgl ~ ' + '+'.join(MORPH), dz, groups=dz.subject_id).fit(reml=True)
        fe = m1.fe_params; Xd = np.column_stack([np.ones(len(dz)), dz[MORPH].values])
        vf = np.var(Xd @ fe.values, ddof=1); vs1, ve1 = float(m1.cov_re.iloc[0, 0]), float(m1.scale)
        R['mixed'] = dict(ICC=vs / (vs + ve), R2_marginal=vf / (vf + vs1 + ve1),
                          R2_conditional=(vf + vs1) / (vf + vs1 + ve1))
    except Exception as e:
        R['mixed'] = dict(ICC=vs / (vs + ve), error=str(e))
    # bootstrap CI for ICC (cluster)
    subs = d.subject_id.unique(); iccs = []
    for b in range(200):
        smp = rng.choice(subs, len(subs), replace=True)
        db = pd.concat([d[d.subject_id == s].assign(subject_id=f'{s}_{i}') for i, s in enumerate(smp)])
        try:
            mb = smf.mixedlm('bgl ~ 1', db, groups=db.subject_id).fit(reml=True)
            iccs.append(float(mb.cov_re.iloc[0, 0]) / (float(mb.cov_re.iloc[0, 0]) + float(mb.scale)))
        except Exception:
            pass
    R['mixed']['ICC_CI95'] = [float(np.percentile(iccs, 2.5)), float(np.percentile(iccs, 97.5))]

    return R
def stage2(R, d):
    y = d.bgl.values
    # ---------- LOSO ----------
    g = d.subject_id.values; logo = LeaveOneGroupOut(); loso = {}
    for sname, fs in [('A_HR+Amp', SET_A), ('B_Morph', SET_B), ('C_Morph+Demo', SET_C)]:
        for mn, mdl in MODELS.items():
            yt, yp, ss_ = [], [], []
            for tr, te in logo.split(d, groups=g):
                sc = StandardScaler().fit(d.loc[tr, fs])
                m = copy.deepcopy(mdl).fit(sc.transform(d.loc[tr, fs]), y[tr])
                yt += list(y[te]); yp += list(m.predict(sc.transform(d.loc[te, fs]))); ss_ += list(g[te])
            loso[f'{sname}|{mn}'] = metrics(yt, yp, ss_)
    # mean-prediction baseline (training-fold mean, no features)
    yt, yp, ss_ = [], [], []
    for tr, te in logo.split(d, groups=g):
        yt += list(y[te]); yp += [y[tr].mean()] * len(te); ss_ += list(g[te])
    loso['BASELINE_train_mean'] = metrics(yt, yp, ss_)
    R['loso'] = loso

    return R
def stage3(R, d):
    y = d.bgl.values
    # ---------- personalised: subjects with N>=4 ----------
    pers_subj = [s for s, gg in d.groupby('subject_id') if len(gg) >= 4]
    R['pers_subjects'] = {int(s): int((d.subject_id == s).sum()) for s in pers_subj}
    pers = {}
    for s in pers_subj:
        gg = d[d.subject_id == s].sort_values('record_idx').reset_index(drop=True)
        X, yy = gg[MORPH].values, gg.bgl.values
        res = {}
        for mn in ['Ridge', 'RandomForest', 'SVR']:
            yt, yp = [], []
            for tr, te in LeaveOneOut().split(X):
                sc = StandardScaler().fit(X[tr])
                m = copy.deepcopy(MODELS[mn]).fit(sc.transform(X[tr]), yy[tr])
                yt.append(yy[te][0]); yp.append(m.predict(sc.transform(X[te]))[0])
            res[mn] = metrics(yt, yp)
        yt = list(yy); yp_mean = [np.delete(yy, i).mean() for i in range(len(yy))]
        res['CALIB_mean'] = metrics(yt, yp_mean)
        res['CALIB_median'] = metrics(yt, [np.median(np.delete(yy, i)) for i in range(len(yy))])
        # LOSO (pre-trained on other subjects, SetC SVR) evaluated on SAME recordings
        tr = d.subject_id != s
        sc = StandardScaler().fit(d.loc[tr, SET_C])
        m = copy.deepcopy(MODELS['SVR']).fit(sc.transform(d.loc[tr, SET_C]), d.loc[tr, 'bgl'])
        res['LOSO_same_records'] = metrics(yy, m.predict(sc.transform(gg[SET_C])))
        res['bgl_range'] = [float(yy.min()), float(yy.max())]
        pers[int(s)] = res
    R['personalised_LOO'] = pers

    # ---------- temporally held-out (forward chaining) for subjects with N>=20 ----------
    temporal = {}
    for s in [s for s in pers_subj if (d.subject_id == s).sum() >= 20]:
        gg = d[d.subject_id == s].sort_values('record_idx').reset_index(drop=True)
        X, yy = gg[MORPH].values, gg.bgl.values
        out = {}
        for K in [5, 10]:
            yt, yp_r, yp_c, yp_fix, yp_resel = [], [], [], [], []
            for t in range(K, len(gg)):
                trn = np.arange(0, t)               # all past sessions (expanding window)
                sc = StandardScaler().fit(X[trn])
                yp_r.append(Ridge(alpha=1.0).fit(sc.transform(X[trn]), yy[trn]).predict(sc.transform(X[[t]]))[0])
                yp_c.append(yy[trn].mean())
                yt.append(yy[t])
                # fixed feature: chosen once from the first K sessions; reselected: chosen from last K sessions
                def best_f(idx):
                    best, bf = -1, None
                    for j, f in enumerate(MORPH):
                        if np.std(X[idx, j]) < 1e-12: continue
                        r_ = abs(pearsonr(X[idx, j], yy[idx])[0])
                        if r_ > best: best, bf = r_, j
                    return bf
                fixed_j = best_f(np.arange(0, K)); resel_j = best_f(np.arange(t - K, t))
                win = np.arange(t - K, t)
                for j, store in [(fixed_j, yp_fix), (resel_j, yp_resel)]:
                    lr = LinearRegression().fit(X[win][:, [j]], yy[win])
                    store.append(lr.predict(X[[t]][:, [j]])[0])
            out[f'K={K}'] = dict(n_test=len(yt), ridge_expanding=metrics(yt, yp_r), calib_mean_expanding=metrics(yt, yp_c),
                                 single_feature_fixed=metrics(yt, yp_fix), single_feature_reselected=metrics(yt, yp_resel))
        temporal[int(s)] = out
    R['temporal'] = temporal

    return R
def stage4(R, d):
    R['corr'] = json.load(open(f'{ROOT}/results/{DS}_{TAG}_1.json'))['corr']
    y = d.bgl.values
    # ---------- honest out-of-sample best |r| + permutation ----------
    def honest(gg, n_splits=30, rng_=rng):
        n_ = len(gg); idx = np.arange(n_); vals = []
        if n_ < 6: return np.nan
        Xg, yg = gg[MORPH].values, gg.bgl.values
        for _ in range(n_splits):
            rng_.shuffle(idx); a, b = idx[:n_ // 2], idx[n_ // 2:]
            best, bf = -1, None
            for j in range(len(MORPH)):
                if np.std(Xg[a, j]) < 1e-12: continue
                r_ = abs(pearsonr(Xg[a, j], yg[a])[0])
                if r_ > best: best, bf = r_, j
            if bf is None or np.std(Xg[b, bf]) < 1e-12: continue
            vals.append(abs(pearsonr(Xg[b, bf], yg[b])[0]))
        return float(np.mean(vals))

    def insample(gg):
        return max(abs(pearsonr(gg[f], gg.bgl)[0]) for f in MORPH if gg[f].std() > 1e-12)
    elig = [s for s, gg in d.groupby('subject_id') if len(gg) >= 6]
    hon = {}
    for s in elig:
        gg = d[d.subject_id == s].reset_index(drop=True)
        obs = honest(gg); null = []
        for p in range(200):
            gp = gg.copy(); gp['bgl'] = rng.permutation(gp.bgl.values); null.append(honest(gp, n_splits=10))
        hon[int(s)] = dict(n=len(gg), insample_best_r=insample(gg), honest_best_r=obs,
                           null_mean=float(np.mean(null)), null_95=float(np.percentile(null, 95)),
                           p_perm=float((np.array(null) >= obs).mean()))
    R['honest_r'] = hon
    # in-sample best |r| for N>=3 (manuscript-style), for comparison
    R['insample_best_r_N3'] = {int(s): insample(gg) for s, gg in d.groupby('subject_id') if len(gg) >= 3}

    # ---------- feature stability (N>=6, halves >=3) + empirical null ----------
    def stability(gg, n_splits=50, rng_=rng):
        Xg, yg = gg[MORPH].values, gg.bgl.values; n_ = len(gg); idx = np.arange(n_); same = []
        def bf(ii):
            best, b_ = -1, None
            for j in range(len(MORPH)):
                if np.std(Xg[ii, j]) < 1e-12: continue
                r_ = abs(pearsonr(Xg[ii, j], yg[ii])[0])
                if r_ > best: best, b_ = r_, j
            return b_
        for _ in range(n_splits):
            rng_.shuffle(idx); same.append(bf(idx[:n_ // 2]) == bf(idx[n_ // 2:]))
        return float(np.mean(same))
    stab = {}
    for s in elig:
        gg = d[d.subject_id == s].reset_index(drop=True)
        obs = stability(gg); null = []
        for p in range(200):
            gp = gg.copy(); gp['bgl'] = rng.permutation(gp.bgl.values); null.append(stability(gp, n_splits=20))
        stab[int(s)] = dict(n=len(gg), consistency=obs, null_mean=float(np.mean(null)),
                            null_95=float(np.percentile(null, 95)), p_perm=float((np.array(null) >= obs).mean()))
    R['stability'] = stab

    # ---------- per-subject direction of top rmcorr feature ----------
    top = R['corr'][0]['feature']; dirs = {}
    for s, gg in d.groupby('subject_id'):
        if len(gg) >= 3 and gg[top].std() > 1e-12:
            dirs[int(s)] = float(pearsonr(gg[top], gg.bgl)[0])
    R['direction_top_feature'] = dict(feature=top, per_subject_r=dirs,
                                      n_pos=sum(v > 0 for v in dirs.values()), n_neg=sum(v < 0 for v in dirs.values()))


    return R

R = globals()['stage'+STAGE](R, d)
fn = f'{ROOT}/results/{DS}_{TAG}_{STAGE}.json'
json.dump(R, open(fn,'w'), indent=1, default=float)
print('saved', fn)
