import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import json, copy, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneGroupOut

OUT = f'{ROOT}/figures'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9,
                     'ytick.labelsize': 9, 'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})
CM, CS = '#1f5f8b', '#c0504d'
M = pd.read_csv(f'{ROOT}/features/must_features.csv')
S = pd.read_csv(f'{ROOT}/features/stuba_features.csv')
RM = json.load(open(f'{ROOT}/results/must_all_records_all.json'))
RS = {}
for st in '1234': RS.update(json.load(open(f'{ROOT}/results/stuba_all_records_{st}.json')))
MORPH = ['hr_bpm', 'ibi_mean', 'ibi_std', 'ibi_rmssd', 'sys_amp', 'dias_amp', 'dias_ratio', 'pulse_width',
         'rise_time', 'fall_time', 'auc_sys', 'auc_dias', 'auc_ratio', 'skewness', 'kurtosis_', 'spectral_ratio']
which = sys.argv[1]

if which == '2':
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    order = MORPH[::-1]
    for ax, R, name, col in [(axes[0], RM, 'MUST', CM), (axes[1], RS, 'STU-BA', CS)]:
        c = {x['feature']: x for x in R['corr']}
        yy = np.arange(len(order))
        ax.barh(yy + 0.2, [c[f]['r_pooled'] for f in order], 0.38, color='0.7', label='Pooled Pearson')
        rm = [c[f]['r_rm'] for f in order]
        lo = [c[f]['r_rm'] - c[f]['ci_rm'][0] for f in order]; hi = [c[f]['ci_rm'][1] - c[f]['r_rm'] for f in order]
        ax.barh(yy - 0.2, rm, 0.38, color=col, label='Repeated-measures r', xerr=[lo, hi],
                error_kw=dict(lw=0.8, capsize=2))
        for i, f in enumerate(order):
            if c[f]['p_rm'] < 0.05: ax.text(max(rm[i], 0) + 0.32, i - 0.2, '*', va='center', fontsize=12)
        ax.axvline(0, color='k', lw=0.8); ax.set_xlim(-0.7, 0.7); ax.set_xlabel('Correlation with BGL')
        ax.set_title(f'{name}'); ax.set_yticks(yy); ax.set_yticklabels([f.rstrip('_') for f in order])
        ax.legend(loc='lower right', frameon=False)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig2_correlations.pdf'); plt.savefig(f'{OUT}/fig2_correlations.png')

if which == '3':
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = ['OLS unique\n(subject)', 'Adjusted $R^2$\n(subject)', 'ICC\n(subject)', 'OLS unique\n(PPG)',
              'Adjusted $R^2$\n(PPG)', 'Marginal $R^2$\n(PPG, mixed)']
    def vals(R):
        v = R['vardecomp_ols']; m = R['mixed']
        return [v['unique_subject'], v['adjR2_subject'], m['ICC'], v['unique_ppg'], v['adjR2_ppg'], m['R2_marginal']]
    x = np.arange(len(labels))
    for off, R, name, col in [(-0.2, RM, 'MUST', CM), (0.2, RS, 'STU-BA', CS)]:
        v = vals(R)
        ax.bar(x + off, v, 0.38, color=col, label=name)
        ci = R['mixed']['ICC_CI95']
        ax.errorbar(x[2] + off, v[2], yerr=[[v[2] - ci[0]], [ci[1] - v[2]]], color='k', capsize=3, lw=1)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + (0.012 if vi >= 0 else -0.04), f'{vi:.2f}', ha='center', fontsize=8)
    ax.axhline(0, color='k', lw=0.8); ax.axvline(2.5, color='0.6', ls=':', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Proportion of BGL variance')
    ax.legend(frameon=False); ax.set_ylim(-0.25, 0.45)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig3_variance.pdf'); plt.savefig(f'{OUT}/fig3_variance.png')

if which == '4':
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for D, name, col, mk in [(M, 'MUST', CM, 'o'), (S, 'STU-BA', CS, 's')]:
        first = True
        for sid, g in D.groupby('subject_id'):
            if len(g) < 3: continue
            X, y = g[MORPH].values, g.bgl.values
            def best(yv):
                return max(abs(pearsonr(X[:, j], yv)[0]) for j in range(X.shape[1]) if np.std(X[:, j]) > 1e-12)
            obs = best(y); null = [best(rng.permutation(y)) for _ in range(300)]
            n = len(g)
            ax.vlines(n, np.percentile(null, 5), np.percentile(null, 95), color=col, alpha=0.35, lw=6)
            ax.scatter(n, obs, color=col, marker=mk, edgecolor='k', zorder=3, s=40,
                       label=f'{name}: observed' if first else None)
            first = False
    ax.plot([], [], color='0.5', lw=6, alpha=0.4, label='Null 5th\u201395th percentile\n(BGL permuted within subject)')
    ax.set_xscale('log'); ax.set_xticks([3, 4, 5, 6, 7, 10, 20, 32, 53]); ax.set_xticklabels([3, 4, 5, 6, 7, 10, 20, 32, 53])
    ax.set_xlabel('Recordings per subject (log scale)'); ax.set_ylabel('In-sample best |r| over 16 features')
    ax.set_ylim(0, 1.05); ax.legend(frameon=False, loc='lower left')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig4_selection_bias.pdf'); plt.savefig(f'{OUT}/fig4_selection_bias.png')

if which == '5':
    C = json.load(open(f'{ROOT}/results/cross_results.json'))
    rows = []
    for R, name in [(RM, 'MUST LOSO'), (RS, 'STU-BA LOSO')]:
        for k, v in R['loso'].items():
            rows.append((name, k, v['mae'], 'BASELINE' in k))
    for k, v in C['cross'].items():
        if 'per_dataset_z' in k: continue
        rows.append((k.split('|')[0].replace('->', ' \u2192 '), k, v['mae'], 'BASELINE' in k))
    for k, v in C['pooled47'].items():
        rows.append(('Pooled LOSO (47 subj.)', k, v['mae'], 'BASELINE' in k))
    df = pd.DataFrame(rows, columns=['scheme', 'key', 'mae', 'base'])
    schemes = ['MUST LOSO', 'STU-BA \u2192 MUST', 'STU-BA LOSO', 'MUST \u2192 STU-BA', 'Pooled LOSO (47 subj.)']
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, sname in enumerate(schemes):
        d = df[df.scheme == sname]
        m = d[~d.base]
        ax.scatter(m.mae, np.full(len(m), i) + np.linspace(-0.18, 0.18, len(m)),
                   s=14, color='0.45', alpha=0.8, label='Feature models (3 sets \u00d7 5 regressors)' if i == 0 else None)
        b = d[d.base].mae.values[0]
        ax.scatter([b], [i], marker='D', s=60, color='#e3a21a', edgecolor='k', zorder=3,
                   label='Training-mean baseline (no features)' if i == 0 else None)
    ax.set_yticks(range(len(schemes))); ax.set_yticklabels(schemes); ax.invert_yaxis()
    ax.set_xscale('log'); ax.set_xticks([10, 15, 20, 30, 40, 60, 100, 200]); ax.set_xticklabels([10, 15, 20, 30, 40, 60, 100, 200])
    ax.set_xlabel('MAE (mg/dL, log scale)'); ax.legend(frameon=False, loc='lower right')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig5_generalisation.pdf'); plt.savefig(f'{OUT}/fig5_generalisation.png')

if which == '6':
    def loso_pred(D, fs):
        g = D.subject_id.values; y = D.bgl.values; yp = np.zeros(len(y)); yb = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(D, groups=g):
            sc = StandardScaler().fit(D.iloc[tr][fs])
            m = SVR(kernel='rbf', C=10, epsilon=0.5).fit(sc.transform(D.iloc[tr][fs]), y[tr])
            yp[te] = m.predict(sc.transform(D.iloc[te][fs])); yb[te] = y[tr].mean()
        return y, yp, yb
    def grid(ax):
        ax.plot([0, 400], [0, 400], 'k:', lw=0.8)
        ax.plot([0, 175 / 3], [70, 70], 'k-', lw=0.8); ax.plot([175 / 3, 400 / 1.2], [70, 400], 'k-', lw=0.8)
        ax.plot([70, 70], [84, 400], 'k-', lw=0.8); ax.plot([0, 70], [180, 180], 'k-', lw=0.8)
        ax.plot([70, 290], [180, 400], 'k-', lw=0.8); ax.plot([70, 70], [0, 56], 'k-', lw=0.8)
        ax.plot([70, 400], [56, 320], 'k-', lw=0.8); ax.plot([180, 180], [0, 70], 'k-', lw=0.8)
        ax.plot([180, 400], [70, 70], 'k-', lw=0.8); ax.plot([240, 240], [70, 180], 'k-', lw=0.8)
        ax.plot([240, 400], [180, 180], 'k-', lw=0.8); ax.plot([130, 180], [0, 70], 'k-', lw=0.8)
        for t, xy in [('A', (30, 15)), ('B', (370, 260)), ('B', (280, 370)), ('C', (160, 370)), ('C', (160, 15)),
                      ('D', (30, 140)), ('D', (370, 120)), ('E', (30, 370)), ('E', (370, 15))]:
            ax.text(*xy, t, fontsize=12, fontweight='bold', color='0.35')
        ax.set_xlim(0, 400); ax.set_ylim(0, 400); ax.set_aspect('equal')
    from clarke import zones
    MORPHS = MORPH
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.6))
    for r, (D, name, fs, col) in enumerate([(M, 'MUST', MORPHS + ['age', 'bmi', 'gender_enc'], CM),
                                             (S, 'STU-BA', ['hr_bpm', 'sys_amp'], CS)]):
        y, yp, yb = loso_pred(D.reset_index(drop=True), fs)
        for c, (pred, lab) in enumerate([(yp, 'Best LOSO model (SVR)'), (yb, 'Training-mean baseline')]):
            ax = axes[r, c]; grid(ax)
            ax.scatter(y, pred, s=16, color=col, edgecolor='k', lw=0.3, alpha=0.8)
            z = zones(y, pred)
            ax.set_title(f'{name}: {lab}', fontsize=10)
            ax.text(0.03, 0.80, f"A {z['A']:.1f}%\nB {z['B']:.1f}%\nC\u2013E {z['C']+z['D']+z['E']:.1f}%",
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', fc='white', ec='0.6'))
            ax.set_xlabel('Reference BGL (mg/dL)'); ax.set_ylabel('Predicted BGL (mg/dL)')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig6_clarke.pdf'); plt.savefig(f'{OUT}/fig6_clarke.png')

if which == '7':
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axes[0]; keys = [('RandomForest', 'RF'), ('SVR', 'SVR'), ('CALIB_mean', 'Calibration mean'),
                          ('CALIB_median', 'Calibration median'), ('LOSO_same_records', 'LOSO model')]
    cols = ['#1f5f8b', '#5b9bd5', '#e3a21a', '#f2c46d', '0.6']
    labels = []; xs = 0; ticks = []
    for R, name in [(RM, 'MUST'), (RS, 'STU-BA')]:
        for s, res in R['personalised_LOO'].items():
            for i, ((k, lab), c) in enumerate(zip(keys, cols)):
                ax.bar(xs + i * 0.16, res[k]['mae'], 0.16, color=c, label=lab if xs == 0 else None)
            ticks.append(xs + 0.32); labels.append(f"{name}\nS{s} (n={R['pers_subjects'][s]})"); xs += 1
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel('MAE (mg/dL)')
    ax.set_yscale('log'); ax.set_yticks([2, 5, 10, 20, 50]); ax.set_yticklabels([2, 5, 10, 20, 50])
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper left'); ax.set_title('(a) Leave-one-out within subject')
    ax = axes[1]
    tk = [('ridge_expanding', 'Ridge, all past sessions'), ('calib_mean_expanding', 'Calibration mean, all past'),
          ('single_feature_fixed', 'Single feature, fixed'), ('single_feature_reselected', 'Single feature, reselected')]
    tc = ['#1f5f8b', '#e3a21a', '0.55', '#c0504d']
    for j, s in enumerate(RS['temporal']):
        v = RS['temporal'][s]['K=10']
        for i, ((k, lab), c) in enumerate(zip(tk, tc)):
            ax.bar(j + i * 0.2, v[k]['mae'], 0.2, color=c, label=lab if j == 0 else None)
    ax.set_xticks([0.3, 1.3]); ax.set_xticklabels([f"STU-BA S{s}\n({RS['temporal'][s]['K=10']['n_test']} test sessions)" for s in RS['temporal']], fontsize=8)
    ax.set_ylabel('MAE (mg/dL)'); ax.legend(frameon=False, fontsize=8); ax.set_title('(b) Temporally held-out (K = 10)')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig7_personalised.pdf'); plt.savefig(f'{OUT}/fig7_personalised.png')
print('ok', which)
