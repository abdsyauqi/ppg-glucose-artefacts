import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
RM=json.load(open(f'{ROOT}/results/must_all_records_all.json')); RS={}
for s in '1234': RS.update(json.load(open(f'{ROOT}/results/stuba_all_records_{s}.json')))
C=json.load(open(f'{ROOT}/results/cross_results.json')); SG=json.load(open(f'{ROOT}/results/subgroup.json'))
MN={'LinearRegression':'Linear','Ridge':'Ridge','RandomForest':'RF','GradientBoosting':'GB','SVR':'SVR','LR':'Linear','RF':'RF','GB':'GB'}
SN={'A_HR+Amp':'A','B_Morph':'B','C_Morph+Demo':'C'}
def f(x,d=2): return f'{x:.{d}f}'.replace('-','$-$')
out=[]
# S1: within-dataset LOSO full
out.append(r'''\begin{table}[htbp]\centering\small
\caption{Leave-one-subject-out (LOSO) results for every feature set and regressor. MAE is computed on pooled predictions (each recording weighted equally); MAE$_\mathrm{s}$ averages per-subject MAE (each subject weighted equally). Clarke A+B in \%.}\label{tab:s1}
\begin{tabular}{llrrrrr|rrrrr}\hline
&&\multicolumn{5}{c|}{MUST}&\multicolumn{5}{c}{STU-BA}\\
Set&Model&MAE&MAE$_\mathrm{s}$&RMSE&$R^2$&A+B&MAE&MAE$_\mathrm{s}$&RMSE&$R^2$&A+B\\\hline''')
for sk in ['A_HR+Amp','B_Morph','C_Morph+Demo']:
    for mk in ['LinearRegression','Ridge','RandomForest','GradientBoosting','SVR']:
        k=f'{sk}|{mk}'; a=RM['loso'][k]; b=RS['loso'][k]
        out.append(f"{SN[sk]}&{MN[mk]}&{f(a['mae'])}&{f(a['mae_subject_weighted'])}&{f(a['rmse'])}&{f(a['r2'],3)}&{f(a['clarke_AB'],1)}&{f(b['mae'])}&{f(b['mae_subject_weighted'])}&{f(b['rmse'])}&{f(b['r2'],3)}&{f(b['clarke_AB'],1)}\\\\")
a=RM['loso']['BASELINE_train_mean']; b=RS['loso']['BASELINE_train_mean']
out.append(r'\hline'+f"\\multicolumn{{2}}{{l}}{{Training mean}}&{f(a['mae'])}&{f(a['mae_subject_weighted'])}&{f(a['rmse'])}&{f(a['r2'],3)}&{f(a['clarke_AB'],1)}&{f(b['mae'])}&{f(b['mae_subject_weighted'])}&{f(b['rmse'])}&{f(b['r2'],3)}&{f(b['clarke_AB'],1)}\\\\\\hline\\end{{tabular}}\\end{{table}}")
# S2: cross & pooled
out.append(r'''\begin{table}[htbp]\centering\small
\caption{Cross-dataset transfer (scaler fitted on the training dataset) and pooled LOSO over 47 subjects (a dataset indicator added to every feature set). MAE in mg/dL with subject-level cluster-bootstrap 95\% CI.}\label{tab:s2}
\begin{tabular}{llrrrrrr}\hline
&&\multicolumn{2}{c}{MUST$\to$STU-BA}&\multicolumn{2}{c}{STU-BA$\to$MUST}&\multicolumn{2}{c}{Pooled LOSO}\\
Set&Model&MAE [95\% CI]&$R^2$&MAE [95\% CI]&$R^2$&MAE [95\% CI]&$R^2$\\\hline''')
def ci(v): return f"{v['mae']:.1f} [{v['mae_ci'][0]:.1f}, {v['mae_ci'][1]:.1f}]"
for sk in 'ABC':
    for mk in ['LR','Ridge','RF','GB','SVR']:
        a=C['cross'][f'MUST->STU-BA|{sk}|{mk}|pooled_scaler']; b=C['cross'][f'STU-BA->MUST|{sk}|{mk}|pooled_scaler']; p=C['pooled47'][f'{sk}|{mk}']
        out.append(f"{sk}&{MN[mk]}&{ci(a)}&{f(a['r2'],2)}&{ci(b)}&{f(b['r2'],2)}&{ci(p)}&{f(p['r2'],2)}\\\\")
a=C['cross']['MUST->STU-BA|BASELINE_train_mean']; b=C['cross']['STU-BA->MUST|BASELINE_train_mean']; p=C['pooled47']['BASELINE_train_mean']
out.append(r'\hline'+f"\\multicolumn{{2}}{{l}}{{Training mean}}&{ci(a)}&{f(a['r2'],2)}&{ci(b)}&{f(b['r2'],2)}&{ci(p)}&{f(p['r2'],2)}\\\\\\hline\\end{{tabular}}\\end{{table}}")
open(f'{ROOT}/results/table_s1_s3.tex','w').write('\n'.join(out))
# main subgroup table
o=[r'''\begin{table}[htbp]\centering\small
\caption{Subgroup errors of the best LOSO model in each dataset. MAE is the mean of per-subject MAE with a subject-level bootstrap 95\% CI; $p$ is a two-sided Mann--Whitney $U$ test on per-subject MAE. All comparisons are exploratory.}\label{tab:subgroup}
\begin{tabular}{lllrrrr}\hline
Dataset&Factor&Level&Subjects&Recordings&MAE [95\% CI] (mg/dL)&$p$\\\hline''']
for ds in ['MUST','STU-BA']:
    for r in SG[ds]:
        lv=r['level'].replace('<=','$\\le$').replace('>=','$\\ge$').replace('<','$<$').replace('>','$>$')
        o.append(f"{ds}&{r['var']}&{lv}&{r['n_subj']}&{r['n_rec']}&{r['mae']:.1f} [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]&{r['p']:.2f}\\\\")
o.append(r'\hline\end{tabular}\end{table}')
open(f'{ROOT}/results/table_subgroup.tex','w').write('\n'.join(o))
print('ok')

# S3: temporally held-out
o=[r'''\begin{table}[htbp]\centering\small
\caption{Temporally held-out prediction in the two STU-BA subjects with many sessions. Each session $t\ge K$ is predicted from sessions before $t$. MAE in mg/dL; Clarke A+B in \%.}\label{tab:s3}
\begin{tabular}{llrrrr}\hline
Subject&Method&MAE ($K=5$)&A+B ($K=5$)&MAE ($K=10$)&A+B ($K=10$)\\\hline''']
names=[('ridge_expanding','Ridge, all past sessions'),('calib_mean_expanding','Calibration mean, all past'),('single_feature_fixed','Single feature, fixed'),('single_feature_reselected','Single feature, reselected')]
for s,v in RS['temporal'].items():
    for k,lab in names:
        a=v['K=5'][k]; b=v['K=10'][k]
        o.append(f"S{s} (n={RS['pers_subjects'][s]})&{lab}&{a['mae']:.1f}&{a['clarke_AB']:.1f}&{b['mae']:.1f}&{b['clarke_AB']:.1f}\\\\")
o.append(r'\hline\end{tabular}\end{table}')
open(f'{ROOT}/results/table_s1_s3.tex','a').write('\n'+'\n'.join(o))
print('S3 ok')
