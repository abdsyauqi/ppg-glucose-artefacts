import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneGroupOut
MORPH=['hr_bpm','ibi_mean','ibi_std','ibi_rmssd','sys_amp','dias_amp','dias_ratio','pulse_width','rise_time','fall_time','auc_sys','auc_dias','auc_ratio','skewness','kurtosis_','spectral_ratio']
rng=np.random.default_rng(3); out={}
for name,path,fs in [('MUST',f'{ROOT}/features/must_features.csv',MORPH+['age','bmi','gender_enc']),('STU-BA',f'{ROOT}/features/stuba_features.csv',['hr_bpm','sys_amp'])]:
    D=pd.read_csv(path).reset_index(drop=True); g=D.subject_id.values; y=D.bgl.values; yp=np.zeros(len(y))
    for tr,te in LeaveOneGroupOut().split(D,groups=g):
        sc=StandardScaler().fit(D.iloc[tr][fs]); yp[te]=SVR(kernel='rbf',C=10,epsilon=0.5).fit(sc.transform(D.iloc[tr][fs]),y[tr]).predict(sc.transform(D.iloc[te][fs]))
    D['ae']=np.abs(y-yp)
    D['Sex']=np.where(D.gender_enc==1,'Male','Female')
    D['Age']=np.where(D.age>30,'>30 y','<=30 y')
    D['BMI']=np.where(D.bmi>=25,'>=25','<25')
    res=[]
    for var in ['Sex','Age','BMI']:
        subj=D.groupby(['subject_id',var]).ae.mean().reset_index()
        grp={}
        for lv,gg in D.groupby(var):
            se=subj[subj[var]==lv].ae.values
            boots=[rng.choice(se,len(se)).mean() for _ in range(2000)]
            grp[lv]=dict(n_subj=int(gg.subject_id.nunique()),n_rec=len(gg),mae_subj=float(se.mean()),ci=[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],vals=se)
        lv=list(grp); p=mannwhitneyu(grp[lv[0]]['vals'],grp[lv[1]]['vals']).pvalue if min(len(grp[l]['vals']) for l in lv)>=2 else np.nan
        for l in lv:
            r=grp[l]; res.append(dict(var=var,level=l,n_subj=r['n_subj'],n_rec=r['n_rec'],mae=r['mae_subj'],ci=r['ci'],p=float(p)))
    out[name]=res
    for r in res: print(name,r['var'],r['level'],r['n_subj'],r['n_rec'],round(r['mae'],2),np.round(r['ci'],1),round(r['p'],3))
json.dump(out,open(f'{ROOT}/results/subgroup.json','w'),indent=1)
