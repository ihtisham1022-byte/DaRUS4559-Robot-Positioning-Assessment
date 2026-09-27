"""R1C3: spatial adjustment, prespecified OLS comparisons, CV and VIF.
Run from any directory: python analyze_r1c3.py
All six joints retained; no feature selection or hyperparameter tuning.
"""
from pathlib import Path
import hashlib,json,platform
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.stats import rankdata,spearmanr
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score,mean_squared_error
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
f=ROOT/'raw'/'validation_results.csv'
assert hashlib.md5(f.read_bytes()).hexdigest()=='e3b86e528c8497922cd332c6c05fb316'
df=pd.read_csv(f); assert df.shape==(100,13) and np.isfinite(df.values).all()
n=len(df); seed=4559
jn=[f'Theta_{j}' for j in range(1,7)]
sn=['LT_x','LT_y','LT_z']
S=df[sn].values; J=df[jn].values; E=1000*np.linalg.norm(df[['E_x','E_y','E_z']].values,axis=1)
# Polynomial expansion has no learned parameters. Subsequent scaling fits training folds only.
poly=PolynomialFeatures(2,include_bias=False)
S2=poly.fit_transform(S)
models={'J':J,'S':S,'S+J':np.c_[S,J],'S2':S2,'S2+J':np.c_[S2,J]}
names={'J':jn,'S':sn,'S+J':sn+jn,'S2':list(poly.get_feature_names_out(sn)),'S2+J':list(poly.get_feature_names_out(sn))+jn}
def estimator(): return make_pipeline(StandardScaler(),LinearRegression())
def metric(y,p): return {'r2':float(r2_score(y,p)),'rmse_mm':float(np.sqrt(mean_squared_error(y,p)))}
def resid(y,X): return y-np.c_[np.ones(len(X)),X]@np.linalg.lstsq(np.c_[np.ones(len(X)),X],y,rcond=None)[0]
# Partial Spearman: Pearson correlation of ranked-variable residuals after linear rank-space adjustment for x,y,z.
R=rankdata(S,axis=0); er=resid(rankdata(E),R)
partial=[]
for k,name in enumerate(jn):
 pr=np.corrcoef(er,resid(rankdata(J[:,k]),R))[0,1]
 partial.append({'joint':name,'spearman':float(spearmanr(J[:,k],E).statistic),'partial_spearman_xyz':float(pr)})
pd.DataFrame(partial).to_csv(OUT/'joint_correlations.csv',index=False)
# Shared repeated folds: each repetition has one OOF prediction for every pose.
cv=list(RepeatedKFold(n_splits=5,n_repeats=20,random_state=seed).split(S))
# Euclidean metre-space regions, using coordinates only (no E or joint inputs).
regions=KMeans(n_clusters=5,n_init=50,random_state=seed).fit_predict(S)
pd.DataFrame({'row_1based':np.arange(1,n+1),'spatial_region':regions}).to_csv(OUT/'spatial_regions.csv',index=False)
fold_rows=[]; rep_rows=[]; summ=[]; vifs=[]; preds=[]
for model,X in models.items():
 fit=estimator().fit(X,E); ins=metric(E,fit.predict(X))
 Z=StandardScaler().fit_transform(X)
 assert np.linalg.matrix_rank(np.c_[np.ones(n),Z])==X.shape[1]+1
 for k,name in enumerate(names[model]):
  rr=resid(Z[:,k],np.delete(Z,k,axis=1))
  vif=np.sum((Z[:,k]-Z[:,k].mean())**2)/np.sum(rr**2)
  vifs.append({'model':model,'predictor':name,'VIF':float(vif)})
 oof=np.full((20,n),np.nan)
 for a,(tr,te) in enumerate(cv):
  rep,fold=divmod(a,5)
  oof[rep,te]=estimator().fit(X[tr],E[tr]).predict(X[te])
  if model=='J':
   fold_rows.extend({'repeat':rep+1,'fold':fold+1,'row_1based':int(i+1)} for i in te)
 assert np.isfinite(oof).all()
 for rep in range(20):
  rep_rows.append({'model':model,'repeat':rep+1,**metric(E,oof[rep])})
 blocked=np.full(n,np.nan)
 for region in range(5):
  te=np.where(regions==region)[0];tr=np.where(regions!=region)[0]
  blocked[te]=estimator().fit(X[tr],E[tr]).predict(X[te])
 assert np.isfinite(blocked).all()
 rm=np.array([metric(E,oof[r])['rmse_mm'] for r in range(20)])
 r2=np.array([metric(E,oof[r])['r2'] for r in range(20)])
 bm=metric(E,blocked)
 summ.append({'model':model,'predictors':X.shape[1],'in_sample_r2':ins['r2'],'random_cv_r2_mean':r2.mean(),'random_cv_r2_sd':r2.std(ddof=1),'random_cv_rmse_mean_mm':rm.mean(),'random_cv_rmse_sd_mm':rm.std(ddof=1),'spatial_cv_r2':bm['r2'],'spatial_cv_rmse_mm':bm['rmse_mm'],'condition_number_standardized':float(np.linalg.cond(Z))})
 for i in range(n):
  preds.append({'model':model,'row_1based':i+1,'observed_E_mm':E[i],'spatial_oof_prediction_mm':blocked[i],**{f'random_oof_repeat_{r+1}':oof[r,i] for r in range(20)}})
pd.DataFrame(summ).to_csv(OUT/'model_comparison.csv',index=False)
pd.DataFrame(vifs).to_csv(OUT/'vif.csv',index=False)
pd.DataFrame(fold_rows).to_csv(OUT/'random_folds.csv',index=False)
pd.DataFrame(rep_rows).to_csv(OUT/'cv_repetitions.csv',index=False)
pd.DataFrame(preds).to_csv(OUT/'out_of_fold_predictions.csv',index=False)
meta={'Python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,'seed':seed,'file_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'spatial_group_sizes':np.bincount(regions).tolist(),'partial_spearman_control':'linear regression on average ranks of LT_x LT_y LT_z; intercept included','models':names}
(OUT/'settings.json').write_text(json.dumps(meta,indent=2))
print(pd.DataFrame(partial).to_string(index=False))
print(pd.DataFrame(summ).to_string(index=False))
print('VIF S+J\n',pd.DataFrame(vifs).query('model=="S+J"').to_string(index=False))
print('Spatial group sizes',np.bincount(regions))
