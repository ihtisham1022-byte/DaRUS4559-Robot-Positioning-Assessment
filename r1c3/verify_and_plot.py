from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent; O=ROOT/'outputs'
df=pd.read_csv(ROOT/'raw/validation_results.csv')
S=df[['LT_x','LT_y','LT_z']].values
J=df[[f'Theta_{j}' for j in range(1,7)]].values
E=1000*np.linalg.norm(df[['E_x','E_y','E_z']].values,axis=1)
S2=PolynomialFeatures(2,include_bias=False).fit_transform(S)
models={'J':J,'S':S,'S+J':np.c_[S,J],'S2':S2,'S2+J':np.c_[S2,J]}
summary=pd.read_csv(O/'model_comparison.csv').set_index('model')
vifs=pd.read_csv(O/'vif.csv')
for key,X in models.items():
 D=np.c_[np.ones(len(E)),X]
 pred=D@np.linalg.lstsq(D,E,rcond=None)[0]
 assert np.isclose(r2_score(E,pred),summary.loc[key,'in_sample_r2'],atol=1e-11)
 if key=='S+J':
  Z=np.c_[np.ones(len(E)),StandardScaler().fit_transform(X)]
  vf=np.diag(np.linalg.inv(np.corrcoef(X,rowvar=False)))
  assert np.allclose(vf,vifs.query('model=="S+J"').VIF,rtol=1e-10)
c=pd.read_csv(O/'joint_correlations.csv')
# Independent partial-correlation identity via inverse rank-correlation matrix.
for j in range(6):
 C=np.corrcoef(rankdata(np.c_[E,J[:,j],S],axis=0),rowvar=False)
 P=np.linalg.inv(C)
 assert np.isclose(-P[0,1]/np.sqrt(P[0,0]*P[1,1]),c.partial_spearman_xyz[j],atol=1e-10)
folds=pd.read_csv(O/'random_folds.csv')
assert all(g.row_1based.nunique()==100 and len(g)==100 for _,g in folds.groupby('repeat'))
pred=pd.read_csv(O/'out_of_fold_predictions.csv')
for model,g in pred.groupby('model',sort=False):
 assert len(g)==100 and g.row_1based.nunique()==100
 assert np.isclose(np.sqrt(np.mean((g.observed_E_mm-g.spatial_oof_prediction_mm)**2)),summary.loc[model,'spatial_cv_rmse_mm'])
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(4.8,3.5),layout='constrained')
x=np.arange(6); w=.35
ax.bar(x-w/2,c.spearman,w,label='Unadjusted Spearman',color='.78',edgecolor='black',linewidth=.6)
ax.bar(x+w/2,c.partial_spearman_xyz,w,label='Partial Spearman (x, y, z)',color='.30',edgecolor='black',linewidth=.6)
ax.axhline(0,color='black',lw=.7)
ax.set_xticks(x,[rf'$\Theta_{j}$' for j in range(1,7)])
ax.set_ylabel('Signed correlation with 3D error')
ax.set_ylim(-1,1);ax.set_yticks(np.arange(-1,1.1,.5))
ax.legend(loc='upper right',fontsize=8,frameon=False)
ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.savefig(O/'R1C3_joint_spatial_adjustment.pdf')
fig.savefig(O/'R1C3_joint_spatial_adjustment.png',dpi=400)
fig.savefig(O/'preview.png',dpi=140)
print('PASS: independent OLS, partial correlation, VIF, fold coverage and pooled spatial RMSE checks.')
