"""Reproduce baseline residual statistics, analytical intervals, integrity audit and Figures 1,2,3,7,8."""
from pathlib import Path
import json,hashlib,platform
import numpy as np,pandas as pd,scipy,matplotlib
from scipy.stats import t,norm,pearsonr,spearmanr
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'data_manifest.json').read_text());frames={};audit=[]
for f in manifest['files']:
 p=ROOT/'data'/f['name'];assert hashlib.md5(p.read_bytes()).hexdigest()==f['md5']
 d=pd.read_csv(p);frames[f['name']]=d
 audit.append({'file':f['name'],'rows':len(d),'columns':list(d.columns),'missing_cells':int(d.isna().sum().sum()),'duplicate_rows':int(d.duplicated().sum()),'constant_columns':[c for c in d if d[c].nunique()==1],'md5':f['md5']})
 for a in audit[-1:]:
  if 'timeseries' in f['name']:
   x=d.to_numpy(float);a['first_third_coordinate_equal']=bool(np.array_equal(x[:,0],x[:,2]));a['coordinate_ranges']=np.ptp(x,axis=0).tolist();a['timestamp_column_present']=any('time' in c.lower() for c in d.columns)
   a['mean_step_native_units']=float(np.linalg.norm(np.diff(x,axis=0),axis=1).mean())
(OUT/'integrity_audit.json').write_text(json.dumps(audit,indent=2))
C=frames['regression_results.csv'];V=frames['validation_results.csv']; ec=1000*C[['E_x','E_y','E_z']].values;e=1000*V[['E_x','E_y','E_z']].values
E=np.linalg.norm(e,axis=1);Ec=np.linalg.norm(ec,axis=1);mean=e.mean(0);cent=e-mean;Ebc=np.linalg.norm(cent,axis=1);N=len(E)
def stats(x):
 return dict(n=len(x),mean=np.mean(x),sd=np.std(x,ddof=1),rmse=np.sqrt(np.mean(x*x)),median=np.median(x),E95=np.quantile(x,.95,method='linear'),E99=np.quantile(x,.99,method='linear'),maximum=max(x))
sC,sV,sB=map(stats,[Ec,E,Ebc]);pd.DataFrame([dict(dataset=k,**v) for k,v in [('calibration',sC),('validation',sV),('mean_centred',sB)]]).to_csv(OUT/'overall_statistics.csv',index=False)
pd.DataFrame([dict(axis=a,bias=e[:,i].mean(),sd=e[:,i].std(ddof=1),mae=np.abs(e[:,i]).mean(),rmse=np.sqrt(np.mean(e[:,i]**2)),max_abs=np.abs(e[:,i]).max()) for i,a in enumerate('xyz')]).to_csv(OUT/'axis_statistics.csv',index=False)
S=np.cov(e,rowvar=False,ddof=1);u=np.sqrt(np.trace(S));b=np.linalg.norm(mean);rmse2=np.mean(E*E)
assert np.isclose(rmse2,b*b+(N-1)/N*u*u)
pd.DataFrame(S,index=list('xyz'),columns=list('xyz')).to_csv(OUT/'covariance_mm2.csv')
metrics=dict(u_res_mm=u,U_res_k2_mm=2*u,bias_norm_mm=b,bias_fraction=b*b/rmse2,scatter_fraction=((N-1)/N*u*u)/rmse2,coverage_count=int(sum(E<=2*u)),mean_reduction_pct=100*(sV['mean']-sB['mean'])/sV['mean'],rmse_reduction_pct=100*(sV['rmse']-sB['rmse'])/sV['rmse'],G_mean=sV['mean']/sC['mean'],G_rmse=sV['rmse']/sC['rmse'],G_E95=sV['E95']/sC['E95'],G_max=sV['maximum']/sC['maximum'])
(OUT/'metrological_and_gap_statistics.json').write_text(json.dumps(metrics,indent=2))
ps=[50,68,90,95,99];q=np.quantile(E,np.array(ps)/100,method='linear');pd.DataFrame({'percentile':ps,'error_mm':q}).to_csv(OUT/'empirical_quantiles.csv',index=False)
intervals=[]
for name,x in [('Mean 3D error',E)]+[(f'Mean {a}',e[:,i]) for i,a in enumerate('xyz')]:
 m=x.mean();h=t.ppf(.975,N-1)*x.std(ddof=1)/np.sqrt(N);intervals.append(dict(statistic=name,estimate=m,lower=m-h,upper=m+h))
rho=np.linalg.norm(V[['LT_x','LT_y','LT_z']].values,axis=1)
for name,fun in [('Pearson',pearsonr),('Spearman',spearmanr)]:
 r=fun(rho,E).statistic;lo,hi=np.tanh(np.arctanh(r)+np.array([-1,1])*norm.ppf(.975)/np.sqrt(N-3));intervals.append(dict(statistic=name,estimate=r,lower=lo,upper=hi))
pd.DataFrame(intervals).to_csv(OUT/'analytical_intervals.csv',index=False)
# Numeric gates check meaningful baseline values without changing their computation.
assert np.isclose(sV['mean'],2.0017185643183755) and np.isclose(sV['rmse'],2.3042855434449376)
assert np.isclose(u,1.8487696906836562) and metrics['coverage_count']==90
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(4.8,3.4));
for x,l,c in [(Ec,'Calibration','.6'),(E,'Validation','black')]:ax.step(np.sort(x),np.arange(1,len(x)+1)/len(x),where='post',label=l,color=c)
ax.set(xlabel='3D positioning error (mm)',ylabel='Empirical cumulative probability');ax.legend(frameon=False);ax.grid(alpha=.2);save(fig,'Figure2_error_cdf')
fig,ax=plt.subplots(figsize=(4.8,3.4));ax.boxplot(e,tick_labels=[r'$x_{LT}$',r'$y_{LT}$',r'$z_{LT}$'],showmeans=True);ax.axhline(0,color='black',ls='--',lw=.8);ax.set(ylabel='Validation residual (mm)');save(fig,'Figure3_axis_residuals')
fig,ax=plt.subplots(figsize=(4.8,3.4));ax.plot(ps+[100],list(q)+[max(E)],'o-',color='black');ax.set(xlabel='Empirical coverage percentile',ylabel='3D positioning error (mm)');ax.grid(alpha=.2);save(fig,'Figure7_empirical_coverage')
fig,ax=plt.subplots(figsize=(4.8,3.4));w=.18
for i,(k,l) in enumerate([('mean','Mean'),('rmse','RMSE'),('E95','95th percentile'),('maximum','Maximum')]):ax.bar(np.arange(2)+(i-1.5)*w,[sV[k],sB[k]],w,label=l,color=str(.25+i*.18),edgecolor='black',lw=.6)
ax.set_xticks([0,1],['Original','Mean-centred']);ax.set_ylabel('Positioning error (mm)');ax.legend(fontsize=8,ncol=2,frameon=False);save(fig,'Figure8_bias_diagnostic')
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(9,1.8));ax.axis('off')
labels=['Verify files\nand units','Separate\ncalibration and\nvalidation','Bias, scatter\nand error tails','Spatial and joint\ndiagnostics','Qualified\nindicators and\ncoverage']
for i,label in enumerate(labels):
 x=.1+.2*i
 ax.add_patch(FancyBboxPatch((x-.077,.25),.154,.5,boxstyle='round,pad=.005',facecolor='white',edgecolor='black',transform=ax.transAxes))
 ax.text(x,.5,label,ha='center',va='center',fontsize=9,transform=ax.transAxes)
 if i<4:ax.annotate('',xy=(x+.118,.5),xytext=(x+.082,.5),arrowprops=dict(arrowstyle='->'),xycoords='axes fraction')
save(fig,'Figure1_workflow_schematic')
print(pd.DataFrame(intervals).round(6).to_string(index=False)); print('Core analysis and figures completed.');print(json.dumps(audit[:2],indent=2))
