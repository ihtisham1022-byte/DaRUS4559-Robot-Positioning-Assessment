"""R1C4: reproducible vector-dispersion and empirical coverage check.
Run: python check_coverage.py (numpy, pandas required).
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parent
p=root/'validation_results.csv'
assert hashlib.md5(p.read_bytes()).hexdigest()=='e3b86e528c8497922cd332c6c05fb316'
f=pd.read_csv(p);e=1000*f[['E_x','E_y','E_z']].to_numpy()
assert e.shape==(100,3) and np.isfinite(e).all()
n=len(e); mean=e.mean(0); c=e-mean; S=np.cov(e,rowvar=False,ddof=1)
u=np.sqrt(np.trace(S)); U=2*u; E=np.linalg.norm(e,axis=1)
assert np.isclose(u*u,np.sum(c*c)/(n-1),rtol=1e-12)
assert np.isclose(np.mean(E*E),np.dot(mean,mean)+(n-1)/n*u*u,rtol=1e-12)
count=int(np.count_nonzero(E<=U));assert count==90
# Independent squared-distance comparison avoids the square root.
assert count==np.count_nonzero(np.sum(e*e,axis=1)<=4*np.sum(c*c)/(n-1))
assert count==np.count_nonzero(E<=round(U,3))
result={'n':n,'u_res_mm':float(u),'U_res_k2_mm':float(U),'covered_original_errors':count,'empirical_coverage':count/n,'centred_rms_mm':float(np.sqrt(np.mean(np.sum(c*c,axis=1)))),'norm_SD_mm':float(E.std(ddof=1)),'E95_mm':float(np.quantile(E,.95,method='linear')),'smallest_exceedance_mm':float(E[E>U].min()),'largest_covered_mm':float(E[E<=U].max()),'coverage_definition':'mean(norm(e_i)<=2*sqrt(trace(sample_cov(e)))); original uncentred residual vectors, full precision','file_md5':hashlib.md5(p.read_bytes()).hexdigest()}
(root/'coverage_results.json').write_text(json.dumps(result,indent=2))
pd.DataFrame({'pose_row_1based':np.arange(1,n+1),'E_mm':E,'covered_by_U_k2':E<=U}).to_csv(root/'pose_coverage.csv',index=False)
print(json.dumps(result,indent=2))
