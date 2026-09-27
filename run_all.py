"""One-command reproduction. Run from any working directory: python run_all.py."""
import os,sys,subprocess,shutil,json,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
for folder in ['r1c2','r1c3']:
 (ROOT/folder/'raw').mkdir(exist_ok=True)
 for f in ['regression_results.csv','validation_results.csv']:shutil.copy2(ROOT/'data'/f,ROOT/folder/'raw'/f)
shutil.copy2(ROOT/'data/validation_results.csv',ROOT/'r1c4/validation_results.csv')
for script in ['analyze_core.py','r1c2/analyze_r1c2.py','r1c3/analyze_r1c3.py','r1c3/verify_and_plot.py','r1c4/check_coverage.py']:
 print('\nRUN',script,flush=True);subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True)
import numpy,pandas,scipy,matplotlib,sklearn
settings={'Python':platform.python_version(),'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'scikit-learn':sklearn.__version__,'seed':4559,'sample_covariance_ddof':1,'quantile_method':'linear','cross_validation':'RepeatedKFold 5 folds, 20 repetitions','regions':'KMeans, Euclidean unscaled metre coordinates, k=5, n_init=50','OLS':'intercept; train-fold StandardScaler; no tuning','partial_spearman':'average ranks, residualize on ranked x,y,z plus intercept','thread_limit':1}
(ROOT/'outputs/software_settings.json').write_text(json.dumps(settings,indent=2))
print('\nAll analyses and verification checks completed.')
