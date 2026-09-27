"""Reproduce Reviewer 1 Comment 2 analysis from original DaRUS-4559 V1 CSVs.
Run: python analyze_r1c2.py
Dependencies: numpy, pandas, scipy, matplotlib. No random operations.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SPEC = [('regression_results.csv', '9c58c023299bb51d119c2abaf8b5f851', 45),
        ('validation_results.csv', 'e3b86e528c8497922cd332c6c05fb316', 100)]
frames = []
provenance = []
for name, md5, count in SPEC:
    path = ROOT / 'raw' / name
    assert hashlib.md5(path.read_bytes()).hexdigest() == md5, name
    frame = pd.read_csv(path)
    assert frame.shape == (count, 13)
    assert np.isfinite(frame.to_numpy()).all()
    frames.append(frame)
    provenance.append({'name': name, 'md5': md5,
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
cal, val = frames
pos = ['LT_x', 'LT_y', 'LT_z']
err = ['E_x', 'E_y', 'E_z']
C, V = cal[pos].to_numpy(), val[pos].to_numpy()
Ec, E = [1000*np.linalg.norm(f[err].to_numpy(), axis=1) for f in frames]
distances = cdist(V, C)
d = distances.min(axis=1)
rho = np.linalg.norm(V, axis=1)
low, high = C.min(axis=0), C.max(axis=0)
inside = ((V >= low) & (V <= high)).all(axis=1)
q1, q2 = np.quantile(d, [1/3, 2/3], method='linear')
groups = np.where(d <= q1, 1, np.where(d <= q2, 2, 3))

def metrics(values):
    return {'n': len(values), 'mean_mm': float(np.mean(values)),
            'rmse_mm': float(np.sqrt(np.mean(values**2))),
            'median_mm': float(np.median(values))}

rows = []
for label, mask in [('Inside calibration bounding box', inside),
                    ('Outside calibration bounding box', ~inside),
                    ('Nearest-distance lower tertile', groups == 1),
                    ('Nearest-distance middle tertile', groups == 2),
                    ('Nearest-distance upper tertile', groups == 3)]:
    rows.append({'group': label, **metrics(E[mask]),
                 'min_d_cal_m': float(d[mask].min()),
                 'max_d_cal_m': float(d[mask].max())})
pd.DataFrame(rows).to_csv(OUT / 'stratified_statistics.csv', index=False)
correlations = []
for name, a, b in [('tracker_distance_vs_error', rho, E),
                    ('nearest_calibration_distance_vs_error', d, E),
                    ('tracker_distance_vs_nearest_calibration_distance', rho, d)]:
    correlations.append({'comparison': name,
                         'pearson_r': float(pearsonr(a, b).statistic),
                         'spearman_rho': float(spearmanr(a, b).statistic)})
pd.DataFrame(correlations).to_csv(OUT / 'distance_correlations.csv', index=False)
poses = val.copy()
poses.insert(0, 'source_row_1based', np.arange(1, len(val)+1))
poses['error_mm'] = E
poses['tracker_distance_m'] = rho
poses['nearest_calibration_distance_m'] = d
poses['nearest_calibration_row_1based'] = distances.argmin(axis=1)+1
poses['inside_calibration_box'] = inside
poses['nearest_distance_tertile'] = groups
poses.to_csv(OUT / 'validation_pose_audit.csv', index=False)
summary = {'dataset_doi': '10.18419/DARUS-4559', 'version': '1.0',
           'provenance': provenance, 'calibration': metrics(Ec), 'validation': metrics(E),
           'calibration_min_m': low.tolist(), 'calibration_max_m': high.tolist(),
           'calibration_ranges_m': np.ptp(C, axis=0).tolist(),
           'validation_ranges_m': np.ptp(V, axis=0).tolist(),
           'bounding_box_ratio': float(np.prod(np.ptp(V,axis=0))/np.prod(np.ptp(C,axis=0))),
           'tertile_method': 'linear (Hyndman-Fan type 7), q=1/3 and 2/3',
           'tertile_thresholds_m': [float(q1), float(q2)],
           'strata': rows, 'correlations': correlations,
           'software': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scipy': scipy.__version__,
                        'matplotlib': matplotlib.__version__}}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

# Independent numerical checks: direct Euclidean distances and rank correlation.
direct_d = np.array([min(np.sqrt(sum((p-c)**2)) for c in C) for p in V])
assert np.allclose(d, direct_d, rtol=0, atol=1e-12)
assert np.isclose(spearmanr(d, E).statistic,
                  np.corrcoef(pd.Series(d).rank(), pd.Series(E).rank())[0,1])
assert int(inside.sum()) == 11
assert list(np.bincount(groups)[1:]) == [34, 33, 33]
assert np.isclose(sum(r['n']*r['mean_mm'] for r in rows[:2])/100, E.mean())
assert np.isclose(sum(r['n']*r['mean_mm'] for r in rows[2:])/100, E.mean())

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                     'font.size': 9, 'axes.labelsize': 10, 'legend.fontsize': 8,
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(3.65, 4.0), layout='constrained')
sc = ax.scatter(V[:,0], V[:,1], c=E, cmap='Greys', vmin=0, vmax=5.1,
                s=23, edgecolors='0.25', linewidths=.4, label='Validation')
ax.scatter(C[:,0], C[:,1], s=29, marker='s', facecolors='none',
           edgecolors='black', linewidths=.8, label='Calibration', zorder=4)
ax.add_patch(Rectangle(low[:2], high[0]-low[0], high[1]-low[1],
                       fill=False, linestyle='--', linewidth=.8, color='black'))
ax.set(xlabel=r'$x_{LT}$ (m)', ylabel=r'$y_{LT}$ (m)')
ax.legend(loc='upper left', framealpha=1)
ax.grid(alpha=.18)
fig.colorbar(sc, ax=ax, label='Validation 3D error (mm)', shrink=.86)
for ext in ['pdf','png']:
    fig.savefig(OUT / f'R1C2_workspace_map.{ext}', dpi=600)
plt.close(fig)
fig, axs = plt.subplots(2, 1, figsize=(3.65, 5.3), layout='constrained')
for ax, x, lab, letter in zip(axs, [rho, d],
        [r'Tracker-origin distance, $\rho_{LT}$ (m)',
         r'Nearest-calibration distance, $d_{\mathrm{cal}}$ (m)'], ['(a)', '(b)']):
    ax.scatter(x, E, facecolors='white', edgecolors='black', s=17, linewidths=.6)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, np.polyval(np.polyfit(x, E, 1), xx), 'k--', linewidth=1)
    if letter == '(b)':
        for threshold in [q1, q2]:
            ax.axvline(threshold, color='0.55', linestyle=':', linewidth=.8)
    ax.set(xlabel=lab, ylabel='Validation 3D error (mm)', ylim=(0,5.5))
    ax.text(.03, .97, f'{letter}  Pearson r = {pearsonr(x,E).statistic:.3f}\n'
            +f'     Spearman '+r'$\rho_s$'+f' = {spearmanr(x,E).statistic:.3f}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
    ax.grid(alpha=.18)
for ext in ['pdf','png']:
    fig.savefig(OUT / f'R1C2_distance_comparison.{ext}', dpi=600)
plt.close(fig)
print(json.dumps(summary, indent=2))
