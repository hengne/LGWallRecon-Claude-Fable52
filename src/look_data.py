# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""First look at the measured scan: plateau map, radial scatter, residual from the azimuthal median
profile, azimuthal cuts.  Writes ../out/data_look.png.

usage: look_data.py
"""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_scan import load
d = load(); x, y, V = d['x'], d['y'], d['V']
ped = np.median(V[d['passed'] == 0]); print('pedestal', ped, 'std', V[d['passed']==0].std(), 'n', (d['passed']==0).sum())
S = V - ped
# centroid / edge-based centre
thr = 0.5 * np.percentile(S, 90)
m = S > thr
X, Y = np.meshgrid(x, y, indexing='ij')
cx, cy = X[m].mean(), Y[m].mean(); area = m.sum() * 0.04
print('centre', cx, cy, 'R_half', np.sqrt(area / np.pi))
R = np.hypot(X - cx, Y - cy); PH = np.arctan2(Y - cy, X - cx)
fig, ax = plt.subplots(2, 3, figsize=(19, 11))
ext = [x[0]-.1, x[-1]+.1, y[0]-.1, y[-1]+.1]
im = ax[0,0].imshow(S.T, origin='lower', extent=ext, cmap='viridis'); plt.colorbar(im, ax=ax[0,0]); ax[0,0].set_title('signal - pedestal [V]')
pl = np.percentile(S[R < 6], [1, 99]); im = ax[0,1].imshow(S.T, origin='lower', extent=ext, cmap='turbo', vmin=pl[0]*0.97, vmax=pl[1]*1.01); plt.colorbar(im, ax=ax[0,1]); ax[0,1].set_title('plateau zoom')
ax[0,2].plot(R.ravel(), S.ravel(), ',k'); ax[0,2].set_xlabel('r [mm]'); ax[0,2].set_title('radial scatter'); ax[0,2].grid()
# radial profile & residual
rb = np.arange(0, 11, 0.2); prof = np.array([np.median(S[(R >= a) & (R < a + .2)]) if ((R >= a) & (R < a + .2)).any() else np.nan for a in rb])
res = S - np.interp(R, rb + .1, prof)
im = ax[1,0].imshow(res.T, origin='lower', extent=ext, cmap='RdBu_r', vmin=-0.002, vmax=0.002); plt.colorbar(im, ax=ax[1,0]); ax[1,0].set_title('residual from azimuthal median profile')
for r0 in [2, 4, 6, 7, 8, 8.5]:
    mm = np.abs(R - r0) < 0.15; o = np.argsort(PH[mm]); ax[1,1].plot(np.degrees(PH[mm][o]), S[mm][o], '.-', ms=3, lw=.5, label=f'r={r0}')
ax[1,1].legend(); ax[1,1].set_xlabel('phi [deg]'); ax[1,1].grid(); ax[1,1].set_title('azimuthal cuts')
ax[1,2].plot(x, S[:, np.argmin(abs(y - cy))], label='along x through centre'); ax[1,2].plot(y - cy + cx, S[np.argmin(abs(x - cx)), :], label='along y (shifted)'); ax[1,2].legend(); ax[1,2].grid()
for a in ax.ravel()[:2]: a.plot(cx, cy, 'r+')
plt.tight_layout(); plt.savefig('../out/data_look.png', dpi=80)
# noise estimate on plateau: high-pass
from scipy.ndimage import uniform_filter
hp = S - uniform_filter(S, 3); print('plateau hp rms', hp[R < 6].std(), 'plateau mean', S[R < 6].mean(), 'rel', hp[R<6].std()/S[R<6].mean())
print('n_events inside', np.unique(d['n_events'][R<8]), 'time order: x outer loop; first/last timestamps ok')
