# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Study: radial response of the ideal cone for several mirror reflectivities, next to the measured
azimuthally averaged profile.  Needs ../out/edge.npz (run edge.py first).

usage: study_reflectivity.py
"""
import numpy as np, lgtrace as lg
from load_scan import load
d = load(); b = np.load('../out/edge.npz'); S = d['V'] - b['ped']
X, Y = np.meshgrid(d['x'], d['y'], indexing='ij'); R = np.hypot(X - b['cx'], Y - b['cy'])
rho = np.arange(0.25, 8.6, 0.5)
dat = np.array([S[np.abs(R - r) < 0.25].mean() for r in rho]); dat /= dat[:8].mean()
print('rho   ', ' '.join(f'{r:5.2f}' for r in rho))
print('data  ', ' '.join(f'{t:5.3f}' for t in dat))
for refl in [1.0, 0.98, 0.95, 0.9, 0.8]:
    T = lg.transmission(rho, 0 * rho, nray=300000, smax=0.75, refl=refl)
    print(f'R={refl:4.2f}', ' '.join(f'{t:5.3f}' for t in T / T[:8].mean()), ' abs T(0)=%.4f' % T[0])
