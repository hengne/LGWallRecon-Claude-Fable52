# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Study: how far the launch-angle cut can be lowered.  Rays with sin(theta) above ~0.5 never reach the
exit of the fitted guide, so the production runs sample only sin(theta) <= 0.55.

usage: study_launch_cone.py [fit npz = ../out/fitB_datab.npz]
"""
import sys, numpy as np, lgtrace as lg
f = np.load(sys.argv[1] if len(sys.argv) > 1 else '../out/fitB_datab.npz'); c = f['c']; xs, ys = f['xs'][::40], f['ys'][::40]
ref = lg.transmission(xs, ys, coeffs=c, nray=150000, smax=1.0, sig=-0.34, seed=3)
for sm in (0.5, 0.55, 0.6, 0.7):
    T = lg.transmission(xs, ys, coeffs=c, nray=150000, smax=sm, sig=-0.34, seed=3)
    print(f'smax {sm}: mean T {T.mean():.5f} (full hemisphere {ref.mean():.5f})  mean diff {np.mean(T-ref):+.2e}  rms diff {np.std(T-ref):.2e}')
