# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Test: the OpenCL tracer against the CPU tracer on the same wall and the same quasi-random rays.
Double precision on the GPU must agree ray by ray (at most one ray per point); single precision must be
unbiased.  Also prints the throughput of each.

usage: test_gpu_vs_cpu.py [fit npz = ../out/fitB_datab.npz]
"""
import sys, time, numpy as np, lgtrace as lg, lgcl
f = np.load(sys.argv[1] if len(sys.argv) > 1 else '../out/fitB_datab.npz'); c = f['c']; xs, ys = f['xs'][::10], f['ys'][::10]
cp = lg.pad_coeffs(c); dlo, dhi = lg.delta_bounds(cp)
NR = 8000
lg.transmission(xs[:4], ys[:4], coeffs=c, nray=64, gpu=False)
t0 = time.time(); Tc = lg.transmission(xs, ys, coeffs=c, nray=NR, smax=0.55, sig=-0.34, seed=5, gpu=False); tc = time.time() - t0
print(f'CPU double : {len(xs)*NR/tc/1e6:6.2f} M rays/s  mean T {Tc.mean():.6f}')
for dbl in (True, False):
    lgcl.transmission(xs[:4], ys[:4], cp, dlo, dhi, 64, 0.55, -0.34, 1.0, 5, double=dbl)
    t0 = time.time(); Tg = lgcl.transmission(xs, ys, cp, dlo, dhi, NR, 0.55, -0.34, 1.0, 5, double=dbl); tg = time.time() - t0
    d = Tg - Tc
    print(f'GPU {"double" if dbl else "single"} : {len(xs)*NR/tg/1e6:6.2f} M rays/s  mean T {Tg.mean():.6f}  vs CPU: mean diff {d.mean():+.2e}  rms {d.std():.2e}  max {np.abs(d).max():.2e}  (1 ray = {0.55**2/NR:.1e})')
