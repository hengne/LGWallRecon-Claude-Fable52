# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Test: the OpenCL tracer against the CPU tracer on the same wall and the same quasi-random rays.

Compared quantity: the transmission of each source point (a sum over its rays), not the individual rays.
  double precision on the GPU: every point must agree with the CPU within 2 rays out of NR
  single precision (the production setting): the mean difference must vanish within its statistical error,
                                             and the point-to-point scatter must stay below 1e-3 of T
Prints the throughput of each, writes ../out/gpu_check.json and exits non-zero if a criterion fails.

usage: test_gpu_vs_cpu.py [fit npz = ../out/fitB_datab.npz]
"""
import sys, json, time, numpy as np, lgtrace as lg, lgcl
from load_scan import spot, wall_sha1
f = np.load(sys.argv[1] if len(sys.argv) > 1 else '../out/fitB_datab.npz'); c = f['c']; xs, ys = f['xs'][::10], f['ys'][::10]
cp = lg.pad_coeffs(c); dlo, dhi = lg.delta_bounds(cp); SIG = spot(np.load('../out/edge.npz'))
NR = 8000; one = 0.55 ** 2 / NR
lg.transmission(xs[:4], ys[:4], coeffs=c, nray=64, gpu=False)
t0 = time.time(); Tc = lg.transmission(xs, ys, coeffs=c, nray=NR, smax=0.55, sig=SIG, seed=5, gpu=False); tc = time.time() - t0
print(f'CPU double : {len(xs)*NR/tc/1e6:6.2f} M rays/s  mean T {Tc.mean():.6f}')
out = dict(wall_sha1=wall_sha1(c), npoints=int(len(xs)), nray=NR, cpu_rate=len(xs) * NR / tc)
ok = True
for dbl in (True, False):
    lgcl.transmission(xs[:4], ys[:4], cp, dlo, dhi, 64, 0.55, SIG, 1.0, 5, double=dbl)
    t0 = time.time(); Tg = lgcl.transmission(xs, ys, cp, dlo, dhi, NR, 0.55, SIG, 1.0, 5, double=dbl); tg = time.time() - t0
    d = Tg - Tc; err = d.std() / np.sqrt(len(d)); name = 'double' if dbl else 'single'
    print(f'GPU {name} : {len(xs)*NR/tg/1e6:6.2f} M rays/s  mean T {Tg.mean():.6f}  vs CPU: mean diff {d.mean():+.2e} +- {err:.1e}  rms {d.std():.2e}  max {np.abs(d).max():.2e}  (1 ray = {one:.1e})')
    if dbl:
        good = bool(np.abs(d).max() <= 2.5 * one)
        out['double'] = dict(max_diff_rays=float(np.abs(d).max() / one), identical_points=int(np.sum(d == 0)), rate=len(xs) * NR / tg, ok=good)
    else:
        good = bool(abs(d.mean()) <= 4 * err + 1e-7 and d.std() <= 1e-3 * Tc.mean())
        out['single'] = dict(mean_diff_rel=float(d.mean() / Tc.mean()), mean_diff_err_rel=float(err / Tc.mean()), rms_diff_rel=float(d.std() / Tc.mean()), rate=len(xs) * NR / tg, ok=good)
    print('   ->', 'PASS' if good else 'FAIL'); ok = ok and good
out['ok'] = ok
json.dump(out, open('../out/gpu_check.json', 'w'), indent=1)
sys.exit(0 if ok else 1)
