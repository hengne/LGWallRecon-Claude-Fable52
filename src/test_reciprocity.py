# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Self-consistency test of the deformed-wall tracer: reciprocity / etendue conservation.
For a lossless specular guide and Lambertian illumination of equal radiance on both ends,
   flux(entrance -> exit) = flux(exit -> entrance)   i.e.  int_entrance T_f dA = int_exit T_b dA.
A wrong intersection or a normal inconsistent with the surface breaks this identity.
Every case must give a ratio compatible with 1 (within 4 sigma); exits non-zero otherwise.

usage: test_reciprocity.py [fit npz]        (with a fit: also writes ../out/recip.json for that wall)"""
import sys, json, numpy as np, lgtrace as lg
from load_scan import wall_sha1
rng = np.random.default_rng(3)
def flux(C, back, n=40000, nray=3000):
    half = 10.0 if not back else 5.5
    xy = (rng.random((n, 2)) - 0.5) * 2 * half
    T = lg.transmission(xy[:, 0], xy[:, 1], coeffs=C, nray=nray, smax=1.0, back=back, seed=int(back) + 1, gpu=False)
    return T.mean() * (2 * half) ** 2, T.std() / np.sqrt(n) * (2 * half) ** 2
cases = {'zero deformation': np.zeros((24, 13))}
c = 0.25 * rng.standard_normal((24, 13)); cases['random rough wall, rms 0.25 mm'] = c
c2 = 0.1 * rng.standard_normal((24, 43)); cases['random rough wall, fine grid, rms 0.1 mm'] = c2
if len(sys.argv) > 1: cases['fitted wall ' + sys.argv[1]] = np.load(sys.argv[1])['c']
ok = True; res = {}
for name, C in cases.items():
    f, ef = flux(C, False); b, eb = flux(C, True); r, er = f / b, f / b * np.hypot(ef / f, eb / b)
    good = abs(r - 1) < 4 * er; ok = ok and good
    print(f'{name:36s} forward {f:8.4f} +- {ef:.4f} mm^2   backward {b:8.4f} +- {eb:.4f} mm^2   ratio {r:.4f} +- {er:.4f}   {"PASS" if good else "FAIL"}')
    if name.startswith('fitted wall'): res = dict(ratio=float(r), err=float(er), ok=bool(good), wall_sha1=wall_sha1(C))
print('ideal cone expectation: pi R2^2 =', np.pi * 16)
if res: json.dump(res, open('../out/recip.json', 'w'), indent=1)
sys.exit(0 if ok else 1)
