# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Compare the wall of this result with a wall fitted earlier to the same scan.

Two walls that both reproduce the scan but differ from each other show directly, on the measured data,
how much of the wall the scan leaves open.  The earlier wall is traced with the present tracer, aperture
and light spot, so the two are compared on equal terms.

usage: compare_walls.py <final npz> <earlier wall_coeffs json> [nray]      (writes ../out/compare_prev.json)"""
import sys, json, numpy as np, lgtrace as lg
from load_scan import spot, rms, wall_sha1
f = np.load(sys.argv[1]); old = json.load(open(sys.argv[2])); NRAY = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
c0 = np.array(old['c'], dtype=float); c1 = f['c']
if c0.shape != c1.shape: sys.exit(f'compare_walls: spline grids differ, {c0.shape} against {c1.shape}')
e = np.load('../out/edge.npz'); SIG = spot(e)
S, R, z = f['S'], f['R'], f['z']
X, Y = np.meshgrid(f['x'], f['y'], indexing='ij'); xr, yr = X - f['cx'], Y - f['cy']
inn = R < 8.0; plat = float(S[inn].mean())
# the earlier wall under the present tracer; amplitude fitted the same way as in final.py
T0 = lg.transmission(xr[inn], yr[inn], coeffs=c0, nray=NRAY, smax=0.55, sig=SIG, refl=float(f['refl']), seed=21)
A0 = np.sum(S[inn] * T0) / np.sum(T0 ** 2); M0 = A0 * T0
ph = f['phi'][:-1]; cp = lg.pad_coeffs(c0)
D0 = np.array([[lg.delta_eval(cp, p, zz)[0] for zz in z] for p in ph]); D1 = f['DEL'][:-1]
p0, p1 = D0.mean(axis=0), D1.mean(axis=0); L0, L1 = D0 - p0, D1 - p1
cor = lambda a, b: float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
out = dict(
    wall_sha1=wall_sha1(c1), earlier_sha1=wall_sha1(c0), nray=NRAY,
    rms_earlier=rms(S[inn] - M0) / plat * 100,            # data - earlier wall, present tracer [% of plateau]
    rms_now=rms((S - f['M'])[inn]) / plat * 100,
    map_diff=rms(M0 - f['M'][inn]) / plat * 100,          # the two simulated maps against each other [%]
    corr_local=cor(L0, L1),                               # local unevenness (azimuthal mean removed)
    loc_rms_earlier=rms(L0) * 1000, loc_rms_now=rms(L1) * 1000, loc_diff=rms(L0 - L1) * 1000,     # [um]
    prof_diff=rms(p0 - p1) * 1000, prof_diff_max=float(np.abs(p0 - p1).max() * 1000),             # [um]
    bands=[dict(z=[lo, lo + 20], corr=cor(L0[:, (z >= lo) & (z <= lo + 20)], L1[:, (z >= lo) & (z <= lo + 20)])) for lo in (0, 20, 40, 60, 80)])
json.dump(out, open('../out/compare_prev.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
