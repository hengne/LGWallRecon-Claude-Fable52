# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Validate the Fourier-mode Jacobian against exact (high statistics) finite changes of the map.
The perturbations are smooth random non-axisymmetric fields of several amplitudes and one axisymmetric step.
For each the exact change of the map is compared with the linear prediction: correlation (after removing the
mean) and slope.  Writes ../out/jacobian.json.
usage: test_jacobian.py <kmodes tag> [fit npz to test 'at current fit']"""
import sys, json, numpy as np, lgtrace as lg
from kernel_modes import jacobian
from load_scan import load, spot, wall_sha1
kn = np.load(f'../out/kmodes_{sys.argv[1]}.npz'); e = np.load('../out/edge.npz'); d = load(); SIG = spot(e)
NPHI, NZ = int(kn['NPHI']), int(kn['NZ']); nk = NZ + 3
X, Y = np.meshgrid(d['x'], d['y'], indexing='ij'); xr, yr = X - e['cx'], Y - e['cy']; m = np.hypot(xr, yr) < 8.0
xs, ys = xr[m][::2], yr[m][::2]; rho = np.hypot(xs, ys); psi = np.arctan2(ys, xs)
J = jacobian(kn, rho, psi)
base = np.tile(kn['p'], (NPHI, 1))
fw = lambda cc: lg.transmission(xs, ys, coeffs=cc, nray=200000, smax=0.55, sig=SIG, seed=5)
rng = np.random.default_rng(2)
def smooth(rms, nsm):
    D = rng.standard_normal((NPHI, nk))
    for _ in range(nsm): D = (D + np.roll(D, 1, 0) + np.roll(D, -1, 0)) / 3; D[:, 1:-1] = (D[:, :-2] + D[:, 1:-1] + D[:, 2:]) / 3
    D -= D.mean(axis=0); return D * rms / D.std()
starts = [('axisymmetric base', base)]
if len(sys.argv) > 2: starts.append(('current fit', np.load(sys.argv[2])['c']))
res = {'wall_sha1': wall_sha1(starts[-1][1]) if len(starts) > 1 else '', 'starts': []}
for name, c0 in starts:
    T0 = fw(c0); T0b = lg.transmission(xs, ys, coeffs=c0, nray=200000, smax=0.55, sig=SIG, seed=6)
    print(f'{name}: MC noise of a 200k-ray map (two seeds) {np.std(T0 - T0b) / np.sqrt(2):.2e}')
    rec = {'name': name, 'smooth': []}; res['starts'].append(rec)
    for rms, nsm in ((0.01, 2), (0.03, 2), (0.06, 2), (0.03, 6), (0.1, 6)):
        D = smooth(rms, nsm); ex = fw(c0 + D) - T0; li = J @ D.ravel()
        rec['smooth'].append(dict(rms_um=1000 * rms, nsm=nsm, corr=float(np.corrcoef(ex, li)[0, 1]), slope=float(np.sum(ex * li) / np.sum(li * li))))
        print(f'  smooth x{nsm} rms {rms:.2f} mm: exact dT rms {ex.std():.5f}  linear {li.std():.5f}  corr {np.corrcoef(ex, li)[0, 1]:.3f}  slope {np.sum(ex * li) / np.sum(li * li):.3f}')
    # axisymmetric changes: a sharp 50 um step at mid length, and a gentle 30 um bump (sigma 10 mm) there
    zk = (np.arange(nk) - 1) * lg.L / NZ
    for key, label, prof in (('m0_step', 'm=0 step +0.05 mm for z > 50', 0.05 * (np.arange(nk) > nk // 2)),
                             ('m0_smooth', 'm=0 bump 0.03 mm, sigma 10 mm', 0.03 * np.exp(-(zk - 50.0) ** 2 / 200.0))):
        D = np.tile(prof, (NPHI, 1)); ex = fw(c0 + D) - T0; li = J @ D.ravel()
        rec[key] = dict(corr=float(np.corrcoef(ex, li)[0, 1]), slope=float(np.sum(ex * li) / np.sum(li * li)), mean_ratio=float(ex.mean() / li.mean()))
        print(f'  {label}: exact dT mean {ex.mean():+.5f} rms {ex.std():.5f}  linear mean {li.mean():+.5f} rms {li.std():.5f}  corr {rec[key]["corr"]:.3f}  slope {rec[key]["slope"]:.3f}')
json.dump(res, open('../out/jacobian.json', 'w'), indent=1)
