# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""What part of the wall does the scan determine?  Linear resolution analysis of the closure test.
The Jacobian J (map change per spline coefficient) is decomposed by SVD; wall patterns with large singular
values are 'visible' in the scan, the others are not.  The true and the reconstructed closure-test walls are
compared inside the visible subspace.
usage: study_resolution.py <kmodes tag> <synth npz> <closure fit npz>"""
import sys, json, numpy as np, lgtrace as lg
from kernel_modes import jacobian
from refine import zbasis
from load_scan import load
kn = np.load(f'../out/kmodes_{sys.argv[1]}.npz'); syn = np.load(sys.argv[2]); fit = np.load(sys.argv[3])
NPHI, NZ = int(kn['NPHI']), int(kn['NZ']); nk = NZ + 3
e = np.load('../out/edge.npz'); d = load()
X, Y = np.meshgrid(d['x'], d['y'], indexing='ij'); xr, yr = X - e['cx'], Y - e['cy']; m = np.hypot(xr, yr) < 8.0
J = jacobian(kn, np.hypot(xr[m], yr[m]), np.arctan2(yr[m], xr[m]))
# true wall re-expressed on the fit grid (least squares on samples)
ph = np.arange(96) * 2 * np.pi / 96; z = np.linspace(0, 100, 201)
cpt = lg.pad_coeffs(syn['c_true'])
tru = np.array([[lg.delta_eval(cpt, p, zz)[0] for zz in z] for p in ph])
def phibasis(ph, n):
    B = np.zeros((len(ph), n)); h = 2 * np.pi / n
    for k, p in enumerate(ph):
        i = int(p / h); u = p / h - i; um = 1 - u
        for q, w in zip((-1, 0, 1, 2), (um ** 3 / 6, (3 * u ** 3 - 6 * u ** 2 + 4) / 6, (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6, u ** 3 / 6)):
            B[k, (i + q) % n] += w
    return B
Bp, Bz = phibasis(ph, NPHI), zbasis(z, NZ)
ct = np.linalg.lstsq(np.kron(Bp, Bz), tru.ravel(), rcond=None)[0].reshape(NPHI, nk)
cr = fit['c']
dt = (ct - ct.mean(axis=0)).ravel(); dr = (cr - cr.mean(axis=0)).ravel()      # non-axisymmetric parts
sigT = 2.1e-4 / float(fit['A'])                                                # noise of one scan point in transmission units
U, s, Vt = np.linalg.svd(J, full_matrices=False)
print(f'{J.shape[0]} scan points, {J.shape[1]} wall coefficients; noise per point {sigT:.2e}')
print('singular values: ', ' '.join(f'{v:.3f}' for v in s[[0, 9, 49, 99, 199, 299, 499, 799, 1031]]))
# a pattern with coefficient amplitude a along singular vector k changes the map by a * s_k (rms a s_k / sqrt(N));
# it is 'visible' at one sigma if a * s_k > sigT   ->  a_min = sigT / s_k
out = {'k': [], 'corr': [], 'frac_true': [], 'amin_um': []}
for k in (10, 30, 60, 100, 150, 200, 300, 500):
    P = Vt[:k]
    a, b = P @ dt, P @ dr
    c = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    out['k'].append(k); out['corr'].append(c); out['frac_true'].append(float((a @ a) / (dt @ dt))); out['amin_um'].append(float(sigT / s[k - 1] * 1000))
    print(f'top {k:4d} patterns: smallest visible amplitude {sigT / s[k - 1] * 1000:7.2f} um; true wall has {100 * (a @ a) / (dt @ dt):5.1f} % of its power here;  corr(true, reconstructed) inside = {c:+.3f}')
print('whole space: corr', float(dt @ dr / np.sqrt((dt @ dt) * (dr @ dr))))
# does the linear map of the true wall explain the synthetic map?  (checks J itself)
json.dump(out, open('../out/resolution.json', 'w'), indent=1)
