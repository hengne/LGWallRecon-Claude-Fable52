# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""What part of the wall does the scan determine?  Linear resolution analysis of the closure test.

The Jacobian J (change of the map per spline coefficient) is linearised around the axisymmetric base wall
of kernel_modes.py.  The overall amplitude is a free parameter of every fit, so the direction of the map
itself is projected out of J first.  J is then decomposed by SVD: wall patterns with large singular values
change the map a lot, the others hardly.  The true and the reconstructed closure-test walls are compared
inside the span of the k best-determined patterns, for several k.

There is no sharp boundary between 'seen' and 'unseen' patterns: the singular values fall smoothly, and
whether a pattern is determined depends on its amplitude.  For every k the script prints the one-sigma
uncertainty with which the scan fixes the amplitude of pattern k (sigma_T / s_k), both as the norm of the
coefficient vector (the metric of the SVD) and as the RMS height of that pattern on the wall surface.

usage: study_resolution.py <kmodes tag> <synth npz> <closure fit npz>      (writes ../out/resolution.json)"""
import sys, json, numpy as np, lgtrace as lg
from kernel_modes import jacobian
from refine import zbasis
from load_scan import load, rms
kn = np.load(f'../out/kmodes_{sys.argv[1]}.npz'); syn = np.load(sys.argv[2]); fit = np.load(sys.argv[3])
NPHI, NZ = int(kn['NPHI']), int(kn['NZ']); nk = NZ + 3
e = np.load('../out/edge.npz'); d = load()
X, Y = np.meshgrid(d['x'], d['y'], indexing='ij'); xr, yr = X - e['cx'], Y - e['cy']; m = np.hypot(xr, yr) < 8.0
J = jacobian(kn, np.hypot(xr[m], yr[m]), np.arctan2(yr[m], xr[m]))
T = fit['T']                                   # fitted map at the same points
if len(T) == len(J):
    J = J - np.outer(T, T @ J) / (T @ T)       # free amplitude: remove the component along the map itself
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
G = np.kron(Bp, Bz)                            # samples of the wall surface per coefficient
ct = np.linalg.lstsq(G, tru.ravel(), rcond=None)[0].reshape(NPHI, nk)
print(f'true wall on the fit grid: representation error {rms(G @ ct.ravel() - tru.ravel()) * 1000:.3f} um RMS')
cr = fit['c']
dt = (ct - ct.mean(axis=0)).ravel(); dr = (cr - cr.mean(axis=0)).ravel()      # non-axisymmetric parts
sigT = 2.0e-4 / float(fit['A'])                                                # noise of one scan point in transmission units
U, s, Vt = np.linalg.svd(J, full_matrices=False)
N = J.shape[0]
print(f'{N} scan points, {J.shape[1]} wall coefficients; noise per point {sigT:.2e}')
print('singular values: ', ' '.join(f'{v:.3f}' for v in s[[0, 9, 49, 99, 199, 299, 499, 799, 1031]]))
# Pattern k with coefficient-vector norm a changes the map by a * s_k * u_k, i.e. chi2 by (a s_k / sigT)^2:
# the scan fixes its amplitude to  +- sigT / s_k  (one sigma).
out = {'k': [], 'corr': [], 'amp_ratio': [], 'frac_true': [], 'amin_um': [], 'amin_surf_um': []}
for k in (10, 30, 60, 100, 150, 200, 300, 500):
    P = Vt[:k]
    a, b = P @ dt, P @ dr
    c = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    amin = sigT / s[k - 1]; surf = amin * rms(G @ Vt[k - 1])
    out['k'].append(k); out['corr'].append(c); out['amp_ratio'].append(float(np.sqrt((b @ b) / (a @ a)))); out['frac_true'].append(float((a @ a) / (dt @ dt)))
    out['amin_um'].append(float(amin * 1000)); out['amin_surf_um'].append(float(surf * 1000))
    print(f'best {k:4d} patterns: amplitude of pattern {k} fixed to +- {surf * 1000:7.3f} um surface RMS; the true wall has {100 * (a @ a) / (dt @ dt):5.1f} % of its power here;  corr(true, reconstructed) = {c:+.3f}, amplitude ratio {np.sqrt((b @ b) / (a @ a)):.2f}')
out['corr_all'] = float(dt @ dr / np.sqrt((dt @ dt) * (dr @ dr)))
out['base_sha1'] = str(syn['base_sha1']) if 'base_sha1' in syn.files else ''
print('whole coefficient space: corr', out['corr_all'])
json.dump(out, open('../out/resolution.json', 'w'), indent=1)
