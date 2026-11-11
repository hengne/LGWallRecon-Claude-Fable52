# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Study: response maps of a set of simple test deformations of the wall (uniform shrink, barrel,
hourglass, single bumps and dents, exit offset, ellipticity, entrance flare and lip, exit pinch), and the
radial response of the ideal cone for non-Lambertian sources.  Writes ../out/explore.png.

usage: study_deformations.py
"""
import time, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lgtrace as lg
NPHI, NZ = 24, 20
phi = np.arange(NPHI) * 2 * np.pi / NPHI
zk = (np.arange(NZ + 3) - 1) * lg.L / NZ
PH, Z = np.meshgrid(phi, zk, indexing='ij')
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
r0 = lg.R1 + lg.SLOPE * Z
tests = {
 'ideal': 0 * Z,
 'shrink -0.1': -0.1 + 0 * Z,
 'barrel +0.2': 0.2 * np.sin(np.pi * np.clip(Z, 0, 100) / 100),
 'hourglass -0.2': -0.2 * np.sin(np.pi * np.clip(Z, 0, 100) / 100),
 'bump -0.3 @z30,phi0': -0.3 * np.exp(-(wrap(PH) * r0) ** 2 / 2 / 3 ** 2 - (Z - 30) ** 2 / 2 / 8 ** 2),
 'bump -0.3 @z70,phi0': -0.3 * np.exp(-(wrap(PH) * r0) ** 2 / 2 / 3 ** 2 - (Z - 70) ** 2 / 2 / 8 ** 2),
 'dent +0.3 @z30,phi0': +0.3 * np.exp(-(wrap(PH) * r0) ** 2 / 2 / 3 ** 2 - (Z - 30) ** 2 / 2 / 8 ** 2),
 'exit offset 0.2 cos(phi) z/L': 0.2 * np.cos(PH) * Z / 100,
 'ellipse 0.1 cos2phi': 0.1 * np.cos(2 * PH),
 'entrance flare: +0.3 exp(-z/10)': 0.3 * np.exp(-np.clip(Z, 0, 100) / 10),
 'entrance lip: -0.3 exp(-z/10)': -0.3 * np.exp(-np.clip(Z, 0, 100) / 10),
 'exit pinch: -0.2 exp(-(100-z)/10)': -0.2 * np.exp(-(100 - np.clip(Z, 0, 100)) / 10),
}
g = np.arange(-9.0, 9.01, 0.4); X, Y = np.meshgrid(g, g, indexing='ij'); m = np.hypot(X, Y) < 9.2
fig, ax = plt.subplots(3, 4, figsize=(22, 15))
for a, (name, C) in zip(ax.ravel(), tests.items()):
    t0 = time.time()
    T = lg.transmission(X[m], Y[m], coeffs=C, nray=6000, smax=0.75)
    M = np.full(X.shape, np.nan); M[m] = T
    R = np.hypot(X, Y)[m]
    prof = [T[(R >= a_) & (R < a_ + 1)].mean() for a_ in range(0, 9)]
    print(f'{name:34s} {time.time()-t0:5.1f}s  mean {np.nanmean(T[T>0]):.4f}  radial:', ' '.join(f'{p:.3f}' for p in prof))
    im = a.imshow(np.where(M > 0.02, M, np.nan).T, origin='lower', extent=[-9.2, 9.2, -9.2, 9.2], cmap='turbo'); plt.colorbar(im, ax=a); a.set_title(name)
plt.tight_layout(); plt.savefig('../out/explore.png', dpi=60)
# non-Lambertian source in the ideal cone
rho = np.linspace(0, 8.6, 9)
for n in [1, 2, 4, 8]:
    print('cos^%d source, ideal cone, T(rho):' % n, ' '.join(f'{t:.3f}' for t in lg.transmission(rho, 0 * rho, nray=200000, lamb_n=float(n))))
