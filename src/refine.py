# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Re-express a fitted wall on a finer (or different) B-spline grid along z (exact when the grids are nested).
usage: refine.py <in npz with c> <NZ new> <out npz>"""
import sys, numpy as np, lgtrace as lg
def zbasis(z, NZ):
    """matrix B[len(z), NZ+3] of the uniform cubic B-spline basis used by lgtrace along z"""
    hz = lg.L / NZ; B = np.zeros((len(z), NZ + 3))
    for n, zz in enumerate(z):
        j = min(max(int(zz / hz), 0), NZ - 1); v = zz / hz - j; vm = 1 - v
        B[n, j:j + 4] = [vm ** 3 / 6, (3 * v ** 3 - 6 * v ** 2 + 4) / 6, (-3 * v ** 3 + 3 * v ** 2 + 3 * v + 1) / 6, v ** 3 / 6]
    return B
if __name__ == '__main__':
    f = np.load(sys.argv[1]); NZn = int(sys.argv[2])
    if 'c' in f:
        c = f['c']; refl = float(f['refl'])
    else:                                   # a Stage-A result: axisymmetric profile p -> same row for every phi knot
        NZa = int(f['NZ']); p = f['p']; c = np.tile(p[:NZa + 3], (24, 1)); refl = float(p[NZa + 3]) if len(p) > NZa + 3 else 1.0
    NZo = c.shape[1] - 3
    z = np.linspace(0, lg.L, 20 * NZn + 1)
    vals = c @ zbasis(z, NZo).T                       # [NPHI, len(z)] samples of every phi-coefficient row
    Bn = zbasis(z, NZn)
    # tiny curvature penalty pins the two coefficients that are only weakly seen inside [0, L]
    D2 = np.zeros((NZn + 1, NZn + 3))
    for i in range(NZn + 1): D2[i, i:i + 3] = [1, -2, 1]
    M = np.vstack([Bn, 1e-3 * D2])
    cn = np.linalg.lstsq(M, np.vstack([vals.T, np.zeros((NZn + 1, c.shape[0]))]), rcond=None)[0].T
    print('refine', c.shape, '->', cn.shape, ' max abs difference on samples', np.abs(cn @ Bn.T - vals).max())
    np.savez(sys.argv[3], c=cn, NZ=NZn, NPHI=c.shape[0], refl=refl)
