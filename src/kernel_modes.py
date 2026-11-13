# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Linear response of the scan map to wall deformations, around an axisymmetric base wall.

The base state is rotationally symmetric, so the response to an azimuthal Fourier mode of the wall,
    delta c[i, j] = a cos(m phi_i)   (only z-knot j),
is a sum of a few azimuthal harmonics of the source angle psi,
    dT(rho, psi) = a * sum_q A[j, m, q](rho) cos(q psi),     q in {m, NPHI - m, NPHI + m}
(the extra q are the aliases of the periodic spline).  Each run perturbs the whole circumference at once
(strong signal with small wall slopes) and the projection on cos(q psi) averages the Monte Carlo noise
over all source angles.  Central differences (+a, -a) remove the second-order term.

usage: kernel_modes.py <npz with c (fit) or p (stage A)> <NPHI> <tag> [nray=12000] [amp=0.06]
"""
import sys, time, numpy as np
import lgtrace as lg
from load_scan import spot

def jacobian(kn, rho_d, psi_d):
    """J[n, i * nk + j] = dT(point n) / dc[i, j]  from the stored mode responses."""
    A, QS, rho = kn['A'], kn['QS'], kn['rho']; NPHI = int(kn['NPHI']); nk = A.shape[0]
    N = len(rho_d); J = np.zeros((N, NPHI, nk))
    r = np.clip(rho_d, rho[0], rho[-1]); k = np.clip(np.searchsorted(rho, r) - 1, 0, len(rho) - 2); f = (r - rho[k]) / (rho[k + 1] - rho[k])
    for i in range(NPHI):
        dpsi = psi_d - i * 2 * np.pi / NPHI
        for m in range(A.shape[1]):
            eps = (1.0 if m in (0, NPHI // 2) else 2.0) / NPHI
            for n in range(3):
                q = int(QS[m, n])
                if q < 0: continue
                Aq = A[:, m, n, :]                                   # [nk, nrho]
                J[:, i, :] += eps * np.cos(q * dpsi)[:, None] * (Aq[:, k] * (1 - f) + Aq[:, k + 1] * f).T
    return J.reshape(N, NPHI * nk)


if __name__ == '__main__':
    a = np.load(sys.argv[1]); NPHI = int(sys.argv[2]); tag = sys.argv[3]
    NRAY = int(sys.argv[4]) if len(sys.argv) > 4 else 12000
    AMP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.06
    NZ = int(a['NZ']); nk = NZ + 3
    if 'c' in a:
        p = a['c'].mean(axis=0); refl = float(a['refl'])
    else:
        p = a['p'][:nk]; refl = float(a['p'][nk]) if len(a['p']) > nk else 1.0
    SIG = spot(np.load('../out/edge.npz'))
    rho = 0.075 + 0.15 * np.arange(57)                  # up to 8.475 mm: includes the partly clipped edge zone
    NPS = 48                                              # psi samples on the half circle [0, pi] (reflection symmetry)
    psi = np.arange(NPS + 1) * np.pi / NPS
    RR, PP = np.meshgrid(rho, psi, indexing='ij')
    xs, ys = (RR * np.cos(PP)).ravel(), (RR * np.sin(PP)).ravel()
    wq = np.full(NPS + 1, 1.0); wq[0] = wq[-1] = 0.5      # trapezoid weights
    base = np.tile(p, (NPHI, 1)); phik = np.arange(NPHI) * 2 * np.pi / NPHI
    run = lambda C: lg.transmission(xs, ys, coeffs=C, nray=NRAY, smax=0.55, sig=SIG, refl=refl, seed=11).reshape(RR.shape)
    MMAX = NPHI // 2
    QS = [[q for q in dict.fromkeys((m, NPHI - m, NPHI + m)) if q >= 0] for m in range(MMAX + 1)]
    A = np.zeros((nk, MMAX + 1, 3, len(rho)))
    t0 = time.time(); T0 = run(base)
    print(f'base: mean T {T0.mean():.4f}; {nk} z-knots x {MMAX + 1} modes, {len(xs)} source points x {NRAY} rays', flush=True)
    for j in range(nk):
        for m in range(MMAX + 1):
            am = min(AMP, 0.18 / max(m, 1))               # finite-difference amplitude of this mode [mm]
            D = np.zeros((NPHI, nk)); D[:, j] = am * np.cos(m * phik)
            R = (run(base + D) - run(base - D)) / (2 * am)
            for n, q in enumerate(QS[m]):
                A[j, m, n] = (R * wq * np.cos(q * psi)).sum(axis=1) / NPS * (1.0 if q == 0 else 2.0)
        print(f'z-knot {j:2d} done ({time.time() - t0:.0f} s)  rms A(m=0..4): ' + ' '.join(f'{A[j, m, 0].std():.4f}' for m in range(5)), flush=True)
    np.savez(f'../out/kmodes_{tag}.npz', A=A, QS=np.array([q + [-1] * (3 - len(q)) for q in QS]), rho=rho, p=p, T0=T0, NPHI=NPHI, NZ=NZ, refl=refl)
