# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Stage B: the full 2-D wall deviation delta(phi, z).

Damped (Levenberg-Marquardt) regularised least squares.  The Jacobian comes from the Fourier-mode linear
response computed around an axisymmetric base wall (kernel_modes.py); every trial step is evaluated with
a full ray trace of all fitted scan points, and the Jacobian is corrected with what the step actually did
(Broyden update).  The iteration stops when no damped step lowers the objective any more.

usage: stageB.py <kernel tag> <out tag> [lam = 1.0] [niter = 8] [start npz]
       reads  ../out/kmodes_<kernel tag>.npz, ../out/edge.npz, the scan (or LG_SYNTH)
       writes ../out/fitB_<out tag>.npz   (c = spline coefficients [NPHI, NZ + 3], A = amplitude, hist, ...)
environment: LG_RCUT (fit radius, 8.0 mm), LG_NRAY (rays per point, 20000), LG_MU (initial damping),
             LG_BROYDEN (1), LG_VERB (0), LG_SYNTH (synthetic scan npz for the closure test)
"""
import sys, os, time, numpy as np
import lgtrace as lg
from load_scan import load, spot, aperture_fourier

ktag, tag = sys.argv[1], sys.argv[2]
LAM = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
NITER = int(sys.argv[4]) if len(sys.argv) > 4 else 8
kn = np.load(f'../out/kmodes_{ktag}.npz'); e = np.load('../out/edge.npz'); d = load()
NPHI, NZ = int(kn['NPHI']), int(kn['NZ']); nk = NZ + 3; refl = float(kn['refl'])
SIG = spot(e); hphi = 2 * np.pi / NPHI
X, Y = np.meshgrid(d['x'], d['y'], indexing='ij'); S = d['V'] - e['ped']
if os.environ.get('LG_SYNTH'):          # closure test: replace the measured scan by a synthetic one
    S = np.load(os.environ['LG_SYNTH'])['S']; print('using synthetic scan', os.environ['LG_SYNTH'])
xr, yr = X - e['cx'], Y - e['cy']; R = np.hypot(xr, yr); PSI = np.arctan2(yr, xr)
RCUT = float(os.environ.get('LG_RCUT', 8.0))
m = R < RCUT
xs, ys, dat, rho, psi = xr[m], yr[m], S[m], R[m], PSI[m]
N = len(dat); SIGD = 2.0e-4
print(f'{N} interior data points, sigma_data {SIGD:.1e} V, NPHI {NPHI} NZ {NZ} -> {NPHI*nk} parameters, refl {refl}', flush=True)

# --- Jacobian (transmission per mm of spline coefficient) from the linear-response kernels
from kernel_modes import jacobian
J = jacobian(kn, rho, psi)

# --- regularisation operators on the coefficient grid
idx = np.arange(NPHI * nk).reshape(NPHI, nk)
rows = []
def add(pairs):
    r = np.zeros(NPHI * nk)
    for k, v in pairs: r[k] += v
    rows.append(r)
for i in range(NPHI):
    for j in range(nk):
        add([(idx[(i + 1) % NPHI, j], 1), (idx[i, j], -1)])                       # d/dphi
        if j + 1 < nk: add([(idx[i, j + 1], 1), (idx[i, j], -1)])                   # d/dz
P = np.array(rows)
# edge constraint: delta(phi_i, z=0) = measured aperture radius - R1
E = np.zeros((NPHI, NPHI * nk)); etar = np.zeros(NPHI)
F = aperture_fourier(e)
for i in range(NPHI):
    ph = i * hphi
    etar[i] = sum(F[mm, 0] * np.cos(mm * ph) + F[mm, 1] * np.sin(mm * ph) for mm in range(9)) - lg.R1
    for k, w in zip((-1, 0, 1), (1 / 6, 4 / 6, 1 / 6)):
        for j, v in zip((0, 1, 2), (1 / 6, 4 / 6, 1 / 6)):
            E[i, idx[(i + k) % NPHI, j]] += w * v
WE = 1 / 0.005; RIDGE = 0.05

base = np.tile(kn['p'], (NPHI, 1))
if len(sys.argv) > 5:
    c = np.load(sys.argv[5])['c'].copy()
else:
    c = base.copy()
NRAY = int(os.environ.get('LG_NRAY', 20000))
def forward(c, nray=NRAY, seed=5):
    return lg.transmission(xs, ys, coeffs=c, nray=nray, smax=0.55, sig=SIG, refl=refl, seed=seed)
# MC noise of the forward model
Ta, Tb = forward(c, seed=5), forward(c, seed=6)
A = np.sum(dat * Ta) / np.sum(Ta * Ta)
smc = A * np.std(Ta - Tb) / np.sqrt(2); sig = np.hypot(SIGD, smc)
print(f'MC noise of model per point {smc:.2e} V  -> total sigma {sig:.2e}', flush=True)
T = Ta; hist = []
VERB = os.environ.get('LG_VERB', '0') == '1'
BROYDEN = os.environ.get('LG_BROYDEN', '1') == '1'
mu = float(os.environ.get('LG_MU', 3000.0))        # LM damping [1/mm^2 in chi2 units]
for it in range(NITER + 1):
    A = np.sum(dat * T) / np.sum(T * T)
    res = (dat - A * T)
    chi = np.sum(res ** 2) / sig ** 2 / N
    rough = np.sum((P @ c.ravel()) ** 2)
    print(f'it {it}: chi2/N {chi:.3f}  rms resid {res.std()/dat.mean()*100:.3f} %  A {A:.5f}  |delta-base| rms {np.std(c-base):.3f} max {np.abs(c-base).max():.3f}  rough {rough:.3f}', flush=True)
    hist.append([chi, res.std(), A])
    np.savez(f'../out/fitB_{tag}.npz', c=c, base=base, A=A, hist=np.array(hist), NPHI=NPHI, NZ=NZ, refl=refl, lam=LAM, T=T, xs=xs, ys=ys, dat=dat)
    if it == NITER: break
    # linear system for the update D (and relative amplitude change a):  A (T + J D)(1 + a) ~ dat
    # Levenberg-Marquardt damping mu keeps the step inside the range where the linear kernels hold.
    npar = NPHI * nk
    Mfix = np.vstack([np.hstack([A * J / sig, (A * T / sig)[:, None]]),
                      np.hstack([LAM * P, np.zeros((len(P), 1))]),
                      np.hstack([RIDGE * LAM * np.eye(npar), np.zeros((npar, 1))]),
                      np.hstack([WE * E, np.zeros((NPHI, 1))])])
    rhs = np.concatenate([res / sig, -LAM * (P @ c.ravel()), -RIDGE * LAM * (c - base).ravel(), WE * (etar - E @ c.ravel())])
    def objective(cc, TT):
        AA = np.sum(dat * TT) / np.sum(TT * TT)
        return (np.sum((dat - AA * TT) ** 2) / sig ** 2 + LAM ** 2 * np.sum((P @ cc.ravel()) ** 2)
                + (RIDGE * LAM) ** 2 * np.sum((cc - base) ** 2) + WE ** 2 * np.sum((etar - E @ cc.ravel()) ** 2))
    obj0 = objective(c, T)
    best = None
    for tr in range(8):
        damp = np.hstack([np.sqrt(mu) * np.eye(npar), np.zeros((npar, 1))])
        sol = np.linalg.lstsq(np.vstack([Mfix, damp]), np.concatenate([rhs, np.zeros(npar)]), rcond=None)[0]
        D = sol[:-1].reshape(NPHI, nk)
        pred = res - A * (J @ sol[:-1]) - sol[-1] * A * T
        cn = c + D; Tn = forward(cn); obj = objective(cn, Tn)
        An = np.sum(dat * Tn) / np.sum(Tn * Tn)
        # Broyden (secant) correction of the Jacobian with what this trial step actually did to the map
        dvec = D.ravel(); lin = J @ dvec
        if BROYDEN and lin.std() > 0.3 * smc / A:
            J += np.outer((Tn - T) - lin, dvec) / (dvec @ dvec)
            Mfix[:N, :npar] = A * J / sig
        if VERB: print(f'    mu {mu:8.1f}: step rms {D.std():.4f} max {np.abs(D).max():.3f} mm  predicted chi2/N {np.sum(pred ** 2) / sig ** 2 / N:.3f}  achieved {np.sum((dat - An * Tn) ** 2) / sig ** 2 / N:.3f}  objective {obj:.1f} (was {obj0:.1f})', flush=True)
        if obj < obj0:
            best = (cn, Tn); mu = max(mu / 2.0, 1.0)
            break
        mu *= 4.0
    if best is None:
        print('no improving step; stop'); break
    c, T = best
