# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Stage A: axisymmetric part.  Fit the m = 0 wall profile delta0(z) (and optionally the mirror reflectivity)
to the azimuthally averaged radial response profile.  Gauss-Newton / LM with finite differences and
common random numbers.

usage: [LG_TAG=<tag>] [LG_START=<A_*.npz>] python stageA.py [NZ] [fix|refl] [LAM2]
LG_START starts from the profile of an earlier fit (reflectivity 1), so that a fit with the reflectivity
free can be compared directly with the perfect-mirror fit it started from."""
import sys, os, time, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lgtrace as lg
from load_scan import load, spot, aperture_fourier

NZ = int(sys.argv[1]) if len(sys.argv) > 1 else 10
FIT_REFL = (sys.argv[2] == 'refl') if len(sys.argv) > 2 else False
tag = f'A_nz{NZ}' + ('_refl' if FIT_REFL else '')
if os.environ.get('LG_TAG'):            # explicit output name: out/A_<LG_TAG>.npz
    tag = 'A_' + os.environ['LG_TAG']
e = np.load('../out/edge.npz'); d = load()
X, Y = np.meshgrid(d['x'], d['y'], indexing='ij'); S = d['V'] - e['ped']
if os.environ.get('LG_SYNTH'):          # closure test: replace the measured scan by a synthetic one
    S = np.load(os.environ['LG_SYNTH'])['S']
    if not os.environ.get('LG_TAG'): tag += '_synth'
R = np.hypot(X - e['cx'], Y - e['cy'])
SIG = spot(e); REDGE = float(aperture_fourier(e)[0, 0])
rho = np.arange(0.1, 8.0, 0.2)
dat = np.array([S[np.abs(R - r) < 0.1].mean() for r in rho])
grad = np.abs(np.gradient(dat, rho))
err = np.hypot(0.003 * dat.max(), 0.03 * grad)          # 0.3 % + 0.03 mm radial uncertainty
NPHI = 4
nk = NZ + 3
zk = (np.arange(nk) - 1) * lg.L / NZ

def model(p, nray=120000, seed=7):
    C = np.tile(p[:nk], (NPHI, 1))
    refl = p[nk] if FIT_REFL else 1.0
    return lg.transmission(rho, 0 * rho, coeffs=C, nray=nray, smax=0.7, sig=SIG, refl=refl, seed=seed)

# regularisation: curvature of delta0(z) + tiny ridge; soft constraint delta0(0) = REDGE - R1
D2 = np.zeros((nk - 2, nk)); 
for i in range(nk - 2): D2[i, i:i + 3] = [1, -2, 1]
LAM2 = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
def resid(p, T):
    A = np.sum(dat * T / err ** 2) / np.sum(T * T / err ** 2)
    r = [(A * T - dat) / err, LAM2 * (D2 @ p[:nk]) , 0.02 * p[:nk],
         [((p[0] + 4 * p[1] + p[2]) / 6 - (REDGE - lg.R1)) / 0.003]]
    return np.concatenate([np.atleast_1d(q).ravel() for q in r]), A

npar = nk + (1 if FIT_REFL else 0)
p = np.full(npar, REDGE - lg.R1); 
if FIT_REFL: p[nk] = 0.95
if os.environ.get('LG_START'):          # continue from an earlier fit; the mirror starts perfect
    p0 = np.load(os.environ['LG_START'])['p']; p[:nk] = p0[:nk]
    if FIT_REFL: p[nk] = 1.0
T = model(p); r, A = resid(p, T); chi = r @ r; mu = 1e-2
print(f'{tag}: start chi2 {chi:.1f}  (data part {np.sum(r[:len(rho)]**2):.1f} / {len(rho)} pts)', flush=True)
for it in range(25):
    J = np.zeros((len(r), npar))
    for k in range(npar):
        h = -0.01 if k >= nk else 0.04              # reflectivity: backward difference, it is capped at 1
        q = p.copy(); q[k] += h
        J[:, k] = (resid(q, model(q))[0] - r) / h
    improved = False
    for tr in range(8):
        dp = np.linalg.solve(J.T @ J + mu * np.diag(np.diag(J.T @ J) + 1e-9), -J.T @ r)
        q = p + dp
        if FIT_REFL: q[nk] = min(q[nk], 1.0)
        Tq = model(q); rq, Aq = resid(q, Tq)
        if rq @ rq < chi:
            p, T, r, A, improved = q, Tq, rq, Aq, True; mu = max(mu / 3, 1e-6)
            dchi = chi - rq @ rq; chi = rq @ rq
            break
        mu *= 4
    print(f' it {it:2d} chi2 {chi:8.2f} data {np.sum(r[:len(rho)]**2):8.2f} mu {mu:.1e} |dp| {np.abs(dp).max():.3f}' + (f' refl {p[nk]:.4f}' if FIT_REFL else ''), flush=True)
    if not improved or dchi < 0.01 * max(chi, 1): break
zz = np.linspace(0, 100, 201)
cp = lg.pad_coeffs(np.tile(p[:nk], (NPHI, 1)))
d0 = np.array([lg.delta_eval(cp, 0.3, z)[0] for z in zz])
np.savez(f'../out/{tag}.npz', p=p, rho=rho, dat=dat, err=err, T=T, A=A, zz=zz, d0=d0, NZ=NZ)
Thi = model(p, nray=400000, seed=99)
fig, ax = plt.subplots(1, 3, figsize=(19, 5))
ax[0].errorbar(rho, dat, err, fmt='k.', label='data (azimuthal mean)'); ax[0].plot(rho, A * Thi, 'r-', label='fit')
Tid = lg.transmission(rho, 0 * rho, coeffs=np.full((NPHI, nk), REDGE - lg.R1), nray=400000, smax=0.7, sig=SIG)
ax[0].plot(rho, dat[:20].mean() / Tid[:20].mean() * Tid, 'b--', label='ideal cone (shrunk)'); ax[0].legend(); ax[0].grid(); ax[0].set_xlabel('rho [mm]')
ax[1].plot(rho, (A * Thi - dat) / dat.max() * 100, 'r.-'); ax[1].set_ylim(-3, 3); ax[1].grid(); ax[1].set_title('residual [% of plateau]')
ax[2].plot(zz, d0, 'r-'); ax[2].plot(zk, p[:nk], 'k.'); ax[2].grid(); ax[2].set_xlabel('z [mm]'); ax[2].set_ylabel('delta0 [mm]'); ax[2].set_title('m=0 wall offset' + (f'  refl={p[nk]:.3f}' if FIT_REFL else ''))
plt.tight_layout(); plt.savefig(f'../out/{tag}.png', dpi=70)
print('delta0(z) at z=0,10,..,100:', ' '.join(f'{v:+.3f}' for v in d0[::20]))
