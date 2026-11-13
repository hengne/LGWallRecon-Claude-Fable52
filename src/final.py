# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""High-statistics simulation of the full scan with a fitted wall, comparison with data, exports.
usage: final.py <fitB npz> <tag> [nray]"""
import sys, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lgtrace as lg
from load_scan import load, spot, aperture_fourier, rms
f = np.load(sys.argv[1]); tag = sys.argv[2]; NRAY = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
c = f['c']; refl = float(f['refl'])
e = np.load('../out/edge.npz'); d = load(); SIG = spot(e)
X, Y = np.meshgrid(d['x'], d['y'], indexing='ij'); S = d['V'] - e['ped']
import os
if os.environ.get('LG_SYNTH'):          # closure test: replace the measured scan by a synthetic one
    S = np.load(os.environ['LG_SYNTH'])['S']; print('using synthetic scan', os.environ['LG_SYNTH'])
xr, yr = X - e['cx'], Y - e['cy']; R = np.hypot(xr, yr)
m = R < 9.5
T = np.zeros(S.shape); T[m] = lg.transmission(xr[m], yr[m], coeffs=c, nray=NRAY, smax=0.55, sig=SIG, refl=refl, seed=21)
T2 = np.zeros(S.shape); T2[m] = lg.transmission(xr[m], yr[m], coeffs=c, nray=NRAY, smax=0.55, sig=SIG, refl=refl, seed=22)
Tid = np.zeros(S.shape); Tid[m] = lg.transmission(xr[m], yr[m], coeffs=np.full((4, 7), float(aperture_fourier(e)[0, 0]) - lg.R1), nray=NRAY, smax=0.7, sig=SIG, refl=refl, seed=21)
inn = R < 8.0
A = np.sum(S[inn] * T[inn]) / np.sum(T[inn] ** 2); Aid = np.sum(S[inn] * Tid[inn]) / np.sum(Tid[inn] ** 2)
M = A * T; res = S - M; resid_ideal = S - Aid * Tid
plat = S[inn].mean()
mc = A * np.std((T - T2)[inn]) / np.sqrt(2)
print(f'amplitude A = {A:.5f} V per unit transmission; plateau {plat:.5f} V')
print(f'interior (r<8.0, {inn.sum()} pts): rms(data-model) = {rms(res[inn])/plat*100:.3f} %   rms(data-ideal cone) = {rms(resid_ideal[inn])/plat*100:.3f} %   model MC noise {mc/plat*100:.3f} %')
print(f'chi2/N with sigma_data=2.0e-4 (+MC): {np.sum(res[inn]**2)/(2.0e-4**2+mc**2)/inn.sum():.3f}')
corr = np.corrcoef((S - Aid * Tid)[inn], (M - Aid * Tid)[inn])[0, 1]
print(f'correlation of non-ideal structure, data vs model: {corr:.3f}')
allm = R < 8.9
print(f'whole aperture incl. edge (r<8.9): rms resid {rms(res[allm])/plat*100:.3f} %')
# deformation on a fine grid
ph = np.linspace(0, 2 * np.pi, 361); zz = np.linspace(0, 100, 201)
cp = lg.pad_coeffs(c)
DEL = np.array([[lg.delta_eval(cp, p % (2 * np.pi), z)[0] for z in zz] for p in ph])
print(f'wall deformation: mean {DEL.mean():+.3f} mm, rms about m=0 profile {np.std(DEL - DEL.mean(axis=0)):.3f} mm, min {DEL.min():+.3f}, max {DEL.max():+.3f}')
np.savez(f'../out/final_{tag}.npz', x=d['x'], y=d['y'], S=S, M=M, T=T, A=A, Mideal=Aid * Tid, R=R, cx=e['cx'], cy=e['cy'], phi=ph, z=zz, DEL=DEL, c=c, refl=refl, mc=mc)
fig, ax = plt.subplots(2, 3, figsize=(20, 12)); ext = [d['x'][0] - .1, d['x'][-1] + .1, d['y'][0] - .1, d['y'][-1] + .1]
lo, hi = np.percentile(S[inn], [0.5, 99.5])
for a, img, t in ((ax[0, 0], S, 'data'), (ax[0, 1], M, 'simulation with fitted wall')):
    im = a.imshow(img.T, origin='lower', extent=ext, cmap='turbo', vmin=lo * 0.97, vmax=hi * 1.01); plt.colorbar(im, ax=a); a.set_title(t)
im = ax[0, 2].imshow((res / plat * 100).T, origin='lower', extent=ext, cmap='RdBu_r', vmin=-4, vmax=4); plt.colorbar(im, ax=ax[0, 2]); ax[0, 2].set_title('data - simulation [% of plateau]')
im = ax[1, 0].imshow(DEL.T, origin='lower', extent=[0, 360, 0, 100], aspect='auto', cmap='RdBu_r', vmin=-np.abs(DEL).max(), vmax=np.abs(DEL).max()); plt.colorbar(im, ax=ax[1, 0]); ax[1, 0].set_title('delta(phi, z) [mm]  (<0: bump into cavity)'); ax[1, 0].set_xlabel('phi [deg]'); ax[1, 0].set_ylabel('z [mm]')
D1 = DEL - DEL.mean(axis=0)
im = ax[1, 1].imshow(D1.T, origin='lower', extent=[0, 360, 0, 100], aspect='auto', cmap='RdBu_r', vmin=-np.abs(D1).max(), vmax=np.abs(D1).max()); plt.colorbar(im, ax=ax[1, 1]); ax[1, 1].set_title('non-axisymmetric part [mm]')
ax[1, 2].plot(zz, DEL.mean(axis=0)); ax[1, 2].set_title('m=0 profile [mm]'); ax[1, 2].grid()
plt.tight_layout(); plt.savefig(f'../out/final_{tag}.png', dpi=60)
