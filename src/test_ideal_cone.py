# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Test: the ideal cone.

For a Lambertian source the transmission averaged over the entrance area equals (R2 / R1)^2 = 0.20898
exactly (conservation of etendue: every ray started backwards from the exit reaches the entrance).
At a single source position it is only nearly that: the closest approach of a ray to the apex of the cone
is conserved by every reflection, so a ray is transmitted if it would pass a sphere through the exit rim
centred on the apex; the exit is a flat disc, not that spherical cap, and the transmission therefore falls
slowly from +0.22 % (centre) to -0.14 % (rim) relative to the area ratio.

Checked: (1) the spline path with zero deformation gives the same numbers as the analytic-cone path,
(2) the area-weighted mean equals (R2 / R1)^2 within 2e-4 (relative), (3) the centre and the rim deviate by
+0.22 % and about -0.14 %, (4) truncating the launch cone at sin(theta) = 0.7 loses nothing.
Writes ../out/ideal_cone.json; exits non-zero if a check fails.

usage: test_ideal_cone.py
"""
import sys, os, json, numpy as np
import lgtrace as lg
NR = 400000
rho = np.linspace(0.0, 8.75, 36)[:-1] + 0.125          # mid-points of 35 radial bins
z = np.zeros_like(rho)
T = lg.transmission(rho, z, nray=NR, gpu=False)
T0 = lg.transmission(rho, z, coeffs=np.zeros((12, 11)), nray=NR, gpu=False)
Ts = lg.transmission(rho, z, nray=NR, smax=0.7, gpu=False)
Tc = float(lg.transmission(np.array([0.0]), np.array([0.0]), nray=4 * NR, gpu=False)[0])
ar = (lg.R2 / lg.R1) ** 2; mean = float(np.sum(T * rho) / np.sum(rho))
out = dict(area_ratio=ar, mean_T=mean, mean_dev_rel=mean / ar - 1, centre_dev_rel=Tc / ar - 1, rim_dev_rel=float(T[-1] / ar - 1),
           max_flat_vs_spline=float(np.abs(T - T0).max()), max_trunc_loss=float(np.abs(Ts - T).max()))
out['ok'] = bool(abs(out['mean_dev_rel']) < 2e-4 and abs(out['centre_dev_rel'] - 0.00224) < 3e-4 and -0.0025 < out['rim_dev_rel'] < -0.0005
                 and out['max_flat_vs_spline'] < 1e-12 and out['max_trunc_loss'] < 2e-4)
print(f"area ratio (R2/R1)^2 = {ar:.6f};  area-weighted mean T = {mean:.6f}  ({out['mean_dev_rel']:+.2e} relative)")
print(f"centre: {out['centre_dev_rel']:+.3%}   rim (rho = {rho[-1]:.2f} mm): {out['rim_dev_rel']:+.3%}   relative to the area ratio")
print(f"max |analytic cone - spline(0)| = {out['max_flat_vs_spline']:.1e};  max change when sin(theta) <= 0.7 only: {out['max_trunc_loss']:.1e}")
for r, t in zip(rho[::5], T[::5]): print(f'  rho {r:5.2f} mm   T {t:.5f}')
print('PASS' if out['ok'] else 'FAIL')
if os.path.isdir('../out'): json.dump(out, open('../out/ideal_cone.json', 'w'), indent=1)
sys.exit(0 if out['ok'] else 1)
