# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Test: the ideal cone.  For a Lambertian point source the transmission must equal (R2 / R1)^2 = 0.2090
at every source position (conservation of etendue), for the analytic-cone path and for the spline path
with zero deformation alike.

usage: test_ideal_cone.py
"""
import time, numpy as np
import lgtrace as lg
rho = np.linspace(0, 8.7, 30)
z = np.zeros_like(rho)
t0=time.time(); T = lg.transmission(rho, z, nray=200); print('compile', time.time()-t0)
t0=time.time(); T = lg.transmission(rho, z, nray=200000); print('flat   ', time.time()-t0)
C0 = np.zeros((12, 11))
t0=time.time(); T0 = lg.transmission(rho, z, coeffs=C0, nray=200); print('compile', time.time()-t0)
t0=time.time(); T0 = lg.transmission(rho, z, coeffs=C0, nray=200000); print('spline0', time.time()-t0)
print('max |flat - spline(0)|', np.abs(T - T0).max())
for sm in [0.6, 0.7, 0.8]:
    Ts = lg.transmission(rho, z, nray=200000, smax=sm); print('smax', sm, 'max diff vs full', np.abs(Ts - T).max())
print('etendue limit (R2/R1)^2 =', (4/8.75)**2, ' area-weighted mean T =', np.trapezoid(T*rho, rho)/np.trapezoid(rho, rho))
for r, t in zip(rho, T): print(f'{r:5.2f} {t:.4f}')
