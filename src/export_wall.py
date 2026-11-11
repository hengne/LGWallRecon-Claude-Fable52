# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Numerical export of the fitted wall.  usage: export_wall.py <final npz> <out dir>"""
import sys, os, json, numpy as np, lgtrace as lg
f = np.load(sys.argv[1]); OUT = sys.argv[2]; c = f['c']; cp = lg.pad_coeffs(c)
ph = np.arange(0, 360, 1.0); z = np.arange(0, 100.01, 0.5)
D = np.array([[lg.delta_eval(cp, np.radians(p), zz)[0] for zz in z] for p in ph])
prof = D.mean(axis=0)
with open(os.path.join(OUT, 'wall_delta.csv'), 'w') as o:
    o.write('# radial deviation of the reconstructed inner wall from the CAD cone, delta(phi, z) [mm]\n')
    o.write('# r(phi, z) = 8.75 - 0.0475 * z + delta ; delta < 0: wall bulges into the cavity ; phi counter-clockwise from scan +x, seen from the light source\n')
    o.write('phi_deg,z_mm,r_cad_mm,delta_mm,delta_azimuthal_mean_mm,delta_local_mm\n')
    for i, p in enumerate(ph):
        for j, zz in enumerate(z):
            o.write(f'{p:.0f},{zz:.1f},{8.75 - 0.0475 * zz:.4f},{D[i, j]:.4f},{prof[j]:.4f},{D[i, j] - prof[j]:.4f}\n')
json.dump(dict(comment='uniform cubic B-spline coefficients c[i][j] of delta(phi, z) in mm; phi periodic with NPHI knots (phi_i = 2 pi i / NPHI), z knots at (j - 1) * 100 / NZ mm; see src/lgtrace.py',
               NPHI=int(c.shape[0]), NZ=int(c.shape[1] - 3), c=[[round(float(v), 6) for v in row] for row in c]), open(os.path.join(OUT, 'wall_coeffs.json'), 'w'))
print('wall_delta.csv', os.path.getsize(os.path.join(OUT, 'wall_delta.csv')) // 1024, 'kB')
