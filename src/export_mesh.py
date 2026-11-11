# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Export a fitted wall as a (phi, z) radius mesh for the Geant4 tessellated cavity, and a list of test points.
usage: export_mesh.py <fit/final npz with c> <mesh.txt> <points.txt> [nphi=1440] [nz=251] [npoints=300]"""
import sys, numpy as np, lgtrace as lg
f = np.load(sys.argv[1]); c = f['c']; cp = lg.pad_coeffs(c)
NPH = int(sys.argv[4]) if len(sys.argv) > 4 else 1440; NZM = int(sys.argv[5]) if len(sys.argv) > 5 else 251
NPT = int(sys.argv[6]) if len(sys.argv) > 6 else 300
g = lg._delta_grid(cp, NPH, NZM) / 1.0                       # delta[phi_i, z_j]  (phi_i = 2 pi i / NPH * 0.999999)
z = np.linspace(0, lg.L, NZM)
r = lg.R1 + lg.SLOPE * z[None, :] + g
with open(sys.argv[2], 'w') as o:
    o.write(f'{NPH} {NZM}\n')
    for j in range(NZM):
        o.write(f'{z[j]:.6f} ' + ' '.join(f'{v:.6f}' for v in r[:, j]) + '\n')
# test points: golden-angle spiral, uniform in area, out to r = 8.0
k = np.arange(NPT) + 0.5; rr = 8.0 * np.sqrt(k / NPT); th = k * np.pi * (3 - np.sqrt(5))
np.savetxt(sys.argv[3], np.column_stack([rr * np.cos(th), rr * np.sin(th)]), fmt='%.5f')
print('mesh', r.shape, 'r range', r.min(), r.max(), '; points', NPT)
