# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Step 1: measure the entrance aperture directly from the sharp edge of the scan:
centre (x0, y0), edge radius r_ap(phi), light-spot sigma, pedestal, noise."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import ndtr
from load_scan import load
d = load(); x, y, V = d['x'], d['y'], d['V']
X, Y = np.meshgrid(x, y, indexing='ij')
# first guess of the centre: centroid of the points above half of the plateau
rough = V - np.median(V[d['passed'] == 0])
bright = rough > 0.5 * np.percentile(rough, 90)
cx, cy = float(X[bright].mean()), float(Y[bright].mean())
far = np.hypot(X - cx, Y - cy) > 10.0
ped = np.median(V[far]); ped_rms = V[far].std()
print(f'pedestal {ped:.6f} V   rms(outside, 30 events) {ped_rms:.2e}')
S = V - ped
NB = 48
def edge_fit(cx, cy):
    R = np.hypot(X - cx, Y - cy); PH = np.arctan2(Y - cy, X - cx) % (2 * np.pi)
    out = []
    for b in range(NB):
        pc = (b + 0.5) * 2 * np.pi / NB
        dphi = (PH - pc + np.pi) % (2 * np.pi) - np.pi
        m = (np.abs(dphi) < np.pi / NB) & (R > 7.6) & (R < 9.6)
        r, s = R[m], S[m]
        f = lambda p: (p[0] + p[1] * (r - 8.5)) * ndtr((p[2] - r) / p[3]) - s
        res = least_squares(f, [0.021, -0.003, 8.65, 0.15], bounds=([0, -0.05, 8, 0.02], [0.1, 0.05, 9.3, 0.6]))
        out.append([pc, *res.x, np.sqrt(np.mean(res.fun ** 2))])
    return np.array(out)
for it in range(6):
    E = edge_fit(cx, cy)
    c1 = 2 * np.mean(E[:, 3] * np.cos(E[:, 0])); s1 = 2 * np.mean(E[:, 3] * np.sin(E[:, 0]))
    cx += c1; cy += s1
    print(f'iter {it}: centre ({cx:.4f}, {cy:.4f})  shift ({c1:+.4f}, {s1:+.4f})')
E = edge_fit(cx, cy)
ph, lev, slope, re, sg, rms = E.T
print(f'mean edge radius {re.mean():.4f} mm (nominal 8.75)   spot sigma median {np.median(sg):.3f} mm  [{sg.min():.3f}, {sg.max():.3f}]')
four = {}
for m in range(0, 9):
    a = np.mean(re * np.cos(m * ph)) * (1 if m == 0 else 2); b = np.mean(re * np.sin(m * ph)) * 2
    four[m] = (a, b); print(f'  m={m}: cos {a:+.4f}  sin {b:+.4f}  amp {np.hypot(a, b):.4f}')
np.savez('../out/edge.npz', cx=cx, cy=cy, ped=ped, ped_rms=ped_rms, phi=ph, r_edge=re, sigma=sg, level=lev,
         fourier=np.array([four[m] for m in range(9)]))
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(np.degrees(ph), re, 'o-'); ax[0].axhline(8.75, color='k', ls='--'); ax[0].set_title('edge radius vs phi'); ax[0].grid()
ax[1].plot(np.degrees(ph), sg, 'o-'); ax[1].set_title('edge sigma'); ax[1].grid()
ax[2].plot(np.degrees(ph), lev, 'o-'); ax[2].set_title('level at edge'); ax[2].grid()
plt.tight_layout(); plt.savefig('../out/edge.png', dpi=70)
