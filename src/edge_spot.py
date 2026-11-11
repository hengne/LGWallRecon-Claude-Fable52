# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Edge analysis, second pass: which light-spot shape does the edge of the scan prefer?
Gaussian (sigma) versus uniform disc (radius a).  Global fit of all edge points with the aperture
radius r_ap(phi) from edge.npz (Fourier series) + common offset dr."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr
from load_scan import load
e = np.load('../out/edge.npz'); d = load()
X, Y = np.meshgrid(d['x'], d['y'], indexing='ij'); S = d['V'] - e['ped']
xr, yr = X - e['cx'], Y - e['cy']; R = np.hypot(xr, yr); PH = np.arctan2(yr, xr)
F = e['fourier']; rap = sum(F[m, 0] * np.cos(m * PH) + F[m, 1] * np.sin(m * PH) for m in range(9))
sel = (R > 7.7) & (R < 9.3); r, s, ra, ph = R[sel], S[sel], rap[sel], PH[sel]
def disc(t):                         # fraction of a unit disc on the inner side of a straight edge at signed distance t
    t = np.clip(t, -1, 1); return 1 - (np.arccos(t) - t * np.sqrt(1 - t * t)) / np.pi
def level(p, r, ph):                 # local plateau just inside the edge: mean, slope, m=1,2 modulation
    return (p[0] + p[1] * (r - 8.3) + p[2] * (r - 8.3) ** 2) * (1 + p[3] * np.cos(ph) + p[4] * np.sin(ph) + p[5] * np.cos(2 * ph) + p[6] * np.sin(2 * ph))
out = {}
for name, fn in (('gauss', lambda t: ndtr(t)), ('disc', disc)):
    f = lambda p: level(p, np.minimum(r, ra + p[8]), ph) * fn((ra + p[8] - r) / p[7]) - s
    res = least_squares(f, [0.021, -0.004, 0, 0, 0, 0, 0, 0.2 if name == 'gauss' else 0.33, 0.0])
    print(f'{name:6s}: width {res.x[7]:.4f} mm   radius offset {res.x[8]:+.4f} mm   rms resid {np.sqrt(np.mean(res.fun**2)):.2e} V  ({len(s)} pts)')
    out[name] = res.x
np.savez('../out/edge.npz', **{k: e[k] for k in e.files if k not in ('spot_radius', 'dr_disc')}, spot_radius=out['disc'][7], dr_disc=out['disc'][8])
