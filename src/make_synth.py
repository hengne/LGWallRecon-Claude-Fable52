# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Closure test input: a KNOWN wall (axisymmetric base + a handful of bumps and dents) is ray traced into a
synthetic scan with the measured noise.  The same reconstruction chain is then run on it.
The truth lives on a finer spline grid (48 x 23) than the fit (24 x 13), so it is not exactly representable.
usage: make_synth.py <stageA npz> <out npz> [seed]"""
import sys, numpy as np, lgtrace as lg
from load_scan import load, spot
a = np.load(sys.argv[1]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rng = np.random.default_rng(seed)
e = np.load('../out/edge.npz'); d = load(); SIG = spot(e)
NZ0 = int(a['NZ']); p = a['p'][:NZ0 + 3]
NPHI, NZ = 48, 20; nk = NZ + 3
phi = np.arange(NPHI) * 2 * np.pi / NPHI; zk = (np.arange(nk) - 1) * lg.L / NZ
cp0 = lg.pad_coeffs(np.tile(p, (4, 1)))
base = np.array([lg.delta_eval(cp0, 0.1, float(np.clip(z, 0, 100)))[0] for z in zk])
PH, Z = np.meshgrid(phi, zk, indexing='ij'); r0 = lg.R1 + lg.SLOPE * np.clip(Z, 0, 100)
c = np.tile(base, (NPHI, 1)); bumps = []
for k in range(7):
    ph0 = rng.uniform(0, 2 * np.pi); z0 = rng.uniform(12, 92); amp = rng.choice([-1, 1]) * rng.uniform(0.15, 0.3)
    sa = rng.uniform(2.0, 4.0); sz = rng.uniform(6, 12)       # arc sigma [mm], z sigma [mm]
    dphi = (PH - ph0 + np.pi) % (2 * np.pi) - np.pi
    c += amp * np.exp(-(dphi * r0) ** 2 / 2 / sa ** 2 - (Z - z0) ** 2 / 2 / sz ** 2)
    bumps.append([ph0, z0, amp, sa, sz])
X, Y = np.meshgrid(d['x'], d['y'], indexing='ij'); xr, yr = X - e['cx'], Y - e['cy']; R = np.hypot(xr, yr)
m = R < 9.5; T = np.zeros(X.shape)
T[m] = lg.transmission(xr[m], yr[m], coeffs=c, nray=20000, smax=0.55, sig=SIG, seed=123)
A = 0.0262 / T[R < 5].mean()
S = A * T + 2.0e-4 * rng.standard_normal(T.shape)
np.savez(sys.argv[2], S=S, c_true=c, bumps=np.array(bumps), A=A)
print('synthetic scan written; bumps (phi[deg], z, amp, sigma_arc, sigma_z):'); print(np.round(np.array(bumps) * [180 / np.pi, 1, 1, 1, 1], 2))
