# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Test: the quasi-random ray sequence fills its unit cube.

The rays of one source point are u_j(k) = frac(shift_j + k alpha_j), j = 1..6 (direction: 1, 2; position in
the light spot: 3, 4; optional blur of the spot: 5, 6).  If the alpha_j obey an integer relation
sum m_j alpha_j = integer with small m_j, the points lie on a few planes instead of filling the cube (an
earlier choice had alpha_3 + alpha_4 = 1, which put the spot samples on a spiral).  Checked here:
(1) no relation with |m_j| <= 4 comes closer to an integer than 1e-6,
(2) every pair of coordinates fills a 32 x 32 grid evenly with 65536 points (all cells within 25 % of the mean),
(3) the centroid of the sampled light spot is at its centre within 0.5 % of the radius.
Exits non-zero if a check fails.

usage: test_sampler.py
"""
import sys, itertools, numpy as np, lgtrace as lg, lgcl
QA = [int(v) for v in lg._QA]; M = 2 ** 32
worst = min((min(s, M - s), m) for m in itertools.product(range(-4, 5), repeat=6) if any(m)
            for s in [sum(a * b for a, b in zip(m, QA)) % M])
ok1 = worst[0] / M > 1e-6
print(f'closest integer relation with |m| <= 4: m = {worst[1]}, distance {worst[0] / M:.2e}   {"PASS" if ok1 else "FAIL"}')
sh = lgcl.shifts(1, 5)[0].astype(np.uint64); s = [sh[0], sh[1], sh[2], sh[3], (sh[0] * np.uint64(2654435761) + sh[3]) & lg._M32, (sh[1] * np.uint64(2246822519) + sh[2]) & lg._M32]
k = np.arange(1, 65537, dtype=np.uint64)
u = np.array([((s[j] + k * lg._QA[j]) & lg._M32).astype(np.float64) / M for j in range(6)])
dev = 0.0
for i, j in itertools.combinations(range(6), 2):
    h = np.histogram2d(u[i], u[j], bins=32, range=[[0, 1], [0, 1]])[0]; dev = max(dev, float(np.abs(h / 64.0 - 1).max()))
ok2 = dev < 0.25
print(f'largest deviation of a 32 x 32 cell count from its mean, over all 15 coordinate pairs: {dev:.1%}   {"PASS" if ok2 else "FAIL"}')
r = np.sqrt(u[2]); cx, cy = float(np.mean(r * np.cos(2 * np.pi * u[3]))), float(np.mean(r * np.sin(2 * np.pi * u[3])))
ok3 = np.hypot(cx, cy) < 0.005
print(f'centroid of the sampled disc spot: ({cx:+.4f}, {cy:+.4f}) radii   {"PASS" if ok3 else "FAIL"}')
sys.exit(0 if ok1 and ok2 and ok3 else 1)
