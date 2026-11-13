# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Test: a grazing hit is not skipped by the ray-wall intersection.

A ray runs parallel to the nominal cone, 11 um inside it, and a smooth ring-shaped bump 11.3 um high makes it
dip 0.3 um into the wall over 0.7 mm of path.  A march that only looks for a sign change of f between samples
further apart than that misses the hit; the tracer also follows the derivative of f along the ray and tests
the maximum between two samples.  Checked for the CPU tracer and, if node is installed, for web/lgtrace.js.
Then the transmission of a rough random wall is compared between the production march step (0.5 mm) and a
ten times finer one: not a single ray may change.  Exits non-zero if a check fails.

usage: test_grazing_hit.py
"""
import sys, os, shutil, subprocess, importlib.util, tempfile, numpy as np, lgtrace as lg
c = np.zeros((4, 43)); c[:, 22] = -0.017; cp = lg.pad_coeffs(c); dlo, dhi = lg.delta_bounds(cp)
d = np.array([lg.SLOPE, 0.0, 1.0]); d /= np.linalg.norm(d); hit = np.zeros(5)
lg.trace_ray(cp, False, dlo, dhi, 8.739, 0.0, 0.0, d[0], d[1], d[2], 1.0, 2000, hit)
ok = int(hit[3]) == 1 and abs(hit[0] - 3.8203) < 2e-4
print(f'CPU tracer: {int(hit[3])} bounce(s), exit x = {hit[0]:.4f} mm (expected 1 bounce, 3.8203)   {"PASS" if ok else "FAIL"}')
if shutil.which('node'):
    js = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'web', 'lgtrace.js')
    code = (f"const LG=require({js!r});const nk=43,c=new Array(4*nk).fill(0);for(let i=0;i<4;i++)c[i*nk+22]=-0.017;LG.setWall(c,4,nk);"
            f"const r=LG.trace(8.739,0,0,{float(d[0])!r},0,{float(d[2])!r});console.log(r.bounces,r.pts[r.pts.length-3]);")
    nb, x = subprocess.run(['node', '-e', code], capture_output=True, text=True).stdout.split()
    good = int(nb) == 1 and abs(float(x) - 3.8203) < 2e-4; ok = ok and good
    print(f'JS tracer : {nb} bounce(s), exit x = {float(x):.4f} mm   {"PASS" if good else "FAIL"}')
# production step against a ten times finer march, same rays
src = open(lg.__file__, encoding='utf-8').read().replace('STEP = 0.5 ', 'STEP = 0.05 ').replace('cache=True', 'cache=False')
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, 'lgtrace_fine.py'); open(p, 'w', encoding='utf-8').write(src)
    sp = importlib.util.spec_from_file_location('lgtrace_fine', p); fine = importlib.util.module_from_spec(sp); sp.loader.exec_module(fine)
    rng = np.random.default_rng(11); cw = 0.05 * rng.standard_normal((24, 43)); xy = (rng.random((300, 2)) - 0.5) * 11
    kw = dict(coeffs=cw, nray=10000, smax=0.55, sig=-0.34, seed=3, gpu=False)
    Ta = lg.transmission(xy[:, 0], xy[:, 1], **kw); Tb = fine.transmission(xy[:, 0], xy[:, 1], **kw)
n = int(round(np.abs(Ta - Tb).sum() / (0.55 ** 2 / 10000))); good = n == 0; ok = ok and good
print(f'rough wall (50 um RMS coefficients), 3e6 rays: {n} rays change between a 0.5 mm and a 0.05 mm march   {"PASS" if good else "FAIL"}')
sys.exit(0 if ok else 1)
