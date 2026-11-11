# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Compare Geant4 (tessellated cavity built from the fitted wall) with the ray tracer at the same source points.
usage: g4_compare.py <fit npz> <g4 out file(s)...>"""
import sys, json, numpy as np, lgtrace as lg
c = np.load(sys.argv[1])['c']
rows = np.vstack([np.loadtxt(p, ndmin=2) for p in sys.argv[2:]])
x, y, det, emit, Tg = rows.T
SMAX = 0.55
Tm = lg.transmission(x, y, coeffs=c, nray=400000, smax=SMAX, sig=-0.34, seed=77, blur=0.0)
p = Tg / SMAX ** 2; err = SMAX ** 2 * np.sqrt(np.clip(p * (1 - p), 1e-9, None) / emit)
pull = (Tg - Tm) / err
r = np.hypot(x, y)
out = dict(npoints=int(len(x)), photons_per_point=int(emit[0]), mean_T_geant4=float(Tg.mean()), mean_T_tracer=float(Tm.mean()),
           ratio=float(Tg.mean() / Tm.mean()), ratio_err=float(np.sqrt(np.sum(err ** 2)) / len(x) / Tm.mean()),
           pull_mean=float(pull.mean()), pull_rms=float(pull.std()), corr=float(np.corrcoef(Tg, Tm)[0, 1]),
           T_min=float(Tm.min()), T_max=float(Tm.max()))
print(json.dumps(out, indent=1))
json.dump(out, open('../out/g4_compare.json', 'w'), indent=1)
np.savez('../out/g4_compare.npz', x=x, y=y, Tg=Tg, Tm=Tm, err=err)
