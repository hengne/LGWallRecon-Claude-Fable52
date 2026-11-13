# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Compare Geant4 with the ray tracer at the same source points, with the same source and mirror.

  fitted wall:  g4_compare.py <fit npz> <params.txt> <g4 out file(s)...>     -> ../out/g4_compare.json
  ideal cone:   g4_compare.py ideal     <params.txt> <g4 out file(s)...>     -> ../out/g4_ideal.json
params.txt (written by export_mesh.py) holds: spot  smax  reflectivity, as given to lgsim.
The quoted uncertainty is the binomial error of the Geant4 photon counts; the tracer's own statistical
error (400,000 rays per point) is negligible next to it.
"""
import sys, json, numpy as np, lgtrace as lg
from load_scan import wall_sha1
ideal = sys.argv[1] == 'ideal'
SPOT, SMAX, REFL = (float(v) for v in open(sys.argv[2]).read().split()[:3])
rows = np.vstack([np.loadtxt(p, ndmin=2) for p in sys.argv[3:]])
x, y, det, emit, Tg = rows.T
c = None if ideal else np.load(sys.argv[1])['c']
Tm = lg.transmission(x, y, coeffs=c, nray=400000, smax=SMAX, sig=SPOT, refl=REFL, seed=77, blur=0.0, gpu=False)
p = Tg / SMAX ** 2; err = SMAX ** 2 * np.sqrt(np.clip(p * (1 - p), 1e-9, None) / emit)
pull = (Tg - Tm) / err
out = dict(npoints=int(len(x)), photons_per_point=int(emit[0]), spot=SPOT, smax=SMAX, refl=REFL,
           mean_T_geant4=float(Tg.mean()), mean_T_tracer=float(Tm.mean()), mean_T_err=float(np.sqrt(np.sum(err ** 2)) / len(x)),
           ratio=float(Tg.mean() / Tm.mean()), ratio_err=float(np.sqrt(np.sum(err ** 2)) / len(x) / Tm.mean()),
           pull_mean=float(pull.mean()), pull_rms=float(np.sqrt(np.mean(pull ** 2))), corr=float(np.corrcoef(Tg, Tm)[0, 1]),
           T_min=float(Tm.min()), T_max=float(Tm.max()), wall_sha1='ideal' if ideal else wall_sha1(c))
print(json.dumps(out, indent=1))
name = 'g4_ideal' if ideal else 'g4_compare'
json.dump(out, open(f'../out/{name}.json', 'w'), indent=1)
np.savez(f'../out/{name}.npz', x=x, y=y, Tg=Tg, Tm=Tm, err=err)
