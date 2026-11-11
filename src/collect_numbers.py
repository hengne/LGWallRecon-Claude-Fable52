# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Collect every number quoted in the report and the manuals into one JSON file.

usage: collect_numbers.py <final npz> <results dir>      (writes <results dir>/numbers.json)
"""
import sys, os, re, json, numpy as np

f = np.load(sys.argv[1]); RES = sys.argv[2]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out')
S, M, R, DEL = f['S'], f['M'], f['R'], f['DEL']
inn = R < 8.0; plat = S[inn].mean()
prof = DEL[:-1].mean(axis=0); LOC = DEL - prof
e = np.load(os.path.join(OUT, 'edge.npz'))
ring = (R > 8.0) & (R < 8.5)
mc = float(f['mc']) if 'mc' in f.files else 0.0
n = dict(
    npts=int(inn.sum()),
    rms=float((S - M)[inn].std() / plat * 100),                    # data - simulation, inner aperture [%]
    rms_id=float((S - f['Mideal'])[inn].std() / plat * 100),       # data - ideal cone [%]
    chi2=float(np.sum((S - M)[inn] ** 2) / (2.0e-4 ** 2 + mc ** 2) / inn.sum()),
    corr=float(np.corrcoef((S - f['Mideal'])[inn], (M - f['Mideal'])[inn])[0, 1]),
    rms_ring=float((S - M)[ring].std() / plat * 100),
    loc_rms=float(LOC.std() * 1000), loc_min=float(LOC.min() * 1000), loc_max=float(LOC.max() * 1000),   # [um]
    mean_min=float(prof.min()), mean_max=float(prof.max()),
    r_ap=float(e['fourier'][0, 0]) + (float(e['dr_disc']) if 'dr_disc' in e.files else 0.0),
    spot_r=float(e['spot_radius']) if 'spot_radius' in e.files else 0.34,
    amplitude=float(f['A']), plateau=float(plat))
p = os.path.join(OUT, 'g4_compare.json')
if os.path.exists(p):
    g = json.load(open(p)); n['g4'] = dict(ratio=g['ratio'], ratio_err=g['ratio_err'], npoints=g['npoints'], nphot=g['photons_per_point'], pull_rms=g['pull_rms'], corr=g['corr'])
p = os.path.join(OUT, 'recip.log')
if os.path.exists(p):
    m = re.search(r'fitted wall.*ratio ([\d.]+) \+- ([\d.]+)', open(p).read())
    if m: n['recip'] = dict(ratio=float(m.group(1)), err=float(m.group(2)))
p = os.path.join(RES, 'closure_numbers.json')
if os.path.exists(p):
    c = json.load(open(p)); n['closure'] = dict(corr_all=c['corr_all'], rms_true=c['rms_true'] * 1000, rms_rec=c['rms_rec'] * 1000, map_rms=c['map_rms'], prof_err=c['prof_rms_err'] * 1000)
p = os.path.join(OUT, 'resolution.json')
if os.path.exists(p):
    r = json.load(open(p)); k = r['k']
    # 'visible' subspace = the 100 best-seen wall patterns; the best 30 give the upper end of the range
    n['resolution'] = dict(corr_vis_lo=r['corr'][k.index(100)], corr_vis_hi=r['corr'][k.index(30)], frac_vis=100 * r['frac_true'][k.index(100)],
                           k=k, corr=r['corr'], frac_true=r['frac_true'], amin_um=r['amin_um'])
# structure left unexplained after subtracting the noise in quadrature [% of plateau]
n['excess'] = float(np.sqrt(max(n['rms'] ** 2 - (100 * np.hypot(2.0e-4, mc) / plat) ** 2, 0.0)))
json.dump(n, open(os.path.join(RES, 'numbers.json'), 'w'), indent=1)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in n.items() if not isinstance(v, dict)}))
