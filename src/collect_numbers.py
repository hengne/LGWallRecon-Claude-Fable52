# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Collect every number quoted in the report and the manuals into one JSON file.

All 'rms' entries are root mean squares (a bias counts); the bias is given separately where it matters.
Each check carries `current`: true if it was run on the wall of this result (fingerprint of the spline
coefficients; for the closure test, of the axisymmetric profile its truth is built from), false if the
file is left over from an earlier reconstruction.  The report marks the latter.

usage: collect_numbers.py <final npz> <results dir>      (writes <results dir>/numbers.json)
"""
import sys, os, json, numpy as np
from load_scan import rms, wall_sha1

f = np.load(sys.argv[1]); RES = sys.argv[2]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out')
S, M, R, DEL = f['S'], f['M'], f['R'], f['DEL']
inn = R < 8.0; plat = float(S[inn].mean()); sha = wall_sha1(f['c'])
prof = DEL[:-1].mean(axis=0); LOC = (DEL - prof)[:-1]
e = np.load(os.path.join(OUT, 'edge.npz'))
ring = (R > 8.0) & (R < 8.5)
mc = float(f['mc']) if 'mc' in f.files else 0.0
SIGD = 2.0e-4                                                              # single-point noise of the scan [V]
n = dict(
    wall_sha1=sha, npts=int(inn.sum()),
    rms=rms((S - M)[inn]) / plat * 100,                                    # data - simulation, inner aperture [% of plateau]
    bias=float((S - M)[inn].mean() / plat * 100),
    rms_id=rms((S - f['Mideal'])[inn]) / plat * 100,                       # data - ideal cone [%]
    chi2=float(np.sum((S - M)[inn] ** 2) / (SIGD ** 2 + mc ** 2) / inn.sum()),
    corr=float(np.corrcoef((S - f['Mideal'])[inn], (M - f['Mideal'])[inn])[0, 1]),
    rms_ring=rms((S - M)[ring]) / plat * 100, bias_ring=float((S - M)[ring].mean() / plat * 100),
    loc_rms=rms(LOC) * 1000, loc_min=float(LOC.min() * 1000), loc_max=float(LOC.max() * 1000),   # [um]
    mean_min=float(prof.min()), mean_max=float(prof.max()),
    r_ap=float(e['fourier'][0, 0]) + (float(e['dr_disc']) if 'dr_disc' in e.files else 0.0),
    spot_r=float(e['spot_radius']) if 'spot_radius' in e.files else 0.34,
    amplitude=float(f['A']), plateau=plat, noise=100 * SIGD / plat, mc_noise=100 * mc / plat)
# azimuthal mean of data - simulation in 0.1 mm annuli of the rim ring: the most negative one [% of plateau]
n['ring_worst'] = float(min((S - M)[(R >= a) & (R < a + 0.1)].mean() for a in np.arange(8.0, 8.45, 0.1)) / plat * 100)
# structure left unexplained after subtracting the noise in quadrature [% of plateau]
n['excess'] = float(np.sqrt(max(n['rms'] ** 2 - (100 * np.hypot(SIGD, mc) / plat) ** 2, 0.0)))


def js(name, where=OUT):
    p = os.path.join(where, name)
    return json.load(open(p)) if os.path.exists(p) else None


g = js('g4_compare.json')
if g: n['g4'] = dict(ratio=g['ratio'], ratio_err=g['ratio_err'], npoints=g['npoints'], nphot=g['photons_per_point'], pull_rms=g['pull_rms'],
                     corr=g['corr'], spot=g.get('spot'), refl=g.get('refl'), current=g.get('wall_sha1') == sha)
g = js('g4_ideal.json')
if g: n['g4_ideal'] = dict(T=g['mean_T_geant4'], T_err=g['mean_T_err'], T_tracer=g['mean_T_tracer'], ratio=g['ratio'], ratio_err=g['ratio_err'],
                           npoints=g['npoints'], nphot=g['photons_per_point'], pull_rms=g['pull_rms'], current=True)
g = js('recip.json')
if g: n['recip'] = dict(ratio=g['ratio'], err=g['err'], current=g.get('wall_sha1') == sha)
g = js('gpu_check.json')
if g and g.get('ok'):
    n['gpu'] = dict(npoints=g['npoints'], nray=g['nray'], max_diff_rays=g['double']['max_diff_rays'], mean_diff=g['single']['mean_diff_rel'],
                    mean_diff_err=g['single']['mean_diff_err_rel'], rms_diff=g['single']['rms_diff_rel'], current=g.get('wall_sha1') == sha)
g = js('ideal_cone.json')
if g and g.get('ok'): n['ideal'] = dict(area_ratio=g['area_ratio'], mean_dev=100 * g['mean_dev_rel'], centre_dev=100 * g['centre_dev_rel'], rim_dev=100 * g['rim_dev_rel'])
# axisymmetric fit with the mirror reflectivity free, next to the same fit with a perfect mirror
pa, pr = os.path.join(OUT, 'A_data.npz'), os.path.join(OUT, 'A_refl.npz')
base_sha = ''
if os.path.exists(pa):
    a = np.load(pa); base_sha = wall_sha1(a['p'][:int(a['NZ']) + 3])
    chi = lambda q: float(np.sum(((float(q['A']) * q['T'] - q['dat']) / q['err']) ** 2))
    if os.path.exists(pr):
        r = np.load(pr); n['refl_fit'] = dict(refl=float(r['p'][int(r['NZ']) + 3]), chi2=chi(r), chi2_fixed=chi(a), npts=int(len(a['dat'])))
# linear kernels against exact finite changes of the map (test_jacobian.py)
g = js('jacobian.json')
if g:
    sm = [q for st in g['starts'] for q in st['smooth']]; st_, sm_ = [st['m0_step'] for st in g['starts']], [st['m0_smooth'] for st in g['starts']]
    rng = lambda qs, k: (min(q[k] for q in qs), max(q[k] for q in qs))
    n['jacobian'] = dict(zip(('corr_min', 'corr_max'), rng(sm, 'corr')), **dict(zip(('slope_min', 'slope_max'), rng(sm, 'slope'))),
                         **dict(zip(('m0_step_corr_min', 'm0_step_corr_max'), rng(st_, 'corr'))), **dict(zip(('m0_smooth_corr_min', 'm0_smooth_corr_max'), rng(sm_, 'corr'))),
                         **dict(zip(('m0_smooth_mean_min', 'm0_smooth_mean_max'), rng(sm_, 'mean_ratio'))), current=g.get('wall_sha1') == sha)
# the wall of an earlier fit of the same scan, traced with the present tracer (compare_walls.py)
g = js('compare_prev.json')
if g:
    n['prev'] = dict(rms_earlier=g['rms_earlier'], rms_now=g['rms_now'], map_diff=g['map_diff'], corr_local=g['corr_local'], corr_entrance=g['bands'][0]['corr'],
                     corr_rest=[b['corr'] for b in g['bands'][1:]], loc_rms_earlier=g['loc_rms_earlier'], loc_diff=g['loc_diff'],
                     prof_diff=g['prof_diff'], prof_diff_max=g['prof_diff_max'], current=g.get('wall_sha1') == sha)
c = js('closure_numbers.json', RES)
if c:
    n['closure'] = dict(corr_all=c['corr_all'], rms_true=c['rms_true'] * 1000, rms_rec=c['rms_rec'] * 1000, rms_err=c['rms_err'] * 1000,
                        map_rms=c['map_rms'], chi2=c.get('chi2'), prof_err=c['prof_rms_err'] * 1000, prof_bias=c['prof_mean_err'] * 1000,
                        current=bool(base_sha) and c.get('base_sha1') == base_sha)
r = js('resolution.json')
if r and 'amin_surf_um' in r:
    k = r['k']; at = lambda key, kk: r[key][k.index(kk)]
    n['resolution'] = dict(k=k, corr=r['corr'], amp_ratio=r['amp_ratio'], frac_true=r['frac_true'], amin_um=r['amin_um'], amin_surf_um=r['amin_surf_um'],
                           c30=at('corr', 30), c100=at('corr', 100), c150=at('corr', 150),
                           f30=100 * at('frac_true', 30), f100=100 * at('frac_true', 100), f150=100 * at('frac_true', 150),
                           a100=at('amp_ratio', 100), s30=at('amin_surf_um', 30), s100=at('amin_surf_um', 100),
                           current=bool(base_sha) and r.get('base_sha1') == base_sha)
json.dump(n, open(os.path.join(RES, 'numbers.json'), 'w'), indent=1)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in n.items() if not isinstance(v, dict)}))
print('checks:', ', '.join(f"{k} ({'this wall' if v.get('current', True) else 'EARLIER RUN'})" for k, v in n.items() if isinstance(v, dict)))
