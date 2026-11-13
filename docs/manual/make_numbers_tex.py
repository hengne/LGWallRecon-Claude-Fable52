# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Turn results/numbers.json into LaTeX macros (numbers.tex) so that both manuals quote the same values.

usage: make_numbers_tex.py [github url] [gitlab url]
The repository URLs may also be given through LG_GITHUB_URL / LG_GITLAB_URL or as "github" / "gitlab"
keys of numbers.json.  Optional blocks (ideal, g4_ideal, g4, gpu, recip, jacobian, refl_fit, prev, closure, resolution)
that are missing give "n/a".
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'results', 'numbers.json')
N = json.load(open(SRC)) if os.path.exists(SRC) else {}
NA = 'n/a'


def num(v, fmt, sign=False):
    """Format a number for text mode; the minus sign is typeset in math mode."""
    if v is None:
        return NA
    s = format(abs(v), fmt)
    if v < 0 and float(s) != 0:
        return r'\ensuremath{-}' + s
    return (r'\ensuremath{+}' if sign else '') + s


def get(*keys):
    d = N
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def url(arg_index, env, key):
    u = sys.argv[arg_index] if len(sys.argv) > arg_index and sys.argv[arg_index] else os.environ.get(env) or N.get(key)
    return r'\url{%s}' % u if u else r'\UrlPending'


def sci(v, sign=False):
    """a x 10^b in math mode (two significant digits)."""
    if v is None:
        return NA
    m, ex = f'{abs(v):.1e}'.split('e')
    pre = '-' if v < 0 else ('+' if sign else '')
    return r'\ensuremath{%s%s\times10^{%d}}' % (pre, m, int(ex))


def pct(v):
    """A percentage: one decimal below 10, none above."""
    return num(v, '.1f' if v is not None and abs(v) < 10 else '.0f')


def small(v):
    """A length in micrometres that may be far below one."""
    return num(v, '.3f' if v is not None and abs(v) < 0.1 else ('.2f' if v is not None and abs(v) < 10 else '.1f'))


r_ap = get('r_ap')
macros = [
    # --- the fit to the measured scan
    ('Npts', num(get('npts'), 'd')),                        # scan points inside the fit radius
    ('RmsFit', num(get('rms'), '.1f')),                     # data - simulation, inner aperture, RMS [%]
    ('RmsFitTwo', num(get('rms'), '.2f')),
    ('BiasFit', num(get('bias'), '.2f', sign=True)),        # mean of data - simulation [%]
    ('RmsIdeal', num(get('rms_id'), '.1f')),                # data - ideal cone [%]
    ('ChiTwo', num(get('chi2'), '.2f')),
    ('CorrStruct', num(get('corr'), '.2f')),                # correlation of the non-ideal structure
    ('RmsRing', num(get('rms_ring'), '.0f')),               # per-point RMS in the outer ring [%]
    ('BiasRing', num(get('bias_ring'), '.1f', sign=True)),  # mean residual in the outer ring [%]
    ('RingWorst', num(get('ring_worst'), '.1f', sign=True)),  # most negative azimuthal mean in a 0.1 mm annulus of that ring [%]
    ('NoisePct', num(get('noise'), '.1f')),                 # single-point noise of the scan [% of plateau]
    ('McNoise', num(get('mc_noise'), '.2f')),               # Monte Carlo noise of the final model map [%]
    ('LocRms', num(get('loc_rms'), '.0f')),                 # local unevenness RMS [um]
    ('LocRmsOne', num(get('loc_rms'), '.1f')),
    ('LocMin', num(get('loc_min'), '.0f')),                 # strongest bump [um] (negative)
    ('LocMax', num(get('loc_max'), '.0f', sign=True)),      # strongest dent [um]
    ('MeanMin', num(get('mean_min'), '.2f')),               # azimuthal mean profile range [mm]
    ('MeanMax', num(get('mean_max'), '.2f')),
    ('RAp', num(r_ap, '.2f')),                              # measured entrance radius [mm]
    ('RApDiff', num(None if r_ap is None else 8.75 - r_ap, '.2f')),
    ('SpotR', num(get('spot_r'), '.2f')),                   # light-spot radius [mm]
    ('Amplitude', num(get('amplitude'), '.4f')),            # volts per unit transmission
    ('Plateau', num(get('plateau'), '.4f')),                # mean signal of the inner aperture [V]
    # --- ideal cone: tracer self-test and Geant4 (G4Cons)
    ('IdealCentreDev', num(get('ideal', 'centre_dev'), '.2f', sign=True)),   # T(centre) relative to (R2/R1)^2 [%]
    ('IdealRimDev', num(get('ideal', 'rim_dev'), '.2f', sign=True)),         # T(rim) relative to (R2/R1)^2 [%]
    ('GfourIdealT', num(get('g4_ideal', 'T'), '.4f')),
    ('GfourIdealTErr', num(get('g4_ideal', 'T_err'), '.4f')),
    ('GfourIdealRatio', num(get('g4_ideal', 'ratio'), '.4f')),
    ('GfourIdealRatioErr', num(get('g4_ideal', 'ratio_err'), '.4f')),
    ('GfourIdealNpoints', num(get('g4_ideal', 'npoints'), 'd')),
    ('GfourIdealNphot', num(get('g4_ideal', 'nphot'), 'd')),
    # --- Geant4 on the fitted wall
    ('GfourRatio', num(get('g4', 'ratio'), '.4f')),
    ('GfourRatioErr', num(get('g4', 'ratio_err'), '.4f')),
    ('GfourNpoints', num(get('g4', 'npoints'), 'd')),
    ('GfourNphot', num(get('g4', 'nphot'), 'd')),
    ('GfourPull', num(get('g4', 'pull_rms'), '.2f')),
    ('GfourCorr', num(get('g4', 'corr'), '.2f')),
    # --- GPU against CPU
    ('GpuNpoints', num(get('gpu', 'npoints'), 'd')),
    ('GpuNray', num(get('gpu', 'nray'), 'd')),
    ('GpuMeanDiff', sci(get('gpu', 'mean_diff'), sign=True)),       # single precision, mean relative difference
    ('GpuMeanDiffErr', sci(get('gpu', 'mean_diff_err'))),
    ('GpuRmsDiff', sci(get('gpu', 'rms_diff'))),                    # single precision, point-to-point scatter (relative)
    # --- reciprocity
    ('RecipRatio', num(get('recip', 'ratio'), '.3f')),
    ('RecipErr', num(get('recip', 'err'), '.3f')),
    # --- linear kernels against exact finite changes of the map (test_jacobian.py)
    ('JacCorrMin', num(get('jacobian', 'corr_min'), '.2f')),                 # non-axisymmetric smooth fields
    ('JacCorrMax', num(get('jacobian', 'corr_max'), '.2f')),
    ('JacSlopeMin', num(get('jacobian', 'slope_min'), '.2f')),
    ('JacSlopeMax', num(get('jacobian', 'slope_max'), '.2f')),
    ('JacStepCorrMin', num(get('jacobian', 'm0_step_corr_min'), '.2f')),     # axisymmetric step
    ('JacStepCorrMax', num(get('jacobian', 'm0_step_corr_max'), '.2f')),
    ('JacBumpCorrMin', num(get('jacobian', 'm0_smooth_corr_min'), '.2f')),   # gentle axisymmetric bump
    ('JacBumpCorrMax', num(get('jacobian', 'm0_smooth_corr_max'), '.2f')),
    # --- axisymmetric fit with a free mirror reflectivity
    ('ReflFit', num(get('refl_fit', 'refl'), '.2f')),
    ('ReflChi', num(get('refl_fit', 'chi2'), '.0f')),
    ('ReflChiFixed', num(get('refl_fit', 'chi2_fixed'), '.0f')),
    ('ReflNpts', num(get('refl_fit', 'npts'), 'd')),
    # --- the wall of an earlier fit of the same scan, traced with the present tracer (compare_walls.py)
    ('PrevRmsEarlier', num(get('prev', 'rms_earlier'), '.2f')),              # data - earlier wall [%]
    ('PrevRmsNow', num(get('prev', 'rms_now'), '.2f')),                      # data - present wall [%]
    ('PrevMapDiff', num(get('prev', 'map_diff'), '.2f')),                    # the two simulated maps against each other [%]
    ('PrevCorr', num(get('prev', 'corr_local'), '.2f')),                     # local unevenness, whole wall
    ('PrevCorrEntrance', num(get('prev', 'corr_entrance'), '.2f')),          # ... z = 0-20 mm
    ('PrevCorrRest', ', '.join(num(v, '.2f', sign=True) for v in get('prev', 'corr_rest')) if get('prev', 'corr_rest') else NA),   # ... the four 20 mm bands further in
    ('PrevLocRms', num(get('prev', 'loc_rms_earlier'), '.1f')),              # local unevenness of the earlier wall [um]
    ('PrevLocDiff', num(get('prev', 'loc_diff'), '.0f')),                    # RMS difference of the local unevenness [um]
    ('PrevProfDiff', num(get('prev', 'prof_diff'), '.0f')),                  # azimuthal mean profiles, RMS difference [um]
    ('PrevProfDiffMax', num(get('prev', 'prof_diff_max'), '.0f')),           # ... largest difference [um]
    # --- closure test
    ('ClCorrAll', num(get('closure', 'corr_all'), '.2f')),
    ('ClRmsTrue', num(get('closure', 'rms_true'), '.0f')),
    ('ClRmsRec', num(get('closure', 'rms_rec'), '.0f')),
    ('ClRmsErr', num(get('closure', 'rms_err'), '.0f')),
    ('ClMapRms', num(get('closure', 'map_rms'), '.2f')),
    ('ClChiTwo', num(get('closure', 'chi2'), '.2f')),
    ('ClProfErr', num(get('closure', 'prof_err'), '.0f')),                   # profile error, RMS [um]
    ('ClProfBias', num(get('closure', 'prof_bias'), '.0f', sign=True)),      # profile error, mean [um]
    # --- resolution study (explicit cut-offs: the best 30, 100, 150 patterns)
    ('ResCThirty', num(get('resolution', 'c30'), '.2f')),
    ('ResCHundred', num(get('resolution', 'c100'), '.2f')),
    ('ResCOneFifty', num(get('resolution', 'c150'), '.2f')),
    ('ResFThirty', pct(get('resolution', 'f30'))),
    ('ResFHundred', pct(get('resolution', 'f100'))),
    ('ResFOneFifty', pct(get('resolution', 'f150'))),
    ('ResAHundred', num(get('resolution', 'a100'), '.2f')),                  # amplitude ratio recovered / true
    ('ResSThirty', small(get('resolution', 's30'))),                         # amplitude uncertainty, surface RMS [um]
    ('ResSHundred', small(get('resolution', 's100'))),
]
res = get('resolution')
if res and all(k in res for k in ('k', 'amin_surf_um', 'frac_true', 'corr', 'amp_ratio')):
    rows = ' \\\\\n  '.join(f'{k} & {small(a)} & {num(100 * f, ".1f")} & {num(c, ".2f", sign=True)} & {num(r, ".2f")}'
                           for k, a, f, c, r in zip(res['k'], res['amin_surf_um'], res['frac_true'], res['corr'], res['amp_ratio']))
else:
    rows = ' & '.join([NA] * 5)
out = ['% Generated by make_numbers_tex.py from results/numbers.json -- do not edit by hand.',
       r'\providecommand{\UrlPending}{to be added}']
out += [r'\newcommand{\%s}{%s}' % (k, v) for k, v in macros]
out.append(r'\newcommand{\ResolutionRows}{%' + '\n  ' + rows + r' \\' + '\n}')
out.append(r'\newcommand{\GithubURL}{%s}' % url(1, 'LG_GITHUB_URL', 'github'))
out.append(r'\newcommand{\GitlabURL}{%s}' % url(2, 'LG_GITLAB_URL', 'gitlab'))
open(os.path.join(HERE, 'numbers.tex'), 'w').write('\n'.join(out) + '\n')
print('numbers.tex:', len(macros) + 3, 'macros', '(numbers.json missing!)' if not N else '')
