# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Assemble the single-file report page of one language: text from web/i18n/<lang>.json, numbers from
numbers.json, the SVG figures inlined, the fitted wall and both maps embedded for the live 3-D demonstration.

usage: build_page.py <final npz> <results dir> <out html> <lang: zh|en> [github url] [gitlab url]
"""
import sys, os, re, json, base64, datetime, numpy as np
import svgkit as sk, i18n

f = np.load(sys.argv[1]); RES = sys.argv[2]; OUTF = sys.argv[3]; LANG = sys.argv[4]
GH = sys.argv[5] if len(sys.argv) > 5 else ''; GL = sys.argv[6] if len(sys.argv) > 6 else ''
L = i18n.load(LANG); N = json.load(open(os.path.join(RES, 'numbers.json')))
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, '..', 'web'); FIG = os.path.join(RES, LANG)
x, y, S, M, T, R, c = f['x'], f['y'], f['S'], f['M'], f['T'], f['R'], f['c']

# ---- data for the live demonstration
inn = R < 8.9
lo, hi = np.percentile(S[R < 8.0], [0.5, 99.5])
q8 = lambda img: np.where(inn, 1 + np.clip((img - lo) / (hi - lo), 0, 1) * 254, 0).astype(np.uint8)   # 0 = outside the aperture
tmax = float(T.max())
DEL = f['DEL']; LOC = DEL - DEL[:-1].mean(axis=0)
JS_KEYS = ('btn_pause', 'btn_play', 'btn_wall_fit', 'btn_wall_ideal', 'btn_map_model', 'btn_map_data', 'probe_local')
data = dict(
    c=[round(float(v), 5) for v in c.ravel()], nphi=int(c.shape[0]), nk=int(c.shape[1]),
    cx=float(f['cx']), cy=float(f['cy']), x0=float(x[0]), dx=float(x[1] - x[0]), y0=float(y[0]), dy=float(y[1] - y[0]), nx=len(x), ny=len(y),
    S8=base64.b64encode(q8(S).tobytes()).decode(), M8=base64.b64encode(q8(M).tobytes()).decode(),
    T16=base64.b64encode(np.round(T / tmax * 65535).astype('<u2').tobytes()).decode(), tmax=tmax,
    div=[sk.DIV(t) for t in np.linspace(0, 1, 41)], seq=[sk.SEQ(t) for t in np.linspace(0.08, 1, 33)],
    vloc=float(np.abs(LOC).max()), src=[-2.4, 1.6], spot=float(N['spot_r']),
    frames=[[78, 132, 800, 380], [78, 612, 800, 250]],          # panels A and C of inner_surface.svg (svg units)
    t={k: L[k] for k in JS_KEYS})


def inline_svg(name):
    """An SVG figure without its own <style> (the page stylesheet themes it)."""
    return re.sub(r'<style>.*?</style>', '', open(os.path.join(FIG, name), encoding='utf-8').read(), count=1, flags=re.S)


# ---- composed blocks
cl, rs, g4, gi, rc, gp, rf, pv, jc = (N.get(k) for k in ('closure', 'resolution', 'g4', 'g4_ideal', 'recip', 'gpu', 'refl_fit', 'prev', 'jacobian'))


class Missing:
    """Stands for a number of a check that was not run: prints as a dash whatever the format."""
    def __format__(self, spec): return '–'


class Numbers(dict):
    """numbers.json flattened for the texts: {rms}, {closure_prof_err}, {prev_corr_local}, ..."""
    def __missing__(self, key): return Missing()


NF = Numbers(N)
for blk in ('closure', 'resolution', 'prev', 'refl_fit', 'jacobian'):
    NF.update({f'{blk}_{k}': v for k, v in (N.get(blk) or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
NF['ring_low'], NF['ring_low_worst'] = -N['bias_ring'], -N['ring_worst']
corr_vis = f'{rs["c100"]:.2f}–{rs["c30"]:.2f}' if rs else '–'
minus = lambda t: re.sub(r'(?<![\w.>])-(?=\d)', '−', t)          # typographic minus sign in front of a number
facts = [(f'{N["rms"]:.1f} %', L['fact1'].format(**N)), (f'{N["rms_id"]:.1f} %', L['fact2']),
         (f'{N["loc_rms"]:.0f} µm', minus(L['fact3'].format(**N)))]
if cl and rs: facts.append((corr_vis, L['fact4'].format(**cl)))
stop = '. ' if LANG == 'en' else '。'
def verdict(good): return f'<span class="{"ok" if good else "bad"}">{L["pass"] if good else L["fail"]}</span>' + stop
def note(d, text):          # a check left over from an earlier reconstruction is labelled as such
    return text + ('' if d.get('current', True) else ' <i>' + L['earlier'] + '</i>')
def sci(v):                 # 1.2e-07 -> 1.2×10⁻⁷ as HTML
    m, ex = f'{v:.1e}'.split('e'); return f'{m}×10<sup>{int(ex)}</sup>'.replace('-', '−')
rows = []
if gi: rows.append((L['row_g4ideal_k'], L['row_g4ideal_v'].format(**gi), verdict(abs(gi['ratio'] - 1) < 4 * gi['ratio_err']) + note(gi, L['row_g4ideal_n'].format(**gi))))
if g4: rows.append((L['row_g4wall_k'], L['row_g4wall_v'].format(**g4), verdict(abs(g4['ratio'] - 1) < 4 * g4['ratio_err']) + note(g4, L['row_g4wall_n'].format(**g4))))
if gp: rows.append((L['row_gpu_k'], L['row_gpu_v'], verdict(True) + note(gp, L['row_gpu_n'].format(npoints=gp['npoints'], nray=gp['nray'], mean=sci(gp['mean_diff']), err=sci(gp['mean_diff_err']), rms=sci(gp['rms_diff'])))))
if rc: rows.append((L['row_recip_k'], L['row_recip_v'].format(**rc), verdict(abs(rc['ratio'] - 1) < 4 * rc['err']) + note(rc, L['row_recip_n'])))
if cl:
    rows.append((L['row_closure_k'], L['row_closure_v'].format(**cl), verdict(cl['chi2'] is not None and cl['chi2'] < 1.2) + note(cl, L['row_closure_n'].format(**cl))))
    rows.append((L['row_closure2_k'], L['row_closure2_v'].format(**cl), verdict(cl['corr_all'] > 0.7) + note(cl, L['row_closure2_n'].format(**cl) + (L['row_closure2_res'].format(**rs) if rs else ''))))
if pv: rows.append((L['row_prev_k'], L['row_prev_v'].format(**pv), verdict(pv['corr_local'] > 0.7) + note(pv, L['row_prev_n'].format(**pv))))
if jc: rows.append((L['row_jac_k'], L['row_jac_v'].format(**jc), note(jc, L['row_jac_n'].format(**jc))))
if rf: rows.append((L['row_refl_k'], f'R = {rf["refl"]:.2f}', L['row_refl_n'].format(**rf)))
closure_fig = ''
if cl and os.path.exists(os.path.join(FIG, 'closure.svg')):
    closure_fig = ('<figure><div class="fig"><div class="scrollx"><div class="wide">' + inline_svg('closure.svg') +
                   '</div></div></div><figcaption>' + L['cap_closure'] + '</figcaption></figure>')
link = lambda u: f'<a href="{u}">{u}</a>' if u else '–'
files = [('LGWallRecon', L['file_repo_gh'] + ': ' + link(GH)), ('', L['file_repo_gl'] + ': ' + link(GL)),
         ('docs/manual/', L['file_manual']), ('results/', L['file_wall']), ('run_all.sh', L['file_run'])]
rep = {
    'DATA_JSON': json.dumps(data, separators=(',', ':'), ensure_ascii=False),
    'LGTRACE_JS': open(os.path.join(WEB, 'lgtrace.js'), encoding='utf-8').read().replace("if (typeof module !== 'undefined') module.exports = LG;", ''),
    'SVG_WALL': inline_svg('inner_surface.svg'), 'SVG_MAPS': inline_svg('response_maps.svg'), 'FIG_CLOSURE': closure_fig,
    'LEAD': L['lead'].format(**N),
    'FACTS': ''.join(f'<div><b>{a}</b><span>{b}</span></div>' for a, b in facts),
    'P_MAPS': minus(L['p_maps'].format(**N)),
    'P_METHOD': L['p_method'].format(centre_dev=N['ideal']['centre_dev'] if 'ideal' in N else 0.22, rim_dev=-N['ideal']['rim_dev'] if 'ideal' in N else 0.14),
    'P_WALL': L['p_wall'].format(mean_lo=-N['mean_max'], mean_hi=-N['mean_min'], **N),
    'STEPS': ''.join(f'<li>{L[k].format(**N)}</li>' for k in ('step1', 'step2', 'step3', 'step4')),
    'CHECK_ROWS': ''.join(f'<tr><td>{a}</td><td class="v">{b}</td><td>{c_}</td></tr>' for a, b, c_ in rows),
    'CAVEATS': ''.join(f'<div>{L[k].format_map(NF)}</div>' for k in ('cav1', 'cav2', 'cav7', 'cav3', 'cav4', 'cav5', 'cav6')),
    'FILES': ''.join(f'<b>{a}</b><span>{b}</span>' for a, b in files),
    'FOOT': L['foot'].format(date=datetime.date.today().isoformat()),
}
page = open(os.path.join(WEB, 'template.html'), encoding='utf-8').read()
for k, v in rep.items(): page = page.replace('{{' + k + '}}', v)
page = re.sub(r'\{\{t:([a-z0-9_]+)\}\}', lambda m: L[m.group(1)], page)          # plain strings of the language table
left = re.findall(r'\{\{[A-Za-z_0-9:]+\}\}', page)
if left: print('WARNING unfilled placeholders:', sorted(set(left)))
full = (f'<!doctype html>\n<html lang="{L["lang"]}">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n' + page + '\n</body>\n</html>\n')
open(OUTF, 'w', encoding='utf-8').write(full)
if os.environ.get('LG_BODY_DIR'):          # body-only copy (no <html>/<head> wrapper) for hosts that add their own page skeleton
    open(os.path.join(os.environ['LG_BODY_DIR'], os.path.basename(OUTF)), 'w', encoding='utf-8').write(page)
print(OUTF, os.path.getsize(OUTF) // 1024, 'kB')
