# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Closure test: the true wall against the wall reconstructed from its synthetic scan.
Writes closure.svg and (once) closure_numbers.json.

usage: make_closure_fig.py <synth npz> <final npz of the closure fit> <out dir> <lang: zh|en>
"""
import sys, os, json, numpy as np
import lgtrace as lg, svgkit as sk, i18n
from load_scan import rms
from make_figures_style import svg_open, footer

syn = np.load(sys.argv[1]); f = np.load(sys.argv[2]); OUT = sys.argv[3]; L = i18n.load(sys.argv[4])
phi, z, REC = f['phi'], f['z'], f['DEL']
cp = lg.pad_coeffs(syn['c_true'])
TRU = np.array([[lg.delta_eval(cp, min(p, 2 * np.pi - 1e-9), zz)[0] for zz in z] for p in phi])
pt, pr = TRU[:-1].mean(axis=0), REC[:-1].mean(axis=0)            # azimuthal mean profiles
LT, LR = TRU - pt, REC - pr                                       # local (non-axisymmetric) parts
cc = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
bands = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
sel = lambda b: (z >= b[0]) & (z <= b[1])
inn = f['R'] < 8.0
# all 'rms' are root mean squares (a bias counts); the local parts have zero azimuthal mean by construction
num = dict(corr_all=cc(LT, LR), rms_true=rms(LT[:-1]), rms_rec=rms(LR[:-1]), rms_err=rms((LR - LT)[:-1]),
           bands=[dict(z=b, corr=cc(LT[:, sel(b)], LR[:, sel(b)]), rms_true=rms(LT[:-1][:, sel(b)]),
                       rms_err=rms((LR - LT)[:-1][:, sel(b)])) for b in bands],
           prof_rms_err=rms(pr - pt), prof_mean_err=float((pr - pt).mean()), prof_std_err=float((pr - pt).std()),
           map_rms=rms((f['S'] - f['M'])[inn]) / float(f['S'][inn].mean()) * 100,
           chi2=float(np.sum((f['S'] - f['M'])[inn] ** 2) / (2.0e-4 ** 2 + float(f['mc']) ** 2) / inn.sum()),
           seed=int(syn['seed']) if 'seed' in syn.files else -1,
           base_sha1=str(syn['base_sha1']) if 'base_sha1' in syn.files else '')
json.dump(num, open(os.path.join(OUT, '..', 'closure_numbers.json'), 'w'), indent=1)

W, H = 1200, 584; ph = np.degrees(phi)
s = [svg_open(W, H, L['f3_title'])]
s.append(f'<text class="h1" x="40" y="44">{L["f3_title"]}</text>')
s.append(f'<text class="sub" x="40" y="68">{L["f3_sub"]}</text>')
vm = max(0.1, np.ceil(np.abs(LT).max() * 20) / 20); lev = np.linspace(-vm, vm, 21); cneg = lambda t: sk.DIV(1 - t)
for k, (img, name) in enumerate(((LT, L['f3_true']), (LR, L['f3_rec']))):
    fr = sk.Frame(70 + k * 440, 120, 380, 340, (0, 360), (0, 100))
    s.append(f'<text class="h2" x="{fr.x0}" y="106">{name}</text>')
    s.append(sk.bands(fr, ph, z, img.T, lev, cneg, -vm, vm)); s.append(sk.isolines(fr, ph, z, img.T, 0.0, 'iso0'))
    s.append(sk.axes(fr, range(0, 361, 90), range(0, 101, 20), L['ax_phi'], L['ax_z'] if k == 0 else ''))
s.append(sk.colorbar(900, 120, 12, 340, cneg, -vm, vm, np.linspace(-vm, vm, 5), L['cb_local'], levels=lev))
pm = max(0.6, np.ceil(max(np.abs(pt).max(), np.abs(pr).max()) * 10) / 10)
fb = sk.Frame(1010, 120, 150, 340, (-pm, 0.05), (0, 100))
s.append(f'<text class="h2" x="1010" y="106">{L["f3_mean"]}</text>')
s.append(sk.axes(fb, [-pm, -pm / 2, 0], range(0, 101, 20), L['cb_mean'], '', xfmt='{:+.1f}', grid=True))
P = lambda v: 'M' + 'L'.join(f'{float(fb.X(a)):.2f},{float(fb.Y(b)):.2f}' for a, b in zip(v, z))
s.append(f'<path d="{P(pt)}" fill="none" stroke="var(--ink)" stroke-width="2" stroke-dasharray="5 4"/><path class="line2" d="{P(pr)}"/>')
s.append(f'<text class="lab" x="1012" y="{120 + 340 + 58}">{L["f3_leg_true"]}</text><text class="lab" x="1090" y="{120 + 340 + 58}" style="fill:#c2501f">{L["f3_leg_rec"]}</text>')
s.append(footer(L['f3_foot'].format(map_rms=num['map_rms'], corr_all=num['corr_all'], rms_true=num['rms_true'] * 1000, rms_rec=num['rms_rec'] * 1000), 40, H - 14))
s.append('</svg>')
open(os.path.join(OUT, 'closure.svg'), 'w', encoding='utf-8').write('\n'.join(s))
print('closure.svg written; wall correlation', round(num['corr_all'], 3), ' map residual', round(num['map_rms'], 3), '%')
