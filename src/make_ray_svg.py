# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Animated SVG (SMIL, no script): side view of real traced rays in the fitted light guide.
The rays come from the JavaScript port of the tracer (web/export_rays.js, needs node).

usage: make_ray_svg.py <final npz> <out dir> <lang: zh|en>
"""
import sys, os, json, subprocess, numpy as np
import i18n
from make_figures_style import svg_open, footer

f = np.load(sys.argv[1]); OUT = sys.argv[2]; L = i18n.load(sys.argv[3]); c = f['c']
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, '..', 'web')
wall_json = os.path.join(OUT, '..', 'wall_for_js.json')
json.dump(dict(c=[float(v) for v in c.ravel()], nphi=int(c.shape[0]), nk=int(c.shape[1])), open(wall_json, 'w'))
SRC = (-3.0, 2.0); NR = 44
rays = json.loads(subprocess.run(['node', os.path.join(WEB, 'export_rays.js'), wall_json, str(SRC[0]), str(SRC[1]), str(NR), '11'],
                                 capture_output=True, text=True, check=True).stdout)
os.remove(wall_json)
W, H = 1200, 486; X0, Y0, K = 96, 235, 10.0                 # 10 px per mm, true aspect ratio
PX = lambda z: X0 + K * z; PY = lambda y: Y0 - K * y
DEL = f['DEL']; z = f['z']; phi = f['phi']
rtop = 8.75 - 0.0475 * z + DEL[np.argmin(np.abs(phi - np.pi / 2))]            # wall in the y-z plane, +y side
rbot = 8.75 - 0.0475 * z + DEL[np.argmin(np.abs(phi - 3 * np.pi / 2))]        # -y side
nhit = sum(r['hit'] for r in rays)
s = [svg_open(W, H, L['f4_aria'])]
s.append('<style>.lgfig .wall{fill:var(--axis);opacity:.55}.lgfig .cav{fill:var(--surface)}'
         '.lgfig .hit{stroke:#e39a00;stroke-width:1.5;fill:none;stroke-linejoin:round}'
         '.lgfig .lost{stroke:#7d93ad;stroke-width:1;fill:none;stroke-linejoin:round;opacity:.75}.lgfig .pmt{fill:#e39a00}</style>')
s.append(f'<text class="h1" x="40" y="44">{L["f4_title"]}</text>')
s.append(f'<text class="sub" x="40" y="68">{L["f4_sub"].format(nhit=nhit, n=NR)}</text>')
top = ' '.join(f'{PX(a):.1f},{PY(b):.1f}' for a, b in zip(z, rtop)); bot = ' '.join(f'{PX(a):.1f},{PY(-b):.1f}' for a, b in zip(z[::-1], rbot[::-1]))
s.append(f'<polygon class="wall" points="{PX(0):.1f},{PY(13):.1f} {PX(100):.1f},{PY(13):.1f} {PX(100):.1f},{PY(-13):.1f} {PX(0):.1f},{PY(-13):.1f}"/>')
s.append(f'<polygon class="cav" points="{top} {bot}"/>')
s.append(f'<rect class="pmt" x="{PX(100):.1f}" y="{PY(4.6):.1f}" width="14" height="{K * 9.2:.1f}" rx="2" opacity="0.35"><animate attributeName="opacity" values="0.3;0.95;0.3" dur="1.3s" repeatCount="indefinite"/></rect>')
s.append(f'<text class="al" x="{PX(100) + 22:.1f}" y="{Y0 + 4}">PMT</text><text class="al" x="{PX(0) - 12:.1f}" y="{Y0 - 100}" text-anchor="end">{L["f4_in"]}</text>')
s.append(f'<circle cx="{PX(0):.1f}" cy="{PY(SRC[1]):.1f}" r="4" fill="var(--ink)"/>')
TC, SPEED = 9.0, 420.0                                            # cycle [s], drawing speed [px / s]
rng = np.random.default_rng(4)
for r in rays:
    p = np.array(r['p']).reshape(-1, 3); xs_, ys_ = PX(p[:, 2]), PY(p[:, 1])
    Lp = float(np.hypot(np.diff(xs_), np.diff(ys_)).sum()); d = min(Lp / SPEED, TC * 0.7); b = float(rng.uniform(0, TC * 0.25))
    path = 'M' + 'L'.join(f'{a:.1f},{b_:.1f}' for a, b_ in zip(xs_, ys_))
    kt = f'0;{b / TC:.4f};{(b + d) / TC:.4f};1'
    # static attributes show the complete ray (renderers without SMIL); the animation draws it progressively
    s.append(f'<path class="{"hit" if r["hit"] else "lost"}" d="{path}" stroke-dasharray="{Lp:.1f}" stroke-dashoffset="0">'
             f'<animate attributeName="stroke-dashoffset" values="{Lp:.1f};{Lp:.1f};0;0" keyTimes="{kt}" dur="{TC}s" repeatCount="indefinite"/>'
             f'<animate attributeName="opacity" values="1;1;0" keyTimes="0;0.93;1" dur="{TC}s" repeatCount="indefinite"/></path>')
s.append(f'<line class="hit" x1="60" y1="428" x2="92" y2="428"/><text class="al" x="100" y="432">{L["f4_hit"]}</text>'
         f'<line class="lost" x1="330" y1="428" x2="362" y2="428"/><text class="al" x="370" y="432">{L["f4_lost"]}</text>')
for t in range(0, 101, 20):
    s.append(f'<line class="tick" x1="{PX(t):.1f}" y1="{PY(-13):.1f}" x2="{PX(t):.1f}" y2="{PY(-13) + 5:.1f}"/><text class="tl" x="{PX(t):.1f}" y="{PY(-13) + 19:.1f}" text-anchor="middle">{t}</text>')
s.append(f'<text class="al" x="{PX(50):.1f}" y="{PY(-13) + 38:.1f}" text-anchor="middle">{L["ax_z"]}</text>')
s.append(footer(L['f4_foot'].format(x=SRC[0], y=SRC[1]), 40, H - 12))
s.append('</svg>')
open(os.path.join(OUT, 'ray_animation.svg'), 'w', encoding='utf-8').write('\n'.join(s))
print('ray_animation.svg', os.path.getsize(os.path.join(OUT, 'ray_animation.svg')) // 1024, 'kB;', nhit, 'of', NR, 'detected')
