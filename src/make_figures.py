# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Vector SVG figures of the reconstruction: the wall map (inner_surface.svg) and the comparison of the
measured and simulated response maps (response_maps.svg).

usage: make_figures.py <final npz> <out dir> <lang: zh|en>
"""
import sys, os, numpy as np
import svgkit as sk
import i18n
from make_figures_style import svg_open, footer

f = np.load(sys.argv[1]); OUT = sys.argv[2]; L = i18n.load(sys.argv[3]); os.makedirs(OUT, exist_ok=True)
phi, z, DEL = np.degrees(f['phi']), f['z'], f['DEL']            # DEL[phi, z]
prof = DEL[:-1].mean(axis=0)                                      # azimuthal mean (m = 0) profile
LOC = DEL - prof                                                  # local unevenness (non-axisymmetric part)
x, y, S, M, R = f['x'], f['y'], f['S'], f['M'], f['R']
plat = S[R < 8.0].mean()


def nice_max(v, steps=(0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)):
    return next((s for s in steps if s >= v), v)


# ====================================================================== figure 1: the wall
W, H = 1200, 972
s = [svg_open(W, H, L['f1_title'])]
s.append(f'<text class="h1" x="40" y="44">{L["f1_title"]}</text>')
s.append(f'<text class="sub" x="40" y="68">{L["f1_sub"]}</text>')

# --- panel A: local unevenness
fa = sk.Frame(78, 132, 800, 380, (0, 360), (0, 100))
vm = nice_max(np.abs(LOC).max())
nb = 20 if vm / 20 >= 0.004 else 10
lev = np.linspace(-vm, vm, nb + 1)
cneg = lambda t: sk.DIV(1 - t)       # red = towards the axis (delta < 0, a bump into the cavity)
s.append(f'<text class="h2" x="78" y="104">{L["f1_A"]}</text>')
s.append(f'<text class="note" x="78" y="121">{L["f1_A_note"].format(step=lev[1] - lev[0])}</text>')
s.append(sk.bands(fa, phi, z, LOC.T, lev, cneg, -vm, vm))
s.append(sk.isolines(fa, phi, z, LOC.T, 0.0, 'iso0'))
s.append(sk.axes(fa, range(0, 361, 45), range(0, 101, 20), L['ax_phi'], L['ax_z']))
s.append(sk.colorbar(896, 132, 13, 380, cneg, -vm, vm, np.linspace(-vm, vm, 5), L['cb_local'], levels=lev))
s.append(f'<text class="note" x="896" y="{132 + 380 + 16}">{L["f1_bump_arrow"]}</text>')


def extreme(arr, sign):
    i, j = np.unravel_index(np.argmax(sign * arr[:-1]), arr[:-1].shape); return phi[i], z[j], arr[i, j]


# direct labels for the strongest bump and the strongest dent only
for sign, key in ((-1, 'f1_bump'), (1, 'f1_dent')):
    p0, z0, v0 = extreme(LOC, sign)
    X, Y = float(fa.X(p0)), float(fa.Y(z0)); left = p0 < 200; tx = X + (14 if left else -14); ty = Y - 14 if z0 < 85 else Y + 22
    s.append(f'<circle class="dot" cx="{X:.1f}" cy="{Y:.1f}" r="2.6"/><line class="lead" x1="{X:.1f}" y1="{Y:.1f}" x2="{tx:.1f}" y2="{ty + 3:.1f}"/>')
    s.append(f'<text class="lab" x="{tx:.1f}" y="{ty:.1f}" text-anchor="{"start" if left else "end"}">{L[key].format(v=v0, p=p0, z=z0)}</text>')

# --- panel B: azimuthal mean profile, sharing the z axis with A
pm = nice_max(np.abs(prof).max())
fb = sk.Frame(1000, 132, 160, 380, (-pm, pm * 0.15), (0, 100))
s.append(f'<text class="h2" x="1000" y="104">{L["f1_B"]}</text>')
s.append(f'<text class="note" x="1000" y="121">{L["f1_B_note"]}</text>')
xt = [t for t in np.round(np.arange(-pm, 0.001, pm / 2), 3)]
s.append(sk.axes(fb, xt, range(0, 101, 20), L['cb_mean'], '', xfmt='{:+.2f}', grid=True))
s.append(f'<line class="tick" x1="{float(fb.X(0)):.1f}" y1="{fb.y0}" x2="{float(fb.X(0)):.1f}" y2="{fb.y0 + fb.h}"/>')
s.append('<path class="line" d="M' + 'L'.join(f'{float(fb.X(a)):.2f},{float(fb.Y(b)):.2f}' for a, b in zip(prof, z)) + '"/>')
s.append(f'<text class="note" x="{float(fb.X(0)) + 4:.1f}" y="{fb.y0 + 14}">CAD</text>')

# --- panel C: total deviation
fc = sk.Frame(78, 612, 800, 250, (0, 360), (0, 100))
tm = nice_max(np.abs(DEL).max()); levt = np.linspace(-tm, tm, 21)
s.append(f'<text class="h2" x="78" y="584">{L["f1_C"]}</text>')
s.append(f'<text class="note" x="78" y="601">{L["f1_C_note"].format(step=levt[1] - levt[0], d=-prof[0])}</text>')
s.append(sk.bands(fc, phi, z, DEL.T, levt, cneg, -tm, tm))
s.append(sk.isolines(fc, phi, z, DEL.T, 0.0, 'iso0'))
s.append(sk.axes(fc, range(0, 361, 45), range(0, 101, 20), L['ax_phi'], L['ax_z']))
s.append(sk.colorbar(896, 612, 13, 250, cneg, -tm, tm, np.linspace(-tm, tm, 5), L['cb_total'], levels=levt))

# --- key: how phi and z are defined
kx, ky = 1000, 612
s.append(f'<text class="h2" x="{kx}" y="584">{L["f1_key"]}</text>')
cx0, cy0, rr = kx + 62, ky + 78, 46
s.append(f'<circle cx="{cx0}" cy="{cy0}" r="{rr}" fill="none" class="frame"/>')
s.append(f'<line class="tick" x1="{cx0 - rr - 8}" y1="{cy0}" x2="{cx0 + rr + 14}" y2="{cy0}"/><line class="tick" x1="{cx0}" y1="{cy0 + rr + 8}" x2="{cx0}" y2="{cy0 - rr - 14}"/>')
s.append(f'<text class="tl" x="{cx0 + rr + 17}" y="{cy0 + 4}">x</text><text class="tl" x="{cx0 - 3}" y="{cy0 - rr - 18}">y</text>')
a0 = np.radians(50); ax_, ay_ = cx0 + rr * np.cos(a0), cy0 - rr * np.sin(a0)
s.append(f'<line class="lead" x1="{cx0}" y1="{cy0}" x2="{ax_:.1f}" y2="{ay_:.1f}"/><circle class="dot" cx="{ax_:.1f}" cy="{ay_:.1f}" r="3"/>')
arc = 'M' + 'L'.join(f'{cx0 + 20 * np.cos(t):.1f},{cy0 - 20 * np.sin(t):.1f}' for t in np.linspace(0, a0, 12))
s.append(f'<path d="{arc}" fill="none" class="lead"/><text class="tl" x="{cx0 + 24}" y="{cy0 - 8}">φ</text>')
for k, t in enumerate(L['f1_key_lines']):
    s.append(f'<text class="note" x="{kx}" y="{ky + 158 + 17 * k}">{t}</text>')
s.append(footer(L['f1_foot'], 40, H - 16))
s.append('</svg>')
open(os.path.join(OUT, 'inner_surface.svg'), 'w', encoding='utf-8').write('\n'.join(s))

# ====================================================================== figure 2: data vs simulation
W, H = 1200, 820
s = [svg_open(W, H, L['f2_title'])]
s.append(f'<text class="h1" x="40" y="44">{L["f2_title"]}</text>')
s.append(f'<text class="sub" x="40" y="68">{L["f2_sub"]}</text>')
inn = R < 8.9
lo, hi = np.percentile(S[R < 8.0], [0.5, 99.5]); lo = np.floor(lo * 1e3 * 2) / 2e3; hi = np.ceil(hi * 1e3 * 2) / 2e3
xl, yl = (x[0] - 0.1, x[-1] + 0.1), (y[0] - 0.1, y[-1] + 0.1)
q = lambda t: sk.SEQ(np.round(np.clip(t, 0, 1) * 24) / 24)          # 25 discrete steps of one hue
for k, (img, name) in enumerate(((S, L['f2_data']), (M, L['f2_model']))):
    fr = sk.Frame(70 + k * 370, 120, 300, 300, xl, yl)
    s.append(f'<text class="h2" x="{fr.x0}" y="106">{name}</text>')
    s.append(sk.pixels(fr, x, y, img, q, lo, hi, mask=inn))
    s.append(sk.axes(fr, range(-6, 13, 3), range(-14, 6, 3), L['ax_x'], L['ax_y'] if k == 0 else ''))
s.append(sk.colorbar(748, 120, 12, 300, q, lo, hi, np.linspace(lo, hi, 5), L['f2_signal'], fmt='{:.4f}', levels=np.linspace(lo, hi, 26)))
res = (S - M) / plat * 100
fr = sk.Frame(850, 120, 300, 300, xl, yl); rm = 4.0
qd = lambda t: sk.DIV(np.round(np.clip(t, 0, 1) * 20) / 20)
s.append(f'<text class="h2" x="{fr.x0}" y="106">{L["f2_resid"]}</text>')
s.append(sk.pixels(fr, x, y, res, qd, -rm, rm, mask=inn))
s.append(sk.axes(fr, range(-6, 13, 3), range(-14, 6, 3), L['ax_x'], ''))
s.append(sk.colorbar(1156, 120, 12, 300, qd, -rm, rm, [-4, -2, 0, 2, 4], '[%]', fmt='{:+.0f}', levels=np.linspace(-rm, rm, 21)))
# radial profile
rb = np.arange(0.1, 9.0, 0.2)
ring_mean = lambda img: np.array([img[np.abs(R - r) < 0.1].mean() for r in rb])
pd, pmod, pid = ring_mean(S), ring_mean(M), ring_mean(f['Mideal'])
fr = sk.Frame(70, 520, 480, 200, (0, 9), (0, 0.03))
s.append(f'<text class="h2" x="{fr.x0}" y="500">{L["f2_radial"]}</text>')
s.append(sk.axes(fr, range(0, 10), [0, 0.01, 0.02, 0.03], L['f2_radial_x'], L['f2_signal'], yfmt='{:.2f}', grid=True))
P = lambda v: 'M' + 'L'.join(f'{float(fr.X(a)):.2f},{float(fr.Y(b)):.2f}' for a, b in zip(rb, v))
s.append(f'<path d="{P(pid)}" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="5 4"/>')
s.append(f'<path class="line2" d="{P(pmod)}"/>')
s.append(''.join(f'<circle class="dot" cx="{float(fr.X(a)):.2f}" cy="{float(fr.Y(b)):.2f}" r="2.2"/>' for a, b in zip(rb, pd)))
ly = float(fr.Y(0.0282))
s.append(f'<text class="lab" x="{float(fr.X(1.2)):.1f}" y="{ly:.1f}">{L["f2_leg_data"]}</text><text class="lab" x="{float(fr.X(3.0)):.1f}" y="{ly:.1f}" style="fill:#c2501f">{L["f2_leg_model"]}</text><text class="lab" x="{float(fr.X(5.0)):.1f}" y="{ly:.1f}" style="fill:var(--muted)">{L["f2_leg_ideal"]}</text>')
# azimuthal cut on a ring
PH = np.degrees(np.arctan2((y - f['cy'])[None, :], (x - f['cx'])[:, None])) % 360
fr = sk.Frame(670, 520, 480, 200, (0, 360), (lo, hi))
r0 = 4.0; mm = np.abs(R - r0) < 0.3
ab = np.arange(5, 360, 10)
sect = lambda img: np.array([img[mm & (np.abs((PH - a + 180) % 360 - 180) < 5)].mean() for a in ab])
cd, cm = sect(S), sect(M)
s.append(f'<text class="h2" x="{fr.x0}" y="500">{L["f2_ring"].format(r=r0)}</text>')
s.append(sk.axes(fr, range(0, 361, 90), np.linspace(lo, hi, 4), L['ax_phi'], '', yfmt='{:.4f}', grid=True))
s.append('<path class="line2" d="M' + 'L'.join(f'{float(fr.X(a)):.2f},{float(fr.Y(b)):.2f}' for a, b in zip(ab, cm)) + '"/>')
s.append(''.join(f'<circle class="dot" cx="{float(fr.X(a)):.2f}" cy="{float(fr.Y(b)):.2f}" r="2.4"/>' for a, b in zip(ab, cd)))
s.append(footer(L['f2_foot'].format(rms=res[R < 8.0].std()), 40, H - 16))
s.append('</svg>')
open(os.path.join(OUT, 'response_maps.svg'), 'w', encoding='utf-8').write('\n'.join(s))
for n in ('inner_surface.svg', 'response_maps.svg'):
    print(n, os.path.getsize(os.path.join(OUT, n)) // 1024, 'kB')
