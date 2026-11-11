# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Small helpers to write hand-built, fully vector SVG figures (filled contour bands, contour lines,
pixel maps, axes).  No raster content, so the figures stay sharp at any zoom."""
import numpy as np
from contourpy import contour_generator


# ---------- colour: interpolation in OKLab between hex stops ----------
def _hex2rgb(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0

def _lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def _gam(c): return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)

def _oklab(rgb):
    r, g, b = _lin(rgb)
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])

def _unoklab(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = np.array([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])
    return np.clip(_gam(rgb), 0, 1)

def ramp(stops):
    """stops: list of (position 0..1, hex).  Returns f(t) -> hex, interpolating in OKLab."""
    pos = np.array([p for p, _ in stops]); labs = np.array([_oklab(_hex2rgb(h)) for _, h in stops])
    def f(t):
        t = float(np.clip(t, 0, 1))
        lab = np.array([np.interp(t, pos, labs[:, k]) for k in range(3)])
        r, g, b = (np.round(_unoklab(lab) * 255)).astype(int)
        return f'#{r:02x}{g:02x}{b:02x}'
    return f

# sequential: one hue (blue), light -> dark;  diverging: blue <-> red through a neutral grey
SEQ = ramp([(0.0, '#eef4fc'), (0.15, '#cde2fb'), (0.4, '#86b6ef'), (0.6, '#3987e5'), (0.8, '#1c5cab'), (1.0, '#0d366b')])
DIV = ramp([(0.0, '#0d366b'), (0.2, '#256abf'), (0.36, '#86b6ef'), (0.5, '#f0efec'), (0.64, '#f0a39c'), (0.8, '#d03b3b'), (1.0, '#7a1616')])


# ---------- geometry ----------
class Frame:
    """Linear map from data coordinates to SVG pixels for one panel."""
    def __init__(self, x0, y0, w, h, xlim, ylim):
        self.x0, self.y0, self.w, self.h, self.xlim, self.ylim = x0, y0, w, h, xlim, ylim
    def X(self, x): return self.x0 + (np.asarray(x, float) - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w
    def Y(self, y): return self.y0 + self.h - (np.asarray(y, float) - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

def _path(fr, pts, close):
    xs, ys = fr.X(pts[:, 0]), fr.Y(pts[:, 1])
    d = 'M' + 'L'.join(f'{a:.2f},{b:.2f}' for a, b in zip(xs, ys))
    return d + ('Z' if close else '')

def bands(fr, x, y, Z, levels, cfun, vlo, vhi, cls='band'):
    """Filled contour bands of Z[len(y), len(x)] as SVG paths; colour = cfun((mid - vlo) / (vhi - vlo))."""
    cg = contour_generator(x, y, Z, fill_type='OuterOffset')
    out = []
    for lo, hi in zip(levels[:-1], levels[1:]):
        pts_list, off_list = cg.filled(lo, hi)
        d = ''
        for pts, off in zip(pts_list, off_list):
            for a, b in zip(off[:-1], off[1:]):
                d += _path(fr, pts[a:b], True)
        if d:
            col = cfun(((lo + hi) / 2 - vlo) / (vhi - vlo))
            out.append(f'<path class="{cls}" d="{d}" fill="{col}" stroke="{col}" stroke-width="0.4" fill-rule="evenodd"><title>{lo:+.3f} … {hi:+.3f}</title></path>')
    return '\n'.join(out)

def isolines(fr, x, y, Z, level, cls='iso'):
    cg = contour_generator(x, y, Z)
    d = ''.join(_path(fr, pts, False) for pts in cg.lines(level) if len(pts) > 1)
    return f'<path class="{cls}" d="{d}" fill="none"/>' if d else ''

def pixels(fr, x, y, Z, cfun, vlo, vhi, mask=None, fmt='{:.4f}', unit=''):
    """One <rect> per scan pixel (Z[len(x), len(y)]); pixels are merged in runs of equal colour along x."""
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    pw = dx / (fr.xlim[1] - fr.xlim[0]) * fr.w; ph = dy / (fr.ylim[1] - fr.ylim[0]) * fr.h
    out = []
    for j in range(len(y)):
        i = 0
        while i < len(x):
            if mask is not None and not mask[i, j]:
                i += 1; continue
            col = cfun((Z[i, j] - vlo) / (vhi - vlo)); k = i + 1
            while k < len(x) and (mask is None or mask[k, j]) and cfun((Z[k, j] - vlo) / (vhi - vlo)) == col:
                k += 1
            out.append(f'<rect x="{fr.X(x[i] - dx / 2):.2f}" y="{fr.Y(y[j] + dy / 2):.2f}" width="{pw * (k - i) + 0.15:.2f}" height="{ph + 0.15:.2f}" fill="{col}"/>')
            i = k
    return '\n'.join(out)

def axes(fr, xticks, yticks, xlabel='', ylabel='', xfmt='{:g}', yfmt='{:g}', grid=False):
    s = [f'<rect class="frame" x="{fr.x0:.2f}" y="{fr.y0:.2f}" width="{fr.w:.2f}" height="{fr.h:.2f}" fill="none"/>']
    for t in xticks:
        X = float(fr.X(t))
        if grid: s.append(f'<line class="grid" x1="{X:.2f}" y1="{fr.y0:.2f}" x2="{X:.2f}" y2="{fr.y0 + fr.h:.2f}"/>')
        s.append(f'<line class="tick" x1="{X:.2f}" y1="{fr.y0 + fr.h:.2f}" x2="{X:.2f}" y2="{fr.y0 + fr.h + 5:.2f}"/>')
        s.append(f'<text class="tl" x="{X:.2f}" y="{fr.y0 + fr.h + 19:.2f}" text-anchor="middle">{xfmt.format(t)}</text>')
    for t in yticks:
        Y = float(fr.Y(t))
        if grid: s.append(f'<line class="grid" x1="{fr.x0:.2f}" y1="{Y:.2f}" x2="{fr.x0 + fr.w:.2f}" y2="{Y:.2f}"/>')
        s.append(f'<line class="tick" x1="{fr.x0 - 5:.2f}" y1="{Y:.2f}" x2="{fr.x0:.2f}" y2="{Y:.2f}"/>')
        s.append(f'<text class="tl" x="{fr.x0 - 9:.2f}" y="{Y + 4:.2f}" text-anchor="end">{yfmt.format(t)}</text>')
    if xlabel: s.append(f'<text class="al" x="{fr.x0 + fr.w / 2:.2f}" y="{fr.y0 + fr.h + 40:.2f}" text-anchor="middle">{xlabel}</text>')
    if ylabel: s.append(f'<text class="al" transform="translate({fr.x0 - 44:.2f},{fr.y0 + fr.h / 2:.2f}) rotate(-90)" text-anchor="middle">{ylabel}</text>')
    return '\n'.join(s)

def colorbar(x0, y0, w, h, cfun, vlo, vhi, ticks, label='', fmt='{:+.2f}', n=60, levels=None):
    """Vertical colour key.  If `levels` is given the key shows the same discrete bands as the map."""
    s = []
    edges = np.linspace(vlo, vhi, n + 1) if levels is None else np.asarray(levels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        ya = y0 + h - (hi - vlo) / (vhi - vlo) * h; hh = (hi - lo) / (vhi - vlo) * h
        s.append(f'<rect x="{x0:.2f}" y="{ya:.2f}" width="{w:.2f}" height="{hh + 0.3:.2f}" fill="{cfun(((lo + hi) / 2 - vlo) / (vhi - vlo))}"/>')
    s.append(f'<rect class="frame" x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" fill="none"/>')
    for t in ticks:
        Y = y0 + h - (t - vlo) / (vhi - vlo) * h
        s.append(f'<line class="tick" x1="{x0 + w:.2f}" y1="{Y:.2f}" x2="{x0 + w + 4:.2f}" y2="{Y:.2f}"/>')
        s.append(f'<text class="tl" x="{x0 + w + 8:.2f}" y="{Y + 4:.2f}">{fmt.format(t)}</text>')
    if label: s.append(f'<text class="al" x="{x0:.2f}" y="{y0 - 10:.2f}">{label}</text>')
    return '\n'.join(s)
