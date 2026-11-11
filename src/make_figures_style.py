# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Shared SVG style of the figures."""
import numpy as np
FONT = "system-ui, -apple-system, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif"
STYLE = f"""
.lgfig {{ --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#b5b4ab; --surface:#fcfcfb; --mark:#0b0b0b;
         font-family:{FONT}; }}
@media (prefers-color-scheme: dark) {{ .lgfig {{ --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#55554f; --surface:#1a1a19; --mark:#f0efec; }} }}
.lgfig .bg {{ fill:var(--surface); }}
.lgfig text {{ fill:var(--ink2); font-size:12px; }}
.lgfig .tl {{ fill:var(--muted); font-size:11.5px; font-variant-numeric:tabular-nums; }}
.lgfig .al {{ fill:var(--ink2); font-size:12.5px; }}
.lgfig .h1 {{ fill:var(--ink); font-size:21px; font-weight:650; }}
.lgfig .h2 {{ fill:var(--ink); font-size:14.5px; font-weight:650; }}
.lgfig .sub {{ fill:var(--ink2); font-size:12.5px; }}
.lgfig .note {{ fill:var(--muted); font-size:11.5px; }}
.lgfig .frame {{ stroke:var(--axis); stroke-width:1; }}
.lgfig .tick {{ stroke:var(--axis); stroke-width:1; }}
.lgfig .grid {{ stroke:var(--grid); stroke-width:1; }}
.lgfig .iso0 {{ stroke:var(--mark); stroke-width:0.7; stroke-opacity:0.55; stroke-dasharray:3 3; }}
.lgfig .iso {{ stroke:var(--mark); stroke-width:0.5; stroke-opacity:0.28; }}
.lgfig .line {{ stroke:#2a78d6; stroke-width:2; fill:none; }}
.lgfig .line2 {{ stroke:#eb6834; stroke-width:2; fill:none; }}
.lgfig .dot {{ fill:var(--ink); }}
.lgfig .lead {{ stroke:var(--mark); stroke-width:0.8; }}
.lgfig .lab {{ fill:var(--ink); font-size:11.5px; font-weight:600; paint-order:stroke; stroke:var(--surface); stroke-width:3px; }}
"""

def svg_open(w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="lgfig" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{title}">\n'
            f'<style>{STYLE}</style>\n<rect class="bg" x="0" y="0" width="{w}" height="{h}"/>\n')



def wrap(text, limit=176):
    """Break a caption into lines of at most `limit` width units (a CJK character counts as two)."""
    width = lambda s: sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)
    words, lines, cur = [], [], ''
    for tok in text.split(' '):                      # CJK text has no spaces: split it into single characters
        if width(tok) > len(tok):
            words.extend(tok)
        else:
            words.append(' ' + tok if words else tok)
    for w in words:
        if width(cur + w) > limit and cur:
            lines.append(cur.strip()); cur = w.lstrip()
        else:
            cur += w
    if cur.strip(): lines.append(cur.strip())
    return lines


def footer(text, x, bottom, cls='note', limit=176, dy=16):
    """A caption block whose LAST line sits at y = bottom (it grows upwards when it needs more lines)."""
    lines = wrap(text, limit)
    return '\n'.join(f'<text class="{cls}" x="{x}" y="{bottom - dy * (len(lines) - 1 - k)}">{ln}</text>' for k, ln in enumerate(lines))
