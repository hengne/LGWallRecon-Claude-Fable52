# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Load the measured XY scan (scan_summary.csv) into regular grids; small shared helpers."""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# the scan: $LG_SCAN, else data/scan_summary.csv of this package
CSV = os.environ.get('LG_SCAN') or os.path.join(HERE, '..', 'data', 'scan_summary.csv')

def load(path=CSV):
    rows = list(csv.DictReader(open(path)))
    ix = np.array([int(r['idx_x']) for r in rows]); iy = np.array([int(r['idx_y']) for r in rows])
    nx, ny = ix.max() + 1, iy.max() + 1
    out = {}
    for k in ['pos_x_mm', 'pos_y_mm', 'probe_V', 'n_events', 'passed', 'direction', 'acq_time_s']:
        g = np.full((nx, ny), np.nan); g[ix, iy] = [float(r[k]) for r in rows]; out[k] = g
    out['x'] = out['pos_x_mm'][:, 0].copy(); out['y'] = out['pos_y_mm'][0, :].copy()
    out['V'] = out['probe_V']          # V[ix, iy]
    return out


def rms(a):
    """Root mean square (not the standard deviation: a bias counts)."""
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(a * a)))


def wall_sha1(c):
    """Fingerprint of a set of spline coefficients; ties a check result to the wall it was run on."""
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(np.round(np.asarray(c, float), 9)).tobytes()).hexdigest()[:16]


def spot(e):
    """Light-spot parameter for lgtrace (`sig`): the edge of the scan prefers a uniform disc
    (radius ~0.34 mm, passed as a negative number) over a Gaussian."""
    if 'spot_radius' in e.files:
        if 'spot_blur' in e.files:
            import lgtrace; lgtrace.SPOT_BLUR = float(e['spot_blur'])
        return -float(e['spot_radius'])
    return float(np.median(e['sigma']))


def aperture_fourier(e):
    """Fourier coefficients (cos, sin; m = 0, 1, ...) of the entrance aperture radius r_ap(phi) [mm]:
    measured from the edge of the scan (edge.py), or, in a closure test (LG_SYNTH), the aperture of the
    synthetic wall itself, so that the known truth is consistent with every input of the fit."""
    syn = os.environ.get('LG_SYNTH')
    if syn:
        s = np.load(syn)
        if 'fourier' in s.files:
            return s['fourier'].copy()
    F = e['fourier'].copy()
    if 'dr_disc' in e.files:
        F[0, 0] += float(e['dr_disc'])      # small correction of the mean radius from the disc-spot edge fit
    return F
