# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Load the measured XY scan (scan_summary.csv) into regular grids."""
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


def spot(e):
    """Light-spot parameter for lgtrace (`sig`): the edge of the scan prefers a uniform disc
    (radius ~0.34 mm, passed as a negative number) over a Gaussian."""
    if 'spot_radius' in e.files:
        if 'spot_blur' in e.files:
            import lgtrace; lgtrace.SPOT_BLUR = float(e['spot_blur'])
        return -float(e['spot_radius'])
    return float(np.median(e['sigma']))


def aperture_fourier(e):
    """Fourier coefficients (cos, sin; m = 0..8) of the measured entrance aperture radius r_ap(phi) [mm]."""
    F = e['fourier'].copy()
    if 'dr_disc' in e.files:
        F[0, 0] += float(e['dr_disc'])      # small correction of the mean radius from the disc-spot edge fit
    return F
