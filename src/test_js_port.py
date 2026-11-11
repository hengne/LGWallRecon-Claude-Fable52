# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Check the JavaScript port of the tracer (web/lgtrace.js, used by the report page) against the Python
tracer on a rough random wall.  Needs node.  The two use different random numbers, so the comparison is
statistical: the last column is the difference in units of the JS statistical error.

usage: test_js_port.py
"""
import os, json, subprocess, tempfile, numpy as np
import lgtrace as lg

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'web')
rng = np.random.default_rng(5); c = 0.2 * rng.standard_normal((24, 13))
pts = [(0, 0), (3, 1), (-5, 4), (2, -7.5), (-6.5, -3)]
T = lg.transmission([p[0] for p in pts], [p[1] for p in pts], coeffs=c, nray=2000000, smax=1.0)
with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
    json.dump({'c': c.tolist(), 'nphi': 24, 'nk': 13, 'points': [[p[0], p[1], float(t)] for p, t in zip(pts, T)]}, f)
print(subprocess.run(['node', os.path.join(WEB, 'test_js.js'), f.name], capture_output=True, text=True).stdout)
os.remove(f.name)
