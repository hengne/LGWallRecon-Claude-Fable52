// LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
// Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
// compare the JS port with the Python tracer on the same wall (statistically)
const LG = require('./lgtrace.js'); const fs = require('fs');
const w = JSON.parse(fs.readFileSync(process.argv[2]));
LG.setWall(w.c.flat(), w.nphi, w.nk);
let s = 12345; const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
const N = 200000;
for (const [x, y, Tpy] of w.points) {
  let det = 0; for (let k = 0; k < N; k++) { const r = LG.lambert(x, y, rnd(), rnd()); if (r && r.detected) det++; }
  console.log(`(${x}, ${y})  JS ${(det / N).toFixed(4)}  python ${Tpy.toFixed(4)}  diff ${((det / N - Tpy) / Math.sqrt(Tpy * (1 - Tpy) / N)).toFixed(1)} sigma(JS stat)`);
}
