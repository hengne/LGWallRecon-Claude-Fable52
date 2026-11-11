// LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
// Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
// Trace N Lambertian rays from one source point with the fitted wall and print their polylines as JSON.
// usage: node export_rays.js <wall.json> <x> <y> <N> [seed]
const LG = require('./lgtrace.js'); const fs = require('fs');
const w = JSON.parse(fs.readFileSync(process.argv[2]));
LG.setWall(w.c, w.nphi, w.nk);
const x = +process.argv[3], y = +process.argv[4], N = +process.argv[5]; let s = +(process.argv[6] || 7);
const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
const out = [];
while (out.length < N) { const r = LG.lambert(x, y, rnd(), rnd()); if (r) out.push({ hit: r.detected, n: r.bounces, p: r.pts.map(v => +v.toFixed(3)) }); }
console.log(JSON.stringify(out));
