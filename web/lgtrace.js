// LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
// Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
// JavaScript port of src/lgtrace.py (same geometry, same B-spline wall, same marching + regula-falsi
// intersection).  Used by the 3-D demonstration page so that every ray drawn is a real traced ray.
const LG = (() => {
  const R1 = 8.75, R2 = 4.0, L = 100.0, SLOPE = (R2 - R1) / L, TWO_PI = 2 * Math.PI;
  let cp = null, NPHI = 0, NZ = 0, dmax = 0;       // padded coefficients cp[(i)*(NZ+3)+j], i = 0..NPHI+2
  function setWall(c, nphi, nk) {                 // c: flat array [nphi][nk]
    NPHI = nphi; NZ = nk - 3; cp = new Float64Array((nphi + 3) * nk); dmax = 0;
    for (let i = 0; i < nphi + 3; i++) for (let j = 0; j < nk; j++) {
      const v = c[((i - 1 + nphi) % nphi) * nk + j]; cp[i * nk + j] = v; if (Math.abs(v) > dmax) dmax = Math.abs(v);
    }
    dmax = dmax * 1.0001 + 1e-6;
  }
  const out3 = new Float64Array(3);
  function deltaEval(phi, z) {
    const nk = NZ + 3, hp = TWO_PI / NPHI, hz = L / NZ;
    let a = phi / hp, i = Math.floor(a); const u = a - i; if (i >= NPHI) i -= NPHI; if (i < 0) i += NPHI;
    let zz = z / hz, j = Math.floor(zz); if (j < 0) j = 0; else if (j > NZ - 1) j = NZ - 1; const v = zz - j;
    const um = 1 - u, vm = 1 - v;
    const bu = [um * um * um / 6, (3 * u * u * u - 6 * u * u + 4) / 6, (-3 * u * u * u + 3 * u * u + 3 * u + 1) / 6, u * u * u / 6];
    const du = [-0.5 * um * um, 0.5 * (3 * u * u - 4 * u), 0.5 * (-3 * u * u + 2 * u + 1), 0.5 * u * u];
    const bv = [vm * vm * vm / 6, (3 * v * v * v - 6 * v * v + 4) / 6, (-3 * v * v * v + 3 * v * v + 3 * v + 1) / 6, v * v * v / 6];
    const dv = [-0.5 * vm * vm, 0.5 * (3 * v * v - 4 * v), 0.5 * (-3 * v * v + 2 * v + 1), 0.5 * v * v];
    let val = 0, dph = 0, dz = 0;
    for (let k = 0; k < 4; k++) {
      let r = 0, s = 0; const o = (i + k) * nk + j;
      for (let m = 0; m < 4; m++) { r += cp[o + m] * bv[m]; s += cp[o + m] * dv[m]; }
      val += r * bu[k]; dph += r * du[k]; dz += s * bu[k];
    }
    out3[0] = val; out3[1] = dph / hp; out3[2] = dz / hz; return out3;
  }
  function wallRadius(phi, z) { return R1 + SLOPE * z + deltaEval(((phi % TWO_PI) + TWO_PI) % TWO_PI, z)[0]; }
  function f(x, y, z) {
    let ph = Math.atan2(y, x); if (ph < 0) ph += TWO_PI;
    return Math.hypot(x, y) - (R1 + SLOPE * z + deltaEval(ph, z)[0]);
  }
  function coneExit(px, py, pz, dx, dy, dz, off) {
    const a0 = R1 + off + SLOPE * pz, A = dx * dx + dy * dy - SLOPE * SLOPE * dz * dz,
      B = px * dx + py * dy - SLOPE * a0 * dz, C = px * px + py * py - a0 * a0;
    let best = 1e30;
    if (Math.abs(A) < 1e-14) { if (Math.abs(B) > 1e-300) { const t = -C / (2 * B); if (t > 0 && a0 + SLOPE * dz * t > 0) best = t; } return best; }
    const disc = B * B - A * C; if (disc < 0) return best;
    const sq = Math.sqrt(disc), q = -(B + (B >= 0 ? sq : -sq)), t1 = q / A, t2 = q !== 0 ? C / q : 1e30;
    if (t1 > 1e-12 && a0 + SLOPE * dz * t1 > 0 && t1 < best) best = t1;
    if (t2 > 1e-12 && a0 + SLOPE * dz * t2 > 0 && t2 < best) best = t2;
    return best;
  }
  // Trace one ray; returns {pts: [x,y,z,...] polyline, detected: bool, bounces: n}
  function trace(px, py, pz, dx, dy, dz, maxb = 400) {
    const pts = [px, py, pz]; let nb = 0;
    for (;;) {
      const tend = dz > 0 ? (L - pz) / dz : dz < 0 ? -pz / dz : 1e30;
      const r = Math.hypot(px, py);
      let tlo = 0; if (r < R1 + SLOPE * pz - dmax) tlo = coneExit(px, py, pz, dx, dy, dz, -dmax);
      if (tlo >= tend) break;
      let thi = coneExit(px, py, pz, dx, dy, dz, dmax); if (thi > tend) thi = tend;
      const n = Math.floor((thi - tlo) / 3.0) + 1, h = (thi - tlo) / n;
      let ta = tlo, fa = f(px + ta * dx, py + ta * dy, pz + ta * dz); if (fa > 0) fa = -1e-12;
      let tb = ta, fb = fa, found = false;
      for (let k = 0; k < n; k++) {
        tb = tlo + (k + 1) * h; fb = f(px + tb * dx, py + tb * dy, pz + tb * dz);
        if (fb >= 0) { found = true; break; }
        ta = tb; fa = fb;
      }
      if (!found) break;
      let t = tb, side = 0;
      for (let it = 0; it < 60; it++) {
        t = (ta * fb - tb * fa) / (fb - fa);
        const ft = f(px + t * dx, py + t * dy, pz + t * dz);
        if (Math.abs(ft) < 1e-10 || tb - ta < 1e-10) break;
        if (ft > 0) { tb = t; fb = ft; if (side === 1) fa *= 0.5; side = 1; } else { ta = t; fa = ft; if (side === -1) fb *= 0.5; side = -1; }
      }
      px += t * dx; py += t * dy; pz += t * dz;
      const rr = Math.hypot(px, py), ex = px / rr, ey = py / rr;
      let ph = Math.atan2(py, px); if (ph < 0) ph += TWO_PI;
      const d = deltaEval(ph, pz), g = d[1] / rr;
      let nx = ex + g * ey, ny = ey - g * ex, nz = -(SLOPE + d[2]);
      const inv = 1 / Math.hypot(nx, ny, nz); nx *= inv; ny *= inv; nz *= inv;
      const dn = dx * nx + dy * ny + dz * nz;
      if (dn > 0) { dx -= 2 * dn * nx; dy -= 2 * dn * ny; dz -= 2 * dn * nz; }
      pts.push(px, py, pz);
      px -= 1e-9 * nx; py -= 1e-9 * ny; pz -= 1e-9 * nz;
      if (++nb >= maxb) return { pts, detected: false, bounces: nb };
    }
    const tend = dz > 0 ? (L - pz) / dz : -pz / dz;
    pts.push(px + tend * dx, py + tend * dy, pz + tend * dz);
    return { pts, detected: dz > 0, bounces: nb };
  }
  // one Lambertian ray from the entrance plane; u1, u2 uniform in [0, 1)
  function lambert(x, y, u1, u2) {
    if (f(x, y, 0) >= 0) return null;
    const s = Math.sqrt(u1), ps = TWO_PI * u2;
    return trace(x, y, 0, s * Math.cos(ps), s * Math.sin(ps), Math.sqrt(1 - s * s));
  }
  return { R1, R2, L, SLOPE, setWall, deltaEval, wallRadius, trace, lambert, inside: (x, y, z) => f(x, y, z) < 0 };
})();
if (typeof module !== 'undefined') module.exports = LG;
