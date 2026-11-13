# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""OpenCL (GPU) version of the light-guide ray tracer in lgtrace.py.

Same geometry, same B-spline wall, same wall-zone marching + regula-falsi intersection, same quasi-random rays
(integer additive recurrence, so CPU and GPU trace the same rays).  Single precision by default
(the Radeon 8060S is ~12x slower in double); set double=True for a bit-for-bit style cross-check.

Work is submitted in slices of well under a second so the GPU is never monopolised and the driver's
job watchdog is never approached.  Host buffers are a few MB.
"""
import os, time
import numpy as np

_SRC = r"""
#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real;
#define TOLF ((real)1e-10)
#define TOLT ((real)1e-10)
#define TOLG ((real)1e-10)
#define NUDGE ((real)1e-9)
#define U01(u) ((real)(u) / (real)4294967296.0)
#else
typedef float real;
#define TOLF ((real)2e-6)
#define TOLT ((real)2e-6)
#define TOLG ((real)1e-6)
#define NUDGE ((real)2e-5)
#define U01(u) ((real)((u) >> 8) / (real)16777216.0)
#endif
#define R1 ((real)8.75)
#define LEN ((real)100.0)
#define SLOPE ((real)-0.0475)
#define TWO_PI ((real)6.283185307179586)
#define BIG ((real)1e30)
#define STEP ((real)0.5)
#define MAXB 600

static inline void bas(real u, real *b) {
    real um = (real)1 - u;
    b[0] = um * um * um / (real)6;
    b[1] = ((real)3 * u * u * u - (real)6 * u * u + (real)4) / (real)6;
    b[2] = ((real)-3 * u * u * u + (real)3 * u * u + (real)3 * u + (real)1) / (real)6;
    b[3] = u * u * u / (real)6;
}
static inline void dbas(real u, real *d) {
    real um = (real)1 - u;
    d[0] = (real)-0.5 * um * um;
    d[1] = (real)0.5 * ((real)3 * u * u - (real)4 * u);
    d[2] = (real)0.5 * ((real)-3 * u * u + (real)2 * u + (real)1);
    d[3] = (real)0.5 * u * u;
}
static inline void cell(int nphi, int nz, real phi, real z, int *i, int *j, real *u, real *v) {
    real a = phi * (real)nphi / TWO_PI;
    int ii = (int)a; *u = a - (real)ii; if (ii >= nphi) ii -= nphi; if (ii < 0) ii = 0;
    real zz = z * (real)nz / LEN;
    int jj = (int)floor(zz); if (jj < 0) jj = 0; else if (jj > nz - 1) jj = nz - 1;
    *v = zz - (real)jj; *i = ii; *j = jj;
}
static real delta_val(__global const real *cp, int nphi, int nk, real phi, real z) {
    int i, j; real u, v, bu[4], bv[4];
    cell(nphi, nk - 3, phi, z, &i, &j, &u, &v); bas(u, bu); bas(v, bv);
    real val = 0;
    for (int k = 0; k < 4; k++) {
        int o = (i + k) * nk + j;
        val += bu[k] * (cp[o] * bv[0] + cp[o + 1] * bv[1] + cp[o + 2] * bv[2] + cp[o + 3] * bv[3]);
    }
    return val;
}
static void delta_grad(__global const real *cp, int nphi, int nk, real phi, real z, real *dph, real *dz) {
    int i, j; real u, v, bu[4], bv[4], du[4], dv[4];
    cell(nphi, nk - 3, phi, z, &i, &j, &u, &v); bas(u, bu); bas(v, bv); dbas(u, du); dbas(v, dv);
    real a = 0, b = 0;
    for (int k = 0; k < 4; k++) {
        int o = (i + k) * nk + j;
        a += du[k] * (cp[o] * bv[0] + cp[o + 1] * bv[1] + cp[o + 2] * bv[2] + cp[o + 3] * bv[3]);
        b += bu[k] * (cp[o] * dv[0] + cp[o + 1] * dv[1] + cp[o + 2] * dv[2] + cp[o + 3] * dv[3]);
    }
    *dph = a * (real)nphi / TWO_PI; *dz = b * (real)(nk - 3) / LEN;
}
static inline real wallf(__global const real *cp, int nphi, int nk, real x, real y, real z) {
    real ph = atan2(y, x); if (ph < 0) ph += TWO_PI;
    return sqrt(x * x + y * y) - (R1 + SLOPE * z + delta_val(cp, nphi, nk, ph, z));
}
/* f and df/dt along the ray at parameter t (value and gradient of the spline in one pass) */
static inline real wallfg(__global const real *cp, int nphi, int nk, real px, real py, real pz,
                          real dx, real dy, real dz, real t, real *g) {
    real x = px + t * dx, y = py + t * dy, z = pz + t * dz;
    real r = sqrt(x * x + y * y);
    real ph = atan2(y, x); if (ph < 0) ph += TWO_PI;
    int i, j; real u, v, bu[4], bv[4], du[4], dv[4];
    cell(nphi, nk - 3, ph, z, &i, &j, &u, &v); bas(u, bu); bas(v, bv); dbas(u, du); dbas(v, dv);
    real val = 0, a = 0, b = 0;
    for (int k = 0; k < 4; k++) {
        int o = (i + k) * nk + j;
        real rv = cp[o] * bv[0] + cp[o + 1] * bv[1] + cp[o + 2] * bv[2] + cp[o + 3] * bv[3];
        real sv = cp[o] * dv[0] + cp[o + 1] * dv[1] + cp[o + 2] * dv[2] + cp[o + 3] * dv[3];
        val += bu[k] * rv; a += du[k] * rv; b += bu[k] * sv;
    }
    real dph = a * (real)nphi / TWO_PI, ddz = b * (real)(nk - 3) / LEN;
    *g = (x * dx + y * dy) / r - SLOPE * dz - dph * (x * dy - y * dx) / (r * r) - ddz * dz;
    return r - (R1 + SLOPE * z + val);
}
static real cone_exit(real px, real py, real pz, real dx, real dy, real dz, real off) {
    real a0 = R1 + off + SLOPE * pz;
    real A = dx * dx + dy * dy - SLOPE * SLOPE * dz * dz;
    real B = px * dx + py * dy - SLOPE * a0 * dz;
    real C = px * px + py * py - a0 * a0;
    real best = BIG;
    if (fabs(A) < (real)1e-12) {
        if (fabs(B) > (real)1e-30) { real t = -C / ((real)2 * B); if (t > 0 && a0 + SLOPE * dz * t > 0) best = t; }
        return best;
    }
    real disc = B * B - A * C;
    if (disc < 0) return best;
    real sq = sqrt(disc);
    real q = -(B + (B >= 0 ? sq : -sq));
    real t1 = q / A, t2 = (q != 0) ? C / q : BIG;
    if (t1 > (real)1e-7 && a0 + SLOPE * dz * t1 > 0 && t1 < best) best = t1;
    if (t2 > (real)1e-7 && a0 + SLOPE * dz * t2 > 0 && t2 < best) best = t2;
    return best;
}
/* +w : left through z = LEN (detected);  -w : back out through z = 0;  0 : lost */
static real trace_ray(__global const real *cp, int nphi, int nk, real dlo, real dhi,
                      real px, real py, real pz, real dx, real dy, real dz, real refl) {
    real w = 1; real tend = 0;
    for (int nb = 0; nb < MAXB; nb++) {
        tend = dz > 0 ? (LEN - pz) / dz : (dz < 0 ? -pz / dz : BIG);
        /* wall zone between the bounding cones: only there can the ray hit the wall (see lgtrace.py) */
        real thi = cone_exit(px, py, pz, dx, dy, dz, dhi);
        if (thi > tend) thi = tend;
        real a0 = R1 + dlo + SLOPE * pz;
        real A = dx * dx + dy * dy - SLOPE * SLOPE * dz * dz;
        real B = px * dx + py * dy - SLOPE * a0 * dz;
        real C = px * px + py * py - a0 * a0;
        real s0 = 0, e0 = thi, s1 = 0, e1 = -1;
        real disc = B * B - A * C;
        if (fabs(A) < (real)1e-12) {
            if (fabs(B) > (real)1e-30) { real tr = (real)-0.5 * C / B; if (B < 0) e0 = fmin(thi, tr); else s0 = fmax((real)0, tr); }
            else if (C < 0) e0 = -1;
        } else if (disc <= 0) {
            if (A < 0) e0 = -1;
        } else {
            real sq = sqrt(disc);
            real q = -(B + (B >= 0 ? sq : -sq));
            real ra = q / A, rb = (q != 0) ? C / q : ra;
            if (ra > rb) { real tmp = ra; ra = rb; rb = tmp; }
            if (A > 0) { if (rb > 0 && ra < thi) { e0 = ra; s1 = rb; e1 = thi; } }
            else { s0 = fmax((real)0, ra); e0 = fmin(thi, rb); }
        }
        real ta = 0, fa = 0, tb = 0, fb = 0; int found = 0;
        for (int seg = 0; seg < 2 && !found; seg++) {
            real sa = seg == 0 ? s0 : s1, sb = seg == 0 ? e0 : e1;
            if (sb <= sa) continue;
            int n = (int)((sb - sa) / STEP) + 1;
            real h = (sb - sa) / (real)n, ga, gb;
            ta = sa; fa = wallfg(cp, nphi, nk, px, py, pz, dx, dy, dz, ta, &ga);
            if (fa > 0) fa = (real)-1e-7;
            for (int k = 0; k < n; k++) {
                tb = sa + (real)(k + 1) * h;
                fb = wallfg(cp, nphi, nk, px, py, pz, dx, dy, dz, tb, &gb);
                if (fb >= 0) { found = 1; break; }
                if (ga > 0 && gb < 0) {          /* maximum of f between the samples: a grazing hit? */
                    real tl = ta, gl = ga, tr = tb, gr = gb; int sd = 0;
                    for (int it = 0; it < 30; it++) {
                        real tm = (tl * gr - tr * gl) / (gr - gl), gm;
                        real fm = wallfg(cp, nphi, nk, px, py, pz, dx, dy, dz, tm, &gm);
                        if (fm >= 0) { tb = tm; fb = fm; found = 1; break; }
                        if (fabs(gm) < TOLG || (tr - tl) < TOLT) break;
                        if (gm > 0) { tl = tm; gl = gm; if (sd == 1) gr *= (real)0.5; sd = 1; }
                        else { tr = tm; gr = gm; if (sd == -1) gl *= (real)0.5; sd = -1; }
                    }
                    if (found) break;
                }
                ta = tb; fa = fb; ga = gb;
            }
        }
        if (!found) return dz > 0 ? w : -w;
        real t = tb; int side = 0;
        for (int it = 0; it < 40; it++) {
            t = (ta * fb - tb * fa) / (fb - fa);
            real ft = wallf(cp, nphi, nk, px + t * dx, py + t * dy, pz + t * dz);
            if (fabs(ft) < TOLF || (tb - ta) < TOLT) break;
            if (ft > 0) { tb = t; fb = ft; if (side == 1) fa *= (real)0.5; side = 1; }
            else { ta = t; fa = ft; if (side == -1) fb *= (real)0.5; side = -1; }
        }
        px += t * dx; py += t * dy; pz += t * dz;
        real r = sqrt(px * px + py * py);
        real ex = px / r, ey = py / r;
        real ph = atan2(py, px); if (ph < 0) ph += TWO_PI;
        real dph, ddz; delta_grad(cp, nphi, nk, ph, pz, &dph, &ddz);
        real g = dph / r;
        real nx = ex + g * ey, ny = ey - g * ex, nz = -(SLOPE + ddz);
        real inv = (real)1 / sqrt(nx * nx + ny * ny + nz * nz);
        nx *= inv; ny *= inv; nz *= inv;
        real dn = dx * nx + dy * ny + dz * nz;
        if (dn > 0) { dx -= (real)2 * dn * nx; dy -= (real)2 * dn * ny; dz -= (real)2 * dn * nz; }
        px -= NUDGE * nx; py -= NUDGE * ny; pz -= NUDGE * nz;
        w *= refl;
        if (w < (real)1e-6) return 0;
    }
    return 0;
}
/* additive-recurrence constants, the same as _QA in lgtrace.py (R_4 sequence, g = 1.1673039782614187) */
#define A1 3679390609u
#define A2 3152041523u
#define A3 2700274807u
#define A4 2313257605u
#define A5 1779033703u
#define A6 3144134277u
__kernel void trace_map(__global const real *cp, int nphi, int nk, real dlo, real dhi,
                        __global const float *xy, __global const uint *sh,
                        int nchunk, int rch, real smax, real spot, real blur, real refl, int pt0,
                        __global float *out)
{
    int gid = get_global_id(0);
    int ip = pt0 + gid / nchunk, ic = gid % nchunk;
    real x0 = xy[2 * ip], y0 = xy[2 * ip + 1];
    uint s0 = sh[4 * ip], s1 = sh[4 * ip + 1], s2 = sh[4 * ip + 2], s3 = sh[4 * ip + 3];
    real acc = 0;
    for (int kk = 0; kk < rch; kk++) {
        uint k = (uint)(ic * rch + kk + 1);
        real u1 = U01(s0 + k * A1), u2 = U01(s1 + k * A2), u3 = U01(s2 + k * A3), u4 = U01(s3 + k * A4);
        real st = smax * sqrt(u1), ps = TWO_PI * u2;
        real dx = st * cos(ps), dy = st * sin(ps), dz = sqrt((real)1 - st * st);
        real px = x0, py = y0;
        if (spot > 0) { real rr = spot * sqrt((real)-2 * log((real)1 - u3)); px += rr * cos(TWO_PI * u4); py += rr * sin(TWO_PI * u4); }
        else if (spot < 0) {
            real rr = -spot * sqrt(u3); px += rr * cos(TWO_PI * u4); py += rr * sin(TWO_PI * u4);
            if (blur > 0) {
                uint s4 = s0 * 2654435761u + s3, s5 = s1 * 2246822519u + s2;
                real u5 = U01(s4 + k * A5), u6 = U01(s5 + k * A6);
                rr = blur * sqrt((real)-2 * log((real)1 - u5)); px += rr * cos(TWO_PI * u6); py += rr * sin(TWO_PI * u6);
            }
        }
        if (wallf(cp, nphi, nk, px, py, (real)0) >= 0) continue;
        real res = trace_ray(cp, nphi, nk, dlo, dhi, px, py, (real)0, dx, dy, dz, refl);
        if (res > 0) acc += res;
    }
    out[gid] = (float)acc;
}
"""

_state = {}


def _get(double):
    key = 'd' if double else 'f'
    if key not in _state:
        import pyopencl as cl
        dev = [d for p in cl.get_platforms() for d in p.get_devices() if d.type & cl.device_type.GPU][0]
        ctx = cl.Context([dev]); q = cl.CommandQueue(ctx)
        prg = cl.Program(ctx, _SRC).build(options=['-DUSE_DOUBLE'] if double else [])
        _state[key] = (cl, ctx, q, cl.Kernel(prg, 'trace_map'), {'rate': 2.0e6})
    return _state[key]


def shifts(npt, seed):
    """Per-point Cranley-Patterson shifts of the quasi-random ray sequence: uint32 [npt, 4] (shared by CPU and GPU)."""
    with np.errstate(over='ignore'):
        s = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + (np.arange(npt, dtype=np.uint64) + np.uint64(1)) * np.uint64(0xBF58476D1CE4E5B9)
        out = np.empty((npt, 4), np.uint32)
        for k in range(4):
            s ^= s >> np.uint64(30); s *= np.uint64(0xBF58476D1CE4E5B9)
            s ^= s >> np.uint64(27); s *= np.uint64(0x94D049BB133111EB)
            s ^= s >> np.uint64(31)
            out[:, k] = (s >> np.uint64(32)).astype(np.uint32)
    return out


def transmission(xs, ys, cp, dlo, dhi, nray, smax, sig, refl, seed, double=False, slice_s=0.6, blur=0.0):
    """cp: padded coefficients [NPHI+3, NZ+3] (lgtrace.pad_coeffs).  Returns T per point (float64)."""
    cl, ctx, q, kern, st = _get(double)
    real = np.float64 if double else np.float32
    npt = len(xs); nphi = cp.shape[0] - 3; nk = cp.shape[1]
    rch = 64 if nray >= 64 else int(nray)
    nchunk = max(1, int(round(nray / rch))); ntot = nchunk * rch
    mf = cl.mem_flags
    b_cp = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(cp, dtype=real))
    xy = np.ascontiguousarray(np.column_stack([xs, ys]), dtype=np.float32)
    b_xy = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xy)
    b_sh = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shifts(npt, seed))
    out = np.zeros(npt)
    pt0 = 0
    while pt0 < npt:
        nb = int(min(npt - pt0, max(1, st['rate'] * slice_s / ntot)))
        b_out = cl.Buffer(ctx, mf.WRITE_ONLY, nb * nchunk * 4)
        t0 = time.time()
        kern(q, (nb * nchunk,), None, b_cp, np.int32(nphi), np.int32(nk), real(dlo), real(dhi), b_xy, b_sh,
             np.int32(nchunk), np.int32(rch), real(smax), real(sig), real(blur), real(refl), np.int32(pt0), b_out).wait()
        dt = max(time.time() - t0, 1e-4)
        res = np.empty(nb * nchunk, np.float32); cl.enqueue_copy(q, res, b_out).wait(); b_out.release()
        out[pt0:pt0 + nb] = res.reshape(nb, nchunk).sum(axis=1, dtype=np.float64)
        rate = nb * ntot / dt
        st['rate'] = 0.5 * st['rate'] + 0.5 * rate if dt > 0.05 else max(st['rate'], min(rate, 4 * st['rate']))
        pt0 += nb
    for b in (b_cp, b_xy, b_sh): b.release()
    return smax * smax * out / ntot
