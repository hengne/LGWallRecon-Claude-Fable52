# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""
Specular ray tracer for a hollow conical light guide with a free-form deformed inner wall.

Geometry (from 4cells_17.5mm_8mm_circle_to_circle_100mm-Cut.step, one cell):
    entrance  z = 0    radius R1 = 8.75 mm   (diameter 17.5 mm)
    exit      z = L    radius R2 = 4.00 mm   (diameter 8 mm), L = 100 mm   -> photodetector
    nominal wall   r0(z) = R1 + (R2 - R1) z / L           (half angle 2.72 deg)
    real wall      r(phi, z) = r0(z) + delta(phi, z)

delta(phi, z) is a tensor-product uniform cubic B-spline: periodic in phi (NPHI coefficients),
open in z (NZ cells -> NZ + 3 coefficients).  delta > 0 : wall further from the axis (a dent in the
wall, cavity locally wider); delta < 0 : wall bulging into the cavity (a bump).

Source: a small diffuse (Lambertian) spot at (xs, ys) in the plane z = -gap, Gaussian width `sig`.
Detector: every photon crossing z = L inside the wall is counted (uniform, angle independent).
Wall: perfect mirror (reflectivity `refl`, 1 by default).

All lengths in mm.
"""
import math
import numpy as np
from numba import njit, prange

import os
R1, R2, L = 8.75, 4.0, 100.0
SPOT_BLUR = 0.0      # Gaussian softening (mm) of the disc-shaped light spot; set from the edge fit by load_scan.spot()
USE_GPU = os.environ.get('LG_GPU', '0') == '1'      # run the wall tracer on the GPU (OpenCL, see lgcl.py)
SLOPE = (R2 - R1) / L
TWO_PI = 2.0 * math.pi


def pad_coeffs(c):
    """c[NPHI, NZ+3] -> periodic padding in phi, shape [NPHI+3, NZ+3] (row k holds index k-1 mod NPHI)."""
    c = np.ascontiguousarray(c, dtype=np.float64)
    return np.ascontiguousarray(np.concatenate([c[-1:], c, c[:2]], axis=0))


@njit(cache=True, fastmath=True, inline='always')
def delta_eval(cp, phi, z):
    """value, d/dphi, d/dz of the B-spline deformation."""
    nphi = cp.shape[0] - 3
    nz = cp.shape[1] - 3
    hp = TWO_PI / nphi
    hz = L / nz
    a = phi / hp
    i = int(a)
    u = a - i
    if i >= nphi:
        i -= nphi
    zz = z / hz
    j = int(zz)
    if j < 0:
        j = 0
    elif j > nz - 1:
        j = nz - 1
    v = zz - j
    um = 1.0 - u
    bu0 = um * um * um / 6.0
    bu1 = (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0
    bu2 = (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0
    bu3 = u * u * u / 6.0
    du0 = -0.5 * um * um
    du1 = 0.5 * (3.0 * u * u - 4.0 * u)
    du2 = 0.5 * (-3.0 * u * u + 2.0 * u + 1.0)
    du3 = 0.5 * u * u
    vm = 1.0 - v
    bv0 = vm * vm * vm / 6.0
    bv1 = (3.0 * v * v * v - 6.0 * v * v + 4.0) / 6.0
    bv2 = (-3.0 * v * v * v + 3.0 * v * v + 3.0 * v + 1.0) / 6.0
    bv3 = v * v * v / 6.0
    dv0 = -0.5 * vm * vm
    dv1 = 0.5 * (3.0 * v * v - 4.0 * v)
    dv2 = 0.5 * (-3.0 * v * v + 2.0 * v + 1.0)
    dv3 = 0.5 * v * v
    # contract along z for each of the four phi rows
    r0 = cp[i, j] * bv0 + cp[i, j + 1] * bv1 + cp[i, j + 2] * bv2 + cp[i, j + 3] * bv3
    r1 = cp[i + 1, j] * bv0 + cp[i + 1, j + 1] * bv1 + cp[i + 1, j + 2] * bv2 + cp[i + 1, j + 3] * bv3
    r2 = cp[i + 2, j] * bv0 + cp[i + 2, j + 1] * bv1 + cp[i + 2, j + 2] * bv2 + cp[i + 2, j + 3] * bv3
    r3 = cp[i + 3, j] * bv0 + cp[i + 3, j + 1] * bv1 + cp[i + 3, j + 2] * bv2 + cp[i + 3, j + 3] * bv3
    s0 = cp[i, j] * dv0 + cp[i, j + 1] * dv1 + cp[i, j + 2] * dv2 + cp[i, j + 3] * dv3
    s1 = cp[i + 1, j] * dv0 + cp[i + 1, j + 1] * dv1 + cp[i + 1, j + 2] * dv2 + cp[i + 1, j + 3] * dv3
    s2 = cp[i + 2, j] * dv0 + cp[i + 2, j + 1] * dv1 + cp[i + 2, j + 2] * dv2 + cp[i + 2, j + 3] * dv3
    s3 = cp[i + 3, j] * dv0 + cp[i + 3, j + 1] * dv1 + cp[i + 3, j + 2] * dv2 + cp[i + 3, j + 3] * dv3
    val = r0 * bu0 + r1 * bu1 + r2 * bu2 + r3 * bu3
    dph = r0 * du0 + r1 * du1 + r2 * du2 + r3 * du3
    dz = s0 * bu0 + s1 * bu1 + s2 * bu2 + s3 * bu3
    return val, dph / hp, dz / hz


@njit(cache=True, fastmath=True, inline='always')
def _f(cp, flat, x, y, z):
    """signed 'radial distance' to the wall: < 0 inside the cavity."""
    r = math.sqrt(x * x + y * y)
    if flat:
        return r - (R1 + SLOPE * z)
    ph = math.atan2(y, x)
    if ph < 0.0:
        ph += TWO_PI
    d, _, _ = delta_eval(cp, ph, z)
    return r - (R1 + SLOPE * z + d)


@njit(cache=True, fastmath=True, inline='always')
def _cone_exit(px, py, pz, dx, dy, dz, off):
    """smallest t > 0 where the ray crosses the cone r = R1 + off + SLOPE z (on the physical nappe)."""
    a0 = R1 + off + SLOPE * pz
    A = dx * dx + dy * dy - SLOPE * SLOPE * dz * dz
    B = px * dx + py * dy - SLOPE * a0 * dz
    C = px * px + py * py - a0 * a0
    best = 1e30
    if abs(A) < 1e-14:
        if abs(B) > 1e-300:
            t = -C / (2.0 * B)
            if t > 0.0 and a0 + SLOPE * dz * t > 0.0:
                best = t
        return best
    disc = B * B - A * C
    if disc < 0.0:
        return best
    sq = math.sqrt(disc)
    q = -(B + (sq if B >= 0.0 else -sq))
    t1 = q / A
    t2 = C / q if q != 0.0 else 1e30
    if t1 > 1e-12 and a0 + SLOPE * dz * t1 > 0.0 and t1 < best:
        best = t1
    if t2 > 1e-12 and a0 + SLOPE * dz * t2 > 0.0 and t2 < best:
        best = t2
    return best


@njit(cache=True, fastmath=True)
def trace_ray(cp, flat, dlo, dhi, px, py, pz, dx, dy, dz, refl, maxb, hit):
    """Follow one ray from inside the cavity.  Returns +weight if it leaves through z = L, -weight if it leaves through z = 0 (0 if lost).
    hit[0..2] <- exit position/‑ unused unless detected; hit[3] <- number of bounces."""
    w = 1.0
    nb = 0
    STEP = 3.0
    while True:
        # distance to the end planes
        if dz > 0.0:
            tend = (L - pz) / dz
        elif dz < 0.0:
            tend = (0.0 - pz) / dz
        else:
            tend = 1e30
        if flat:
            th = _cone_exit(px, py, pz, dx, dy, dz, 0.0)
            if th >= tend:
                break
            t = th
        else:
            # bracket: the wall lies between the cones offset by dlo and dhi (bounds of delta)
            r = math.sqrt(px * px + py * py)
            tlo = 0.0
            if r < R1 + SLOPE * pz + dlo:
                tlo = _cone_exit(px, py, pz, dx, dy, dz, dlo)
            if tlo >= tend:
                break
            thi = _cone_exit(px, py, pz, dx, dy, dz, dhi)
            if thi > tend:
                thi = tend
            # march to the first sign change of f
            n = int((thi - tlo) / STEP) + 1
            h = (thi - tlo) / n
            ta = tlo
            fa = _f(cp, flat, px + ta * dx, py + ta * dy, pz + ta * dz)
            if fa > 0.0:        # started (numerically) outside: treat start as on the wall
                fa = -1e-12
            found = False
            tb = ta; fb = fa
            for k in range(n):
                tb = tlo + (k + 1) * h
                fb = _f(cp, flat, px + tb * dx, py + tb * dy, pz + tb * dz)
                if fb >= 0.0:
                    found = True
                    break
                ta = tb; fa = fb
            if not found:
                break                      # reached an end plane while still inside
            # Illinois regula falsi on [ta, tb]
            side = 0
            t = tb
            for it in range(60):
                t = (ta * fb - tb * fa) / (fb - fa)
                ft = _f(cp, flat, px + t * dx, py + t * dy, pz + t * dz)
                if abs(ft) < 1e-10 or (tb - ta) < 1e-10:
                    break
                if ft > 0.0:
                    tb = t; fb = ft
                    if side == 1:
                        fa *= 0.5
                    side = 1
                else:
                    ta = t; fa = ft
                    if side == -1:
                        fb *= 0.5
                    side = -1
        # hit point and normal
        px += t * dx; py += t * dy; pz += t * dz
        r = math.sqrt(px * px + py * py)
        ex = px / r; ey = py / r
        if flat:
            nx = ex; ny = ey; nz = -SLOPE
        else:
            ph = math.atan2(py, px)
            if ph < 0.0:
                ph += TWO_PI
            d, dph, ddz = delta_eval(cp, ph, pz)
            g = dph / r
            nx = ex + g * ey           # r_hat - (1/r) d_phi(delta) phi_hat ; phi_hat = (-ey, ex)
            ny = ey - g * ex
            nz = -(SLOPE + ddz)
        inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
        nx *= inv; ny *= inv; nz *= inv
        dn = dx * nx + dy * ny + dz * nz
        if dn > 0.0:
            dx -= 2.0 * dn * nx; dy -= 2.0 * dn * ny; dz -= 2.0 * dn * nz
        # step a hair inside
        px -= 1e-9 * nx; py -= 1e-9 * ny; pz -= 1e-9 * nz
        w *= refl
        nb += 1
        if nb >= maxb or w < 1e-6:
            hit[3] = nb
            return 0.0
    hit[3] = nb
    hit[0] = px + tend * dx; hit[1] = py + tend * dy; hit[2] = dx; hit[4] = dy
    if dz > 0.0:
        return w          # left through the exit plane z = L
    return -w             # went back out through the entrance plane z = 0


# additive-recurrence (R_4) quasi random numbers in 32-bit integer arithmetic (identical on CPU and GPU):
#   u_j(k) = frac(shift_j + k * g^-j),  g = 1.2207440846...,  stored as multiples of 2^-32
_QA = np.array([3518319155, 2882110345, 2360945575, 1934021721, 1584297434, 1297812911], dtype=np.uint64)
_M32 = np.uint64(0xFFFFFFFF)


@njit(cache=True, fastmath=True)
def _inside_entrance(cp, flat, x, y, z):
    return _f(cp, flat, x, y, z) < 0.0


@njit(cache=True, parallel=True, fastmath=True)
def trace_map(xs, ys, cp, flat, dlo, dhi, nray, smax, sig, gap, refl, lamb_n, sh, back, blur):
    """Transmission for every source position (xs[i], ys[i]) (cell-centred coordinates, mm).

    nray   rays per position (quasi random, with a fixed per-position random shift `sh` -> common random numbers)
    smax   only directions with sin(theta) <= smax are sampled (the rest is known not to reach the exit)
    sig    light spot: > 0 Gaussian sigma (mm), < 0 uniform disc of radius -sig (mm);  gap : distance of the source plane in front of z = 0
    lamb_n angular law  I ~ cos^lamb_n(theta)   (1 = Lambertian)
    Returns the detected fraction of ALL light emitted into the forward hemisphere.
    """
    npt = xs.shape[0]
    out = np.zeros(npt)
    # normalisation of cos^n over the hemisphere, sampling uniform in the projected disc (pdf ~ cos)
    # fraction = smax^2 * mean( (n+1)/2 * cos^(n-1) ) over samples (with zeros for lost rays)
    for i in prange(npt):
        s0 = np.uint64(sh[i, 0]); s1 = np.uint64(sh[i, 1]); s2 = np.uint64(sh[i, 2]); s3 = np.uint64(sh[i, 3])
        hit = np.zeros(5)
        acc = 0.0
        for k in range(nray):
            kk = np.uint64(k + 1)
            u1 = float((s0 + kk * _QA[0]) & _M32) / 4294967296.0
            u2 = float((s1 + kk * _QA[1]) & _M32) / 4294967296.0
            u3 = float((s2 + kk * _QA[2]) & _M32) / 4294967296.0
            u4 = float((s3 + kk * _QA[3]) & _M32) / 4294967296.0
            st = smax * math.sqrt(u1)
            ps = TWO_PI * u2
            dx = st * math.cos(ps); dy = st * math.sin(ps); dz = math.sqrt(1.0 - st * st)
            wt = 1.0
            if lamb_n != 1.0:
                wt = 0.5 * (lamb_n + 1.0) * dz ** (lamb_n - 1.0)
            px = xs[i]; py = ys[i]
            if sig > 0.0:          # Gaussian spot
                rr = sig * math.sqrt(-2.0 * math.log(1.0 - u3))
                px += rr * math.cos(TWO_PI * u4); py += rr * math.sin(TWO_PI * u4)
            elif sig < 0.0:        # uniform disc of radius -sig
                rr = -sig * math.sqrt(u3)
                px += rr * math.cos(TWO_PI * u4); py += rr * math.sin(TWO_PI * u4)
                if blur > 0.0:     # soft edge: Gaussian blur of the disc
                    s4 = (s0 * np.uint64(2654435761) + s3) & _M32; s5 = (s1 * np.uint64(2246822519) + s2) & _M32
                    u5 = float((s4 + kk * _QA[4]) & _M32) / 4294967296.0
                    u6 = float((s5 + kk * _QA[5]) & _M32) / 4294967296.0
                    rr = blur * math.sqrt(-2.0 * math.log(1.0 - u5))
                    px += rr * math.cos(TWO_PI * u6); py += rr * math.sin(TWO_PI * u6)
            if gap > 0.0:
                tt = gap / dz
                px += tt * dx; py += tt * dy
            if back:
                # reverse run (used for the reciprocity test): start on the exit plane, count rays reaching z = 0
                if not _inside_entrance(cp, flat, px, py, L):
                    continue
                res = trace_ray(cp, flat, dlo, dhi, px, py, L, dx, dy, -dz, refl, 2000, hit)
                if res < 0.0:
                    acc -= wt * res
                continue
            if not _inside_entrance(cp, flat, px, py, 0.0):
                continue
            res = trace_ray(cp, flat, dlo, dhi, px, py, 0.0, dx, dy, dz, refl, 2000, hit)
            if res > 0.0:
                acc += wt * res
        out[i] = smax * smax * acc / nray
    return out


def transmission(xs, ys, coeffs=None, nray=4000, smax=1.0, sig=0.0, gap=0.0, refl=1.0, lamb_n=1.0, seed=1, back=False, gpu=None, blur=None):
    if blur is None:
        blur = SPOT_BLUR
    xs = np.ascontiguousarray(xs, dtype=np.float64).ravel()
    ys = np.ascontiguousarray(ys, dtype=np.float64).ravel()
    if coeffs is None:
        cp = np.zeros((7, 7)); flat = True; dlo = dhi = 0.0
    else:
        cp = pad_coeffs(coeffs); flat = False
        dlo, dhi = delta_bounds(cp)
    if USE_GPU and not flat and not back and gap == 0.0 and lamb_n == 1.0 and gpu is not False or gpu is True:
        import lgcl
        return lgcl.transmission(xs, ys, cp, dlo, dhi, nray, smax, sig, refl, seed, blur=blur)
    from lgcl import shifts
    return trace_map(xs, ys, cp, flat, dlo, dhi, nray, smax, sig, gap, refl, lamb_n, shifts(len(xs), seed), back, blur)


@njit(cache=True)
def _delta_grid(cp, n_phi, n_z):
    out = np.empty((n_phi, n_z))
    for i in range(n_phi):
        for j in range(n_z):
            out[i, j] = delta_eval(cp, TWO_PI * i / n_phi * 0.999999, L * j / (n_z - 1))[0]
    return out


def delta_bounds(cp):
    """Tight lower / upper bound of delta inside the guide (sampled on a 6x finer grid than the spline cells,
    plus a small safety margin).  They define the two bounding cones that bracket every wall hit."""
    g = _delta_grid(cp, 6 * (cp.shape[0] - 3), 6 * (cp.shape[1] - 3) + 1)
    lo, hi = float(g.min()), float(g.max()); m = 0.03 * (hi - lo) + 2e-3
    return lo - m, hi + m
