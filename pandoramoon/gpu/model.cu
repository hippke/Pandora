// Pandora GPU model: FP32 planet+moon log-likelihood, one warp per call (GPU_PLAN.md, GPU_MODEL.md).
//
// Compiled at run time by NVRTC (pandoramoon/gpu/_cuda.py) with these macros:
//   SS        supersampling factor (sub-exposures per cadence)
//   WARPS     warps (= calls) per block;  MINB  minimum resident blocks per SM (__launch_bounds__)
//   LY0, LSC, IY0, ISC   the sqrt(k) maps of the occultation tables (occult.LIMB_TAB_*, INT_TAB_*)
//   GDAY      G DAY^2 / (4 pi^2)  [SI]
//
// Precision: FP32 everywhere except a handful of FP64 operations per call (parameter validity, the per-call
// reference offsets of the planet's epoch and the moon's phase) and the final chi^2 reduction. Every formula and
// its error analysis: GPU_MODEL.md. The CPU model (pandoramoon.kernel.pandora_loglike) is the reference.

#define LM1 13          // LIMB_TAB_M + 1
#define IM1 33          // INT_TAB_M + 1
#define PI_F 3.14159265358979f
#define TWO_PI_F 6.28318530717959f
#define ROW 36          // shared-memory row of one coefficient set (36: bank-shifted)
#ifdef NO_ECC
#define ECC_ON 0        // compiled for circular moons only (GPULikelihood(eccentric=False)): no Kepler loop
#else
#define ECC_ON 1
#endif

// ------------------------------------------------------------------------------------------- occultation
// deficit 1 - F of one body at squared distance z2 [R_star^2], from the per-call coefficient rows in shared
// memory: interior (0 <= z < 1 - k, P21c: Horner in x = 2 s / (1 - k) - 1, s = sqrt((1-k)^2 - z^2)) or limb
// (|z - 1| < k, P21b: outer half z >= 1 in x = 2 sqrt((k - t) / k) - 1, inner half in x = 2 sqrt((k + t) / k) - 1,
// t = z - 1). All three polynomials are evaluated as one Horner of degree 15 over float4 loads (the unused top
// coefficients are 0), so lanes in different regions do not diverge in the Horner.
struct Body {
    float k, omk2, inv_h, one_k2, inv_k;
    const float* s_int;     // 16 coefficients
    const float* s_limb;    // outer [0, 16), inner [16, 32)
};

__device__ __forceinline__ void body_init(Body& b, float k, const float* s_int, const float* s_limb) {
    b.k = k;
    float omk = 1.f - k;
    b.omk2 = omk * omk;
    b.inv_h = 2.f / omk;
    b.one_k2 = (1.f + k) * (1.f + k);
    b.inv_k = 1.f / k;
    b.s_int = s_int;
    b.s_limb = s_limb;
}

__device__ __forceinline__ float deficit(float z2, const Body& b) {
    if (!(z2 < b.one_k2)) return 0.f;              // off the star (also NaN)
    float x;
    const float4* c4;
    if (z2 < b.omk2) {
        x = sqrtf(b.omk2 - z2) * b.inv_h - 1.f;
        c4 = reinterpret_cast<const float4*>(b.s_int);
    } else {
        float t = sqrtf(z2) - 1.f;
        bool out = t >= 0.f;
        float r = (out ? (b.k - t) : (b.k + t)) * b.inv_k;
        x = 2.f * sqrtf(fmaxf(r, 0.f)) - 1.f;
        c4 = reinterpret_cast<const float4*>(b.s_limb + (out ? 0 : 16));
    }
    float4 q3 = c4[3], q2 = c4[2], q1 = c4[1], q0 = c4[0];
    float P = q3.w;
    P = fmaf(P, x, q3.z); P = fmaf(P, x, q3.y); P = fmaf(P, x, q3.x);
    P = fmaf(P, x, q2.w); P = fmaf(P, x, q2.z); P = fmaf(P, x, q2.y); P = fmaf(P, x, q2.x);
    P = fmaf(P, x, q1.w); P = fmaf(P, x, q1.z); P = fmaf(P, x, q1.y); P = fmaf(P, x, q1.x);
    P = fmaf(P, x, q0.w); P = fmaf(P, x, q0.z); P = fmaf(P, x, q0.y); P = fmaf(P, x, q0.x);
    return P > 0.f ? P : 0.f;
}

// limb coefficient `lane` (0..31) of two bodies: coef_j = sum_m T_m(t) (w0 LT[m, j] + w1 LT[M1 + m, j] +
// w2 LT[2 M1 + m, j]), t linear in sqrt(k) (occult._limb_coeffs_tab). Both bodies in one loop.
__device__ __forceinline__ void cheb_limb2(int lane, float ka, float kb, float w0, float w1, float w2,
                                           const float* __restrict__ LT, float& sa, float& sb) {
    float ta = (sqrtf(ka) - LY0) * LSC - 1.f;
    float tb = (sqrtf(kb) - LY0) * LSC - 1.f;
    float g0 = w0 * LT[lane] + w1 * LT[LM1 * 32 + lane] + w2 * LT[2 * LM1 * 32 + lane];
    float g1 = w0 * LT[32 + lane] + w1 * LT[(LM1 + 1) * 32 + lane] + w2 * LT[(2 * LM1 + 1) * 32 + lane];
    sa = fmaf(ta, g1, g0);
    sb = fmaf(tb, g1, g0);
    float am1 = 1.f, am = ta, bm1 = 1.f, bm = tb;
#pragma unroll
    for (int m = 2; m < LM1; ++m) {
        float an = 2.f * ta * am - am1; am1 = am; am = an;
        float bn = 2.f * tb * bm - bm1; bm1 = bm; bm = bn;
        float g = w0 * LT[m * 32 + lane] + w1 * LT[(LM1 + m) * 32 + lane] + w2 * LT[(2 * LM1 + m) * 32 + lane];
        sa = fmaf(an, g, sa);
        sb = fmaf(bn, g, sb);
    }
}

// interior coefficient j (0..15) of one body (occult._int_coeffs_tab)
__device__ __forceinline__ float cheb_int(int j, float k, float u1, float u2, const float* __restrict__ IT) {
    float om = 1.f / (1.f - u1 * (1.f / 3.f) - u2 * (1.f / 6.f));
    float c1 = 1.f - u1 - 2.f * u2, c2 = u1 + 2.f * u2, k2 = k * k;
    float t = (sqrtf(k) - IY0) * ISC - 1.f;
    float s = fmaf(IT[16 + j], t, IT[j]);
    float tm1 = 1.f, tm = t;
#pragma unroll
    for (int m = 2; m < IM1; ++m) {
        float tn = 2.f * t * tm - tm1; tm1 = tm; tm = tn;
        s = fmaf(IT[m * 16 + j], tn, s);
    }
    s *= c2 * om;
    float q = k2 * (1.f - k) * (1.f - k);
    if (j == 0) s += om * (c1 * k2 + u2 * (0.5f * k2 * k2 + 0.75f * q));
    else if (j == 1) s -= om * u2 * 0.5f * q;
    else if (j == 2) s -= om * u2 * 0.25f * q;
    return s;
}

// ------------------------------------------------------------------------------------------- disk overlaps
// Common overlap of N = 2 or 3 disks (GPU_MODEL.md section 4): boundary = for every circle the arcs inside all other
// disks; area = sum over arcs of r^2 (L - sin L) / 2 + (P_s x P_e) / 2. Coordinates relative to the moon centre.
struct Circ { float x, y, r; };

// circle i restricted to disk j: 0 disjoint (the overlap is empty), 1 all of circle i, 2 none of it,
// 3 the interval [s, s + L]
__device__ __forceinline__ int interval(const Circ& ci, const Circ& cj, float& s, float& L) {
    float dx = cj.x - ci.x, dy = cj.y - ci.y;
    float d2 = fmaf(dx, dx, dy * dy);
    float d = sqrtf(d2);
    if (d >= ci.r + cj.r) return 0;
    if (d <= cj.r - ci.r) return 1;
    if (d <= ci.r - cj.r) return 2;
    float dmi = fmaf(dx, dx, fmaf(dy, dy, -ci.r * ci.r)) / (d + ci.r);    // d - r_i without cancellation
    float dmj = fmaf(dx, dx, fmaf(dy, dy, -cj.r * cj.r)) / (d + cj.r);    // d - r_j
    float p = (ci.r + cj.r + d) * (cj.r - dmi) * (dmj + ci.r) * (dmi + cj.r);   // (2 d h)^2, Heron
    float h = sqrtf(fmaxf(p, 0.f)) / (2.f * d);
    float a = (d2 + (ci.r - cj.r) * (ci.r + cj.r)) / (2.f * d);
    float alpha = atan2f(h, a);
    s = atan2f(dy, dx) - alpha;
    L = 2.f * alpha;
    return 3;
}

__device__ __forceinline__ float seg(float L) {                          // L - sin L
    if (L < 0.5f) {
        float L2 = L * L;
        return L * L2 * (1.f / 6.f - L2 * (1.f / 120.f - L2 * (1.f / 5040.f - L2 * (1.f / 362880.f))));
    }
    return L - sinf(L);
}

__device__ __forceinline__ float arc_terms(const Circ& c, float s, float L) {
    float ss, cs, se, ce;
    sincosf(s, &ss, &cs);
    sincosf(s + L, &se, &ce);
    float psx = fmaf(c.r, cs, c.x), psy = fmaf(c.r, ss, c.y);
    float pex = fmaf(c.r, ce, c.x), pey = fmaf(c.r, se, c.y);
    return 0.5f * c.r * c.r * seg(L) + 0.5f * (psx * pey - psy * pex);
}

template <int N>
__device__ float area_disks(const Circ* C) {
    float area = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        float s0 = 0.f, L0 = 0.f, s1 = 0.f, L1 = 0.f;
        int nv = 0;
        bool empty = false;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            float sj, Lj;
            int t = interval(C[i], C[j], sj, Lj);
            if (t == 0) return 0.f;
            if (t == 2) empty = true;
            else if (t == 3) {
                if (nv == 0) { s0 = sj; L0 = Lj; } else { s1 = sj; L1 = Lj; }
                ++nv;
            }
        }
        if (empty) continue;
        if (nv == 0) return PI_F * C[i].r * C[i].r;          // circle i inside all other disks
        if (nv == 1) {
            if (L0 > 0.f) area += arc_terms(C[i], s0, L0);
        } else {
            float dl = s1 - s0;                               // offset of interval 1 in [0, 2 pi)
            dl -= TWO_PI_F * floorf(dl * (1.f / TWO_PI_F));
            if (dl < L0) {
                float L = fminf(L0, dl + L1) - dl;
                if (L > 0.f) area += arc_terms(C[i], s0 + dl, L);
            }
            if (dl + L1 > TWO_PI_F) {
                float L = fminf(L0, dl + L1 - TWO_PI_F);
                if (L > 0.f) area += arc_terms(C[i], s0, L);
            }
        }
    }
    return area > 0.f ? area : 0.f;
}

// er = min(1, A(star & planet & moon) / A(star & moon)) (overlap_analytic.er_fast)
__device__ float er3(float xp, float yp, float rp, float xm, float ym, float rm) {
    Circ C[3];
    C[0].x = -xm; C[0].y = -ym; C[0].r = 1.f;
    C[1].x = 0.f; C[1].y = 0.f; C[1].r = rm;
    float a_sm = area_disks<2>(C);
    if (!(a_sm > 0.f)) return 1.f;
    C[2].x = xp - xm; C[2].y = yp - ym; C[2].r = rp;
    return fminf(1.f, area_disks<3>(C) / a_sm);
}

// moon deficit after the planet-moon overlap (eclipse.eclipse for one sample): er from the two-circle ratio
// (eclipse_ratio) or, with a body on the stellar limb, from er3; plus the T4 gradient term (lens_gradient_term)
// for partial overlaps
template <bool EXACT>
__device__ __forceinline__ float eclipse_dm(float dm, float xp, float yp, float xm, float ym, float d2, float kp,
                                            float km, float zm2, float zp2, float u1, float u2, float inv_f0,
                                            bool on_limb) {
    float d = sqrtf(d2), zm = sqrtf(zm2);
    if (kp - km > d) return 0.f;                       // moon fully behind the planet: er = 1
    float km2 = km * km, kp2 = kp * kp, A, sdist;
    if (d <= km - kp) { A = PI_F * kp2; sdist = 0.f; }
    else {
        float a = (d2 + kp2 - km2) / (2.f * d);
        float h = sqrtf(fmaxf(kp2 - a * a, 0.f));
        float b = d - a;
        float A_p = kp2 * acosf(fminf(fmaxf(a / kp, -1.f), 1.f)) - a * h;
        float A_m = km2 * acosf(fminf(fmaxf(b / km, -1.f), 1.f)) - b * h;
        A = A_p + A_m;
        if (!(A > 0.f)) return dm;
        sdist = d * A_m / A;
    }
    float er = A / (PI_F * km2);
    if (EXACT && on_limb) er = er3(xp, yp, kp, xm, ym, km);
    if (!(er > 0.f)) return dm;
    float dmn = dm * (1.f - er);
    if (er < 1.f && zm < 1.f) {
        float wt = fminf(1.f, (1.f - zm) / km);
        float mu = sqrtf(1.f - zm2);
        float e_dot_m = ((xm - xp) * xm + (ym - yp) * ym) / d;
        float grad = -(u1 + 2.f * u2 * (1.f - mu)) / mu * (sdist - d) * e_dot_m;
        dmn -= wt * A * grad * inv_f0;
    }
    return dmn > 0.f ? dmn : 0.f;
}


// ------------------------------------------------------------------------------------------- one cadence
struct CallP {
    float B, t0s, dt0, dP, ph0, ipm_hi, ipm_lo, w2pi, cO, sO, sc, cc, bb, ap, am, kp, km, u1, u2, reach, rs2, inv_f0;
    float e, ax, bx, ay, by, sq;     // eccentric moon (e > 0): direction (ax X + bx Y, ay X + by Y), X = cos E - e,
                                     // Y = sq sin E (ellipse._orbit_axes)
    bool moon_on, ecl_on;
};
static_assert(sizeof(CallP) <= 128, "CallP must fit the 32-float slot of limb_fix");
struct Data {
    const float* __restrict__ DT; const int* __restrict__ EP; const float* __restrict__ EDT;
    const float* __restrict__ EN; const float* __restrict__ EHI; const float* __restrict__ ELO;
};
#ifndef QCAP
#define QCAP 64
#endif

// (cos E, sin E) of the eccentric anomaly at the phase f in [-1/2, 1/2) turns from tau (sin E signed by the half
// orbit), 0 < e < 1: E = M + 0.85 e and four Halley steps in FP32 (ellipse._kepler_cs_fast; GPU_MODEL.md 2.1)
__device__ __forceinline__ void kepler_cs(float f, float e, float& cE, float& sE) {
    float M = TWO_PI_F * fabsf(f);
    float E = M + 0.85f * e;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
        float s, c;
        sincosf(E, &s, &c);
        float g = E - e * s - M;
        float fp = 1.f - e * c;
        E -= 2.f * g * fp / (2.f * fp * fp - g * e * s);
    }
    sincosf(E, &sE, &cE);
    if (f < 0.f) sE = -sE;
}

// mean deficit of cadence c over its SS sub-exposures. EXACT = false: a sample with a limb eclipse sets defer and
// returns (the cadence is redone with EXACT = true); EXACT = true: the three-circle area there (er3).
template <bool EXACT, bool ECC>
__device__ __forceinline__ float cadence_deficit(int c, const CallP& P, const Data& D, const Body& Pb,
                                                 const Body& Mb, bool& defer) {
    int e = __ldg(&D.EP[c]);
    // planet: x = B (EDT_e - dt0 - dP n_e + dt) - t0s   (GPU_MODEL.md 1.2)
    float base = __ldg(&D.EDT[e]) - P.dt0 - P.dP * __ldg(&D.EN[e]);
    // moon phase at TE_e: ph0 + frac((TE_e - TE_0) / per_moon), the product as a two-float (GPU_MODEL.md 1.3)
    float eh = __ldg(&D.EHI[e]), el = __ldg(&D.ELO[e]);
    float p = eh * P.ipm_hi;
    float perr = fmaf(eh, P.ipm_hi, -p) + fmaf(eh, P.ipm_lo, el * P.ipm_hi);
    float phe = (p - floorf(p)) + perr + P.ph0;
    float dtc = __ldg(&D.DT[c * SS + SS / 2]);
    float dsum = 0.f;
    if (!(fabsf(P.B * (base + dtc) - P.t0s) < P.reach)) return 0.f;
    float ph = fmaf(dtc, P.ipm_hi, phe);
    ph -= floorf(ph + 0.5f);
    float sA, cA;
    sincospif(ph + ph, &sA, &cA);
#pragma unroll
    for (int j = 0; j < SS; ++j) {
        float dt = __ldg(&D.DT[c * SS + j]);
        float x = P.B * (base + dt) - P.t0s;
        float dl = (dt - dtc) * P.w2pi, d2 = dl * dl, sQ, cQ;
        float vx, vy;
        if (ECC) {               // eccentric moon: Kepler's equation per sample (a separate instantiation)
            float p3 = fmaf(dt, P.ipm_hi, phe);
            p3 -= floorf(p3 + 0.5f);
            float cE, sE;
            kepler_cs(p3, P.e, cE, sE);
            float X = cE - P.e, Y = P.sq * sE;
            vx = P.ax * X + P.bx * Y;
            vy = P.ay * X + P.by * Y;
        } else {
        if (d2 < 0.25f) {        // rotation from the cadence anchor (|dl| < 0.5 rad), short series
            float sd = dl * (1.f + d2 * (-1.f / 6.f + d2 * (1.f / 120.f + d2 * (-1.f / 5040.f + d2 * (1.f / 362880.f)))));
            float cd = 1.f + d2 * (-0.5f + d2 * (1.f / 24.f + d2 * (-1.f / 720.f + d2 * (1.f / 40320.f))));
            sQ = fmaf(sA, cd, cA * sd);
            cQ = fmaf(cA, cd, -sA * sd);
        } else {
            float p2 = fmaf(dt, P.ipm_hi, phe);
            p2 -= floorf(p2 + 0.5f);
            sincospif(p2 + p2, &sQ, &cQ);
        }
        vx = P.cO * cQ - P.sc * sQ;
        vy = P.sO * cQ + P.cc * sQ;
        }
        float xp = x - vx * P.ap, yp = P.bb - vy * P.ap;
        float zp2 = fmaf(xp, xp, yp * yp);
        float dp = deficit(zp2, Pb), dm = 0.f;
        if (P.moon_on) {
            float xm = x + vx * P.am, ym = P.bb + vy * P.am;
            float zm2 = fmaf(xm, xm, ym * ym);
            dm = deficit(zm2, Mb);
            if (P.ecl_on && dm > 0.f) {
                float ddx = xm - xp, ddy = ym - yp, dd2 = fmaf(ddx, ddx, ddy * ddy);
                if (dd2 < P.rs2) {
                    bool on_limb = fabsf(1.f - sqrtf(zm2)) < P.km || fabsf(1.f - sqrtf(zp2)) < P.kp;
                    if (!EXACT) defer |= on_limb;    // finished with the two-circle value, then discarded
                    dm = eclipse_dm<EXACT>(dm, xp, yp, xm, ym, dd2, P.kp, P.km, zm2, zp2, P.u1, P.u2, P.inv_f0,
                                           on_limb);
                }
            }
        }
        dsum += dp + dm;
    }
    return dsum * (1.f / SS);
}

// ------------------------------------------------------------------------------------------- the likelihood
// TH: (n, 20) FP64 Pandora parameters (the order of pandora_loglike).
// Data (per light curve; GPU_MODEL.md section 1):
//   DT  [nc * SS]  FP32  sample time minus its epoch's reference time TE_e
//   EP  [nc]       int   epoch index of each cadence
//   EDT [ne]       FP32  TE_e - t0_ref - epoch_distance * n_e   (n_e the epoch number)
//   EN  [ne]       FP32  n_e
//   EHI, ELO [ne]  FP32  TE_e - TE_0 as a two-float (hi + lo)
//   IV, E [nc]     FP32  inv_var = 1 / yerr^2 (0 for invalid data), E = 1 - data
// Scalars: t0_ref, TE_0, epoch_distance (FP64), hill, hspan (max |dt - dt_anchor| within a cadence).
// chi^2 = sum over ALL cadences of IV (d - E)^2, d the model deficit (GPU_MODEL.md section 5: summed directly, not
// as the CPU's difference to the flat model, whose large in-transit terms cost FP32 precision; Neumaier-compensated
// per lane, FP64 across the warp).
struct Scal { double t0_ref, te0, ed; float hill, hspan; };

// per-call setup: validity, CallP, the interpolant coefficients into the warp's shared rows (lane j = coefficient j)
__device__ __forceinline__ bool call_setup(const double* __restrict__ th, const Scal& S, int lane,
                                           const float* __restrict__ LT, const float* __restrict__ IT,
                                           float (*sh)[ROW], CallP& P, Body& Pb, Body& Mb) {
    // validity (params_valid): FP64 comparisons; NaN / inf via FP32 (finite FP64 values beyond the FP32 range are
    // physically meaningless here)
    double u1d = th[0], u2d = th[1];
    bool valid = th[2] > 0 && th[3] > 0 && th[4] > 0 && th[11] > 0 && th[13] > 0 && th[5] >= 0 && th[12] >= 0
                 && th[19] >= 0 && u1d >= 0 && u1d + u2d <= 1 + 1e-12 && u1d + 2 * u2d >= -1e-12
                 && th[17] >= 0 && th[17] < 1 && th[8] >= 0 && th[8] < 1;
    int nonfinite = 0;                          // exponent bits all set: inf or NaN (integer test, no FP64 work)
#pragma unroll
    for (int i = 0; i < 20; ++i) nonfinite |= ((__double2hiint(th[i]) & 0x7ff00000) == 0x7ff00000);
    valid = valid && !nonfinite;
    // FP64: the reference offsets (GPU_MODEL.md 1.2, 1.3)
    double per_b = th[3], per_m = th[13];
    P.dt0 = (float)(th[9] - S.t0_ref);          // t0_bary - t0_ref
    P.dP = (float)(per_b - S.ed);               // per_bary - epoch_distance
    double ipmd = 1.0 / per_m;
    double ph0d = S.te0 * ipmd - th[14];        // moon phase at TE_0 [turns]
    P.ph0 = (float)(ph0d - floor(ph0d));
    P.ipm_hi = (float)ipmd;
    P.ipm_lo = (float)(ipmd - (double)P.ipm_hi);
    // FP32
    float u1 = (float)u1d, u2 = (float)u2d, R_star = (float)th[2], a_b = (float)th[4], kp = (float)th[5];
    float t0_off = (float)th[10], M_p = (float)th[11], km = (float)th[12], pm = (float)per_m;
    float Om = (float)th[15] * (PI_F / 180.f), inc = (float)th[16] * (PI_F / 180.f), M_m = (float)th[19];
    float pb = (float)per_b;
    // _moon_setup: a_moon [R_star]; Hill radius (G M_p P^2 / (12 pi^2))^(1/3) / R_star (no stellar mass: FP32 range)
    float a_m = cbrtf(GDAY * (M_p + M_m) * pm * pm) / R_star;
    float r_hill = cbrtf(GDAY * M_p * pb * pb * (1.f / 3.f)) / R_star;
    float em = (float)th[17];
    bool unphys = (a_m * (1.f + em) / r_hill > S.hill) || (a_m * (1.f - em) < kp + km);
    float q = M_m / M_p;
    float a_pl = a_m * q / (1.f + q);
    P.ap = unphys ? 0.f : a_pl;
    P.am = a_m - a_pl;
    float tdur = pb / PI_F * asinf(1.f / a_b);
    P.B = 2.f / tdur;
    P.t0s = t0_off / (0.5f * tdur);
    float ci = cosf(inc);
    sincosf(Om, &P.sO, &P.cO);
    P.sc = P.sO * ci;
    P.cc = P.cO * ci;
    P.e = em;
    if (em > 0.f) {                 // ellipse._orbit_axes (w in the orbit plane, then Omega, i); e = 0 ignores w
        float sw, cw;
        sincosf((float)th[18] * (PI_F / 180.f), &sw, &cw);
        P.ax = P.cO * cw - P.sO * ci * sw;
        P.bx = -P.cO * sw - P.sO * ci * cw;
        P.ay = P.sO * cw + P.cO * ci * sw;
        P.by = -P.sO * sw + P.cO * ci * cw;
        P.sq = sqrtf((1.f - em) * (1.f + em));
    } else {
        P.ax = P.bx = P.ay = P.by = 0.f;
        P.sq = 1.f;
    }
    P.bb = (float)th[6];
    P.kp = kp; P.km = km; P.u1 = u1; P.u2 = u2;
    P.moon_on = !unphys && km > 0.f;
    // eclipse_impossible_circular with a relative margin 1e-6 (never claims "impossible" where the CPU could eclipse)
    P.ecl_on = P.moon_on && !(fabsf(a_m * (1.f - em) * ci) * (1.f - 1e-6f) > kp + km);
    P.w2pi = TWO_PI_F * P.ipm_hi;
    P.reach = 1.f + fmaxf(kp, km) + (P.am + P.ap) * (1.f + em) + P.B * S.hspan;
    P.rs2 = (kp + km) * (kp + km);
    P.inv_f0 = 1.f / (PI_F * (1.f - u1 * (1.f / 3.f) - u2 * (1.f / 6.f)));
    float om = 1.f / (1.f - u1 * (1.f / 3.f) - u2 * (1.f / 6.f));
    float w0 = (1.f - u1 - 2.f * u2) * om, w1 = (u1 + 2.f * u2) * om, w2 = u2 * om;
    float kmc = fmaxf(km, 1e-4f);
    float vpl, vml;
    cheb_limb2(lane, kp, kmc, w0, w1, w2, LT, vpl, vml);
    sh[1][lane] = vpl;
    sh[3][lane] = vml;
    if (lane < 16) sh[0][lane] = cheb_int(lane, kp, u1, u2, IT);
    else sh[2][lane - 16] = cheb_int(lane - 16, kmc, u1, u2, IT);
    __syncwarp();
    body_init(Pb, kp, sh[0], sh[1]);
    body_init(Mb, kmc, sh[2], sh[3]);
    return valid;
}

// kernel 1's cadence loop (lane l: cadences l, l + 32, ...): chi^2 of the cadences without a limb eclipse
// (Neumaier-compensated), the others queued for limb_fix
template <bool ECC>
__device__ __forceinline__ void main_loop(int lane, int nc, int call, bool valid, const CallP& P, const Data& D,
                                          const Body& Pb, const Body& Mb, const float* __restrict__ IV,
                                          const float* __restrict__ E, int2* __restrict__ Q, int* __restrict__ QN,
                                          int qcap, float& chi, float& comp, int& nlimb, bool& overflow) {
    for (int c = lane; c < nc; c += 32) {
        bool defer = false;
        float d = cadence_deficit<false, ECC>(c, P, D, Pb, Mb, defer);
        if (defer) {
            ++nlimb;
            if (valid) {
                int iq = atomicAdd(QN, 1);
                if (iq < qcap) Q[iq] = make_int2(call, c);
                else overflow = true;
            }
            continue;
        }
        float r = d - __ldg(&E[c]);
        float t = __ldg(&IV[c]) * r * r;
        float s2 = chi + t;                                   // Neumaier summation
        comp += (fabsf(chi) >= fabsf(t)) ? (chi - s2) + t : (t - s2) + chi;
        chi = s2;
    }
}

// Kernel 1 (the hot path): one warp per call. Cadences with a limb eclipse are left out of the chi^2 and queued
// as (call, cadence) in Q; limb_fix adds them (GPU_LOG Round 3: the three-circle code in this kernel slowed every
// call by ~30 %, executed or not).
extern "C" __global__ void __launch_bounds__(32 * WARPS, MINB)
loglike(const double* __restrict__ TH, int n,
        const float* __restrict__ DT, const int* __restrict__ EP, const float* __restrict__ EDT,
        const float* __restrict__ EN, const float* __restrict__ EHI, const float* __restrict__ ELO,
        const float* __restrict__ IV, const float* __restrict__ E, int nc,
        double t0_ref, double te0, double ed, float hill, float hspan,
        const float* __restrict__ LT, const float* __restrict__ IT,
        double* __restrict__ OUT, int* __restrict__ NLIMB, int2* __restrict__ Q, int* __restrict__ QN, int qcap,
        CallP* __restrict__ PCG, float* __restrict__ COEF, const int* __restrict__ ACT) {
    __shared__ __align__(16) float sh[WARPS][4][ROW];
    int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int call = blockIdx.x * WARPS + w;
    if (call >= n) return;
    if (ACT && !ACT[call]) return;                 // the sampler's idle chains
    Scal S; S.t0_ref = t0_ref; S.te0 = te0; S.ed = ed; S.hill = hill; S.hspan = hspan;
    CallP P; Body Pb, Mb;
    bool valid = call_setup(TH + (size_t)call * 20, S, lane, LT, IT, sh[w], P, Pb, Mb);
    // the call's set-up and coefficients for limb_fix, stored BEFORE the loop (stored after it, all CallP fields
    // stayed live through the loop and spilled: GPU_LOG Round 7); ~3.5 us of stores per 2048 calls
    {
        float* co = COEF + (size_t)call * 4 * ROW;
#pragma unroll
        for (int r = 0; r < 4; ++r) co[r * ROW + lane] = sh[w][r][lane];
        if (lane == 0) PCG[call] = P;
    }
    Data D;
    D.DT = DT; D.EP = EP; D.EDT = EDT; D.EN = EN; D.EHI = EHI; D.ELO = ELO;
    float chi = 0.f, comp = 0.f;
    int nlimb = 0;
    bool overflow = false;
    if (ECC_ON && P.e > 0.f)                       // uniform per call (warp): two loops, the circular one lean
        main_loop<true>(lane, nc, call, valid, P, D, Pb, Mb, IV, E, Q, QN, qcap, chi, comp, nlimb, overflow);
    else
        main_loop<false>(lane, nc, call, valid, P, D, Pb, Mb, IV, E, Q, QN, qcap, chi, comp, nlimb, overflow);
    double v = (double)chi + (double)comp;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
        nlimb += __shfl_down_sync(0xffffffffu, nlimb, off);
    }
    overflow = __any_sync(0xffffffffu, overflow);
    if (lane == 0) {
        double out = valid ? -0.5 * v : -__longlong_as_double(0x7ff0000000000000LL);
        OUT[call] = overflow ? __longlong_as_double(0x7ff8000000000000LL) : out;   // NaN: queue overflow
        if (NLIMB) NLIMB[call] = nlimb;
    }
}

// Kernel 2: the queued limb-eclipse cadences, one thread per entry (grid-stride), with the call's setup and
// coefficients as kernel 1 stored them (PCG, COEF); adds -chi^2/2 of each cadence to OUT. The last block resets the
// queue counter QN (and DONE) for the next batch.
extern "C" __global__ void limb_fix(const float* __restrict__ DT, const int* __restrict__ EP,
                                    const float* __restrict__ EDT, const float* __restrict__ EN,
                                    const float* __restrict__ EHI, const float* __restrict__ ELO,
                                    const float* __restrict__ IV, const float* __restrict__ E,
                                    double* __restrict__ OUT, const int2* __restrict__ Q, int* __restrict__ QN,
                                    int qcap, const CallP* __restrict__ PCG, const float* __restrict__ COEF,
                                    unsigned int* __restrict__ DONE) {
    int nq = min(*((volatile int*)QN), qcap);
    Data D;
    D.DT = DT; D.EP = EP; D.EDT = EDT; D.EN = EN; D.EHI = EHI; D.ELO = ELO;
    for (int iq = blockIdx.x * blockDim.x + threadIdx.x; iq < nq; iq += gridDim.x * blockDim.x) {
        int2 qc = Q[iq];
        CallP P = PCG[qc.x];
        const float* co = COEF + (size_t)qc.x * 4 * ROW;
        Body Pb, Mb;
        body_init(Pb, P.kp, co, co + ROW);
        body_init(Mb, fmaxf(P.km, 1e-4f), co + 2 * ROW, co + 3 * ROW);
        bool defer = false;
        float d = (ECC_ON && P.e > 0.f) ? cadence_deficit<true, true>(qc.y, P, D, Pb, Mb, defer)
                            : cadence_deficit<true, false>(qc.y, P, D, Pb, Mb, defer);
        float r = d - E[qc.y];
        atomicAdd(&OUT[qc.x], -0.5 * (double)(IV[qc.y] * r * r));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned int t = atomicAdd(DONE, 1u);
        if (t == gridDim.x - 1) { *QN = 0; *DONE = 0u; __threadfence(); }
    }
}

// ------------------------------------------------------------------------------------------- prior transform
// sampling._theta on the device, FP64, one thread per point (GPU_MODEL.md section 6a): unit cube U (n, ndim) ->
// Pandora parameters TH (n, 20) and the log weight LW (n) of the sampled coordinates (LOGZERO = -1e300 outside the
// prior). The prior is packed by pandoramoon.gpu.transform.pack():
//   IS: [nk, ld_mode, mp, pm_k, ld_c0, ld_c1, mp_c0, mp_c1, mp_c2, ncdf, nat_p[nk], nat_c[nk], nat_k[nk]]
//   DS: [base[20], nat_par[nk * 4], ld_par[6], mp_f[6], pm_par[4], CDF[ncdf]]
#define LOGZERO_D (-1e300)
#define INF_D __longlong_as_double(0x7ff0000000000000LL)

__device__ double ndtri_d(double p) {          // sampling._ndtri: Acklam + one Halley step
    if (!(p > 0.0 && p < 1.0)) return p <= 0.0 ? -INF_D : INF_D;
    const double a1 = -3.969683028665376e+01, a2 = 2.209460984245205e+02, a3 = -2.759285104469687e+02;
    const double a4 = 1.383577518672690e+02, a5 = -3.066479806614716e+01, a6 = 2.506628277459239e+00;
    const double b1 = -5.447609879822406e+01, b2 = 1.615858368580409e+02, b3 = -1.556989798598866e+02;
    const double b4 = 6.680131188771972e+01, b5 = -1.328068155288572e+01;
    const double c1 = -7.784894002430293e-03, c2 = -3.223964580411365e-01, c3 = -2.400758277161838e+00;
    const double c4 = -2.549732539343734e+00, c5 = 4.374664141464968e+00, c6 = 2.938163982698783e+00;
    const double d1 = 7.784695709041462e-03, d2 = 3.224671290700398e-01, d3 = 2.445134137142996e+00;
    const double d4 = 3.754408661907416e+00;
    const double plow = 0.02425;
    double x;
    if (p < plow) {
        double q = sqrt(-2.0 * log(p));
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    } else if (p <= 1.0 - plow) {
        double q = p - 0.5, r = q * q;
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    } else {
        double q = sqrt(-2.0 * log(1.0 - p));
        x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    double e = 0.5 * erfc(-x / sqrt(2.0)) - p;
    double u = e * sqrt(2.0 * 3.141592653589793) * exp(0.5 * x * x);
    return x - u / (1.0 + 0.5 * x * u);
}

__device__ __forceinline__ double ppf_d(int kind, const double* par, double u) {
    if (kind == 0) return par[0] + u * (par[1] - par[0]);
    if (kind == 1) return exp(par[0] + u * (par[1] - par[0]));
    return par[0] + par[1] * ndtri_d(par[2] + u * (par[3] - par[2]));
}

__device__ double logpdf_d(int kind, const double* par, double x) {
    double a = par[0], b = par[1], c = par[2], d = par[3];
    if (kind == 0) return (x < a || x > b) ? -INF_D : -log(b - a);
    if (kind == 1) {
        if (x <= 0.0 || log(x) < a || log(x) > b) return -INF_D;
        return -log(x) - log(b - a);
    }
    double z = (x - a) / b;
    double lo = c > 0.0 ? a + b * ndtri_d(c) : -INF_D;
    double hi = d < 1.0 ? a + b * ndtri_d(d) : INF_D;
    if (x < lo || x > hi) return -INF_D;
    return -0.5 * z * z - log(b * sqrt(2.0 * 3.141592653589793)) - log(d - c);
}

extern "C" __global__ void transform(const double* __restrict__ U, int n, int ndim, const int* __restrict__ IS,
                                     const double* __restrict__ DS, double* __restrict__ TH,
                                     double* __restrict__ LW, const int* __restrict__ ACT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ACT && !ACT[i]) return;
    const double* u = U + (size_t)i * ndim;
    double* th = TH + (size_t)i * 20;
    int nk = IS[0], ld_mode = IS[1], mp = IS[2], pm_k = IS[3], ncdf = IS[9];
    const int* nat_p = IS + 10;
    const int* nat_c = nat_p + nk;
    const int* nat_k = nat_c + nk;
    const double* base = DS;
    const double* nat_par = DS + 20;
    const double* ld_par = nat_par + 4 * nk;
    const double* mp_f = ld_par + 6;
    const double* pm_par = mp_f + 6;
    const double* CDF = pm_par + 4;
    for (int j = 0; j < 20; ++j) th[j] = base[j];
    double lw = 0.0;
    for (int j = 0; j < nk; ++j) th[nat_p[j]] = ppf_d(nat_k[j], nat_par + 4 * j, u[nat_c[j]]);
    bool ok = true;
    if (ld_mode == 0) {
        double q1 = th[0], q2 = th[1];
        if (!(q1 >= 0.0 && q1 <= 1.0 && q2 >= 0.0 && q2 <= 1.0)) ok = false;
        else {
            double sq = sqrt(q1);
            th[0] = 2.0 * sq * q2;
            th[1] = sq * (1.0 - 2.0 * q2);
        }
    } else if (ld_mode == 1 || ld_mode == 2) {
        if (ld_mode == 2) {
            double z1 = ndtri_d(u[IS[4]]), z2 = ndtri_d(u[IS[5]]);
            th[0] = ld_par[0] + ld_par[2] * z1;
            th[1] = ld_par[1] + ld_par[3] * z1 + ld_par[4] * z2;
        }
        if (!(th[0] >= 0.0 && th[0] + th[1] <= 1.0 && th[0] + 2.0 * th[1] >= 0.0)) ok = false;
        lw += ld_par[5];
    }
    if (ok && mp == 1) {
        // _to_native: phase-at-transit coordinates -> (per_moon, tau, log w)
        double phi_c = u[IS[6]], delta = u[IS[7]], u_m = u[IS[8]];
        int lo = 0, hi = ncdf - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (CDF[mid] > u_m) hi = mid; else lo = mid + 1;
        }
        double m = mp_f[0] + lo;
        double N = m + delta;
        double per_bary = th[3];
        double per_moon = per_bary / N;
        double t_c = th[9] + th[10] + mp_f[3] * per_bary;
        double x = t_c * N / per_bary - phi_c;
        double tau = x - floor(x);
        double logw = per_moon > mp_f[2] ? LOGZERO_D : log(per_bary / mp_f[2] * mp_f[1] * m * (m + 1.0) / (N * N));
        if (!(logw > LOGZERO_D) || per_moon < mp_f[4]) ok = false;
        else {
            double lpdf = logpdf_d(pm_k, pm_par, per_moon);
            if (!(lpdf > -INF_D)) ok = false;
            else {
                lw += logw + lpdf + mp_f[5];
                th[13] = per_moon;
                th[14] = tau;
            }
        }
    }
    LW[i] = ok ? lw : LOGZERO_D;
}

// ------------------------------------------------------------------------------------------- component tests
// deficit of one body (k, u1, u2) at squared distances Z2 (one warp)
extern "C" __global__ void t_deficit(float k, float u1, float u2, const float* __restrict__ Z2, int m,
                                     const float* __restrict__ LT, const float* __restrict__ IT,
                                     float* __restrict__ OUT) {
    __shared__ __align__(16) float sh[2][ROW];
    int lane = threadIdx.x & 31;
    float om = 1.f / (1.f - u1 * (1.f / 3.f) - u2 * (1.f / 6.f));
    float va, vb;
    cheb_limb2(lane, k, k, (1.f - u1 - 2.f * u2) * om, (u1 + 2.f * u2) * om, u2 * om, LT, va, vb);
    sh[1][lane] = va;
    if (lane < 16) sh[0][lane] = cheb_int(lane, k, u1, u2, IT);
    __syncwarp();
    Body b;
    body_init(b, k, sh[0], sh[1]);
    for (int s = lane; s < m; s += 32) OUT[s] = deficit(Z2[s], b);
}

// er3 at configurations X[i] = (xp, yp, rp, xm, ym, rm)
extern "C" __global__ void t_er3(const float* __restrict__ X, int m, float* __restrict__ OUT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const float* x = X + 6 * i;
    OUT[i] = er3(x[0], x[1], x[2], x[3], x[4], x[5]);
}
