// GPU nested slice sampler (GPU_PLAN.md phase 3): the algorithm of pandoramoon.nss.ns() with R independent runs of
// n live points, k slice chains per run and iteration (parallel nested sampling), all state on the device.
//
// Per tick the host launches (no host synchronisation):
//   chain_tick  one thread per chain: consume the last evaluation, advance the slice state machine, write the next
//               proposal PU (ACT = 1) or finish the chain
//   run_tick    one block per run: when all k chains of the run's iteration are done, finish that iteration
//               (insertion ranks, mixing, replacement, scale adaptation) and begin the next one (the k lowest,
//               dead points and weights, stopping test, spread, neighbour lists, start points)
//   evaluate    (separate module) L = LW + LLK at the proposals with ACT = 1
//
// Same arithmetic as nss.ns (FP64 sampler state); different random streams (Philox), so runs agree statistically.

#define MAXEXP 100
#define NMAX 2048              // live points per run selected in shared memory (more: global scratch)
#define DMAX 32                // dimensions
#ifndef DIM
#define DIM DMAX               // the run's dimension at compile time (sampler_module(D)); DMAX = generic
#endif
#define KPMAX 8                // neighbours per point
#define LOGZERO (-1e300)
#define INF_D __longlong_as_double(0x7ff0000000000000LL)
#define INF_F __int_as_float(0x7f800000)

typedef unsigned long long u64;

// ------------------------------------------------------------------------------------------- Philox4x32-10
__device__ __forceinline__ void philox(unsigned int k0, unsigned int k1, u64 ctr, unsigned int& o0, unsigned int& o1) {
    unsigned int c0 = (unsigned int)ctr, c1 = (unsigned int)(ctr >> 32), c2 = 0x9e3779b9u, c3 = 0x7f4a7c15u;
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        unsigned int hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        unsigned int hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        unsigned int n0 = hi1 ^ c1 ^ k0, n2 = hi0 ^ c3 ^ k1;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    o0 = c0; o1 = c1;
}

struct Rng { unsigned int k0, k1; u64 ctr; };

__device__ __forceinline__ double runif(Rng& g) {      // uniform in [0, 1), 53 bits
    unsigned int a, b;
    philox(g.k0, g.k1, g.ctr++, a, b);
    u64 x = ((u64)a << 32) | b;
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ int rint_(Rng& g, int n) {
    int i = (int)(runif(g) * n);
    return i < n ? i : n - 1;
}

// ------------------------------------------------------------------------------------------- state
struct Run {                   // layout mirrored in sampler.py (RUN_DTYPE)
    double logX, logZ, Lmin, scale;
    long long ncall, nexp_tot, nstep_tot, nstuck, nmaxexp;
    int it, nd, nins, nrecl, repl_since, status, nmix, overflow;
    u64 seed;                  // the run's seed: all its random streams derive from it (slot-independent)
    int phase, nsteps;         // nsteps: slice steps per chain (per run: the arms of nss.fit differ)
};
// status: 0 free slot, 1 running, 2 finished, 3 aborted (buffer overflow), 4 aborted by the host;
// phase: 0 begin an iteration, 1 chains running

struct Chain {                 // layout mirrored in sampler.py (CHAIN_DTYPE)
    double XL, left, right, t;
    long long nc, nexp, nstuck, nmaxexp;
    u64 ctr;
    int state, step, e, start;
};
// state: 0 done, 1 start a step, 2 step out left, 3 step out right, 4 shrink

struct Cfg {
    int R, k, n, D, kpair, nsteps, cap;
    double log_frac, recluster;
    unsigned int seed;
    int sd_every;
};

__device__ __forceinline__ bool wrap_inside(double* x, const int* __restrict__ WR, int D) {
    for (int i = 0; i < D; ++i) {
        if (WR[i]) x[i] -= floor(x[i]);
        else if (!(x[i] > 0.0 && x[i] < 1.0)) return false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------- chain_tick
__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// one warp per chain: lane i holds coordinate i of the chain's point x, direction v and proposal xn; the scalar state
// is uniform (all lanes draw the same random numbers and take the same branches)
extern "C" __global__ void chain_tick(int R_, int k_, int n_, int D_, int kpair_, int nsteps_, unsigned int seed_,
                                      Chain* __restrict__ CH, const Run* __restrict__ RS,
                                      double* __restrict__ X, double* __restrict__ V, double* __restrict__ PU,
                                      const double* __restrict__ LW, const double* __restrict__ LLK,
                                      int* __restrict__ ACT, const double* __restrict__ LU,
                                      const unsigned char* __restrict__ ALIVE, const int* __restrict__ NBR,
                                      const int* __restrict__ WR) {
    int lane = threadIdx.x & 31;
    int c = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int k = k_, n = n_, D = D_;
    if (c >= R_ * k) return;
    int r = c / k;
    int st = RS[r].status, ph = RS[r].phase;
    bool had = ACT[c] != 0;
    __syncwarp();
    if (lane == 0) ACT[c] = 0;
    if (st != 1 || ph != 1) return;
    Chain ch = CH[c];
    if (ch.state == 0) return;
    double Lmin = RS[r].Lmin, scale = RS[r].scale;
    int nsteps_r = RS[r].nsteps;
    u64 rseed = RS[r].seed;
    bool act = lane < D;
    size_t o = (size_t)c * D + lane;
    double xi = act ? X[o] : 0.0, vi = act ? V[o] : 0.0, xni = act ? PU[o] : 0.0;
    int wr = act ? WR[lane] : 0;
    const double* lu = LU + (size_t)r * n * D;
    const unsigned char* alive = ALIVE + (size_t)r * n;
    Rng g; g.k0 = (unsigned int)rseed ^ 0x85ebca6bu; g.k1 = (unsigned int)(rseed >> 32) ^ (0xc2b2ae35u * (unsigned int)(c % k + 1));
    g.ctr = ch.ctr;
    double L = 0.0;
    if (had) {
        double lw = LW[c];
        L = lw > LOGZERO ? lw + LLK[c] : LOGZERO;
    }
    bool pending = had, need = false;
    for (int guard = 0; guard < 4 * MAXEXP + 10000; ++guard) {
        if (ch.state == 1) {                                   // a new slice step
            if (ch.step >= nsteps_r) { ch.state = 0; break; }
            int pa;
            do { pa = rint_(g, n); } while (!alive[pa]);
            int pb = NBR[((size_t)r * n + pa) * kpair_ + rint_(g, kpair_)];
            double sg = runif(g) < 0.5 ? 1.0 : -1.0;
            vi = act ? sg * (lu[(size_t)pa * D + lane] - lu[(size_t)pb * D + lane]) : 0.0;
            double vv = warp_sum(vi * vi);
            if (vv == 0.0) { ch.step++; continue; }            // duplicate points: no direction (as nss)
            ch.left = -scale * runif(g);
            ch.right = ch.left + scale;
            ch.e = 0;
            ch.state = 2;
            pending = false;
        }
        if (ch.state == 2 || ch.state == 3) {                   // stepping out
            bool adv = false;
            if (pending) {
                pending = false;
                ch.nc++;
                if (L > Lmin) {
                    if (ch.state == 2) ch.left -= scale; else ch.right += scale;
                    ch.nexp++;
                    if (++ch.e >= MAXEXP) { ch.nmaxexp++; adv = true; }
                } else adv = true;
            }
            if (!adv) {
                double sv = ch.state == 2 ? ch.left : ch.right;
                xni = xi + vi * sv;
                bool ok = true;
                if (act) { if (wr) xni -= floor(xni); else ok = xni > 0.0 && xni < 1.0; }
                if (__all_sync(0xffffffffu, ok)) { need = true; break; }
                adv = true;
            }
            if (adv) {
                if (ch.state == 2) { ch.state = 3; ch.e = 0; } else ch.state = 4;
                continue;
            }
        }
        if (ch.state == 4) {                                    // shrinking
            if (pending) {
                pending = false;
                ch.nc++;
                if (L > Lmin) {                                 // accepted: xn is the new point (wrapped)
                    xi = xni;
                    ch.XL = L;
                    ch.step++;
                    ch.state = 1;
                    continue;
                }
                if (ch.t < 0.0) ch.left = ch.t; else ch.right = ch.t;
                if (ch.right - ch.left < 1e-14) { ch.nstuck++; ch.step++; ch.state = 1; continue; }
            }
            ch.t = ch.left + (ch.right - ch.left) * runif(g);
            xni = xi + vi * ch.t;
            bool ok = true;
            if (act) { if (wr) xni -= floor(xni); else ok = xni > 0.0 && xni < 1.0; }
            if (__all_sync(0xffffffffu, ok)) { need = true; break; }
            if (ch.t < 0.0) ch.left = ch.t; else ch.right = ch.t;
            if (ch.right - ch.left < 1e-14) { ch.nstuck++; ch.step++; ch.state = 1; }
            continue;
        }
        if (ch.state == 0) break;
    }
    ch.ctr = g.ctr;
    if (act) { X[o] = xi; V[o] = vi; PU[o] = xni; }
    if (lane == 0) { CH[c] = ch; ACT[c] = need ? 1 : 0; }
}

// ------------------------------------------------------------------------------------------- run_tick helpers
__device__ __forceinline__ double logaddexp_(double a, double b) {
    if (a == -INF_D) return b;
    if (b == -INF_D) return a;
    double m = fmax(a, b);
    return m + log1p(exp(-fabs(a - b)));
}

// bitonic sort of (key, idx) ascending over m = power of 2 elements in shared memory, whole block
__device__ void bitonic(double* key, int* idx, int m) {
    for (int size = 2; size <= m; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < m; i += blockDim.x) {
                int j = i ^ stride;
                if (j > i) {
                    bool up = (i & size) == 0;
                    if ((key[i] > key[j]) == up) {
                        double tk = key[i]; key[i] = key[j]; key[j] = tk;
                        int ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
                    }
                }
            }
        }
    }
    __syncthreads();
}

// ------------------------------------------------------------------------------------------- run_tick
extern "C" __global__ void __launch_bounds__(512)
run_tick(int R_, int k_, int n_, int D_, int kpair_, int nsteps_, int cap_, double log_frac_, double recluster_,
         unsigned int seed_, int sd_every_, Run* __restrict__ RS, Chain* __restrict__ CH, double* __restrict__ X,
         double* __restrict__ LU, double* __restrict__ LL, unsigned char* __restrict__ ALIVE, int* __restrict__ NBR,
         double* __restrict__ SD, double* __restrict__ MIX, int* __restrict__ SEL, int* __restrict__ STARTS,
         double* __restrict__ DU, double* __restrict__ DL, double* __restrict__ DLW, int* __restrict__ INS,
         const int* __restrict__ WR, long long* __restrict__ PROF, double* __restrict__ GK, int* __restrict__ GI,
         int ng) {
    long long t_last = clock64();
#define SECTION(i) do { __syncthreads(); if (PROF && threadIdx.x == 0) { long long t_ = clock64(); \
        PROF[blockIdx.x * 16 + (i)] += t_ - t_last; t_last = t_; } } while (0)
    Cfg cfg;
    cfg.R = R_; cfg.k = k_; cfg.n = n_; cfg.D = D_; cfg.kpair = kpair_; cfg.nsteps = nsteps_; cfg.cap = cap_;
    cfg.log_frac = log_frac_; cfg.recluster = recluster_; cfg.seed = seed_; cfg.sd_every = sd_every_;
    __shared__ double skey[NMAX];
    __shared__ int sidx[NMAX];
    // the selection buffers: shared memory for n <= NMAX, else the run's row of the global scratch GK / GI (ng >= n)
    double* key = skey;
    int* kix = sidx;
    if (n_ > NMAX) { key = GK + (size_t)blockIdx.x * ng; kix = GI + (size_t)blockIdx.x * ng; }
    __shared__ int s_flag;
    __shared__ long long s_sum[5];
    __shared__ float s_inv[DMAX];
    __shared__ double s_s1[DMAX], s_s2[DMAX];
    __shared__ double s_w[512], s_inc[512], s_pre[513];
    __shared__ Run run;                        // the block's copy of RS[r] (written back at `done`)
    int r = blockIdx.x;
    if (RS[r].status != 1) return;
    if (threadIdx.x == 0) run = RS[r];
    __syncthreads();
    do {
    int k = cfg.k, n = cfg.n, D = cfg.D, tid = threadIdx.x, nt = blockDim.x;
    double* lu = LU + (size_t)r * n * D;
    double* ll = LL + (size_t)r * n;
    unsigned char* alive = ALIVE + (size_t)r * n;
    int* nbr = NBR + (size_t)r * n * cfg.kpair;
    int* sel = SEL + (size_t)r * k;
    int* starts = STARTS + (size_t)r * k;
    Chain* ch = CH + (size_t)r * k;
    double* sd = SD + (size_t)r * D;

    if (run.phase == 1) {
        // all chains of the iteration done?
        if (tid == 0) s_flag = 1;
        __syncthreads();
        for (int j = tid; j < k; j += nt) if (ch[j].state != 0) s_flag = 0;
        __syncthreads();
        if (!s_flag) return;                     // nothing modified yet
        // ---- finish the iteration (nss.ns: ranks, mixing, replacement, counters, scale) ----
        // insertion index: rank of each new logL among the n - k survivors (alive, not yet replaced);
        // shared copies of the live logL and the alive flags, one thread per new point
        if (run.nins + k > cfg.cap) { if (tid == 0) { run.overflow = 1; run.status = 3; } break; }
        for (int q = tid; q < n; q += nt) { key[q] = ll[q]; kix[q] = alive[q]; }
        __syncthreads();
        for (int j = tid; j < k; j += nt) {
            double Lc = ch[j].XL;
            int cnt = 0;
            for (int q = 0; q < n; ++q) cnt += (kix[q] && key[q] < Lc) ? 1 : 0;
            INS[(size_t)r * cfg.cap + run.nins + j] = cnt;
        }
        __syncthreads();
    SECTION(0);
        // mixing diagnostic: (end - start)^2 / (2 var_live) per dimension
        for (int i = tid; i < D; i += nt) s_s1[i] = 0.0;
        __syncthreads();
        for (int t = tid; t < k * D; t += nt) {
            int j = t / D, i = t % D;
            double dlt = X[((size_t)r * k + j) * D + i] - lu[(size_t)starts[j] * D + i];
            if (WR[i]) dlt -= floor(dlt + 0.5);
            atomicAdd(&s_s1[i], dlt * dlt / (2.0 * sd[i] * sd[i]));
        }
        __syncthreads();
        for (int i = tid; i < D; i += nt) MIX[(size_t)r * D + i] += s_s1[i];
        __syncthreads();
    SECTION(1);
        // replacement (the new point inherits its start's neighbour list)
        for (int t = tid; t < k * D; t += nt) {
            int j = t / D, i = t % D;
            lu[(size_t)sel[j] * D + i] = X[((size_t)r * k + j) * D + i];
        }
        for (int t = tid; t < k * cfg.kpair; t += nt) {
            int j = t / cfg.kpair, a = t % cfg.kpair;
            nbr[(size_t)sel[j] * cfg.kpair + a] = nbr[(size_t)starts[j] * cfg.kpair + a];
        }
        if (tid < 5) s_sum[tid] = 0;
        __syncthreads();
        for (int j = tid; j < k; j += nt) {
            ll[sel[j]] = ch[j].XL;
            alive[sel[j]] = 1;
            atomicAdd((unsigned long long*)&s_sum[0], (unsigned long long)ch[j].nc);
            atomicAdd((unsigned long long*)&s_sum[1], (unsigned long long)ch[j].nexp);
            atomicAdd((unsigned long long*)&s_sum[2], (unsigned long long)ch[j].nstuck);
            atomicAdd((unsigned long long*)&s_sum[3], (unsigned long long)ch[j].nmaxexp);
        }
        __syncthreads();
        if (tid == 0) {
            run.nins += k;
            run.nmix += k;
            run.ncall += s_sum[0];
            run.nexp_tot += s_sum[1];
            run.nstuck += s_sum[2];
            run.nmaxexp += s_sum[3];
            run.nstep_tot += (long long)k * run.nsteps;
            // scale adaptation: aim at ~1 expansion per step
            if (s_sum[1] > (long long)k * run.nsteps) run.scale *= 1.1; else run.scale /= 1.1;
            run.scale = fmin(fmax(run.scale, 1e-3), 1e3);
            run.it += 1;
            run.phase = 0;
        }
        __syncthreads();
    }

    SECTION(2);
    // ---- begin an iteration (nss.ns loop head) ----
    // the k lowest live points (all are alive here), ascending
    int m = 1;
    while (m < n) m <<= 1;
    for (int q = tid; q < m; q += nt) {
        key[q] = q < n ? ll[q] : INF_D;
        kix[q] = q;
    }
    bitonic(key, kix, m);
    double Lmax = key[n - 1];
    if (run.nd + k + n > cfg.cap) { if (tid == 0) { run.overflow = 1; run.status = 3; } break; }
    SECTION(3);
    // record the k removed points: dead weights (parallel nested sampling: shrinkage 1 / (n - j))
    for (int t = tid; t < k * D; t += nt) {
        int j = t / D, i = t % D;
        DU[((size_t)r * cfg.cap + run.nd + j) * D + i] = lu[(size_t)kix[j] * D + i];
    }
    __syncthreads();
    SECTION(4);
    // weights in parallel: thread jj has logX_jj = logX - sum_{i < jj} 1 / (n - i) and
    // logw_jj = logX_jj + log(1 - exp(-1 / (n - jj))); logZ += logsumexp_jj(logw_jj + L_jj)
    for (int jj = tid; jj < k; jj += nt) s_inc[jj] = 1.0 / (n - jj);
    __syncthreads();
    if (tid == 0) { double S = 0.0; for (int jj = 0; jj < k; ++jj) { s_pre[jj] = S; S += s_inc[jj]; } s_pre[k] = S; }
    __syncthreads();
    for (int jj = tid; jj < k; jj += nt) {
        int q = kix[jj];
        double lwt = run.logX - s_pre[jj] + log(-expm1(-s_inc[jj]));
        DL[(size_t)r * cfg.cap + run.nd + jj] = ll[q];
        DLW[(size_t)r * cfg.cap + run.nd + jj] = lwt;
        s_w[jj] = lwt + ll[q];
        sel[jj] = q;
    }
    __syncthreads();
    if (tid == 0) {
        double mx = -INF_D;
        for (int jj = 0; jj < k; ++jj) mx = fmax(mx, s_w[jj]);
        double sm = 0.0;
        for (int jj = 0; jj < k; ++jj) sm += exp(s_w[jj] - mx);
        double logZ = logaddexp_(run.logZ, mx + log(sm));
        double logX = run.logX - s_pre[k];
        run.nd += k;
        run.logX = logX;
        run.logZ = logZ;
        run.Lmin = ll[kix[k - 1]];
        s_flag = (isfinite(logZ) && logX + Lmax - logZ < cfg.log_frac) ? 1 : 0;
    }
    __syncthreads();
    SECTION(5);
    if (s_flag) {
        // stop: all n live points (the k just recorded included, as nss.ns) with weight X / n
        double lwl = run.logX - log((double)n);
        for (int t = tid; t < n * D; t += nt)
            DU[((size_t)r * cfg.cap + run.nd) * D + t] = lu[t];
        for (int q = tid; q < n; q += nt) {
            DL[(size_t)r * cfg.cap + run.nd + q] = ll[q];
            DLW[(size_t)r * cfg.cap + run.nd + q] = lwl;
        }
        __syncthreads();
        if (tid == 0) { run.nd += n; run.status = 2; }
        break;
    }
    SECTION(6);
    // spread of the live points every sd_every iterations: sums of x and x^2 over all threads (shared atomics)
    if (run.it % cfg.sd_every == 0) {
        for (int i = tid; i < D; i += nt) { s_s1[i] = 0.0; s_s2[i] = 0.0; }
        __syncthreads();
        {   // thread tid: dimension tid % D, points tid / D, tid / D + nt / D, ...; one atomic pair per thread
            int per = nt / D;
            if (tid < per * D) {
                int i = tid % D;
                double a1 = 0.0, a2 = 0.0;
                for (int a = tid / D; a < n; a += per) { double xv = lu[(size_t)a * D + i]; a1 += xv; a2 += xv * xv; }
                atomicAdd(&s_s1[i], a1);
                atomicAdd(&s_s2[i], a2);
            }
        }
        __syncthreads();
        for (int i = tid; i < D; i += nt) {
            double mu = s_s1[i] / n;
            sd[i] = sqrt(fmax(s_s2[i] / n - mu * mu, 0.0)) + 1e-300;
        }
    }
    __syncthreads();
    for (int j = tid; j < k; j += nt) alive[sel[j]] = 0;
    __syncthreads();
    SECTION(7);
    // neighbour lists (dirs = "pairs"): the kpair nearest alive points in the metric scaled by sd, FP32 distances;
    // reference set = all alive points (n <= nbsub = 2000 in every NestedSampler setting)
    if (run.repl_since >= cfg.recluster * n) {
        for (int i = tid; i < D; i += nt) s_inv[i] = (float)(1.0 / sd[i]);
        __syncthreads();
        float* tile = reinterpret_cast<float*>(skey);          // TILE x D floats (the sort buffer is free here)
        int* tidx = sidx;                                        // TILE flags
        const int TILE = (NMAX * 2) / DMAX;                     // 128 points of up to 32 coordinates
        for (int qb = 0; qb < n; qb += nt) {
            int q = qb + tid;
            bool mine = q < n && alive[q];
            float qc[DIM];
#pragma unroll
            for (int i = 0; i < DIM; ++i) qc[i] = (mine && i < D) ? (float)lu[(size_t)q * D + i] * s_inv[i] : 0.f;
            float bd[KPMAX];
            int bi[KPMAX];
            int mm = 0;
            float worst = INF_F;
            for (int t0 = 0; t0 < n; t0 += TILE) {
                int tn = min(TILE, n - t0);
                __syncthreads();
                for (int t = tid; t < tn * D; t += nt) {
                    int a = t / D, i = t % D;
                    tile[a * DMAX + i] = (float)lu[(size_t)(t0 + a) * D + i] * s_inv[i];
                }
                for (int a = tid; a < tn; a += nt) tidx[a] = alive[t0 + a];
                __syncthreads();
                if (mine) {
                    for (int a = 0; a < tn; ++a) {
                        if (t0 + a == q || !tidx[a]) continue;
                        float sd2 = 0.f;
#pragma unroll
                        for (int i = 0; i < DIM; ++i) {
                            float dd = (i < D) ? tile[a * DMAX + i] - qc[i] : 0.f;
                            sd2 = fmaf(dd, dd, sd2);
                        }
                        if (sd2 >= worst) continue;
                        int p = mm < cfg.kpair ? mm++ : cfg.kpair - 1;
                        while (p > 0 && bd[p - 1] > sd2) { bd[p] = bd[p - 1]; bi[p] = bi[p - 1]; --p; }
                        bd[p] = sd2; bi[p] = t0 + a;
                        if (mm == cfg.kpair) worst = bd[cfg.kpair - 1];
                    }
                }
            }
            if (mine) for (int t = 0; t < cfg.kpair; ++t) nbr[(size_t)q * cfg.kpair + t] = t < mm ? bi[t] : q;
        }
        __syncthreads();
        if (tid == 0) { run.nrecl += 1; run.repl_since = 0; }
    }
    __syncthreads();
    SECTION(8);
    // start points (random alive) and the chains
    for (int j = tid; j < k; j += nt) {
        Rng g; g.k0 = (unsigned int)run.seed ^ 0x27d4eb2fu ^ (0x9E3779B9u * (unsigned int)(j + 1));
        g.k1 = (unsigned int)(run.seed >> 32) ^ 0x165667b1u; g.ctr = ((u64)run.it << 20);
        int q;
        do { q = rint_(g, n); } while (!alive[q]);
        starts[j] = q;
    }
    if (tid == 0) run.repl_since += k;
    __syncthreads();
    for (int t = tid; t < k * D; t += nt) {
        int j = t / D, i = t % D;
        X[((size_t)r * k + j) * D + i] = lu[(size_t)starts[j] * D + i];
    }
    for (int j = tid; j < k; j += nt) {
        Chain c0 = ch[j];
        c0.XL = ll[starts[j]];
        c0.nc = c0.nexp = c0.nstuck = c0.nmaxexp = 0;
        c0.state = 1;
        c0.step = 0;
        c0.start = starts[j];
        ch[j] = c0;
    }
    __syncthreads();
    SECTION(9);
    if (tid == 0) run.phase = 1;
    } while (0);
    __syncthreads();
    if (threadIdx.x == 0) RS[r] = run;
}

// ------------------------------------------------------------------------------------------- synthetic targets
// Gaussian at 0.5 with sd sigma in every dimension (unit cube), normalised: lnZ = sum ln(mass inside (0, 1)) ~ 0
extern "C" __global__ void t_gauss(const double* __restrict__ PU, const int* __restrict__ ACT, int nc, int D,
                                   double sigma, double* __restrict__ LW, double* __restrict__ LLK) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc || (ACT && !ACT[c])) return;
    double s = 0.0;
    for (int i = 0; i < D; ++i) { double z = (PU[(size_t)c * D + i] - 0.5) / sigma; s += z * z; }
    LW[c] = 0.0;
    LLK[c] = -0.5 * s - D * (log(sigma) + 0.5 * log(2.0 * 3.141592653589793));
}

// two Gaussians (weights 1/2) at 0.3 and 0.7 in every dimension: lnZ ~ 0
extern "C" __global__ void t_twomode(const double* __restrict__ PU, const int* __restrict__ ACT, int nc, int D,
                                     double sigma, double* __restrict__ LW, double* __restrict__ LLK) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc || (ACT && !ACT[c])) return;
    double s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < D; ++i) {
        double z1 = (PU[(size_t)c * D + i] - 0.3) / sigma, z2 = (PU[(size_t)c * D + i] - 0.7) / sigma;
        s1 += z1 * z1; s2 += z2 * z2;
    }
    double norm = -D * (log(sigma) + 0.5 * log(2.0 * 3.141592653589793)) - log(2.0);
    LW[c] = 0.0;
    LLK[c] = norm + logaddexp_(-0.5 * s1, -0.5 * s2);
}
