// cudasynth: exact Clifford+T synthesis by level-synchronous branch-and-bound (docs/PLAN.md).
//
// A frontier node is a gate prefix p (indices into the spec's candidate list). Its residual S = U p^-1 is what
// the rest of the circuit must implement; it is recomputed from p (cheap next to the bounds) instead of stored,
// so a node is ~64 bytes. Per level: expand every node by every gate allowed by the W2/W3 window rules, update S,
// report S == I (a circuit), prune when lower_bound(S) > depth - t, then keep one node per distinct S (the
// lexicographically smallest prefix, which keeps the lex-least optimal circuit and so stays sound with W2/W3).
// An empty frontier proves that no circuit of <= depth gates exists.
//
// Exact arithmetic: Z[w] entries (a + b w + c w^2 + d w^3, w = e^{i pi/4}) with one sqrt2 exponent per matrix.
// Same source for both backends: nvcc -> CUDA; g++ -x c++ -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
// -> CPU (development and tests without an NVIDIA GPU).
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif
#define HD __host__ __device__

#ifndef NQ_MIN
#define NQ_MIN 1
#endif
#ifndef NQ_MAX
#define NQ_MAX 4
#endif
constexpr int DMAX = 48;          // longest prefix
constexpr int GMAX = 64;          // most candidates
constexpr int LMAX = 8;           // largest local gate dimension (3-qubit gates)
constexpr int CUTMAX = 16;
constexpr uint64_t PRIME = 998244353ULL;   // p = 1 mod 8: Z[w] -> F_p for exact-or-lower ranks

// ---------------- Z[w] ----------------
struct Zw { int64_t a, b, c, d; };
HD inline Zw zadd(Zw x, Zw y) { return {x.a + y.a, x.b + y.b, x.c + y.c, x.d + y.d}; }
HD inline Zw zmul(Zw x, Zw y) {
  return {x.a * y.a - x.b * y.d - x.c * y.c - x.d * y.b, x.a * y.b + x.b * y.a - x.c * y.d - x.d * y.c,
          x.a * y.c + x.b * y.b + x.c * y.a - x.d * y.d, x.a * y.d + x.b * y.c + x.c * y.b + x.d * y.a};
}
HD inline Zw zconj(Zw x) { return {x.a, -x.d, -x.c, -x.b}; }
HD inline bool zzero(Zw x) { return !(x.a | x.b | x.c | x.d); }
HD inline bool zeq(Zw x, Zw y) { return x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d; }
HD inline bool zdiv2ok(Zw x) { return !((x.a - x.c) & 1) && !((x.b - x.d) & 1); }      // x / sqrt2 in Z[w]
HD inline Zw zdown(Zw x) { return {(x.b - x.d) / 2, (x.a + x.c) / 2, (x.b + x.d) / 2, (x.c - x.a) / 2}; }
HD inline Zw zwpow(int p) {        // w^p
  p &= 7;
  Zw r{0, 0, 0, 0};
  int64_t s = p < 4 ? 1 : -1;
  switch (p & 3) { case 0: r.a = s; break; case 1: r.b = s; break; case 2: r.c = s; break; default: r.d = s; }
  return r;
}
HD inline int zval(Zw x) {         // sqrt2-adic valuation, x != 0
  int v = 0;
  while (zdiv2ok(x)) { x = zdown(x); v++; if (v > 200) break; }
  return v;
}

// Storage of residual matrices: int32 (default) or int64 (-DSTORE64); arithmetic in 64 bits, narrowed with an
// overflow check. Coefficients of sqrt2^k U grow like 2^(k/2) (each H raises k by at most 1), and a surviving
// prefix has k <= its remaining budget <= DMAX, so int32 holds every surviving residual; a target or child above
// that is caught by the check (the run aborts) - build with -DSTORE64 for such inputs.
#ifdef STORE64
using StoreT = int64_t;
HD inline bool fits(int64_t) { return true; }
#else
using StoreT = int32_t;
HD inline bool fits(int64_t v) { return v >= INT32_MIN && v <= INT32_MAX; }
#endif
struct Zw32 { StoreT a, b, c, d; };
HD inline Zw ld(Zw32 x) { return {x.a, x.b, x.c, x.d}; }
HD inline Zw32 st(Zw x, int &ovf) {
  if (!fits(x.a) || !fits(x.b) || !fits(x.c) || !fits(x.d)) ovf = 1;
  return {(StoreT)x.a, (StoreT)x.b, (StoreT)x.c, (StoreT)x.d};
}
HD inline bool zzero32(Zw32 x) { return !(x.a | x.b | x.c | x.d); }
HD inline bool zeq32(Zw32 x, Zw32 y) { return x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d; }
HD inline bool zdiv2ok32(Zw32 x) { return !((x.a - x.c) & 1) && !((x.b - x.d) & 1); }
HD inline Zw32 zdown32(Zw32 x) { return {(x.b - x.d) / 2, (x.a + x.c) / 2, (x.b + x.d) / 2, (x.c - x.a) / 2}; }

// ---------------- spec (device-readable) ----------------
struct Gate {
  int nq, q[3], ik;                // inverse gate: local matrix / sqrt2^ik
  Zw inv[LMAX * LMAX];
};
struct Spec {
  int n, N, G, depth, standard, ncuts, rad[CUTMAX], cutmask[CUTMAX], cutsize[CUTMAX], have_table;
  uint64_t w8;                     // w mod PRIME
  const Gate *gates;
  const uint8_t *bad2;             // G x G
  const uint32_t *bad3;            // bitset G^3
  const Zw *target;                // start matrix (unitary: the target U; states: input rows), exponent tk
  int tk;
  const int *cxtable;
  int lite, tdiv;                  // gate set with ccx: only h = k and ceil(T bound / tdiv) are sound
  int states, m, gk;               // states: m pairs, rows of the residual are the images p|in_j> (transposed)
  const Zw *goal;                  // the search ends when the residual equals goal / sqrt2^gk (unitary: I)
  const int *ginfo;                // states: per pair sde, nullity, Schmidt rank per cut of the goal row
};

// ---------------- exact matrices ----------------
template <int NQ> struct Mat {
  static constexpr int N = 1 << NQ;
  Zw32 e[N * N];
  int k, ovf;                      // ovf: a value left the int32 range (the run aborts: never a silent wrong answer)
};

template <int NQ> HD inline void normalise(Mat<NQ> &S) {
  constexpr int N = 1 << NQ;
  while (S.k > 0) {
    for (int i = 0; i < N * N; i++) if (!zdiv2ok32(S.e[i])) return;
    for (int i = 0; i < N * N; i++) S.e[i] = zdown32(S.e[i]);
    S.k--;
  }
}

// S <- S g^-1: column c mixes the columns that agree with c outside g's qubits
template <int NQ> HD inline void apply_inv(Mat<NQ> &S, const Gate &g) {
  constexpr int N = 1 << NQ;
  const int L = 1 << g.nq;
  int mask = 0;
  for (int j = 0; j < g.nq; j++) mask |= 1 << g.q[j];
  Zw col[LMAX];
  for (int base = 0; base < N; base++) {
    if (base & mask) continue;
    int idx[LMAX];
    for (int l = 0; l < L; l++) {
      int x = base;
      for (int j = 0; j < g.nq; j++) if ((l >> j) & 1) x |= 1 << g.q[j];
      idx[l] = x;
    }
    for (int r = 0; r < N; r++) {
      for (int lc = 0; lc < L; lc++) {
        Zw acc{0, 0, 0, 0};
        for (int m = 0; m < L; m++) {
          const Zw gm = g.inv[m * L + lc];
          if (!zzero(gm)) acc = zadd(acc, zmul(ld(S.e[r * N + idx[m]]), gm));
        }
        col[lc] = acc;
      }
      for (int lc = 0; lc < L; lc++) S.e[r * N + idx[lc]] = st(col[lc], S.ovf);
    }
  }
  S.k += g.ik;
  normalise(S);
}

template <int NQ> HD inline void residual(Mat<NQ> &S, const Spec &sp, const uint8_t *seq, int len) {
  constexpr int N = 1 << NQ;
  S.ovf = 0;
  for (int i = 0; i < N * N; i++) S.e[i] = st(sp.target[i], S.ovf);
  S.k = sp.tk;
  for (int t = 0; t < len; t++) apply_inv(S, sp.gates[seq[t]]);
}

template <int NQ> HD inline bool is_identity(const Mat<NQ> &S) {
  constexpr int N = 1 << NQ;
  if (S.k) return false;
  for (int r = 0; r < N; r++)
    for (int c = 0; c < N; c++) {
      const Zw32 x = S.e[r * N + c];
      if (r == c ? !(x.a == 1 && !x.b && !x.c && !x.d) : !zzero32(x)) return false;
    }
  return true;
}

template <int NQ> HD inline bool is_goal(const Mat<NQ> &S, const Spec &sp) {
  if (S.k != sp.gk) return false;
  for (int i = 0; i < (1 << NQ) * (1 << NQ); i++) {
    const Zw g = sp.goal[i];
    const Zw32 x = S.e[i];
    if (x.a != g.a || x.b != g.b || x.c != g.c || x.d != g.d) return false;
  }
  return true;
}

HD inline uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 27;
  return h;
}
// a sum of per-entry hashes, so a warp can hash a matrix in parallel (equal residuals are still compared exactly)
HD inline void hash_entry(int i, Zw32 x, uint64_t &a, uint64_t &b) {
  a += mix(mix(mix(mix(0x12345678ULL + (uint64_t)i, (uint64_t)(int64_t)x.a), (uint64_t)(int64_t)x.b), (uint64_t)(int64_t)x.c), (uint64_t)(int64_t)x.d);
  b += mix(mix(mix(mix(0x9abcdef0ULL + 31 * (uint64_t)i, (uint64_t)((int64_t)x.d * 31 + 7)), (uint64_t)(int64_t)x.c), (uint64_t)(int64_t)x.b), (uint64_t)(int64_t)x.a);
}
HD inline void hash_final(int k, uint64_t a, uint64_t b, uint64_t &h1, uint64_t &h2) {
  h1 = mix(a, (uint64_t)k); h2 = mix(b, (uint64_t)k ^ 0x5555ULL);
}
template <int NQ> HD inline void hash2(const Mat<NQ> &S, uint64_t &h1, uint64_t &h2) {
  uint64_t a = 0, b = 0;
  for (int i = 0; i < (1 << NQ) * (1 << NQ); i++) hash_entry(i, S.e[i], a, b);
  hash_final(S.k, a, b, h1, h2);
}

// ---------------- lower bounds (milq depth_bounds.lower_bound, strong, -v all) ----------------
HD inline uint64_t mulmod(uint64_t a, uint64_t b) { return (a * b) % PRIME; }
HD inline uint64_t powmod(uint64_t a, uint64_t e) {
  uint64_t r = 1;
  while (e) { if (e & 1) r = mulmod(r, a); a = mulmod(a, a); e >>= 1; }
  return r;
}
HD inline uint64_t tomod(int64_t v) { int64_t m = v % (int64_t)PRIME; return (uint64_t)(m < 0 ? m + (int64_t)PRIME : m); }
HD inline uint64_t zmod(Zw x, uint64_t w) {
  const uint64_t w2 = mulmod(w, w), w3 = mulmod(w2, w);
  return (tomod(x.a) + mulmod(tomod(x.b), w) + mulmod(tomod(x.c), w2) + mulmod(tomod(x.d), w3)) % PRIME;
}

// rank over F_p of the operator-Schmidt matrix of S across the cut (qubit mask A): never above the true rank,
// so the bound stays sound if reduction mod p ever loses rank
template <int NQ> HD inline int cut_rank(const Mat<NQ> &S, int A, int sa, uint64_t w) {
  constexpr int N = 1 << NQ;
  constexpr int MAXR = (NQ / 2 >= 1) ? (1 << (2 * (NQ / 2))) : 1;     // rows 4^|A|, |A| <= n/2
  constexpr int MAXC = 1 << (2 * NQ);                                  // generous
  const int R = 1 << (2 * sa), C = (N * N) / R;
  uint64_t M[MAXR][MAXC];
  // row = (oA, iA), col = (oB, iB): bits of o and i split by the mask A
  for (int o = 0; o < N; o++)
    for (int i = 0; i < N; i++) {
      int oa = 0, ob = 0, ia = 0, ib = 0, na = 0, nb = 0;
      for (int q = 0; q < NQ; q++) {
        if ((A >> q) & 1) { oa |= ((o >> q) & 1) << na; ia |= ((i >> q) & 1) << na; na++; }
        else { ob |= ((o >> q) & 1) << nb; ib |= ((i >> q) & 1) << nb; nb++; }
      }
      M[(oa << na) | ia][(ob << nb) | ib] = zmod(ld(S.e[o * N + i]), w);
    }
  int rank = 0;
  for (int c = 0; c < C && rank < R; c++) {
    int piv = -1;
    for (int r = rank; r < R; r++) if (M[r][c]) { piv = r; break; }
    if (piv < 0) continue;
    if (piv != rank) for (int j = 0; j < C; j++) { uint64_t t = M[piv][j]; M[piv][j] = M[rank][j]; M[rank][j] = t; }
    const uint64_t inv = powmod(M[rank][c], PRIME - 2);
    for (int r = rank + 1; r < R; r++) {
      if (!M[r][c]) continue;
      const uint64_t f = mulmod(M[r][c], inv);
      for (int j = c; j < C; j++) M[r][j] = (M[r][j] + PRIME - mulmod(f, M[rank][j])) % PRIME;
    }
    rank++;
  }
  return rank;
}

template <int NQ> HD inline int support(const Mat<NQ> &S) {
  constexpr int N = 1 << NQ;
  int cnt = 0;
  for (int q = 0; q < NQ; q++) {
    const int bit = 1 << q;
    bool trivial = true;
    for (int r = 0; r < N && trivial; r++)
      for (int c = 0; c < N && trivial; c++) {
        if ((r ^ c) & bit) { if (!zzero32(S.e[r * N + c])) trivial = false; }
        else if (!((r | c) & bit) && !zeq32(S.e[r * N + c], S.e[(r | bit) * N + (c | bit)])) trivial = false;
      }
    cnt += !trivial;
  }
  return cnt;
}

// 2 if S is monomial with a non-affine permutation (needs >= 2 H: lb_h_struct), else 0
template <int NQ> HD inline int h_struct(const Mat<NQ> &S) {
  constexpr int N = 1 << NQ;
  int pi[N];
  for (int c = 0; c < N; c++) {
    int nz = 0, row = 0;
    for (int r = 0; r < N; r++) if (!zzero32(S.e[r * N + c])) { nz++; row = r; }
    if (nz != 1) return 0;
    pi[c] = row;
  }
  const int c0 = pi[0];
  for (int j = 0; j < N; j++) {
    int x = c0;
    for (int i = 0; i < NQ; i++) if ((j >> i) & 1) x ^= pi[1 << i] ^ c0;
    if (pi[j] != x) return 2;
  }
  return 0;
}

// sde of the Pauli-transfer matrix R[a][b] = Tr(P_a S P_b S^dag) / 2^n (Gosset et al.; >= T count). With
// S' = sqrt2^k S integral, X_ab = Tr(P_a S' P_b S'^dag) = 2^(n+k) R_ab, so sde(R) = max 2(n+k) - v(X_ab).
// Pauli digit per qubit: 0 I, 1 X, 2 Y, 3 Z; P|x> = phase |x ^ flip>.
HD inline void pauli(int b, int nq, int &flip, int &ymask, int &zmask) {
  flip = ymask = zmask = 0;
  for (int q = 0; q < nq; q++) {
    const int d = (b >> (2 * q)) & 3;
    if (d == 1 || d == 2) flip |= 1 << q;
    if (d == 2) ymask |= 1 << q;
    if (d == 3) zmask |= 1 << q;
  }
}
HD inline int popc(int x) { int c = 0; while (x) { c += x & 1; x >>= 1; } return c; }
HD inline int pphase(int x, int ymask, int zmask) {      // power of w: Y|x> = i (-1)^x |x^1>, Z|x> = (-1)^x |x>
  return (2 * popc(ymask) + 4 * popc(x & (ymask | zmask))) & 7;
}

template <int NQ> HD inline int ptm_sde(const Mat<NQ> &S) {
  constexpr int N = 1 << NQ, P = 1 << (2 * NQ);
  // |X_ab| <= ~2^k N^2 (coefficients ~2^(k/2), N^2 products summed): beyond int64, claim nothing (sound)
  if (S.k + 2 * NQ + 6 > 62) return 0;
  int best = 0;
  Zw SP[N * N], M[N * N];
  for (int b = 0; b < P; b++) {
    int fb, yb, zb;
    pauli(b, NQ, fb, yb, zb);
    for (int r = 0; r < N; r++)                          // (S' P_b)[r][c] = S'[r][c ^ fb] * phase(b, c)
      for (int c = 0; c < N; c++) {
        const Zw x = ld(S.e[r * N + (c ^ fb)]);
        SP[r * N + c] = zzero(x) ? x : zmul(x, zwpow(pphase(c, yb, zb)));
      }
    for (int r = 0; r < N; r++)                          // M = (S' P_b) S'^dag
      for (int s = 0; s < N; s++) {
        Zw acc{0, 0, 0, 0};
        for (int c = 0; c < N; c++) {
          const Zw y = ld(S.e[s * N + c]);
          if (!zzero(y) && !zzero(SP[r * N + c])) acc = zadd(acc, zmul(SP[r * N + c], zconj(y)));
        }
        M[r * N + s] = acc;
      }
    for (int a = 0; a < P; a++) {                        // X_ab = sum_x phase(a, x^fa) M[x^fa][x]
      int fa, ya, za;
      pauli(a, NQ, fa, ya, za);
      Zw acc{0, 0, 0, 0};
      for (int x = 0; x < N; x++) {
        const Zw m = M[(x ^ fa) * N + x];
        if (!zzero(m)) acc = zadd(acc, zmul(m, zwpow(pphase(x ^ fa, ya, za))));
      }
      if (!zzero(acc)) {
        const int need = 2 * (NQ + S.k) - zval(acc);
        if (need > best) best = need;
      }
    }
  }
  return best;
}

// The same bound, cheaper. Units (powers of w) don't change the valuation, so with zz = Y|Z digits of a Pauli,
// X_ab ~ sum_{x,c} (-1)^{x.zza + c.zzb} S'[x^fa][c^fb] conj(S'[x][c]): for each (fa, fb), a 2-D Walsh-Hadamard
// transform of N^2 products gives all (zza, zzb) at once - N^4 products in all instead of 32^n. Arithmetic in
// Z[w] mod 2^b (U = uint32 or uint64, wrapping) is enough: v(X) = 2 min_i v2(X_i) + [X / 2^m divisible by sqrt2]
// is read off the low bits whenever X != 0 mod 2^b, and a nonzero X = 0 mod 2^b has v >= b, need <= 0.
template <typename U> struct ZU { U a, b, c, d; };
template <typename U> HD inline ZU<U> zuadd(ZU<U> x, ZU<U> y) { return {U(x.a + y.a), U(x.b + y.b), U(x.c + y.c), U(x.d + y.d)}; }
template <typename U> HD inline ZU<U> zusub(ZU<U> x, ZU<U> y) { return {U(x.a - y.a), U(x.b - y.b), U(x.c - y.c), U(x.d - y.d)}; }
template <typename U> HD inline ZU<U> zumul(ZU<U> x, ZU<U> y) {
  return {U(x.a * y.a - x.b * y.d - x.c * y.c - x.d * y.b), U(x.a * y.b + x.b * y.a - x.c * y.d - x.d * y.c),
          U(x.a * y.c + x.b * y.b + x.c * y.a - x.d * y.d), U(x.a * y.d + x.b * y.c + x.c * y.b + x.d * y.a)};
}
template <typename U> HD inline ZU<U> zuconj(ZU<U> x) { return {x.a, U(0 - x.d), U(0 - x.c), U(0 - x.b)}; }
template <typename U> HD inline ZU<U> zuof(Zw32 x) { return {U((int64_t)x.a), U((int64_t)x.b), U((int64_t)x.c), U((int64_t)x.d)}; }
HD inline int ctz64(uint64_t x) {
#ifdef __CUDA_ARCH__
  return __ffsll((long long)x) - 1;
#else
  return __builtin_ctzll(x);
#endif
}
template <typename U> HD inline int zuval(ZU<U> x) {   // sqrt2-adic valuation from the low bits, -1 if x = 0 mod 2^b
  const U o = x.a | x.b | x.c | x.d;
  if (!o) return -1;
  const int m = ctz64((uint64_t)o);
  const U ya = x.a >> m, yb = x.b >> m, yc = x.c >> m, yd = x.d >> m;
  return 2 * m + (!((ya ^ yc) & 1) && !((yb ^ yd) & 1));
}

template <int NQ, typename U> HD inline int ptm_wht_t(const Mat<NQ> &S) {
  constexpr int N = 1 << NQ, E = N * N;
  const int top = 2 * (NQ + S.k);
  int best = 0;
  ZU<U> Q[E];
  for (int fa = 0; fa < N; fa++)
    for (int fb = 0; fb < N; fb++) {
      for (int e = 0; e < E; e++) {
        const int x = e >> NQ, c = e & (N - 1);
        Q[e] = zumul(zuof<U>(S.e[(x ^ fa) * N + (c ^ fb)]), zuconj(zuof<U>(S.e[e])));
      }
      for (int h = 1; h < E; h <<= 1)
        for (int e = 0; e < E; e++)
          if (!(e & h)) { const ZU<U> u = Q[e], v = Q[e | h]; Q[e] = zuadd(u, v); Q[e | h] = zusub(u, v); }
      for (int e = 0; e < E; e++) {
        const int v = zuval(Q[e]);
        if (v >= 0 && top - v > best) best = top - v;
      }
    }
  return best;
}
template <int NQ> HD inline int ptm_wht(const Mat<NQ> &S) {
  return 2 * (NQ + S.k) <= 64 ? ptm_wht_t<NQ, uint32_t>(S) : ptm_wht_t<NQ, uint64_t>(S);
}

// ---------------- state bounds (rows j < m are the images of the inputs, goal rows fixed) ----------------
// H: sde changes by <= 1 per H. T: stabilizer nullity changes by <= 1 per T (Cliffords keep it). CX: the
// Schmidt rank across a cut changes by a factor <= 2 per crossing CX/CZ. Ranks are exact (minors in int64
// under a coefficient bound; mod-p ranks could be too low and overstate the gap to a higher goal rank).
HD inline int64_t zbig(Zw32 x) {
  int64_t m = 0, v;
  v = x.a < 0 ? -(int64_t)x.a : x.a; if (v > m) m = v;
  v = x.b < 0 ? -(int64_t)x.b : x.b; if (v > m) m = v;
  v = x.c < 0 ? -(int64_t)x.c : x.c; if (v > m) m = v;
  v = x.d < 0 ? -(int64_t)x.d : x.d; if (v > m) m = v;
  return m;
}
HD inline int row_sde(const Zw32 *row, int N, int k) {
  int v = 1 << 30;
  for (int x = 0; x < N; x++) if (!zzero32(row[x])) { const int z = zval(ld(row[x])); if (z < v) v = z; }
  return v == (1 << 30) || k - v < 0 ? 0 : k - v;
}
HD inline Zw zneg(Zw x) { return {-x.a, -x.b, -x.c, -x.d}; }
HD inline bool pauli_stab(const Zw32 *row, int nq, int b, Zw nrm) {   // P_b |v> = +-|v>  <=>  <v|P_b|v> = +-<v|v>
  int f, y, z;
  pauli(b, nq, f, y, z);
  Zw acc{0, 0, 0, 0};
  for (int x = 0; x < (1 << nq); x++) {
    const Zw32 u = row[x ^ f], v = row[x];
    if (zzero32(u) || zzero32(v)) continue;
    acc = zadd(acc, zmul(zconj(ld(u)), zmul(ld(v), zwpow(pphase(x, y, z)))));
  }
  return zeq(acc, nrm) || zeq(acc, zneg(nrm));
}
HD inline Zw row_norm(const Zw32 *row, int N) {
  Zw acc{0, 0, 0, 0};
  for (int x = 0; x < N; x++) if (!zzero32(row[x])) acc = zadd(acc, zmul(zconj(ld(row[x])), ld(row[x])));
  return acc;
}
HD inline bool nullity_ok(const Zw32 *row, int N) {     // int64 safe: N * 4 * B^2 < 2^62
  int64_t B = 0;
  for (int x = 0; x < N; x++) { const int64_t v = zbig(row[x]); if (v > B) B = v; }
  return B < (1LL << 27);
}
HD inline int ilog2(int c) { int r = 0; while ((2 << r) <= c) r++; return r; }

HD inline Zw zsub(Zw x, Zw y) { return {x.a - y.a, x.b - y.b, x.c - y.c, x.d - y.d}; }
HD inline Zw det_small(const Zw *M, int r, int stride) {   // r <= 4, Laplace along the first row
  if (r == 1) return M[0];
  if (r == 2) return zsub(zmul(M[0], M[stride + 1]), zmul(M[1], M[stride]));
  Zw acc{0, 0, 0, 0};
  Zw sub[9];
  for (int j = 0; j < r; j++) {
    if (zzero(M[j])) continue;
    int t = 0;
    for (int i = 1; i < r; i++) for (int c = 0; c < r; c++) if (c != j) sub[t++] = M[i * stride + c];
    const Zw d = zmul(M[j], det_small(sub, r - 1, r - 1));
    acc = (j & 1) ? zsub(acc, d) : zadd(acc, d);
  }
  return acc;
}
// exact Schmidt rank of the row reshaped (qubits A, |A| = sa rows side) x (rest); -1 if coefficients too large
HD inline int row_rank(const Zw32 *row, int nq, int A, int sa) {
  const int R = 1 << sa, C = 1 << (nq - sa), N = 1 << nq;
  Zw M[4 * 16];
  int64_t B = 0;
  for (int x = 0; x < N; x++) {
    int ia = 0, ib = 0, na = 0, nb = 0;
    for (int q = 0; q < nq; q++) { if ((A >> q) & 1) ia |= ((x >> q) & 1) << na++; else ib |= ((x >> q) & 1) << nb++; }
    M[ia * C + ib] = ld(row[x]);
    const int64_t v = zbig(row[x]); if (v > B) B = v;
  }
  const int64_t lim[5] = {0, 1LL << 40, 1LL << 29, 1LL << 18, 1LL << 12};
  for (int r = R < C ? R : C; r >= 2; r--) {
    if (B >= lim[r]) return -1;
    int rs[4], cs[4];                                // r-subsets of rows and columns as bitmasks
    for (int rm = 0; rm < (1 << R); rm++) {
      if (popc(rm) != r) continue;
      int t = 0; for (int i = 0; i < R; i++) if ((rm >> i) & 1) rs[t++] = i;
      for (int cm = 0; cm < (1 << C); cm++) {
        if (popc(cm) != r) continue;
        int u = 0; for (int i = 0; i < C; i++) if ((cm >> i) & 1) cs[u++] = i;
        Zw sub[16];
        for (int i = 0; i < r; i++) for (int j = 0; j < r; j++) sub[i * r + j] = M[rs[i] * C + cs[j]];
        if (!zzero(det_small(sub, r, r))) return r;
      }
    }
  }
  for (int x = 0; x < N; x++) if (!zzero32(row[x])) return 1;
  return 0;
}
HD inline int rank_need(int rv, int rt) {     // fewest crossing CX/CZ between Schmidt ranks rv and rt
  if (rv < 0 || rt <= 0 || rv == 0) return 0;
  int lo = rv < rt ? rv : rt, hi = rv < rt ? rt : rv, c = 0;
  while ((lo << c) < hi) c++;
  return c;
}
HD inline int ginfo_w(const Spec &sp) { return 2 + sp.ncuts; }
HD inline int cx_from_needs(const int *need, const Spec &sp) {
  int idx = 0, mult = 1, mx = 0;
  for (int i = 0; i < sp.ncuts; i++) {
    int nd = need[i];
    if (nd > sp.rad[i] - 1) nd = sp.rad[i] - 1;
    idx += nd * mult;
    mult *= sp.rad[i];
    if (nd > mx) mx = nd;
  }
  return sp.have_table ? sp.cxtable[idx] : mx;
}
template <int NQ> HD inline void state_cheap(const Mat<NQ> &S, const Spec &sp, int &h, int &c2) {
  constexpr int N = 1 << NQ;
  int need[CUTMAX] = {0};
  h = 0;
  for (int j = 0; j < sp.m; j++) {
    const int *gi = sp.ginfo + j * ginfo_w(sp);
    const int d = row_sde(S.e + j * N, N, S.k) - gi[0];
    if ((d < 0 ? -d : d) > h) h = d < 0 ? -d : d;
    for (int c = 0; c < sp.ncuts; c++) {
      const int nd = rank_need(row_rank(S.e + j * N, NQ, sp.cutmask[c], sp.cutsize[c]), gi[2 + c]);
      if (nd > need[c]) need[c] = nd;
    }
  }
  c2 = cx_from_needs(need, sp);
}
template <int NQ> HD inline int state_t(const Mat<NQ> &S, const Spec &sp) {
  constexpr int N = 1 << NQ;
  int best = 0;
  for (int j = 0; j < sp.m; j++) {
    const Zw32 *row = S.e + j * N;
    if (!nullity_ok(row, N)) continue;
    const Zw nrm = row_norm(row, N);
    int cnt = 0;
    for (int b = 0; b < (1 << (2 * NQ)); b++) cnt += pauli_stab(row, NQ, b, nrm);
    const int d = (NQ - ilog2(cnt)) - sp.ginfo[j * ginfo_w(sp) + 1];
    if ((d < 0 ? -d : d) > best) best = d < 0 ? -d : d;
  }
  return best;
}

template <int NQ> HD inline int cx_bound(const Mat<NQ> &S, const Spec &sp) {
  int idx = 0, mult = 1, mx = 0;
  for (int i = 0; i < sp.ncuts; i++) {
    const int rk = cut_rank(S, sp.cutmask[i], sp.cutsize[i], sp.w8);
    int need = 0;
    while ((1 << need) < rk) need++;
    if (need > sp.rad[i] - 1) need = sp.rad[i] - 1;
    idx += need * mult;
    mult *= sp.rad[i];
    if (need > mx) mx = need;
  }
  return sp.have_table ? sp.cxtable[idx] : mx;
}

HD inline int combine(int h, int t, int c2, int supp) {
  int best = 1 << 30;
  const int hi = c2 > supp ? c2 : supp;
  for (int c = c2; c <= hi; c++) {
    const int one = (h + t > supp - 2 * c) ? h + t : supp - 2 * c;
    if (c + one < best) best = c + one;
  }
  return best;
}

// ---------------- frontier ----------------
struct Node {
  uint64_t h1, h2;
  uint8_t len, seq[DMAX];
};

HD inline bool node_less(const Node &x, const Node &y) {      // hash, then lexicographic prefix
  if (x.h1 != y.h1) return x.h1 < y.h1;
  if (x.h2 != y.h2) return x.h2 < y.h2;
  for (int i = 0; i < x.len && i < y.len; i++) if (x.seq[i] != y.seq[i]) return x.seq[i] < y.seq[i];
  return x.len < y.len;
}
struct NodeLess { HD bool operator()(const Node &x, const Node &y) const { return node_less(x, y); } };

enum : uint8_t { DEAD = 0, ALIVE = 1, FOUND = 2, W2CUT = 3, W3CUT = 4, NEEDT = 5, OVF = 6 };

template <int NQ> struct ParentResidual {   // pass 0: S = U p^-1 once per parent
  Spec sp;
  const Node *parents;
  Mat<NQ> *R;
  HD void operator()(int i) const { residual(R[i], sp, parents[i].seq, parents[i].len); }
};

// pass 1, one thread per child: windows, gate, identity, the cheap bounds (H, structure, support, CX). Children the
// cheap bound can't prune become NEEDT with (h, c2, support) packed in `cheap` for pass 2.
template <int NQ> struct Cheap {
  Spec sp;
  const Node *parents;
  const Mat<NQ> *R;
  uint8_t *status;
  int *cheap;
  uint64_t *h1, *h2;
  int t;
  HD void operator()(int o) const {
    const int i = o / sp.G, g = o % sp.G;
    const Node &p = parents[i];
    if (p.len >= 1 && sp.bad2[p.seq[p.len - 1] * sp.G + g]) { status[o] = W2CUT; return; }
    if (p.len >= 2) {
      const uint32_t tri = ((uint32_t)p.seq[p.len - 2] * sp.G + p.seq[p.len - 1]) * sp.G + g;
      if ((sp.bad3[tri >> 5] >> (tri & 31)) & 1) { status[o] = W3CUT; return; }
    }
    Mat<NQ> C = R[i];
    apply_inv(C, sp.gates[g]);
    if (C.ovf || R[i].ovf) { status[o] = OVF; return; }
    if (is_goal(C, sp)) { status[o] = FOUND; return; }
    if (!sp.standard && !sp.lite) { status[o] = ALIVE; hash2(C, h1[o], h2[o]); return; }
    int h, supp = 0, c2 = 0;
    if (sp.lite) h = C.k;
    else
    if (sp.states) state_cheap(C, sp, h, c2);
    else if (!sp.lite) {
      const int hs = h_struct(C);
      h = C.k > hs ? C.k : hs;
      supp = support(C); c2 = cx_bound(C, sp);
    }
    if (combine(h, 0, c2, supp) > sp.depth - t) { status[o] = DEAD; return; }
    status[o] = NEEDT;
    cheap[o] = h | (c2 << 10) | (supp << 20);
  }
};

template <int NQ> struct Full {             // pass 2, on NEEDT children only: T bound (PTM sde), then hash
  Spec sp;
  const Mat<NQ> *R;
  uint8_t *status;
  const int *cheap;
  uint64_t *h1, *h2;
  int t;
  HD void operator()(int o) const {
    const int i = o / sp.G, g = o % sp.G;
    Mat<NQ> C = R[i];
    apply_inv(C, sp.gates[g]);
    const int c = cheap[o], h = c & 1023, c2 = (c >> 10) & 1023, supp = c >> 20;
#ifdef CHECK_T
    if (!sp.states && ptm_wht(C) != ptm_sde(C)) { std::printf("T MISMATCH %d %d k=%d\n", ptm_wht(C), ptm_sde(C), C.k); std::abort(); }
#endif
    const int T = sp.states ? state_t(C, sp) : (ptm_wht(C) + sp.tdiv - 1) / sp.tdiv;
    if (combine(h, T, c2, supp) > sp.depth - t) { status[o] = DEAD; return; }
    status[o] = ALIVE;
    hash2(C, h1[o], h2[o]);
  }
};

#ifdef __CUDACC__
// pass 2 on the GPU: one warp per child. The child residual lives in shared memory (each lane owns entries
// lane, lane + 32, ...), the Walsh-Hadamard butterflies across lanes are shuffles, the max is a warp reduction.
constexpr unsigned FULL_MASK = 0xffffffffu;
constexpr int WPB = 4;                       // warps per block

template <typename U> __device__ inline ZU<U> zushfl(ZU<U> x, int m) {
  return {__shfl_xor_sync(FULL_MASK, x.a, m), __shfl_xor_sync(FULL_MASK, x.b, m), __shfl_xor_sync(FULL_MASK, x.c, m),
          __shfl_xor_sync(FULL_MASK, x.d, m)};
}

template <int NQ, typename U> __device__ int warp_wht(const Zw32 *S, int k, int lane) {
  constexpr int N = 1 << NQ, E = N * N, EPL = E >= 32 ? E / 32 : 1;
  const int top = 2 * (NQ + k);
  int best = 0;
  for (int fa = 0; fa < N; fa++)
    for (int fb = 0; fb < N; fb++) {
      ZU<U> q[EPL];
#pragma unroll
      for (int j = 0; j < EPL; j++) {
        const int e = lane + 32 * j;
        if (e < E) {
          const int x = e >> NQ, c = e & (N - 1);
          q[j] = zumul(zuof<U>(S[(x ^ fa) * N + (c ^ fb)]), zuconj(zuof<U>(S[e])));
        } else q[j] = {0, 0, 0, 0};
      }
#pragma unroll
      for (int h = 1; h < E; h <<= 1) {
        if (h < 32) {
#pragma unroll
          for (int j = 0; j < EPL; j++) {
            const ZU<U> o = zushfl(q[j], h);
            q[j] = (lane & h) ? zusub(o, q[j]) : zuadd(q[j], o);
          }
        } else {
          const int hj = h / 32;
#pragma unroll
          for (int j = 0; j < EPL; j++)
            if (!(j & hj)) { const ZU<U> u = q[j], v = q[j | hj]; q[j] = zuadd(u, v); q[j | hj] = zusub(u, v); }
        }
      }
#pragma unroll
      for (int j = 0; j < EPL; j++) {
        const int v = zuval(q[j]);
        if (v >= 0 && top - v > best) best = top - v;
      }
    }
  for (int m = 16; m; m >>= 1) { const int o = __shfl_xor_sync(FULL_MASK, best, m); if (o > best) best = o; }
  return best;
}

// child residual S = parent g^-1, normalised (as apply_inv + normalise); returns its sqrt2 exponent
template <int NQ> __device__ int warp_child(Zw32 *S, const Mat<NQ> &P, const Gate &gt, int lane, bool &ovf) {
  constexpr int N = 1 << NQ, E = N * N, EPL = E >= 32 ? E / 32 : 1;
  for (int e = lane; e < E; e += 32) S[e] = P.e[e];
  int k = P.k;
  __syncwarp();
  const int L = 1 << gt.nq;
  int mask = 0;
  for (int j = 0; j < gt.nq; j++) mask |= 1 << gt.q[j];
  Zw32 nv[EPL];
  int of = 0;
#pragma unroll
  for (int j = 0; j < EPL; j++) {
    const int e = lane + 32 * j;
    if (e >= E) continue;
    const int r = e >> NQ, col = e & (N - 1), base = col & ~mask;
    int lc = 0;
    for (int jj = 0; jj < gt.nq; jj++) lc |= ((col >> gt.q[jj]) & 1) << jj;
    Zw acc{0, 0, 0, 0};
    for (int m = 0; m < L; m++) {
      const Zw gm = gt.inv[m * L + lc];
      if (zzero(gm)) continue;
      int x = base;
      for (int jj = 0; jj < gt.nq; jj++) if ((m >> jj) & 1) x |= 1 << gt.q[jj];
      acc = zadd(acc, zmul(ld(S[r * N + x]), gm));
    }
    nv[j] = st(acc, of);
  }
  __syncwarp();
#pragma unroll
  for (int j = 0; j < EPL; j++) if (lane + 32 * j < E) S[lane + 32 * j] = nv[j];
  k += gt.ik;
  while (k > 0) {
    bool ok = true;
#pragma unroll
    for (int j = 0; j < EPL; j++) if (lane + 32 * j < E && !zdiv2ok32(S[lane + 32 * j])) ok = false;
    if (!__all_sync(FULL_MASK, ok)) break;
#pragma unroll
    for (int j = 0; j < EPL; j++) if (lane + 32 * j < E) S[lane + 32 * j] = zdown32(S[lane + 32 * j]);
    k--;
  }
  ovf = __any_sync(FULL_MASK, of || P.ovf);
  __syncwarp();
  return k;
}

template <int NQ> __device__ bool warp_identity(const Zw32 *S, int k, int lane) {
  constexpr int N = 1 << NQ, E = N * N;
  if (k) return false;
  bool ok = true;
  for (int e = lane; e < E; e += 32) {
    const Zw32 x = S[e];
    if ((e >> NQ) == (e & (N - 1)) ? !(x.a == 1 && !x.b && !x.c && !x.d) : !zzero32(x)) ok = false;
  }
  return __all_sync(FULL_MASK, ok);
}

template <int NQ> __device__ int warp_h_struct(const Zw32 *S, int lane) {   // lane c: column c (N <= 16)
  constexpr int N = 1 << NQ;
  int nz = 0, row = 0;
  if (lane < N) for (int r = 0; r < N; r++) if (!zzero32(S[r * N + lane])) { nz++; row = r; }
  if (__any_sync(FULL_MASK, lane < N && nz != 1)) return 0;
  const int c0 = __shfl_sync(FULL_MASK, row, 0);
  int x = c0;
  for (int i = 0; i < NQ; i++) {
    const int pb = __shfl_sync(FULL_MASK, row, 1 << i);
    if ((lane >> i) & 1) x ^= pb ^ c0;
  }
  return __any_sync(FULL_MASK, lane < N && row != x) ? 2 : 0;
}

template <int NQ> __device__ int warp_support(const Zw32 *S, int lane) {
  constexpr int N = 1 << NQ, E = N * N;
  int cnt = 0;
  for (int q = 0; q < NQ; q++) {
    const int bit = 1 << q;
    bool nt = false;
    for (int e = lane; e < E; e += 32) {
      const int r = e >> NQ, c = e & (N - 1);
      if ((r ^ c) & bit) { if (!zzero32(S[e])) nt = true; }
      else if (!((r | c) & bit) && !zeq32(S[e], S[(r | bit) * N + (c | bit)])) nt = true;
    }
    cnt += __any_sync(FULL_MASK, nt);
  }
  return cnt;
}

// cut_rank with the R x C matrix (R = 4^|A| <= 16 rows) in shared memory M, row factors in F
template <int NQ> __device__ int warp_cut_rank(const Zw32 *S, uint32_t *M, uint32_t *F, int A, int sa, uint64_t w, int lane) {
  constexpr int N = 1 << NQ, E = N * N;
  const int R = 1 << (2 * sa), C = E / R;
  for (int e = lane; e < E; e += 32) {
    const int o = e >> NQ, i = e & (N - 1);
    int oa = 0, ob = 0, ia = 0, ib = 0, na = 0, nb = 0;
    for (int q = 0; q < NQ; q++) {
      if ((A >> q) & 1) { oa |= ((o >> q) & 1) << na; ia |= ((i >> q) & 1) << na; na++; }
      else { ob |= ((o >> q) & 1) << nb; ib |= ((i >> q) & 1) << nb; nb++; }
    }
    M[((oa << na) | ia) * C + ((ob << nb) | ib)] = (uint32_t)zmod(ld(S[e]), w);
  }
  __syncwarp();
  int rank = 0;
  for (int c = 0; c < C && rank < R; c++) {
    const unsigned bal = __ballot_sync(FULL_MASK, lane >= rank && lane < R && M[lane * C + c] != 0);
    if (!bal) continue;
    const int piv = __ffs(bal) - 1;
    if (piv != rank) {
      for (int j = lane; j < C; j += 32) { const uint32_t x = M[piv * C + j]; M[piv * C + j] = M[rank * C + j]; M[rank * C + j] = x; }
      __syncwarp();
    }
    const uint64_t inv = powmod(M[rank * C + c], PRIME - 2);
    if (lane > rank && lane < R) F[lane] = (uint32_t)mulmod(M[lane * C + c], inv);
    __syncwarp();
    for (int x = lane; x < R * C; x += 32) {
      const int r = x / C, j = x % C;
      if (r > rank && j >= c && F[r]) M[x] = (uint32_t)((M[x] + PRIME - mulmod(F[r], M[rank * C + j])) % PRIME);
    }
    __syncwarp();
    rank++;
  }
  return rank;
}

template <int NQ> __device__ int warp_cx_bound(const Zw32 *S, uint32_t *M, uint32_t *F, const Spec &sp, int lane) {
  int idx = 0, mult = 1, mx = 0;
  for (int i = 0; i < sp.ncuts; i++) {
    const int rk = warp_cut_rank<NQ>(S, M, F, sp.cutmask[i], sp.cutsize[i], sp.w8, lane);
    int need = 0;
    while ((1 << need) < rk) need++;
    if (need > sp.rad[i] - 1) need = sp.rad[i] - 1;
    idx += need * mult;
    mult *= sp.rad[i];
    if (need > mx) mx = need;
  }
  return sp.have_table ? sp.cxtable[idx] : mx;
}

template <int NQ> __device__ bool warp_goal(const Zw32 *S, int k, const Spec &sp, int lane) {
  constexpr int E = (1 << NQ) * (1 << NQ);
  if (k != sp.gk) return false;
  bool ok = true;
  for (int e = lane; e < E; e += 32) {
    const Zw g = sp.goal[e];
    const Zw32 x = S[e];
    if (x.a != g.a || x.b != g.b || x.c != g.c || x.d != g.d) ok = false;
  }
  return __all_sync(FULL_MASK, ok);
}
__device__ inline int warp_max(int v) {
  for (int m = 16; m; m >>= 1) { const int o = __shfl_xor_sync(FULL_MASK, v, m); if (o > v) v = o; }
  return v;
}
template <int NQ> __device__ void warp_state_cheap(const Zw32 *S, int k, const Spec &sp, int lane, int &h, int &c2) {
  constexpr int N = 1 << NQ;
  const int W = ginfo_w(sp);
  int d = 0;
  if (lane < sp.m) { d = row_sde(S + lane * N, N, k) - sp.ginfo[lane * W]; if (d < 0) d = -d; }
  h = warp_max(d);
  int mine[CUTMAX] = {0}, need[CUTMAX];
  for (int it = lane; it < sp.m * sp.ncuts; it += 32) {
    const int j = it / sp.ncuts, c = it % sp.ncuts;
    const int nd = rank_need(row_rank(S + j * N, NQ, sp.cutmask[c], sp.cutsize[c]), sp.ginfo[j * W + 2 + c]);
    if (nd > mine[c]) mine[c] = nd;
  }
  for (int c = 0; c < sp.ncuts; c++) need[c] = warp_max(mine[c]);
  c2 = cx_from_needs(need, sp);
}
template <int NQ> __device__ int warp_state_t(const Zw32 *S, const Spec &sp, int lane) {
  constexpr int N = 1 << NQ, P = 1 << (2 * NQ);
  int best = 0;
  for (int j = 0; j < sp.m; j++) {
    const Zw32 *row = S + j * N;
    if (!nullity_ok(row, N)) continue;
    const Zw nrm = row_norm(row, N);
    int cnt = 0;
    for (int b0 = 0; b0 < P; b0 += 32) {
      const int b = b0 + lane;
      cnt += __popc(__ballot_sync(FULL_MASK, b < P && pauli_stab(row, NQ, b, nrm)));
    }
    int d = (NQ - ilog2(cnt)) - sp.ginfo[j * ginfo_w(sp) + 1];
    if (d < 0) d = -d;
    if (d > best) best = d;
  }
  return best;
}

// one level, fused: one warp per child does windows, child residual, identity, cheap bounds, T bound, hash
template <int NQ, bool ST> __global__ void __launch_bounds__(32 * WPB)
level_warp(Spec sp, const Node *parents, const Mat<NQ> *R, long nc, uint8_t *status, uint64_t *h1, uint64_t *h2, int t) {
  constexpr int N = 1 << NQ, E = N * N;
  __shared__ Zw32 shS[WPB][E];
  __shared__ uint32_t shM[WPB][E], shF[WPB][16];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  Zw32 *S = shS[w];
  for (long o = (long)blockIdx.x * WPB + w; o < nc; o += (long)gridDim.x * WPB) {
    const int i = (int)(o / sp.G), g = (int)(o % sp.G);
    const Node &p = parents[i];
    uint8_t s;
    if (p.len >= 1 && sp.bad2[p.seq[p.len - 1] * sp.G + g]) s = W2CUT;
    else if (p.len >= 2 && ((sp.bad3[(((uint32_t)p.seq[p.len - 2] * sp.G + p.seq[p.len - 1]) * sp.G + g) >> 5] >>
                             ((((uint32_t)p.seq[p.len - 2] * sp.G + p.seq[p.len - 1]) * sp.G + g) & 31)) & 1)) s = W3CUT;
    else {
      bool ovf;
      const int k = warp_child<NQ>(S, R[i], sp.gates[g], lane, ovf);
      if (ovf) s = OVF;
      else if (warp_goal<NQ>(S, k, sp, lane)) s = FOUND;
      else {
        bool alive = true;
        if constexpr (ST) {
          if (sp.standard) {
          int h, c2;
          warp_state_cheap<NQ>(S, k, sp, lane, h, c2);
          if (combine(h, 0, c2, 0) > sp.depth - t) alive = false;
          else alive = combine(h, warp_state_t<NQ>(S, sp, lane), c2, 0) <= sp.depth - t;
          }
        } else if (sp.lite) {
          if (k > sp.depth - t) alive = false;
          else {
            const int T = 2 * (NQ + k) <= 64 ? warp_wht<NQ, uint32_t>(S, k, lane) : warp_wht<NQ, uint64_t>(S, k, lane);
            alive = k + (T + sp.tdiv - 1) / sp.tdiv <= sp.depth - t;
          }
        } else if (sp.standard) {
          const int hs = warp_h_struct<NQ>(S, lane), h = k > hs ? k : hs;
          const int supp = warp_support<NQ>(S, lane), c2 = warp_cx_bound<NQ>(S, shM[w], shF[w], sp, lane);
          if (combine(h, 0, c2, supp) > sp.depth - t) alive = false;
          else {
            const int T = 2 * (NQ + k) <= 64 ? warp_wht<NQ, uint32_t>(S, k, lane) : warp_wht<NQ, uint64_t>(S, k, lane);
            alive = combine(h, T, c2, supp) <= sp.depth - t;
          }
        }
        if (alive) {
          uint64_t a = 0, b = 0;
          for (int e = lane; e < E; e += 32) hash_entry(e, S[e], a, b);
          for (int m = 16; m; m >>= 1) { a += __shfl_xor_sync(FULL_MASK, a, m); b += __shfl_xor_sync(FULL_MASK, b, m); }
          if (!lane) hash_final(k, a, b, h1[o], h2[o]);
        }
        s = alive ? ALIVE : DEAD;
      }
    }
    if (!lane) status[o] = s;
    __syncwarp();
  }
}

static void cuda_check(const char *what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error after %s: %s\n", what, cudaGetErrorString(e)); std::exit(5); }
}
#endif

struct IsStatus { uint8_t v; HD bool operator()(uint8_t s) const { return s == v; } };

struct MakeChild {
  const Node *parents;
  const uint64_t *h1, *h2;
  int G;
  HD Node operator()(int o) const {
    Node c = parents[o / G];
    c.seq[c.len++] = (uint8_t)(o % G);
    c.h1 = h1[o]; c.h2 = h2[o];
    return c;
  }
};

template <int NQ> struct SameResidual {   // exact: hashes equal and the residual matrices equal
  Spec sp;
  HD bool operator()(const Node &x, const Node &y) const {
    if (x.h1 != y.h1 || x.h2 != y.h2) return false;
    Mat<NQ> A, B;
    residual(A, sp, x.seq, x.len);
    residual(B, sp, y.seq, y.len);
    if (A.k != B.k) return false;
    for (int i = 0; i < (1 << NQ) * (1 << NQ); i++) if (!zeq32(A.e[i], B.e[i])) return false;
    return true;
  }
};

// ---------------- host ----------------
struct HostSpec {
  int n = 0, depth = 0, standard = 0, G = 0, tk = 0, lite = 0, tdiv = 1;
  std::vector<std::string> names;
  std::vector<Gate> gates;
  std::vector<uint8_t> bad2;
  std::vector<uint32_t> bad3;
  std::vector<Zw> target;
  std::vector<int> cutmask, cutsize, rad, cxtable;
  int states = 0, m = 0, gk = 0;
  std::vector<Zw> start, goal;
  std::vector<int> ginfo;
};

static HostSpec read_spec(std::istream &in) {
  HostSpec s;
  std::string tok;
  in >> tok;
  if (tok != "cudasynth-spec") { std::cerr << "not a spec\n"; std::exit(2); }
  int version; in >> version;
  while (in >> tok) {
    if (tok == "n") in >> s.n;
    else if (tok == "mode" || tok == "phase") {
      std::string v; in >> v;
      if (v == "states") s.states = 1;
      else if (v != "unitary" && v != "exact") { std::cerr << "unsupported " << tok << " " << v << "\n"; std::exit(2); }
    }
    else if (tok == "pairs") in >> s.m;
    else if (tok == "start") {
      in >> s.tk; const int N = 1 << s.n; s.start.resize(N * N);
      for (auto &x : s.start) in >> x.a >> x.b >> x.c >> x.d;
    } else if (tok == "goal") {
      int m; in >> m; s.ginfo.resize(m * (2 + s.cutmask.size()));
      for (auto &v : s.ginfo) in >> v;
    }
    else if (tok == "depth") in >> s.depth;
    else if (tok == "standard") in >> s.standard;
    else if (tok == "lite") in >> s.lite;
    else if (tok == "tdiv") in >> s.tdiv;
    else if (tok == "gates") {
      in >> s.G;
      if (s.G > GMAX) { std::cerr << "too many gates\n"; std::exit(2); }
      for (int g = 0; g < s.G; g++) {
        std::string name; Gate gt{}; int gk;
        in >> name >> gt.nq;
        for (int j = 0; j < gt.nq; j++) in >> gt.q[j];
        const int L = 1 << gt.nq;
        in >> gk;
        Zw fwd[LMAX * LMAX];
        for (int i = 0; i < L * L; i++) in >> fwd[i].a >> fwd[i].b >> fwd[i].c >> fwd[i].d;
        in >> gt.ik;
        for (int i = 0; i < L * L; i++) in >> gt.inv[i].a >> gt.inv[i].b >> gt.inv[i].c >> gt.inv[i].d;
        if (s.states) {          // rows are images p|in_j> transposed: W <- W g^T, so multiply by the transpose
          gt.ik = gk;
          for (int a = 0; a < L; a++) for (int b = 0; b < L; b++) gt.inv[a * L + b] = fwd[b * L + a];
        }
        s.names.push_back(name); s.gates.push_back(gt);
      }
    } else if (tok == "bad2") {
      int m; in >> m; s.bad2.assign(s.G * s.G, 0);
      for (int i = 0; i < m; i++) { int a, b; in >> a >> b; s.bad2[a * s.G + b] = 1; }
    } else if (tok == "bad3") {
      int m; in >> m; s.bad3.assign((s.G * s.G * s.G + 31) / 32, 0);
      for (int i = 0; i < m; i++) { int a, b, c; in >> a >> b >> c; const uint32_t t = (a * s.G + b) * s.G + c; s.bad3[t >> 5] |= 1u << (t & 31); }
    } else if (tok == "target") {
      int k; in >> k; const int N = 1 << s.n; std::vector<Zw> M(N * N);
      for (auto &x : M) in >> x.a >> x.b >> x.c >> x.d;
      if (s.states) { s.goal = M; s.gk = k; }
      else {                   // unitary: the residual starts at U and must reach I
        s.target = M; s.tk = k; s.gk = 0; s.goal.assign(N * N, Zw{0, 0, 0, 0});
        for (int i = 0; i < N; i++) s.goal[i * N + i] = Zw{1, 0, 0, 0};
      }
    } else if (tok == "cuts") {
      int m; in >> m;
      for (int i = 0; i < m; i++) {
        int sz, mask = 0; in >> sz;
        for (int j = 0; j < sz; j++) { int q; in >> q; mask |= 1 << q; }
        s.cutmask.push_back(mask); s.cutsize.push_back(sz); s.rad.push_back(2 * sz + 1);
      }
    } else if (tok == "cxtable") {
      int m; in >> m; s.cxtable.resize(m); for (auto &v : s.cxtable) in >> v;
    } else if (tok == "end") break;
    else { std::cerr << "unknown spec field " << tok << "\n"; std::exit(2); }
  }
  if (s.states) {
    if (s.start.empty() || s.goal.empty() || s.ginfo.empty() || s.m < 1 || s.m > (1 << s.n)) { std::cerr << "states spec incomplete\n"; std::exit(2); }
    s.target = s.start;
  }
  if (s.bad2.empty()) s.bad2.assign(s.G * s.G, 0);
  if (s.bad3.empty()) s.bad3.assign((s.G * s.G * s.G + 31) / 32, 0);
  return s;
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static bool g_prof = false;            // --profile: per-pass seconds on stderr (synchronises after each pass)
static double g_pt[8];
static const char *g_pn[8] = {"residual", "cheap", "T", "compact", "dedupe", "", "", ""};
struct Pass {
  int i; double t0;
  explicit Pass(int i_) : i(i_), t0(g_prof ? now() : 0) {}
  ~Pass() {
    if (!g_prof) return;
#ifdef __CUDACC__
    cudaDeviceSynchronize();
#endif
    g_pt[i] += now() - t0;
  }
};
static void prof_report() {
  if (!g_prof) return;
  for (int i = 0; i < 8; i++) if (g_pn[i][0]) std::fprintf(stderr, "  %-9s %.3f s\n", g_pn[i], g_pt[i]);
}

template <int NQ> static int run(const HostSpec &hs, int depth, bool dedupe, long chunk) {
  thrust::device_vector<Gate> d_gates(hs.gates.begin(), hs.gates.end());
  thrust::device_vector<uint8_t> d_bad2(hs.bad2.begin(), hs.bad2.end());
  thrust::device_vector<uint32_t> d_bad3(hs.bad3.begin(), hs.bad3.end());
  thrust::device_vector<Zw> d_target(hs.target.begin(), hs.target.end());
  thrust::device_vector<int> d_cx(hs.cxtable.empty() ? std::vector<int>{0} : hs.cxtable);
  thrust::device_vector<Zw> d_goal(hs.goal.begin(), hs.goal.end());
  thrust::device_vector<int> d_ginfo(hs.ginfo.empty() ? std::vector<int>{0} : hs.ginfo);
  Spec sp{};
  sp.n = hs.n; sp.N = 1 << hs.n; sp.G = hs.G; sp.depth = depth; sp.standard = hs.standard;
  sp.ncuts = (int)hs.cutmask.size();
  for (int i = 0; i < sp.ncuts; i++) { sp.rad[i] = hs.rad[i]; sp.cutmask[i] = hs.cutmask[i]; sp.cutsize[i] = hs.cutsize[i]; }
  sp.have_table = !hs.cxtable.empty();
  sp.w8 = powmod(3, (PRIME - 1) / 8);
  sp.gates = thrust::raw_pointer_cast(d_gates.data());
  sp.bad2 = thrust::raw_pointer_cast(d_bad2.data());
  sp.bad3 = thrust::raw_pointer_cast(d_bad3.data());
  sp.target = thrust::raw_pointer_cast(d_target.data());
  sp.tk = hs.tk;
  sp.cxtable = thrust::raw_pointer_cast(d_cx.data());
  sp.lite = hs.lite; sp.tdiv = hs.tdiv < 1 ? 1 : hs.tdiv;
  sp.states = hs.states; sp.m = hs.states ? hs.m : (1 << hs.n); sp.gk = hs.gk;
  sp.goal = thrust::raw_pointer_cast(d_goal.data());
  sp.ginfo = thrust::raw_pointer_cast(d_ginfo.data());

  thrust::device_vector<Node> frontier(1);
  { Node root{}; frontier[0] = root; }
  {   // depth 0: is the target the identity?
    Mat<NQ> S; S.ovf = 0; S.k = hs.tk;
    for (int i = 0; i < (1 << NQ) * (1 << NQ); i++) S.e[i] = st(hs.target[i], S.ovf);
    if (S.ovf) { std::fprintf(stderr, "target coefficients exceed int32: rebuild with -DSTORE64\n"); return 4; }
    bool done = S.k == hs.gk;
    for (int i = 0; done && i < (1 << NQ) * (1 << NQ); i++) done = zeq(ld(S.e[i]), hs.goal[i]);
    if (done) { std::printf("{\"found\": 0, \"circuit\": []}\n"); return 0; }
  }
  for (int t = 1; t <= depth; t++) {
    const double t0 = now();
    const long m = (long)frontier.size();
    thrust::device_vector<Node> next;
    long n_w2 = 0, n_w3 = 0, n_dead = 0, n_alive = 0, n_found = 0;
    std::vector<Node> found;
    for (long off = 0; off < m; off += chunk) {
      const long cm = std::min(chunk, m - off);
      const Node *par = thrust::raw_pointer_cast(frontier.data()) + off;
      const int nc = (int)(cm * hs.G);
      thrust::device_vector<Mat<NQ>> R(cm);                            // pass 0
      { Pass p_(0);
      thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)cm),
                       ParentResidual<NQ>{sp, par, thrust::raw_pointer_cast(R.data())}); }
      thrust::device_vector<uint8_t> status(nc);
      thrust::device_vector<uint64_t> h1(nc), h2(nc);
      uint8_t *st_ = thrust::raw_pointer_cast(status.data());
      uint64_t *h1_ = thrust::raw_pointer_cast(h1.data()), *h2_ = thrust::raw_pointer_cast(h2.data());
#ifdef __CUDACC__
      {   // passes 1 + 2 fused, one warp per child
        Pass p_(1);
        const long blocks = std::min(((long)nc + WPB - 1) / WPB, 65535L * 4);
        if (sp.states) level_warp<NQ, true><<<(unsigned)blocks, 32 * WPB>>>(sp, par, thrust::raw_pointer_cast(R.data()), nc, st_, h1_, h2_, t);
        else level_warp<NQ, false><<<(unsigned)blocks, 32 * WPB>>>(sp, par, thrust::raw_pointer_cast(R.data()), nc, st_, h1_, h2_, t);
        cuda_check("level kernel");
      }
      if (thrust::count(status.begin(), status.end(), (uint8_t)OVF)) {
        std::fprintf(stderr, "residual coefficients exceed the storage type: rebuild with -DSTORE64\n");
        return 4;
      }
#else
      thrust::device_vector<int> cheap(nc);
      { Pass p_(1);
      thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(nc),   // pass 1
                       Cheap<NQ>{sp, par, thrust::raw_pointer_cast(R.data()), st_, thrust::raw_pointer_cast(cheap.data()), h1_, h2_, t}); }
      if (thrust::count(status.begin(), status.end(), (uint8_t)OVF)) {
        std::fprintf(stderr, "residual coefficients exceed the storage type: rebuild with -DSTORE64\n");
        return 4;
      }
      const long nt = thrust::count(status.begin(), status.end(), (uint8_t)NEEDT);
      if (nt) {                                                        // pass 2 on the compacted NEEDT children
        Pass p_(2);
        thrust::device_vector<int> idx(nt);
        thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(nc), status.begin(),
                        idx.begin(), IsStatus{NEEDT});
        thrust::for_each(idx.begin(), idx.end(), Full<NQ>{sp, thrust::raw_pointer_cast(R.data()), st_,
                                                          thrust::raw_pointer_cast(cheap.data()), h1_, h2_, t});
      }
#endif
      Pass p_(3);
      n_w2 += thrust::count(status.begin(), status.end(), (uint8_t)W2CUT);
      n_w3 += thrust::count(status.begin(), status.end(), (uint8_t)W3CUT);
      n_dead += thrust::count(status.begin(), status.end(), (uint8_t)DEAD);
      const long nf = thrust::count(status.begin(), status.end(), (uint8_t)FOUND);
      MakeChild mk{thrust::raw_pointer_cast(frontier.data()) + off, thrust::raw_pointer_cast(h1.data()),
                   thrust::raw_pointer_cast(h2.data()), hs.G};
      if (nf) {
        n_found += nf;
        thrust::device_vector<int> idx(nf);
        thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)(cm * hs.G)),
                        status.begin(), idx.begin(), IsStatus{FOUND});
        thrust::host_vector<int> hidx(idx);
        thrust::host_vector<Node> par(frontier.begin() + off, frontier.begin() + off + cm);
        for (int o : hidx) { Node c = par[o / hs.G]; c.seq[c.len++] = (uint8_t)(o % hs.G); found.push_back(c); }
      }
      const long na = thrust::count(status.begin(), status.end(), (uint8_t)ALIVE);
      n_alive += na;
      if (na && found.empty() && t < depth) {          // last level: survivors are never expanded
        thrust::device_vector<int> idx(na);
        thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)(cm * hs.G)),
                        status.begin(), idx.begin(), IsStatus{ALIVE});
        const long old = (long)next.size();
        next.resize(old + na);
        thrust::transform(idx.begin(), idx.end(), next.begin() + old, mk);
      }
    }
    if (!found.empty()) {
      std::sort(found.begin(), found.end(), [](const Node &a, const Node &b) {
        for (int i = 0; i < a.len; i++) if (a.seq[i] != b.seq[i]) return a.seq[i] < b.seq[i];
        return false; });
      std::printf("{\"t\": %d, \"found\": %ld, \"circuit\": [", t, n_found);
      for (int i = 0; i < found[0].len; i++) std::printf("%s\"%s\"", i ? ", " : "", hs.names[found[0].seq[i]].c_str());
      std::printf("], \"seconds\": %.2f}\n", now() - t0);
      return 0;
    }
    long distinct = (long)next.size();
    if (dedupe && distinct > 1) {
      Pass p_(4);
      thrust::sort(next.begin(), next.end(), NodeLess{});
      auto end = thrust::unique(next.begin(), next.end(), SameResidual<NQ>{sp});
      next.erase(end, next.end());
      distinct = (long)next.size();
    }
    std::printf("{\"t\": %d, \"parents\": %ld, \"prefixes\": %ld, \"cut_w3\": %ld, \"cut_lb\": %ld, \"survivors\": %ld, "
                "\"distinct\": %ld, \"proved_unsat\": %s, \"seconds\": %.2f}\n",
                t, m, m * hs.G - n_w2, n_w3, n_dead, n_alive, distinct, next.empty() ? "true" : "false", now() - t0);
    std::fflush(stdout);
    if (next.empty()) return 1;
    frontier.swap(next);
  }
  // every prefix of <= depth gates was checked against the goal (pruning, windows and dedupe are sound), so a
  // frontier left at the last level is still a proof: no circuit of <= depth gates
  std::printf("{\"depth\": %d, \"proved_unsat\": true, \"frontier\": %ld}\n", depth, (long)frontier.size());
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: cudasynth SPEC [--depth D] [--no-dedupe] [--chunk PARENTS]\n"
                         "exit 0: circuit found, 1: no circuit of <= depth gates (proof)\n");
    return 3;
  }
  std::ifstream f(argv[1]);
  HostSpec hs = read_spec(f);
  int depth = hs.depth; bool dedupe = true; long chunk = 1 << 17;
  for (int i = 2; i < argc; i++) {
    if (!std::strcmp(argv[i], "--depth") && i + 1 < argc) depth = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-dedupe")) dedupe = false;
    else if (!std::strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = std::atol(argv[++i]);
    else if (!std::strcmp(argv[i], "--profile")) g_prof = true;
  }
  if (depth > DMAX) { std::fprintf(stderr, "depth > %d\n", DMAX); return 3; }
  std::fprintf(stderr, "cudasynth: n %d, %d candidates, depth %d, dedupe %d\n", hs.n, hs.G, depth, dedupe);
#ifdef __CUDACC__
  if (hs.states) {   // the GPU state-target kernel aborts on some inputs (under investigation); CPU path validated
    std::fprintf(stderr, "state targets: GPU path not ready yet, use build/cudasynth-cpu\n");
    return 3;
  }
#endif
  int rc = 3;
  switch (hs.n) {   // qubit counts compiled in: NQ_MIN..NQ_MAX (each one is a full instantiation; nvcc is slow)
#if NQ_MIN <= 1 && NQ_MAX >= 1
    case 1: rc = run<1>(hs, depth, dedupe, chunk); break;
#endif
#if NQ_MIN <= 2 && NQ_MAX >= 2
    case 2: rc = run<2>(hs, depth, dedupe, chunk); break;
#endif
#if NQ_MIN <= 3 && NQ_MAX >= 3
    case 3: rc = run<3>(hs, depth, dedupe, chunk); break;
#endif
#if NQ_MIN <= 4 && NQ_MAX >= 4
    case 4: rc = run<4>(hs, depth, dedupe, chunk); break;
#endif
    default: std::fprintf(stderr, "n = %d not built (NQ_MIN..NQ_MAX)\n", hs.n); return 3;
  }
  prof_report();
  return rc;
}
