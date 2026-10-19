// cheese: exact Clifford+T synthesis by level-synchronous branch-and-bound (docs/PLAN.md).
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
#include <thrust/binary_search.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/functional.h>
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
  int nsym, nogoal;                // symmetry group size (1: off); nogoal: never report the goal (table build)
  const uint8_t *symperm;          // nsym x N: basis-index map of each group element (a qubit relabelling)
  const uint8_t *symconj;          // nsym: element also conjugates entries (T <-> Tdg, S <-> Sdg)
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

// Symmetry classes. Group element s maps S to S' with S'[pi(r)][pi(c)] = S[r][c] (qubit relabelling pi, as a
// basis-index map), entries conjugated if symconj[s]. The remaining gate count is the same for every member of
// a class when the candidates are closed under relabelling (and conjugation). The class key is the smallest
// hash over the orbit; element 0 is the identity, so with nsym = 1 this is hash2.
HD inline Zw32 zconj32(Zw32 x) { return {x.a, (StoreT)-x.d, (StoreT)-x.c, (StoreT)-x.b}; }
HD inline bool hless(uint64_t a1, uint64_t a2, uint64_t b1, uint64_t b2) { return a1 != b1 ? a1 < b1 : a2 < b2; }
template <int NQ> HD inline void class_hash(const Mat<NQ> &S, const Spec &sp, uint64_t &h1, uint64_t &h2) {
  constexpr int N = 1 << NQ;
  h1 = h2 = ~0ULL;
  for (int s = 0; s < sp.nsym; s++) {
    const uint8_t *pm = sp.symperm + s * N;
    uint64_t a = 0, b = 0, x1, x2;
    for (int e = 0; e < N * N; e++)
      hash_entry(pm[e >> NQ] * N + pm[e & (N - 1)], sp.symconj[s] ? zconj32(S.e[e]) : S.e[e], a, b);
    hash_final(S.k, a, b, x1, x2);
    if (hless(x1, x2, h1, h2)) { h1 = x1; h2 = x2; }
  }
}
template <int NQ> HD inline bool same_class(const Mat<NQ> &A, const Mat<NQ> &B, const Spec &sp) {
  constexpr int N = 1 << NQ;
  if (A.k != B.k) return false;
  for (int s = 0; s < sp.nsym; s++) {
    const uint8_t *pm = sp.symperm + s * N;
    bool eq = true;
    for (int e = 0; eq && e < N * N; e++) {
      const Zw32 v = sp.symconj[s] ? zconj32(A.e[e]) : A.e[e];
      eq = zeq32(v, B.e[pm[e >> NQ] * N + pm[e & (N - 1)]]);
    }
    if (eq) return true;
  }
  return false;
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
// The frontier is a tree: level l keeps, per node, its parent's index in level l-1 and its last gate (5 bytes);
// a prefix is read back by walking up. Children are generated parent-major, gate-minor, so with parents in
// lexicographic order every level is in lexicographic order too, and a node's index orders it.
struct Hist {
  const uint32_t *par[DMAX + 1];
  const uint8_t *gat[DMAX + 1];
};
HD inline int path(const Hist &H, int t, uint32_t i, uint8_t *seq) {
  for (int l = t; l >= 1; l--) { seq[l - 1] = H.gat[l][i]; i = H.par[l][i]; }
  return t;
}
HD inline void last2(const Hist &H, int tp, uint32_t pi, int &g1, int &g2) {   // the parent's last two gates
  g1 = tp >= 1 ? H.gat[tp][pi] : -1;
  g2 = tp >= 2 ? H.gat[tp - 1][H.par[tp][pi]] : -1;
}
HD inline bool window_cut(const Spec &sp, int g1, int g2, int g, uint8_t &s) {
  if (g1 >= 0 && sp.bad2[g1 * sp.G + g]) { s = 3; return true; }
  if (g2 >= 0) {
    const uint32_t tri = ((uint32_t)g2 * sp.G + g1) * sp.G + g;
    if ((sp.bad3[tri >> 5] >> (tri & 31)) & 1) { s = 4; return true; }
  }
  return false;
}
struct Key { uint64_t h1, h2; uint32_t idx; };
struct KeyLess {        // hash only
  HD bool operator()(const Key &x, const Key &y) const { return x.h1 != y.h1 ? x.h1 < y.h1 : x.h2 < y.h2; }
};
struct KeyLessIdx {     // hash, then index (= lexicographic prefix order)
  HD bool operator()(const Key &x, const Key &y) const {
    if (x.h1 != y.h1) return x.h1 < y.h1;
    if (x.h2 != y.h2) return x.h2 < y.h2;
    return x.idx < y.idx;
  }
};
struct IsTrue { HD bool operator()(bool b) const { return b; } };
struct KeyByIdx { HD bool operator()(const Key &x, const Key &y) const { return x.idx < y.idx; } };

enum : uint8_t { DEAD = 0, ALIVE = 1, FOUND = 2, W2CUT = 3, W3CUT = 4, NEEDT = 5, OVF = 6 };

template <int NQ> struct ParentResidual {   // pass 0: S = U p^-1 once per parent
  Spec sp;
  Hist H;
  int tp;
  uint32_t off;
  Mat<NQ> *R;
  HD void operator()(int i) const {
    uint8_t seq[DMAX];
    const int len = path(H, tp, off + i, seq);
    residual(R[i], sp, seq, len);
  }
};

// pass 1, one thread per child: windows, gate, identity, the cheap bounds (H, structure, support, CX). Children the
// cheap bound can't prune become NEEDT with (h, c2, support) packed in `cheap` for pass 2.
template <int NQ> struct Cheap {
  Spec sp;
  Hist H;
  int tp;
  uint32_t off;
  const Mat<NQ> *R;
  uint8_t *status;
  int *cheap;
  uint64_t *h1, *h2;
  int t;
  HD void operator()(int o) const {
    const int i = o / sp.G, g = o % sp.G;
    int g1, g2;
    last2(H, tp, off + i, g1, g2);
    uint8_t wc;
    if (window_cut(sp, g1, g2, g, wc)) { status[o] = wc; return; }
    Mat<NQ> C = R[i];
    apply_inv(C, sp.gates[g]);
    if (C.ovf || R[i].ovf) { status[o] = OVF; return; }
    if (is_goal(C, sp)) { status[o] = sp.nogoal ? DEAD : FOUND; return; }
    if (!sp.standard && !sp.lite) { status[o] = ALIVE; class_hash(C, sp, h1[o], h2[o]); return; }
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
    class_hash(C, sp, h1[o], h2[o]);
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
level_warp(Spec sp, Hist H, int tp, uint32_t off, const Mat<NQ> *R, long nc, uint8_t *status, uint64_t *h1, uint64_t *h2, int t) {
  constexpr int N = 1 << NQ, E = N * N;
  __shared__ Zw32 shS[WPB][E];
  __shared__ uint32_t shM[WPB][E], shF[WPB][16];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  Zw32 *S = shS[w];
  for (long o = (long)blockIdx.x * WPB + w; o < nc; o += (long)gridDim.x * WPB) {
    const int i = (int)(o / sp.G), g = (int)(o % sp.G);
    int g1, g2;
    last2(H, tp, off + i, g1, g2);
    uint8_t s;
    if (window_cut(sp, g1, g2, g, s)) {}
    else {
      bool ovf;
      const int k = warp_child<NQ>(S, R[i], sp.gates[g], lane, ovf);
      if (ovf) s = OVF;
      else if (warp_goal<NQ>(S, k, sp, lane)) s = sp.nogoal ? DEAD : FOUND;
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
        if (alive) {                                 // class key: smallest hash over the symmetry orbit
          uint64_t b1 = ~0ULL, b2 = ~0ULL;
          for (int sy = 0; sy < sp.nsym; sy++) {
            const uint8_t *pm = sp.symperm + sy * N;
            const bool cj = sp.symconj[sy];
            uint64_t a = 0, b = 0, x1, x2;
            for (int e = lane; e < E; e += 32) hash_entry(pm[e >> NQ] * N + pm[e & (N - 1)], cj ? zconj32(S[e]) : S[e], a, b);
            for (int m = 16; m; m >>= 1) { a += __shfl_xor_sync(FULL_MASK, a, m); b += __shfl_xor_sync(FULL_MASK, b, m); }
            hash_final(k, a, b, x1, x2);
            if (hless(x1, x2, b1, b2)) { b1 = x1; b2 = x2; }
          }
          if (!lane) { h1[o] = b1; h2[o] = b2; }
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

struct Emit {            // alive child o of the chunk -> parent, gate, key (index base + j)
  const int *idx;
  uint32_t off, base;
  int G;
  const uint64_t *h1, *h2;
  uint32_t *CP;
  uint8_t *CG;
  Key *CK;
  HD void operator()(int j) const {
    const int o = idx[j];
    CP[base + j] = off + o / G; CG[base + j] = (uint8_t)(o % G);
    CK[base + j] = Key{h1[o], h2[o], base + j};
  }
};

template <int NQ> struct SameChild {   // exact: hashes equal and the residual matrices equal
  Spec sp;
  Hist H;
  int tp;
  const uint32_t *CP;
  const uint8_t *CG;
  HD void child(uint32_t c, Mat<NQ> &A) const {
    uint8_t seq[DMAX];
    const int len = path(H, tp, CP[c], seq);
    seq[len] = CG[c];
    residual(A, sp, seq, len + 1);
  }
  HD bool operator()(const Key &x, const Key &y) const {
    if (x.h1 != y.h1 || x.h2 != y.h2) return false;
    Mat<NQ> A, B;
    child(x.idx, A);
    child(y.idx, B);
    return same_class(A, B, sp);
  }
};

struct Gather {          // kept keys -> the new level's parent / gate arrays and hashes
  const Key *K;
  const uint32_t *CP;
  const uint8_t *CG;
  uint32_t *P;
  uint8_t *Gt;
  uint64_t *F1, *F2;
  HD void operator()(int j) const { const uint32_t i = K[j].idx; P[j] = CP[i]; Gt[j] = CG[i]; F1[j] = K[j].h1; F2[j] = K[j].h2; }
};

struct DKey { uint64_t h1, h2; uint32_t idx, lvl; };
struct DKeyLess { HD bool operator()(const DKey &x, const DKey &y) const { return hless(x.h1, x.h2, y.h1, y.h2); } };
HD inline long dkey_lower(const DKey *K, long n, uint64_t h1, uint64_t h2) {   // first entry with hash >= (h1, h2)
  long lo = 0, hi = n;
  while (lo < hi) { const long mid = (lo + hi) / 2; if (hless(K[mid].h1, K[mid].h2, h1, h2)) lo = mid + 1; else hi = mid; }
  return lo;
}
template <int NQ> struct SeenBefore {    // child's class already at a lower level (hash match, then exact)
  Spec sp;
  Hist H;
  int tp;
  const uint32_t *CP;
  const uint8_t *CG;
  const DKey *seen;
  long ns;
  HD bool operator()(const Key &x) const {
    long i = dkey_lower(seen, ns, x.h1, x.h2);
    if (i >= ns || seen[i].h1 != x.h1 || seen[i].h2 != x.h2) return false;
    Mat<NQ> A, B;
    uint8_t seq[DMAX];
    int len = path(H, tp, CP[x.idx], seq);
    seq[len] = CG[x.idx];
    residual(A, sp, seq, len + 1);
    for (; i < ns && seen[i].h1 == x.h1 && seen[i].h2 == x.h2; i++) {
      len = path(H, seen[i].lvl, seen[i].idx, seq);
      residual(B, sp, seq, len);
      if (same_class(A, B, sp)) return true;
    }
    return false;
  }
};
struct ToDKey {
  const uint64_t *F1, *F2;
  uint32_t lvl;
  HD DKey operator()(int j) const { return {F1[j], F2[j], (uint32_t)j, lvl}; }
};

struct Tree {            // host handle of the search tree; level 0 is the empty prefix
  std::vector<thrust::device_vector<uint32_t>> P;
  std::vector<thrust::device_vector<uint8_t>> Gt;
  std::vector<long> n;
  thrust::device_vector<uint64_t> F1, F2;   // hashes of the top level's residuals
  bool filter = false;                      // distance table: drop classes already seen at a lower level
  thrust::device_vector<DKey> seen;         // every kept class so far, sorted by hash
  Tree() : P(1), Gt(1), n(1, 1) {}
  int top() const { return (int)n.size() - 1; }
  Hist hist() const {
    Hist H{};
    for (int l = 1; l <= top(); l++) { H.par[l] = thrust::raw_pointer_cast(P[l].data()); H.gat[l] = thrust::raw_pointer_cast(Gt[l].data()); }
    return H;
  }
  std::vector<int> seq(int t, uint32_t i) const {
    std::vector<int> s(t);
    for (int l = t; l >= 1; l--) { s[l - 1] = Gt[l][i]; i = P[l][i]; }
    return s;
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
  if (tok != "cheese-spec") { std::cerr << "not a spec\n"; std::exit(2); }
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

// levels 1..stop of the search on sp's target, pruned against budget depth, growing tree T. 0: a circuit
// (found_out, the lexicographically first at its level), 1: proof, 2: stopped at level stop (T.top() == stop)
template <int NQ> static int levels(const HostSpec &hs, const Spec &sp, int depth, int stop, bool dedupe, long chunk,
                                    Tree &T, std::vector<int> &found_out, long &n_found_out, const char *tag) {
  for (int t = T.top() + 1; t <= stop; t++) {
    const double t0 = now();
    const int tp = t - 1;
    const long m = T.n[tp];
    const Hist H = T.hist();
    const bool keep = t < depth;                       // last level: survivors are never expanded
    thrust::device_vector<uint32_t> CP;
    thrust::device_vector<uint8_t> CG;
    thrust::device_vector<Key> CK;
    long n_w2 = 0, n_w3 = 0, n_dead = 0, n_alive = 0, n_found = 0;
    std::vector<int> fseq;
    for (long off = 0; off < m; off += chunk) {
      const long cm = std::min(chunk, m - off);
      const int nc = (int)(cm * hs.G);
      thrust::device_vector<Mat<NQ>> R(cm);                            // pass 0
      { Pass p_(0);
      thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)cm),
                       ParentResidual<NQ>{sp, H, tp, (uint32_t)off, thrust::raw_pointer_cast(R.data())}); }
      thrust::device_vector<uint8_t> status(nc);
      thrust::device_vector<uint64_t> h1(nc), h2(nc);
      uint8_t *st_ = thrust::raw_pointer_cast(status.data());
      uint64_t *h1_ = thrust::raw_pointer_cast(h1.data()), *h2_ = thrust::raw_pointer_cast(h2.data());
#ifdef __CUDACC__
      {   // passes 1 + 2 fused, one warp per child
        Pass p_(1);
        const long blocks = std::min(((long)nc + WPB - 1) / WPB, 65535L * 4);
        if (sp.states) level_warp<NQ, true><<<(unsigned)blocks, 32 * WPB>>>(sp, H, tp, (uint32_t)off, thrust::raw_pointer_cast(R.data()), nc, st_, h1_, h2_, t);
        else level_warp<NQ, false><<<(unsigned)blocks, 32 * WPB>>>(sp, H, tp, (uint32_t)off, thrust::raw_pointer_cast(R.data()), nc, st_, h1_, h2_, t);
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
                       Cheap<NQ>{sp, H, tp, (uint32_t)off, thrust::raw_pointer_cast(R.data()), st_, thrust::raw_pointer_cast(cheap.data()), h1_, h2_, t}); }
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
      if (nf) {
        if (fseq.empty()) {                            // first in generation order = lexicographically first
          const int o = (int)(thrust::find(status.begin(), status.end(), (uint8_t)FOUND) - status.begin());
          fseq = T.seq(tp, (uint32_t)(off + o / hs.G));
          fseq.push_back(o % hs.G);
        }
        n_found += nf;
      }
      const long na = thrust::count(status.begin(), status.end(), (uint8_t)ALIVE);
      n_alive += na;
      if (na && fseq.empty() && keep) {
        thrust::device_vector<int> idx(na);
        thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(nc), status.begin(),
                        idx.begin(), IsStatus{ALIVE});
        const uint32_t base = (uint32_t)CK.size();
        CP.resize(base + na); CG.resize(base + na); CK.resize(base + na);
        thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)na),
                         Emit{thrust::raw_pointer_cast(idx.data()), (uint32_t)off, base, hs.G, h1_, h2_,
                              thrust::raw_pointer_cast(CP.data()), thrust::raw_pointer_cast(CG.data()),
                              thrust::raw_pointer_cast(CK.data())});
      }
    }
    if (!fseq.empty()) { found_out = fseq; n_found_out = n_found; return 0; }
    long distinct = (long)CK.size();
    if (dedupe && distinct > 1) {
      Pass p_(4);
      thrust::sort(CK.begin(), CK.end(), KeyLessIdx{});
      auto end = thrust::unique(CK.begin(), CK.end(), SameChild<NQ>{sp, H, tp, thrust::raw_pointer_cast(CP.data()),
                                                                   thrust::raw_pointer_cast(CG.data())});
      CK.erase(end, CK.end());
      thrust::sort(CK.begin(), CK.end(), KeyByIdx{});
      distinct = (long)CK.size();
    }
    if (T.filter && distinct) {
      Pass p_(4);
      thrust::device_vector<bool> old(distinct);
      thrust::transform(CK.begin(), CK.end(), old.begin(),
                        SeenBefore<NQ>{sp, H, tp, thrust::raw_pointer_cast(CP.data()), thrust::raw_pointer_cast(CG.data()),
                                       thrust::raw_pointer_cast(T.seen.data()), (long)T.seen.size()});
      auto end = thrust::remove_if(CK.begin(), CK.end(), old.begin(), IsTrue{});
      CK.erase(end, CK.end());
      distinct = (long)CK.size();
    }
    std::printf("{%s\"t\": %d, \"parents\": %ld, \"prefixes\": %ld, \"cut_w3\": %ld, \"cut_lb\": %ld, \"survivors\": %ld, "
                "\"distinct\": %ld, \"proved_unsat\": %s, \"seconds\": %.2f}\n",
                tag, t, m, m * hs.G - n_w2, n_w3, n_dead, n_alive, distinct, distinct ? "false" : "true", now() - t0);
    std::fflush(stdout);
    // a proof: nothing survives, or the last level was checked (every prefix of <= depth gates compared to the goal)
    if (!distinct) return 1;
    T.P.emplace_back(distinct); T.Gt.emplace_back(distinct); T.n.push_back(distinct);
    T.F1.resize(distinct); T.F2.resize(distinct);
    thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)distinct),
                     Gather{thrust::raw_pointer_cast(CK.data()), thrust::raw_pointer_cast(CP.data()),
                            thrust::raw_pointer_cast(CG.data()), thrust::raw_pointer_cast(T.P.back().data()),
                            thrust::raw_pointer_cast(T.Gt.back().data()), thrust::raw_pointer_cast(T.F1.data()),
                            thrust::raw_pointer_cast(T.F2.data())});
    if (T.filter) {
      thrust::device_vector<DKey> nk(distinct), merged(T.seen.size() + distinct);
      thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)distinct), nk.begin(),
                        ToDKey{thrust::raw_pointer_cast(T.F1.data()), thrust::raw_pointer_cast(T.F2.data()), (uint32_t)t});
      thrust::sort(nk.begin(), nk.end(), DKeyLess{});
      thrust::merge(T.seen.begin(), T.seen.end(), nk.begin(), nk.end(), merged.begin(), DKeyLess{});
      T.seen.swap(merged);
    }
  }
  return 2;
}

// ---------------- symmetry group (host) ----------------
// Candidate names are L_<gate>_<q>[_<q>...]. An element: qubit relabelling perm (qubit i -> perm[i]) and an
// optional conjugation (t <-> tdg, s <-> sdg; x, z, h, cx, cz, ccx real). It is used only if it maps every
// candidate to a candidate, so every member of a class has the same remaining gate count.
static int sym_gate(const HostSpec &hs, const std::vector<int> &perm, bool conj, int g) {
  const std::string &nm = hs.names[g];
  std::vector<std::string> parts;
  size_t a = 0, b;
  while ((b = nm.find('_', a)) != std::string::npos) { parts.push_back(nm.substr(a, b - a)); a = b + 1; }
  parts.push_back(nm.substr(a));
  std::string gate = parts[1];
  if (conj) {
    if (gate == "t") gate = "tdg"; else if (gate == "tdg") gate = "t";
    else if (gate == "s") gate = "sdg"; else if (gate == "sdg") gate = "s";
    else if (gate != "x" && gate != "z" && gate != "h" && gate != "cx" && gate != "cz" && gate != "ccx") return -1;
  }
  std::vector<int> qs;
  for (size_t i = 2; i < parts.size(); i++) qs.push_back(perm[std::atoi(parts[i].c_str())]);
  if (gate == "ccx" && qs.size() == 3 && qs[0] > qs[1]) std::swap(qs[0], qs[1]);   // controls unordered
  std::string want = "L_" + gate;
  for (int q : qs) want += "_" + std::to_string(q);
  for (int h = 0; h < hs.G; h++) if (hs.names[h] == want) return h;
  return -1;
}
struct SymGroup {
  std::vector<std::vector<int>> perm;    // per element: qubit map
  std::vector<uint8_t> conj, bperm;      // bperm: nsym x N basis-index maps
  std::vector<std::vector<int>> gmap;    // per element: candidate -> candidate
};
static SymGroup sym_group(const HostSpec &hs, bool on) {
  const int n = hs.n, N = 1 << n;
  SymGroup g;
  std::vector<int> p(n);
  for (int i = 0; i < n; i++) p[i] = i;
  std::vector<std::vector<int>> perms;
  do perms.push_back(p); while (on && std::next_permutation(p.begin(), p.end()));
  for (int cj = 0; cj < (on ? 2 : 1); cj++)
    for (auto &pm : perms) {
      std::vector<int> gm(hs.G);
      bool ok = true;
      for (int c = 0; c < hs.G && ok; c++) ok = (gm[c] = sym_gate(hs, pm, cj, c)) >= 0;
      if (!ok) continue;
      g.perm.push_back(pm); g.conj.push_back((uint8_t)cj); g.gmap.push_back(gm);
      for (int x = 0; x < N; x++) {
        int y = 0;
        for (int i = 0; i < n; i++) if ((x >> i) & 1) y |= 1 << pm[i];
        g.bperm.push_back((uint8_t)y);
      }
    }
  return g;
}

// ---------------- distance table for the last k gates ----------------
// Every symmetry class of unitaries with at most k gates, with its exact gate count: a search from the
// identity without bounds (residual p^-1 of sequence p) that drops classes already seen at a lower level, so a
// class sits at the level of its gate count. Built once per candidate list, k and group, cached on disk. In
// the search, a residual whose class is in the table has exactly that remaining gate count, and one that is not
// needs more than k gates - so the search stops at level D - k.
static uint64_t fnv(uint64_t h, const void *p, size_t n) {
  const unsigned char *c = (const unsigned char *)p;
  for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 0x100000001b3ULL; }
  return h;
}
static std::string table_path(const HostSpec &hs, const SymGroup &grp, int k) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (auto &nm : hs.names) h = fnv(h, nm.data(), nm.size() + 1);
  h = fnv(h, &hs.n, sizeof hs.n); h = fnv(h, &k, sizeof k);
  h = fnv(h, grp.bperm.data(), grp.bperm.size()); h = fnv(h, grp.conj.data(), grp.conj.size());
  h = fnv(h, hs.bad2.data(), hs.bad2.size()); h = fnv(h, hs.bad3.data(), hs.bad3.size() * 4);
  char buf[64];
  std::snprintf(buf, sizeof buf, "build/table-n%d-k%d-%016llx.bin", hs.n, k, (unsigned long long)h);
  return buf;
}
static bool table_load(const std::string &path, Tree &D) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  int nl = 0;
  bool ok = std::fread(&nl, sizeof nl, 1, f) == 1 && nl >= 0 && nl <= DMAX;
  for (int l = 1; ok && l <= nl; l++) {
    long n = 0;
    ok = std::fread(&n, sizeof n, 1, f) == 1 && n > 0;
    if (!ok) break;
    std::vector<uint32_t> P(n); std::vector<uint8_t> G(n);
    ok = std::fread(P.data(), 4, n, f) == (size_t)n && std::fread(G.data(), 1, n, f) == (size_t)n;
    D.P.emplace_back(P.begin(), P.end()); D.Gt.emplace_back(G.begin(), G.end()); D.n.push_back(n);
  }
  long ns = 0;
  ok = ok && std::fread(&ns, sizeof ns, 1, f) == 1 && ns > 0;
  if (ok) {
    std::vector<DKey> K(ns);
    ok = std::fread(K.data(), sizeof(DKey), ns, f) == (size_t)ns;
    D.seen.assign(K.begin(), K.end());
  }
  std::fclose(f);
  return ok;
}
static void table_save(const std::string &path, const Tree &D) {
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) { std::fprintf(stderr, "cannot write %s (table not cached)\n", path.c_str()); return; }
  const int nl = D.top();
  std::fwrite(&nl, sizeof nl, 1, f);
  for (int l = 1; l <= nl; l++) {
    const long n = D.n[l];
    thrust::host_vector<uint32_t> P(D.P[l]); thrust::host_vector<uint8_t> G(D.Gt[l]);
    std::fwrite(&n, sizeof n, 1, f); std::fwrite(P.data(), 4, n, f); std::fwrite(G.data(), 1, n, f);
  }
  thrust::host_vector<DKey> K(D.seen);
  const long ns = (long)K.size();
  std::fwrite(&ns, sizeof ns, 1, f); std::fwrite(K.data(), sizeof(DKey), ns, f);
  std::fclose(f);
}

template <int NQ> static int table_get(const HostSpec &hs, const Spec &sp, const SymGroup &grp, int k, long chunk,
                                       const thrust::device_vector<Zw> &d_I, Tree &D) {
  constexpr int N = 1 << NQ;
  const std::string path = table_path(hs, grp, k);
  const double t0 = now();
  if (table_load(path, D)) {
    std::fprintf(stderr, "table %s: %ld classes (loaded, %.1f s)\n", path.c_str(), (long)D.seen.size(), now() - t0);
    return 0;
  }
  Spec sd = sp;                                  // from the identity, no bounds, never "found"
  sd.target = thrust::raw_pointer_cast(d_I.data()); sd.tk = 0;
  sd.standard = 0; sd.lite = 0; sd.states = 0; sd.nogoal = 1;
  Spec hsd = sd;                                 // the identity's class key, on the host
  std::vector<Zw> I(N * N, Zw{0, 0, 0, 0});
  for (int i = 0; i < N; i++) I[i * N + i] = Zw{1, 0, 0, 0};
  hsd.symperm = grp.bperm.data(); hsd.symconj = grp.conj.data();
  Mat<NQ> M; M.ovf = 0; M.k = 0;
  for (int i = 0; i < N * N; i++) M.e[i] = st(I[i], M.ovf);
  DKey root{0, 0, 0, 0};
  class_hash(M, hsd, root.h1, root.h2);
  D.filter = true;
  D.seen.assign(1, root);
  std::vector<int> found;
  long nf = 0;
  const int rc = levels<NQ>(hs, sd, k + 1, k, true, chunk, D, found, nf, "\"dir\": \"table\", ");
  if (rc == 4) return 4;
  std::fprintf(stderr, "table: %ld classes within %d gates (%.1f s), cached in %s\n", (long)D.seen.size(), D.top(),
               now() - t0, path.c_str());
  table_save(path, D);
  return 0;
}

struct NonNeg { HD bool operator()(long x) const { return x >= 0; } };
struct InTable {          // node j of the top level: index into the table keys of its class, or -1
  const uint64_t *F1, *F2;
  const DKey *K;
  long nk;
  int maxc;
  HD long operator()(int j) const {
    const long i = dkey_lower(K, nk, F1[j], F2[j]);
    for (long x = i; x < nk && K[x].h1 == F1[j] && K[x].h2 == F2[j]; x++) if ((int)K[x].lvl <= maxc) return x;
    return -1;
  }
};

// a circuit from the table: prefix (level t of T, or the target itself) + continuation for its residual S;
// candidates confirmed exactly (class member, then the whole circuit == U). Returns the shortest, or empty.
template <int NQ> static std::vector<int> table_hit(const HostSpec &hs, const Spec &hsp, const SymGroup &grp,
                                                    const std::vector<int> &inv, const Tree &T, int t, const Tree &D,
                                                    int maxc) {
  constexpr int N = 1 << NQ;
  thrust::host_vector<DKey> keys;               // lazily: only when a hash matched
  const long nt = T.n[t];
  std::vector<std::pair<int, long>> cand;        // (cost, node j)
  thrust::device_vector<long> pos(nt);
  thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)nt), pos.begin(),
                    InTable{thrust::raw_pointer_cast(T.F1.data()), thrust::raw_pointer_cast(T.F2.data()),
                            thrust::raw_pointer_cast(D.seen.data()), (long)D.seen.size(), maxc});
  const long nh = (long)thrust::count_if(pos.begin(), pos.end(), NonNeg{});
  if (!nh) return {};
  thrust::host_vector<long> hp(pos);
  keys = D.seen;
  for (long j = 0; j < nt; j++) if (hp[j] >= 0) cand.push_back({(int)keys[hp[j]].lvl, j});
  std::sort(cand.begin(), cand.end());
  Spec hI = hsp;
  std::vector<Zw> I(N * N, Zw{0, 0, 0, 0});
  for (int i = 0; i < N; i++) I[i * N + i] = Zw{1, 0, 0, 0};
  hI.target = I.data(); hI.tk = 0;
  for (auto &c : cand) {
    const std::vector<int> pre = T.seq(t, (uint32_t)c.second);
    std::vector<uint8_t> p8(pre.begin(), pre.end());
    Mat<NQ> S;
    residual(S, hsp, p8.data(), (int)p8.size());
    const DKey &k0 = keys[hp[c.second]];
    for (long x = hp[c.second]; x < (long)keys.size() && keys[x].h1 == k0.h1 && keys[x].h2 == k0.h2; x++) {
      if ((int)keys[x].lvl > maxc) continue;
      const std::vector<int> w = D.seq(keys[x].lvl, keys[x].idx);   // table sequence: its residual E = W^-1
      std::vector<uint8_t> w8(w.begin(), w.end());
      Mat<NQ> E;
      residual(E, hI, w8.data(), (int)w8.size());
      for (size_t sy = 0; sy < grp.conj.size(); sy++) {             // sigma(E) == S ?
        const uint8_t *pm = grp.bperm.data() + sy * N;
        bool eq = E.k == S.k;
        for (int e = 0; eq && e < N * N; e++)
          eq = zeq32(grp.conj[sy] ? zconj32(E.e[e]) : E.e[e], S.e[pm[e >> NQ] * N + pm[e & (N - 1)]]);
        if (!eq) continue;
        std::vector<int> full = pre;                                 // sequence with product E: w reversed, inverted
        for (int i = (int)w.size() - 1; i >= 0; i--) full.push_back(grp.gmap[sy][inv[w[i]]]);
        std::vector<uint8_t> f8(full.begin(), full.end());
        Mat<NQ> Z;
        residual(Z, hsp, f8.data(), (int)f8.size());
        bool ok = Z.k == 0;
        for (int r = 0; ok && r < N; r++)
          for (int cc = 0; ok && cc < N; cc++) {
            const Zw32 v = Z.e[r * N + cc];
            ok = r == cc ? (v.a == 1 && !v.b && !v.c && !v.d) : zzero32(v);
          }
        if (ok) return full;
      }
    }
  }
  return {};
}

// ---------------- meet in the middle ----------------
// Budget D = a + b. Forward: the search on U, pruned against D, up to level a: residuals R = U p^-1. Backward: the
// same search on U^dag, levels 1..b: a node q is a suffix s reversed and inverted (q = s^-1), its residual
// R' = U^dag q^-1 = U^dag s. A circuit p then s of length a + |q| exists iff R = U R' for some pair (s = U p^-1).
// Circuits shorter than a are found by the forward levels themselves. Every circuit of <= D gates has a
// canonical prefix and suffix in the two trees (windows, bounds and dedupe are sound on both sides), so no match
// is a proof. The backward tree itself is the backward table.
struct FwdKey {
  const uint64_t *F1, *F2;
  HD Key operator()(int i) const { return {F1[i], F2[i], (uint32_t)i}; }
};
template <int NQ> HD inline void mul_left(const Zw *U, int ku, const Mat<NQ> &R, Mat<NQ> &K) {   // K = U R
  constexpr int N = 1 << NQ;
  K.ovf = R.ovf; K.k = ku + R.k;
  for (int r = 0; r < N; r++)
    for (int c = 0; c < N; c++) {
      Zw acc{0, 0, 0, 0};
      for (int m = 0; m < N; m++) {
        const Zw u = U[r * N + m];
        if (!zzero(u)) acc = zadd(acc, zmul(u, ld(R.e[m * N + c])));
      }
      K.e[r * N + c] = st(acc, K.ovf);
    }
  normalise(K);
}
template <int NQ> struct BwdKey {    // flat index over backward levels 1..top -> hash of U R'
  Spec spB;
  Hist H;
  long off[DMAX + 2];
  int top;
  const Zw *U;
  int ku;
  int *ovf;
  HD Key operator()(long j) const {
    int l = 1;
    while (l < top && j >= off[l + 1]) l++;
    uint8_t seq[DMAX];
    const int len = path(H, l, (uint32_t)(j - off[l]), seq);
    Mat<NQ> R, K;
    residual(R, spB, seq, len);
    mul_left(U, ku, R, K);
    if (K.ovf) *ovf = 1;
    Key key{0, 0, (uint32_t)j};
    hash2(K, key.h1, key.h2);
    return key;
  }
};

static std::vector<int> inverse_gates(const HostSpec &hs) {   // candidate index of each gate's inverse
  std::vector<int> inv(hs.G, -1);
  for (int g = 0; g < hs.G; g++) {
    std::string nm = hs.names[g];
    const size_t a = nm.find('_'), b = nm.find('_', a + 1);
    std::string gate = nm.substr(a + 1, b - a - 1);
    const char *pairs[4][2] = {{"t", "tdg"}, {"tdg", "t"}, {"s", "sdg"}, {"sdg", "s"}};
    for (auto &pr : pairs) if (gate == pr[0]) { gate = pr[1]; break; }
    const std::string want = nm.substr(0, a + 1) + gate + nm.substr(b);
    for (int h = 0; h < hs.G; h++) if (hs.names[h] == want) inv[g] = h;
    if (inv[g] < 0) { std::fprintf(stderr, "--mitm: no inverse candidate for %s\n", nm.c_str()); std::exit(3); }
  }
  return inv;
}

static void print_circuit(const HostSpec &hs, const std::vector<int> &c, long n_found, const char *extra, double t0) {
  std::printf("{\"t\": %d, \"found\": %ld, \"circuit\": [", (int)c.size(), n_found ? n_found : 1L);
  for (size_t i = 0; i < c.size(); i++) std::printf("%s\"%s\"", i ? ", " : "", hs.names[c[i]].c_str());
  std::printf("]%s, \"seconds\": %.2f}\n", extra, now() - t0);
}

template <int NQ> static int mitm_join(const HostSpec &hs, const Spec &sp, int depth, bool dedupe, long chunk,
                                       const thrust::device_vector<Zw> &d_U, double t0) {
  constexpr int N = 1 << NQ;
  if (hs.states) { std::fprintf(stderr, "--mitm: unitary targets only\n"); return 3; }
  const std::vector<int> inv = inverse_gates(hs);
  std::vector<int> found;
  long n_found = 0;
  std::vector<Zw> Ud(N * N);                                   // U^dag, same sqrt2 exponent
  for (int r = 0; r < N; r++) for (int c = 0; c < N; c++) Ud[r * N + c] = zconj(hs.target[c * N + r]);
  thrust::device_vector<Zw> d_Ud(Ud.begin(), Ud.end());
  Spec spB = sp;
  spB.target = thrust::raw_pointer_cast(d_Ud.data());
  // grow the side with the smaller frontier (the next level costs about frontier x candidates) until the depths
  // add up to D. A dead forward side is a proof; a dead backward side at level b+1 means no suffix is longer than
  // b, so the join with levels 1..b is already complete.
  Tree TF, TB;
  bool bdead = false;
  while (TF.top() + TB.top() < depth) {
    const bool fwd = bdead || TF.n[TF.top()] <= TB.n[TB.top()];
    Tree &T = fwd ? TF : TB;
    n_found = 0;
    const int rc = levels<NQ>(hs, fwd ? sp : spB, depth, T.top() + 1, dedupe, chunk, T, found, n_found,
                              fwd ? "\"dir\": \"fwd\", " : "\"dir\": \"bwd\", ");
    if (rc == 0) {
      if (fwd) { print_circuit(hs, found, n_found, "", t0); return 0; }
      std::vector<int> c;                                      // a circuit q for U^dag: U = q^-1
      for (int i = (int)found.size() - 1; i >= 0; i--) c.push_back(inv[found[i]]);
      print_circuit(hs, c, n_found, "", t0);
      return 0;
    }
    if (rc == 4) return 4;
    if (rc == 1) { if (fwd) return 1; bdead = true; break; }
  }
  const int a = TF.top(), b = TB.top();

  const double tj = now();
  const long nf = TF.n[a];
  const int top = TB.top();
  BwdKey<NQ> bk{spB, TB.hist(), {}, top, thrust::raw_pointer_cast(d_U.data()), hs.tk, nullptr};
  long nb = 0;
  for (int l = 1; l <= top; l++) { bk.off[l] = nb; nb += TB.n[l]; }
  bk.off[top + 1] = nb;
  if (!nb) { std::printf("{\"depth\": %d, \"proved_unsat\": true, \"mitm\": [%d, %d]}\n", depth, a, b); return 1; }
  thrust::device_vector<Key> FK(nf), BK(nb);
  thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)nf), FK.begin(),
                    FwdKey{thrust::raw_pointer_cast(TF.F1.data()), thrust::raw_pointer_cast(TF.F2.data())});
  thrust::sort(FK.begin(), FK.end(), KeyLess{});
  thrust::device_vector<int> d_ovf(1, 0);
  bk.ovf = thrust::raw_pointer_cast(d_ovf.data());
  thrust::transform(thrust::counting_iterator<long>(0), thrust::counting_iterator<long>(nb), BK.begin(), bk);
  if (d_ovf[0]) { std::fprintf(stderr, "--mitm: coefficients exceed the storage type: rebuild with -DSTORE64\n"); return 4; }
  thrust::device_vector<bool> hit(nb);
  thrust::binary_search(FK.begin(), FK.end(), BK.begin(), BK.end(), hit.begin(), KeyLess{});
  const long nh = thrust::count(hit.begin(), hit.end(), true);
  std::printf("{\"dir\": \"join\", \"forward\": %ld, \"backward\": %ld, \"hash_matches\": %ld, \"seconds\": %.2f}\n",
              nf, nb, nh, now() - tj);
  std::fflush(stdout);
  if (nh) {                                                    // confirm candidates exactly on the host
    thrust::device_vector<Key> HB(nh);
    thrust::copy_if(BK.begin(), BK.end(), hit.begin(), HB.begin(), IsTrue{});
    thrust::device_vector<int> pos(nh);
    thrust::lower_bound(FK.begin(), FK.end(), HB.begin(), HB.end(), pos.begin(), KeyLess{});
    thrust::host_vector<Key> hb(HB);
    thrust::host_vector<int> hp(pos);
    Spec hF = sp, hB = sp;                                     // host copies for residual()
    hF.gates = hs.gates.data(); hF.target = hs.target.data();
    hB.gates = hs.gates.data(); hB.target = Ud.data();
    for (long j = 0; j < nh; j++) {
      const long fj = hb[j].idx;
      int l = 1;
      while (l < top && fj >= bk.off[l + 1]) l++;
      const std::vector<int> q = TB.seq(l, (uint32_t)(fj - bk.off[l]));
      std::vector<uint8_t> q8(q.begin(), q.end());
      Mat<NQ> Rq, K;
      residual(Rq, hB, q8.data(), (int)q8.size());
      mul_left(hs.target.data(), hs.tk, Rq, K);
      for (long f = hp[j]; f < nf; f++) {
        const Key fk = FK[f];
        if (fk.h1 != hb[j].h1 || fk.h2 != hb[j].h2) break;
        std::vector<int> c = TF.seq(a, fk.idx);
        std::vector<uint8_t> p8(c.begin(), c.end());
        Mat<NQ> R;
        residual(R, hF, p8.data(), (int)p8.size());
        bool eq = R.k == K.k;
        for (int i = 0; eq && i < N * N; i++) eq = zeq32(R.e[i], K.e[i]);
        if (!eq) continue;
        for (int i = (int)q.size() - 1; i >= 0; i--) c.push_back(inv[q[i]]);
        char extra[64];
        std::snprintf(extra, sizeof extra, ", \"mitm\": [%d, %d]", a, (int)q.size());
        print_circuit(hs, c, 1, extra, t0);
        return 0;
      }
    }
  }
  std::printf("{\"depth\": %d, \"proved_unsat\": true, \"mitm\": [%d, %d]}\n", depth, a, b);
  return 1;
}

template <int NQ> static int run(const HostSpec &hs, int depth, bool dedupe, long chunk, int mitm, bool sym, int tk) {
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
  // symmetry classes: plain unitary search only (the meet-in-the-middle join needs exact residuals)
  const SymGroup grp = sym_group(hs, sym && !mitm && !hs.states);
  thrust::device_vector<uint8_t> d_symperm(grp.bperm.begin(), grp.bperm.end()), d_symconj(grp.conj.begin(), grp.conj.end());
  sp.nsym = (int)grp.conj.size(); sp.nogoal = 0;
  sp.symperm = thrust::raw_pointer_cast(d_symperm.data());
  sp.symconj = thrust::raw_pointer_cast(d_symconj.data());
  std::fprintf(stderr, "symmetry classes: %d group elements\n", sp.nsym);

  {   // depth 0: is the target the identity?
    Mat<NQ> S; S.ovf = 0; S.k = hs.tk;
    for (int i = 0; i < (1 << NQ) * (1 << NQ); i++) S.e[i] = st(hs.target[i], S.ovf);
    if (S.ovf) { std::fprintf(stderr, "target coefficients exceed int32: rebuild with -DSTORE64\n"); return 4; }
    bool done = S.k == hs.gk;
    for (int i = 0; done && i < (1 << NQ) * (1 << NQ); i++) done = zeq(ld(S.e[i]), hs.goal[i]);
    if (done) { std::printf("{\"found\": 0, \"circuit\": []}\n"); return 0; }
  }
  const double t0 = now();
  if (mitm) return mitm_join<NQ>(hs, sp, depth, dedupe, chunk, d_target, t0);
  if (tk > 0 && !hs.states) {
    constexpr int N = 1 << NQ;
    std::vector<Zw> I(N * N, Zw{0, 0, 0, 0});
    for (int i = 0; i < N; i++) I[i * N + i] = Zw{1, 0, 0, 0};
    thrust::device_vector<Zw> d_I(I.begin(), I.end());
    Tree D;
    if (table_get<NQ>(hs, sp, grp, tk, chunk, d_I, D) == 4) return 4;
    const std::vector<int> inv = inverse_gates(hs);
    Spec hsp = sp;                               // host copy for exact checks
    hsp.gates = hs.gates.data(); hsp.target = hs.target.data();
    hsp.symperm = grp.bperm.data(); hsp.symconj = grp.conj.data();
    Tree T;
    {   // level 0: the target's own class
      Mat<NQ> U; U.ovf = 0; U.k = hs.tk;
      for (int i = 0; i < N * N; i++) U.e[i] = st(hs.target[i], U.ovf);
      uint64_t a, b;
      class_hash(U, hsp, a, b);
      T.F1.assign(1, a); T.F2.assign(1, b);
    }
    std::vector<int> found;
    long n_found = 0;
    for (int t = 0; t <= depth - tk || t == 0; t++) {
      if (t > 0) {
        const int rc = levels<NQ>(hs, sp, depth, t, dedupe, chunk, T, found, n_found, "");
        if (rc == 0) { print_circuit(hs, found, n_found, "", t0); return 0; }
        if (rc != 2) return rc;
      }
      const std::vector<int> hit = table_hit<NQ>(hs, hsp, grp, inv, T, t, D, depth - t);
      if (!hit.empty()) {
        char extra[64];
        std::snprintf(extra, sizeof extra, ", \"table\": [%d, %d]", t, (int)hit.size() - t);
        print_circuit(hs, hit, 1, extra, t0);
        return 0;
      }
    }
    // the last level searched was D - k: there every residual outside the table needs more than k gates
    std::printf("{\"depth\": %d, \"proved_unsat\": true, \"table\": %d}\n", depth, tk);
    return 1;
  }
  Tree T;
  std::vector<int> found;
  long n_found = 0;
  const int rc = levels<NQ>(hs, sp, depth, depth, dedupe, chunk, T, found, n_found, "");
  if (rc == 0) { print_circuit(hs, found, n_found, "", t0); return 0; }
  return rc == 2 ? 1 : rc;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: cheese SPEC [--depth D] [--no-dedupe] [--chunk PARENTS] [--mitm] [--no-sym] [--table K] [--profile]\n"
                         "exit 0: circuit found, 1: no circuit of <= depth gates (proof)\n");
    return 3;
  }
  std::ifstream f(argv[1]);
  HostSpec hs = read_spec(f);
  int depth = hs.depth, mitm = 0, tablek = 0; bool dedupe = true, sym = true; long chunk = 1 << 17;
  for (int i = 2; i < argc; i++) {
    if (!std::strcmp(argv[i], "--depth") && i + 1 < argc) depth = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-dedupe")) dedupe = false;
    else if (!std::strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = std::atol(argv[++i]);
    else if (!std::strcmp(argv[i], "--profile")) g_prof = true;
    else if (!std::strcmp(argv[i], "--mitm")) mitm = 1;
    else if (!std::strcmp(argv[i], "--no-sym")) sym = false;
    else if (!std::strcmp(argv[i], "--table") && i + 1 < argc) tablek = std::atoi(argv[++i]);
  }
  if (depth > DMAX) { std::fprintf(stderr, "depth > %d\n", DMAX); return 3; }
  std::fprintf(stderr, "cheese: n %d, %d candidates, depth %d, dedupe %d\n", hs.n, hs.G, depth, dedupe);
#ifdef __CUDACC__
  if (hs.states) {   // the GPU state-target kernel aborts on some inputs (under investigation); CPU path validated
    std::fprintf(stderr, "state targets: GPU path not ready yet, use build/cheese-cpu\n");
    return 3;
  }
#endif
  int rc = 3;
  switch (hs.n) {   // qubit counts compiled in: NQ_MIN..NQ_MAX (each one is a full instantiation; nvcc is slow)
#if NQ_MIN <= 1 && NQ_MAX >= 1
    case 1: rc = run<1>(hs, depth, dedupe, chunk, mitm, sym, tablek); break;
#endif
#if NQ_MIN <= 2 && NQ_MAX >= 2
    case 2: rc = run<2>(hs, depth, dedupe, chunk, mitm, sym, tablek); break;
#endif
#if NQ_MIN <= 3 && NQ_MAX >= 3
    case 3: rc = run<3>(hs, depth, dedupe, chunk, mitm, sym, tablek); break;
#endif
#if NQ_MIN <= 4 && NQ_MAX >= 4
    case 4: rc = run<4>(hs, depth, dedupe, chunk, mitm, sym, tablek); break;
#endif
    default: std::fprintf(stderr, "n = %d not built (NQ_MIN..NQ_MAX)\n", hs.n); return 3;
  }
  prof_report();
  return rc;
}
