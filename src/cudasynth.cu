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
  const Zw *target;                // N x N, exponent tk
  int tk;
  const int *cxtable;
};

// ---------------- exact matrices ----------------
template <int NQ> struct Mat {
  static constexpr int N = 1 << NQ;
  Zw e[N * N];
  int k;
};

template <int NQ> HD inline void normalise(Mat<NQ> &S) {
  constexpr int N = 1 << NQ;
  while (S.k > 0) {
    for (int i = 0; i < N * N; i++) if (!zdiv2ok(S.e[i])) return;
    for (int i = 0; i < N * N; i++) S.e[i] = zdown(S.e[i]);
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
          if (!zzero(gm)) acc = zadd(acc, zmul(S.e[r * N + idx[m]], gm));
        }
        col[lc] = acc;
      }
      for (int lc = 0; lc < L; lc++) S.e[r * N + idx[lc]] = col[lc];
    }
  }
  S.k += g.ik;
  normalise(S);
}

template <int NQ> HD inline void residual(Mat<NQ> &S, const Spec &sp, const uint8_t *seq, int len) {
  constexpr int N = 1 << NQ;
  for (int i = 0; i < N * N; i++) S.e[i] = sp.target[i];
  S.k = sp.tk;
  for (int t = 0; t < len; t++) apply_inv(S, sp.gates[seq[t]]);
}

template <int NQ> HD inline bool is_identity(const Mat<NQ> &S) {
  constexpr int N = 1 << NQ;
  if (S.k) return false;
  for (int r = 0; r < N; r++)
    for (int c = 0; c < N; c++) {
      const Zw x = S.e[r * N + c];
      if (r == c ? !(x.a == 1 && !x.b && !x.c && !x.d) : !zzero(x)) return false;
    }
  return true;
}

HD inline uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 27;
  return h;
}
template <int NQ> HD inline void hash2(const Mat<NQ> &S, uint64_t &h1, uint64_t &h2) {
  constexpr int N = 1 << NQ;
  h1 = mix(0x12345678ULL, S.k); h2 = mix(0x9abcdef0ULL, S.k);
  for (int i = 0; i < N * N; i++) {
    const Zw x = S.e[i];
    h1 = mix(mix(mix(mix(h1, x.a), x.b), x.c), x.d);
    h2 = mix(mix(mix(mix(h2, x.d * 31 + 7), x.c), x.b), x.a);
  }
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
      M[(oa << na) | ia][(ob << nb) | ib] = zmod(S.e[o * N + i], w);
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
        if ((r ^ c) & bit) { if (!zzero(S.e[r * N + c])) trivial = false; }
        else if (!((r | c) & bit) && !zeq(S.e[r * N + c], S.e[(r | bit) * N + (c | bit)])) trivial = false;
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
    for (int r = 0; r < N; r++) if (!zzero(S.e[r * N + c])) { nz++; row = r; }
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
  int best = 0;
  Zw SP[N * N], M[N * N];
  for (int b = 0; b < P; b++) {
    int fb, yb, zb;
    pauli(b, NQ, fb, yb, zb);
    for (int r = 0; r < N; r++)                          // (S' P_b)[r][c] = S'[r][c ^ fb] * phase(b, c)
      for (int c = 0; c < N; c++) {
        const Zw x = S.e[r * N + (c ^ fb)];
        SP[r * N + c] = zzero(x) ? x : zmul(x, zwpow(pphase(c, yb, zb)));
      }
    for (int r = 0; r < N; r++)                          // M = (S' P_b) S'^dag
      for (int s = 0; s < N; s++) {
        Zw acc{0, 0, 0, 0};
        for (int c = 0; c < N; c++) {
          const Zw y = S.e[s * N + c];
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

enum : uint8_t { DEAD = 0, ALIVE = 1, FOUND = 2, W2CUT = 3, W3CUT = 4 };

template <int NQ> struct Expand {        // one thread per parent: all its children
  Spec sp;
  const Node *parents;
  uint8_t *status;                       // [parent * G + g]
  uint64_t *h1, *h2;
  int t;                                 // child length
  HD void operator()(int i) const {
    const Node &p = parents[i];
    Mat<NQ> S;
    residual(S, sp, p.seq, p.len);
    const int rem = sp.depth - t;
    for (int g = 0; g < sp.G; g++) {
      const int o = i * sp.G + g;
      if (p.len >= 1 && sp.bad2[p.seq[p.len - 1] * sp.G + g]) { status[o] = W2CUT; continue; }
      if (p.len >= 2) {
        const uint32_t tri = ((uint32_t)p.seq[p.len - 2] * sp.G + p.seq[p.len - 1]) * sp.G + g;
        if ((sp.bad3[tri >> 5] >> (tri & 31)) & 1) { status[o] = W3CUT; continue; }
      }
      Mat<NQ> C = S;
      apply_inv(C, sp.gates[g]);
      if (is_identity(C)) { status[o] = FOUND; h1[o] = h2[o] = 0; continue; }
      int lb = 0;
      if (sp.standard) {
        const int hs = h_struct(C), h = C.k > hs ? C.k : hs;
        const int supp = support(C), c2 = cx_bound(C, sp);
        lb = combine(h, 0, c2, supp);
        if (lb <= rem) lb = combine(h, ptm_sde(C), c2, supp);
      }
      if (lb > rem) { status[o] = DEAD; continue; }
      status[o] = ALIVE;
      uint64_t a, b;
      hash2(C, a, b);
      h1[o] = a; h2[o] = b;
    }
  }
};

struct IsAlive { const uint8_t *s; HD bool operator()(int o) const { return s[o] == ALIVE; } };
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
    for (int i = 0; i < (1 << NQ) * (1 << NQ); i++) if (!zeq(A.e[i], B.e[i])) return false;
    return true;
  }
};

// ---------------- host ----------------
struct HostSpec {
  int n = 0, depth = 0, standard = 0, G = 0, tk = 0;
  std::vector<std::string> names;
  std::vector<Gate> gates;
  std::vector<uint8_t> bad2;
  std::vector<uint32_t> bad3;
  std::vector<Zw> target;
  std::vector<int> cutmask, cutsize, rad, cxtable;
};

static HostSpec read_spec(std::istream &in) {
  HostSpec s;
  std::string tok;
  in >> tok;
  if (tok != "cudasynth-spec") { std::cerr << "not a spec\n"; std::exit(2); }
  int version; in >> version;
  while (in >> tok) {
    if (tok == "n") in >> s.n;
    else if (tok == "mode" || tok == "phase") { std::string v; in >> v; if (v != "unitary" && v != "exact") { std::cerr << "unsupported " << tok << " " << v << "\n"; std::exit(2); } }
    else if (tok == "depth") in >> s.depth;
    else if (tok == "standard") in >> s.standard;
    else if (tok == "gates") {
      in >> s.G;
      if (s.G > GMAX) { std::cerr << "too many gates\n"; std::exit(2); }
      for (int g = 0; g < s.G; g++) {
        std::string name; Gate gt{}; int gk;
        in >> name >> gt.nq;
        for (int j = 0; j < gt.nq; j++) in >> gt.q[j];
        const int L = 1 << gt.nq;
        in >> gk;
        for (int i = 0; i < L * L; i++) { Zw x; in >> x.a >> x.b >> x.c >> x.d; (void)x; }   // forward matrix (unused)
        in >> gt.ik;
        for (int i = 0; i < L * L; i++) in >> gt.inv[i].a >> gt.inv[i].b >> gt.inv[i].c >> gt.inv[i].d;
        s.names.push_back(name); s.gates.push_back(gt);
      }
    } else if (tok == "bad2") {
      int m; in >> m; s.bad2.assign(s.G * s.G, 0);
      for (int i = 0; i < m; i++) { int a, b; in >> a >> b; s.bad2[a * s.G + b] = 1; }
    } else if (tok == "bad3") {
      int m; in >> m; s.bad3.assign((s.G * s.G * s.G + 31) / 32, 0);
      for (int i = 0; i < m; i++) { int a, b, c; in >> a >> b >> c; const uint32_t t = (a * s.G + b) * s.G + c; s.bad3[t >> 5] |= 1u << (t & 31); }
    } else if (tok == "target") {
      in >> s.tk; const int N = 1 << s.n; s.target.resize(N * N);
      for (auto &x : s.target) in >> x.a >> x.b >> x.c >> x.d;
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
  if (s.bad2.empty()) s.bad2.assign(s.G * s.G, 0);
  if (s.bad3.empty()) s.bad3.assign((s.G * s.G * s.G + 31) / 32, 0);
  return s;
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <int NQ> static int run(const HostSpec &hs, int depth, bool dedupe, long chunk) {
  thrust::device_vector<Gate> d_gates(hs.gates.begin(), hs.gates.end());
  thrust::device_vector<uint8_t> d_bad2(hs.bad2.begin(), hs.bad2.end());
  thrust::device_vector<uint32_t> d_bad3(hs.bad3.begin(), hs.bad3.end());
  thrust::device_vector<Zw> d_target(hs.target.begin(), hs.target.end());
  thrust::device_vector<int> d_cx(hs.cxtable.empty() ? std::vector<int>{0} : hs.cxtable);
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

  thrust::device_vector<Node> frontier(1);
  { Node root{}; frontier[0] = root; }
  {   // depth 0: is the target the identity?
    Mat<NQ> S; thrust::host_vector<Zw> T(d_target); for (int i = 0; i < (1 << NQ) * (1 << NQ); i++) S.e[i] = T[i]; S.k = hs.tk;
    if (is_identity(S)) { std::printf("{\"found\": 0, \"circuit\": []}\n"); return 0; }
  }
  for (int t = 1; t <= depth; t++) {
    const double t0 = now();
    const long m = (long)frontier.size();
    thrust::device_vector<Node> next;
    long n_w2 = 0, n_w3 = 0, n_dead = 0, n_alive = 0, n_found = 0;
    std::vector<Node> found;
    for (long off = 0; off < m; off += chunk) {
      const long cm = std::min(chunk, m - off);
      thrust::device_vector<uint8_t> status(cm * hs.G);
      thrust::device_vector<uint64_t> h1(cm * hs.G), h2(cm * hs.G);
      Expand<NQ> ex{sp, thrust::raw_pointer_cast(frontier.data()) + off, thrust::raw_pointer_cast(status.data()),
                    thrust::raw_pointer_cast(h1.data()), thrust::raw_pointer_cast(h2.data()), t};
      thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)cm), ex);
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
      if (na && found.empty()) {
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
  std::printf("{\"depth\": %d, \"result\": \"open\", \"frontier\": %ld}\n", depth, (long)frontier.size());
  return 2;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: cudasynth SPEC [--depth D] [--no-dedupe] [--chunk PARENTS]\n"
                         "exit 0: circuit found, 1: no circuit of <= depth gates (proof), 2: open\n");
    return 3;
  }
  std::ifstream f(argv[1]);
  HostSpec hs = read_spec(f);
  int depth = hs.depth; bool dedupe = true; long chunk = 1 << 18;
  for (int i = 2; i < argc; i++) {
    if (!std::strcmp(argv[i], "--depth") && i + 1 < argc) depth = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-dedupe")) dedupe = false;
    else if (!std::strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = std::atol(argv[++i]);
  }
  if (depth > DMAX) { std::fprintf(stderr, "depth > %d\n", DMAX); return 3; }
  std::fprintf(stderr, "cudasynth: n %d, %d candidates, depth %d, dedupe %d\n", hs.n, hs.G, depth, dedupe);
  switch (hs.n) {
    case 1: return run<1>(hs, depth, dedupe, chunk);
    case 2: return run<2>(hs, depth, dedupe, chunk);
    case 3: return run<3>(hs, depth, dedupe, chunk);
    case 4: return run<4>(hs, depth, dedupe, chunk);
    default: std::fprintf(stderr, "n = %d not built\n", hs.n); return 3;
  }
}
