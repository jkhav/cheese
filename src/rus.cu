// rus: optimal T-count of a two-qubit repeat-until-success circuit, measurement included (separate binary; the
// spec comes from python/cheese/rus.py, which also verifies the result).
//
// Only J = W(|0> x I) matters (ancilla q0 prepared in |0>), and the RUS conditions (success branch proportional to
// the target's, failure branch a Clifford) are kept by any Clifford on the output side up to a finite set of
// "roots": W = C X_t ... X_1 with C a Clifford, so the optimum is the fewest rotations X with
// X_t ... X_1 J00 = C^-1 J for a root J. Everything works on J's Pauli transfer matrix Jc[a][b] =
// Tr(P_a J P_b J^dag) / 2 (16 rows: two-qubit Paulis, 4 columns: one-qubit Paulis), entries (x + y sqrt2) / sqrt2^e.
//
// A left Clifford C permutes rows with signs: (C J)c[a] = s_C(a) Jc[pi_C(a)] with C^dag P_a C = s_C(a) P_pi_C(a).
// A rotation R(P)^-1 on the left mixes rows exactly as in tcount.cu. So the search works on left cosets C J:
//   key: exponent, row 0 (the identity row, fixed by every Clifford) and the sum of hashes of the sign-normalised
//     rows (invariant under every Clifford, so no coset is ever missed);
//   equality: exact, by finding the Clifford. It is fixed by the images of X_0, Z_0, X_1, Z_1, which must be rows
//     equal (up to sign) to the target's generator rows: few candidates, each looked up in a table indexed by those
//     images and confirmed on all 16 rows. Rows equal as a multiset are not enough on their own (an isometry's PTM
//     has repeated and zero rows), hence the explicit Clifford.
// Successors of a coset are well defined: R(P)^-1 C J = C R(+-P')^-1 J and R(-P')^-1 J = S_P'^-1 R(P')^-1 J
// (up to phase) lies in the coset of R(P')^-1 J, so the 15 peels R(P)^-1 reach every successor coset.
//
// Table: cosets reachable from J00 = |0> x I with <= K peels, at the level of their T-count (cached in
// build/rus-table-k<K>.bin). Search: peel the roots level by level; a residual R(P_j)^-1..R(P_1)^-1 J equal to
// C R(Q_c)^-1..R(Q_1)^-1 J00 (table level c) gives W = R(P_1)..R(P_j) C R(Q_c)^-1..R(Q_1)^-1 with j + c T gates
// and W(|0> x I) = J up to phase; the first level j with a hit gives the optimum.
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define HD __host__ __device__
constexpr int NR = 16;                // rows: two-qubit Paulis
constexpr int NC = 4;                 // columns: one-qubit Paulis
constexpr int NE = NR * NC;
constexpr int NCL = 11520;            // two-qubit Cliffords mod phase
constexpr int MAXT = 32;              // deepest peel sequence
__constant__ int GEN[4] = {1, 3, 4, 12};   // X_q0, Z_q0, X_q1, Z_q1

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void cuda_check(const char *what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error after %s: %s\n", what, cudaGetErrorString(e)); std::exit(5); }
}

struct RSpec {
  const int *bx, *by, *be;            // roots: NR x NC matrices and exponents
  int nroot;
  const int *tr, *ts;                 // 15 x NR: partner row and sign per rotation and row (r < 0: commutes)
  const uint8_t *perm;                // NCL x NR: pi_C
  const int8_t *sign;                 // NCL x NR: s_C
  const int *bygen;                   // 2^20: Clifford index from the images of the generators, or -1
};
struct Hist {
  const uint32_t *par[MAXT + 1];
  const uint8_t *gat[MAXT + 1];       // rotation index p (1..15)
};
HD inline uint32_t path(const Hist &H, int t, uint32_t i, uint8_t *seq) {   // peels into seq[0..t), returns the root
  for (int l = t; l >= 1; l--) { seq[l - 1] = H.gat[l][i]; i = H.par[l][i]; }
  return i;
}
HD inline uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 27;
  return h;
}

// ---- block-cooperative ops: blockDim.x == NR, thread a owns row a ----
struct Smat { int *x, *y; int *e; };
__device__ void blk_load(Smat M, const RSpec &sp, uint32_t root) {
  const int a = threadIdx.x;
  for (int c = 0; c < NC; c++) { M.x[a * NC + c] = sp.bx[root * NE + a * NC + c]; M.y[a * NC + c] = sp.by[root * NE + a * NC + c]; }
  if (!a) *M.e = sp.be[root];
  __syncthreads();
}
__device__ void blk_peel(Smat M, int p, const RSpec &sp, int *ovf) {   // M <- R(P_p)^-1 M, normalised
  const int a = threadIdx.x;
  const int r = sp.tr[(p - 1) * NR + a], s = sp.ts[(p - 1) * NR + a];
  int nx[NC], ny[NC];
  bool bad = false;
  for (int c = 0; c < NC; c++) {
    const int64_t ox = M.x[a * NC + c], oy = M.y[a * NC + c];
    int64_t vx, vy;
    if (r < 0) { vx = 2 * oy; vy = ox; }                      // times sqrt2: (x + y s) s = 2y + x s
    else { vx = ox - s * (int64_t)M.x[r * NC + c]; vy = oy - s * (int64_t)M.y[r * NC + c]; }
    if (vx > INT32_MAX || vx < INT32_MIN || vy > INT32_MAX || vy < INT32_MIN) bad = true;
    nx[c] = (int)vx; ny[c] = (int)vy;
  }
  if (bad) atomicOr(ovf, 1);
  __syncthreads();
  for (int c = 0; c < NC; c++) { M.x[a * NC + c] = nx[c]; M.y[a * NC + c] = ny[c]; }
  __syncthreads();
  if (!a) (*M.e)++;
  __syncthreads();
  while (true) {                                             // (x + y s) / s = y + (x / 2) s while all x even
    bool even = *M.e > 0;
    for (int c = 0; even && c < NC; c++) even = !(M.x[a * NC + c] & 1);
    if (!__syncthreads_and(even)) break;
    for (int c = 0; c < NC; c++) { const int x = M.x[a * NC + c]; M.x[a * NC + c] = M.y[a * NC + c]; M.y[a * NC + c] = x / 2; }
    __syncthreads();
    if (!a) (*M.e)--;
    __syncthreads();
  }
}
__device__ void blk_build(Smat M, const RSpec &sp, uint32_t root, const uint8_t *seq, int len, int *ovf) {
  blk_load(M, sp, root);
  for (int i = 0; i < len; i++) blk_peel(M, seq[i], sp, ovf);
}
// key: exponent, row 0, sum of the sign-normalised rows' hashes (order- and sign-independent)
__device__ void blk_label(Smat M, uint64_t *r1, uint64_t *r2, uint64_t *out1, uint64_t *out2) {
  const int a = threadIdx.x;
  int s = 1;
  for (int c = 0; c < NC; c++) {
    const int x = M.x[a * NC + c], y = M.y[a * NC + c];
    if (x) { s = x > 0 ? 1 : -1; break; }
    if (y) { s = y > 0 ? 1 : -1; break; }
  }
  uint64_t h1 = 0x1234567ULL, h2 = 0x89abcdefULL;
  for (int c = 0; c < NC; c++) {
    const int64_t x = s * (int64_t)M.x[a * NC + c], y = s * (int64_t)M.y[a * NC + c];
    h1 = mix(mix(h1, (uint64_t)x), (uint64_t)y);
    h2 = mix(mix(h2, (uint64_t)(y * 7 + 3)), (uint64_t)(x * 13 + c));
  }
  r1[a] = h1; r2[a] = h2;
  __syncthreads();
  if (!a) {
    uint64_t u = 0, v = 0;
    for (int b = 1; b < NR; b++) { u += r1[b]; v += r2[b]; }
    *out1 = mix(mix(u, r1[0]), (uint64_t)*M.e);
    *out2 = mix(mix(v, r2[0]), (uint64_t)*M.e ^ 0x5555ULL);
  }
  __syncthreads();
}

HD inline bool row_eq(const Smat &A, int ra, int s, const Smat &B, int rb) {   // s A[ra] == B[rb]
  for (int c = 0; c < NC; c++)
    if (s * A.x[ra * NC + c] != B.x[rb * NC + c] || s * A.y[ra * NC + c] != B.y[rb * NC + c]) return false;
  return true;
}
// Clifford index c with B = C_c A (B[a] = s_c(a) A[pi_c(a)] for every row), else -1; exact. Thread 0 enumerates
// generator images (rows of A equal to +-B's generator rows, with the commutation pattern of X0 Z0 X1 Z1).
__device__ int blk_equiv(const Smat &A, const Smat &B, const RSpec &sp, int *res) {
  if (!threadIdx.x) {
    int out = -1;
    if (*A.e == *B.e) {
      uint8_t cand[4][30]; int nc[4];
      bool any = true;
      for (int g = 0; g < 4 && any; g++) {
        nc[g] = 0;
        for (int r = 1; r < NR; r++)
          for (int s = 1; s >= -1; s -= 2)
            if (row_eq(A, r, s, B, GEN[g])) cand[g][nc[g]++] = (uint8_t)((r << 1) | (s < 0));
        any = nc[g] > 0;
      }
      // P_p and P_q commute iff tr[(p-1) * NR + q] < 0
      auto com = [&](int p, int q) { return sp.tr[(p - 1) * NR + q] < 0; };
      for (int i0 = 0; any && out < 0 && i0 < nc[0]; i0++) {
        const int b0 = cand[0][i0] >> 1;
        for (int i1 = 0; out < 0 && i1 < nc[1]; i1++) {
          const int b1 = cand[1][i1] >> 1;
          if (com(b0, b1)) continue;                         // X0, Z0 anticommute
          for (int i2 = 0; out < 0 && i2 < nc[2]; i2++) {
            const int b2 = cand[2][i2] >> 1;
            if (!com(b0, b2) || !com(b1, b2)) continue;
            for (int i3 = 0; out < 0 && i3 < nc[3]; i3++) {
              const int b3 = cand[3][i3] >> 1;
              if (com(b2, b3) || !com(b0, b3) || !com(b1, b3)) continue;
              const int key = cand[0][i0] | cand[1][i1] << 5 | cand[2][i2] << 10 | cand[3][i3] << 15;
              const int c = sp.bygen[key];
              if (c < 0) continue;
              bool ok = true;
              for (int a = 0; ok && a < NR; a++) ok = row_eq(A, sp.perm[c * NR + a], sp.sign[c * NR + a], B, a);
              if (ok) out = c;
            }
          }
        }
      }
    }
    *res = out;
  }
  __syncthreads();
  const int r = *res;
  __syncthreads();
  return r;
}

struct Scratch {
  Smat A, B;
  uint64_t *r1, *r2;
  int *res;
};
__device__ Scratch scratch() {
  __shared__ int xa[NE], ya[NE], xb[NE], yb[NE], ea, eb, rs;
  __shared__ uint64_t h1[NR], h2[NR];
  Scratch S;
  S.A = {xa, ya, &ea}; S.B = {xb, yb, &eb};
  S.r1 = h1; S.r2 = h2; S.res = &rs;
  return S;
}

// ---- kernels ----
__global__ void k_roots(RSpec sp, long n, uint64_t *h1, uint64_t *h2, int *ovf) {
  Scratch S = scratch();
  for (long o = blockIdx.x; o < n; o += gridDim.x) {
    blk_build(S.A, sp, (uint32_t)o, nullptr, 0, ovf);
    blk_label(S.A, S.r1, S.r2, h1 + o, h2 + o);
  }
}
// child o = parent (level tp, index off + o / G) peeled by rotation o % G + 1
__global__ void k_children(RSpec sp, Hist H, int tp, uint32_t off, long nc, uint64_t *h1, uint64_t *h2, int *ovf) {
  Scratch S = scratch();
  const int G = NR - 1;
  for (long o = blockIdx.x; o < nc; o += gridDim.x) {
    uint8_t seq[MAXT + 1];
    const uint32_t root = path(H, tp, off + (uint32_t)(o / G), seq);
    seq[tp] = (uint8_t)(o % G + 1);
    blk_build(S.A, sp, root, seq, tp + 1, ovf);
    blk_label(S.A, S.r1, S.r2, h1 + o, h2 + o);
  }
}

struct Key { uint64_t h1, h2; uint32_t idx; };
struct KeyLessIdx {
  HD bool operator()(const Key &x, const Key &y) const {
    if (x.h1 != y.h1) return x.h1 < y.h1;
    if (x.h2 != y.h2) return x.h2 < y.h2;
    return x.idx < y.idx;
  }
};
struct KeyByIdx { HD bool operator()(const Key &x, const Key &y) const { return x.idx < y.idx; } };
struct DKey { uint64_t h1, h2; uint32_t idx, lvl; };
struct DKeyLess { HD bool operator()(const DKey &x, const DKey &y) const { return x.h1 != y.h1 ? x.h1 < y.h1 : x.h2 < y.h2; } };
struct IsTrue { HD bool operator()(uint8_t b) const { return b != 0; } };
struct MakeKey {
  const uint64_t *h1, *h2;
  uint32_t base;
  HD Key operator()(long i) const { return {h1[i], h2[i], base + (uint32_t)i}; }
};
struct GroupStart {                   // index of j if it starts a hash group, else 0 (max-scan gives the head)
  const Key *K;
  HD long operator()(long j) const { return (j == 0 || K[j].h1 != K[j - 1].h1 || K[j].h2 != K[j - 1].h2) ? j : 0; }
};

// dedupe: is sorted key j in the same coset as its group's head? (children of level tp; exact, with a Clifford)
// A group's members that are not the head's coset are kept even if equal to each other (only costs a node).
__global__ void k_dup(RSpec sp, Hist H, int tp, const Key *K, const long *head, long n, uint8_t *dup, int *ovf) {
  Scratch S = scratch();
  const uint32_t G = NR - 1;
  for (long j = blockIdx.x; j < n; j += gridDim.x) {
    const long h = head[j];
    if (h == j) { if (!threadIdx.x) dup[j] = 0; continue; }
    uint8_t sa[MAXT + 1], sb[MAXT + 1];
    const uint32_t ra = path(H, tp, K[h].idx / G, sa); sa[tp] = (uint8_t)(K[h].idx % G + 1);
    const uint32_t rb = path(H, tp, K[j].idx / G, sb); sb[tp] = (uint8_t)(K[j].idx % G + 1);
    blk_build(S.A, sp, ra, sa, tp + 1, ovf);
    blk_build(S.B, sp, rb, sb, tp + 1, ovf);
    const bool eq = blk_equiv(S.A, S.B, sp, S.res) >= 0;
    if (!threadIdx.x) dup[j] = eq;
    __syncthreads();
  }
}
HD inline long dkey_lower(const DKey *K, long n, uint64_t h1, uint64_t h2) {
  long lo = 0, hi = n;
  while (lo < hi) {
    const long mid = (lo + hi) / 2;
    if (K[mid].h1 != h1 ? K[mid].h1 < h1 : K[mid].h2 < h2) lo = mid + 1; else hi = mid;
  }
  return lo;
}
// node j of level t (tree Hq over spec sq) against table keys D (tree Ht over spec st): index into D of a coset
// equal to it with level <= maxc (the smallest such level), or -1; cl[j] = the Clifford C with node = C table entry
__global__ void k_lookup(RSpec sq, Hist Hq, int t, const uint64_t *F1, const uint64_t *F2, long n, RSpec st, Hist Ht,
                         const DKey *D, long nd, int maxc, long *hit, int *cl, int *ovf) {
  Scratch S = scratch();
  for (long j = blockIdx.x; j < n; j += gridDim.x) {
    long res = -1;
    int rl = 1 << 30, rc = -1;
    bool built = false;
    for (long x = dkey_lower(D, nd, F1[j], F2[j]); x < nd && D[x].h1 == F1[j] && D[x].h2 == F2[j]; x++) {
      if ((int)D[x].lvl > maxc || (int)D[x].lvl >= rl) continue;
      uint8_t sa[MAXT + 1], sb[MAXT + 1];
      if (!built) { const uint32_t ra = path(Hq, t, (uint32_t)j, sa); blk_build(S.A, sq, ra, sa, t, ovf); built = true; }
      const uint32_t rb = path(Ht, D[x].lvl, D[x].idx, sb);
      blk_build(S.B, st, rb, sb, D[x].lvl, ovf);
      const int c = blk_equiv(S.B, S.A, sq, S.res);           // node (A) = C table entry (B)
      if (c >= 0) { res = x; rl = (int)D[x].lvl; rc = c; }
    }
    if (!threadIdx.x) { hit[j] = res; if (cl) cl[j] = rc; }
    __syncthreads();
  }
}

struct Gather {
  const Key *K;
  uint32_t G;
  uint32_t *P;
  uint8_t *Gt;
  uint64_t *F1, *F2;
  HD void operator()(long j) const { const uint32_t i = K[j].idx; P[j] = i / G; Gt[j] = (uint8_t)(i % G + 1); F1[j] = K[j].h1; F2[j] = K[j].h2; }
};
struct ToDKey {
  const uint64_t *F1, *F2;
  uint32_t lvl;
  HD DKey operator()(long j) const { return {F1[j], F2[j], (uint32_t)j, lvl}; }
};
struct Tree {
  std::vector<thrust::device_vector<uint32_t>> P;
  std::vector<thrust::device_vector<uint8_t>> Gt;
  std::vector<long> n;
  thrust::device_vector<uint64_t> F1, F2;   // keys of the top level
  thrust::device_vector<DKey> seen;         // table: every coset so far, sorted by key
  Tree() : P(1), Gt(1), n(1, 1) {}
  int top() const { return (int)n.size() - 1; }
  Hist hist() const {
    Hist H{};
    for (int l = 1; l <= top(); l++) { H.par[l] = thrust::raw_pointer_cast(P[l].data()); H.gat[l] = thrust::raw_pointer_cast(Gt[l].data()); }
    return H;
  }
  std::vector<int> seq(int t, uint32_t i, uint32_t *root) const {
    std::vector<int> s(t);
    for (int l = t; l >= 1; l--) { s[l - 1] = Gt[l][i]; i = P[l][i]; }
    *root = i;
    return s;
  }
};

static const long BLOCKS = 8192;
static int *g_ovf;
static void check_ovf() {
  int v = 0;
  cudaMemcpy(&v, g_ovf, sizeof v, cudaMemcpyDeviceToHost);
  if (v) { std::fprintf(stderr, "coefficient overflow (int32)\n"); std::exit(4); }
}

static void roots(const RSpec &sp, Tree &T) {
  T.n[0] = sp.nroot;
  T.F1.resize(sp.nroot); T.F2.resize(sp.nroot);
  k_roots<<<(unsigned)std::min((long)sp.nroot, BLOCKS), NR>>>(sp, sp.nroot, thrust::raw_pointer_cast(T.F1.data()),
                                                              thrust::raw_pointer_cast(T.F2.data()), g_ovf);
  cuda_check("roots");
}

// one more level of tree T (peels off sp's roots); table: drop cosets already seen at a lower level
static long grow(const RSpec &sp, Tree &T, bool table, long chunk, const char *tag) {
  const double t0 = now();
  const int t = T.top() + 1, tp = t - 1;
  const uint32_t G = NR - 1;
  const Hist H = T.hist();
  const long m = T.n[tp];
  thrust::device_vector<Key> CK;
  for (long off = 0; off < m; off += chunk) {
    const long cm = std::min(chunk, m - off), nc = cm * G;
    thrust::device_vector<uint64_t> h1(nc), h2(nc);
    k_children<<<(unsigned)std::min(nc, BLOCKS), NR>>>(sp, H, tp, (uint32_t)off, nc, thrust::raw_pointer_cast(h1.data()),
                                                       thrust::raw_pointer_cast(h2.data()), g_ovf);
    cuda_check("children");
    const long base = (long)CK.size();
    CK.resize(base + nc);
    thrust::transform(thrust::counting_iterator<long>(0), thrust::counting_iterator<long>(nc), CK.begin() + base,
                      MakeKey{thrust::raw_pointer_cast(h1.data()), thrust::raw_pointer_cast(h2.data()), (uint32_t)(off * G)});
  }
  check_ovf();
  const long nk = (long)CK.size();
  thrust::sort(CK.begin(), CK.end(), KeyLessIdx{});
  thrust::device_vector<long> head(nk);
  thrust::transform(thrust::counting_iterator<long>(0), thrust::counting_iterator<long>(nk), head.begin(),
                    GroupStart{thrust::raw_pointer_cast(CK.data())});
  thrust::inclusive_scan(head.begin(), head.end(), head.begin(), thrust::maximum<long>());
  thrust::device_vector<uint8_t> dup(nk);
  k_dup<<<(unsigned)std::min(nk, BLOCKS), NR>>>(sp, H, tp, thrust::raw_pointer_cast(CK.data()),
                                                thrust::raw_pointer_cast(head.data()), nk, thrust::raw_pointer_cast(dup.data()), g_ovf);
  cuda_check("dedupe");
  check_ovf();
  CK.erase(thrust::remove_if(CK.begin(), CK.end(), dup.begin(), IsTrue{}), CK.end());
  thrust::sort(CK.begin(), CK.end(), KeyByIdx{});
  const long kept = (long)CK.size();
  T.P.emplace_back(kept); T.Gt.emplace_back(kept); T.n.push_back(kept);
  T.F1.resize(kept); T.F2.resize(kept);
  thrust::for_each(thrust::counting_iterator<long>(0), thrust::counting_iterator<long>(kept),
                   Gather{thrust::raw_pointer_cast(CK.data()), G, thrust::raw_pointer_cast(T.P.back().data()),
                          thrust::raw_pointer_cast(T.Gt.back().data()), thrust::raw_pointer_cast(T.F1.data()),
                          thrust::raw_pointer_cast(T.F2.data())});
  long fresh = kept;
  if (table && kept) {
    thrust::device_vector<long> hit(kept);
    k_lookup<<<(unsigned)std::min(kept, BLOCKS), NR>>>(sp, T.hist(), t, thrust::raw_pointer_cast(T.F1.data()),
        thrust::raw_pointer_cast(T.F2.data()), kept, sp, T.hist(), thrust::raw_pointer_cast(T.seen.data()),
        (long)T.seen.size(), t - 1, thrust::raw_pointer_cast(hit.data()), nullptr, g_ovf);
    cuda_check("table filter");
    check_ovf();
    thrust::host_vector<long> hh(hit);
    thrust::host_vector<uint32_t> hp(T.P.back());
    thrust::host_vector<uint8_t> hg(T.Gt.back());
    thrust::host_vector<uint64_t> f1(T.F1), f2(T.F2);
    std::vector<uint32_t> np; std::vector<uint8_t> ng; std::vector<uint64_t> n1, n2;
    for (long j = 0; j < kept; j++) if (hh[j] < 0) { np.push_back(hp[j]); ng.push_back(hg[j]); n1.push_back(f1[j]); n2.push_back(f2[j]); }
    fresh = (long)np.size();
    T.P.back().assign(np.begin(), np.end()); T.Gt.back().assign(ng.begin(), ng.end()); T.n.back() = fresh;
    T.F1.assign(n1.begin(), n1.end()); T.F2.assign(n2.begin(), n2.end());
    thrust::device_vector<DKey> nk2(fresh), merged(T.seen.size() + fresh);
    thrust::transform(thrust::counting_iterator<long>(0), thrust::counting_iterator<long>(fresh), nk2.begin(),
                      ToDKey{thrust::raw_pointer_cast(T.F1.data()), thrust::raw_pointer_cast(T.F2.data()), (uint32_t)t});
    thrust::sort(nk2.begin(), nk2.end(), DKeyLess{});
    thrust::merge(T.seen.begin(), T.seen.end(), nk2.begin(), nk2.end(), merged.begin(), DKeyLess{});
    T.seen.swap(merged);
  }
  std::printf("{%s\"t\": %d, \"parents\": %ld, \"children\": %ld, \"cosets\": %ld, \"new\": %ld, \"seconds\": %.2f}\n",
              tag, t, m, nk, kept, fresh, now() - t0);
  std::fflush(stdout);
  return fresh;
}

// the table depends only on K (and the spec's fixed tables): cached in build/rus-table-k<K>.bin
static bool table_load(const char *path, Tree &D) {
  FILE *f = std::fopen(path, "rb");
  if (!f) return false;
  int nl = 0;
  bool ok = std::fread(&nl, sizeof nl, 1, f) == 1 && nl >= 0 && nl <= MAXT;
  for (int l = 1; ok && l <= nl; l++) {
    long n = 0;
    ok = std::fread(&n, sizeof n, 1, f) == 1 && n >= 0;
    if (!ok) break;
    std::vector<uint32_t> P(n); std::vector<uint8_t> G(n);
    ok = std::fread(P.data(), 4, n, f) == (size_t)n && std::fread(G.data(), 1, n, f) == (size_t)n;
    D.P.emplace_back(P.begin(), P.end()); D.Gt.emplace_back(G.begin(), G.end()); D.n.push_back(n);
  }
  long ns = 0;
  ok = ok && std::fread(&ns, sizeof ns, 1, f) == 1 && ns > 0;
  if (ok) { std::vector<DKey> K(ns); ok = std::fread(K.data(), sizeof(DKey), ns, f) == (size_t)ns; D.seen.assign(K.begin(), K.end()); }
  std::fclose(f);
  return ok;
}
static void table_save(const char *path, const Tree &D) {
  FILE *f = std::fopen(path, "wb");
  if (!f) { std::fprintf(stderr, "cannot write %s (table not cached)\n", path); return; }
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

struct HSpec {
  std::vector<int> rx, ry, re;        // roots
  std::vector<int> tx, ty; int te = 0;   // table root (J00)
  std::vector<int> tr, ts;
  std::vector<uint8_t> perm; std::vector<int8_t> sign;
  int nroot = 0;
};
static void read_mat(std::ifstream &in, std::vector<int> &x, std::vector<int> &y, int &e) {
  in >> e;
  for (int i = 0; i < NE; i++) { int a, b; in >> a >> b; x.push_back(a); y.push_back(b); }
}
static HSpec read_spec(const char *path) {
  std::ifstream in(path);
  HSpec s;
  std::string tok;
  in >> tok;
  if (tok != "rus-spec") { std::fprintf(stderr, "not a rus spec\n"); std::exit(2); }
  int ver; in >> ver;
  while (in >> tok) {
    if (tok == "table-root") read_mat(in, s.tx, s.ty, s.te);
    else if (tok == "roots") { in >> s.nroot; s.re.resize(s.nroot); for (int r = 0; r < s.nroot; r++) read_mat(in, s.rx, s.ry, s.re[r]); }
    else if (tok == "rotations") { int m; in >> m; s.tr.resize(m * NR); s.ts.resize(m * NR); for (int i = 0; i < m * NR; i++) in >> s.tr[i] >> s.ts[i]; }
    else if (tok == "cliffords") {
      int m; in >> m;
      if (m != NCL) { std::fprintf(stderr, "expected %d Cliffords\n", NCL); std::exit(2); }
      s.perm.resize(m * NR); s.sign.resize(m * NR);
      for (int i = 0; i < m * NR; i++) { int p, g; in >> p >> g; s.perm[i] = (uint8_t)p; s.sign[i] = (int8_t)g; }
    }
    else if (tok == "end") break;
    else { std::fprintf(stderr, "unknown field %s\n", tok.c_str()); std::exit(2); }
  }
  if (!s.nroot || s.tx.empty() || s.tr.size() != 15 * NR || s.perm.empty()) { std::fprintf(stderr, "incomplete spec\n"); std::exit(2); }
  return s;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: rus SPEC [--table K] [--max T]\n"
                         "optimal RUS T-count (success branch and Pr[success] fixed, failure a Clifford);\n"
                         "exit 0 found, 1 none with T-count <= T\n");
    return 3;
  }
  const HSpec hs = read_spec(argv[1]);
  int K = 8, maxt = 16;
  long chunk = 1 << 16;
  for (int i = 2; i < argc; i++) {
    if (!std::strcmp(argv[i], "--table") && i + 1 < argc) K = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--max") && i + 1 < argc) maxt = std::atoi(argv[++i]);
  }
  thrust::device_vector<int> d_ovf(1, 0);
  g_ovf = thrust::raw_pointer_cast(d_ovf.data());
  const double t0 = now();
  std::vector<int> bygen(1 << 20, -1);
  const int G[4] = {1, 3, 4, 12};
  for (int c = 0; c < NCL; c++) {
    int key = 0;
    for (int g = 0; g < 4; g++) key |= ((hs.perm[c * NR + G[g]] << 1) | (hs.sign[c * NR + G[g]] < 0)) << (5 * g);
    if (bygen[key] >= 0) { std::fprintf(stderr, "Clifford table: generator images not unique\n"); return 2; }
    bygen[key] = c;
  }
  thrust::device_vector<int> d_rx(hs.rx), d_ry(hs.ry), d_re(hs.re), d_tx(hs.tx), d_ty(hs.ty), d_te(1, hs.te);
  thrust::device_vector<int> d_tr(hs.tr), d_ts(hs.ts), d_bygen(bygen);
  thrust::device_vector<uint8_t> d_perm(hs.perm);
  thrust::device_vector<int8_t> d_sign(hs.sign);
  const RSpec su{thrust::raw_pointer_cast(d_rx.data()), thrust::raw_pointer_cast(d_ry.data()), thrust::raw_pointer_cast(d_re.data()),
                 hs.nroot, thrust::raw_pointer_cast(d_tr.data()), thrust::raw_pointer_cast(d_ts.data()),
                 thrust::raw_pointer_cast(d_perm.data()), thrust::raw_pointer_cast(d_sign.data()), thrust::raw_pointer_cast(d_bygen.data())};
  RSpec si = su;
  si.bx = thrust::raw_pointer_cast(d_tx.data()); si.by = thrust::raw_pointer_cast(d_ty.data());
  si.be = thrust::raw_pointer_cast(d_te.data()); si.nroot = 1;

  Tree D;                                         // table: level 0 is the coset of J00
  char tpath[96];
  std::snprintf(tpath, sizeof tpath, "build/rus-table-k%d.bin", K);
  if (table_load(tpath, D)) {
    std::fprintf(stderr, "table %s: %ld cosets (loaded, %.1f s)\n", tpath, (long)D.seen.size(), now() - t0);
  } else {
    roots(si, D);
    { thrust::host_vector<uint64_t> a(D.F1), b(D.F2); D.seen.assign(1, DKey{a[0], b[0], 0, 0}); }
    for (int l = 1; l <= K; l++) if (!grow(si, D, true, chunk, "\"dir\": \"table\", ")) break;
    std::fprintf(stderr, "table: %ld cosets with T-count <= %d (%.1f s), cached in %s\n", (long)D.seen.size(), D.top(),
                 now() - t0, tpath);
    table_save(tpath, D);
  }

  Tree T;                                         // search: peel rotations off the roots
  roots(su, T);
  thrust::host_vector<DKey> dk;
  for (int j = 0; j <= maxt; j++) {
    if (j > 0 && !grow(su, T, false, chunk, "\"dir\": \"search\", ")) break;
    const long n = T.n[j];
    thrust::device_vector<long> hit(n);
    thrust::device_vector<int> cl(n);
    const int maxc = std::min(D.top(), maxt - j);
    k_lookup<<<(unsigned)std::min(n, BLOCKS), NR>>>(su, T.hist(), j, thrust::raw_pointer_cast(T.F1.data()),
        thrust::raw_pointer_cast(T.F2.data()), n, si, D.hist(), thrust::raw_pointer_cast(D.seen.data()),
        (long)D.seen.size(), maxc, thrust::raw_pointer_cast(hit.data()), thrust::raw_pointer_cast(cl.data()), g_ovf);
    cuda_check("lookup");
    check_ovf();
    thrust::host_vector<long> hh(hit);
    long best = -1, bestx = -1;
    int bestc = 1 << 30;
    for (long i = 0; i < n; i++)
      if (hh[i] >= 0) {
        if (dk.empty()) dk = D.seen;
        if ((int)dk[hh[i]].lvl < bestc) { bestc = (int)dk[hh[i]].lvl; best = i; bestx = hh[i]; }
      }
    if (best >= 0) {
      uint32_t root, troot;
      const std::vector<int> fw = T.seq(j, (uint32_t)best, &root), tb = D.seq(bestc, dk[bestx].idx, &troot);
      const int c = cl[best];
      std::printf("{\"tcount\": %d, \"root\": %u, \"fw\": [", j + bestc, root);
      for (size_t i = 0; i < fw.size(); i++) std::printf("%s%d", i ? ", " : "", fw[i]);
      std::printf("], \"cliff\": %d, \"tb\": [", c);
      for (size_t i = 0; i < tb.size(); i++) std::printf("%s%d", i ? ", " : "", tb[i]);
      std::printf("], \"split\": [%d, %d], \"seconds\": %.2f}\n", j, bestc, now() - t0);
      return 0;
    }
    if (j + D.top() >= maxt) break;
  }
  std::printf("{\"tcount_above\": %d, \"seconds\": %.2f}\n", maxt, now() - t0);
  return 1;
}
