// tcount: optimal T-count of a unitary over Clifford+T (separate from the gate-count search in cudasynth.cu).
//
// U = R(P_1) ... R(P_t) C with R(P) = ((1 + w) I + (1 - w) P) / 2 (a pi/8 rotation about a Pauli, one T gate)
// and C a Clifford; global phase ignored (Gosset, Kliuchnikov, Mosca, Russo). Everything works on channel
// representations Uc[a][b] = Tr(P_a U P_b U^dag) / 2^n: real 4^n x 4^n matrices with entries
// (x + y sqrt2) / sqrt2^e, one exponent e per matrix (python/cudasynth/tcount.py writes the spec).
//
// Peeling a rotation off the left, S <- R(P)^-1 S, keeps the rows that commute with P (times sqrt2 in the
// common exponent) and replaces an anticommuting row a by row a - s row r, where i P P_a = s P_r (checked
// exactly against direct multiplication). Right multiplication by a Clifford permutes the columns with signs,
// so a right coset S C is labelled by its exponent and the multiset of sign-normalised columns. Peeling acts on
// the left, so the successors of a coset depend only on the coset: merging cosets loses nothing.
//
// Table: every coset reachable from the identity with <= K peels, at the level of its T-count (a coset already
// seen at a lower level is dropped). Search: peel rotations off U level by level; a residual whose coset is in
// the table at level c gives T-count j + c, and the first level j with a hit gives the optimum (a circuit of
// T-count t' is hit at level max(0, t' - K)). Coset equality is always confirmed exactly (columns compared after
// sorting by hash); the decomposition is re-checked independently by `python -m cudasynth.tcount verify`.
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
constexpr int MAXP = 64;              // 4^n, n <= 3
constexpr int MAXT = 32;              // deepest peel sequence

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void cuda_check(const char *what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error after %s: %s\n", what, cudaGetErrorString(e)); std::exit(5); }
}

struct TSpec {
  int n, P;
  const int *bx, *by;                 // base matrix (P x P): U's channel, or the identity for the table
  int be;
  const int *tr, *ts;                 // (P-1) x P: partner row and sign per rotation and row (r < 0: commutes)
};
struct Hist {
  const uint32_t *par[MAXT + 1];
  const uint8_t *gat[MAXT + 1];       // rotation index p (1..P-1)
};
HD inline int path(const Hist &H, int t, uint32_t i, uint8_t *seq) {
  for (int l = t; l >= 1; l--) { seq[l - 1] = H.gat[l][i]; i = H.par[l][i]; }
  return t;
}
HD inline uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 27;
  return h;
}

// ---- block-cooperative matrix ops: blockDim.x == P, thread a owns row a (and column a) ----
struct Smat { int *x, *y; int *e; };
__device__ void blk_load(Smat M, const int *gx, const int *gy, int ge, int P) {
  const int a = threadIdx.x;
  for (int c = 0; c < P; c++) { M.x[a * P + c] = gx[a * P + c]; M.y[a * P + c] = gy[a * P + c]; }
  if (!a) *M.e = ge;
  __syncthreads();
}
__device__ void blk_peel(Smat M, int p, const TSpec &sp, int *ovf) {   // M <- R(P_p)^-1 M, normalised
  const int P = sp.P, a = threadIdx.x;
  const int r = sp.tr[(p - 1) * P + a], s = sp.ts[(p - 1) * P + a];
  int nx[MAXP], ny[MAXP];
  bool bad = false;
  for (int c = 0; c < P; c++) {
    const int64_t ox = M.x[a * P + c], oy = M.y[a * P + c];
    int64_t vx, vy;
    if (r < 0) { vx = 2 * oy; vy = ox; }                      // times sqrt2: (x + y s) s = 2y + x s
    else { vx = ox - s * (int64_t)M.x[r * P + c]; vy = oy - s * (int64_t)M.y[r * P + c]; }
    if (vx > INT32_MAX || vx < INT32_MIN || vy > INT32_MAX || vy < INT32_MIN) bad = true;
    nx[c] = (int)vx; ny[c] = (int)vy;
  }
  if (bad) atomicOr(ovf, 1);
  __syncthreads();
  for (int c = 0; c < P; c++) { M.x[a * P + c] = nx[c]; M.y[a * P + c] = ny[c]; }
  __syncthreads();
  if (!a) (*M.e)++;
  __syncthreads();
  while (true) {                                             // (x + y s) / s = y + (x / 2) s while all x even
    bool even = *M.e > 0;
    for (int c = 0; even && c < P; c++) even = !(M.x[a * P + c] & 1);
    if (!__syncthreads_and(even)) break;
    for (int c = 0; c < P; c++) { const int x = M.x[a * P + c]; M.x[a * P + c] = M.y[a * P + c]; M.y[a * P + c] = x / 2; }
    __syncthreads();
    if (!a) (*M.e)--;
    __syncthreads();
  }
}
__device__ void blk_build(Smat M, const TSpec &sp, const uint8_t *seq, int len, int *ovf) {
  blk_load(M, sp.bx, sp.by, sp.be, sp.P);
  for (int i = 0; i < len; i++) blk_peel(M, seq[i], sp, ovf);
}
// per column c: the sign of its first nonzero entry and two hashes of the sign-normalised column
__device__ void blk_colhash(Smat M, int P, int *sg, uint64_t *c1, uint64_t *c2) {
  const int c = threadIdx.x;
  int s = 1;
  for (int r = 0; r < P; r++) {
    const int x = M.x[r * P + c], y = M.y[r * P + c];
    if (x || y) { s = (x > 0 || (x == 0 && y > 0)) ? 1 : -1; break; }
  }
  uint64_t h1 = 0x1234567ULL, h2 = 0x89abcdefULL;
  for (int r = 0; r < P; r++) {
    const int64_t x = s * (int64_t)M.x[r * P + c], y = s * (int64_t)M.y[r * P + c];
    h1 = mix(mix(h1, (uint64_t)x), (uint64_t)y);
    h2 = mix(mix(h2, (uint64_t)(y * 7 + 3)), (uint64_t)(x * 13 + r));
  }
  sg[c] = s; c1[c] = h1; c2[c] = h2;
  __syncthreads();
}
// coset key: the exponent and the column hashes summed (order-independent: columns may be permuted)
__device__ void blk_label(Smat M, int P, int *sg, uint64_t *c1, uint64_t *c2, uint64_t *out1, uint64_t *out2) {
  blk_colhash(M, P, sg, c1, c2);
  if (!threadIdx.x) {
    uint64_t a = 0, b = 0;
    for (int c = 0; c < P; c++) { a += c1[c]; b += c2[c]; }
    *out1 = mix(a, (uint64_t)*M.e); *out2 = mix(b, (uint64_t)*M.e ^ 0x5555ULL);
  }
  __syncthreads();
}
// column indices sorted by (c1, c2): bitonic sort (P a power of two)
__device__ void blk_sortcols(int P, const uint64_t *c1, const uint64_t *c2, int *ord) {
  const int t = threadIdx.x;
  ord[t] = t;
  __syncthreads();
  for (int k = 2; k <= P; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int o = t ^ j;
      if (o > t) {
        const int A = ord[t], B = ord[o];
        const bool gt = c1[A] != c1[B] ? c1[A] > c1[B] : c2[A] > c2[B];
        if (((t & k) == 0) == gt) { ord[t] = B; ord[o] = A; }
      }
      __syncthreads();
    }
}

struct Scratch {
  Smat A, B;
  int *sa, *sb, *oa, *ob;
  uint64_t *a1, *a2, *b1, *b2;
};
__device__ Scratch scratch(int P) {
  extern __shared__ int sm[];
  Scratch S;
  int *p = sm;
  S.A.x = p; p += P * P; S.A.y = p; p += P * P; S.A.e = p; p += 2;
  S.B.x = p; p += P * P; S.B.y = p; p += P * P; S.B.e = p; p += 2;
  S.sa = p; p += P; S.sb = p; p += P; S.oa = p; p += P; S.ob = p; p += P;
  uint64_t *q = (uint64_t *)(((uintptr_t)p + 7) & ~(uintptr_t)7);
  S.a1 = q; q += P; S.a2 = q; q += P; S.b1 = q; q += P; S.b2 = q;
  return S;
}
static size_t smem_bytes(int P) { return (size_t)(4 * P * P + 4 + 4 * P) * 4 + 8 + (size_t)4 * P * 8; }

// exact: same exponent and the same multiset of sign-normalised columns (block-uniform result)
__device__ bool blk_same_coset(Scratch &S, int P) {
  if (*S.A.e != *S.B.e) return false;
  blk_colhash(S.A, P, S.sa, S.a1, S.a2);
  blk_colhash(S.B, P, S.sb, S.b1, S.b2);
  blk_sortcols(P, S.a1, S.a2, S.oa);
  blk_sortcols(P, S.b1, S.b2, S.ob);
  const int c = threadIdx.x, ca = S.oa[c], cb = S.ob[c];
  bool eq = S.a1[ca] == S.b1[cb] && S.a2[ca] == S.b2[cb];
  for (int r = 0; eq && r < P; r++)
    eq = S.sa[ca] * S.A.x[r * P + ca] == S.sb[cb] * S.B.x[r * P + cb] &&
         S.sa[ca] * S.A.y[r * P + ca] == S.sb[cb] * S.B.y[r * P + cb];
  return __syncthreads_and(eq);
}

// ---- kernels ----
__global__ void k_root(TSpec sp, uint64_t *h1, uint64_t *h2, int *ovf) {   // the base matrix's own key
  Scratch S = scratch(sp.P);
  blk_build(S.A, sp, nullptr, 0, ovf);
  blk_label(S.A, sp.P, S.sa, S.a1, S.a2, h1, h2);
}
// child o = parent (level tp, index off + o / G) peeled by rotation o % G + 1
__global__ void k_children(TSpec sp, Hist H, int tp, uint32_t off, long nc, uint64_t *h1, uint64_t *h2, int *ovf) {
  Scratch S = scratch(sp.P);
  const int G = sp.P - 1;
  for (long o = blockIdx.x; o < nc; o += gridDim.x) {
    uint8_t seq[MAXT + 1];
    const int len = path(H, tp, off + (uint32_t)(o / G), seq);
    seq[len] = (uint8_t)(o % G + 1);
    blk_build(S.A, sp, seq, len + 1, ovf);
    blk_label(S.A, sp.P, S.sa, S.a1, S.a2, h1 + o, h2 + o);
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

// dedupe: is sorted key j exactly in the coset of its group's head? (children of level tp)
__global__ void k_dup(TSpec sp, Hist H, int tp, const Key *K, const long *head, long n, uint8_t *dup, int *ovf) {
  Scratch S = scratch(sp.P);
  const uint32_t G = sp.P - 1;
  for (long j = blockIdx.x; j < n; j += gridDim.x) {
    const long h = head[j];
    if (h == j) { if (!threadIdx.x) dup[j] = 0; continue; }
    uint8_t sa[MAXT + 1], sb[MAXT + 1];
    int la = path(H, tp, K[h].idx / G, sa); sa[la++] = (uint8_t)(K[h].idx % G + 1);
    int lb = path(H, tp, K[j].idx / G, sb); sb[lb++] = (uint8_t)(K[j].idx % G + 1);
    blk_build(S.A, sp, sa, la, ovf);
    blk_build(S.B, sp, sb, lb, ovf);
    const bool eq = blk_same_coset(S, sp.P);
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
// node j of level t (tree Hq over spec sq) against table keys D (tree Ht over spec st): index into D of an
// exactly equal coset with level <= maxc (the smallest such level), or -1
__global__ void k_lookup(TSpec sq, Hist Hq, int t, const uint64_t *F1, const uint64_t *F2, long n, TSpec st, Hist Ht,
                         const DKey *D, long nd, int maxc, long *hit, int *ovf) {
  Scratch S = scratch(sq.P);
  for (long j = blockIdx.x; j < n; j += gridDim.x) {
    long res = -1;
    int rl = 1 << 30;
    bool built = false;
    for (long x = dkey_lower(D, nd, F1[j], F2[j]); x < nd && D[x].h1 == F1[j] && D[x].h2 == F2[j]; x++) {
      if ((int)D[x].lvl > maxc || (int)D[x].lvl >= rl) continue;
      uint8_t sa[MAXT + 1], sb[MAXT + 1];
      if (!built) { const int la = path(Hq, t, (uint32_t)j, sa); blk_build(S.A, sq, sa, la, ovf); built = true; }
      const int lb = path(Ht, D[x].lvl, D[x].idx, sb);
      blk_build(S.B, st, sb, lb, ovf);
      if (blk_same_coset(S, sq.P)) { res = x; rl = (int)D[x].lvl; }
    }
    if (!threadIdx.x) hit[j] = res;
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
  std::vector<int> seq(int t, uint32_t i) const {
    std::vector<int> s(t);
    for (int l = t; l >= 1; l--) { s[l - 1] = Gt[l][i]; i = P[l][i]; }
    return s;
  }
};

static const long BLOCKS = 4096;
static int *g_ovf;
static void check_ovf() {
  int v = 0;
  cudaMemcpy(&v, g_ovf, sizeof v, cudaMemcpyDeviceToHost);
  if (v) { std::fprintf(stderr, "coefficient overflow (int32)\n"); std::exit(4); }
}

static void root(const TSpec &sp, Tree &T) {
  thrust::device_vector<uint64_t> a(1), b(1);
  k_root<<<1, sp.P, smem_bytes(sp.P)>>>(sp, thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), g_ovf);
  cuda_check("root");
  T.F1 = a; T.F2 = b;
}

// one more level of tree T (peels off sp's base); table: drop cosets already seen at a lower level
static long grow(const TSpec &sp, Tree &T, bool table, long chunk, const char *tag) {
  const double t0 = now();
  const int t = T.top() + 1, tp = t - 1, P = sp.P;
  const uint32_t G = P - 1;
  const size_t sm = smem_bytes(P);
  const Hist H = T.hist();
  const long m = T.n[tp];
  thrust::device_vector<Key> CK;
  for (long off = 0; off < m; off += chunk) {
    const long cm = std::min(chunk, m - off), nc = cm * G;
    thrust::device_vector<uint64_t> h1(nc), h2(nc);
    k_children<<<(unsigned)std::min(nc, BLOCKS), P, sm>>>(sp, H, tp, (uint32_t)off, nc, thrust::raw_pointer_cast(h1.data()),
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
  k_dup<<<(unsigned)std::min(nk, BLOCKS), P, sm>>>(sp, H, tp, thrust::raw_pointer_cast(CK.data()),
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
    k_lookup<<<(unsigned)std::min(kept, BLOCKS), P, sm>>>(sp, T.hist(), t, thrust::raw_pointer_cast(T.F1.data()),
        thrust::raw_pointer_cast(T.F2.data()), kept, sp, T.hist(), thrust::raw_pointer_cast(T.seen.data()),
        (long)T.seen.size(), t - 1, thrust::raw_pointer_cast(hit.data()), g_ovf);
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

// the table depends only on n and K: cached in build/tcount-table-n<n>-k<K>.bin (tree levels + sorted keys)
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

struct HSpec { int n = 0, P = 0, e = 0; std::vector<int> x, y, tr, ts; };
static HSpec read_spec(const char *path) {
  std::ifstream in(path);
  HSpec s;
  std::string tok;
  in >> tok;
  if (tok != "tcount-spec") { std::fprintf(stderr, "not a tcount spec\n"); std::exit(2); }
  int ver; in >> ver;
  while (in >> tok) {
    if (tok == "n") { in >> s.n; s.P = 1 << (2 * s.n); }
    else if (tok == "channel") { in >> s.e; s.x.resize(s.P * s.P); s.y.resize(s.P * s.P); for (int i = 0; i < s.P * s.P; i++) in >> s.x[i] >> s.y[i]; }
    else if (tok == "rotations") { int m; in >> m; s.tr.resize(m * s.P); s.ts.resize(m * s.P); for (int i = 0; i < m * s.P; i++) in >> s.tr[i] >> s.ts[i]; }
    else if (tok == "end") break;
    else { std::fprintf(stderr, "unknown field %s\n", tok.c_str()); std::exit(2); }
  }
  if (s.P > MAXP || s.n < 1) { std::fprintf(stderr, "n = %d not supported (1 <= n <= 3)\n", s.n); std::exit(3); }
  return s;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: tcount SPEC [--table K] [--max T]\n"
                         "optimal T-count up to global phase; exit 0 found, 1 none with T-count <= T\n");
    return 3;
  }
  const HSpec hs = read_spec(argv[1]);
  int K = 4, maxt = 16;
  long chunk = 1 << 14;
  for (int i = 2; i < argc; i++) {
    if (!std::strcmp(argv[i], "--table") && i + 1 < argc) K = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--max") && i + 1 < argc) maxt = std::atoi(argv[++i]);
  }
  const int P = hs.P;
  const size_t sm = smem_bytes(P);
  cudaFuncSetAttribute(k_root, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  cudaFuncSetAttribute(k_children, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  cudaFuncSetAttribute(k_dup, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  cudaFuncSetAttribute(k_lookup, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  thrust::device_vector<int> d_ovf(1, 0);
  g_ovf = thrust::raw_pointer_cast(d_ovf.data());
  const double t0 = now();
  thrust::device_vector<int> d_x(hs.x), d_y(hs.y), d_tr(hs.tr), d_ts(hs.ts);
  std::vector<int> ix(P * P, 0), iy(P * P, 0);
  for (int i = 0; i < P; i++) ix[i * P + i] = 1;
  thrust::device_vector<int> d_ix(ix), d_iy(iy);
  const TSpec su{hs.n, P, thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_y.data()), hs.e,
                 thrust::raw_pointer_cast(d_tr.data()), thrust::raw_pointer_cast(d_ts.data())};
  TSpec si = su;
  si.bx = thrust::raw_pointer_cast(d_ix.data()); si.by = thrust::raw_pointer_cast(d_iy.data()); si.be = 0;

  Tree D;                                         // table: level 0 is the Clifford coset
  char tpath[96];
  std::snprintf(tpath, sizeof tpath, "build/tcount-table-n%d-k%d.bin", hs.n, K);
  if (table_load(tpath, D)) {
    std::fprintf(stderr, "table %s: %ld cosets (loaded, %.1f s)\n", tpath, (long)D.seen.size(), now() - t0);
  } else {
    root(si, D);
    { thrust::host_vector<uint64_t> a(D.F1), b(D.F2); D.seen.assign(1, DKey{a[0], b[0], 0, 0}); }
    for (int l = 1; l <= K; l++) if (!grow(si, D, true, chunk, "\"dir\": \"table\", ")) break;
    std::fprintf(stderr, "table: %ld cosets with T-count <= %d (%.1f s), cached in %s\n", (long)D.seen.size(), D.top(),
                 now() - t0, tpath);
    table_save(tpath, D);
  }

  Tree T;                                         // search: peel rotations off U
  root(su, T);
  thrust::host_vector<DKey> dk;
  for (int j = 0; j <= maxt; j++) {
    if (j > 0 && !grow(su, T, false, chunk, "\"dir\": \"search\", ")) break;
    const long n = T.n[j];
    thrust::device_vector<long> hit(n);
    const int maxc = std::min(D.top(), maxt - j);
    k_lookup<<<(unsigned)std::min(n, BLOCKS), P, sm>>>(su, T.hist(), j, thrust::raw_pointer_cast(T.F1.data()),
        thrust::raw_pointer_cast(T.F2.data()), n, si, D.hist(), thrust::raw_pointer_cast(D.seen.data()),
        (long)D.seen.size(), maxc, thrust::raw_pointer_cast(hit.data()), g_ovf);
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
      // U = R(P_1)..R(P_j) S with S = R(Q_c)^-1 .. R(Q_1)^-1 C' (the table entry's peels Q_1..Q_c from I)
      const std::vector<int> fw = T.seq(j, (uint32_t)best), tb = D.seq(bestc, dk[bestx].idx);
      std::printf("{\"tcount\": %d, \"rotations\": [", j + bestc);
      bool first = true;
      for (int p : fw) { std::printf("%s[%d, 0]", first ? "" : ", ", p); first = false; }
      for (int i = (int)tb.size() - 1; i >= 0; i--) { std::printf("%s[%d, 1]", first ? "" : ", ", tb[i]); first = false; }
      std::printf("], \"split\": [%d, %d], \"seconds\": %.2f}\n", j, bestc, now() - t0);
      return 0;
    }
    if (j + D.top() >= maxt) break;
  }
  std::printf("{\"tcount_above\": %d, \"seconds\": %.2f}\n", maxt, now() - t0);
  return 1;
}
