// cliffopt: fewer Clifford gates around a fixed sequence of T gates (separate from the other two programs).
//
// Input (a job file): n, the T gates in time order (qubit, dagger), and the t + 1 Clifford segments between them
// as gate lists. Output: an equivalent circuit (up to global phase) with the same number of T gates, every
// Clifford segment of minimal length, and the T gates placed so that the sum of segment lengths is small.
//
// 1. Clifford table: breadth-first search over the whole n-qubit Clifford group (n <= 3; 92,897,280 elements up
//    to phase for n = 3) with gates H, S, Sdg, X, Z, CX, on stabiliser tableaux packed into 64 bits, in a GPU hash
//    table: the exact minimal gate count of every Clifford, and an optimal circuit by walking the counts down.
// 2. Placement: T_q commutes with every Clifford fixing Z_q, so replacing T by K T' K^-1 (K Z_q' K^-1 = +-Z_q,
//    T' on q', dagger flipped for the minus sign) changes the circuit only by a global phase and turns segment j
//    into K_{j+1}^-1 D_j K_j. A beam search over all such K for every T gate minimises the summed lengths.
//
// Tableau: row i < n is the image of X_i, row n + i the image of Z_i, each (x bits, z bits, sign r) for the
// Hermitian Pauli (-1)^r i^(x.z) X^x Z^z (Aaronson-Gottesman). Applying gate G after C conjugates every row.
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
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
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void cuda_check(const char *what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error after %s: %s\n", what, cudaGetErrorString(e)); std::exit(5); }
}

// ---------------- tableaux ----------------
enum : int { GH = 0, GS = 1, GSDG = 2, GX = 3, GZ = 4, GCX = 5, GY = 6 };
struct Gate { int k, a, b; };          // kind, qubit (control), target for cx

HD inline int popc(uint32_t v) { int c = 0; while (v) { c += v & 1; v >>= 1; } return c; }
struct Tab {
  int n;
  HD int rb() const { return 2 * n + 1; }
  HD uint32_t row(uint64_t T, int i) const { return (uint32_t)((T >> (i * rb())) & ((1ULL << rb()) - 1)); }
  HD uint64_t set(uint64_t T, int i, uint32_t r) const {
    const uint64_t m = ((1ULL << rb()) - 1) << (i * rb());
    return (T & ~m) | ((uint64_t)r << (i * rb()));
  }
  HD uint32_t mk(uint32_t x, uint32_t z, uint32_t r) const { return x | (z << n) | (r << (2 * n)); }
  HD uint32_t X(uint32_t r) const { return r & ((1u << n) - 1); }
  HD uint32_t Z(uint32_t r) const { return (r >> n) & ((1u << n) - 1); }
  HD uint32_t R(uint32_t r) const { return (r >> (2 * n)) & 1; }
  HD uint64_t identity() const {
    uint64_t T = 0;
    for (int i = 0; i < n; i++) { T = set(T, i, mk(1u << i, 0, 0)); T = set(T, n + i, mk(0, 1u << i, 0)); }
    return T;
  }
  HD uint32_t conj_row(uint32_t w, Gate g) const {   // G P G^dag for one row
    uint32_t x = X(w), z = Z(w), r = R(w);
    const uint32_t a = 1u << g.a;
    const uint32_t xa = (x >> g.a) & 1, za = (z >> g.a) & 1;
    switch (g.k) {
      case GH: r ^= xa & za; x = (x & ~a) | (za << g.a); z = (z & ~a) | (xa << g.a); break;
      case GS: r ^= xa & za; z ^= xa << g.a; break;
      case GSDG: r ^= xa & (za ^ 1); z ^= xa << g.a; break;
      case GX: r ^= za; break;
      case GZ: r ^= xa; break;
      case GY: r ^= xa ^ za; break;
      case GCX: {
        const uint32_t xt = (x >> g.b) & 1, zt = (z >> g.b) & 1;
        r ^= xa & zt & (xt ^ za ^ 1);
        x ^= xa << g.b; z ^= zt << g.a;
        break;
      }
    }
    return mk(x, z, r);
  }
  HD uint64_t apply(uint64_t T, Gate g) const {       // gate g after the Clifford T
    for (int i = 0; i < 2 * n; i++) T = set(T, i, conj_row(row(T, i), g));
    return T;
  }
  // image under T of the Pauli (x, z) with sign +, tracked as i^k X^x Z^z, returned as a row
  HD uint32_t image(uint64_t T, uint32_t x, uint32_t z) const {
    int k = popc(x & z);
    uint32_t ax = 0, az = 0;
    for (int q = 0; q < 2 * n; q++) {
      const bool on = q < n ? ((x >> q) & 1) : ((z >> (q - n)) & 1);
      if (!on) continue;
      const uint32_t w = row(T, q), bx = X(w), bz = Z(w);
      k += 2 * (int)R(w) + popc(bx & bz) + 2 * popc(az & bx);   // (X^ax Z^az)(X^bx Z^bz) = (-1)^(az.bx) X^(ax^bx) Z^(az^bz)
      ax ^= bx; az ^= bz;
    }
    return mk(ax, az, (uint32_t)(((k - popc(ax & az)) & 3) >> 1));
  }
  HD uint64_t compose(uint64_t A, uint64_t B) const { // A first, then B
    uint64_t C = 0;
    for (int i = 0; i < 2 * n; i++) {
      const uint32_t w = row(A, i), im = image(B, X(w), Z(w));
      C = set(C, i, im ^ (R(w) << (2 * n)));
    }
    return C;
  }
  HD uint64_t inverse(uint64_t C) const {              // symplectic part Omega S^T Omega, signs from C
    const int m = 2 * n;
    uint64_t D = 0;
    for (int i = 0; i < m; i++) {
      uint32_t v = 0;                                  // row i of the inverse: bit j = S[w(j)][w(i)], w(i) = i + n mod 2n
      const int wi = (i + n) % m;
      for (int j = 0; j < m; j++) {
        const uint32_t sr = row(C, (j + n) % m), bits = X(sr) | (Z(sr) << n);
        v |= ((bits >> wi) & 1) << j;
      }
      const uint32_t x = v & ((1u << n) - 1), z = v >> n;
      D = set(D, i, mk(x, z, R(image(C, x, z))));     // C h C^dag = (-1)^r g_i: take sign r for h
    }
    return D;
  }
};

// ---------------- hash table: tableau -> minimal gate count ----------------
HD inline uint64_t hmix(uint64_t h) { h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33; return h; }
struct Table {
  uint64_t *keys;   // 0 = empty (no tableau is 0)
  uint8_t *dist;
  uint64_t mask;
  __device__ int find(uint64_t k) const {
    for (uint64_t s = hmix(k) & mask;; s = (s + 1) & mask) {
      const uint64_t v = keys[s];
      if (v == k) return dist[s];
      if (!v) return -1;
    }
  }
  __device__ bool insert(uint64_t k, uint8_t d) {     // true if new
    for (uint64_t s = hmix(k) & mask;; s = (s + 1) & mask) {
      const unsigned long long old = atomicCAS((unsigned long long *)&keys[s], 0ULL, (unsigned long long)k);
      if (old == 0) { dist[s] = d; return true; }
      if (old == k) return false;
    }
  }
};

struct GateSet { int n, ng; Gate g[32]; };
static GateSet gate_set(int n) {
  GateSet G{n, 0, {}};
  for (int q = 0; q < n; q++) for (int k : {GH, GS, GSDG, GX, GZ}) G.g[G.ng++] = Gate{k, q, 0};
  for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) if (a != b) G.g[G.ng++] = Gate{GCX, a, b};
  return G;
}
HD inline Gate inv_gate(Gate g) { if (g.k == GS) g.k = GSDG; else if (g.k == GSDG) g.k = GS; return g; }

__global__ void k_expand(Tab tb, GateSet G, Table T, const uint64_t *front, long nf, uint64_t *next, unsigned long long *nn,
                         long cap, uint8_t d, int *ovf) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < nf; i += (long)gridDim.x * blockDim.x)
    for (int j = 0; j < G.ng; j++) {
      const uint64_t c = tb.apply(front[i], G.g[j]);
      if (T.insert(c, d)) {
        const unsigned long long p = atomicAdd(nn, 1ULL);
        if ((long)p < cap) next[p] = c; else *ovf = 1;
      }
    }
}

// placements for a T on qubit q: every K with K Z_q' K^-1 = +-Z_q (q' and the sign recorded)
struct Cand { uint64_t K, Kinv; int qp, neg; };
__global__ void k_cands(Tab tb, Table T, long cap, int q, Cand *out, unsigned long long *nc, long maxc) {
  const int n = tb.n;
  for (long s = blockIdx.x * (long)blockDim.x + threadIdx.x; s < cap; s += (long)gridDim.x * blockDim.x) {
    const uint64_t K = T.keys[s];
    if (!K) continue;
    for (int qp = 0; qp < n; qp++) {
      const uint32_t w = tb.row(K, n + qp);
      if (tb.X(w) == 0 && tb.Z(w) == (1u << q)) {
        const unsigned long long p = atomicAdd(nc, 1ULL);
        if ((long)p < maxc) out[p] = Cand{K, tb.inverse(K), qp, (int)tb.R(w)};
      }
    }
  }
}
// T = Tdg S and Tdg = T Sdg exactly: flipping a T's dagger moves an S (Sdg) into the segment before it
HD inline uint64_t flipped_inv(const Tab &tb, const Cand &c, int dagger) {
  return tb.apply(c.Kinv, Gate{dagger ? GSDG : GS, c.qp, 0});
}
// beam step: state b (previous placement Kprev), segment D, candidate c (low bit: dagger flipped): cost of the
// segment Kc^-1 D Kprev; dg = the T gate's dagger before any flip
__global__ void k_step(Tab tb, Table T, const uint64_t *Kprev, int nb, bool first, uint64_t D, const Cand *C, long ncand,
                       int dg, const int *bcost, uint64_t *packed) {
  const long tot = (long)nb * ncand * 2;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < tot; i += (long)gridDim.x * blockDim.x) {
    const int b = (int)(i / (ncand * 2));
    const long c2 = i % (ncand * 2), c = c2 >> 1;
    const uint64_t left = first ? D : tb.compose(Kprev[b], D);
    const uint64_t kinv = (c2 & 1) ? flipped_inv(tb, C[c], dg ^ C[c].neg) : C[c].Kinv;
    const int d = T.find(tb.compose(left, kinv));
    const uint64_t cost = (uint64_t)(bcost[b] + (d < 0 ? 999 : d));
    packed[i] = (cost << 40) | ((uint64_t)b << 32) | (uint64_t)c2;  // b < 256, c2 < 2^32
  }
}
__global__ void k_final(Tab tb, Table T, const uint64_t *Kprev, int nb, uint64_t D, int *out) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < nb) { const int d = T.find(tb.compose(Kprev[b], D)); out[b] = d < 0 ? 999 : d; }
}
__global__ void k_find(Table T, const uint64_t *q, int nq, int *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nq) out[i] = T.find(q[i]);
}

// ---------------- host ----------------
struct Job { int n = 0; std::vector<int> tq, td; std::vector<std::vector<Gate>> seg; };
static Gate parse_gate(const std::string &name, std::istream &in) {
  Gate g{0, 0, 0};
  if (name == "h") g.k = GH; else if (name == "s") g.k = GS; else if (name == "sdg") g.k = GSDG;
  else if (name == "x") g.k = GX; else if (name == "z") g.k = GZ; else if (name == "y") g.k = GY;
  else if (name == "cx") g.k = GCX;
  else { std::fprintf(stderr, "unknown Clifford gate %s\n", name.c_str()); std::exit(2); }
  in >> g.a;
  if (g.k == GCX) in >> g.b;
  return g;
}
static Job read_job(const char *path) {
  std::ifstream in(path);
  Job J;
  int t = 0;
  in >> J.n >> t;
  J.tq.resize(t); J.td.resize(t);
  for (int i = 0; i < t; i++) in >> J.tq[i] >> J.td[i];
  J.seg.resize(t + 1);
  for (auto &s : J.seg) {
    int m = 0; in >> m;
    for (int i = 0; i < m; i++) { std::string nm; in >> nm; s.push_back(parse_gate(nm, in)); }
  }
  if (!in || J.n < 1 || J.n > 3) { std::fprintf(stderr, "bad job file\n"); std::exit(2); }
  return J;
}
static const char *gname(int k) { static const char *nm[] = {"h", "s", "sdg", "x", "z", "cx", "y"}; return nm[k]; }

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: cliffopt JOB [--beam B]\n"); return 3; }
  const Job J = read_job(argv[1]);
  int beam = 16;
  for (int i = 2; i < argc; i++) if (!std::strcmp(argv[i], "--beam") && i + 1 < argc) beam = std::max(1, std::min(255, std::atoi(argv[++i])));
  const int n = J.n, t = (int)J.tq.size();
  const Tab tb{n};
  const GateSet G = gate_set(n);
  const double t0 = now();

  // 1. Clifford table
  const long order = n == 1 ? 24 : n == 2 ? 11520 : 92897280;
  int lg = 4;
  while ((1L << lg) < order * 4 / 3) lg++;
  const long cap = 1L << lg;
  thrust::device_vector<uint64_t> keys(cap, 0);
  thrust::device_vector<uint8_t> dist(cap, 0);
  const Table T{thrust::raw_pointer_cast(keys.data()), thrust::raw_pointer_cast(dist.data()), (uint64_t)(cap - 1)};
  const long fcap = std::max(1024L, order / 2);
  thrust::device_vector<uint64_t> fa(fcap), fb(fcap);
  thrust::device_vector<unsigned long long> cnt(1);
  thrust::device_vector<int> ovf(1, 0);
  const uint64_t id = tb.identity();
  fa[0] = id;
  keys[hmix(id) & (cap - 1)] = id;                     // empty table: the identity takes its own slot, distance 0
  long nf = 1, total = 1;
  for (int d = 1; nf; d++) {
    cnt[0] = 0;
    k_expand<<<2048, 256>>>(tb, G, T, thrust::raw_pointer_cast(fa.data()), nf, thrust::raw_pointer_cast(fb.data()),
                            thrust::raw_pointer_cast(cnt.data()), fcap, (uint8_t)d, thrust::raw_pointer_cast(ovf.data()));
    cuda_check("bfs");
    if (ovf[0]) { std::fprintf(stderr, "frontier buffer too small\n"); return 4; }
    nf = (long)(unsigned long long)cnt[0];
    total += nf;
    if (nf) std::fprintf(stderr, "  clifford gates %d: %ld\n", d, nf);
    fa.swap(fb);
  }
  fa.clear(); fa.shrink_to_fit(); fb.clear(); fb.shrink_to_fit();
  std::fprintf(stderr, "clifford table: %ld elements (group order %ld), %.1f s\n", total, order, now() - t0);
  if (total != order) { std::fprintf(stderr, "group order mismatch\n"); return 4; }

  std::vector<uint64_t> D(t + 1);                       // segment tableaux
  for (int j = 0; j <= t; j++) { uint64_t X = tb.identity(); for (const Gate &g : J.seg[j]) X = tb.apply(X, g); D[j] = X; }

  // 2. beam search over the placements of every T gate
  std::vector<thrust::device_vector<Cand>> cands(n);
  const long maxc = order / ((1L << (2 * n)) - 1) * n + 64;   // n x 2 signs x order / (2 (4^n - 1))
  for (int q = 0; q < n; q++) {
    cands[q].resize(maxc);
    cnt[0] = 0;
    k_cands<<<2048, 256>>>(tb, T, cap, q, thrust::raw_pointer_cast(cands[q].data()), thrust::raw_pointer_cast(cnt.data()), maxc);
    cuda_check("candidates");
    const long nc = (long)(unsigned long long)cnt[0];
    if (nc > maxc) { std::fprintf(stderr, "candidate buffer too small\n"); return 4; }
    cands[q].resize(nc);
  }
  std::vector<std::vector<int>> parent(t);
  std::vector<std::vector<Cand>> chosen(t);
  std::vector<std::vector<int>> flips(t);
  thrust::device_vector<uint64_t> Kprev(1, 0);
  thrust::device_vector<int> bcost(1, 0);
  int nb = 1;
  for (int j = 0; j < t; j++) {
    const thrust::device_vector<Cand> &C = cands[J.tq[j]];
    const long nc = (long)C.size(), tot = (long)nb * nc * 2;
    thrust::device_vector<uint64_t> packed(tot);
    k_step<<<4096, 256>>>(tb, T, thrust::raw_pointer_cast(Kprev.data()), nb, j == 0, D[j], thrust::raw_pointer_cast(C.data()),
                          nc, J.td[j], thrust::raw_pointer_cast(bcost.data()), thrust::raw_pointer_cast(packed.data()));
    cuda_check("beam step");
    thrust::sort(packed.begin(), packed.end());
    const int keep = (int)std::min<long>(beam, tot);
    thrust::host_vector<uint64_t> top(packed.begin(), packed.begin() + keep);
    std::vector<uint64_t> nk(keep);
    std::vector<int> ncost(keep);
    parent[j].resize(keep); chosen[j].resize(keep); flips[j].resize(keep);
    for (int i = 0; i < keep; i++) {
      const int b = (int)((top[i] >> 32) & 0xff);
      const long c2 = (long)(top[i] & 0xffffffffULL);
      const Cand cd = C[c2 >> 1];
      parent[j][i] = b; chosen[j][i] = cd; flips[j][i] = (int)(c2 & 1);
      nk[i] = cd.K; ncost[i] = (int)(top[i] >> 40);
    }
    Kprev.assign(nk.begin(), nk.end()); bcost.assign(ncost.begin(), ncost.end());
    nb = keep;
    std::fprintf(stderr, "  T gate %d: %ld placements x beam, best partial %d\n", j + 1, nc, ncost[0]);
  }
  int best = 0, bestv = 1 << 30;
  {
    thrust::device_vector<int> fin(nb);
    k_final<<<(nb + 63) / 64, 64>>>(tb, T, thrust::raw_pointer_cast(Kprev.data()), nb, D[t], thrust::raw_pointer_cast(fin.data()));
    cuda_check("final");
    thrust::host_vector<int> hf(fin), hb(bcost);
    for (int b = 0; b < nb; b++) if (hb[b] + hf[b] < bestv) { bestv = hb[b] + hf[b]; best = b; }
  }
  std::vector<Cand> K(t);
  std::vector<int> flip(t), dag(t);
  for (int j = t - 1, b = best; j >= 0; j--) { K[j] = chosen[j][b]; flip[j] = flips[j][b]; b = parent[j][b]; }
  for (int j = 0; j < t; j++) {                         // the T gate actually placed: dagger after sign and flip
    const int d0 = J.td[j] ^ K[j].neg;
    if (flip[j]) K[j].Kinv = flipped_inv(tb, K[j], d0);
    dag[j] = d0 ^ flip[j];
  }

  // segments K_j^-1 D_j K_{j-1}, each read back as an optimal gate list
  auto dist_of = [&](const std::vector<uint64_t> &q) {
    thrust::device_vector<uint64_t> dq(q.begin(), q.end());
    thrust::device_vector<int> out(q.size());
    k_find<<<((int)q.size() + 63) / 64, 64>>>(T, thrust::raw_pointer_cast(dq.data()), (int)q.size(), thrust::raw_pointer_cast(out.data()));
    cuda_check("find");
    thrust::host_vector<int> h(out);
    return std::vector<int>(h.begin(), h.end());
  };
  std::vector<std::vector<Gate>> out(t + 1);
  int clifford_total = 0;
  for (int j = 0; j <= t; j++) {
    uint64_t X = j > 0 ? tb.compose(K[j - 1].K, D[j]) : D[j];
    if (j < t) X = tb.compose(X, K[j].Kinv);
    int d = dist_of({X})[0];
    if (d < 0) { std::fprintf(stderr, "segment not in the table\n"); return 4; }
    clifford_total += d;
    while (d > 0) {                                     // remove the last gate: G^-1 X is one step closer
      std::vector<uint64_t> pre(G.ng);
      for (int g = 0; g < G.ng; g++) pre[g] = tb.apply(X, inv_gate(G.g[g]));
      const std::vector<int> pd = dist_of(pre);
      int g = 0;
      while (g < G.ng && pd[g] != d - 1) g++;
      if (g == G.ng) { std::fprintf(stderr, "path reconstruction failed\n"); return 4; }
      out[j].push_back(G.g[g]);
      X = pre[g]; d--;
    }
    std::reverse(out[j].begin(), out[j].end());
  }
  int orig = 0;
  for (int j = 0; j <= t; j++) orig += (int)J.seg[j].size();
  std::printf("{\"clifford_gates\": %d, \"clifford_gates_before\": %d, \"t_gates\": %d, \"seconds\": %.2f}\n",
              clifford_total, orig, t, now() - t0);
  for (int j = 0; j <= t; j++) {
    for (const Gate &g : out[j]) {
      if (g.k == GCX) std::printf("cx %d %d\n", g.a, g.b);
      else std::printf("%s %d\n", gname(g.k), g.a);
    }
    if (j < t) std::printf("%s %d\n", dag[j] ? "tdg" : "t", K[j].qp);
  }
  return 0;
}
