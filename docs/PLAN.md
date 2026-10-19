# cudasynth: GPU branch-and-bound synthesis, plan

Copied from milq `misc/GPU_SEARCH_PLAN.md` (2026-10-19). Paths below starting with `misc/` refer to the milq
repository. The CPU prototype there, `misc/prototypes/prune_probe.py` (Python, 10 processes), proves depths no
SAT run reached (Toffoli 9, CCZ 8, half_adder_3 9, Fredkin 9; 30-180 s each) and finds optimal circuits (CS = 5).
This repository moves it to GPUs (an RTX card first, a university cluster later) and extends it to RUS targets.

## 0. Decisions (2026-10-19)

- CUDA C++ from the start (the Python prototype in milq is the correctness oracle; no CuPy stage).
- Exact arithmetic: entries in Z[w] as four integers (a, b, c, d) = a + b w + c w^2 + d w^3, one sqrt2 exponent
  k per vector (MILQ's five-tuple with the exponent shared by the vector), normalised so equal states have equal
  bits; int32 with an overflow check, int64 fallback.
- Input = a spec: number of qubits, the gate set (each gate's exact local matrix and the qubits it acts on), the
  vector pairs to map (in_j -> out_j; full unitary = every basis state), the phase mode (exact / one global phase),
  the depth bound and the mode (prove / find / all). Gate classes for the bounds (sqrt2-raising, non-Clifford,
  entangling) are derived from the exact matrices; a bound whose assumption fails is switched off.
- Search: level-synchronous branch-and-bound first, then meet-in-the-middle (exact and complete; memory-bound).
- Every per-prefix function is `__host__ __device__`, so the same code runs on the CPU (development, tests on a
  machine without an NVIDIA GPU) and on the GPU.

## Status (2026-10-19 evening)

Implemented (`src/cudasynth.cu`, one source for both backends):
- Exact unitary-target branch-and-bound (phase 1 of section 6): exact Z[w] arithmetic, W2/W3 window rules,
  every bound of milq's `lower_bound(strong=True)` (H via sde, H structure, T via the exact PTM sde, CX via
  operator-Schmidt ranks over F_p + the precomputed ILP table, support), exact deduplication of residuals keeping
  the lexicographically smallest prefix (sound together with W2/W3).
- Validated: per-level survivor counts identical to milq's `prune_probe.py`; found circuits re-checked exactly
  (`python -m cudasynth.verify`). Regression: `tests/check.sh`.
- Layout for the GPU:
  - pass 0: each parent's residual S = U p^-1 computed once per level chunk;
  - pass 1: one thread per child (W2/W3, gate, identity check, cheap bounds);
  - pass 2: the expensive T bound only on the compacted children the cheap bounds kept;
  - then compaction, sort by (hash, prefix) and exact unique.
- Storage int32 (coefficients of sqrt2^k U grow like 2^(k/2); surviving residuals have k <= remaining budget
  <= DMAX = 48, so ~2^25), arithmetic in int64, every narrowing checked: overflow aborts the run (exit 4) with
  "rebuild with -DSTORE64". The T bound claims nothing when k is too large for int64 (k + 2n + 6 > 62).
- Tooling: `install.sh` (driver check; nvcc from PATH, /usr/local/cuda or NVIDIA's pip wheels in .venv; build;
  tests), `run.sh` (optimum by increasing budgets, every failed budget a proof, verified QASM out),
  `tests/bench.sh` (timings).

Results (CPU build, 12-thread laptop): Toffoli 9 7.9 s, CCZ 8 7.0 s, Toffoli 10 23 s, Toffoli 11 2.3 min (all
impossible; no SAT run proved any of these). Dedupe removes 6-33% of survivors from level 4 on (Toffoli 11,
level 6: 221,049 -> 147,656).

GPU memory layout (2026-10-19 night, RTX 3060 Ti; per-level counts identical to the previous version on every
test and bench instance):
- First GPU measurement: the T bound took 27 of 28.5 s on Toffoli 11 (4-6 KB of local memory per thread).
- T bound rewritten: units don't change sqrt2-valuations, so for each (fa, fb) one 2-D Walsh-Hadamard transform
  of N^2 products gives all X_ab: N^4 products instead of 32^n, in Z[w] mod 2^32 (exact valuations whenever
  2(n+k) <= 64, else mod 2^64).
- One fused kernel, one warp per child: windows, child residual in shared memory, identity, H / structure /
  support / CX bounds (Gaussian elimination across lanes), T bound (butterflies by shuffles), hash (sum of
  per-entry hashes). No stack, 64 registers. The CPU build keeps the scalar two-pass path.
- Toffoli 10 5.2 s -> 0.54 s, Toffoli 11 30.5 s -> 1.25 s, Toffoli 12 impossible in 7.9 s (new). Remaining
  time at Toffoli 12: kernel 4.7 s, dedupe 1.6 s, parent residuals 0.9 s (`--profile`).

RUS benchmarks (rus-synth/benchmarks/rus, spec.qasm as a full unitary, measurement dropped; optimum with every
smaller budget proven impossible, verified exactly; GPU, seconds each): test 4, RUS 1 11, 2 7, 3 11, 4 18, 5 29,
6 28, 7 7 (RUS 7 uses ccx).

Gate sets with ccx (`lite` in the spec): only the H bound (k; ccx is a permutation) and ceil(T bound / tdiv)
(tdiv = largest PTM sde of one gate: T 1, ccx 2) are sound; CX, support and structure bounds are off. Checked
against bound-free runs (RUS 7, budgets 3-6). A frontier left after the last level is a proof (exit 1, no
longer "open"), and last-level survivors are not stored.

State targets (item 2, partial): `spec.py --ancilla Q.. | --inputs X,..` writes C|in_j> = Ref|in_j> specs; exact
per-pair bounds (sde, stabilizer nullity, Schmidt ranks by exact minors). CPU build only for now - the GPU
kernel aborts on some inputs.

Frontier as a tree (parent index + last gate per node, 5 bytes instead of 72; prefixes read back by walking
up). Children are generated parent-major, so every level is in lexicographic order and dedupe on (hash, index)
keeps the same representative as before; per-level output identical on all test, bench and RUS instances.

Meet in the middle (`--mitm`, unitaries): forward to level ceil(D/2), backward on U^dag (the backward tree is the
table), join on hash(U p^-1) = hash(U R'), candidates confirmed exactly; complete for every length <= D. Same
verdicts as the plain search everywhere tested. Faster when the expensive levels lie past the middle (RUS 14
budget 12: 20 s -> 10 s, RUS 27 budget 14: 71 s -> 63 s); slower when they sit in the middle (RUS 13 budget 16:
214 s plain, 311 s mitm, peak 6 GB - both halves pay the 124M/227M-prefix levels 7-8).

Not implemented yet (in order):
1. ~~GPU memory layout~~ (done, above).
2. RUS / state targets: vector pairs instead of a full unitary, per-pair state bounds (section 5).
3. Global-phase mode.
4. ~~Meet-in-the-middle~~ (done, above; next: choose the split level by frontier size, partition the join).
5. Multi-GPU / cluster split (phase 4).

Each extra depth costs ~6-7x the work (Toffoli 10 -> 11).

## 1. What the algorithm is

Level-synchronous branch-and-bound over gate sequences ("brute force with proofs"):

    frontier_0 = {empty prefix};  for t = 1..D:
      expand   every prefix p in frontier_{t-1} by every allowed gate g (W2 table)        -> p·g
      update   residual S = U (p·g)^-1   (one sparse gate applied to the stored S)
      check    S == I  -> circuit found (depth t)
      bound    lb(S) = H + T + CX + support lower bounds (depth_bounds.py); drop if lb(S) > D - t
      dedupe   keep one prefix per distinct S (transposition table): equal S = equal subtree
      frontier_t = survivors;  empty -> depth D proven impossible

Everything per prefix is independent, so one level is one big batch of identical small jobs: a GPU workload.
The work is *not* large-state simulation: it is 10^5-10^9 tiny exact matrices (4x4 to 32x32).

Meet-in-the-middle (MITM) doubles the reach: tables of forward prefixes P (length <= a) and backward suffixes
S (length <= b), both bound-pruned, joined on U = S·P via hashes of P and S^-1·U (sort-merge or hash join).

## 2. Existing options (what to reuse, what to write)

| option | what it is | fit for us |
|---|---|---|
| cuQuantum cuStateVec, incl. `apply_matrix_batched` [1] | NVIDIA's state-vector simulator; batched API for many small same-size state vectors | partial: batched gate application exists, but floating point only, one library call per gate over the whole batch, and no way to fuse our bounds, hashing or pruning into it. Use for final numerical cross-checks at most |
| Qiskit Aer GPU, qsim, CUDA-Q | single large circuits (20-40 qubits), shots | no: wrong regime (one big state, not 10^8 tiny unitaries) |
| CuPy / PyTorch / JAX | GPU array libraries: batched matmul, int64, sort, unique, compaction; CuPy RawKernel = inline CUDA | yes for the prototype: our numpy code ports almost 1:1; sort/unique give the transposition table |
| Thrust / CUB (CUDA C++) | sort, unique, copy_if (stream compaction), scans | yes for the production version: all the "glue" of a level |
| cuCollections (cuco) | GPU hash sets / maps | yes: transposition table and MITM join without full sorts |
| RAPIDS cuDF | GPU dataframes, hash joins | optional: MITM join on very large tables |
| mjarret/Exact-Synthesis [2] (2026) | CPU (12-core Ryzen), 2-qubit Clifford+T, T-count optimal: MITM + Clifford canonicalisation + exact SO(6) arithmetic in Z[1/sqrt2], concurrent hash sets; T-count 12 from identity in 414 s / 22 GB | reference design for the 2-qubit case (RUS 1-6, 12, 28 are 2-qubit): its packed exact arithmetic and canonicalisation are directly reusable ideas. Not GPU, T-count not gate count |
| Amy-Maslov-Mosca-Roetteler MITM [3]; parallel deterministic walks [4] | the classic exact-synthesis searches (CPU) | algorithms to borrow (MITM, collision finding); [4] parallelises the search over workers |
| Paetznick-Svore RUS direct search [5] | RUS circuits found by brute-force search over {Clifford,T} circuits with one ancilla, canonical single-qubit forms (>10^5 smaller space), split into thousands of cluster jobs | the RUS precedent: exactly this approach, on a cluster, without bounds or GPUs |
| Golubitsky-Maslov 4-bit reversible [6] | optimal 4-bit reversible circuits by canonical-form BFS + MITM (1.2*10^11 circuits up to 9 gates) | canonical forms and storage tricks for large tables |

Not found: a public GPU exact gate-count synthesiser for Clifford+T. GPU work in the area is stabilizer
equivalence checking [7] and numerical instantiation (BQSKit/QFactor, JAX) - neither exact nor enumerative.

**Decision: write the search ourselves, on library primitives.** The core (exact ring arithmetic, gate
application, bounds, hashing) is small and specific; the glue (sort, unique, compaction, hash tables, batched
matmul) comes from CuPy now and Thrust/CUB/cuco later. Off-the-shelf simulators would cost exactness and
fusion and solve the wrong problem size.

## 3. Representation (exact, hashable)

- Matrix entries in Z[w]/sqrt2^k (w = e^{i pi/4}): four integers (a, b, c, d) + one shared exponent k per
  matrix, as bvarr.py already does. int32 suffices for the depths here (coefficients grow ~sqrt2 per H);
  overflow check per level, int64 fallback.
- Sizes per residual: 2 qubits 16 entries (256 B), 3 qubits 64 (1 KB), 4 qubits 256 (4 KB), 5 qubits 1024 (16 KB).
- Canonical key: reduce k while every entry is divisible by sqrt2; global phase fixed when the target allows
  it; then a 128-bit hash of the exact integers (collisions checked on the exact data when merging).
- Gate application is sparse: Paulis/S/T/CX/CZ are permutations with phases (index shuffles), H mixes row
  pairs and raises k by 1. No general matmul needed for the search step.

## 4. Bounds on the GPU (cheap first, exact, sound)

| bound | computation | cost |
|---|---|---|
| H count: sde of the residual | max sqrt2-adic valuation over entries, integer ops | cheap |
| support: qubits the residual touches | per-qubit identity test | cheap |
| CX count: Schmidt ranks across each qubit cut | rank of reshaped matrices (exact Gaussian elimination mod a prime, or float SVD with a safe margin), then a lookup table over the small set of rank patterns (the ILP is precomputed) | moderate |
| T count: sde of the Pauli-transfer matrix (Gosset et al.), exact via the Galois conjugate | batched matmul, 64x64 at 3 qubits, 256x256 at 4 | expensive: run only on prefixes that survive the cheap bounds |
| W2 / W3 windows | table lookups on the last 2-3 gates | cheap |

Soundness is checked like on the CPU: never prune a prefix of a known optimal circuit (the 33-circuit test),
and GPU verdicts must equal the CPU prototype's on every instance where both finish.

## 5. RUS

RUS targets (no measurement, as in fastsynth `-v rus`): the circuit must map each input state |in_j> to its
target |t_j> exactly. The residual is the set of pairs (p|in_j>, |t_j>), and the bounds become the state
versions already in depth_bounds.py (sde per state, stabilizer nullity for T, Schmidt ranks for CX).
MITM keys on the stacked images: forward P|in_j>, backward S^-1|t_j>. RUS instances here have 2-7 qubits and
optimal sizes around 10-20 gates, so MITM is required for the larger ones; the 2-qubit ones (RUS 1-6, 12, 28)
are the first targets, with [2] as a reference. Post-selection (the measured variant) is a later extension.

## 6. Phases

| phase | where | deliverable | gate to pass |
|---|---|---|---|
| 0. CPU reference | laptop, now | numpy-vectorised level evaluation + transposition table in `prune_probe.py`; measure the dedupe factor (distinct residuals / prefixes) | same verdicts as now; speed-up recorded |
| 1. GPU prototype | RTX | CuPy: exact batched representation, sparse gate kernels, cheap bounds, sort/unique dedupe, compaction; T bound via batched matmul on survivors | identical verdicts on CS, Toffoli 7-9, CCZ 7-8, half_adder 7-9, Fredkin 9; >= 100x the Python prototype |
| 2. MITM | RTX | forward/backward tables with pruning, GPU sort-merge or cuco hash join; unitary and RUS (state) keys | finds the known optima (CS 5, relative-phase Toffoli 9, RUS 4 = 18, RUS 8 = 11, RUS 14 = 12) |
| 3. Production kernels | RTX | hot loops in CUDA C++ (Thrust/CUB/cuco), int32 packed arithmetic, fused expand+bound+hash kernel | >= 10x over phase 1 on the largest levels |
| 4. Cluster | university cluster | subtree split at a shallow level into independent jobs (Slurm array, no communication; cross-job duplicates accepted); MITM join partitioned by hash bits | open depths: Toffoli 10+, CCZ 9+, Grover_3 10, RUS 5/6/13/17/21 |
| 5. Integration | prototype | as a lower-bound oracle for the SAT walk and as the pruning source of a SAT propagator (§3.19 plan) | walk results improve on the 71 suite |

## 7. Risks

- Exponential growth stays: ~13x per level at 3 qubits until the bounds bite. Hardware buys about 2-3 levels;
  dedupe and MITM buy more.
- Weak bounds (CCZ starts at 4, RUS state bounds) mean more levels before pruning.
- Finding circuits (SAT side) gets less pruning near the leaves than proving impossibility; MITM is the remedy.
- Qubit count: 5+ qubits make the exact T bound (1024x1024 transfer matrix) too expensive; use the state
  bounds there.
- MITM memory: 10^8-10^9 keys of 16 B fit an RTX card's memory only partly; partition by hash bits, spill to host/disk.

## Sources

1. NVIDIA cuQuantum, custatevec `apply_matrix_batched`: https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/overview/index.html
2. High-Performance Exact Synthesis of Two-Qubit Quantum Circuits (2026): https://arxiv.org/abs/2601.19166, code https://github.com/mjarret/Exact-Synthesis
3. Amy, Maslov, Mosca, Roetteler, A meet-in-the-middle algorithm for fast synthesis of depth-optimal quantum circuits: https://arxiv.org/abs/1206.0758
4. Di Matteo, Mosca, Parallelizing quantum circuit synthesis: https://arxiv.org/abs/1606.07413
5. Paetznick, Svore, Repeat-Until-Success: non-deterministic decomposition of single-qubit unitaries: https://arxiv.org/abs/1311.1074
6. Golubitsky, Maslov, A study of optimal 4-bit reversible Toffoli circuits and their synthesis: https://arxiv.org/abs/1103.2686
7. GPU-accelerated simulation / equivalence checking of stabilizer circuits: https://arxiv.org/abs/2603.14641
