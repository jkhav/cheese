# cheese

**C**lifford+T **H**ardware-accelerated **E**xact **E**numerative **S**ynthesis **E**ngine, a companion to
[milq](https://github.com/QuantumFIT/milq).

Exact Clifford+T circuit synthesis by GPU branch-and-bound: enumerate gate sequences level by level, prune every
prefix whose remaining work provably exceeds the depth budget, and either find a circuit or prove that none of
the given depth exists. Targets: unitaries, state maps, and repeat-until-success (RUS) circuits without
measurement. See [docs/PLAN.md](docs/PLAN.md).

Status: early development using AI agents. How the search and its pruning work: [docs/ALGORITHM.md](docs/ALGORITHM.md).

## Notable results

All proofs are exhaustive searches with sound lower bounds; every circuit is re-checked exactly. RTX 3060 Ti.
Details: [benchmarks/README.md](benchmarks/README.md), [benchmarks/optimal/README.md](benchmarks/optimal/README.md),
[docs/PLAN.md](docs/PLAN.md).

**Optimal gate counts, proved** (every smaller budget impossible; global phase included):

| target | input gates | proved optimum | time |
|---|---|---|---|
| Toffoli | 15 | **15** (every circuit of <= 14 gates ruled out) | 27 s with `--table 8` |
| Fredkin | 17 | **17** (<= 16 ruled out) | ~100 s |
| RUS 1 / 3 / 4 / 5 / 6 | 13 / 12 / 20 / 37 / 33 | **11 / 11 / 18 / 29 / 28** | 4-21 s |
| RUS 7 / 13 / 14 / 27 (3 qubits) | 9 / 18 / 14 / 20 | **7 / 16 / 12 / 14** | 12-269 s |
| RUS 2, 8, 12, 20, 28 | 7, 11, 8, 9, 3 | same as input: input proved optimal | 2-13 s |

For context: milq's SAT runs proved Toffoli impossible up to 8 gates, its Python branch-and-bound prototype up to
10 (Fredkin 9); this search reaches 14 (Fredkin 16). RUS circuits are taken as their full
unitary (measurement dropped), so a circuit that is only right on the success branch can be shorter.

**Optimal T-counts, proved** (up to global phase):
- RUS 7: 14 -> **4** (two Toffolis replaced), RUS 13: 14 -> **8**, RUS 14: 12 -> **6** (ccx counted as 7 T).
- RUS 8: T-count 4, described as the best known in its benchmark README, is now proved optimal.
- The other RUS instances run (2-3 qubits: 1-6, 12, 20, 27, 28): the benchmark circuit's T-count proved optimal.
  The 4-7 qubit instances (9-11, 15-19, 21-26) are not run yet.
- Toffoli, Fredkin, Peres, CCZ 7 (the known optimum, as a check), relative-phase Toffoli 4, CS 3, QFT_2 3,
  about 1 s each once the coset table is built (41 s, cached).

**Fewer Clifford gates at optimal T-count** (Clifford mode, below; segments optimal, total not proved minimal):
Toffoli 28 -> 20 gates, Fredkin 67 -> 21, Peres 28 -> 15, CCZ 24 -> 16, relative-phase Toffoli 29 -> 12; CS 5 and
QFT_2 7 reach their gate-count optimum. The whole 3-qubit Clifford group (92,897,280 elements) is tabulated in
about 3 s on the GPU.

**Speed:** Toffoli 11 proved impossible in 0.5 s on the GPU (2.3 min on the CPU build), Toffoli 12 in 7.9 s.

## Install and build

    ./install.sh              # checks the NVIDIA driver, finds nvcc (or installs NVIDIA's CUDA compiler wheels into
                              # .venv, no root), builds the GPU + CPU binaries, runs tests/check.sh
    ./install.sh --cpu-only   # any machine without an NVIDIA GPU
    NQ="1 4" ./install.sh     # compile 1-4 qubit support (default 2-3; nvcc takes a few minutes per qubit count)

The NVIDIA driver must already be installed (`nvidia-smi` works). Manual builds: `./build.sh cpu|cuda`
(settings in build/config.sh, written by install.sh).

## Run

    ./run.sh TARGET.qasm                # optimal circuit: budgets 1, 2, ... until one is found; every failed budget is a proof
    ./run.sh TARGET.qasm --from 9       # start at budget 9 (when smaller budgets are already known impossible)
    ./run.sh TARGET.qasm --depth 10     # one budget: a circuit of <= 10 gates, or a proof that none exists
    ./run.sh TARGET.qasm --cpu          # CPU binary even when the GPU one exists

Found circuits are re-checked exactly and written to results/<name>-opt.qasm; per-level statistics (one JSON
line per level: prefixes, cut by W3 / by the bounds, survivors, distinct residuals) to results/<name>-d<D>.log.
Timings: `tests/bench.sh`. Low level: `python3 -m cheese.spec T.qasm D > t.spec; build/cheese t.spec`
(exit 0 circuit found, 1 proof, 2 open).

Validated against milq's `prune_probe.py` (identical survivor counts per level, same circuits): CS 4/5,
Toffoli 7/8/9, CCZ 8. CPU build on a 12-thread laptop: Toffoli 9 in 7.9 s, CCZ 8 in 7.0 s (Python prototype:
175 s and 163 s on 10 processes; SAT: no proof within its runs); Toffoli 10 in 23 s and Toffoli 11 in 2.5 min.

## T-count mode

    ./build.sh tcount                   # separate binary build/tcount (src/tcount.cu)
    ./tcount.sh TARGET.qasm             # optimal T-count up to global phase, decomposition verified exactly

Toffoli, Fredkin, Peres, CCZ (T-count 7) take about 1 s each on an RTX 3060 Ti once the 3-qubit coset table
(41 s, cached in build/) exists. Method in [docs/ALGORITHM.md](docs/ALGORITHM.md) section 10, results in
docs/PLAN.md and benchmarks/README.md.

## Clifford mode

    ./build.sh cliffopt                                                    # build/cliffopt (src/clifford.cu)
    python3 -m cheese.tcircuit TARGET.qasm DECOMP.json OUT.qasm --opt     # run from python/; needs Qiskit

Turns a T-count decomposition (from `./tcount.sh`) into an explicit circuit with exactly that many T gates, then
replaces every Clifford segment between T gates by an optimal one from a GPU table of the whole Clifford group
(up to 3 qubits) and beam-searches where the T gates sit to make the total small. Every result is checked
exactly. Method: [docs/ALGORITHM.md](docs/ALGORITHM.md) section 10.


MIT, see [LICENSE](LICENSE). Benchmark circuits in `benchmarks/` come from milq and the sources named in
their READMEs (MQT Bench, published RUS papers).
