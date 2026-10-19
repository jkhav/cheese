# cudasynth

Exact Clifford+T circuit synthesis by GPU branch-and-bound: enumerate gate sequences level by level, prune every
prefix whose remaining work provably exceeds the depth budget, and either find a circuit or prove that none of
the given depth exists. Targets: unitaries, state maps, and repeat-until-success (RUS) circuits without
measurement. See [docs/PLAN.md](docs/PLAN.md).

Status: early development. The CPU reference prototype lives in milq (`misc/prototypes/prune_probe.py`).

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
Timings: `tests/bench.sh`. Low level: `python3 -m cudasynth.spec T.qasm D > t.spec; build/cudasynth t.spec`
(exit 0 circuit found, 1 proof, 2 open).

Validated against milq's `prune_probe.py` (identical survivor counts per level, same circuits): CS 4/5,
Toffoli 7/8/9, CCZ 8. CPU build on a 12-thread laptop: Toffoli 9 in 7.9 s, CCZ 8 in 7.0 s (Python prototype:
175 s and 163 s on 10 processes; SAT: no proof within its runs); Toffoli 10 in 23 s and Toffoli 11 in 2.5 min.
