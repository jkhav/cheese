# cudasynth

Exact Clifford+T circuit synthesis by GPU branch-and-bound: enumerate gate sequences level by level, prune every
prefix whose remaining work provably exceeds the depth budget, and either find a circuit or prove that none of
the given depth exists. Targets: unitaries, state maps, and repeat-until-success (RUS) circuits without
measurement. See [docs/PLAN.md](docs/PLAN.md).

Status: early development. The CPU reference prototype lives in milq (`misc/prototypes/prune_probe.py`).

## Build

    ./build.sh cpu                 # g++ + OpenMP; Thrust headers from the CUDA toolkit or CCCL=/path/to/cccl
    ARCH=sm_89 ./build.sh cuda     # nvcc; ARCH = your GPU (sm_86 RTX 30xx, sm_89 RTX 40xx, sm_120 RTX 50xx)

For a CPU-only machine, `pip install nvidia-cuda-cccl` into `.venv` provides the headers (build.sh finds them).

## On a machine with an NVIDIA GPU

    nvidia-smi && nvcc --version            # driver + CUDA toolkit (12.x or 13.x)
    git clone git@github.com:jkhav/cudasynth.git && cd cudasynth
    NQ="2 3" ARCH=native ./build.sh cuda    # 2-3 qubits keeps the nvcc build short; NQ="1 4" for everything
    tests/check.sh                          # known verdicts, found circuits verified exactly
    tests/bench.sh                          # timings on the proofs SAT could not reach

Python >= 3.8, standard library only (the spec generator and the verifier).

## Run

    cd python && python3 -m cudasynth.spec TARGET.qasm DEPTH > ../target.spec
    ./build/cudasynth target.spec [--depth D] [--no-dedupe] [--chunk PARENTS]

One JSON line per level (prefixes examined, cut by W3 / by the bounds, survivors, distinct residuals). Exit code
0: a circuit was found (printed), 1: no circuit of <= D gates exists (proof), 2: open.

Validated against milq's `prune_probe.py` (identical survivor counts per level, same circuits): CS 4/5,
Toffoli 7/8/9, CCZ 8. CPU build on a 12-thread laptop: Toffoli 9 in 7.9 s, CCZ 8 in 7.0 s (Python prototype:
175 s and 163 s on 10 processes; SAT: no proof within its runs).
