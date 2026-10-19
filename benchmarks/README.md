# Benchmarks (copied from milq)

Every circuit from milq's `benchmarks/` (QASM + READMEs), unchanged:
`rus/` (RUS 1-28: `spec.qasm` input, `synth.qasm` / `optimal-nm.qasm` known results, README per instance),
`gates/`, `ghz/`, `qec/`, `mqt-bench/`, `bernstein-vazirani/`.

`rus/found-nm/<config>/<n>.qasm`: milq's own results for RUS without measurement (`-v rus`), one folder per solver
configuration; each was checked with Qiskit statevectors.

Run as a unitary: `./run.sh benchmarks/rus/7/spec.qasm`. As a state target (RUS without measurement, CPU build only
for now): `./run.sh benchmarks/rus/7/spec.qasm --cpu --ancilla 1 2`. Note milq's `-v rus` also requires |+..+>|0..>
as an input; `--ancilla` uses the basis inputs only, so its optima can be lower.

## Results: RUS as unitaries (cudasynth, RTX 3060 Ti)

`rus/<n>/unitary-opt.qasm`: a circuit with the fewest gates whose unitary equals `spec.qasm`'s exactly
(measurements dropped, global phase included; gate set Clifford+T plus the spec's own gates, so ccx counts as
one gate where the spec uses it). Optimal: `./run.sh` tried budgets 1, 2, ... and proved every budget below the
optimum impossible; each circuit was re-checked exactly (`python -m cudasynth.verify`). "total time" covers all
budgets, "last budget" the one that found the circuit.

| RUS | qubits | spec gates | unitary optimum | total time | last budget | gate set |
|---|---|---|---|---|---|---|
| 1 | 2 | 13 | 11 | 5 s | 0.3 s | Clifford+T |
| 2 | 2 | 7 | 7 | 3 s | 0.2 s | Clifford+T |
| 3 | 2 | 12 | 11 | 4 s | 0.3 s | Clifford+T |
| 4 | 2 | 20 | 18 | 8 s | 0.5 s | Clifford+T |
| 5 | 2 | 37 | 29 | 21 s | 5.5 s | Clifford+T |
| 6 | 2 | 33 | 28 | 15 s | 2.2 s | Clifford+T |
| 7 | 3 | 9 | 7 | 12 s | 0.3 s | Clifford+T + ccx |
| 8 | 3 | 11 | 11 | 13 s | 0.5 s | Clifford+T |
| 12 | 2 | 8 | 8 | 3 s | 0.3 s | Clifford+T |
| 13 | 3 | 18 | 16 | 269 s | 214.4 s | Clifford+T + ccx |
| 14 | 3 | 14 | 12 | 43 s | 19.2 s | Clifford+T + ccx |
| 20 | 3 | 9 | 9 | 10 s | 0.3 s | Clifford+T |
| 27 | 3 | 20 | 14 | 100 s | 71.3 s | Clifford+T |
| 28 | 2 | 3 | 3 | 2 s | 0.2 s | Clifford+T |

## Results: optimal T-count of the RUS unitaries (cudasynth T-count mode)

`./tcount.sh rus/<n>/spec.qasm`: the fewest T gates of any Clifford+T circuit with exactly the spec's unitary
(measurements dropped, all branches, global phase ignored), proved optimal and verified exactly. Circuit
columns count T/Tdg gates, each ccx as 7 (its optimal T-count). RTX 3060 Ti, tables cached (built once: 2 qubits
T-count <= 6, 3 qubits T-count <= 4).

| RUS | qubits | spec.qasm | synth.qasm | optimal-nm.qasm | unitary-opt.qasm (gate count) | **optimal T-count** | time |
|---|---|---|---|---|---|---|---|
| 28 | 2 | 1 | - | - | 1 | **1** | 0.2 s |
| 2 | 2 | 2 | 2 | - | 2 | **2** | 0.2 s |
| 12 | 2 | 2 | - | - | 2 | **2** | 0.2 s |
| 20 | 3 | 2 | - | - | 2 | **2** | 0.4 s |
| 1 | 2 | 4 | 4 | - | 4 | **4** | 0.2 s |
| 3 | 2 | 4 | 4 | - | 4 | **4** | 0.2 s |
| 27 | 3 | 4 | - | - | 4 | **4** | 0.3 s |
| 8 | 3 | 4 | - | 4 | 4 | **4** (README: best known; now proved optimal) | 0.4 s |
| 4 | 2 | 6 | - | 6 | 6 | **6** | 0.2 s |
| 5 | 2 | 11 | - | - | 11 | **11** | 0.3 s |
| 6 | 2 | 11 | - | - | 11 | **11** | 0.3 s |
| 14 | 3 | 12 (5 T + ccx) | - | 16 (2 T + 2 ccx) | 16 (2 T + 2 ccx) | **6** (improved) | 0.3 s |
| 13 | 3 | 14 (7 T + ccx) | - | - | 18 (4 T + 2 ccx) | **8** (improved) | 44 s |
| 7 | 3 | 14 (2 ccx) | 14 (2 ccx) | - | 14 (2 ccx) | **4** (improved) | 0.4 s |

Each decomposition is stored as `rus/<n>/tcount.json` (format in optimal/README.md; check with
`python -m cudasynth.tcount verify rus/<n>/spec.qasm rus/<n>/tcount.json`) and as an explicit circuit
`rus/<n>/tcount-opt.qasm` with exactly that many T gates, equal to the spec up to global phase, Clifford gates
reduced by `python -m cudasynth.tcircuit ... --opt` (optimal Clifford segments, beam-searched T placement; the
total is not proved minimal). Total gates of these circuits (T-count optimal) vs the gate-count optimum
(`unitary-opt.qasm`, whose T-count can be higher):

| RUS | 28 | 2 | 12 | 1 | 3 | 4 | 6 | 5 | 7 | 8 | 20 | 14 | 27 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T-count | 1 | 2 | 2 | 4 | 4 | 6 | 11 | 11 | 4 | 4 | 2 | 6 | 4 | 8 |
| tcount-opt.qasm gates | 3 | 7 | 11 | 11 | 12 | 18 | 30 | 29 | 15 | 15 | 9 | 19 | 17 | 23 |
| unitary-opt.qasm gates | 3 | 7 | 8 | 11 | 11 | 18 | 28 | 29 | 7 (14 T via ccx) | 11 | 9 | 12 (16 T) | 14 | 16 (18 T) |

The T-count is of the full unitary; a RUS circuit only has to be right on its success branch, so its own optimum
can be lower. The output is a decomposition into Pauli rotations and a Clifford, not yet an explicit gate list.

Not run yet: RUS 9, 21, 22, 23, 25, 26 (4 qubits: need an NQ=4 build) and RUS 10, 11, 15-19, 24 (5-7 qubits:
not supported by the unitary search).
