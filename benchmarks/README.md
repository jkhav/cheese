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

Not run yet: RUS 9, 21, 22, 23, 25, 26 (4 qubits: need an NQ=4 build) and RUS 10, 11, 15-19, 24 (5-7 qubits:
not supported by the unitary search).
