# Benchmarks (copied from milq)

Every circuit from milq's `benchmarks/` (QASM + READMEs), unchanged:
`rus/` (RUS 1-28: `spec.qasm` input, `synth.qasm` / `optimal-nm.qasm` known results, README per instance),
`gates/`, `ghz/`, `qec/`, `mqt-bench/`, `bernstein-vazirani/`.

`rus/found-nm/<config>/<n>.qasm`: milq's own results for RUS without measurement (fastsynth `-v rus`,
misc/prototypes/rus_run.py), one folder per solver configuration; each was checked with Qiskit statevectors.

Run as a unitary: `./run.sh benchmarks/rus/7/spec.qasm`. As a state target (RUS without measurement, CPU build only
for now): `./run.sh benchmarks/rus/7/spec.qasm --cpu --ancilla 1 2`. Note milq's `-v rus` also requires |+..+>|0..>
as an input; `--ancilla` uses the basis inputs only, so its optima can be lower.
