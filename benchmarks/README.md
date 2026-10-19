# Benchmarks (copied from milq)

Every circuit from milq's `benchmarks/` (QASM + READMEs), unchanged:
`rus/` (RUS 1-28: `spec.qasm` input, `synth.qasm` / `optimal-nm.qasm` known results, README per instance),
`gates/`, `ghz/`, `qec/`, `mqt-bench/`, `bernstein-vazirani/`.

`rus/found-nm/<config>/<n>.qasm`: milq's own results for RUS without measurement (fastsynth `-v rus`,
misc/prototypes/rus_run.py), one folder per solver configuration; each was checked with Qiskit statevectors.

Not yet searchable here: cudasynth's spec only has `mode unitary`; RUS needs the state-pair mode (docs/PLAN.md §5).
