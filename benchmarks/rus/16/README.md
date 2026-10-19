# rus/16: Comparator rotation, n = 3, k = 5

- **Gate on success:** (4 − iZ)/√17 (up to phase diag(1, (15 + 8i)/17)), the inverse of the gate `rus/5` should implement
- **Measurement:** success when every ancilla reads 0; q[1]–q[3] (comparator register, prepared in \|+⟩) and q[4] (work qubit holding an AND, measured in the X basis) all read 0
- **Qubits:** 1 target(s) q[0], 4 ancilla(s)
- **Success probability:** 17/64
- **Spec cost:** 8 T
- **On failure:** Z, per the paper
- **Source:** [Hindlycke & Larsson, Phys. Rev. Research 6, L042027 (2024), arXiv:2404.05618](https://arxiv.org/abs/2404.05618), Fig. 1b

A different circuit for a (4 ± iZ)/√17 gate than `rus/5` (Paetznick–Svore Fig. 10b): 5 qubits, but 8 T instead of 11. The work AND uses Gidney's 4-T construction.

```console
./milq -v rus -T 1 -A 4 -c Classic -s gurobi benchmarks/rus/16/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
