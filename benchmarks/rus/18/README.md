# rus/18: Comparator rotation, n = 4, k = 11

- **Gate on success:** (8 − 3iX)/√73
- **Measurement:** success when every ancilla reads 0; q[1]–q[6] in the Z basis; q[1]–q[4] are the register, q[5] and q[6] work qubits
- **Qubits:** 1 target(s) q[0], 6 ancilla(s)
- **Success probability:** 73/128
- **Spec cost:** 6 `ccx`
- **On failure:** Z, per the paper (not checked)
- **Source:** [Hindlycke & Larsson, Phys. Rev. Research 6, L042027 (2024), arXiv:2404.05618](https://arxiv.org/abs/2404.05618), Fig. 1b

The spec uses `ccx`, so (as with `rus/7`) `ccx` joins the synthesis gate set.

```console
./milq -v rus -T 1 -A 6 -c Classic -s gurobi benchmarks/rus/18/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
