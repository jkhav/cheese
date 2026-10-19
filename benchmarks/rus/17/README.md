# rus/17: Comparator rotation, n = 3, k = 7

- **Gate on success:** (4 − 3iX)/5, Clifford-equivalent to V₃²
- **Measurement:** success when every ancilla reads 0; q[1]–q[4] in the Z basis (the `h` gates make q[1]–q[3] X-basis)
- **Qubits:** 1 target(s) q[0], 4 ancilla(s)
- **Success probability:** 25/32
- **Spec cost:** 4 `ccx`
- **On failure:** Clifford (checked for all 7 failure outcomes)
- **Source:** [Hindlycke & Larsson, Phys. Rev. Research 6, L042027 (2024), arXiv:2404.05618](https://arxiv.org/abs/2404.05618), Fig. 1b

The spec uses `ccx`, so (as with `rus/7`) `ccx` joins the synthesis gate set.

```console
./milq -v rus -T 1 -A 4 -c Classic -s gurobi benchmarks/rus/17/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
