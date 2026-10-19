# rus/12: Gearbox, one control

- **Gate on success:** Rx(2·atan(tan²(π/8))) ∝ I − i·tan²(π/8)·X
- **Measurement:** success when every ancilla reads 0; q[1] in the Z basis
- **Qubits:** 1 target(s) q[0], 1 ancilla(s)
- **Success probability:** 3/4
- **Spec cost:** 2 T
- **On failure:** not checked
- **Source:** [Wiebe & Kliuchnikov, arXiv:1305.5528](https://arxiv.org/abs/1305.5528), Fig. 1 (C⁽¹⁾(HTH)); [Wiebe & Roetteler, arXiv:1406.2040](https://arxiv.org/abs/1406.2040), Fig. 2

Built from the figure: the control gets U = HTH, then controlled −iX (`cx` plus `sdg` on the control), then U†. Same gate as `rus/2` up to Clifford: 2·atan(√2) − π/2 = 2·atan(tan²(π/8)).

```console
./milq -v rus -T 1 -A 1 -c Classic -s gurobi benchmarks/rus/12/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
