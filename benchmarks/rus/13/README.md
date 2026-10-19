# rus/13: Gearbox, two controls

- **Gate on success:** Rx(2·atan(tan²(asin(sin²(π/8))))) ≈ Rx(0.0439)
- **Measurement:** success when every ancilla reads 0; q[1] and q[2] in the Z basis
- **Qubits:** 1 target(s) q[0], 2 ancilla(s)
- **Success probability:** ≈ 0.958
- **Spec cost:** 1 `ccx` + 7 T
- **On failure:** not checked
- **Source:** [Wiebe & Kliuchnikov, arXiv:1305.5528](https://arxiv.org/abs/1305.5528), Fig. 1 (C⁽²⁾(HTH, HTH)); [Wiebe & Roetteler, arXiv:1406.2040](https://arxiv.org/abs/1406.2040), Fig. 2, Eq. 1

Doubly controlled −iX, written as `ccx` plus a controlled-S† on the controls. A small angle is new to this set. The spec uses `ccx`, so (as with `rus/7`) `ccx` joins the synthesis gate set.

```console
./milq -v rus -T 1 -A 2 -c Classic -s gurobi benchmarks/rus/13/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
