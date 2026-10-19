# rus/14: PAR, two inputs

- **Gate on success:** Rx(2·atan(tan²(π/8))), the same gate as `rus/12`
- **Measurement:** success when every ancilla reads 0; q[1] and q[2] in the GHZ basis (`cx q[1],q[2]; h q[1]` before `measure`)
- **Qubits:** 1 target(s) q[0], 2 ancilla(s)
- **Success probability:** 3/8
- **Spec cost:** 1 `ccx` + 5 T
- **On failure:** not checked
- **Source:** [Wiebe & Roetteler, arXiv:1406.2040](https://arxiv.org/abs/1406.2040), Fig. 3 (programmable ablation of rotations, PAR), Eq. 18: GB(φ) = PAR(φ, φ)

The same gate as the gearbox with a different circuit and half the success probability. CC(iX) has a 4-T phase polynomial, so 6 T in total is possible. The spec uses `ccx`, so (as with `rus/7`) `ccx` joins the synthesis gate set.

```console
./milq -v rus -T 1 -A 2 -c Classic -s gurobi benchmarks/rus/14/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
