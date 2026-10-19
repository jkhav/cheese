# rus/11: Magic-state ladder, rung 4

- **Gate on success:** Rz(θ), tan(θ/2) = (√2−1)⁵, θ ≈ 0.0244; equal to (5 + 2iZ)/√29 times an odd power of T
- **Measurement:** success when every ancilla reads 0; all of q[1]–q[5] in the Z basis
- **Qubits:** 1 target(s) q[0], 5 ancilla(s)
- **Success probability:** 29/128
- **Spec cost:** 5 T
- **On failure:** not Pauli (see `rus/9`)
- **Source:** [Duclos-Cianci & Svore, arXiv:1210.1980](https://arxiv.org/abs/1210.1980), Figs. 1–2; also used in [Bocharov, Gurevich & Svore, arXiv:1303.1411](https://arxiv.org/abs/1303.1411), App. A

```console
./milq -v rus -T 1 -A 5 -c Classic -s gurobi benchmarks/rus/11/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
