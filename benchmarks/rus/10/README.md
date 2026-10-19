# rus/10: Magic-state ladder, rung 3

- **Gate on success:** Rz(θ), tan(θ/2) = (√2−1)⁴, θ ≈ 0.0589
- **Measurement:** success when every ancilla reads 0; all of q[1]–q[4] in the Z basis
- **Qubits:** 1 target(s) q[0], 4 ancilla(s)
- **Success probability:** 17/64
- **Spec cost:** 4 T
- **On failure:** not Pauli (see `rus/9`)
- **Source:** [Duclos-Cianci & Svore, arXiv:1210.1980](https://arxiv.org/abs/1210.1980), Figs. 1–2; also used in [Bocharov, Gurevich & Svore, arXiv:1303.1411](https://arxiv.org/abs/1303.1411), App. A

One more resource state and parity check than `rus/9`.

```console
./milq -v rus -T 1 -A 4 -c Classic -s gurobi benchmarks/rus/10/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
