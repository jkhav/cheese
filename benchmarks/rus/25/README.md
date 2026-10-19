# rus/25: Quantum neuron, two inputs (OR)

- **Gate on success:** Ry(g = 2·atan(tan²(π/8))) on q[2] when q[0] = q[1] = 0, otherwise Ry(π − g); g ≈ 0.340
- **Measurement:** success when every ancilla reads 0; q[3] in the Z basis
- **Qubits:** 3 target(s) q[0]–q[2], 1 ancilla(s)
- **Success probability:** 3/4
- **Spec cost:** 10 T
- **On failure:** a Clifford −π/2 Y rotation on q[2], independent of the inputs (Moreira et al., Eq. 1)
- **Source:** [Cao, Guerreschi & Aspuru-Guzik, arXiv:1711.11240](https://arxiv.org/abs/1711.11240); [Moreira et al., arXiv:2212.10742](https://arxiv.org/abs/2212.10742), Fig. 3, with weights w₁ = w₂ = π/2 and bias b = π/4

Equal weights make the success probability input-independent, so the map is unitary. I believe the rotation's double control rules out Clifford equivalence to the gearbox ⊗ I, but have not proved it.

```console
./milq -v rus -T 3 -A 1 -c Classic -s gurobi benchmarks/rus/25/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
