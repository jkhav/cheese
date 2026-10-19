# rus/26: Quantum neuron, two inputs (AND)

- **Gate on success:** Ry(π − g) on q[2] when q[0] = q[1] = 1, otherwise Ry(g); g = 2·atan(tan²(π/8))
- **Measurement:** success when every ancilla reads 0; q[3] in the Z basis
- **Qubits:** 3 target(s) q[0]–q[2], 1 ancilla(s)
- **Success probability:** 3/4
- **Spec cost:** 10 T
- **On failure:** as `rus/25`
- **Source:** [Cao, Guerreschi & Aspuru-Guzik, arXiv:1711.11240](https://arxiv.org/abs/1711.11240); [Moreira et al., arXiv:2212.10742](https://arxiv.org/abs/2212.10742), Fig. 3, with w₁ = w₂ = π/2 and b = −π/4

```console
./milq -v rus -T 3 -A 1 -c Classic -s gurobi benchmarks/rus/26/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
