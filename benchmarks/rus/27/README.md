# rus/27: Distorted conditional gearbox

- **Gate on success:** not unitary: I when q[0] = 0, (1/√2)·Ry(π/2) on q[1] when q[0] = 1
- **Measurement:** success when every ancilla reads 0; q[2] in the Z basis
- **Qubits:** 2 target(s) q[0]–q[1], 1 ancilla(s)
- **Success probability:** 1 for q[0] = 0, 1/2 for q[0] = 1
- **Spec cost:** 4 T
- **On failure:** not checked
- **Source:** [Cao, Guerreschi & Aspuru-Guzik, arXiv:1711.11240](https://arxiv.org/abs/1711.11240); [Moreira et al., arXiv:2212.10742](https://arxiv.org/abs/2212.10742), Fig. 1a (b = 0, w = π/2); [Guerreschi, arXiv:1808.02900](https://arxiv.org/abs/1808.02900), §III

The success probability depends on the input, so after one success a superposed input is distorted. MILQ compares the normalised outputs per input state, and the \|+⟩ input sees the distortion. Use this one only if that is what you want to test.

```console
./milq -v rus -T 2 -A 1 -c Classic -s gurobi benchmarks/rus/27/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
