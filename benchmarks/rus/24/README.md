# rus/24: Post-selected C³X

- **Gate on success:** C³X, controls q[0]–q[2], target q[3]
- **Measurement:** success when every ancilla reads 0; q[4] and q[5] (two chained temporary ANDs) in the X basis
- **Qubits:** 4 target(s) q[0]–q[3], 2 ancilla(s)
- **Success probability:** 1/4
- **Spec cost:** 8 T
- **On failure:** not checked
- **Source:** [Jones, arXiv:1212.5069](https://arxiv.org/abs/1212.5069), Fig. 1b, with the temporary AND of [Gidney, arXiv:1709.06648](https://arxiv.org/abs/1709.06648), Fig. 3

No exact Clifford+T circuit for C³X exists on 4 qubits without ancillas (it fails the determinant condition), so this also needs the ancillas.

```console
./milq -v rus -T 4 -A 2 -c Classic -s gurobi benchmarks/rus/24/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
