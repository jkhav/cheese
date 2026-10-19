# rus/22: Post-selected Toffoli

- **Gate on success:** Toffoli, controls q[0], q[1], target q[2]
- **Measurement:** success when every ancilla reads 0; q[3] (the temporary AND) in the X basis
- **Qubits:** 3 target(s) q[0]–q[2], 1 ancilla(s)
- **Success probability:** 1/2
- **Spec cost:** 4 T
- **On failure:** Toffoli·CZ(q[0], q[1]), fixed by a Clifford
- **Source:** [Jones, arXiv:1212.5069](https://arxiv.org/abs/1212.5069), Fig. 1b, with the temporary AND of [Gidney, arXiv:1709.06648](https://arxiv.org/abs/1709.06648), Fig. 3

The best unitary Toffoli without ancillas needs 7 T; this one needs 4.

```console
./milq -v rus -T 3 -A 1 -c Classic -s gurobi benchmarks/rus/22/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
