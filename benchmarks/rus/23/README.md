# rus/23: Post-selected CCH

- **Gate on success:** controlled-controlled-H, controls q[0], q[1], target q[2]
- **Measurement:** success when every ancilla reads 0; q[3] in the X basis
- **Qubits:** 3 target(s) q[0]–q[2], 1 ancilla(s)
- **Success probability:** 1/2
- **Spec cost:** 6 T
- **On failure:** not checked
- **Source:** [Jones, arXiv:1212.5069](https://arxiv.org/abs/1212.5069), Fig. 1b, with the temporary AND of [Gidney, arXiv:1709.06648](https://arxiv.org/abs/1709.06648), Fig. 3; Jones §II (the CNOT replaced by a controlled gate)

The controlled-H is qelib1's `ch` expansion (2 T).

```console
./milq -v rus -T 3 -A 1 -c Classic -s gurobi benchmarks/rus/23/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
