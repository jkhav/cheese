# rus/28: T by gate teleportation

- **Gate on success:** T
- **Measurement:** success when every ancilla reads 0; q[1] in the X basis
- **Qubits:** 1 target(s) q[0], 1 ancilla(s)
- **Success probability:** 1/2
- **Spec cost:** 1 T
- **On failure:** Z·T, so a Z fixes it (the circuit is effectively deterministic)
- **Source:** [NTU-ALComLab/SliQEC](https://github.com/NTU-ALComLab/SliQEC), `PEC_benchmarks/RUS_T_1.qasm`

The original ends with `swap q[0], q[1]`; it is left out so the ancilla is the high qubit. The target is itself one T gate, so this is a smoke test.

```console
./milq -v rus -T 1 -A 1 -c Classic -s gurobi benchmarks/rus/28/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
