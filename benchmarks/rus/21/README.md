# rus/21: Fan-out RUS, three targets

- **Gate on success:** (I + i√2·X⊗X⊗X)/√3 on q[0]–q[2]
- **Measurement:** success when every ancilla reads 0; q[3] in the Z basis
- **Qubits:** 3 target(s) q[0]–q[2], 1 ancilla(s)
- **Success probability:** 3/4
- **Spec cost:** 2 T
- **On failure:** I
- **Source:** [munich-quantum-toolkit/core](https://github.com/munich-quantum-toolkit/core), `mlir/bench/programs/RepeatUntilSuccess.cpp`

Run with `-T 3`.

```console
./milq -v rus -T 3 -A 1 -c Classic -s gurobi benchmarks/rus/21/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
