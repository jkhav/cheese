# rus/20: Fan-out RUS, two targets

- **Gate on success:** (I + i√2·X⊗X)/√3 on q[0], q[1]
- **Measurement:** success when every ancilla reads 0; q[2] in the Z basis
- **Qubits:** 2 target(s) q[0]–q[1], 1 ancilla(s)
- **Success probability:** 3/4
- **Spec cost:** 2 T
- **On failure:** I
- **Source:** [munich-quantum-toolkit/core](https://github.com/munich-quantum-toolkit/core), `mlir/bench/programs/RepeatUntilSuccess.cpp`

The two-qubit version of the `rus/2` gate (conjugated by CNOTs). Run with `-T 2`.

```console
./milq -v rus -T 2 -A 1 -c Classic -s gurobi benchmarks/rus/20/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
