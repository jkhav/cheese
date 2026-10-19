# rus/8: V₃ with two ancillas

- **Gate on success:** V₃ = (I + 2iZ)/√5
- **Measurement:** success when every ancilla reads 0; q[1] and q[2] start in \|+⟩ (the `h` gates) and are measured in the X basis (`h` before `measure`)
- **Qubits:** 1 target(s) q[0], 2 ancilla(s)
- **Success probability:** 5/8
- **Spec cost:** 4 T
- **On failure:** Z or I, depending on the outcome (Clifford)
- **Source:** [Paetznick & Svore, arXiv:1311.1074](https://arxiv.org/abs/1311.1074), Fig. 1c. The same circuit is `VGateRUS.qs` in [microsoft/Quantum](https://github.com/microsoft/Quantum/tree/main/samples/algorithms/repeat-until-success) and ran on Quantinuum H1-1 in [arXiv:2310.12106](https://arxiv.org/abs/2310.12106), Fig. 2

Same gate as `rus/3` (one ancilla) and `rus/7` (two Toffolis), with a different circuit. Its T count, 4, is the best known.

```console
./milq -v rus -T 1 -A 2 -c Classic -s gurobi benchmarks/rus/8/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
