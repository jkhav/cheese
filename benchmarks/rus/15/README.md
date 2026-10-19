# rus/15: PAR, three inputs

- **Gate on success:** Rx(2·atan(tan³(π/8))) ≈ Rx(0.142)
- **Measurement:** success when every ancilla reads 0; q[1]–q[3] in the GHZ basis; q[4] is a work qubit that returns to \|0⟩ and is measured too
- **Qubits:** 1 target(s) q[0], 4 ancilla(s)
- **Success probability:** 5/16
- **Spec cost:** 3 `ccx` + 3 T
- **On failure:** identity or the opposite rotation; the paper fixes the latter with oblivious amplitude amplification (Lemma 2)
- **Source:** [Wiebe & Roetteler, arXiv:1406.2040](https://arxiv.org/abs/1406.2040), Fig. 3 and Thm. 1

A low-T variant exists: uncompute the work AND by an X measurement, for 14 T and p = 5/32. The spec uses `ccx`, so (as with `rus/7`) `ccx` joins the synthesis gate set.

```console
./milq -v rus -T 1 -A 4 -c Classic -s gurobi benchmarks/rus/15/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
