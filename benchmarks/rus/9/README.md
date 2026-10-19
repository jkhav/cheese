# rus/9: Magic-state ladder, rung 2

- **Gate on success:** Rz(θ) = diag(1, e^{iθ}) up to phase, tan(θ/2) = (√2−1)³, θ ≈ 0.1419
- **Measurement:** success when every ancilla reads 0; all of q[1]–q[3] in the Z basis; q[1] carries the rung state, q[2] and q[3] are fresh resource states, and the `cx` gates are parity checks
- **Qubits:** 1 target(s) q[0], 3 ancilla(s)
- **Success probability:** 5/16
- **Spec cost:** 3 T
- **On failure:** not Pauli: a failed injection applies Rz(−θ), a failed parity check a T-type rotation
- **Source:** [Duclos-Cianci & Svore, arXiv:1210.1980](https://arxiv.org/abs/1210.1980), Figs. 1–2; also used in [Bocharov, Gurevich & Svore, arXiv:1303.1411](https://arxiv.org/abs/1303.1411), App. A

Rung 1 is Clifford-equivalent to `rus/2`. This gate equals V₃·T†, but it is not Clifford-equivalent to V₃. MILQ checks only the success branch, so the non-Pauli failures are fine there. It is not an RUS circuit in the strict Paetznick–Svore sense (Clifford recovery).

```console
./milq -v rus -T 1 -A 3 -c Classic -s gurobi benchmarks/rus/9/spec.qasm
```

`tests/test_bench.py::test_rus_literature_success_map` checks the success map and probability.
