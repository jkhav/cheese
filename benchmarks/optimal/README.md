# Proven optima outside the RUS set

- `toffoli-gatecount15.qasm`: a 15-gate Clifford+T circuit for Toffoli (exact, global phase included). Optimal
  gate count: every budget <= 14 was proved impossible (`./run.sh tests/qasm/toffoli.qasm --table 8`, 27 s).
  Fredkin's optimum is its 17-gate input `tests/qasm/fredkin.qasm` (every budget <= 16 proved impossible).
- `<name>-tcount.json`: optimal T-count decompositions (up to global phase) from `./tcount.sh tests/qasm/<name>.qasm`.
  `"rotations": [[p, inv], ...]` means U = X_1 ... X_t C with X = R(P_p) (inv 0) or R(P_p)^-1 (inv 1),
  R(P) = ((1 + w) I + (1 - w) P) / 2, P_p the Pauli with digit (p >> 2q) & 3 on qubit q (0 I, 1 X, 2 Y, 3 Z),
  and C a Clifford. Check: `python -m cheese.tcount verify tests/qasm/<name>.qasm benchmarks/optimal/<name>-tcount.json`.
  Toffoli, Fredkin, Peres, CCZ: 7; relative-phase Toffoli 4; CS 3; QFT_2 3.
- `<name>-tcount.qasm`: the same decompositions as explicit Clifford+T circuits with exactly that many T/Tdg
  gates, equal to the target up to global phase (checked exactly). Built by
  `python -m cheese.tcircuit TARGET.qasm DECOMPOSITION.json OUT.qasm`: each rotation is V T V^dag with a
  Clifford V (CNOT ladder + H / S H), the Clifford remainder is synthesised with Qiskit. Then the Clifford gates are
  reduced by `--opt` (build/cliffopt: optimal Clifford segments from a GPU table of the whole Clifford group and
  a beam search over where the T gates sit). The T-count is optimal; the total is small but not proved minimal.
  Total gates before -> after `--opt`: Toffoli 28 -> 20, Fredkin 67 -> 21, Peres 28 -> 15, CCZ 24 -> 16,
  relative-phase Toffoli 29 -> 12, CS 7 -> 5 (= the gate-count optimum), QFT_2 14 -> 7 (= the gate-count optimum).
