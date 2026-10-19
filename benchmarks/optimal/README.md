# Proven optima outside the RUS set

- `toffoli-gatecount15.qasm`: a 15-gate Clifford+T circuit for Toffoli (exact, global phase included). Optimal
  gate count: every budget <= 14 was proved impossible (`./run.sh tests/qasm/toffoli.qasm --table 8`, 27 s).
  Fredkin's optimum is its 17-gate input `tests/qasm/fredkin.qasm` (every budget <= 16 proved impossible).
- `<name>-tcount.json`: optimal T-count decompositions (up to global phase) from `./tcount.sh tests/qasm/<name>.qasm`.
  `"rotations": [[p, inv], ...]` means U = X_1 ... X_t C with X = R(P_p) (inv 0) or R(P_p)^-1 (inv 1),
  R(P) = ((1 + w) I + (1 - w) P) / 2, P_p the Pauli with digit (p >> 2q) & 3 on qubit q (0 I, 1 X, 2 Y, 3 Z),
  and C a Clifford. Check: `python -m cudasynth.tcount verify tests/qasm/<name>.qasm benchmarks/optimal/<name>-tcount.json`.
  Toffoli, Fredkin, Peres, CCZ: 7; relative-phase Toffoli 4; CS 3; QFT_2 3.
