// rus/8 without measurement (MILQ -v rus -nm): optimal, 11 gates (spec 11, so the spec is optimal).
// Exact output state on every basis state of the target (q0) and |+>, ancillas |0>, no global phase.
// Proved by milq's SAT search (smart walk, W2, yices2 -> CryptoMiniSat): 10 gates impossible; Qiskit check passes.
// 2026-10-19.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[2];
h q[1];
t q[2];
cx q[1],q[2];
tdg q[2];
h q[2];
t q[0];
z q[0];
cx q[0],q[1];
t q[1];
h q[1];
