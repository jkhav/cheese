// rus/4 without measurement (MILQ -v rus -nm): optimal, 18 gates (spec 20), T-count 6.
// Exact output state on |0>|0>, |1>|0>, |+>|0> (target q0, ancilla q1), no global phase.
// Proved optimal by misc/prototypes/fastsynth.py (smart walk, W2, yices2 -> CryptoMiniSat):
// 17 gates impossible, 18 found; Qiskit statevector check passes. 2026-10-19, see misc/PROTOTYPE_RESULTS.md §3.12.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[1];
tdg q[1];
h q[1];
t q[1];
h q[1];
s q[1];
cx q[0],q[1];
t q[1];
h q[1];
t q[1];
cx q[0],q[1];
s q[1];
h q[1];
s q[1];
t q[1];
h q[1];
tdg q[1];
h q[1];
