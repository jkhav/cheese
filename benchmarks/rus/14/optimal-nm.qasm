// rus/14 without measurement (MILQ -v rus -nm): optimal, 12 gates (spec 14).
// Exact output state on every basis state of the target (q0) and |+>, ancillas |0>, no global phase.
// Proved by misc/prototypes/fastsynth.py (smart walk, W2, yices2 -> CryptoMiniSat): 11 gates impossible; Qiskit check passes.
// 2026-10-19, see misc/PROTOTYPE_RESULTS.md §3.12.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[1];
h q[2];
t q[1];
t q[2];
h q[2];
ccx q[0],q[2],q[1];
h q[1];
sdg q[0];
ccx q[1],q[2],q[0];
cx q[1],q[2];
h q[1];
s q[0];
