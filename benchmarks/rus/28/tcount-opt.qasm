OPENQASM 2.0;
include "qelib1.inc";
// T-count 1 (optimal, up to global phase), 3 gates
qreg q[2];
cx q[0],q[1];
t q[1];
h q[1];
