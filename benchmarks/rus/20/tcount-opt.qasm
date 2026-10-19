OPENQASM 2.0;
include "qelib1.inc";
// T-count 2 (optimal, up to global phase), 9 gates
qreg q[3];
h q[2];
t q[2];
cx q[2],q[1];
cx q[2],q[0];
h q[2];
t q[2];
cx q[2],q[1];
cx q[2],q[0];
h q[2];
