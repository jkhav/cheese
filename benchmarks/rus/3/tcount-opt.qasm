OPENQASM 2.0;
include "qelib1.inc";
// T-count 4 (optimal, up to global phase), 12 gates
qreg q[2];
h q[1];
t q[1];
h q[1];
tdg q[1];
h q[1];
cx q[0],q[1];
t q[1];
h q[1];
cx q[0],q[1];
t q[1];
h q[1];
z q[0];
