OPENQASM 2.0;
include "qelib1.inc";
// T-count 4 (optimal, up to global phase), 11 gates
qreg q[2];
h q[1];
tdg q[1];
cx q[0],q[1];
h q[0];
t q[0];
h q[0];
tdg q[0];
h q[0];
cx q[0],q[1];
tdg q[1];
h q[1];
