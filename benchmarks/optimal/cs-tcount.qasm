OPENQASM 2.0;
include "qelib1.inc";
// T-count 3 (optimal, up to global phase), 7 gates
qreg q[2];
s q[0];
s q[1];
tdg q[0];
tdg q[1];
cx q[1],q[0];
tdg q[0];
cx q[1],q[0];
