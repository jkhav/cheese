OPENQASM 2.0;
include "qelib1.inc";
// T-count 3 (optimal, up to global phase), 5 gates
qreg q[2];
t q[0];
t q[1];
cx q[1],q[0];
tdg q[0];
cx q[1],q[0];
