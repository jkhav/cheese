OPENQASM 2.0;
include "qelib1.inc";
// T-count 1 (optimal, up to global phase), 8 gates
qreg q[2];
z q[0];
z q[1];
cx q[0],q[1];
sdg q[1];
h q[1];
h q[1];
tdg q[1];
h q[1];
