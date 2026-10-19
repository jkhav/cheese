OPENQASM 2.0;
include "qelib1.inc";
// T-count 3 (optimal, up to global phase), 14 gates
qreg q[2];
z q[0];
sdg q[0];
h q[0];
h q[1];
s q[1];
h q[0];
tdg q[0];
h q[0];
tdg q[1];
h q[0];
cx q[1],q[0];
tdg q[0];
cx q[1],q[0];
h q[0];
