// rus/8 without measurement: 11 gates, T 4, config smart+w2+cms, qiskit check True
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[2];
h q[1];
t q[2];
cx q[1],q[2];
tdg q[2];
h q[2];
t q[0];
z q[0];
cx q[0],q[1];
t q[1];
h q[1];
