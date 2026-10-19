// rus/1 without measurement: 11 gates, T 4, config inc, qiskit check True
OPENQASM 2.0;
include "qelib1.inc";
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
