// rus/12 without measurement: 8 gates, T 2, config inc+w2+ycms, qiskit check True
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[1];
t q[1];
h q[1];
cx q[1],q[0];
sdg q[1];
h q[1];
tdg q[1];
h q[1];
