// rus/20 without measurement: 9 gates, T 2, config inc+w2+ycms, qiskit check True
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[2];
cx q[2],q[0];
cx q[2],q[1];
t q[2];
h q[2];
cx q[2],q[0];
cx q[2],q[1];
t q[2];
h q[2];
