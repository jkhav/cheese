// rus/28 without measurement: 3 gates, T 1, config inc+w2+ycms, qiskit check True
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
cx q[0],q[1];
h q[1];
t q[0];
