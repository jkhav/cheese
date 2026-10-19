// rus/7 without measurement: 7 gates, T 0, config inc+w2+ycms, qiskit check True
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[1];
ccx q[0],q[1],q[2];
h q[0];
s q[0];
h q[0];
ccx q[0],q[1],q[2];
h q[1];
