// rus/7 without measurement: 7 gates, T 0, config inc, qiskit check True
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[2];
ccx q[0],q[2],q[1];
h q[0];
s q[0];
h q[0];
ccx q[0],q[2],q[1];
h q[2];
