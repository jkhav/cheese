// rus/14 without measurement: 12 gates, T 2, config smart+w2+cms, qiskit check True
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[1];
h q[2];
t q[1];
t q[2];
h q[2];
ccx q[0],q[2],q[1];
h q[1];
sdg q[0];
ccx q[1],q[2],q[0];
cx q[1],q[2];
h q[1];
s q[0];
