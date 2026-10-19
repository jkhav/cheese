// rus/2 without measurement: 7 gates, T 2, config smart+kissat, qiskit check True
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[1];
t q[1];
cx q[1],q[0];
h q[1];
cx q[1],q[0];
t q[1];
h q[1];
