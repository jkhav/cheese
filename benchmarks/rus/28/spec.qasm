OPENQASM 2.0;
include "stdgates.inc";
qreg q[2];
creg c[2];
cx q[0], q[1];
t q[1];
h q[1];
measure q[1] -> c[1];
