// Benchmark created by MQT Bench on 2025-12-08
// For more info: https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 2.1.0
// Qiskit version: 2.1.1
// Output format: qasm3
// Level: nativegates
// Target: clifford+t
// Used gateset: ['id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg', 'cx', 'cy', 'cz', 'swap', 'iswap', 'dcx', 'ecr', 'reset', 'delay', 'measure']

OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] meas;
h q[0];
cx q[0], q[1];
h q[0];
t q[0];
cx q[0], q[1];
tdg q[1];
cx q[0], q[1];
t q[1];
h q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
