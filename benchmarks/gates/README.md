# Gate targets

The files contain only Clifford+T primitives, so target operations such as `ccx`
do not accidentally become unit-cost synthesis gates.

- CS: `T(a) T(b) CX(a,b) Tdg(b) CX(a,b)`.
- Toffoli: Qiskit 2.2.3's CCX decomposition, transpiled at level 0.
- CCZ: H–CCX–H with adjacent H gates cancelled (level 1).
- Relative-phase Toffoli: Qiskit 2.2.3 RCCX decomposition, level 0.

`tests/test_bench.py` compares each complete matrix against the independent Qiskit
gate. These are targets with known constructions, not frozen total-gate optima.
The proposed literature T/CX bounds use different objectives from this baseline.
