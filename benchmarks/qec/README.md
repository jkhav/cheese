# QEC targets

- `repetition-bit`: three-qubit bit-flip encoder, arbitrary logical input on q[0].
- `repetition-phase`: bit-flip encoder followed by H on each qubit.
- `repetition-syndrome`: two parity checks on three data qubits and two ancillas;
  terminal measurements are stripped for this unitary synthesis experiment.
- `code422-zero`: logical |00> = (|0000> + |1111>)/sqrt(2) of [[4,2,2]].
  This is state preparation, not a general two-logical-qubit encoder.
- `steane-zero`: uniform superposition over the span of the three binary rows
  {0,3,4,6}, {1,3,5,6}, {2,4,5,6}. These form a self-orthogonal simplex code,
  giving the Steane logical-zero state in this qubit labeling.
- `shor-zero`: Shor encoder restricted to zero input, yielding three GHZ blocks.

The last three use zero-state synthesis. Tests check the explicit code-state
amplitudes and syndrome parity action. No total-gate optimum is claimed.
