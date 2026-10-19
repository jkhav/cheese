#!/bin/sh
# Timing on the proofs the milq SAT runs could not reach (all expected: exit 1 = no circuit of <= depth gates).
# CPU build (12-thread laptop) for reference: toffoli 9 7.9 s, ccz 8 7.0 s, toffoli 10 23 s.
cd "$(dirname "$0")/.."
BIN=${BIN:-$( [ -x build/cheese ] && echo build/cheese || echo build/cheese-cpu )}
T=$(mktemp -d)
for a in "toffoli 8" "toffoli 9" "ccz 8" "half_adder_3 9" "fredkin 9" "peres 9" "grover_3 10" "toffoli 10" ${EXTRA}; do
  set -- $a
  (cd python && python3 -m cheese.spec "../tests/qasm/$1.qasm" "$2" > "$T/s" 2>/dev/null)
  s=$(date +%s.%N); "$BIN" "$T/s" > "$T/out" 2>/dev/null; rc=$?; e=$(date +%s.%N)
  echo "$1 depth $2: exit $rc, $(python3 -c "print(round($e - $s, 2))") s, last level: $(tail -1 "$T/out" | cut -c1-120)"
done
rm -rf "$T"
