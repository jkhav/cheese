#!/bin/sh
# Regression: known verdicts (milq SAT runs and prune_probe.py agree on all of these). Needs a build; uses
# build/cheese if present, else build/cheese-cpu. Targets in tests/qasm (copied from milq's benchmarks).
set -e
cd "$(dirname "$0")/.."
BIN=${BIN:-$( [ -x build/cheese ] && echo build/cheese || echo build/cheese-cpu )}
B=$(pwd)/tests/qasm
T=$(mktemp -d)
fail=0
check() {   # qasm depth expected-exit
  (cd python && python3 -m cheese.spec "$B/$1" "$2" > "$T/s" 2>/dev/null)
  set +e; "$BIN" "$T/s" > "$T/out" 2>/dev/null; got=$?; set -e
  if [ "$got" = "$3" ]; then echo "ok   $1 depth $2 -> $got"; else echo "FAIL $1 depth $2 -> $got (want $3)"; fail=1; fi
  if [ "$got" = 0 ]; then (cd python && python3 -m cheese.verify "$B/$1" "$T/out") || fail=1; fi
}
check cs.qasm 4 1                  # proof
check cs.qasm 5 0                  # circuit (optimum 5)
check toffoli.qasm 8 1
check ccz.qasm 7 1
check qft_2.qasm 7 0     # optimum 7
check qft_2.qasm 6 1
check relative-phase-toffoli.qasm 9 0                     # optimum 9 (SAT, milq)
rm -rf "$T"
exit $fail
