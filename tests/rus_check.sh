#!/bin/sh
# Regression for RUS mode (rus.sh): known optima, cross-checked across table depths and with the Python prototype.
# Targets from ricotta's reproduced Paetznick-Svore database (q0 ancilla, q1 data). Needs build/rus.
set -e
cd "$(dirname "$0")/.."
fail=0
check() {   # qasm table expected-rus-tcount
  got=$(./rus.sh "tests/qasm/$1" --table "$2" 2>/dev/null | grep -oE '"(rus_)?tcount": [0-9]+' | tail -1 | grep -oE '[0-9]+$')
  if [ "$got" = "$3" ]; then echo "ok   $1 table $2 -> $got"; else echo "FAIL $1 table $2 -> $got (want $3)"; fail=1; fi
}
check ricotta-rus-0003.qasm 4 4        # input optimal (none with 3)
check ricotta-rus-0003.qasm 1 4        # same, the search side doing the work
check ricotta-rus-0229.qasm 7 10       # raw T 12 -> 10, circuit found and verified exactly
check ricotta-rus-0229.qasm 5 10
exit $fail
