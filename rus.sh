#!/bin/sh
# Optimal T-count of a two-qubit repeat-until-success circuit, measurement included (q0 ancilla in |0>, measured;
# q1 data): fewest T gates with the same success branch (unitary up to Cliffords, same Pr[success]) and a
# Clifford failure branch. Separate from tcount.sh, which counts T gates of the whole unitary.
#
#   ./rus.sh TARGET.qasm                  search below the input's own T-count; none found = the input is optimal
#   ./rus.sh TARGET.qasm --table K        table depth K (default 8, built once and cached in build/)
#   ./rus.sh TARGET.qasm --max T          give up above T-count T
#   ./rus.sh TARGET.qasm --outcome K      success outcome (default: the "success outcome K" comment, else detected)
#   ./rus.sh TARGET.qasm --opt            minimise the Clifford gates of the found circuit (build/cliffopt)
#
# Output: one JSON line; a found circuit is checked exactly and written to results/<name>-rus.qasm. Build the
# binary with ./build.sh rus. Method: docs/ALGORITHM.md section 12.
set -e
cd "$(dirname "$0")"
[ $# -ge 1 ] || { sed -n '2,13p' "$0"; exit 1; }
TARGET=$(realpath "$1"); shift
[ -x build/rus ] || { echo "no binary: ./build.sh rus"; exit 1; }
TABLE=8; MAX=; OUT=; OPT=
while [ $# -gt 0 ]; do
  case "$1" in
    --table) TABLE=$2; shift ;;
    --max) MAX=$2; shift ;;
    --outcome) OUT="--outcome $2"; shift ;;
    --opt) OPT=--opt ;;
  esac
  shift
done
T_IN=$(grep -cE '^\s*(t|tdg) ' "$TARGET" || true)
DEFAULT_MAX=0; [ -z "$MAX" ] && MAX=$((T_IN - 1)) && DEFAULT_MAX=1
mkdir -p results
NAME=$(basename "$TARGET" .qasm)
(cd python && python3 -m cheese.rus spec "$TARGET" $OUT > "../results/$NAME.rspec")
if [ "$MAX" -lt 0 ]; then
  echo "{\"rus_tcount\": 0, \"optimal\": true, \"witness\": \"input\"}"; exit 0
fi
set +e; build/rus "results/$NAME.rspec" --table "$TABLE" --max "$MAX" > "results/$NAME-rus.log"; rc=$?; set -e
tail -1 "results/$NAME-rus.log"
if [ $rc = 0 ]; then
  (cd python && python3 -m cheese.rus verify "$TARGET" "../results/$NAME-rus.log" "../results/$NAME-rus.qasm" $OUT $OPT)
elif [ $rc = 1 ] && [ $DEFAULT_MAX = 1 ]; then
  echo "{\"rus_tcount\": $T_IN, \"optimal\": true, \"witness\": \"input\"}"; exit 0
fi
exit $rc
