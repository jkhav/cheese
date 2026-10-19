#!/bin/sh
# Synthesise a circuit with the fewest gates for TARGET.qasm (exact, Clifford+T plus the target's own gates).
#
#   ./run.sh TARGET.qasm                  optimum: depth budgets 1, 2, 3, ... until a circuit is found
#   ./run.sh TARGET.qasm --from 7         same, starting at budget 7 (only if depths < 7 are known impossible)
#   ./run.sh TARGET.qasm --depth 9        one budget: a circuit of <= 9 gates, or a proof that none exists
#   ./run.sh TARGET.qasm --ancilla 2      state target (RUS without measurement): qubit 2 starts in |0>, the circuit
#                                         must map every |x,0> to TARGET|x,0> exactly; --inputs 0,3,5 lists inputs
#   options: --out FILE.qasm (default results/<name>-opt.qasm), --cpu (CPU binary), --no-dedupe, --chunk N,
#            --mitm (meet in the middle: forward to level ceil(D/2), backward on U^dag, exact join; unitaries only)
#
# A budget that fails is a proof (no circuit of <= that many gates). One run with budget D returns the shortest
# circuit of <= D gates, but small budgets prune far harder, hence the increasing loop. Each budget's per-level
# JSON goes to results/<name>-d<D>.log; found circuits are re-checked exactly (python -m cudasynth.verify).
set -e
cd "$(dirname "$0")"
[ $# -ge 1 ] || { sed -n '2,16p' "$0"; exit 1; }
TARGET=$(realpath "$1"); shift
NAME=$(basename "$TARGET" .qasm)
FROM=1; ONLY=; OUT=results/$NAME-opt.qasm; BIN=; PASS=; ST=
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM=$2; shift ;;
    --depth) ONLY=$2; shift ;;
    --out) OUT=$2; shift ;;
    --cpu) BIN=build/cudasynth-cpu ;;
    --no-dedupe) PASS="$PASS --no-dedupe" ;;
    --mitm) PASS="$PASS --mitm" ;;
    --chunk) PASS="$PASS --chunk $2"; shift ;;
    --ancilla) ST="--ancilla $2"; shift; while [ $# -gt 1 ] && [ "${2#-}" = "$2" ]; do ST="$ST $2"; shift; done ;;
    --inputs) ST="--inputs $2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done
[ -n "$BIN" ] || BIN=$( [ -x build/cudasynth ] && echo build/cudasynth || echo build/cudasynth-cpu )
[ -x "$BIN" ] || { echo "no binary: run ./install.sh first"; exit 1; }
mkdir -p results
GATES=$(cd python && python3 -c "import sys; from cudasynth.spec import parse_qasm; print(len(parse_qasm(sys.argv[1])[1]))" "$TARGET")
LAST=${ONLY:-$GATES}; D=${ONLY:-$FROM}
echo "$NAME: $GATES gates in the input, $BIN, budgets $D..$LAST"
T0=$(date +%s)
while [ "$D" -le "$LAST" ]; do
  (cd python && python3 -m cudasynth.spec "$TARGET" "$D" $ST > "../results/$NAME.spec" 2>/dev/null)
  s=$(date +%s.%N)
  set +e; "$BIN" "results/$NAME.spec" $PASS > "results/$NAME-d$D.log" 2>/dev/null; rc=$?; set -e
  dt=$(python3 -c "import time; print(round(time.time() - $s, 1))")
  case $rc in
    0) t=$(python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1]).get('t', 0))" "results/$NAME-d$D.log")
       echo "budget $D: circuit with $t gates found ($dt s)"
       (cd python && python3 -m cudasynth.verify "$TARGET" "../results/$NAME-d$D.log" --qasm "../$OUT" $ST)
       [ -z "$ONLY" ] && echo "optimum: $t gates (every smaller budget proven impossible above), total $(( $(date +%s) - T0 )) s"
       exit 0 ;;
    1) echo "budget $D: no circuit of <= $D gates (proof, $dt s)" ;;
    *) echo "budget $D: open or error (exit $rc), see results/$NAME-d$D.log"; exit $rc ;;
  esac
  D=$((D + 1))
done
[ -z "$ONLY" ] && echo "no circuit within the budgets tried"
exit 1
