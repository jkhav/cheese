#!/bin/sh
# Optimal T-count of TARGET.qasm's unitary over Clifford+T, up to global phase (separate from run.sh's gate count).
#
#   ./tcount.sh TARGET.qasm               table of all cosets with T-count <= 4 (3 qubits) / 6 (1-2 qubits)
#   ./tcount.sh TARGET.qasm --table K     table depth K (built once per qubit count and K, cached in build/)
#   ./tcount.sh TARGET.qasm --max T       give up above T-count T (default 16)
#
# Output: one JSON line, U = X_1 ... X_t C with X = R(P_p) or R(P_p)^-1 ([p, inv]) and C a Clifford, checked
# exactly by `python -m cudasynth.tcount verify`. Build the binary with ./build.sh tcount. Method: docs/ALGORITHM.md.
set -e
cd "$(dirname "$0")"
[ $# -ge 1 ] || { sed -n '2,10p' "$0"; exit 1; }
TARGET=$(realpath "$1"); shift
[ -x build/tcount ] || { echo "no binary: ./build.sh tcount"; exit 1; }
N=$(cd python && python3 -c "import sys; from cudasynth.spec import parse_qasm; print(parse_qasm(sys.argv[1])[0])" "$TARGET")
K=4; [ "$N" -le 2 ] && K=6
mkdir -p results
NAME=$(basename "$TARGET" .qasm)
(cd python && python3 -m cudasynth.tcount spec "$TARGET" > "../results/$NAME.tspec")
set +e; build/tcount "results/$NAME.tspec" --table "$K" "$@" > "results/$NAME-tcount.log"; rc=$?; set -e
tail -1 "results/$NAME-tcount.log"
[ $rc = 0 ] && (cd python && python3 -m cudasynth.tcount verify "$TARGET" "../results/$NAME-tcount.log")
exit $rc
