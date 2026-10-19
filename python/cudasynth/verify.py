"""Check a found circuit independently of the search: exact unitary of the printed gate list == the target's.

  python -m cudasynth.verify TARGET.qasm OUTPUT     (OUTPUT: cudasynth's stdout; uses its "circuit" line)
"""
import json, sys

from .spec import parse_qasm, unitary


def gates_of(names):
    out = []
    for nm in names:                                # L_<gate>_<q>[_<q>...]
        _, g, *qs = nm.split("_")
        out.append((g, tuple(int(q) for q in qs)))
    return out


def main():
    n, circ = parse_qasm(sys.argv[1])
    line = next(l for l in open(sys.argv[2]) if '"circuit"' in l)
    found = gates_of(json.loads(line)["circuit"])
    ok = unitary(n, found) == unitary(n, circ)
    print(f"{'equivalent' if ok else 'NOT EQUIVALENT'}: {len(found)} gates vs {len(circ)} in the input")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
