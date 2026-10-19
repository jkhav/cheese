"""Check a found circuit independently of the search: exact unitary of the printed gate list == the target's.

  python -m cudasynth.verify TARGET.qasm OUTPUT [--qasm OUT.qasm]
OUTPUT: cudasynth's stdout (its "circuit" line). --qasm: also write the circuit as OpenQASM 2.0.
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
    if ok and "--qasm" in sys.argv:
        out = sys.argv[sys.argv.index("--qasm") + 1]
        with open(out, "w") as f:
            f.write(f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{n}];\n")
            f.write("".join(f"{g} " + ",".join(f"q[{q}]" for q in qs) + ";\n" for g, qs in found))
        print(f"written {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
