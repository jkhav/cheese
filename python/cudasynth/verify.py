"""Check a found circuit independently of the search: exact unitary of the printed gate list == the target's.

  python -m cudasynth.verify TARGET.qasm OUTPUT [--qasm OUT.qasm] [--ancilla Q ... | --inputs X,Y,...]
OUTPUT: cudasynth's stdout (its "circuit" line). --qasm: also write the circuit as OpenQASM 2.0.
--ancilla / --inputs (as for the spec): state target, only the images of those input basis states must agree.
"""
import json, sys

from .spec import parse_qasm, unitary, normalise


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
    inputs = None
    if "--inputs" in sys.argv:
        inputs = [int(x) for x in sys.argv[sys.argv.index("--inputs") + 1].split(",")]
    elif "--ancilla" in sys.argv:
        i, anc = sys.argv.index("--ancilla") + 1, []
        while i < len(sys.argv) and sys.argv[i].isdigit():
            anc.append(int(sys.argv[i])); i += 1
        mask = sum(1 << q for q in anc)
        inputs = [x for x in range(1 << n) if not x & mask]
    if inputs is None:
        ok = unitary(n, found) == unitary(n, circ)
    else:
        cols = lambda U: normalise(U[0], [[U[1][r][x] for r in range(1 << n)] for x in inputs])
        ok = cols(unitary(n, found)) == cols(unitary(n, circ))
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
