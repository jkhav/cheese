"""Explicit Clifford+T circuits from T-count decompositions (build/tcount output).

U = X_1 ... X_t C with X = R(P) or R(P)^-1. Each rotation is V T_q V^dag with one T (or Tdg) on qubit q and a
Clifford V mapping Z_q to P (a CNOT ladder spreading Z_q over P's support, then H for X and S H for Y), so the
circuit has exactly t T gates. The Clifford remainder C is synthesised by Qiskit (optional dependency, only for
this step). The whole circuit is then checked with exact arithmetic: its unitary equals U up to a global phase
w^k, as the T-count is defined.

  python -m cudasynth.tcircuit TARGET.qasm DECOMPOSITION.json OUT.qasm [--opt [BEAM]]

--opt: then minimise the Clifford gates with build/cliffopt (src/clifford.cu): every Clifford segment between
two T gates replaced by an optimal one from a GPU table of the whole Clifford group, and the T gates moved
(through Cliffords that commute with them) to make the sum small. The result is checked exactly again.
"""
import json, os, subprocess, sys, tempfile

from .spec import parse_qasm, unitary, mul, wpow
from .tcount import rot, matmul, dagger


def rotation(n, p, inv):
    """gate list (time order) for R(P_p), or R(P_p)^-1 if inv"""
    sup = [(q, (p >> (2 * q)) & 3) for q in range(n) if (p >> (2 * q)) & 3]
    q0 = sup[0][0]
    ladder = [("cx", (q, q0)) for q, _ in sup[1:]]            # CX(c, q0) maps Z_q0 -> Z_c Z_q0
    pre, post = [], []
    for q, d in sup:
        if d == 1:                                            # X = H Z H
            pre.append(("h", (q,)))
            post.append(("h", (q,)))
        elif d == 2:                                          # Y = (S H) Z (S H)^dag
            pre += [("sdg", (q,)), ("h", (q,))]
            post += [("h", (q,)), ("s", (q,))]
    return pre + ladder + [("tdg" if inv else "t", (q0,))] + ladder + post


def clifford_gates(n, C):
    """Clifford unitary C (exact) -> gate list over h, s, sdg, x, y, z, cx (time order), via Qiskit"""
    import numpy as np
    from qiskit import transpile
    from qiskit.quantum_info import Clifford, Operator
    k, M = C
    w = np.exp(1j * np.pi / 4)
    val = lambda z: (z[0] + z[1] * w + z[2] * w ** 2 + z[3] * w ** 3) / np.sqrt(2) ** k
    A = np.array([[val(z) for z in row] for row in M])       # bit q of the index = qubit q, as in Qiskit
    qc = Clifford.from_operator(Operator(A)).to_circuit()
    qc = transpile(qc, basis_gates=["h", "s", "sdg", "x", "y", "z", "cx"], optimization_level=1)
    return [(ins.operation.name, tuple(qc.find_bit(b).index for b in ins.qubits)) for ins in qc.data]


def equal_up_to_phase(A, B):
    (ka, a), (kb, b) = A, B
    return ka == kb and any(all(mul(wpow(ph), x) == y for ra, rb in zip(a, b) for x, y in zip(ra, rb))
                            for ph in range(8))


def build(n, U, rotations):
    """time-ordered gate list: C first, then X_t, ..., X_1"""
    W = U
    for p, inv in rotations:                                   # C = X_t^-1 ... X_1^-1 U
        X = rot(n, p)
        W = matmul(X if inv else dagger(X), W)
    gates = clifford_gates(n, W)
    for p, inv in reversed(rotations):
        gates += rotation(n, p, inv)
    return gates


def optimise(n, gates, beam=16):
    """same T gates, minimal Clifford segments and a beam-searched placement of the T gates (build/cliffopt)"""
    tq, segs, cur = [], [], []
    for g, qs in gates:
        if g in ("t", "tdg"):
            tq.append((qs[0], int(g == "tdg")))
            segs.append(cur)
            cur = []
        else:
            cur.append((g, qs))
    segs.append(cur)
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    with tempfile.NamedTemporaryFile("w", suffix=".job", delete=False) as f:
        f.write(f"{n} {len(tq)}\n" + "".join(f"{q} {d}\n" for q, d in tq))
        for sg in segs:
            f.write(f"{len(sg)} " + " ".join(f"{g} " + " ".join(map(str, qs)) for g, qs in sg) + "\n")
        job = f.name
    try:
        res = subprocess.run([os.path.join(root, "build", "cliffopt"), job, "--beam", str(beam)],
                             capture_output=True, text=True, check=True)
    finally:
        os.unlink(job)
    lines = res.stdout.splitlines()
    info = json.loads(lines[0])
    out = []
    for l in lines[1:]:
        g, *qs = l.split()
        out.append((g, tuple(int(q) for q in qs)))
    return out, info


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    n, circ = parse_qasm(sys.argv[1])
    U = unitary(n, circ)
    res = json.loads(next(l for l in open(sys.argv[2]) if '"rotations"' in l))
    gates = build(n, U, res["rotations"])
    if "--opt" in sys.argv:
        i = sys.argv.index("--opt")
        beam = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit() else 16
        before = len(gates)
        gates, info = optimise(n, gates, beam)
        print(f"cliffopt: Clifford gates {info['clifford_gates_before']} -> {info['clifford_gates']}, "
              f"total {before} -> {len(gates)} ({info['seconds']} s)")
    tc = sum(g in ("t", "tdg") for g, _ in gates)
    ok = equal_up_to_phase(unitary(n, gates), U) and tc == res["tcount"]
    print(f"{'verified' if ok else 'NOT EQUIVALENT'}: {len(gates)} gates, T-count {tc}, equal to the target up to global phase")
    if not ok:
        sys.exit(1)
    with open(sys.argv[3], "w") as f:
        f.write(f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\n// T-count {tc} (optimal, up to global phase), {len(gates)} gates\n")
        f.write(f"qreg q[{n}];\n")
        f.write("".join(f"{g} " + ",".join(f"q[{q}]" for q in qs) + ";\n" for g, qs in gates))


if __name__ == "__main__":
    main()
