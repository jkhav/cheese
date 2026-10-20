"""RUS mode: spec writer and exact verifier for build/rus (src/rus.cu).

Optimal T-count of a two-qubit repeat-until-success circuit with measurement taken into account: the fewest T
gates over all Clifford+T circuits W on (ancilla q0, data q1) whose ancilla-|0> half W(|0> x I) has
  - success branch (ancilla outcome k): proportional to the given circuit's, with the same Pr[success], up to
    Cliffords on either side of the success unitary,
  - failure branch: proportional to a Clifford (any).
Only J = W(|0> x I) matters, so the search works on J modulo left Cliffords (W = C X_t ... X_1 with the Clifford
on the output side), in the Pauli transfer matrix Jc[a][b] = Tr(P_a J P_b J^dag) / 2 (16 x 4, rows two-qubit
Paulis, columns one-qubit Paulis): a left Clifford permutes rows with signs, a rotation R(P)^-1 mixes rows as in
T-count mode. Roots: every valid J for this success branch (finitely many up to left Cliffords: a unit-modulus
rescaling u of the success block, a failure amplitude gamma with |gamma|^2 = 1 - p, a failure Clifford D).

  python -m cheese.rus spec TARGET.qasm [--outcome K] > target.rspec
  python -m cheese.rus verify TARGET.qasm OUTPUT OUT.qasm [--outcome K] [--opt [BEAM]]   (OUTPUT: build/rus stdout)

Without --outcome the success outcome is read from a "success outcome K" comment, else detected (the outcome
whose branch is not proportional to a Clifford).
"""
import itertools, json, re, sys
from fractions import Fraction

from .spec import ZERO, add, mul, wpow, normalise, matmul, gate_matrix, identity, parse_qasm, unitary
from .tcount import pauli_matrix, pauli_tables, dagger, real_sqrt2, norm_real

GENS = (1, 3, 4, 12)                       # X_q0, Z_q0, X_q1, Z_q1 (Pauli index: digit per qubit at bits 2q)
CGEN = [("h", (0,)), ("h", (1,)), ("s", (0,)), ("s", (1,)), ("cz", (0, 1))]


# ---------- exact rectangular helpers ----------
def adj(A):
    k, a = A
    return k, tuple(tuple((x[0], -x[3], -x[2], -x[1]) for x in (a[j][i] for j in range(len(a)))) for i in range(len(a[0])))


def trace(A):
    k, a = A
    acc = ZERO
    for i in range(len(a)):
        acc = add(acc, a[i][i])
    return k, acc


def scalar(A):
    """(k, ((z,),)) if A = z / sqrt2^k times the identity (normalised), else None"""
    k, a = A
    if any(a[i][j] != ZERO for i in range(len(a)) for j in range(len(a)) if i != j) or len({r[i] for i, r in enumerate(a)}) != 1:
        return None
    return normalise(k, ((a[0][0],),))


def conjz(z):
    return (z[0], -z[3], -z[2], -z[1])


# ---------- Clifford group mod phase, as row actions C^dag P_a C = s P_pi(a); fixed order (BFS over CGEN) ----------
_CL = None


def cliffords():
    """[(perm, sign, word)] for all 11520 two-qubit Cliffords mod phase; word = CGEN indices in time order"""
    global _CL
    if _CL:
        return _CL
    P16 = [pauli_matrix(2, p) for p in range(16)]

    def action(C):
        Cd = dagger(C)
        pi, sg = [0] * 16, [1] * 16
        for a in range(16):
            _, A = matmul(matmul(Cd, P16[a]), C)
            pi[a], sg[a] = next((b, s) for b in range(16) for s in (1, -1)
                                if all(A[i][j] == tuple(s * v for v in P16[b][1][i][j]) for i in range(4) for j in range(4)))
        return tuple(pi), tuple(sg)

    gact = [action(gate_matrix(g, q, 2)) for g, q in CGEN]
    ident = (tuple(range(16)), (1,) * 16)
    seen, order, frontier = {ident: ()}, [ident], [ident]
    while frontier:
        nxt = []
        for a in frontier:
            for gi, (pg, sgn) in enumerate(gact):          # G C: (G C)^dag P (G C) = C^dag (G^dag P G) C
                b = (tuple(a[0][pg[x]] for x in range(16)), tuple(sgn[x] * a[1][pg[x]] for x in range(16)))
                if b not in seen:
                    seen[b] = seen[a] + (gi,)
                    order.append(b)
                    nxt.append(b)
        frontier = nxt
    assert len(order) == 11520
    _CL = [(p, s, seen[(p, s)]) for p, s in order]
    return _CL


# ---------- PTM of an isometry ----------
def ptm(J):
    """J (4 x 2 exact) -> (e, rows of (x, y)): Jc[a][b] = Tr(P_a J P_b J^dag) / 2 = (x + y sqrt2) / sqrt2^e"""
    P16 = [pauli_matrix(2, p) for p in range(16)]
    P1 = [pauli_matrix(1, p) for p in range(4)]
    Jd = adj(J)
    rows = [[None] * 4 for _ in range(16)]
    for b in range(4):
        A = matmul(matmul(J, P1[b]), Jd)
        for a in range(16):
            k, tr = trace(matmul(P16[a], A))
            rows[a][b] = (k, real_sqrt2(tr))
    E = max(k for r in rows for k, _ in r) + 2
    out = []
    for r in rows:
        row = []
        for k, (x, y) in r:
            for _ in range(E - 2 - k):                       # times sqrt2: (x + y s) s = 2y + x s
                x, y = 2 * y, x
            row.append((x, y))
        out.append(row)
    e, M = norm_real(E, out)
    return e, tuple(tuple(r) for r in M)


_BYGEN = None


def equivalent(A, B):
    """Clifford index c with B = C_c A (rows: B[a] = s_c(a) A[pi_c(a)]), else -1; A, B = (e, rows). A Clifford is
    fixed by the images of the four generator Paulis, so only rows matching B's generator rows are tried."""
    global _BYGEN
    (ea, a), (eb, b) = A, B
    if ea != eb:
        return -1
    if _BYGEN is None:
        _BYGEN = {tuple((p[g], s[g]) for g in GENS): c for c, (p, s, _) in enumerate(cliffords())}
    neg = lambda r: tuple((-x, -y) for x, y in r)
    cand = [[(r, s) for r in range(1, 16) for s in (1, -1) if (a[r] if s == 1 else neg(a[r])) == b[g]] for g in GENS]
    for combo in itertools.product(*cand):
        c = _BYGEN.get(combo)
        if c is not None:
            p, s, _ = cliffords()[c]
            if all((a[p[x]] if s[x] == 1 else neg(a[p[x]])) == b[x] for x in range(16)):
                return c
    return -1


def invariant(A):
    """Clifford-invariant key: exponent, row 0 (identity, fixed by every Clifford), multiset of sign-normalised rows"""
    e, rows = A
    norm = lambda r: max(r, tuple((-x, -y) for x, y in r))
    return e, rows[0], tuple(sorted(norm(r) for r in rows[1:]))


# ---------- roots ----------
def norm2(z):                                              # |z|^2 = A + B sqrt2 for z in Z[w]
    x0, x1, x2, x3 = z
    return x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3, x0 * x1 + x1 * x2 + x2 * x3 - x0 * x3


def norm_solutions(A, B):
    """every z in Z[w] with |z|^2 = A + B sqrt2 (its coefficients' squares then sum to A)"""
    out, r = [], int(A ** 0.5) + 1
    sq = {i * i: i for i in range(r + 1)}
    for x0 in range(-r, r + 1):
        for x1 in range(-r, r + 1):
            for x2 in range(-r, r + 1):
                rem = A - x0 * x0 - x1 * x1 - x2 * x2
                if rem in sq:
                    for x3 in {sq[rem], -sq[rem]}:
                        if x0 * x1 + x1 * x2 + x2 * x3 - x0 * x3 == B:
                            out.append((x0, x1, x2, x3))
    return out


def qmul(x, y):                                            # Q(w) elements as 4 Fractions
    out = [Fraction(0)] * 4
    for i, a in enumerate(x):
        for j, b in enumerate(y):
            out[(i + j) % 4] += a * b if i + j < 4 else -a * b
    return out


def up(z):                                                 # times sqrt2 (sqrt2 = w - w^3)
    return [z[1] - z[3], z[0] + z[2], z[1] + z[3], z[2] - z[0]]


def blocks(W):
    """(k, [B_0, B_1]): B_o[i][j] = <o, i| W |0, j> (ancilla q0 = bit 0 of the index), entries / sqrt2^k"""
    k, w = W
    return k, [[[w[o + 2 * i][2 * j] for j in range(2)] for i in range(2)] for o in range(2)]


def clifford_block(k, B):
    """B / sqrt2^k proportional to a one-qubit Clifford (exact): B B^dag = q I and B P B^dag Q = +-q I"""
    Bm = (k, tuple(tuple(r) for r in B))
    G = scalar(matmul(Bm, adj(Bm)))
    if G is None or G[1][0][0] == ZERO:
        return False
    negG = (G[0], ((tuple(-v for v in G[1][0][0]),),))
    P1 = [pauli_matrix(1, p) for p in range(4)]
    for p in (1, 3):
        A = matmul(matmul(Bm, P1[p]), adj(Bm))
        if not any(scalar(matmul(A, P1[q])) in (G, negG) for q in (1, 2, 3)):
            return False
    return True


def success_outcome(path, W):
    m = re.search(r"success outcome (\d)", open(path).read())
    if m:
        return int(m.group(1))
    k, B = blocks(W)
    cl = [clifford_block(k, B[o]) for o in range(2)]
    assert cl.count(True) == 1, "not a RUS circuit: need exactly one branch proportional to a Clifford"
    return cl.index(False)


def roots(W, ks):
    """every valid J up to left Cliffords and phase, as distinct PTMs (success outcome moved to 0)"""
    kW, B = blocks(W)
    M = B[ks]
    A = sum(norm2(z)[0] for r in M for z in r)
    Bs = sum(norm2(z)[1] for r in M for z in r)
    m = kW + 1                                             # p = (A + Bs sqrt2) / 2^m
    gammas = norm_solutions(2 ** m - A, -Bs)               # gamma = g / sqrt2^m, |gamma|^2 = 1 - p
    m0 = next(z for r in M for z in r if z != ZERO)
    a0, b0 = norm2(m0)
    d = Fraction(a0 * a0 - 2 * b0 * b0)
    inv = [a0 / d, -b0 / d, Fraction(0), b0 / d]           # 1 / |m0|^2
    us = []
    for h in norm_solutions(a0, b0):                       # u = h / m0 (|u| = 1), kept if u M stays exact
        u = qmul(qmul([Fraction(v) for v in h], [Fraction(v) for v in conjz(m0)]), inv)
        if all(c.denominator & (c.denominator - 1) == 0 for r in M for z in r for c in qmul(u, [Fraction(v) for v in z])):
            us.append(u)
    D1, seen, fr = [], set(), [identity(2)]                # the 24 one-qubit Cliffords mod phase
    while fr:
        nx = []
        for U in fr:
            key = min(tuple(mul(wpow(ph), x) for r in U[1] for x in r) for ph in range(8)) + (U[0],)
            if key not in seen:
                seen.add(key)
                D1.append(U)
                nx += [matmul(gate_matrix(g, (0,), 1), U) for g in ("h", "s")]
        fr = nx
    assert len(D1) == 24
    found, out = {}, []
    for u in us:
        for g in gammas:
            for kD, dm in D1:
                vals = []                                  # row 2i + anc: anc 0 = u M, anc 1 = gamma D
                for i in range(2):
                    for anc in range(2):
                        for j in range(2):
                            v, k = ((qmul(u, [Fraction(x) for x in M[i][j]]), kW) if anc == 0 else
                                    ([Fraction(x) for x in mul(g, dm[i][j])], m + kD))
                            for _ in range(k):
                                v = [c / 2 for c in up(v)]
                            vals.append(v)
                E = 0
                while not all(c.denominator == 1 for v in vals for c in v):
                    vals = [up(v) for v in vals]
                    E += 1
                J = normalise(E, [[tuple(int(c) for c in vals[4 * i + 2 * anc + j]) for j in range(2)]
                                  for i in range(2) for anc in range(2)])
                X = ptm(J)
                bucket = found.setdefault(invariant(X), [])
                if all(equivalent(Y, X) < 0 for Y in bucket):
                    bucket.append(X)
                    out.append(X)
    return out, dict(p=[A, Bs, m], rescalings=len(us), failure_amplitudes=len(gammas))


J00 = (0, tuple(tuple((1, 0, 0, 0) if i == 2 * j else ZERO for j in range(2)) for i in range(4)))


def write_spec(out, R):
    w = out.write

    def mat(X):
        e, rows = X
        w(f"{e}\n" + "\n".join(" ".join(f"{x} {y}" for x, y in r) for r in rows) + "\n")
    w("rus-spec 1\ntable-root\n")
    mat(ptm(J00))
    w(f"roots {len(R)}\n")
    for X in R:
        mat(X)
    tab = pauli_tables(2)
    w(f"rotations {len(tab)}\n")
    for row in tab:
        w(" ".join(f"{r} {s}" for r, s in row) + "\n")
    cl = cliffords()
    w(f"cliffords {len(cl)}\n")
    for p, s, _ in cl:
        w(" ".join(f"{a} {b}" for a, b in zip(p, s)) + "\n")
    w("end\n")


# ---------- circuit and exact check ----------
def circuit(res):
    """time-ordered gates of W' = R(P_1)..R(P_j) C R(Q_c)^-1..R(Q_1)^-1 (search peels fw, table peels tb)"""
    from .tcircuit import rotation
    gates = []
    for q in res["tb"]:
        gates += rotation(2, q, True)
    gates += [CGEN[gi] for gi in cliffords()[res["cliff"]][2]]
    for p in reversed(res["fw"]):
        gates += rotation(2, p, False)
    return gates


def check_rus(W, ks, V):
    """exact: V's outcome-0 branch = (unit) x W's outcome-ks branch and V's outcome-1 branch is proportional to a
    Clifford (so V is a RUS circuit with the same success unitary and probability)"""
    kW, BW = blocks(W)
    kV, BV = blocks(V)
    M = (kW, tuple(tuple(r) for r in BW[ks]))
    S = (kV, tuple(tuple(r) for r in BV[0]))
    pM = scalar(matmul(M, adj(M)))                         # M M^dag = p I
    mu = scalar(matmul(S, adj(M)))                         # S M^dag = mu I, so S = (mu / p) M
    if pM is None or mu is None:
        return False
    sq = lambda s, z: normalise(2 * s[0], ((mul(s[1][0][0], z),),))
    return sq(mu, conjz(mu[1][0][0])) == sq(pM, pM[1][0][0]) and clifford_block(kV, BV[1])   # |mu| = p


def main():
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "spec":
        n, circ = parse_qasm(args[1])
        assert n == 2, "RUS mode: two qubits (q0 ancilla, measured; q1 data)"
        W = unitary(n, circ)
        ks = int(args[args.index("--outcome") + 1]) if "--outcome" in args else success_outcome(args[1], W)
        R, info = roots(W, ks)
        print(json.dumps(dict(outcome=ks, roots=len(R), **info)), file=sys.stderr)
        write_spec(sys.stdout, R)
    elif len(args) >= 4 and args[0] == "verify":
        n, circ = parse_qasm(args[1])
        W = unitary(n, circ)
        ks = int(args[args.index("--outcome") + 1]) if "--outcome" in args else success_outcome(args[1], W)
        res = json.loads(next(l for l in open(args[2]) if '"tcount"' in l))
        gates = circuit(res)
        if "--opt" in args:
            from .tcircuit import optimise
            i = args.index("--opt")
            beam = int(args[i + 1]) if i + 1 < len(args) and args[i + 1].isdigit() else 16
            before = len(gates)
            gates, info = optimise(2, gates, beam)
            print(f"cliffopt: Clifford gates {info['clifford_gates_before']} -> {info['clifford_gates']}, "
                  f"total {before} -> {len(gates)} ({info['seconds']} s)")
        tc = sum(g in ("t", "tdg") for g, _ in gates)
        ok = check_rus(W, ks, unitary(2, gates)) and tc == res["tcount"]
        print(f"{'verified' if ok else 'NOT VERIFIED'}: {len(gates)} gates, T-count {tc}; success branch (outcome 0) "
              f"= the target's (outcome {ks}) up to phase, failure branch a Clifford: {ok}")
        if not ok:
            sys.exit(1)
        with open(args[3], "w") as f:
            f.write(f"OPENQASM 2.0;\ninclude \"qelib1.inc\";\n// RUS T-count {tc} (optimal for this success branch and "
                    f"Pr[success]), {len(gates)} gates; success outcome 0, failure branch a Clifford\n")
            f.write("qreg q[2];\ncreg c[1];\n")
            f.write("".join(f"{g} " + ",".join(f"q[{q}]" for q in qs) + ";\n" for g, qs in gates))
            f.write("measure q[0] -> c[0];\n")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
