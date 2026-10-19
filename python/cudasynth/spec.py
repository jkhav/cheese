"""Spec generator: QASM target -> the text spec read by the cudasynth search binary.

Exact arithmetic throughout: an entry of Z[w] (w = e^{i pi/4}) is (a, b, c, d) = a + b w + c w^2 + d w^3, a
matrix is (k, entries) meaning entries / sqrt2^k, normalised (k as small as possible), so equal matrices have
equal representations. Gate table, candidate order and the W2/W3 window rules follow milq
(misc/perf/bvarr.py, misc/prototypes/fastsynth.py sym_rules), so results compare one to one with
milq's prune_probe.py.

  python -m cudasynth.spec TARGET.qasm DEPTH > target.spec
"""
import argparse, itertools, re, sys
from pathlib import Path

# gate -> (sqrt2 exponent, local matrix: M[l][m] = phase exponents p, entry = sum w^p); bit j of a local index
# is qubit qs[j] (milq bvarr.GATES)
D1 = lambda p: (((0,), ()), ((), (p,)))
GATES = {
    "x": (0, (((), (0,)), ((0,), ()))), "y": (0, (((), (6,)), ((2,), ()))),
    "z": (0, D1(4)), "s": (0, D1(2)), "sdg": (0, D1(6)), "t": (0, D1(1)), "tdg": (0, D1(7)),
    "h": (1, (((0,), (0,)), ((0,), (4,)))),
    "cx": (0, (((0,), (), (), ()), ((), (), (), (0,)), ((), (), (0,), ()), ((), (0,), (), ()))),
    "cz": (0, (((0,), (), (), ()), ((), (0,), (), ()), ((), (), (0,), ()), ((), (), (), (4,)))),
    "ccx": (0, tuple(tuple((0,) if m == {3: 7, 7: 3}.get(l, l) else () for m in range(8)) for l in range(8))),
}
DAGGER = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}       # every other gate here is its own inverse
CLIFFORD_T = ["x", "z", "h", "s", "sdg", "t", "tdg", "cx"]
STANDARD = {"x", "y", "z", "s", "sdg", "t", "tdg", "cx", "cz", "h"}   # H the only sqrt2 gate, T/Tdg the only non-Clifford
ZERO = (0, 0, 0, 0)


# ---------- Z[w] ----------
def mul(x, y):
    out = [0] * 4
    for i, a in enumerate(x):
        if a:
            for j, b in enumerate(y):
                if b:
                    s = i + j
                    out[s % 4] += a * b if s < 4 else -a * b
    return tuple(out)


def add(x, y):
    return tuple(a + b for a, b in zip(x, y))


def wpow(p):
    p %= 8
    out = [0] * 4
    out[p % 4] = 1 if p < 4 else -1
    return tuple(out)


def up(x):                                          # x * sqrt2, sqrt2 = w - w^3
    a, b, c, d = x
    return (b - d, a + c, b + d, c - a)


def down(x):                                        # x / sqrt2, or None if not in Z[w]
    a, b, c, d = x
    if (a - c) % 2 or (b - d) % 2:
        return None
    return ((b - d) // 2, (a + c) // 2, (b + d) // 2, (c - a) // 2)


# ---------- exact matrices: (k, rows) with rows a tuple of tuples of Z[w] ----------
def normalise(k, M):
    while k > 0:
        D = [[down(x) for x in row] for row in M]
        if any(x is None for row in D for x in row):
            break
        M, k = D, k - 1
    return k, tuple(tuple(row) for row in M)


def matmul(A, B):
    (ka, a), (kb, b) = A, B
    n = len(a)
    out = [[ZERO] * len(b[0]) for _ in range(n)]
    for i in range(n):
        for l in range(len(b)):
            if a[i][l] != ZERO:
                for j in range(len(b[0])):
                    if b[l][j] != ZERO:
                        out[i][j] = add(out[i][j], mul(a[i][l], b[l][j]))
    return normalise(ka + kb, out)


def identity(N):
    return 0, tuple(tuple((1, 0, 0, 0) if i == j else ZERO for j in range(N)) for i in range(N))


def gate_matrix(g, qs, n):
    """2^n x 2^n exact matrix of gate g on qubits qs (bit q of an index = qubit q)"""
    gk, M = GATES[g]
    N, mask = 1 << n, sum(1 << q for q in qs)
    out = [[ZERO] * N for _ in range(N)]
    for base in range(N):
        if base & mask:
            continue
        idx = [base | sum(((l >> j) & 1) << q for j, q in enumerate(qs)) for l in range(len(M))]
        for l, row in enumerate(M):
            for m, ps in enumerate(row):
                for p in ps:
                    out[idx[l]][idx[m]] = add(out[idx[l]][idx[m]], wpow(p))
    return normalise(gk, out)


def unitary(n, circ):
    U = identity(1 << n)
    for g, qs in circ:
        U = matmul(gate_matrix(g, qs, n), U)
    return U


def parse_qasm(path):
    n, circ = 0, []
    for line in Path(path).read_text().splitlines():
        line = line.split("//")[0].strip()
        if m := re.match(r"(qreg\s+\w+\[(\d+)\]|qubit\[(\d+)\])", line):
            n = int(m.group(2) or m.group(3))
        elif (m := re.match(r"([a-z]+)\s+(.*);", line)) and m.group(1) in GATES:
            circ.append((m.group(1), tuple(int(q) for q in re.findall(r"\[(\d+)\]", m.group(2)))))
    return n, circ


def candidates(n, gates):
    """(name, gate, qubits) in milq's order: gates as given, then qubit tuples"""
    out = []
    for g in gates:
        ar = len(GATES[g][1]).bit_length() - 1
        for qs in ([(q,) for q in range(n)] if ar == 1 else
                   [(a, b) for a in range(n) for b in range(n) if a != b] if ar == 2 else
                   [qs for qs in itertools.permutations(range(n), ar) if qs[0] < qs[1]]):   # ccx: controls unordered
            out.append((f"L_{g}_" + "_".join(map(str, qs)), g, qs))
    return out


# ---------- W2 / W3 window rules (milq fastsynth.sym_rules, exact keys) ----------
def canon(n_loc, qubits, P):
    """key of the product P on the sorted qubit list, after stripping qubits it acts trivially on"""
    k, M = P
    qubits = list(qubits)
    for i in reversed(range(len(qubits))):
        N, bit = len(M), 1 << i
        trivial = all(M[r][c] == ZERO for r in range(N) for c in range(N) if (r ^ c) & bit) and \
            all(M[r][c] == M[r | bit][c | bit] for r in range(N) for c in range(N) if not (r | c) & bit)
        if trivial:
            keep = [r for r in range(N) if not r & bit]
            sub = [[M[r][c] for c in keep] for r in keep]
            # re-index: drop bit i from the indices
            k, M = normalise(k, sub)
            del qubits[i]
    return tuple(qubits), k, M


def product(cs, seq):
    U = sorted({q for g in seq for q in cs[g][2]})
    P = identity(1 << len(U))
    for g in seq:
        P = matmul(gate_matrix(cs[g][1], tuple(U.index(q) for q in cs[g][2]), len(U)), P)
    return canon(len(U), U, P)


def sym_rules(cs, use3=True, max_triples=300_000):
    G = len(cs)
    short = {canon(0, [], identity(1))} | {product(cs, [g]) for g in range(G)}
    first, bad2, keys2 = {}, [], set()
    for pair in itertools.product(range(G), repeat=2):
        k = product(cs, pair)
        keys2.add(k)
        if k in short or first.setdefault(k, pair) != pair:
            bad2.append(pair)
    bad3 = []
    ok2 = set(itertools.product(range(G), repeat=2)) - set(bad2)
    nxt = {g: [h for g2, h in ok2 if g2 == g] for g in range(G)}
    triples = [(f, g, h) for f, g in sorted(ok2) for h in nxt[g]]
    if use3 and len(triples) <= max_triples:
        short |= keys2
        first = {}
        for t in triples:
            k = product(cs, t)
            if k in short or first.setdefault(k, t) != t:
                bad3.append(t)
    return bad2, bad3


# ---------- CX lower bound: cuts and the integer program as a lookup table ----------
def cuts(n):
    out = []
    for r in range(1, n // 2 + 1):
        for A in itertools.combinations(range(n), r):
            if r == n - r and 0 not in A:
                continue                            # each bipartition once
            out.append(A)
    return out


def cx_table(n):
    """table[need tuple, mixed radix 2|A|+1 per cut] = fewest CX/CZ pair counts meeting every cut's demand
    (milq depth_bounds.lb_2q's ILP, solved for every demand vector by enumeration + superset minimum)"""
    cs = cuts(n)
    rad = [2 * len(A) + 1 for A in cs]
    size = 1
    for r in rad:
        size *= r
    if size > 2_000_000:
        return cs, rad, None                         # too big: the search falls back to max over cuts
    pairs = list(itertools.combinations(range(n), 2))
    top = max(rad) - 1
    INF = 1 << 30
    best = [INF] * size
    for x in itertools.product(range(top + 1), repeat=len(pairs)):
        idx, mult = 0, 1
        for A, r in zip(cs, rad):
            cross = sum(c for p, c in zip(pairs, x) if (p[0] in A) != (p[1] in A))
            idx += min(cross, r - 1) * mult
            mult *= r
        best[idx] = min(best[idx], sum(x))
    mult = 1
    for r in rad:                                    # superset minimum, one dimension at a time
        for i in range(size - 1, -1, -1):
            if (i // mult) % r < r - 1:
                best[i] = min(best[i], best[i + mult])
        mult *= r
    return cs, rad, best


# ---------- state targets (RUS without measurement): C |in_j> = Ref |in_j> exactly ----------
def conj(x):
    a, b, c, d = x
    return (a, -d, -c, -b)


def row_sde(k, row):
    """least s >= 0 with sqrt2^s * (row / sqrt2^k) integral"""
    v = min((val(x) for x in row if x != ZERO), default=0)
    return max(0, k - v)


def val(x):
    v = 0
    while (y := down(x)) is not None:
        x, v = y, v + 1
        if v > 400:
            break
    return v


def pauli_apply(n, b, row):
    """P_b |row> for Pauli digits per qubit (0 I, 1 X, 2 Y, 3 Z)"""
    out = [ZERO] * len(row)
    for x, amp in enumerate(row):
        if amp == ZERO:
            continue
        flip = ph = 0
        for q in range(n):
            dgt = (b >> (2 * q)) & 3
            bit = (x >> q) & 1
            if dgt == 1:
                flip |= 1 << q
            elif dgt == 2:
                flip |= 1 << q
                ph += 2 + 4 * bit
            elif dgt == 3:
                ph += 4 * bit
        out[x ^ flip] = mul(amp, wpow(ph))
    return out


def inner(u, v):
    acc = ZERO
    for x, y in zip(u, v):
        if x != ZERO and y != ZERO:
            acc = add(acc, mul(conj(x), y))
    return acc


def nullity(n, row):
    """stabilizer nullity n - log2 |{P : P|v> = +-|v>}| (Clifford-invariant, changes by <= 1 per T)"""
    nrm = inner(row, row)
    neg = tuple(-c for c in nrm)
    cnt = sum(1 for b in range(4 ** n) if inner(row, pauli_apply(n, b, row)) in (nrm, neg))
    return n - (cnt.bit_length() - 1)


def det(M):
    if len(M) == 1:
        return M[0][0]
    acc = ZERO
    for j in range(len(M)):
        if M[0][j] != ZERO:
            sub = [r[:j] + r[j + 1:] for r in M[1:]]
            t = mul(M[0][j], det(sub))
            acc = add(acc, t if j % 2 == 0 else tuple(-c for c in t))
    return acc


def schmidt_rank(n, A, row):
    """exact rank of the state reshaped (qubits A) x (rest): largest nonzero minor"""
    B = [q for q in range(n) if q not in A]
    R, C = 1 << len(A), 1 << len(B)
    M = [[ZERO] * C for _ in range(R)]
    for x, amp in enumerate(row):
        ia = sum(((x >> q) & 1) << i for i, q in enumerate(A))
        ib = sum(((x >> q) & 1) << i for i, q in enumerate(B))
        M[ia][ib] = amp
    for r in range(min(R, C), 0, -1):
        for rows in itertools.combinations(range(R), r):
            for cols in itertools.combinations(range(C), r):
                if det([[M[i][j] for j in cols] for i in rows]) != ZERO:
                    return r
    return 0


def state_pairs(n, U, inputs):
    """start rows e_x and goal rows Ref e_x (the transposed images), shared exponents"""
    k, M = U
    N = 1 << n
    start = [[(1, 0, 0, 0) if c == x else ZERO for c in range(N)] for x in inputs]
    goal = [[M[r][x] for r in range(N)] for x in inputs]
    pad = [[ZERO] * N for _ in range(N - len(inputs))]
    gk, G = normalise(k, goal + pad)
    return (0, tuple(tuple(r) for r in start + pad)), (gk, G)


# ---------- gate sets with ccx: H and (weighted) T bounds stay sound ----------
def ptm_sde(g):
    """sde of the Pauli-transfer matrix of gate g on its own qubits: the most one gate can change the T bound"""
    gk, M = GATES[g]
    nq = len(M).bit_length() - 1
    k, U = gate_matrix(g, tuple(range(nq)), nq)
    N, best = 1 << nq, 0
    cols = [[U[r][c] for r in range(N)] for c in range(N)]
    for b in range(4 ** nq):
        PB = [pauli_apply(nq, b, [ZERO] * c + [(1, 0, 0, 0)] + [ZERO] * (N - c - 1)) for c in range(N)]
        UPB = [[sum_z(mul(U[r][x], PB[c][x]) for x in range(N)) for c in range(N)] for r in range(N)]   # U P_b
        for a in range(4 ** nq):
            acc = ZERO
            for c in range(N):                       # Tr(P_a U P_b U^dag) = sum_c <col_c(U)| P_a |col_c(U P_b)>
                v = pauli_apply(nq, a, [UPB[r][c] for r in range(N)])
                acc = add(acc, inner(cols[c], v))
            if acc != ZERO:
                best = max(best, 2 * (nq + k) - val(acc))
    return best


def sum_z(it):
    acc = ZERO
    for x in it:
        acc = add(acc, x)
    return acc


# ---------- spec ----------
def write_spec(out, n, circ, depth, gates=None, use3=True, inputs=None):
    gates = sorted(set(CLIFFORD_T) | {g for g, _ in circ}) if gates is None else gates
    cs = candidates(n, gates)
    bad2, bad3 = sym_rules(cs, use3)
    k, U = unitary(n, circ)
    cuts_, rad, table = cx_table(n)
    standard = set(gates) <= STANDARD
    lite = not standard and set(gates) <= STANDARD | {"ccx"}     # ccx: a permutation, no sqrt2
    tdiv = max(ptm_sde(g) for g in gates if g in ("t", "tdg", "ccx")) if lite else 1
    w = out.write
    w("cudasynth-spec 1\n")
    w(f"n {n}\nmode {'unitary' if inputs is None else 'states'}\nphase exact\ndepth {depth}\n")
    w(f"standard {int(standard)}\n")                # H-, T- and structure bounds valid for this gate set
    if lite:                                         # only H (sde) and T / tdiv (T and ccx change the PTM sde by <= tdiv)
        w(f"lite 1\ntdiv {tdiv}\n")
    w(f"gates {len(cs)}\n")
    for name, g, qs in cs:
        gk, M = GATES[g]
        dim = len(M)
        ent = []
        for l in range(dim):
            for m in range(dim):
                x = ZERO
                for p in M[l][m]:
                    x = add(x, wpow(p))
                ent.append(x)
        # inverse gate: used to update the residual S <- S g^-1
        ig = DAGGER.get(g, g)
        ik, iM = GATES[ig]
        inv = []
        for l in range(dim):
            for m in range(dim):
                x = ZERO
                for p in iM[l][m]:
                    x = add(x, wpow(p))
                inv.append(x)
        w(f"{name} {len(qs)} {' '.join(map(str, qs))} {gk} {' '.join(' '.join(map(str, x)) for x in ent)}"
          f" {ik} {' '.join(' '.join(map(str, x)) for x in inv)}\n")
    w(f"bad2 {len(bad2)}\n" + "".join(f"{a} {b}\n" for a, b in bad2))
    w(f"bad3 {len(bad3)}\n" + "".join(f"{a} {b} {c}\n" for a, b, c in bad3))
    mat = lambda kk, M: f"{kk}\n" + "".join(" ".join(" ".join(map(str, x)) for x in row) + "\n" for row in M)
    if inputs is None:
        w("target " + mat(k, U))
    else:
        (sk, S), (gk, G) = state_pairs(n, (k, U), inputs)
        w(f"pairs {len(inputs)}\n")
        w("start " + mat(sk, S))
        w("target " + mat(gk, G))
    w(f"cuts {len(cuts_)}\n" + "".join(f"{len(A)} {' '.join(map(str, A))}\n" for A in cuts_))
    if inputs is not None:                           # goal side of the state bounds: sde, nullity, Schmidt ranks
        w(f"goal {len(inputs)}\n")
        for j in range(len(inputs)):
            row = G[j]
            w(f"{row_sde(gk, row)} {nullity(n, row)} " + " ".join(str(schmidt_rank(n, A, row)) for A in cuts_) + "\n")
    if table is None:
        w("cxtable 0\n")
    else:
        w(f"cxtable {len(table)}\n" + " ".join(map(str, table)) + "\n")
    w("end\n")
    return cs, bad2, bad3


def state_inputs(n, o):
    if o.inputs:
        return [int(x) for x in o.inputs.split(",")]
    if o.ancilla:
        mask = sum(1 << q for q in o.ancilla)
        return [x for x in range(1 << n) if not x & mask]
    return None


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("qasm"); a.add_argument("depth", type=int)
    a.add_argument("--no-w3", action="store_true")
    a.add_argument("--ancilla", type=int, nargs="+", help="state target: these qubits start in |0>; inputs = "
                   "every basis state with them 0 (RUS without measurement: C|x,0> = Ref|x,0>)")
    a.add_argument("--inputs", help="state target: comma-separated input basis states (integers, bit q = qubit q)")
    o = a.parse_args()
    n, circ = parse_qasm(o.qasm)
    cs, bad2, bad3 = write_spec(sys.stdout, n, circ, o.depth, use3=not o.no_w3, inputs=state_inputs(n, o))
    print(f"{Path(o.qasm).stem}: {n} qubits, {len(cs)} candidates, W2 pairs {len(bad2)}, W3 triples {len(bad3)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
