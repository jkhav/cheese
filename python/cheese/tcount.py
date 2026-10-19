"""T-count mode: spec writer and exact verifier (separate from the gate-count search).

A unitary U is written as U = R(P_t) ... R(P_1) C with R(P) = ((1 + w) I + (1 - w) P) / 2 (a pi/8 rotation
about the Pauli P, one T gate), C a Clifford and t the T-count; global phase is ignored. The search works on
channel representations Uc[a][b] = Tr(P_a U P_b U^dag) / 2^n: real 4^n x 4^n matrices with entries in
Z[1/sqrt2], here (x + y sqrt2) / sqrt2^e with one exponent e per matrix.

  python -m cheese.tcount spec TARGET.qasm > target.tspec
  python -m cheese.tcount verify TARGET.qasm OUTPUT     (OUTPUT: build/tcount's stdout)

The output lists rotations [p, inv]: U = X_1 ... X_t C with X = R(P_p), or R(P_p)^-1 when inv is 1.
"""
import json, sys

from .spec import ZERO, add, mul, wpow, normalise, parse_qasm, unitary, conj

ONE = (1, 0, 0, 0)


def pauli_matrix(n, p):
    """P_p, digits per qubit (0 I, 1 X, 2 Y, 3 Z), as (0, rows) over Z[w]"""
    N = 1 << n
    M = [[ZERO] * N for _ in range(N)]
    for x in range(N):
        flip = ph = 0
        for q in range(n):
            d, bit = (p >> (2 * q)) & 3, (x >> q) & 1
            if d == 1:
                flip |= 1 << q
            elif d == 2:
                flip |= 1 << q
                ph += 2 + 4 * bit
            elif d == 3:
                ph += 4 * bit
        M[x ^ flip][x] = wpow(ph)
    return 0, tuple(tuple(r) for r in M)


def matmul(A, B):
    (ka, a), (kb, b) = A, B
    N = len(a)
    out = [[ZERO] * N for _ in range(N)]
    for i in range(N):
        for l in range(N):
            if a[i][l] != ZERO:
                for j in range(N):
                    if b[l][j] != ZERO:
                        out[i][j] = add(out[i][j], mul(a[i][l], b[l][j]))
    return normalise(ka + kb, out)


def dagger(A):
    k, a = A
    return k, tuple(tuple(conj(a[j][i]) for j in range(len(a))) for i in range(len(a)))


def trace(A):
    k, a = A
    acc = ZERO
    for i in range(len(a)):
        acc = add(acc, a[i][i])
    return k, acc


def real_sqrt2(z):
    """a + b w + c w^2 + d w^3 real -> (x, y) with value x + y sqrt2 (sqrt2 = w - w^3)"""
    a, b, c, d = z
    assert c == 0 and d == -b, f"not real: {z}"
    return a, b


def norm_real(e, M):
    while e > 0 and all(x % 2 == 0 for r in M for x, _ in r):  # (x + y s) / s = y + (x / 2) s
        M = [[(y, x // 2) for x, y in r] for r in M]
        e -= 1
    return e, M


def channel(n, U):
    """(e, rows of (x, y)): Uc[a][b] = (x + y sqrt2) / sqrt2^e, normalised"""
    P = 4 ** n
    paulis = [pauli_matrix(n, p) for p in range(P)]
    Ud = dagger(U)
    rows = [[None] * P for _ in range(P)]
    for b in range(P):
        A = matmul(matmul(U, paulis[b]), Ud)                   # U P_b U^dag
        for a in range(P):
            k, tr = trace(matmul(paulis[a], A))
            rows[a][b] = (k, real_sqrt2(tr))
    # value = tr / sqrt2^k / 2^n = (x + y sqrt2) / sqrt2^(k + 2n); bring all to the largest exponent
    E = max(k for r in rows for k, _ in r) + 2 * n
    out = []
    for r in rows:
        row = []
        for k, (x, y) in r:
            for _ in range(E - 2 * n - k):                     # times sqrt2: (x + y s) s = 2y + x s
                x, y = 2 * y, x
            row.append((x, y))
        out.append(row)
    return norm_real(E, out)


def pauli_tables(n):
    """for each P (p >= 1) and each Pauli c: (-1, 0) if they commute, else (r, sign) with i P P_c = sign P_r"""
    P = 4 ** n
    paulis = [pauli_matrix(n, p) for p in range(P)]
    iw = (0, 0, 1, 0)                                          # i = w^2
    tab = []
    for p in range(1, P):
        row = []
        for c in range(P):
            _, M = matmul(paulis[p], paulis[c])
            _, M2 = matmul(paulis[c], paulis[p])
            if M == M2:
                row.append((-1, 0))
                continue
            M = tuple(tuple(mul(iw, x) for x in r) for r in M)
            hit = None
            for r in range(P):
                Q = paulis[r][1]
                for sign in (1, -1):
                    if all(M[i][j] == tuple(sign * v for v in Q[i][j]) for i in range(len(M)) for j in range(len(M))):
                        hit = (r, sign)
                if hit:
                    break
            assert hit, "i P Q not a signed Pauli"
            row.append(hit)
        tab.append(row)
    return tab


def rot(n, p):
    """R(P) = ((1 + w) I + (1 - w) P) / 2 as an exact matrix"""
    N = 1 << n
    _, Pm = pauli_matrix(n, p)
    a, b = (1, 1, 0, 0), (1, -1, 0, 0)
    return normalise(2, [[add(a if i == j else ZERO, mul(b, Pm[i][j])) for j in range(N)] for i in range(N)])


def phase_match(A, Q):
    """A = w^ph Q for some ph"""
    return any(all(A[i][j] == mul(wpow(ph), Q[i][j]) for i in range(len(A)) for j in range(len(A))) for ph in range(8))


def is_clifford(n, C):
    """C P C^dag is a Pauli up to phase for every non-identity Pauli P"""
    P = 4 ** n
    paulis = [pauli_matrix(n, p) for p in range(P)]
    Cd = dagger(C)
    for p in range(1, P):
        k, A = matmul(matmul(C, paulis[p]), Cd)
        if k != 0 or not any(phase_match(A, paulis[r][1]) for r in range(1, P)):
            return False
    return True


def write_spec(out, n, U):
    e, M = channel(n, U)
    tab = pauli_tables(n)
    w = out.write
    w(f"tcount-spec 1\nn {n}\nchannel {e}\n")
    for r in M:
        w(" ".join(f"{x} {y}" for x, y in r) + "\n")
    w(f"rotations {len(tab)}\n")
    for row in tab:
        w(" ".join(f"{r} {s}" for r, s in row) + "\n")
    w("end\n")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "spec":
        n, circ = parse_qasm(sys.argv[2])
        write_spec(sys.stdout, n, unitary(n, circ))
    elif len(sys.argv) >= 4 and sys.argv[1] == "verify":
        n, circ = parse_qasm(sys.argv[2])
        W = unitary(n, circ)
        res = json.loads(next(l for l in open(sys.argv[3]) if '"rotations"' in l))
        for p, inv in res["rotations"]:                        # U = X_1 X_2 ... X_t C, X = R(P) or R(P)^-1
            X = rot(n, p)
            W = matmul(X if inv else dagger(X), W)            # peel X_1, X_2, ... off the left
        ok = is_clifford(n, W) and len(res["rotations"]) == res["tcount"]
        print(f"{'verified' if ok else 'NOT VERIFIED'}: T-count {res['tcount']}, remainder is a Clifford: {ok}")
        sys.exit(0 if ok else 1)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
