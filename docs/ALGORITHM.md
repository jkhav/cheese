# How cheese works

cheese finds a circuit with the fewest gates that implements a given unitary **exactly** (global phase
included), over Clifford+T plus the target's own gates, or proves that no circuit of a given size exists.
This document describes the search, every rule that removes part of the search space, and why none of them can
remove the optimum. Code references are to `src/cheese.cu` (search), `python/cheese/spec.py` (spec
generator) and `run.sh` (outer loop).

## 1. Problem and guarantees

Input: a target unitary U on n qubits (from a QASM circuit; measurements are dropped) and a budget D.
Candidates: every gate of the gate set on every qubit placement (`L_h_0`, `L_cx_1_0`, `L_ccx_0_1_2`, ...); the
default set is `x z h s sdg t tdg cx` plus the gates the target uses. A circuit is a sequence of candidates
g_1, ..., g_L applied in that order, with product g_L ... g_1.

Output of one run with budget D (exit codes):
- `0`: a circuit with product exactly U and at most D gates; it is the shortest circuit of at most D gates.
- `1`: a proof that no circuit of at most D gates exists.

`run.sh` calls the binary with budgets 1, 2, 3, ... (or from `--from N`). The first budget that succeeds gives
an **optimal** circuit: every smaller budget was proved impossible. Each found circuit is re-checked exactly by
`python -m cheese.verify` (independent Python code multiplying the gate matrices).

## 2. Exact arithmetic

Every entry of every matrix in the search lies in Z[1/sqrt2, i] and is stored as

    (a + b w + c w^2 + d w^3) / sqrt2^k,   w = e^{i pi/4},  a, b, c, d integers,

with one exponent k per matrix. A matrix is **normalised**: k is lowered while every entry stays divisible by
sqrt2 (`normalise`), so equal matrices have equal integers and equal k. No floating point is used anywhere in
the search or in the verification.

Storage is int32 per coefficient (`Zw32`), arithmetic is int64, and every narrowing is checked: if a value
leaves the int32 range the run aborts with exit 4 ("rebuild with -DSTORE64") instead of continuing with a
wrong value. Coefficients of sqrt2^k U grow like 2^(k/2), so int32 holds every residual of the depths used here.

## 3. The search

### 3.1 Residuals

A **prefix** is the first t gates p = g_1 ... g_t of a would-be circuit, with product P. Its **residual** is

    S = U P^-1,

the unitary the remaining gates must implement: a circuit p r has product U exactly when r's product is S. The
search starts from S = U (the empty prefix) and is done when S = I. Appending a gate g to a prefix updates the
residual by S <- S g^-1 (`apply_inv`): g^-1 acts on at most 3 qubits, so this is a sparse update that mixes
groups of at most 8 columns, followed by normalisation.

The remaining work of a prefix depends only on its residual: two prefixes with equal residuals have exactly
the same completions. This is what makes deduplication (section 5) possible.

Residual matrices are not stored. A frontier node holds only its gate sequence (section 8); the residual is
recomputed from the sequence when needed, because a stored 3-qubit residual is about 1 KB and the frontier
reaches tens of millions of nodes.

### 3.2 Level-synchronous expansion

The search is a breadth-first enumeration of prefixes, one level (= one gate) at a time:

    frontier_0 = { empty prefix }
    for t = 1 .. D:
        for each prefix p in frontier_{t-1} and each candidate g:     (one GPU warp per child)
            skip p.g if a window rule forbids it                       (section 4.1)
            S = residual of p.g
            if S == I: report the circuit p.g                          (found, length t)
            if a lower bound on the gates S needs exceeds D - t: drop  (section 4.2)
            otherwise keep p.g
        frontier_t = kept children, one per residual (class)          (section 5)
        if frontier_t is empty: no circuit of <= D gates exists        (proof)

Levels are processed in order, so the first level where a circuit appears gives the shortest circuit of at
most D gates. If level D is reached without a circuit, that is also a proof: every prefix of length <= D was
compared with the target (the survivors of level D are not even stored).

Within a level, all children of all parents are independent identical small jobs, which is why the work suits a
GPU: 10^5 - 10^8 tiny exact matrices (4x4 to 16x16), not one big state.

### 3.3 Why a pruning rule may remove a prefix

All pruning rules below are **sound** in one precise sense: let C* be the lexicographically smallest circuit
among all shortest circuits for U (candidates ordered by index). Then no rule ever removes a prefix of C*.
Since C*'s prefix of each length survives every level, C* itself is reported at level |C*| if D >= |C*|.
Conversely, a circuit is only reported after its residual was compared with I exactly, so a reported circuit
is always correct. Both directions together give: exit 0 iff a circuit of <= D gates exists, and the reported
one is shortest.

## 4. Pruning

### 4.1 Window rules W2 / W3 (`spec.py: sym_rules`)

Precomputed per gate set and qubit count, by exact multiplication of short gate windows:

- **W2** forbids the pair (g, h) if the product of g then h equals the product of a single candidate or of the
  empty circuit (e.g. `t t` = `s`, `h h` = I), or equals the product of a lexicographically smaller pair
  (e.g. commuting gates on disjoint qubits: only one order is allowed).
- **W3** forbids a triple whose product equals that of a shorter window, or of a lexicographically smaller
  allowed triple. It is only computed when the number of allowed triples is <= 300,000.

Products are compared after removing qubits the window acts trivially on (`canon`), so the rules are about
the gates themselves, not about where in the circuit they sit.

Soundness: if C* contained a forbidden window, replacing it by the shorter or lexicographically smaller
equivalent window would give a shorter circuit for U, or an equally short and lexicographically smaller one.
Both contradict the choice of C*. So C* contains no forbidden window. The search applies the rule to the last
two gates of the parent and the new gate (`window_cut`).

Effect: removes 15-20% of the children at every level (`cut_w3` in the per-level output also counts W2).

### 4.2 Lower bounds on the remaining gates

For a residual S the search computes lower bounds on the number of gates any circuit for S needs. If the
combined bound exceeds the remaining budget D - t, no completion of the prefix fits and it is dropped. A lower
bound never overestimates, so the remaining part of C* (which has exactly the true cost) is never dropped.

The bounds assume the **standard** gate set (spec field `standard 1`): H is the only gate with a sqrt2,
T/Tdg the only non-Clifford gates, CX/CZ the only entangling gates, all others one-qubit Cliffords.

- **H count: h >= k** (the sqrt2 exponent of the normalised residual). Every gate except H has entries that
  are powers of w (no sqrt2), and H changes the exponent by at most 1.
- **H structure (`h_struct`): h >= 2** if S is a monomial matrix whose permutation is not affine over GF(2).
  Without H, Clifford+T circuits only produce monomial matrices with affine permutations; one H cannot fix a
  non-affine permutation.
- **T count (`ptm_wht`)**: the Pauli transfer matrix R_ab = Tr(P_a S P_b S^dag) / 2^n has entries in
  Z[1/sqrt2]; its sqrt2-denominator exponent (sde) is a lower bound on the T count (Gosset et al.): Cliffords
  permute signed Paulis (sde 0), each T raises the sde by at most 1.
  Computation: with S' = sqrt2^k S integral, X_ab = Tr(P_a S' P_b S'^dag) and sde = max over a, b of
  2(n + k) - v(X_ab), where v is the sqrt2-adic valuation. Two facts make this cheap:
  1. Powers of w are units, so they don't change v. Writing each Pauli as a bit flip f and a sign pattern z,
     X_ab equals, up to a unit, sum over x, c of (-1)^(x.za + c.zb) S'[x^fa][c^fb] conj(S'[x][c]). For each
     (fa, fb) that is one 2-D Walsh-Hadamard transform of N^2 products, giving all (za, zb) at once: N^4
     products in total instead of 32^n.
  2. v(X) only needs the low bits: v(X) = 2m + [X / 2^m divisible by sqrt2] with m the smallest 2-adic valuation
     of the four coefficients. Arithmetic modulo 2^32 is therefore exact whenever 2(n + k) <= 64 (a nonzero
     X that is 0 mod 2^32 has v >= 64 and cannot raise the bound); otherwise modulo 2^64.
- **CX count (`cx_bound`)**: for every cut A | B of the qubits, the operator Schmidt rank of S across the cut
  is at most 2^(number of CX/CZ crossing the cut), since each crossing CX/CZ has operator Schmidt rank 2 and
  gates within a side don't change it. So the cut needs ceil(log2 rank) crossing gates. Ranks are computed over
  F_p (p = 998244353, which has an 8th root of unity, so w maps to F_p): the rank mod p is never above the true
  rank, so the bound stays a lower bound. The demands of all cuts are combined by a precomputed table
  (`cx_table` in spec.py): the fewest CX/CZ pairs meeting every cut's demand, an integer program solved for
  every demand vector at spec time.
- **Support**: the number of qubits S acts on non-trivially. Each one-qubit gate covers 1 qubit, each CX/CZ 2.

Combination (`combine`): H, T and CX/CZ gates are distinct gates, so with c CX/CZ gates the circuit needs at
least c + max(h + t, support - 2c) gates; the bound is the minimum of that over c from the CX bound up.

Order of evaluation (one warp per child): windows, then the residual, the identity check, the cheap bounds
(H, structure, support, CX), and only if the child survives those, the T bound (the expensive one). In
practice the bounds remove 70-90% of the children at deep levels (`cut_lb`).

**Gate sets with ccx** (spec field `lite 1`): ccx is a 3-qubit non-Clifford permutation, which breaks the
structure, CX and support bounds. Only two bounds stay valid: h >= k (ccx has no sqrt2) and the T bound
divided by `tdiv`, the largest PTM sde of a single gate (T: 1, ccx: 2, computed exactly by `ptm_sde` in
spec.py), since each T or ccx raises the PTM sde by at most `tdiv`. Weighted CX/support bounds counting ccx
were tried; they were sound but cost more time than they saved.

## 5. Deduplication

After a level, the kept children are sorted by a 128-bit hash of their residual and, among children with the
same residual (compared exactly, not just by hash), only the **lexicographically smallest prefix** is kept.

Soundness: two prefixes with the same residual have the same completions. If C*'s prefix p* were dropped in
favour of a smaller p' with the same residual, then p' followed by the rest of C* would be a circuit for U of
the same length, lexicographically smaller than C*: a contradiction. So p* is the smallest in its group and
is kept. Keeping the lexicographically smallest (rather than any) representative is what makes deduplication
compatible with the window rules.

### 5.1 Symmetry classes

Two residuals related by a qubit relabelling pi (S' = P_pi S P_pi^-1) need the same number of gates, because
the candidate list contains every gate on every qubit placement: relabelling a circuit for S gives a circuit
for S' of the same length. The same holds for complex conjugation when the gate set is closed under it
(T <-> Tdg, S <-> Sdg; H, X, Z, CX, CZ, ccx are real). The search therefore deduplicates **classes**: the
key is the smallest hash over the orbit of the residual (`class_hash`), equality is checked exactly over the
group (`same_class`), and the lexicographically smallest prefix of each class is kept. The group is built from
the candidate names and an element is used only if it maps every candidate to a candidate (`sym_group`); at 3
qubits it has up to 3! x 2 = 12 elements.

Soundness: if C*'s prefix p* were dropped for a smaller p' whose residual is sigma(S*) for a group element sigma,
then p' followed by sigma(rest of C*) - the remaining gates relabelled / conjugated, which has product
sigma(S*) - is a circuit for U of the same length, lexicographically smaller than C*. Contradiction.

Transpose and inverse also preserve the gate count but reverse the gate order; they are not used, because the
distance table (section 6) relies on the group acting gate by gate. Meet in the middle (section 7) uses exact
residuals (no classes). `--no-sym` switches classes off; per-level output is then identical to plain
deduplication.

Effect: depends on the target's own symmetry, since equivalent residuals only meet when U is close to
symmetric. Toffoli 11: largest frontier 172,054 -> 49,503 classes, 1.4 s -> 0.5 s; CCZ 8: 4,960 -> 501;
RUS 27 budget 14: 6.2M -> 3.3M, 64 s -> 34 s; RUS 14: no change.

## 6. Distance table for the last k gates (`--table K`)

The table holds every symmetry class of unitaries that needs at most K gates, with its exact gate count. It is
built by the same level search, started from the identity (residual of a sequence w is W^-1), without bounds
and without a goal, and with one more filter: a class already seen at a lower level is dropped, so each class
sits at the level of its gate count (`SeenBefore`, hash match then exact class check). It depends only on the
candidate list, K and the symmetry group, and is cached in `build/table-n<n>-k<K>-<hash>.bin`.

Completeness: for a class of gate count t, take the lexicographically smallest sequence of length t whose
inverse product lies in the class. It contains no forbidden window (a replacement would give a shorter or
smaller sequence for a member of the class), each of its prefixes has the minimal gate count of its own class,
and by the argument of section 5.1 every prefix is the representative kept at its level. So every class of
gate count <= K appears, at the right level.

Use in the search: after each level t, every kept residual is looked up (class key, binary search in the
sorted table keys). If its class is in the table with gate count c <= D - t, the circuit is the prefix
followed by a continuation read off the table entry (its sequence reversed and inverted, mapped by the group
element that relates the two residuals); the candidate is confirmed exactly (class member, then the whole
circuit multiplied out and compared with U) before it is reported. The levels are searched in order and the
shortest confirmed candidate is reported, so the result is still the shortest circuit of <= D gates.

The search stops at level D - K: there every residual not in the table needs more than K = D - t gates and
cannot be completed. So the last K levels - the ones where the frontier is largest in a deep search - are
replaced by one lookup per residual. A residual whose class is in the table always produces the stored key,
so "not in the table" is never wrong; a hash collision can only cause a candidate that fails the exact check.

Table sizes (Clifford+T, 3 qubits, classes): K=4 5,429; K=5 45,924; growth about 8x per gate.

## 7. Meet in the middle (`--mitm`, unitary targets)

For budget D the circuit is split into a prefix p of a gates and a suffix s of b = D - a gates. The forward
search on U gives residuals R = U P^-1. A second search on U^dag gives sequences q with residuals
R' = U^dag Q^-1; q is the suffix reversed and inverted (q = s^-1, so Q^-1 = S). A circuit p.s exists iff
R = S = U R' for some pair, so the two sides are joined on the hash of R and of U R' (sorted keys, binary
search on the GPU), and every hash match is confirmed exactly on the host. All backward levels 1..b are kept,
so circuits of every length from a to D are covered; shorter ones are found by the forward levels directly.

Completeness: the forward side contains the lexicographically smallest sequence of every unitary of minimal
length a that extends to a solution, the backward side the same for the inverted suffixes, by the arguments of
sections 4.1 and 5 (window rules and deduplication keep, for every unitary, its lexicographically smallest
minimal sequence; bounds never cut a prefix that extends to a solution of <= D gates).

The split is chosen greedily: grow whichever side has the smaller frontier (the next level costs about frontier
x candidates) until the depths add up to D. A side that dies is handled exactly: a dead forward side is a
proof; a dead backward side at level b+1 means no suffix is longer than b, so the join with levels 1..b is
complete.

When it helps: when the expensive levels lie beyond the middle (RUS 14 budget 12: 21 s -> 10 s). When the
largest frontier sits in the middle, both halves pay for it (RUS 13 budget 16: 214 s plain vs 311 s).

## 8. GPU layout

- **Frontier as a tree.** Level l stores, per node, its parent's index in level l-1 and its last gate (5 bytes);
  a prefix is read back by walking up (`path`). Children are generated parent-major, gate-minor, so with
  parents in lexicographic order every level is in lexicographic order and a node's index orders it: dedupe
  sorts by (hash, index) and keeps the first of each group, then restores index order.
- **Per level, in chunks of 2^17 parents:**
  pass 0 computes each parent's residual once (`ParentResidual`);
  one fused kernel (`level_warp`) runs **one warp per child**: window rules, the child residual in shared
  memory (each lane owns entries e, e+32, ...), identity check, all bounds, class hash. Ranks use Gaussian
  elimination across lanes; the T bound's Walsh-Hadamard butterflies are warp shuffles; maxima and hash sums
  are warp reductions. No stack, 64 registers. A separate instantiation is compiled for state targets so the
  unitary kernel carries none of their registers.
  Then survivors are compacted, their keys sorted, deduplicated exactly, and the new level is gathered.
- `--profile` prints the time per pass.

Measured on an RTX 3060 Ti: Toffoli 11 (proof) 30.5 s -> 1.3 s after the T-bound rewrite and fused kernel,
0.5 s with symmetry classes; Toffoli 12 (proof) 8 s.

The CPU build (`./build.sh cpu`, Thrust on OpenMP) runs the same per-child functions as two passes (cheap
bounds, then the T bound on the compacted survivors) and gives the same results.

## 9. State targets (RUS without measurement; CPU build only for now)

`spec.py --ancilla Q... | --inputs X,...` asks for C|in_j> = U|in_j> on chosen basis inputs only. The residual
is the stack of images p|in_j> (transposed, so a gate acts as right-multiplication by g^T), and the goal is the
stack of U|in_j>. Bounds per pair: sde of the vector (one per H), stabilizer nullity (Cliffords keep it, one T
changes it by <= 1) and Schmidt ranks across cuts, computed **exactly** with minors in int64 under a coefficient
bound: a rank mod p could be too low and overstate the gap to a higher target rank. The GPU binary refuses
state targets for now.

## 10. T-count mode (`tcount.sh`, `src/tcount.cu`; separate binary)

A second, independent program finds the optimal **T-count** (T gates only; Cliffords are free; global phase
ignored), the metric of Gosset, Kliuchnikov, Mosca and Russo and of Di Matteo and Mosca (arXiv:1606.07413).

- Form: every Clifford+T unitary is U = R(P_1) ... R(P_t) C with R(P) = ((1 + w) I + (1 - w) P) / 2 a pi/8
  rotation about a non-identity Pauli (one T gate each) and C a Clifford; t minimal is the T-count.
- Representation: the channel matrix Uc[a][b] = Tr(P_a U P_b U^dag) / 2^n (64 x 64 at 3 qubits), entries
  (x + y sqrt2) / sqrt2^e with one exponent per matrix, exact. Peeling a rotation off the left, S <- R(P)^-1 S,
  keeps rows that commute with P and replaces an anticommuting row a by (row a - s row r) / sqrt2 with
  i P P_a = s P_r (checked against direct multiplication for every Pauli on CS, QFT_2 and Toffoli).
- Cosets: right multiplication by a Clifford permutes columns with signs, so the right coset S C is labelled by
  e and the multiset of sign-normalised columns (two unitary channels with equal labels differ by a Clifford).
  Peeling acts on the left, so successors depend only on the coset: deduplicating cosets is exact.
- Table: every coset with T-count <= K, built from the identity level by level (a coset seen at a lower level is
  dropped), cached in `build/tcount-table-n<n>-k<K>.bin`; it does not depend on the target.
- Search: peel rotations off U level by level (deduplicated by coset) and look every residual up. A residual in
  the table at level c gives T-count j + c, and the first level j with a hit gives the optimum, because a
  decomposition of T-count t' is hit at level max(0, t' - K). Coset equality is always confirmed exactly
  (columns sorted by hash, then compared entry by entry); the decomposition is re-checked by independent Python.
- GPU layout: one 64-thread block per coset, the 64 x 64 matrix in shared memory (64 KB with the comparison
  buffer), thread per row for peeling and per column for labels, bitonic sort of the columns for comparisons.

Explicit circuits: `python -m cheese.tcircuit` turns a decomposition into Clifford+T gates. R(P) = V T_q V^dag
with V a CNOT ladder (Z_q -> Z on P's support) followed by H (X) or S H (Y) on each qubit, checked exactly for
every Pauli; the Clifford remainder is synthesised by Qiskit; the whole circuit is checked exactly against U up
to a phase w^k.

Fewer Clifford gates (`--opt`, `src/clifford.cu`, binary `build/cliffopt`): the circuit is cut at its T gates
into Clifford segments D_0 T D_1 ... T D_t. A breadth-first search over the whole n-qubit Clifford group on the
GPU (92,897,280 tableaux for n = 3, each packed into 64 bits, in a GPU hash table; the count is checked against
the group order) gives the minimal gate count of every Clifford over H, S, Sdg, X, Z, CX, and an optimal circuit
by walking the counts down. The T gates are then re-placed: T_q commutes with every Clifford K that fixes Z_q,
so K T' K^-1 with K Z_q' K^-1 = +-Z_q (T' on q', dagger flipped for the minus sign) leaves the circuit unchanged
up to phase, and T = Tdg S, Tdg = T Sdg let a dagger flip move an S into the segment before. A beam search over
all these choices (about 8.8M per T gate at 3 qubits) minimises the summed optimal segment lengths. This gives
optimal segments and a good placement, not a proof of the fewest total gates (docs/PLAN.md item 6).

Unlike Di Matteo and Mosca's parallel collision search (deterministic walks, distinguished points, nothing kept
between targets), the table is built once per qubit count and reused by every target.

## 11. What is checked

- `tests/check.sh`: known verdicts (proofs and found optima) with exact verification of every circuit.
- `tests/rus_check.sh`: RUS mode optima across table depths (every found circuit verified exactly).
- Every new pruning rule was checked against a run without it: identical verdicts and optimal lengths, and
  for rules that should not change the frontier (T-bound rewrite, tree layout, `--no-sym`) identical per-level
  counts on all test, bench and RUS instances.
- Every reported circuit is multiplied out exactly (in the binary for table and meet-in-the-middle results,
  and again by `cheese.verify`).

## 12. RUS mode (`rus.sh`, `src/rus.cu`; separate binary)

T-count of a two-qubit repeat-until-success circuit **with its measurement**: the fewest T gates over all
Clifford+T circuits W (q0 ancilla prepared in |0> and measured, q1 data) whose success branch is the target's
(same success unitary up to Cliffords on either side, same Pr[success]) and whose failure branch is a Clifford.
T-count mode (section 10) bounds the whole unitary instead, which only gives an upper bound here: the other half of
W (ancilla input |1>) and the Cliffords applied after the measurement are free.

- Only J = W(|0> x I) (a 4 x 2 isometry) matters. Write W = C X_t ... X_1 with C a Clifford on the output side;
  J is a valid RUS isometry iff C^-1 J is one of finitely many **roots** up to left Cliffords. Roots
  (`python -m cheese.rus spec`): the success outcome moved to 0, failure block gamma D with D one of the 24
  one-qubit Cliffords and gamma in D[w] with |gamma|^2 = 1 - p, success block u M with M the target's block and
  |u| = 1, u M exact (u = h / m for an entry m of M and |h| = |m|). Both norm equations are solved exactly by
  enumerating Z[w] elements whose coefficients' squares sum to the rational part. Typically 12 roots.
- Representation: J's Pauli transfer matrix Jc[a][b] = Tr(P_a J P_b J^dag) / 2, 16 x 4 (rows two-qubit Paulis,
  columns one-qubit Paulis), entries (x + y sqrt2) / sqrt2^e. A rotation R(P)^-1 on the left mixes rows exactly
  as in section 10; a left Clifford permutes rows with signs. The search works on **left** cosets C J (the
  quotient and the peels are on the same side): the successors of a coset are still well defined, since
  R(P)^-1 C J = C R(+-P')^-1 J and R(-P')^-1 J lies in the coset of R(P')^-1 J (they differ by S_P' on the left).
- Coset key: exponent, row 0 (the identity row, fixed by every Clifford) and the sum of hashes of the
  sign-normalised rows: Clifford-invariant, so no coset is missed. Coset equality is exact: equal row multisets
  are **not** enough (an isometry's PTM has zero and repeated rows; such collisions do occur: the GPU table keeps
  692 cosets at level 4 where exact merging gives 683), so the Clifford is found explicitly. It is fixed by the
  images of X_0, Z_0, X_1, Z_1; candidates are the rows equal (up to sign) to the target's generator rows,
  filtered by their commutation pattern, looked up in a 2^20 table indexed by the four images, and confirmed on
  all 16 rows. In the Python prototype this is 182 us per check against 6.4 ms for trying all 11520 Cliffords.
- Table: cosets reachable from J00 = |0> x I with <= K peels (default K = 8: 5.7M cosets, 1.7 s, cached in
  `build/rus-table-k<K>.bin`; growth about 9x per level). Search: peel the roots level by level (about 10x per
  level); a hit at level j against table level c gives W = R(P_1)..R(P_j) C R(Q_c)^-1..R(Q_1)^-1 with j + c
  T gates, and the first level with a hit is optimal (as in section 10). `rus.sh` searches below the input's own
  T-count by default, so "none found" proves the input optimal.
- Check (`python -m cheese.rus verify`): the circuit is built (rotations as in section 10, the Clifford from its
  generator word) and its blocks are checked exactly: outcome-0 block S with S M^dag = mu I and |mu| = p, i.e.
  S = (unit) M, and outcome-1 block B with B B^dag = q I and B P B^dag proportional to a Pauli.

Representations compared before the GPU version (Python prototype, per node): PTM peel 2.2 us and key 4.2 us;
canonical PTM (minimum over all Cliffords) 18 ms; the isometry itself over Z[w] (statevector style) has no cheap
invariant key, and its canonical form over 11520 Cliffords x 8 phases costs 1.8 s per node.

Results agree across table depths (K = 4, 5, 7, 8 and with K = 1, where the search side does the work) and with
the prototype (forward search without a table). RUS T-count 15 targets: about 6 s on an RTX 3060 Ti.
