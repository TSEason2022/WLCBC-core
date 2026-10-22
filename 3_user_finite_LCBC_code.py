import math
import itertools
from fractions import Fraction
import torch

# ============================================================
# 0) Utilities
# ============================================================
def is_prime(n: int) -> bool:
    """Deterministic primality check for small/moderate n."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True

def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0

def lcmm(nums) -> int:
    res = 1
    for n in nums:
        res = lcm(res, n)
    return res

# ============================================================
# 1) Linear algebra over GF(p) (p prime)
# ============================================================
def mod_inv(a: int, p: int) -> int:
    """Inverse in GF(p), p prime."""
    a = int(a) % p
    if a == 0:
        raise ZeroDivisionError("No inverse for 0 in GF(p)")
    return pow(a, p - 2, p)  # Fermat, p prime

def rref_modp(M: torch.Tensor, p: int):
    """
    Reduced Row Echelon Form over GF(p).
    Returns (RREF, pivot_cols).
    """
    M = (M.clone() % p).to(torch.int64)
    rows, cols = M.shape
    pivot_cols = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        pivot = None
        for rr in range(r, rows):
            if int(M[rr, c]) % p != 0:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        inv = mod_inv(int(M[r, c]), p)
        M[r, :] = (M[r, :] * inv) % p
        for rr in range(rows):
            if rr == r:
                continue
            if int(M[rr, c]) % p != 0:
                factor = int(M[rr, c]) % p
                M[rr, :] = (M[rr, :] - factor * M[r, :]) % p
        pivot_cols.append(c)
        r += 1
    return M, pivot_cols

def rank_modp(M: torch.Tensor, p: int) -> int:
    if M.numel() == 0 or M.shape[1] == 0:
        return 0
    _, piv = rref_modp(M, p)
    return len(piv)

def column_basis(M: torch.Tensor, p: int) -> torch.Tensor:
    """Column-space basis by selecting pivot columns."""
    if M.numel() == 0 or M.shape[1] == 0:
        return M[:, :0].clone()
    _, piv = rref_modp(M, p)
    if len(piv) == 0:
        return M[:, :0].clone()
    return (M[:, piv].clone() % p)

def conditional_rank(A: torch.Tensor, B: torch.Tensor, p: int) -> int:
    """rk(A|B)=rk([A,B]) - rk(B)."""
    if A is None or A.shape[1] == 0:
        return 0
    if B is None or B.shape[1] == 0:
        return rank_modp(A, p)
    return rank_modp(torch.cat([A, B], dim=1), p) - rank_modp(B, p)

def solve_unique_coeff(A: torch.Tensor, Y: torch.Tensor, p: int) -> torch.Tensor:
    """
    Solve A*X=Y over GF(p), assuming A has full column rank and Y in span(A).
    Return X.
    """
    A = (A.clone() % p).to(torch.int64)
    Y = (Y.clone() % p).to(torch.int64)
    d, n = A.shape
    if n == 0:
        if Y.shape[1] == 0:
            return torch.zeros((0, 0), dtype=torch.int64)
        raise ValueError("A has zero columns but Y is non-empty.")
    Aug = torch.cat([A, Y], dim=1)
    R, piv = rref_modp(Aug, p)
    # A full col rank => pivots must be exactly columns 0..n-1
    if piv[:n] != list(range(n)):
        raise ValueError("A is not full column rank; unique solution not guaranteed.")
    X = R[:n, n:] % p
    # Consistency check
    if R[n:, n:].numel() != 0 and torch.any(R[n:, n:] % p != 0):
        raise ValueError("Inconsistent system: Y not in span(A).")
    return X

def intersection_basis(A: torch.Tensor, B: torch.Tensor, p: int) -> torch.Tensor:
    """
    Basis for span(A) ∩ span(B), assuming A,B are bases (independent columns).
    """
    if A is None or A.shape[1] == 0 or B is None or B.shape[1] == 0:
        return A[:, :0].clone() if A is not None else torch.zeros((0, 0), dtype=torch.int64)
    d = A.shape[0]
    a = A.shape[1]
    b = B.shape[1]
    M = torch.cat([A, (-B) % p], dim=1) % p  # d x (a+b)
    R, piv = rref_modp(M, p)
    piv_set = set(piv)
    nvars = a + b
    free_cols = [j for j in range(nvars) if j not in piv_set]
    if len(free_cols) == 0:
        return A[:, :0].clone()
    # Nullspace basis
    null_basis = []
    for free in free_cols:
        z = torch.zeros((nvars, 1), dtype=torch.int64)
        z[free, 0] = 1
        for i, pc in enumerate(piv):
            val = 0
            for fc in free_cols:
                if int(z[fc, 0]) != 0:
                    val = (val - int(R[i, fc])) % p
            z[pc, 0] = val
        null_basis.append(z)
    Z = torch.cat(null_basis, dim=1) % p
    U = Z[:a, :]
    inter = (A @ U) % p
    return column_basis(inter, p)

def complement_basis(space_basis: torch.Tensor, sub_basis: torch.Tensor, p: int) -> torch.Tensor:
    """
    Find C such that [sub_basis, C] spans span(space_basis) and C complements sub_basis.
    """
    if sub_basis is None or sub_basis.shape[1] == 0:
        return space_basis.clone() % p
    space_basis = space_basis.clone() % p
    sub_ind = column_basis(sub_basis, p)
    target_rank = rank_modp(space_basis, p)
    cur = sub_ind.clone()
    C_cols = []
    for j in range(space_basis.shape[1]):
        v = space_basis[:, j:j+1]
        test = torch.cat([cur, v], dim=1)
        if rank_modp(test, p) > rank_modp(cur, p):
            C_cols.append(v)
            cur = test
        if rank_modp(cur, p) == target_rank:
            break
    C = torch.cat(C_cols, dim=1) if C_cols else space_basis[:, :0].clone()
    if rank_modp(torch.cat([sub_ind, C], dim=1), p) != target_rank:
        raise ValueError("Failed to build complement basis.")
    return C % p

def reduce_side_demand(Vp: torch.Tensor, V: torch.Tensor, p: int):
    """
    Remove redundancy:
      - side: keep a basis
      - demand: keep only columns that increase rk([Vp, V])
    """
    Vp_b = column_basis(Vp, p) if Vp is not None else torch.zeros((V.shape[0], 0), dtype=torch.int64)
    cur = Vp_b.clone()
    V_cols = []
    if V is not None and V.shape[1] > 0:
        for j in range(V.shape[1]):
            v = V[:, j:j+1] % p
            test = torch.cat([cur, v], dim=1) if cur.numel() != 0 else v
            if rank_modp(test, p) > rank_modp(cur, p):
                V_cols.append(v)
                cur = test
    V_b = torch.cat(V_cols, dim=1) if V_cols else torch.zeros((Vp_b.shape[0], 0), dtype=torch.int64)
    return Vp_b % p, V_b % p

# ============================================================
# 2) Lemma-2 decomposition (Appendix C) with P20 enforcement
# ============================================================
def lemma2_decomposition(U1: torch.Tensor, U2: torch.Tensor, U3: torch.Tensor, p: int):
    """
    Construct B matrices satisfying Lemma-2 and enforce P20: B1_23 + B2_13 = B3_12.
    """
    # Intersections
    U12 = intersection_basis(U1, U2, p)
    U13 = intersection_basis(U1, U3, p)
    U23 = intersection_basis(U2, U3, p)
    U123 = intersection_basis(U12, U3, p)

    U23_union = column_basis(torch.cat([U2, U3], dim=1), p)
    U13_union = column_basis(torch.cat([U1, U3], dim=1), p)
    U12_union = column_basis(torch.cat([U1, U2], dim=1), p)

    U1_23 = intersection_basis(U1, U23_union, p)
    U2_13 = intersection_basis(U2, U13_union, p)
    U3_12 = intersection_basis(U3, U12_union, p)

    # Steps 1-4
    B123 = U123.clone() % p
    B12  = complement_basis(U12, B123, p)
    B13  = complement_basis(U13, B123, p)
    B23  = complement_basis(U23, B123, p)

    # Steps 5-7
    B1_23 = complement_basis(U1_23, torch.cat([B123, B12, B13], dim=1), p)
    B2_13 = complement_basis(U2_13, torch.cat([B123, B12, B23], dim=1), p)
    B3_12 = complement_basis(U3_12, torch.cat([B123, B13, B23], dim=1), p)

    # Steps 8-10 (for completeness; needed in P20 enforcement)
    B1c = complement_basis(U1, torch.cat([B123, B12, B13, B1_23], dim=1), p)
    B2c = complement_basis(U2, torch.cat([B123, B12, B23, B2_13], dim=1), p)

    # Same width check
    b = B1_23.shape[1]
    if not (B2_13.shape[1] == b and B3_12.shape[1] == b):
        raise ValueError("Expected B1_23, B2_13, B3_12 to have same number of columns.")

    # Steps 11-14 enforce P20 (Appendix C)
    A12 = torch.cat([B123, B12, B13, B23, B1_23, B1c, B2_13, B2c], dim=1)
    R = solve_unique_coeff(A12, B3_12, p)

    idx = 0
    nB123 = B123.shape[1]; R1 = R[idx:idx+nB123, :]; idx += nB123
    nB12  = B12.shape[1];  R2 = R[idx:idx+nB12,  :]; idx += nB12
    nB13  = B13.shape[1];  R3 = R[idx:idx+nB13,  :]; idx += nB13
    nB23  = B23.shape[1];  R4 = R[idx:idx+nB23,  :]; idx += nB23
    nB1_23= B1_23.shape[1];R5 = R[idx:idx+nB1_23,:]; idx += nB1_23
    nB1c  = B1c.shape[1];  _  = R[idx:idx+nB1c,  :]; idx += nB1c
    nB2_13= B2_13.shape[1];R7 = R[idx:idx+nB2_13,:]; idx += nB2_13

    if b > 0:
        if rank_modp(R5, p) != b or rank_modp(R7, p) != b:
            raise ValueError("Unexpected: R5 or R7 not invertible.")

    B3_new = (B3_12
              - (B123 @ R1 if nB123 else 0)
              - (B13  @ R3 if nB13  else 0)
              - (B23  @ R4 if nB23  else 0)) % p
    B2_new = ((B2_13 @ R7 if b else B2_13)
              + (B12   @ R2 if nB12 else 0)) % p
    B1_new = (B1_23 @ R5 if b else B1_23) % p

    if b > 0 and torch.any((B1_new + B2_new - B3_new) % p != 0):
        raise ValueError("P20 enforcement failed: B1_23 + B2_13 != B3_12")

    return {
        "U123": U123 % p, "U12": U12 % p, "U13": U13 % p, "U23": U23 % p,
        "U1_23": U1_23 % p, "U2_13": U2_13 % p, "U3_12": U3_12 % p,
        "B1_23": B1_new % p, "B2_13": B2_new % p, "B3_12": B3_new % p
    }

# ============================================================
# 3) LP solver (Theorem-2) by vertex enumeration (exact Fractions)
# ============================================================
def solve_linear_system_fraction(A, b):
    """Solve A x = b in Fractions, return x or None."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    r = 0
    for c in range(n):
        pivot = None
        for rr in range(r, n):
            if M[rr][c] != 0:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            M[r], M[pivot] = M[pivot], M[r]
        pivval = M[r][c]
        for cc in range(c, n + 1):
            M[r][cc] /= pivval
        for rr in range(n):
            if rr == r:
                continue
            if M[rr][c] != 0:
                factor = M[rr][c]
                for cc in range(c, n + 1):
                    M[rr][cc] -= factor * M[r][cc]
        r += 1
        if r == n:
            break
    for rr in range(n):
        if all(M[rr][cc] == 0 for cc in range(n)) and M[rr][n] != 0:
            return None
    if r < n:
        return None
    return [M[i][n] for i in range(n)]

def solve_lp_lambdas(m1, m2, m3, ranks_dict):
    """
    Variables x = [λ123, λ12, λ13, λ23, λ] (Fractions)
    Objective: minimize f = (m1+m2+m3) - 2λ123 - λ12 - λ13 - λ23 - λ
    Constraints: (21)-(24) + nonnegativity.
    """
    cons = []
    def add_con(a, b):
        cons.append((tuple(Fraction(ai) for ai in a), Fraction(b)))

    # (21)
    for i in [1, 2, 3]:
        add_con([1, 0, 0, 0, 0], ranks_dict[f"c21_{i}"])
    # (22)
    add_con([1, 1, 0, 0, 0], ranks_dict["c22_12_1"])
    add_con([1, 1, 0, 0, 0], ranks_dict["c22_12_2"])
    add_con([1, 0, 1, 0, 0], ranks_dict["c22_13_1"])
    add_con([1, 0, 1, 0, 0], ranks_dict["c22_13_3"])
    add_con([1, 0, 0, 1, 0], ranks_dict["c22_23_2"])
    add_con([1, 0, 0, 1, 0], ranks_dict["c22_23_3"])
    # (23)
    add_con([1, 1, 1, 0, 0], ranks_dict["c23_1"])
    add_con([1, 1, 0, 1, 0], ranks_dict["c23_2"])
    add_con([1, 0, 1, 1, 0], ranks_dict["c23_3"])
    # (24)
    add_con([1, 1, 1, 0, 1], ranks_dict["c24_1"])
    add_con([1, 1, 0, 1, 1], ranks_dict["c24_2"])
    add_con([1, 0, 1, 1, 1], ranks_dict["c24_3"])
    # λ >= 0
    for j in range(5):
        a = [0] * 5
        a[j] = -1
        add_con(a, 0)

    nvar = 5
    best_x = [Fraction(0)] * nvar
    best_f = Fraction(m1 + m2 + m3)

    for eq_idx in itertools.combinations(range(len(cons)), nvar):
        A = []
        b = []
        for idx in eq_idx:
            a, rhs = cons[idx]
            A.append(list(a))
            b.append(rhs)
        x = solve_linear_system_fraction(A, b)
        if x is None:
            continue
        # feasibility
        ok = True
        for a, rhs in cons:
            lhs = sum(a[j] * x[j] for j in range(nvar))
            if lhs > rhs:
                ok = False
                break
        if not ok:
            continue
        f = Fraction(m1 + m2 + m3) - (Fraction(2) * x[0] + x[1] + x[2] + x[3] + x[4])
        if f < best_f:
            best_f = f
            best_x = x
    return best_x, best_f

# ============================================================
# 4) Field extension GF(p^z) via multiplication matrices over GF(p)
# ============================================================
def poly_trim(a):
    while len(a) > 0 and a[-1] == 0:
        a.pop()
    return a

def poly_divmod(a, b, p):
    a = a[:]
    b = b[:]
    poly_trim(a); poly_trim(b)
    if not b:
        raise ZeroDivisionError
    if not a:
        return [], []
    deg_a = len(a) - 1
    deg_b = len(b) - 1
    if deg_a < deg_b:
        return [], a
    inv_lead = mod_inv(b[-1], p)
    q = [0] * (deg_a - deg_b + 1)
    r = a[:]
    while r and (len(r) - 1) >= deg_b:
        deg_r = len(r) - 1
        lead_coeff = r[-1] * inv_lead % p
        shift = deg_r - deg_b
        q[shift] = lead_coeff
        for i in range(deg_b + 1):
            r[i + shift] = (r[i + shift] - lead_coeff * b[i]) % p
        poly_trim(r)
    poly_trim(q)
    return q, r

def poly_mod(a, mod_poly, p):
    _, r = poly_divmod(a, mod_poly, p)
    return r

def all_monic_polys(deg, p):
    if deg == 0:
        yield [1]
        return
    for coeffs in itertools.product(range(p), repeat=deg):
        yield list(coeffs) + [1]

def is_irreducible(f, p):
    m = len(f) - 1
    for d in range(1, m // 2 + 1):
        for g in all_monic_polys(d, p):
            _, r = poly_divmod(f, g, p)
            if not r:
                return False
    return True

def find_irreducible_poly(p, m):
    for f in all_monic_polys(m, p):
        if is_irreducible(f, p):
            return f
    raise RuntimeError(f"Cannot find irreducible polynomial for p={p}, m={m}")

def int_to_coeffs(a, p, m):
    coeffs = []
    x = a
    for _ in range(m):
        coeffs.append(int(x % p))
        x //= p
    return coeffs

def mul_by_element_matrix(elem_int, p, m, irr_poly):
    """
    Return m×m matrix M so that for row coeff vector x: coeffs(x*elem) = x @ M (mod p).
    """
    a = int_to_coeffs(elem_int, p, m)
    rows = []
    for i in range(m):
        tmp = [0] * i + a[:]      # x^i * a(x)
        tmp_red = poly_mod(tmp, irr_poly, p)
        tmp_red = tmp_red + [0] * (m - len(tmp_red))
        rows.append(tmp_red[:m])
    return torch.tensor(rows, dtype=torch.int64) % p

def precompute_mul_matrices(p, m, irr_poly):
    q = p ** m
    mulM = torch.zeros((q, m, m), dtype=torch.int64)
    for a in range(q):
        mulM[a] = mul_by_element_matrix(a, p, m, irr_poly)
    return mulM

def random_coeff_block(r, c, p, m, mulM, rng):
    """
    Random (r×c) over GF(p^m), represented as (r*m)×(c*m) over GF(p)
    by mapping each entry to its multiplication matrix.
    """
    if r == 0 or c == 0:
        return torch.zeros((r * m, c * m), dtype=torch.int64)
    q = mulM.shape[0]
    idx = torch.randint(0, q, (r, c), dtype=torch.int64, generator=rng)
    blocks = mulM[idx]  # (r,c,m,m)
    big = blocks.permute(0, 2, 1, 3).contiguous().view(r * m, c * m) % p
    return big

def expand_scalar_matrix_to_block(M_scalar, p, m):
    """
    Expand (a×b) scalar matrix over GF(p) into (a*m)×(b*m) block matrix with blocks scalar*I_m.
    """
    if M_scalar.numel() == 0 or M_scalar.shape[1] == 0:
        return torch.zeros((M_scalar.shape[0] * m, 0), dtype=torch.int64)
    I = torch.eye(m, dtype=torch.int64)
    return torch.kron(M_scalar % p, I) % p

def choose_field_extension_degree(p, d_ext):
    """
    Guarantee bound from random-coding existence argument:
    choose smallest z with p^z > 3*d_ext.
    """
    z = 1
    while p ** z <= 3 * d_ext:
        z += 1
    return z

# ============================================================
# 5) Main builder: Theorem-2 + matrix extension + (auto) field extension
# ============================================================
def kron_eye(L: int, M: torch.Tensor) -> torch.Tensor:
    if L == 1:
        return M.clone()
    I = torch.eye(L, dtype=torch.int64)
    return torch.kron(I, M)

def build_lcbc_3user_code(
    p: int,
    Vp_list,
    V_list,
    seed=0,
    max_tries=5000,
    verbose=False,
    z_max=None,
):
    """
    Build optimal 3-user LCBC code over GF(p), following the paper's construction:
      - Solve Theorem-2 LP for λ
      - Matrix extension to make λ integral
      - Random mixing; if base field too small, auto-try GF(p^z) via multiplication-matrix representation
    Output is a *base-field* vector code:
      batch size L_total = L_scale * z_field,
      data length = d * L_total,
      broadcast length = N (already base-field length).
    """
    if not is_prime(p):
        raise ValueError("p must be prime.")
    assert len(Vp_list) == 3 and len(V_list) == 3
    d = V_list[0].shape[0]

    # (A) redundancy removal
    Vp = []
    V = []
    for k in range(3):
        Vp_k, V_k = reduce_side_demand(Vp_list[k], V_list[k], p)
        Vp.append(Vp_k)
        V.append(V_k)

    m = [V[k].shape[1] for k in range(3)]

    # (B) Lemma-2 decomposition (base field)
    U = [column_basis(torch.cat([Vp[k], V[k]], dim=1), p) for k in range(3)]
    dec = lemma2_decomposition(U[0], U[1], U[2], p)
    U123, U12, U13, U23 = dec["U123"], dec["U12"], dec["U13"], dec["U23"]
    U1_23, U2_13, U3_12 = dec["U1_23"], dec["U2_13"], dec["U3_12"]
    B1_23, B2_13, B3_12 = dec["B1_23"], dec["B2_13"], dec["B3_12"]

    # (C) LP ranks
    ranks = {}
    for i in [1, 2, 3]:
        ranks[f"c21_{i}"] = conditional_rank(U123, Vp[i - 1], p)
    ranks["c22_12_1"] = conditional_rank(U12, Vp[0], p)
    ranks["c22_12_2"] = conditional_rank(U12, Vp[1], p)
    ranks["c22_13_1"] = conditional_rank(U13, Vp[0], p)
    ranks["c22_13_3"] = conditional_rank(U13, Vp[2], p)
    ranks["c22_23_2"] = conditional_rank(U23, Vp[1], p)
    ranks["c22_23_3"] = conditional_rank(U23, Vp[2], p)
    ranks["c23_1"] = conditional_rank(column_basis(torch.cat([U12, U13], dim=1), p), Vp[0], p)
    ranks["c23_2"] = conditional_rank(column_basis(torch.cat([U12, U23], dim=1), p), Vp[1], p)
    ranks["c23_3"] = conditional_rank(column_basis(torch.cat([U13, U23], dim=1), p), Vp[2], p)
    ranks["c24_1"] = conditional_rank(U1_23, Vp[0], p)
    ranks["c24_2"] = conditional_rank(U2_13, Vp[1], p)
    ranks["c24_3"] = conditional_rank(U3_12, Vp[2], p)

    x_opt, f_opt = solve_lp_lambdas(m[0], m[1], m[2], ranks)
    lam123, lam12, lam13, lam23, lam = x_opt
    lambdas = {
        "lam123": lam123,
        "lam12": lam12,
        "lam13": lam13,
        "lam23": lam23,
        "lam": lam,
        "f_opt": f_opt,
    }

    # (D) matrix extension factor (to make lambdas integers)
    L_scale = lcmm([lam123.denominator, lam12.denominator, lam13.denominator, lam23.denominator, lam.denominator]) or 1
    lam123_i = int(lam123 * L_scale)
    lam12_i  = int(lam12  * L_scale)
    lam13_i  = int(lam13  * L_scale)
    lam23_i  = int(lam23  * L_scale)
    lam_i    = int(lam    * L_scale)

    t1_i = m[0] * L_scale - (lam123_i + lam12_i + lam13_i + lam_i)
    t2_i = m[1] * L_scale - (lam123_i + lam12_i + lam23_i + lam_i)
    t3_i = m[2] * L_scale - (lam123_i + lam13_i + lam23_i + lam_i)
    if min(t1_i, t2_i, t3_i) < 0:
        raise ValueError("Negative t_i after scaling; inconsistent.")

    d_ext = d * L_scale
    z_bound = choose_field_extension_degree(p, d_ext)
    if z_max is not None:
        z_bound = min(z_bound, int(z_max))

    # (E) scalar matrices after matrix extension (over GF(p))
    Vp_ext = [kron_eye(L_scale, Vp[k]) % p for k in range(3)]
    V_ext  = [kron_eye(L_scale, V[k]) % p for k in range(3)]
    U_ext  = [column_basis(torch.cat([Vp_ext[k], V_ext[k]], dim=1), p) for k in range(3)]
    U123e  = kron_eye(L_scale, U123) % p
    U12e   = kron_eye(L_scale, U12) % p
    U13e   = kron_eye(L_scale, U13) % p
    U23e   = kron_eye(L_scale, U23) % p
    B1e    = kron_eye(L_scale, B1_23) % p
    B2e    = kron_eye(L_scale, B2_13) % p
    B3e    = kron_eye(L_scale, B3_12) % p

    r123, r12, r13, r23, rb = U123e.shape[1], U12e.shape[1], U13e.shape[1], U23e.shape[1], B1e.shape[1]

    # (F) try z=1..z_bound until random mixing succeeds
    for z in range(1, z_bound + 1):
        rng = torch.Generator()
        rng.manual_seed(seed + 1000 * z)

        irr_poly = find_irreducible_poly(p, z)
        mulM = precompute_mul_matrices(p, z, irr_poly)

        # expand scalar matrices to block matrices (coefficient-domain over GF(p))
        Vp_base = [expand_scalar_matrix_to_block(Vp_ext[k], p, z) for k in range(3)]
        V_base  = [expand_scalar_matrix_to_block(V_ext[k],  p, z) for k in range(3)]
        U1_base_basis = expand_scalar_matrix_to_block(U_ext[0], p, z)
        U2_base_basis = expand_scalar_matrix_to_block(U_ext[1], p, z)
        U3_base_basis = expand_scalar_matrix_to_block(U_ext[2], p, z)

        U123_base = expand_scalar_matrix_to_block(U123e, p, z)
        U12_base  = expand_scalar_matrix_to_block(U12e,  p, z)
        U13_base  = expand_scalar_matrix_to_block(U13e,  p, z)
        U23_base  = expand_scalar_matrix_to_block(U23e,  p, z)

        B1_base   = expand_scalar_matrix_to_block(B1e, p, z)
        B2_base   = expand_scalar_matrix_to_block(B2e, p, z)
        B3_base   = expand_scalar_matrix_to_block(B3e, p, z)

        tries = max_tries if z == z_bound else min(500, max_tries)

        for attempt in range(1, tries + 1):
            # random coefficient matrices over GF(p^z) represented as GF(p) blocks
            N123 = random_coeff_block(r123, lam123_i, p, z, mulM, rng)
            N12  = random_coeff_block(r12,  lam12_i,  p, z, mulM, rng)
            N13  = random_coeff_block(r13,  lam13_i,  p, z, mulM, rng)
            N23  = random_coeff_block(r23,  lam23_i,  p, z, mulM, rng)

            M12  = random_coeff_block(r12, lam_i, p, z, mulM, rng)
            M13  = random_coeff_block(r13, lam_i, p, z, mulM, rng)
            M23  = random_coeff_block(r23, lam_i, p, z, mulM, rng)
            M    = random_coeff_block(rb,  lam_i, p, z, mulM, rng)

            # broadcast blocks
            G123 = (U123_base @ N123) % p
            G12  = (U12_base  @ N12 ) % p
            G13  = (U13_base  @ N13 ) % p
            G23  = (U23_base  @ N23 ) % p

            Gl1  = (U12_base @ M12 + U13_base @ M13 + B1_base @ M) % p
            Gl2  = ((-U12_base % p) @ M12 + U23_base @ M23 + B2_base @ M) % p
            Gl3  = (Gl1 + Gl2) % p  # derived, for user3 decoding

            # rank checks (full col rank)
            A1 = torch.cat([Vp_base[0], G123, G12, G13, Gl1], dim=1)
            A2 = torch.cat([Vp_base[1], G123, G12, G23, Gl2], dim=1)
            A3 = torch.cat([Vp_base[2], G123, G13, G23, Gl3], dim=1)
            if rank_modp(A1, p) != A1.shape[1] or rank_modp(A2, p) != A2.shape[1] or rank_modp(A3, p) != A3.shape[1]:
                continue

            # 1-bird-1-stone completion
            U1t = complement_basis(U1_base_basis, A1, p)
            U2t = complement_basis(U2_base_basis, A2, p)
            U3t = complement_basis(U3_base_basis, A3, p)
            if not (U1t.shape[1] == t1_i * z and U2t.shape[1] == t2_i * z and U3t.shape[1] == t3_i * z):
                continue

            # final transmitted matrix
            blocks = {
                "G123": G123, "G12": G12, "G13": G13, "G23": G23,
                "Gl1": Gl1, "Gl2": Gl2,
                "U1t": U1t, "U2t": U2t, "U3t": U3t,
            }
            order = ["G123", "G12", "G13", "G23", "Gl1", "Gl2", "U1t", "U2t", "U3t"]
            G = torch.cat([blocks[name] for name in order], dim=1) % p

            # slices
            slices = {}
            st = 0
            for name in order:
                en = st + blocks[name].shape[1]
                slices[name] = slice(st, en)
                st = en

            # decoding matrices in base-field coefficient domain
            A1_full = torch.cat([Vp_base[0], G123, G12, G13, Gl1, U1t], dim=1)
            A2_full = torch.cat([Vp_base[1], G123, G12, G23, Gl2, U2t], dim=1)
            A3_full = torch.cat([Vp_base[2], G123, G13, G23, Gl3, U3t], dim=1)

            if rank_modp(A1_full, p) != U1_base_basis.shape[1] or rank_modp(A2_full, p) != U2_base_basis.shape[1] or rank_modp(A3_full, p) != U3_base_basis.shape[1]:
                continue

            T1 = solve_unique_coeff(A1_full, V_base[0], p)
            T2 = solve_unique_coeff(A2_full, V_base[1], p)
            T3 = solve_unique_coeff(A3_full, V_base[2], p)

            if verbose:
                N = G.shape[1]
                L_total = L_scale * z
                print(f"[OK] z={z}, attempts={attempt}, N={N}, L_total={L_total}, Δ={N/L_total:.6f}, irr_poly={irr_poly}")

            return {
                "p": p,
                "d": d,
                "L_scale": L_scale,
                "z_field": z,
                "L_total": L_scale * z,
                "lambdas": lambdas,
                "lambdas_int": {
                    "lam123": lam123_i, "lam12": lam12_i, "lam13": lam13_i, "lam23": lam23_i,
                    "lam": lam_i, "t1": t1_i, "t2": t2_i, "t3": t3_i, "L_scale": L_scale,
                },
                "G": G % p,
                "slices": slices,
                "decode_Tv": [T1 % p, T2 % p, T3 % p],
                "Vp_base": Vp_base,
                "V_base": V_base,
                "order": order,
            }

        if verbose:
            print(f"[WARN] z={z} failed after {tries} attempts; trying larger z...")

    raise RuntimeError("Failed to build a code. Increase max_tries or use a larger prime p.")

# ============================================================
# 6) p-PAM modulation + AWGN + hard decision (side/channel SNR knobs)
# ============================================================
def pam_scale(p: int) -> float:
    # avg power of levels {-(p-1),-(p-3),...,p-1} is (p^2-1)/3
    return math.sqrt((p * p - 1.0) / 3.0)

def awgn_hard_demod(symbols_int: torch.Tensor, p: int, snr_db, rng: torch.Generator):
    """
    symbols_int: int64 in {0,...,p-1}
    snr_db: float (dB) or None/inf for noiseless
    """
    if snr_db is None or snr_db == float("inf"):
        return symbols_int.clone()
    scale = pam_scale(p)
    x = (2.0 * symbols_int.to(torch.float32) - float(p - 1)) / scale  # normalized to unit avg power
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    noise_var = 1.0 / snr_lin
    n = torch.randn(x.shape, generator=rng) * math.sqrt(noise_var)
    y = x + n
    y_unnorm = y * scale
    s_hat = torch.round((y_unnorm + float(p - 1)) / 2.0).to(torch.int64)
    s_hat = torch.clamp(s_hat, 0, p - 1)
    return s_hat

# ============================================================
# 7) End-to-end simulation
# ============================================================
def simulate_lcbc(code, num_batches=2000, snr_side_db=None, snr_channel_db=None, seed=0):
    """
    SER simulation with adjustable SNR:
      - snr_side_db: None or float or (snr1,snr2,snr3)
      - snr_channel_db: None or float or (snr1,snr2,snr3)
    """
    p = code["p"]
    d = code["d"]
    L_total = code["L_total"]
    G = code["G"]
    slices = code["slices"]
    Tv = code["decode_Tv"]
    Vp_base = code["Vp_base"]
    V_base = code["V_base"]

    rng = torch.Generator()
    rng.manual_seed(seed)

    X = torch.randint(0, p, (num_batches, d * L_total), dtype=torch.int64, generator=rng)
    S = (X @ G) % p
    Wp_true = [(X @ Vp_base[k]) % p for k in range(3)]
    W_true = [(X @ V_base[k]) % p for k in range(3)]

    def pick(snr, k):
        if snr is None:
            return None
        if isinstance(snr, (list, tuple)):
            return snr[k]
        return snr

    # side noise
    Wp_obs = [awgn_hard_demod(Wp_true[k], p, pick(snr_side_db, k), rng) for k in range(3)]
    # broadcast channel noise (each user sees its own channel)
    S_rx = [awgn_hard_demod(S, p, pick(snr_channel_db, k), rng) for k in range(3)]

    # decode
    W_hat = []
    for k in range(3):
        if k == 0:
            Y = torch.cat([
                Wp_obs[0],
                S_rx[0][:, slices["G123"]],
                S_rx[0][:, slices["G12"]],
                S_rx[0][:, slices["G13"]],
                S_rx[0][:, slices["Gl1"]],
                S_rx[0][:, slices["U1t"]],
            ], dim=1) % p
        elif k == 1:
            Y = torch.cat([
                Wp_obs[1],
                S_rx[1][:, slices["G123"]],
                S_rx[1][:, slices["G12"]],
                S_rx[1][:, slices["G23"]],
                S_rx[1][:, slices["Gl2"]],
                S_rx[1][:, slices["U2t"]],
            ], dim=1) % p
        else:
            # λ3 block is computed as (λ1 + λ2) mod p, per the construction
            lam3 = (S_rx[2][:, slices["Gl1"]] + S_rx[2][:, slices["Gl2"]]) % p
            Y = torch.cat([
                Wp_obs[2],
                S_rx[2][:, slices["G123"]],
                S_rx[2][:, slices["G13"]],
                S_rx[2][:, slices["G23"]],
                lam3,
                S_rx[2][:, slices["U3t"]],
            ], dim=1) % p

        W_hat_k = (Y @ Tv[k]) % p
        W_hat.append(W_hat_k)

    ser = [(W_hat[k] != W_true[k]).to(torch.float32).mean().item() for k in range(3)]
    return ser

# ============================================================
# 8) Demo: Paper Example 2 (3-user, GF(3))
# ============================================================
def demo_example2():
    # Field GF(3), data X^T=(A,B,C)
    p = 3
    e1 = torch.tensor([[1],[0],[0]], dtype=torch.int64)
    e2 = torch.tensor([[0],[1],[0]], dtype=torch.int64)
    e3 = torch.tensor([[0],[0],[1]], dtype=torch.int64)

    # User1: side A, wants B+C
    Vp1 = e1
    V1  = (e2 + e3) % p

    # User2: side B, wants A+C
    Vp2 = e2
    V2  = (e1 + e3) % p

    # User3: side C, wants A+2B
    Vp3 = e3
    V3  = (e1 + 2*e2) % p

    code = build_lcbc_3user_code(
        p,
        [Vp1, Vp2, Vp3],
        [V1,  V2,  V3],
        seed=1,
        max_tries=3000,
        verbose=True
    )

    print("lambdas (Fractions):", code["lambdas"])
    print("L_total:", code["L_total"], "z_field:", code["z_field"], "G shape:", tuple(code["G"].shape))

    # noiseless correctness check
    ser0 = simulate_lcbc(code, num_batches=2000, snr_side_db=None, snr_channel_db=None, seed=0)
    print("SER (noiseless):", ser0)

    # Adjustable SNR demo (single value => all users)
    ser1 = simulate_lcbc(code, num_batches=10000, snr_side_db=20.0, snr_channel_db=10.0, seed=0)
    print("SER (side=20dB, channel=10dB):", ser1)

    # Per-user SNR demo (list/tuple of length 3)
    ser2 = simulate_lcbc(code, num_batches=10000,
                         snr_side_db=[25.0, 20.0, 15.0],
                         snr_channel_db=[12.0, 10.0, 8.0],
                         seed=0)
    print("SER (per-user SNR):", ser2)

if __name__ == "__main__":
    demo_example2()
