# ============================================================
# 3-user LCBC (Yao & Jafar) construction over REAL field R
# - Strictly follows the same pipeline as the finite-field code:
#   (1) redundancy removal
#   (2) Lemma-2 decomposition (Appendix C) + P20 enforcement
#   (3) Theorem-2 LP to get lambdas
#   (4) matrix extension to make lambdas integers
#   (5) random mixing to build G and per-user decoding matrices T_k
# - Messages X are Gaussian (used in simulation), and we add:
#   side-noise + channel-noise (AWGN)
# - NO "anti-noise": no MMSE, no denoising, no quantization/hard-decision.
#   Decoder is the fixed noiseless linear map derived from the construction.
# Coded by: Shuo Tan
# Date: 2026.1.24
# ============================================================

import math
import itertools
from fractions import Fraction
import torch


# ============================================================
# 0) Utilities
# ============================================================
def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def lcmm(nums) -> int:
    res = 1
    for n in nums:
        res = lcm(res, int(n))
    return res


# ============================================================
# 1) Linear algebra over REAL field R (numerical)
#    - basis via SVD
#    - intersection via principal angles
#    - complement via orthogonal projection
# ============================================================
def rank_real(M: torch.Tensor, tol: float = None) -> int:
    """
    Numerical rank over R using SVD.
    """
    if M is None or M.numel() == 0 or M.shape[1] == 0:
        return 0
    M = M.to(torch.float64)
    S = torch.linalg.svdvals(M)
    if tol is None:
        eps = torch.finfo(S.dtype).eps
        tol = float(S.max().item()) * max(M.shape) * eps
        tol = max(tol, 1e-12)
    return int((S > tol).sum().item())


def orth_basis(M: torch.Tensor, tol: float = None) -> torch.Tensor:
    """
    Orthonormal basis for column space of M (d x n) using SVD.
    Returns d x r with orthonormal columns.
    """
    if M is None or M.numel() == 0 or M.shape[1] == 0:
        d = 0 if M is None else M.shape[0]
        return torch.zeros((d, 0), dtype=torch.float64)

    M = M.to(torch.float64)
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    if tol is None:
        eps = torch.finfo(S.dtype).eps
        tol = float(S.max().item()) * max(M.shape) * eps
        tol = max(tol, 1e-12)
    r = int((S > tol).sum().item())
    if r == 0:
        return torch.zeros((M.shape[0], 0), dtype=torch.float64)
    return U[:, :r].contiguous()


def conditional_rank_real(A: torch.Tensor, B: torch.Tensor, tol: float = None) -> int:
    """
    rk(A|B) = rk([A,B]) - rk(B)
    """
    if A is None or A.shape[1] == 0:
        return 0
    if B is None or B.shape[1] == 0:
        return rank_real(A, tol)
    return rank_real(torch.cat([A, B], dim=1), tol) - rank_real(B, tol)


def intersection_basis_real(
    A: torch.Tensor,
    B: torch.Tensor,
    tol: float = None,
    tol_angle: float = 1e-8
) -> torch.Tensor:
    """
    Orthonormal basis for span(A) ∩ span(B) using principal angles.

    If QA and QB are orthonormal bases, singular values of QA^T QB are cos(theta_i).
    Intersection corresponds to theta_i = 0 => cos=1.
    """
    if A is None or A.numel() == 0 or A.shape[1] == 0 or B is None or B.numel() == 0 or B.shape[1] == 0:
        d = 0 if A is None else A.shape[0]
        return torch.zeros((d, 0), dtype=torch.float64)

    QA = orth_basis(A, tol)
    QB = orth_basis(B, tol)
    if QA.shape[1] == 0 or QB.shape[1] == 0:
        return torch.zeros((A.shape[0], 0), dtype=torch.float64)

    C = QA.T @ QB
    U, S, _ = torch.linalg.svd(C, full_matrices=False)
    idx = (S > 1.0 - tol_angle).nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return torch.zeros((A.shape[0], 0), dtype=torch.float64)

    inter = QA @ U[:, idx]
    return orth_basis(inter, tol)


def complement_basis_real(space_basis: torch.Tensor, sub_basis: torch.Tensor, tol: float = None) -> torch.Tensor:
    """
    Return C such that [sub_in_space, C] spans span(space_basis),
    and C is an orthonormal basis for the complement subspace.

    We robustify by projecting sub_basis onto the space first.
    """
    Qs = orth_basis(space_basis, tol)
    if Qs.shape[1] == 0:
        return torch.zeros((space_basis.shape[0], 0), dtype=torch.float64)

    if sub_basis is None or sub_basis.numel() == 0 or sub_basis.shape[1] == 0:
        return Qs

    Qb = orth_basis(sub_basis, tol)
    if Qb.shape[1] == 0:
        return Qs

    # subspace of (sub_basis ∩ space_basis) via projection
    Qb_proj = Qs @ (Qs.T @ Qb)
    Qb_in = orth_basis(Qb_proj, tol)
    if Qb_in.shape[1] == 0:
        return Qs

    # complement inside Qs
    Qcomp_raw = Qs - Qb_in @ (Qb_in.T @ Qs)
    Qcomp = orth_basis(Qcomp_raw, tol)
    return Qcomp


def solve_unique_coeff_real(A: torch.Tensor, Y: torch.Tensor, tol: float = None) -> torch.Tensor:
    """
    Solve A X = Y over R, assuming A has full column rank and Y in span(A).
    Return X (n x k).
    """
    A = A.to(torch.float64)
    Y = Y.to(torch.float64)

    d, n = A.shape
    if n == 0:
        if Y.numel() == 0 or Y.shape[1] == 0:
            return torch.zeros((0, 0), dtype=torch.float64)
        raise ValueError("A has zero columns but Y is non-empty.")

    rA = rank_real(A, tol)
    if rA != n:
        raise ValueError(f"A is not full column rank: rank={rA}, n={n}")

    # least-squares solution (unique due to full column rank + consistency)
    X = torch.linalg.lstsq(A, Y).solution

    # consistency check (numerical)
    resid = torch.linalg.norm(A @ X - Y)
    denom = max(1.0, float(torch.linalg.norm(Y).item()))
    if float(resid.item()) / denom > 1e-6:
        raise ValueError(f"Inconsistent system (numerical): rel_resid={float(resid.item())/denom}")
    return X


def reduce_side_demand_real(Vp: torch.Tensor, V: torch.Tensor, tol: float = None):
    """
    Remove redundancy:
      - side: keep a basis
      - demand: keep only new directions beyond side
    """
    if Vp is None:
        Vp = torch.zeros((V.shape[0], 0), dtype=torch.float64)
    if V is None:
        V = torch.zeros((Vp.shape[0], 0), dtype=torch.float64)

    Vp_b = orth_basis(Vp, tol)
    union = orth_basis(torch.cat([Vp_b, V.to(torch.float64)], dim=1), tol)
    V_b = complement_basis_real(union, Vp_b, tol)
    return Vp_b, V_b


# ============================================================
# 2) Lemma-2 decomposition (Appendix C) + P20 enforcement
# ============================================================
def lemma2_decomposition_real(U1: torch.Tensor, U2: torch.Tensor, U3: torch.Tensor, tol: float = None):
    """
    Real-field analogue of Lemma-2 decomposition (Appendix C),
    with enforcement of P20:
        B1_23 + B2_13 = B3_12
    """
    U1 = orth_basis(U1, tol)
    U2 = orth_basis(U2, tol)
    U3 = orth_basis(U3, tol)

    # Intersections
    U12 = intersection_basis_real(U1, U2, tol)
    U13 = intersection_basis_real(U1, U3, tol)
    U23 = intersection_basis_real(U2, U3, tol)
    U123 = intersection_basis_real(U12, U3, tol)

    # Unions
    U23_union = orth_basis(torch.cat([U2, U3], dim=1), tol)
    U13_union = orth_basis(torch.cat([U1, U3], dim=1), tol)
    U12_union = orth_basis(torch.cat([U1, U2], dim=1), tol)

    U1_23 = intersection_basis_real(U1, U23_union, tol)
    U2_13 = intersection_basis_real(U2, U13_union, tol)
    U3_12 = intersection_basis_real(U3, U12_union, tol)

    # Steps 1-4
    B123 = U123.clone()
    B12 = complement_basis_real(U12, B123, tol)
    B13 = complement_basis_real(U13, B123, tol)
    B23 = complement_basis_real(U23, B123, tol)

    # Steps 5-7
    B1_23 = complement_basis_real(U1_23, torch.cat([B123, B12, B13], dim=1), tol)
    B2_13 = complement_basis_real(U2_13, torch.cat([B123, B12, B23], dim=1), tol)
    B3_12 = complement_basis_real(U3_12, torch.cat([B123, B13, B23], dim=1), tol)

    # Steps 8-10 (needed for P20 enforcement)
    B1c = complement_basis_real(U1, torch.cat([B123, B12, B13, B1_23], dim=1), tol)
    B2c = complement_basis_real(U2, torch.cat([B123, B12, B23, B2_13], dim=1), tol)

    b = B1_23.shape[1]
    if not (B2_13.shape[1] == b and B3_12.shape[1] == b):
        raise ValueError("Expected B1_23, B2_13, B3_12 to have same number of columns (b).")

    # Steps 11-14 enforce P20 by re-basing (Appendix C idea)
    A12 = torch.cat([B123, B12, B13, B23, B1_23, B1c, B2_13, B2c], dim=1)
    R = solve_unique_coeff_real(A12, B3_12, tol)

    idx = 0
    nB123 = B123.shape[1]; R1 = R[idx:idx+nB123, :]; idx += nB123
    nB12  = B12.shape[1];  R2 = R[idx:idx+nB12,  :]; idx += nB12
    nB13  = B13.shape[1];  R3 = R[idx:idx+nB13,  :]; idx += nB13
    nB23  = B23.shape[1];  R4 = R[idx:idx+nB23,  :]; idx += nB23
    nB1_23= B1_23.shape[1];R5 = R[idx:idx+nB1_23,:]; idx += nB1_23
    nB1c  = B1c.shape[1];  _  = R[idx:idx+nB1c,  :]; idx += nB1c
    nB2_13= B2_13.shape[1];R7 = R[idx:idx+nB2_13,:]; idx += nB2_13

    if b > 0:
        if rank_real(R5, tol) != b or rank_real(R7, tol) != b:
            raise ValueError("Unexpected: R5 or R7 not full-rank (numerical).")

    # New bases that satisfy P20 exactly in algebra (up to numerical eps)
    B3_new = B3_12.clone()
    if nB123: B3_new = B3_new - B123 @ R1
    if nB13:  B3_new = B3_new - B13  @ R3
    if nB23:  B3_new = B3_new - B23  @ R4

    if b:
        B2_new = B2_13 @ R7
        if nB12:
            B2_new = B2_new + B12 @ R2
        B1_new = B1_23 @ R5
    else:
        B2_new = B2_13.clone()
        B1_new = B1_23.clone()

    if b > 0:
        rel = float(torch.linalg.norm(B1_new + B2_new - B3_new).item()) / max(1.0, float(torch.linalg.norm(B3_new).item()))
        if rel > 1e-6:
            raise ValueError(f"P20 enforcement failed (numerical): rel_err={rel}")

    return {
        "U123": U123, "U12": U12, "U13": U13, "U23": U23,
        "U1_23": U1_23, "U2_13": U2_13, "U3_12": U3_12,
        "B1_23": B1_new, "B2_13": B2_new, "B3_12": B3_new,
    }


# ============================================================
# 3) Theorem-2 LP solver by vertex enumeration (exact Fractions)
# ============================================================
def solve_linear_system_fraction(A, b):
    """
    Solve A x = b in Fractions, return x or None.
    A is n x n.
    """
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

    # inconsistency
    for rr in range(n):
        if all(M[rr][cc] == 0 for cc in range(n)) and M[rr][n] != 0:
            return None
    if r < n:
        return None
    return [M[i][n] for i in range(n)]


def solve_lp_lambdas(m1, m2, m3, ranks_dict):
    """
    Variables x = [λ123, λ12, λ13, λ23, λ] (Fractions)

    Objective (Theorem-2 style):
      minimize f = (m1+m2+m3) - 2λ123 - λ12 - λ13 - λ23 - λ

    Constraints: (21)-(24) + nonnegativity, exactly like your GF(p) code.
    """
    cons = []

    def add_con(a, b):
        cons.append((tuple(Fraction(ai) for ai in a), Fraction(int(b))))

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
# 4) Matrix extension: kron(I_L, M) = block_diag(M,...,M)
#    (because we only need kron with identity)
# ============================================================
def kron_eye(L: int, M: torch.Tensor) -> torch.Tensor:
    """
    kron(I_L, M) as block diagonal with L copies of M.
    This avoids torch.kron issues on some builds/strides.
    """
    M = M.to(torch.float64)
    if L == 1:
        return M.clone()
    blocks = [M.clone() for _ in range(L)]
    return torch.block_diag(*blocks)


# ============================================================
# 5) Main builder: Theorem-2 + matrix extension + random mixing over R
# ============================================================
def build_lcbc_3user_code_real(
    Vp_list,
    V_list,
    seed: int = 0,
    max_tries: int = 2000,
    tol: float = None,
    verbose: bool = False,
):
    """
    Build 3-user LCBC linear code over real field R.

    Output:
      - G: encoding matrix (d*L_total) x N (real)
      - decode_Tv: list of decoding matrices T1,T2,T3 (real)
      - Vp_ext, V_ext: extended side/demand matrices (real)
      - lambdas (Fractions) and ints, slices for decoding
    """
    assert len(Vp_list) == 3 and len(V_list) == 3
    d = V_list[0].shape[0]

    # (A) redundancy removal
    Vp = []
    V = []
    for k in range(3):
        Vp_k, V_k = reduce_side_demand_real(Vp_list[k], V_list[k], tol)
        Vp.append(Vp_k)
        V.append(V_k)

    m = [V[k].shape[1] for k in range(3)]

    # (B) Lemma-2 decomposition
    U = [orth_basis(torch.cat([Vp[k], V[k]], dim=1), tol) for k in range(3)]
    dec = lemma2_decomposition_real(U[0], U[1], U[2], tol)

    U123, U12, U13, U23 = dec["U123"], dec["U12"], dec["U13"], dec["U23"]
    U1_23, U2_13, U3_12 = dec["U1_23"], dec["U2_13"], dec["U3_12"]
    B1_23, B2_13, B3_12 = dec["B1_23"], dec["B2_13"], dec["B3_12"]

    # (C) LP ranks
    ranks = {}
    for i in [1, 2, 3]:
        ranks[f"c21_{i}"] = conditional_rank_real(U123, Vp[i - 1], tol)

    ranks["c22_12_1"] = conditional_rank_real(U12, Vp[0], tol)
    ranks["c22_12_2"] = conditional_rank_real(U12, Vp[1], tol)
    ranks["c22_13_1"] = conditional_rank_real(U13, Vp[0], tol)
    ranks["c22_13_3"] = conditional_rank_real(U13, Vp[2], tol)
    ranks["c22_23_2"] = conditional_rank_real(U23, Vp[1], tol)
    ranks["c22_23_3"] = conditional_rank_real(U23, Vp[2], tol)

    ranks["c23_1"] = conditional_rank_real(orth_basis(torch.cat([U12, U13], dim=1), tol), Vp[0], tol)
    ranks["c23_2"] = conditional_rank_real(orth_basis(torch.cat([U12, U23], dim=1), tol), Vp[1], tol)
    ranks["c23_3"] = conditional_rank_real(orth_basis(torch.cat([U13, U23], dim=1), tol), Vp[2], tol)

    ranks["c24_1"] = conditional_rank_real(U1_23, Vp[0], tol)
    ranks["c24_2"] = conditional_rank_real(U2_13, Vp[1], tol)
    ranks["c24_3"] = conditional_rank_real(U3_12, Vp[2], tol)

    x_opt, f_opt = solve_lp_lambdas(m[0], m[1], m[2], ranks)
    lam123, lam12, lam13, lam23, lam = x_opt
    lambdas = {
        "lam123": lam123, "lam12": lam12, "lam13": lam13, "lam23": lam23, "lam": lam,
        "f_opt": f_opt
    }

    # (D) matrix extension factor to make lambdas integers
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
        raise ValueError("Negative t_i after scaling; inconsistent LP solution or ranks.")

    # (E) matrix extension
    Vp_ext = [kron_eye(L_scale, Vp[k]) for k in range(3)]
    V_ext  = [kron_eye(L_scale, V[k])  for k in range(3)]
    U_ext  = [orth_basis(torch.cat([Vp_ext[k], V_ext[k]], dim=1), tol) for k in range(3)]

    U123e = kron_eye(L_scale, U123)
    U12e  = kron_eye(L_scale, U12)
    U13e  = kron_eye(L_scale, U13)
    U23e  = kron_eye(L_scale, U23)
    B1e   = kron_eye(L_scale, B1_23)
    B2e   = kron_eye(L_scale, B2_13)
    B3e   = kron_eye(L_scale, B3_12)

    r123 = U123e.shape[1]
    r12  = U12e.shape[1]
    r13  = U13e.shape[1]
    r23  = U23e.shape[1]
    rb   = B1e.shape[1]

    rng = torch.Generator()
    rng.manual_seed(seed)

    # (F) random mixing over R
    for attempt in range(1, max_tries + 1):
        def randn(shape):
            return torch.randn(shape, dtype=torch.float64, generator=rng)

        # Random coefficient matrices
        N123 = randn((r123, lam123_i)) if lam123_i > 0 else torch.zeros((r123, 0), dtype=torch.float64)
        N12  = randn((r12,  lam12_i))  if lam12_i  > 0 else torch.zeros((r12,  0), dtype=torch.float64)
        N13  = randn((r13,  lam13_i))  if lam13_i  > 0 else torch.zeros((r13,  0), dtype=torch.float64)
        N23  = randn((r23,  lam23_i))  if lam23_i  > 0 else torch.zeros((r23,  0), dtype=torch.float64)

        M12  = randn((r12, lam_i)) if lam_i > 0 else torch.zeros((r12, 0), dtype=torch.float64)
        M13  = randn((r13, lam_i)) if lam_i > 0 else torch.zeros((r13, 0), dtype=torch.float64)
        M23  = randn((r23, lam_i)) if lam_i > 0 else torch.zeros((r23, 0), dtype=torch.float64)
        M    = randn((rb,  lam_i)) if lam_i > 0 else torch.zeros((rb,  0), dtype=torch.float64)

        # Broadcast blocks
        dL = d * L_scale
        G123 = U123e @ N123 if lam123_i > 0 else torch.zeros((dL, 0), dtype=torch.float64)
        G12  = U12e  @ N12  if lam12_i  > 0 else torch.zeros((dL, 0), dtype=torch.float64)
        G13  = U13e  @ N13  if lam13_i  > 0 else torch.zeros((dL, 0), dtype=torch.float64)
        G23  = U23e  @ N23  if lam23_i  > 0 else torch.zeros((dL, 0), dtype=torch.float64)

        Gl1 = (U12e @ M12) + (U13e @ M13) + (B1e @ M) if lam_i > 0 else torch.zeros((dL, 0), dtype=torch.float64)
        Gl2 = (-U12e @ M12) + (U23e @ M23) + (B2e @ M) if lam_i > 0 else torch.zeros((dL, 0), dtype=torch.float64)
        Gl3 = Gl1 + Gl2  # required by construction

        # Full column rank checks
        A1 = torch.cat([Vp_ext[0], G123, G12, G13, Gl1], dim=1)
        A2 = torch.cat([Vp_ext[1], G123, G12, G23, Gl2], dim=1)
        A3 = torch.cat([Vp_ext[2], G123, G13, G23, Gl3], dim=1)

        if rank_real(A1, tol) != A1.shape[1]:
            continue
        if rank_real(A2, tol) != A2.shape[1]:
            continue
        if rank_real(A3, tol) != A3.shape[1]:
            continue

        # 1-bird-1-stone: complete basis
        U1t = complement_basis_real(U_ext[0], A1, tol)
        U2t = complement_basis_real(U_ext[1], A2, tol)
        U3t = complement_basis_real(U_ext[2], A3, tol)

        if not (U1t.shape[1] == t1_i and U2t.shape[1] == t2_i and U3t.shape[1] == t3_i):
            continue

        # Final transmitted matrix
        blocks = {
            "G123": G123, "G12": G12, "G13": G13, "G23": G23,
            "Gl1": Gl1, "Gl2": Gl2,
            "U1t": U1t, "U2t": U2t, "U3t": U3t,
        }
        order = ["G123", "G12", "G13", "G23", "Gl1", "Gl2", "U1t", "U2t", "U3t"]
        G = torch.cat([blocks[name] for name in order], dim=1)

        # Slices for decoding
        slices = {}
        st = 0
        for name in order:
            en = st + blocks[name].shape[1]
            slices[name] = slice(st, en)
            st = en

        # Decoding matrices (fixed linear maps derived from noiseless equalities)
        A1_full = torch.cat([Vp_ext[0], G123, G12, G13, Gl1, U1t], dim=1)
        A2_full = torch.cat([Vp_ext[1], G123, G12, G23, Gl2, U2t], dim=1)
        A3_full = torch.cat([Vp_ext[2], G123, G13, G23, Gl3, U3t], dim=1)

        if rank_real(A1_full, tol) != U_ext[0].shape[1]:
            continue
        if rank_real(A2_full, tol) != U_ext[1].shape[1]:
            continue
        if rank_real(A3_full, tol) != U_ext[2].shape[1]:
            continue

        T1 = solve_unique_coeff_real(A1_full, V_ext[0], tol)
        T2 = solve_unique_coeff_real(A2_full, V_ext[1], tol)
        T3 = solve_unique_coeff_real(A3_full, V_ext[2], tol)

        # Noiseless sanity check
        Xtest = torch.randn((1, dL), dtype=torch.float64, generator=rng)
        W1 = Xtest @ V_ext[0]
        W2 = Xtest @ V_ext[1]
        W3 = Xtest @ V_ext[2]
        S = Xtest @ G

        Y1 = torch.cat([
            Xtest @ Vp_ext[0],
            S[:, slices["G123"]],
            S[:, slices["G12"]],
            S[:, slices["G13"]],
            S[:, slices["Gl1"]],
            S[:, slices["U1t"]],
        ], dim=1)
        Y2 = torch.cat([
            Xtest @ Vp_ext[1],
            S[:, slices["G123"]],
            S[:, slices["G12"]],
            S[:, slices["G23"]],
            S[:, slices["Gl2"]],
            S[:, slices["U2t"]],
        ], dim=1)
        lam3_slice = S[:, slices["Gl1"]] + S[:, slices["Gl2"]]
        Y3 = torch.cat([
            Xtest @ Vp_ext[2],
            S[:, slices["G123"]],
            S[:, slices["G13"]],
            S[:, slices["G23"]],
            lam3_slice,
            S[:, slices["U3t"]],
        ], dim=1)

        err = max(
            float(torch.linalg.norm(Y1 @ T1 - W1).item()),
            float(torch.linalg.norm(Y2 @ T2 - W2).item()),
            float(torch.linalg.norm(Y3 @ T3 - W3).item()),
        )
        if err > 1e-6:
            continue

        if verbose:
            N = G.shape[1]
            L_total = L_scale
            print(f"[OK-Real] attempts={attempt}, N={N}, L_total={L_total}, rate=N/L={N/L_total:.6f}")

        return {
            "field": "R",
            "d": d,
            "L_scale": L_scale,
            "L_total": L_scale,
            "lambdas": lambdas,
            "lambdas_int": {
                "lam123": lam123_i, "lam12": lam12_i, "lam13": lam13_i, "lam23": lam23_i,
                "lam": lam_i, "t1": t1_i, "t2": t2_i, "t3": t3_i, "L_scale": L_scale,
            },
            "G": G,
            "order": order,
            "slices": slices,
            "decode_Tv": [T1, T2, T3],
            "Vp_ext": Vp_ext,
            "V_ext": V_ext,
        }

    raise RuntimeError("Failed to build a real-field LCBC code. Increase max_tries or adjust tol.")


# ============================================================
# 6) AWGN + simulation (Gaussian X)
# ============================================================
def add_awgn_real(x: torch.Tensor, snr_db, rng: torch.Generator):
    """
    Add AWGN to x with target SNR in dB defined by:
        SNR = E[x^2] / E[n^2].
    If snr_db is None or inf -> no noise.
    """
    if snr_db is None or snr_db == float("inf"):
        return x.clone()
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    sig_pow = float((x ** 2).mean().item())
    noise_var = sig_pow / snr_lin
    n = torch.randn(x.shape, dtype=x.dtype, generator=rng) * math.sqrt(noise_var)
    return x + n


def simulate_lcbc_real(
    code,
    num_batches: int = 10000,
    snr_side_db=None,
    snr_channel_db=None,
    seed: int = 0
):
    """
    Gaussian X + side noise + channel noise.
    Decoder: fixed linear map from noiseless construction (NO MMSE, NO denoising).

    snr_side_db: None / float / (snr1,snr2,snr3)
    snr_channel_db: None / float / (snr1,snr2,snr3)

    Returns:
      mse  : list of 3 MSEs
      nmse : list of 3 NMSEs (MSE normalized by E[W^2])
    """
    d = code["d"]
    L_total = code["L_total"]
    G = code["G"]
    slices = code["slices"]
    Tv = code["decode_Tv"]
    Vp_ext = code["Vp_ext"]
    V_ext = code["V_ext"]

    rng = torch.Generator()
    rng.manual_seed(seed)

    # X ~ N(0,1)
    X = torch.randn((num_batches, d * L_total), dtype=torch.float64, generator=rng)

    S = X @ G
    Wp_true = [X @ Vp_ext[k] for k in range(3)]
    W_true = [X @ V_ext[k] for k in range(3)]

    def pick(snr, k):
        if snr is None:
            return None
        if isinstance(snr, (list, tuple)):
            return snr[k]
        return snr

    # side noise per user
    Wp_obs = [add_awgn_real(Wp_true[k], pick(snr_side_db, k), rng) for k in range(3)]
    # channel noise per user
    S_rx = [add_awgn_real(S, pick(snr_channel_db, k), rng) for k in range(3)]

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
            ], dim=1)
        elif k == 1:
            Y = torch.cat([
                Wp_obs[1],
                S_rx[1][:, slices["G123"]],
                S_rx[1][:, slices["G12"]],
                S_rx[1][:, slices["G23"]],
                S_rx[1][:, slices["Gl2"]],
                S_rx[1][:, slices["U2t"]],
            ], dim=1)
        else:
            lam3 = S_rx[2][:, slices["Gl1"]] + S_rx[2][:, slices["Gl2"]]
            Y = torch.cat([
                Wp_obs[2],
                S_rx[2][:, slices["G123"]],
                S_rx[2][:, slices["G13"]],
                S_rx[2][:, slices["G23"]],
                lam3,
                S_rx[2][:, slices["U3t"]],
            ], dim=1)

        W_hat.append(Y @ Tv[k])

    mse = []
    nmse = []
    for k in range(3):
        err = W_hat[k] - W_true[k]
        mse_k = float((err ** 2).mean().item())
        sigpow = float((W_true[k] ** 2).mean().item())
        nmse_k = mse_k / sigpow if sigpow > 0 else float("nan")
        mse.append(mse_k)
        nmse.append(nmse_k)

    return mse, nmse


# ============================================================
# 7) Demo: "Paper Example 2" structure, but over R and Gaussian X
# ============================================================
def demo_example2_real():
    # Data X^T = (A,B,C), A,B,C are Gaussian now
    e1 = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    e2 = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64)
    e3 = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float64)

    # User1: side A, wants B+C
    Vp1 = e1
    V1 = e2 + e3

    # User2: side B, wants A+C
    Vp2 = e2
    V2 = e1 + e3

    # User3: side C, wants A+2B  (real coefficient 2)
    Vp3 = e3
    V3 = e1 + 2.0 * e2

    code = build_lcbc_3user_code_real(
        [Vp1, Vp2, Vp3],
        [V1,  V2,  V3],
        seed=1,
        max_tries=5000,
        verbose=True
    )

    print("lambdas (Fractions):", code["lambdas"])
    print("lambdas (ints):", code["lambdas_int"])
    print("L_total:", code["L_total"], "G shape:", tuple(code["G"].shape))

    # Noiseless correctness check (should be ~ numerical zero)
    mse0, nmse0 = simulate_lcbc_real(code, num_batches=2000, snr_side_db=None, snr_channel_db=None, seed=0)
    print("MSE (noiseless):", mse0)
    print("NMSE (noiseless):", nmse0)

    # Add noise
    mse1, nmse1 = simulate_lcbc_real(code, num_batches=20000, snr_side_db=20.0, snr_channel_db=10.0, seed=0)
    print("MSE (side=20dB, channel=10dB):", mse1)
    print("NMSE (side=20dB, channel=10dB):", nmse1)

    # Per-user SNR example
    mse2, nmse2 = simulate_lcbc_real(
        code,
        num_batches=20000,
        snr_side_db=[25.0, 20.0, 15.0],
        snr_channel_db=[12.0, 10.0, 8.0],
        seed=0
    )
    print("MSE (per-user SNR):", mse2)
    print("NMSE (per-user SNR):", nmse2)


if __name__ == "__main__":
    demo_example2_real()
