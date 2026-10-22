# ============================================================
# According to the paper:
#   "The Capacity of Computation Broadcast" by Hua Sun, Syed A. Jafar
#
# This code implements a continuous-field (real numbers) version of the
# 2-user linear computation broadcast coding scheme (LCBC),
# mirroring Theorem 3 but over R instead of finite fields.
#
# It includes:
#   - linear algebra utilities over R
#   - LCBC scheme design (a/b/c subspace decomposition)
#   - encoding/decoding procedures
#   - end-to-end simulation over AWGN channels (broadcast + noisy side info)
#   - analytic MSE predictor (linear system + AWGN)
#
# coded by: Shuo Tan
# Expert-level numerical/engineering refinements: (this version)
# Date: 2026-01-25
# ============================================================

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np


# ============================================================
# 1) Linear algebra over the continuous field R
# ============================================================

def _stable_svd_tol(s: np.ndarray, m: int, n: int, user_tol: float) -> float:
    """
    A mixed absolute/relative threshold:
      thresh = max(user_tol, max(m,n)*eps*s[0])
    user_tol is still honored as a lower bound.
    """
    if s.size == 0:
        return user_tol
    eps = np.finfo(float).eps
    return max(float(user_tol), float(max(m, n) * eps * s[0]))


def orth(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Orthonormal basis for the column space of A (over R)."""
    A = np.array(A, dtype=float)
    if A.size == 0:
        return np.zeros((A.shape[0], 0), dtype=float)

    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if s.size == 0:
        return np.zeros((A.shape[0], 0), dtype=float)

    thresh = _stable_svd_tol(s, A.shape[0], A.shape[1], tol)
    r = int(np.sum(s > thresh))
    return U[:, :r]


def nullspace(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Basis for the right nullspace of A (over R)."""
    A = np.array(A, dtype=float)
    m, n = A.shape
    if n == 0:
        return np.zeros((n, 0), dtype=float)

    # full_matrices=True => Vh is n×n
    _, s, Vh = np.linalg.svd(A, full_matrices=True)
    thresh = _stable_svd_tol(s, m, n, tol)
    r = int(np.sum(s > thresh))
    if r >= n:
        return np.zeros((n, 0), dtype=float)
    return Vh[r:, :].T  # n×(n-r)


def intersection_basis(A: np.ndarray, B: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Basis for span(A) ∩ span(B), spans are column spaces over R.

    Method used here is the original (robust, exact-constraint) approach:
      solve [A_basis, -B_basis] [u;v] = 0
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    m = A.shape[0]

    A_basis = orth(A, tol)
    B_basis = orth(B, tol)
    ra, rb = A_basis.shape[1], B_basis.shape[1]
    if ra == 0 or rb == 0:
        return np.zeros((m, 0), dtype=float)

    C = np.concatenate([A_basis, -B_basis], axis=1)  # m×(ra+rb)
    Z = nullspace(C, tol)  # (ra+rb)×k
    if Z.shape[1] == 0:
        return np.zeros((m, 0), dtype=float)

    inter = A_basis @ Z[:ra, :]
    return orth(inter, tol)


def extend_basis(current: np.ndarray, target: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Extend 'current' to span(target) by selecting directions from target.
    Returns only the extra columns.

    IMPORTANT NUMERICAL REFINEMENT (does NOT change core logic):
      We append the *orthogonal residual direction* u = r/||r||,
      not the raw target column v. This preserves span(target) but yields
      much better conditioning (critical under AWGN).
    Assumes span(current) ⊆ span(target).
    """
    current = np.array(current, dtype=float)
    target = np.array(target, dtype=float)
    m = target.shape[0]

    Q = orth(current, tol)  # orthonormal basis for span(current)
    extras = []

    for j in range(target.shape[1]):
        v = target[:, j:j + 1]  # m×1
        r = v - Q @ (Q.T @ v) if Q.shape[1] > 0 else v
        nr = float(np.linalg.norm(r))
        if nr > tol:
            u = r / nr  # <-- key
            extras.append(u)
            Q = np.concatenate([Q, u], axis=1)

    if len(extras) == 0:
        return np.zeros((m, 0), dtype=float)
    return np.concatenate(extras, axis=1)


def solve_linear(A: np.ndarray, B: np.ndarray, rcond: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Solve A X = B (possibly consistent) via least squares; return (X, relative_residual).
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    X, _, _, _ = np.linalg.lstsq(A, B, rcond=rcond)
    rel = np.linalg.norm(A @ X - B) / (np.linalg.norm(B) + 1e-12)
    return X, float(rel)


def is_orthonormal_columns(B: np.ndarray, tol: float = 1e-8) -> float:
    """
    Return relative deviation ||B^T B - I||_F / ||I||_F for column-orthonormal test.
    """
    B = np.array(B, dtype=float)
    r = B.shape[1]
    if r == 0:
        return 0.0
    G = B.T @ B
    I = np.eye(r)
    return float(np.linalg.norm(G - I) / (np.linalg.norm(I) + 1e-12))


# ============================================================
# 2) Continuous-field LCBC scheme (real numbers), mirroring Theorem 3
# ============================================================

@dataclass
class RealLCBCScheme:
    m: int
    V1: np.ndarray
    V1p: np.ndarray
    V2: np.ndarray
    V2p: np.ndarray

    V1a: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    V1b: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    V1c: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    V2a: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    V2b: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    V2c: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    P1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    P2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Q1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Q2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    M1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    M2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    M2b: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    T1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    T2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    n1a: int = 0
    n1b: int = 0
    n1c: int = 0
    n2a: int = 0
    n2b: int = 0
    n2c: int = 0
    nA: int = 0  # max(n1a, n2a)

    # Cached broadcast matrices
    VSa: np.ndarray = field(init=False, default_factory=lambda: np.zeros((0, 0)))
    VSb: np.ndarray = field(init=False, default_factory=lambda: np.zeros((0, 0)))
    VS: np.ndarray = field(init=False, default_factory=lambda: np.zeros((0, 0)))
    nS: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.m = int(self.m)
        self.V1 = np.array(self.V1, dtype=float)
        self.V2 = np.array(self.V2, dtype=float)
        self.V1p = np.array(self.V1p, dtype=float)
        self.V2p = np.array(self.V2p, dtype=float)

        # Cache broadcast pieces per (99)
        self.VSa = self.V1p @ self.Q1 + self.V2p @ self.P2
        self.VSb = self.V2b @ self.M2b + self.V2p @ self.M2
        self.VS = np.concatenate([self.VSa, self.VSb, self.V1c, self.V2c], axis=1)
        self.nS = int(self.VS.shape[1])

        # Basic dimension checks (fail fast)
        if self.V1.shape[0] != self.m or self.V2.shape[0] != self.m:
            raise ValueError("Row dimension mismatch among V matrices.")
        if self.VSa.shape[1] != self.nA:
            raise ValueError("VSa width must equal nA.")
        if self.VSb.shape[1] != self.n1b:
            raise ValueError("VSb width must equal n1b (nb).")
        if self.V1c.shape[1] != self.n1c or self.V2c.shape[1] != self.n2c:
            raise ValueError("V1c/V2c dimension mismatch.")
        if self.n1b != self.n2b:
            raise ValueError("Expected n1b == n2b (nb).")
        if self.T1.shape[0] != (self.n1a + self.n1b + self.n1c):
            raise ValueError("T1 row dimension mismatch with [V1a,V1b,V1c].")
        if self.T2.shape[0] != (self.n2a + self.n2b + self.n2c):
            raise ValueError("T2 row dimension mismatch with [V2a,V2b,V2c].")

    def broadcast_matrix(self) -> np.ndarray:
        """
        Return V_S such that S = X^T V_S (over R), consistent with (99).
        """
        return self.VS

    def encode(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Encode according to (99), but over R (no mod, no quantization)."""
        X = np.array(X, dtype=float).reshape(self.m, 1)

        W1 = (X.T @ self.V1).reshape(-1)
        W2 = (X.T @ self.V2).reshape(-1)
        W1p = (X.T @ self.V1p).reshape(-1)
        W2p = (X.T @ self.V2p).reshape(-1)

        Sa = (X.T @ self.VSa).reshape(-1)
        Sb = (X.T @ self.VSb).reshape(-1)
        Sc1 = (X.T @ self.V1c).reshape(-1)
        Sc2 = (X.T @ self.V2c).reshape(-1)

        S = np.concatenate([Sa, Sb, Sc1, Sc2])

        return {
            "X": X.reshape(-1),
            "W1": W1,
            "W2": W2,
            "W1p": W1p,
            "W2p": W2p,
            "Sa": Sa,
            "Sb": Sb,
            "Sc1": Sc1,
            "Sc2": Sc2,
            "S": S,
        }

    def decode_user1(self, Y1: np.ndarray, W1p_obs: np.ndarray) -> np.ndarray:
        """User 1 decoder: (105)(106)(109) + reconstruction with T1."""
        Y1 = np.array(Y1, dtype=float).reshape(1, -1)
        W1p_obs = np.array(W1p_obs, dtype=float).reshape(1, -1)

        idx = 0
        Sa_hat = Y1[:, idx: idx + self.nA]
        idx += self.nA
        Sb_hat = Y1[:, idx: idx + self.n1b]
        idx += self.n1b
        Sc1_hat = Y1[:, idx: idx + self.n1c]

        # (105)(106): pad then truncate
        W1a_pad_hat = Sa_hat - (W1p_obs @ self.Q1) + (W1p_obs @ self.P1)
        W1a_hat = W1a_pad_hat[:, : self.n1a]

        W1b_hat = Sb_hat + (W1p_obs @ self.M1)
        W1c_hat = Sc1_hat

        basis_hat = np.concatenate([W1a_hat, W1b_hat, W1c_hat], axis=1)
        return (basis_hat @ self.T1).reshape(-1)

    def decode_user2(self, Y2: np.ndarray, W2p_obs: np.ndarray) -> np.ndarray:
        """User 2 decoder: (107)(108)(109) + reconstruction with T2."""
        Y2 = np.array(Y2, dtype=float).reshape(1, -1)
        W2p_obs = np.array(W2p_obs, dtype=float).reshape(1, -1)

        idx = 0
        Sa_hat = Y2[:, idx: idx + self.nA]
        idx += self.nA
        Sb_hat = Y2[:, idx: idx + self.n1b]
        idx += self.n1b
        idx += self.n1c  # skip Sc1
        Sc2_hat = Y2[:, idx: idx + self.n2c]

        W2a_pad_hat = Sa_hat - (W2p_obs @ self.P2) + (W2p_obs @ self.Q2)
        W2a_hat = W2a_pad_hat[:, : self.n2a]

        if self.n2b == 0:
            W2b_hat = np.zeros((1, 0), dtype=float)
        else:
            tmp = Sb_hat - (W2p_obs @ self.M2)
            # More stable than tmp @ inv(M2b):
            # Solve: M2b^T x^T = tmp^T  => x = tmp @ inv(M2b)
            W2b_hat = np.linalg.solve(self.M2b.T, tmp.T).T

        W2c_hat = Sc2_hat

        basis_hat = np.concatenate([W2a_hat, W2b_hat, W2c_hat], axis=1)
        return (basis_hat @ self.T2).reshape(-1)


def design_lcbc_real(
    V1: np.ndarray, V1p: np.ndarray, V2: np.ndarray, V2p: np.ndarray,
    tol: float = 1e-10,
    check_residual: bool = True,
) -> RealLCBCScheme:
    """
    Continuous-field (real) version of Theorem 3 linear construction:
      - same a/b/c subspace decomposition ((61)-(63))
      - same linear relations (87),(97)-(99),(105)-(109)
      - all interpreted over R.
    """
    V1 = np.array(V1, dtype=float)
    V2 = np.array(V2, dtype=float)
    V1p = np.array(V1p, dtype=float)
    V2p = np.array(V2p, dtype=float)

    m = V1.shape[0]
    if not (V2.shape[0] == m and V1p.shape[0] == m and V2p.shape[0] == m):
        raise ValueError("All V matrices must share the same row dimension m.")

    # Bases for spans
    V1_basis = orth(V1, tol)
    V2_basis = orth(V2, tol)
    side_basis = orth(np.concatenate([V1p, V2p], axis=1), tol)

    # (61)
    V1a = intersection_basis(V1_basis, side_basis, tol)
    V2a = intersection_basis(V2_basis, side_basis, tol)

    # (62)
    V1ab = intersection_basis(V1_basis, orth(np.concatenate([V1p, V2p, V2_basis], axis=1), tol), tol)
    V2ab = intersection_basis(V2_basis, orth(np.concatenate([V1p, V2p, V1_basis], axis=1), tol), tol)

    # Steinitz extension with numerically-healthy basis choices
    V1b = extend_basis(V1a, V1ab, tol)
    V2b = extend_basis(V2a, V2ab, tol)

    V1c = extend_basis(np.concatenate([V1a, V1b], axis=1), V1_basis, tol)
    V2c = extend_basis(np.concatenate([V2a, V2b], axis=1), V2_basis, tol)

    n1a, n1b, n1c = V1a.shape[1], V1b.shape[1], V1c.shape[1]
    n2a, n2b, n2c = V2a.shape[1], V2b.shape[1], V2c.shape[1]
    if n1b != n2b:
        raise ValueError(f"Expected n1b==n2b (from proof), got {n1b} vs {n2b}.")
    nb = n1b
    nA = max(n1a, n2a)

    # Reconstruction matrices T1,T2: solve B T = V
    B1 = np.concatenate([V1a, V1b, V1c], axis=1)  # m×r1
    B2 = np.concatenate([V2a, V2b, V2c], axis=1)  # m×r2

    ortho_err1 = is_orthonormal_columns(B1, tol=1e-8)
    ortho_err2 = is_orthonormal_columns(B2, tol=1e-8)

    if ortho_err1 < 1e-8 and B1.shape[1] > 0:
        T1 = B1.T @ V1
        rel1 = float(np.linalg.norm(B1 @ T1 - V1) / (np.linalg.norm(V1) + 1e-12))
    else:
        T1, rel1 = solve_linear(B1, V1)

    if ortho_err2 < 1e-8 and B2.shape[1] > 0:
        T2 = B2.T @ V2
        rel2 = float(np.linalg.norm(B2 @ T2 - V2) / (np.linalg.norm(V2) + 1e-12))
    else:
        T2, rel2 = solve_linear(B2, V2)

    # (97)(98): solve [V1p V2p] * [P1;P2] = V1a_pad ; similarly for Q
    V1a_pad = np.concatenate([V1a, np.zeros((m, nA - n1a))], axis=1)
    V2a_pad = np.concatenate([V2a, np.zeros((m, nA - n2a))], axis=1)
    side_concat = np.concatenate([V1p, V2p], axis=1)

    PQ, relPQ = solve_linear(side_concat, V1a_pad)
    QQ, relQQ = solve_linear(side_concat, V2a_pad)

    n1p = V1p.shape[1]
    n2p = V2p.shape[1]
    P1 = PQ[:n1p, :]
    P2 = PQ[n1p:, :]
    Q1 = QQ[:n1p, :]
    Q2 = QQ[n1p:, :]

    # (87): solve [V1p V2p V2b] [M1; M2; M2b] = V1b
    if nb == 0:
        M1 = np.zeros((n1p, 0))
        M2 = np.zeros((n2p, 0))
        M2b = np.zeros((0, 0))
        relb = 0.0
    else:
        A_b = np.concatenate([V1p, V2p, V2b], axis=1)
        coeff, relb = solve_linear(A_b, V1b)
        M1 = coeff[:n1p, :]
        M2 = coeff[n1p:n1p + n2p, :]
        M2b = coeff[n1p + n2p:, :]

        r = np.linalg.matrix_rank(M2b, tol=tol)
        if r != nb:
            raise ValueError(f"M2b not full rank: rank={r}, nb={nb}. Try adjusting tol.")

    if check_residual:
        worst = max(rel1, rel2, relPQ, relQQ, relb)
        if worst > 1e-7:
            print("[warn] design residuals:",
                  dict(rel_T1=rel1, rel_T2=rel2, rel_PQ=relPQ, rel_QQ=relQQ, rel_b=relb,
                       ortho_err1=ortho_err1, ortho_err2=ortho_err2))

    return RealLCBCScheme(
        m=m, V1=V1, V1p=V1p, V2=V2, V2p=V2p,
        V1a=V1a, V1b=V1b, V1c=V1c,
        V2a=V2a, V2b=V2b, V2c=V2c,
        P1=P1, P2=P2, Q1=Q1, Q2=Q2,
        M1=M1, M2=M2, M2b=M2b,
        T1=T1, T2=T2,
        n1a=n1a, n1b=n1b, n1c=n1c,
        n2a=n2a, n2b=n2b, n2c=n2c,
        nA=nA
    )


# ============================================================
# 3) AWGN channels with adjustable SNRs (broadcast + side-info)
# ============================================================

def avg_power_from_matrix(V: np.ndarray) -> float:
    """
    For X~N(0,I), if Z = X^T V (1×n), then E[mean(Z^2)] = ||V||_F^2 / n.
    """
    V = np.array(V, dtype=float)
    n = V.shape[1]
    if n == 0:
        return 0.0
    return float(np.sum(V * V) / n)


def add_awgn_batch(
    x: np.ndarray,
    snr_db: Optional[float],
    rng: np.random.Generator,
    ref_power: Optional[float] = None,
) -> np.ndarray:
    """
    Add AWGN with target SNR in dB:
        SNR = ref_power / noise_var

    If ref_power is None:
      - for 1D x: use empirical mean(x^2)
      - for 2D x: use per-row empirical mean(x^2) (keeps old semantics when use_avg_power=False)

    If snr_db is inf or None, returns x unchanged.
    """
    x = np.array(x, dtype=float)
    if snr_db is None or np.isinf(snr_db):
        return x.copy()

    snr_lin = 10.0 ** (float(snr_db) / 10.0)

    if ref_power is not None:
        p = float(ref_power)
        noise_var = (p / snr_lin) if p > 0 else (1.0 / snr_lin)
        noise = rng.normal(0.0, math.sqrt(noise_var), size=x.shape)
        return x + noise

    # empirical power
    if x.ndim == 1:
        p = float(np.mean(x * x))
        noise_var = (p / snr_lin) if p > 0 else (1.0 / snr_lin)
        noise = rng.normal(0.0, math.sqrt(noise_var), size=x.shape)
        return x + noise

    # x is 2D: per-row power
    p = np.mean(x * x, axis=1, keepdims=True)
    noise_var = np.where(p > 0, p / snr_lin, 1.0 / snr_lin)
    noise = rng.normal(0.0, 1.0, size=x.shape) * np.sqrt(noise_var)
    return x + noise


def simulate_real_lcbc(
    scheme: RealLCBCScheme,
    snr_ch_db: Union[Optional[float], Tuple[Optional[float], Optional[float]]],
    snr_si1_db: Optional[float],
    snr_si2_db: Optional[float],
    num_trials: int = 2000,
    seed: int = 0,
    use_avg_power: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end simulation (pure analog uncoded AWGN), vectorized.

    If use_avg_power=True:
      noise variance is fixed based on theoretical average power:
        E[mean(S^2)] = ||V_S||_F^2 / nS  (Gaussian X)

    If use_avg_power=False:
      noise variance is set per trial using empirical mean(x^2) per sample (row).
    """
    rng = np.random.default_rng(seed)

    if isinstance(snr_ch_db, tuple):
        snr_ch1_db, snr_ch2_db = snr_ch_db
    else:
        snr_ch1_db, snr_ch2_db = snr_ch_db, snr_ch_db

    # reference powers for fixed-variance AWGN
    if use_avg_power:
        pS = avg_power_from_matrix(scheme.broadcast_matrix())
        pW1p = avg_power_from_matrix(scheme.V1p)
        pW2p = avg_power_from_matrix(scheme.V2p)
    else:
        pS = pW1p = pW2p = None

    # Generate all trials at once
    X = rng.normal(0.0, 1.0, size=(num_trials, scheme.m))

    # Core signals
    S = X @ scheme.VS
    W1 = X @ scheme.V1
    W2 = X @ scheme.V2
    W1p = X @ scheme.V1p
    W2p = X @ scheme.V2p

    # Add noises
    Y1 = add_awgn_batch(S, snr_ch1_db, rng, ref_power=pS)
    Y2 = add_awgn_batch(S, snr_ch2_db, rng, ref_power=pS)

    W1p_obs = add_awgn_batch(W1p, snr_si1_db, rng, ref_power=pW1p)
    W2p_obs = add_awgn_batch(W2p, snr_si2_db, rng, ref_power=pW2p)

    # Vectorized decoding (same logic as decode_user1/decode_user2)
    nA = scheme.nA
    nb = scheme.n1b
    n1c = scheme.n1c
    n2c = scheme.n2c

    # ---- User 1 ----
    Sa1 = Y1[:, 0:nA]
    Sb1 = Y1[:, nA:nA + nb]
    Sc1 = Y1[:, nA + nb:nA + nb + n1c]

    W1a_pad_hat = Sa1 - (W1p_obs @ scheme.Q1) + (W1p_obs @ scheme.P1)
    W1a_hat = W1a_pad_hat[:, :scheme.n1a]
    W1b_hat = Sb1 + (W1p_obs @ scheme.M1)
    W1c_hat = Sc1
    basis1 = np.concatenate([W1a_hat, W1b_hat, W1c_hat], axis=1)
    W1_hat = basis1 @ scheme.T1

    # ---- User 2 ----
    Sa2 = Y2[:, 0:nA]
    Sb2 = Y2[:, nA:nA + nb]
    Sc2 = Y2[:, nA + nb + n1c:nA + nb + n1c + n2c]

    W2a_pad_hat = Sa2 - (W2p_obs @ scheme.P2) + (W2p_obs @ scheme.Q2)
    W2a_hat = W2a_pad_hat[:, :scheme.n2a]

    if scheme.n2b == 0:
        W2b_hat = np.zeros((num_trials, 0), dtype=float)
    else:
        tmp = Sb2 - (W2p_obs @ scheme.M2)
        # Solve row-wise: tmp @ inv(M2b) via transposed solve
        # (np.linalg.solve handles multiple RHS columns, so do transpose trick)
        W2b_hat = np.linalg.solve(scheme.M2b.T, tmp.T).T

    W2c_hat = Sc2
    basis2 = np.concatenate([W2a_hat, W2b_hat, W2c_hat], axis=1)
    W2_hat = basis2 @ scheme.T2

    # MSEs
    mse_S1 = float(np.mean((Y1 - S) ** 2))
    mse_S2 = float(np.mean((Y2 - S) ** 2))
    mse_W1 = float(np.mean((W1_hat - W1) ** 2))
    mse_W2 = float(np.mean((W2_hat - W2) ** 2))
    mse_W1p = float(np.mean((W1p_obs - W1p) ** 2))
    mse_W2p = float(np.mean((W2p_obs - W2p) ** 2))

    return {
        "snr_ch1_db": snr_ch1_db,
        "snr_ch2_db": snr_ch2_db,
        "snr_si1_db": snr_si1_db,
        "snr_si2_db": snr_si2_db,
        "num_trials": num_trials,
        "use_avg_power": use_avg_power,
        "MSE_S1": mse_S1,
        "MSE_S2": mse_S2,
        "MSE_W1": mse_W1,
        "MSE_W2": mse_W2,
        "MSE_W1p": mse_W1p,
        "MSE_W2p": mse_W2p,
    }


# ============================================================
# 3.5) Analytic MSE predictor (linear decode + AWGN)
# ============================================================

def _snr_to_noise_var(ref_power: float, snr_db: Optional[float]) -> float:
    if snr_db is None or np.isinf(snr_db):
        return 0.0
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    if ref_power <= 0:
        return 1.0 / snr_lin
    return ref_power / snr_lin


def predict_mse_real_lcbc(
    scheme: RealLCBCScheme,
    snr_ch_db: Union[Optional[float], Tuple[Optional[float], Optional[float]]],
    snr_si1_db: Optional[float],
    snr_si2_db: Optional[float],
) -> Dict[str, Any]:
    """
    Analytic MSE prediction (no Monte Carlo), valid because:
      - encoding is linear
      - decoding is linear
      - noiseless decode is exact
      - noise is i.i.d AWGN on broadcast and side info

    We compute linear maps:
      e1 = Nch1 @ G1y + Nsi1 @ G1si
      e2 = Nch2 @ G2y + Nsi2 @ G2si
    then
      MSE(W1) = (σch1^2||G1y||_F^2 + σsi1^2||G1si||_F^2) / nW1
      MSE(W2) = (σch2^2||G2y||_F^2 + σsi2^2||G2si||_F^2) / nW2
    """
    if isinstance(snr_ch_db, tuple):
        snr_ch1_db, snr_ch2_db = snr_ch_db
    else:
        snr_ch1_db, snr_ch2_db = snr_ch_db, snr_ch_db

    pS = avg_power_from_matrix(scheme.broadcast_matrix())
    pW1p = avg_power_from_matrix(scheme.V1p)
    pW2p = avg_power_from_matrix(scheme.V2p)

    sig2_ch1 = _snr_to_noise_var(pS, snr_ch1_db)
    sig2_ch2 = _snr_to_noise_var(pS, snr_ch2_db)
    sig2_si1 = _snr_to_noise_var(pW1p, snr_si1_db)
    sig2_si2 = _snr_to_noise_var(pW2p, snr_si2_db)

    nA = scheme.nA
    nb = scheme.n1b
    n1c = scheme.n1c
    n2c = scheme.n2c
    nS = scheme.nS

    nW1 = scheme.V1.shape[1]
    nW2 = scheme.V2.shape[1]

    # Partition T1
    T1a = scheme.T1[:scheme.n1a, :]                    # n1a×nW1
    T1b = scheme.T1[scheme.n1a:scheme.n1a + nb, :]     # nb×nW1
    T1c = scheme.T1[scheme.n1a + nb:, :]               # n1c×nW1

    # Channel map for user1: Nch1 (1×nS) -> e1 (1×nW1)
    G1y = np.zeros((nS, nW1), dtype=float)
    if scheme.n1a > 0:
        G1y[0:scheme.n1a, :] = T1a
    if nb > 0:
        G1y[nA:nA + nb, :] = T1b
    if n1c > 0:
        G1y[nA + nb:nA + nb + n1c, :] = T1c
    # Sc2 rows unused => already zeros

    # Side-info map for user1: Nsi1 (1×n1p) -> e1
    G1si = np.zeros((scheme.V1p.shape[1], nW1), dtype=float)
    if scheme.n1a > 0:
        G1si += (scheme.P1 - scheme.Q1)[:, :scheme.n1a] @ T1a
    if nb > 0:
        G1si += scheme.M1 @ T1b

    # Partition T2
    T2a = scheme.T2[:scheme.n2a, :]                    # n2a×nW2
    T2b = scheme.T2[scheme.n2a:scheme.n2a + nb, :]     # nb×nW2
    T2c = scheme.T2[scheme.n2a + nb:, :]               # n2c×nW2

    # Precompute L = inv(M2b) @ T2b stably via solve
    if nb > 0:
        L = np.linalg.solve(scheme.M2b, T2b)  # nb×nW2
    else:
        L = np.zeros((0, nW2), dtype=float)

    # Channel map for user2
    G2y = np.zeros((nS, nW2), dtype=float)
    if scheme.n2a > 0:
        G2y[0:scheme.n2a, :] = T2a
    if nb > 0:
        G2y[nA:nA + nb, :] = L
    if n2c > 0:
        start = nA + nb + n1c
        G2y[start:start + n2c, :] = T2c

    # Side-info map for user2
    G2si = np.zeros((scheme.V2p.shape[1], nW2), dtype=float)
    if scheme.n2a > 0:
        G2si += (scheme.Q2 - scheme.P2)[:, :scheme.n2a] @ T2a
    if nb > 0:
        G2si += (-scheme.M2) @ L

    mseW1 = (sig2_ch1 * (np.linalg.norm(G1y, "fro") ** 2) +
             sig2_si1 * (np.linalg.norm(G1si, "fro") ** 2)) / max(nW1, 1)

    mseW2 = (sig2_ch2 * (np.linalg.norm(G2y, "fro") ** 2) +
             sig2_si2 * (np.linalg.norm(G2si, "fro") ** 2)) / max(nW2, 1)

    # Broadcast observation error is just noise power per component
    mseS1 = sig2_ch1
    mseS2 = sig2_ch2

    return {
        "snr_ch1_db": snr_ch1_db,
        "snr_ch2_db": snr_ch2_db,
        "snr_si1_db": snr_si1_db,
        "snr_si2_db": snr_si2_db,
        "use_avg_power": True,
        "Pred_MSE_S1": float(mseS1),
        "Pred_MSE_S2": float(mseS2),
        "Pred_MSE_W1": float(mseW1),
        "Pred_MSE_W2": float(mseW2),
    }


# ============================================================
# 4) Example instance (paper Sec. 3.2 coefficients, but over R)
# ============================================================

def build_paper_section_3_2_example_real() -> RealLCBCScheme:
    """
    Use the coefficient matrices in the paper's Section 3.2 example,
    but interpret them over R (no modulo arithmetic).

      m = 7, X = [x1..x7]^T, xi ~ N(0,1)
      W1' = [x1, x3],  W2' = [x2, x4]
      W1  = [x1+2x2, x3+x5, x1+x4+x6, x7]
      W2  = [2x1+x2, x5, x2+x4+2x6]
    """
    m = 7
    I = np.eye(m)
    e = [I[:, i:i + 1] for i in range(m)]

    V1p = np.concatenate([e[0], e[2]], axis=1)
    V2p = np.concatenate([e[1], e[3]], axis=1)

    V1 = np.concatenate([
        e[0] + 2 * e[1],
        e[2] + e[4],
        e[0] + e[3] + e[5],
        e[6]
    ], axis=1)

    V2 = np.concatenate([
        2 * e[0] + e[1],
        e[4],
        e[1] + e[3] + 2 * e[5]
    ], axis=1)

    return design_lcbc_real(V1, V1p, V2, V2p, tol=1e-10)


# ============================================================
# 5) "反复检验" (repeat checks)
# ============================================================

def self_test() -> None:
    scheme = build_paper_section_3_2_example_real()
    rng = np.random.default_rng(0)

    # (A) Noiseless exactness
    for _ in range(300):
        X = rng.normal(0.0, 1.0, size=(scheme.m,))
        comp = scheme.encode(X)
        W1_hat = scheme.decode_user1(comp["S"], comp["W1p"])
        W2_hat = scheme.decode_user2(comp["S"], comp["W2p"])
        if not np.allclose(W1_hat, comp["W1"], atol=1e-10, rtol=1e-10):
            raise AssertionError("Noiseless decode failed for User 1.")
        if not np.allclose(W2_hat, comp["W2"], atol=1e-10, rtol=1e-10):
            raise AssertionError("Noiseless decode failed for User 2.")

    # (B) Infinite SNR => numerical MSE ~ 0
    res = simulate_real_lcbc(
        scheme,
        snr_ch_db=float("inf"),
        snr_si1_db=float("inf"),
        snr_si2_db=float("inf"),
        num_trials=500,
        seed=1,
        use_avg_power=True
    )
    if res["MSE_W1"] > 1e-20 or res["MSE_W2"] > 1e-20:
        raise AssertionError("Infinite-SNR sanity check failed.")

    print("[self_test] PASS: noiseless exactness + infinite-SNR checks.")


# ============================================================
# 6) Demo run
# ============================================================

if __name__ == "__main__":
    self_test()

    scheme = build_paper_section_3_2_example_real()
    print("Dimensions:",
          dict(n1a=scheme.n1a, n1b=scheme.n1b, n1c=scheme.n1c,
               n2a=scheme.n2a, n2b=scheme.n2b, n2c=scheme.n2c, nA=scheme.nA,
               nS=scheme.nS))

    # Optional: show predicted vs simulated at a few SNR points
    print("\n--- Compare Analytic Prediction vs Monte Carlo (use_avg_power=True) ---")
    for snr in [40, 20, 10, 0]:
        pred = predict_mse_real_lcbc(
            scheme,
            snr_ch_db=snr,
            snr_si1_db=snr,
            snr_si2_db=snr
        )
        sim = simulate_real_lcbc(
            scheme,
            snr_ch_db=snr,
            snr_si1_db=snr,
            snr_si2_db=snr,
            num_trials=20000,
            seed=0,
            use_avg_power=True
        )
        print(
            f"SNR={snr:>3} dB | "
            f"Pred MSE(W1)={pred['Pred_MSE_W1']:.6g}, Sim={sim['MSE_W1']:.6g} | "
            f"Pred MSE(W2)={pred['Pred_MSE_W2']:.6g}, Sim={sim['MSE_W2']:.6g}"
        )

    print("\n--- Sweep SNR (use_avg_power=True) ---")
    for snr in [float("inf"), 40, 30, 20, 10, 5, 0]:
        res = simulate_real_lcbc(
            scheme,
            snr_ch_db=snr,      # scalar => both users same channel SNR (independent noises)
            snr_si1_db=snr,     # side-info SNR at user 1
            snr_si2_db=snr,     # side-info SNR at user 2
            num_trials=50000,
            seed=0,
            use_avg_power=True
        )
        print(
            f"SNR={snr:>5} dB | "
            f"MSE(W1)={res['MSE_W1']:.6g}, MSE(W2)={res['MSE_W2']:.6g}, "
            f"MSE(S1)={res['MSE_S1']:.6g}, MSE(S2)={res['MSE_S2']:.6g}, "
            f"MSE(W1')={res['MSE_W1p']:.6g}, MSE(W2')={res['MSE_W2p']:.6g}"
        )
