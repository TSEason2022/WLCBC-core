#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Over-the-Air LCBC WSMSE Solver (AO/BCD) - Optimized Implementation
- PyTorch implementation (CPU/GPU)
- SVD-based side-information reduction (stores U,Sigma,(optional)T)
- Per-user channel-noise and side-noise scaling (whitening)
- LMMSE decoders via low-rank Pi without forming n x n
- Precoder update via eigen-decomposition + safeguarded Newton (faster than bisection)
- Saves/plots WSMSE(MSE) descent curve

Noise model (per user i):
  y_i = H_i s + n_i,          n_i ~ N(0, sigma_ch_i^2 I)
  w'_i = V'_i^T x + n'_i,     n'_i ~ N(0, sigma_side_i^2 I)

We whiten observations (invertible scaling, no loss of optimality):
  ytilde_i = y_i / sigma_ch_i  -> effective channel H_i_eff = H_i / sigma_ch_i
  wtilde_i = w'_i / sigma_side_i -> effective side matrix V'_i_eff = V'_i / sigma_side_i
Then run the same canonical algorithm with unit noise.

Key implementation optimizations (without changing the core AO logic):
  1) Avoid forming n×d residuals in decoder update:
       U^T(V - R^T Gy) = (U^T V) - (R U)^T Gy
     so we precompute UtV and reuse RU in each iteration.
  2) Avoid storing PiV (n×d): compute RHS = R V - (R U diag(lambda) UtV) on-the-fly.
  3) Optional: align final decoders with final precoder E before returning outputs.
  4) Faster/stabler SPD solves using cholesky_ex (no Python exceptions).

Author: (Shuo Tan)
Edits/Optimizations: (assistant)
"""

from __future__ import annotations

import sys
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch


# =========================
# Utilities
# =========================
def fro_norm_sq(X: torch.Tensor) -> torch.Tensor:
    return torch.sum(X * X)


def qr_orthonormal_columns(X: torch.Tensor) -> torch.Tensor:
    """
    Return Q with orthonormal columns from QR (reduced).
    """
    if X.numel() == 0:
        return X
    Q, _ = torch.linalg.qr(X, mode="reduced")
    return Q


def _running_in_notebook() -> bool:
    return ("ipykernel" in sys.modules) or ("google.colab" in sys.modules)


def _parse_csv_floats(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def _broadcast_param_list(val_list: List[float], K: int, default_scalar: float) -> List[float]:
    """
    If val_list provided (len>0), must match K.
    Else broadcast default_scalar to length K.
    """
    if len(val_list) == 0:
        return [float(default_scalar) for _ in range(K)]
    if len(val_list) != K:
        raise ValueError(f"List length mismatch: got {len(val_list)} but K={K}.")
    return [float(x) for x in val_list]


def safe_cholesky_solve(
    A: torch.Tensor,
    B: torch.Tensor,
    jitter: float = 1e-12,
    max_tries: int = 3,
) -> torch.Tensor:
    """
    Solve AX=B for SPD A using Cholesky, with diagonal jitter for stability.

    Uses torch.linalg.cholesky_ex (no Python exception overhead). If Cholesky
    fails after max_tries, falls back to torch.linalg.solve.

    NOTE: A is assumed to be (approximately) symmetric.
    """
    if A.numel() == 0:
        return torch.zeros_like(B)

    # Try a few jitter scales
    jit = float(jitter)
    for _ in range(max_tries):
        Aj = A.clone()
        if jit > 0.0:
            Aj.diagonal().add_(jit)
        L, info = torch.linalg.cholesky_ex(Aj, check_errors=False)
        if int(info.item()) == 0:
            return torch.cholesky_solve(B, L)
        jit *= 10.0

    return torch.linalg.solve(A, B)


# =========================
# Data structures
# =========================
@dataclass
class SideInfoReduced:
    """
    Reduced side information model after whitening and projection:
      Vprime_eff = (Vprime_raw / sigma_side) = U Sigma T^T

      reduced side measurement (conceptually):
        w_red = T^T (w'_orig / sigma_side) = Sigma U^T x + n, n~N(0,I)

    Stores:
      U:      (n, d_side)
      sigma:  (d_side,)
      T:      (dprime_raw, d_side)  (semi-orthonormal) or None
      lambda_: sigma^2/(1+sigma^2)
      coeff:  sigma/(1+sigma^2)
      UtV:    (d_side, d) = U^T V   (precomputed to avoid n×d residuals)
      U_sigma: optional (n, d_side) = U * sigma, for faster VprimeGw (memory-speed tradeoff)
    """
    U: torch.Tensor
    sigma: torch.Tensor
    T: Optional[torch.Tensor]
    lambda_: torch.Tensor
    coeff: torch.Tensor
    UtV: torch.Tensor
    U_sigma: Optional[torch.Tensor] = None


@dataclass
class User:
    """
    One user i:
      H_raw: (m, r)   unwhitened
      V:     (n, d)
      Vprime_raw: (n, dprime_raw)  unwhitened
      channel_noise_std: sigma_ch_i
      side_noise_std:    sigma_side_i
      weight: omega_i

    During preprocess:
      H will be set to H_eff = H_raw / sigma_ch_i (unit-noise equivalent)
      side will be set from SVD(Vprime_raw / sigma_side_i)
    """
    H_raw: torch.Tensor
    V: torch.Tensor
    Vprime_raw: torch.Tensor
    weight: float = 1.0
    channel_noise_std: float = 1.0
    side_noise_std: float = 1.0

    # filled by preprocess
    H: Optional[torch.Tensor] = None
    side: Optional[SideInfoReduced] = None

    @property
    def m(self) -> int:
        return int(self.H_raw.shape[0])

    @property
    def r(self) -> int:
        return int(self.H_raw.shape[1])

    @property
    def d(self) -> int:
        return int(self.V.shape[1])

    @property
    def dprime_raw(self) -> int:
        return int(self.Vprime_raw.shape[1])

    @property
    def d_side(self) -> int:
        if self.side is None:
            return 0
        return int(self.side.sigma.numel())


@dataclass
class Decoder:
    """
    Decoder matrices for whitened observations:
      Gy: (m, d) applied to y_tilde = y / sigma_ch
      Gw: (d_side, d) applied to w_red = T^T (w'/sigma_side)

    If you want a decoder acting on original (unwhitened) y and w' directly:
      Gy_full_on_y = Gy / sigma_ch
      Gw_full_on_wprime = (T @ Gw) / sigma_side   (dimension: dprime_raw x d)
    """
    Gy: torch.Tensor
    Gw: torch.Tensor


@dataclass
class SolverConfig:
    # AO loop
    max_iters: int = 200
    rel_tol: float = 1e-6
    power: float = 1.0

    # numerical
    dtype: torch.dtype = torch.float64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # side info SVD
    svd_tol_rel: float = 1e-10
    svd_on_cpu: bool = True
    store_side_T: bool = True          # store T for mapping back to original w' dimension
    precompute_u_sigma: bool = False   # memory-speed tradeoff

    # decoder solve
    chol_jitter: float = 1e-12

    # mu solver (safeguarded Newton)
    mu_tol_rel: float = 1e-12
    newton_max_iters: int = 50
    eig_tol: float = 1e-14

    # logging/plot
    verbose: bool = True
    track_per_user: bool = False  # if True, track per-user MSE curves (extra cost)

    # output alignment
    align_final_decoders: bool = True  # recompute decoders for final E before returning


# =========================
# Preprocessing: whitening + side info SVD reduction
# =========================
@torch.no_grad()
def preprocess_side_info(users: List[User], cfg: SolverConfig) -> None:
    """
    For each user:
      1) move tensors to cfg.device/cfg.dtype
      2) whiten noise by scaling:
           H_eff = H_raw / sigma_ch
           Vprime_eff = Vprime_raw / sigma_side
      3) thin SVD on Vprime_eff
      4) compute lambda, coeff
      5) precompute UtV = U^T V (key for fast Gw update and PiV action)
      6) optional: precompute U_sigma = U * sigma (for faster VprimeGw)
    """
    dev_main = torch.device(cfg.device)
    dtype = cfg.dtype

    for idx, u in enumerate(users):
        if u.channel_noise_std <= 0.0 or u.side_noise_std <= 0.0:
            raise ValueError(f"User {idx}: noise std must be positive.")

        # Move raw matrices to main device/dtype
        u.H_raw = u.H_raw.to(dev_main, dtype=dtype)
        u.V = u.V.to(dev_main, dtype=dtype)
        u.Vprime_raw = u.Vprime_raw.to(dev_main, dtype=dtype)

        # Whiten channel noise (effective H used in solver)
        u.H = u.H_raw / float(u.channel_noise_std)

        # Whiten side noise (effective Vprime for SVD)
        Vp_eff = u.Vprime_raw / float(u.side_noise_std)

        # SVD possibly on CPU for stability/efficiency
        if cfg.svd_on_cpu:
            Vp_svd = Vp_eff.detach().to("cpu", dtype=torch.float64)
        else:
            Vp_svd = Vp_eff.detach().to(dev_main, dtype=dtype)

        # thin SVD: Vp = U S Vh
        Umat, Svals, Vh = torch.linalg.svd(Vp_svd, full_matrices=False)

        if Svals.numel() == 0:
            r_side = 0
        else:
            smax = torch.max(Svals)
            thr = cfg.svd_tol_rel * smax
            r_side = int(torch.sum(Svals > thr).item())

        n = u.V.shape[0]
        d = u.d

        if r_side == 0:
            Ured = torch.zeros((n, 0), dtype=torch.float64, device=Vp_svd.device)
            sigma = torch.zeros((0,), dtype=torch.float64, device=Vp_svd.device)
            Tred = torch.zeros((u.dprime_raw, 0), dtype=torch.float64, device=Vp_svd.device)
        else:
            Ured = Umat[:, :r_side].contiguous()          # (n, r_side)
            sigma = Svals[:r_side].contiguous()           # (r_side,)
            Tred = Vh.T[:, :r_side].contiguous()          # (dprime_raw, r_side)

        # lambda = sigma^2/(1+sigma^2), coeff = sigma/(1+sigma^2)
        sigma2 = sigma * sigma
        denom = 1.0 + sigma2
        lambda_ = sigma2 / denom
        coeff = sigma / denom

        # Move reduced params to main device/dtype
        Ured = Ured.to(dev_main, dtype=dtype)
        sigma = sigma.to(dev_main, dtype=dtype)
        lambda_ = lambda_.to(dev_main, dtype=dtype)
        coeff = coeff.to(dev_main, dtype=dtype)

        if cfg.store_side_T:
            T_keep = Tred.to(dev_main, dtype=dtype)
        else:
            T_keep = None

        # Precompute UtV = U^T V (small but crucial)
        V = u.V
        if r_side > 0:
            UtV = (Ured.T @ V).contiguous()  # (r_side, d)
        else:
            UtV = torch.zeros((0, d), device=dev_main, dtype=dtype)

        # Optional precompute U_sigma = U * sigma
        if cfg.precompute_u_sigma and r_side > 0:
            U_sigma = (Ured * sigma.unsqueeze(0)).contiguous()
        else:
            U_sigma = None

        u.side = SideInfoReduced(
            U=Ured,
            sigma=sigma,
            T=T_keep,
            lambda_=lambda_,
            coeff=coeff,
            UtV=UtV,
            U_sigma=U_sigma,
        )

        if cfg.verbose:
            print(
                f"[preprocess] user {idx}: "
                f"sigma_ch={u.channel_noise_std:g}, sigma_side={u.side_noise_std:g}, "
                f"rank(Vprime_eff) -> d_side={r_side}"
            )


# =========================
# Decoder update
# =========================
@torch.no_grad()
def update_decoders(users: List[User], E: torch.Tensor, cfg: SolverConfig) -> List[Decoder]:
    """
    For fixed E, with whitened H and reduced side model:

      R = H_eff E^T

      Let Pi = I - U diag(lambda) U^T.
      We need:
        Gy = (I + R Pi R^T)^(-1) R (Pi V)
        Gw = diag(coeff) * U^T (V - R^T Gy)

    Implementation tricks (no change to math):
      - Do NOT store PiV (n×d). Compute RHS = R V - (R U diag(lambda) (U^T V)).
      - Do NOT form residual (n×d): U^T(V - R^T Gy) = (U^T V) - (R U)^T Gy.
    """
    decs: List[Decoder] = []
    dev = E.device
    dtype = E.dtype

    for u in users:
        assert u.H is not None and u.side is not None, "Call preprocess_side_info first."

        H = u.H
        V = u.V
        Ured = u.side.U
        lambda_ = u.side.lambda_
        coeff = u.side.coeff
        UtV = u.side.UtV

        m = u.m
        d = u.d
        d_side = u.d_side

        # R = H E^T : (m, r)(r, n)->(m, n)
        R = H @ E.T

        if m == 0:
            Gy = torch.zeros((0, d), device=dev, dtype=dtype)
            Gw = torch.zeros((d_side, d), device=dev, dtype=dtype)
            decs.append(Decoder(Gy=Gy, Gw=Gw))
            continue

        RRt = R @ R.T  # (m, m)
        if d_side > 0:
            # RU = R U, and RPiRt = RRt - (RU diag(lambda)) RU^T
            RU = R @ Ured  # (m, d_side)
            RU_scaled = RU * lambda_.unsqueeze(0)
            RPiRt = RRt - (RU_scaled @ RU.T)

            # RHS = R(Pi V) = R V - (RU diag(lambda)) (U^T V)
            RHS = (R @ V) - (RU_scaled @ UtV)
        else:
            RPiRt = RRt
            RHS = R @ V

        # S = I + RPiRt, implemented without allocating I
        S = RPiRt.clone()
        S.diagonal().add_(1.0)

        Gy = safe_cholesky_solve(S, RHS, jitter=cfg.chol_jitter)

        if d_side > 0:
            # U^T(V - R^T Gy) = UtV - (RU^T Gy)
            Utr = UtV - (RU.T @ Gy)  # (d_side, d)
            Gw = coeff.unsqueeze(1) * Utr
        else:
            Gw = torch.zeros((0, d), device=dev, dtype=dtype)

        decs.append(Decoder(Gy=Gy, Gw=Gw))

    return decs


# =========================
# Objective evaluation
# =========================
@torch.no_grad()
def _compute_VprimeGw(u: User, Gw: torch.Tensor) -> torch.Tensor:
    """
    Compute (U Sigma) Gw as (n, d) without changing the algorithm.
    Uses precomputed U_sigma if available; otherwise uses U @ (sigma*Gw).
    """
    assert u.side is not None
    if u.d_side == 0:
        return torch.zeros_like(u.V)

    Ured = u.side.U
    sigma = u.side.sigma
    if u.side.U_sigma is not None:
        return u.side.U_sigma @ Gw
    return Ured @ (sigma.unsqueeze(1) * Gw)


@torch.no_grad()
def compute_wsmse(users: List[User], decs: List[Decoder], E: torch.Tensor) -> torch.Tensor:
    """
    With whitened/reduced model (unit noises), use:
      M = Gy^T H_eff
      C_T = V - (U Sigma) Gw
      E_i = ||C_T - E M^T||_F^2 + ||Gy||_F^2 + ||Gw||_F^2
      F = sum_i omega_i E_i
    """
    total = torch.zeros((), device=E.device, dtype=E.dtype)
    for u, dec in zip(users, decs):
        assert u.H is not None and u.side is not None
        H = u.H
        V = u.V

        Gy = dec.Gy
        Gw = dec.Gw

        M = Gy.T @ H  # (d, r)

        VprimeGw = _compute_VprimeGw(u, Gw)  # (n, d)
        C_T = V - VprimeGw
        residual = C_T - (E @ M.T)

        mse_i = fro_norm_sq(residual) + fro_norm_sq(Gy) + fro_norm_sq(Gw)
        total = total + (u.weight * mse_i)

    return total


@torch.no_grad()
def compute_user_mse(users: List[User], decs: List[Decoder], E: torch.Tensor) -> List[float]:
    """
    Return per-user (unweighted) MSE_i under the whitened/reduced model.
    This is useful for per-user curves; it costs extra compute if called every iter.
    """
    out: List[float] = []
    for u, dec in zip(users, decs):
        assert u.H is not None and u.side is not None
        H = u.H
        V = u.V

        Gy = dec.Gy
        Gw = dec.Gw

        M = Gy.T @ H
        VprimeGw = _compute_VprimeGw(u, Gw)

        C_T = V - VprimeGw
        residual = C_T - (E @ M.T)

        mse_i = fro_norm_sq(residual) + fro_norm_sq(Gy) + fro_norm_sq(Gw)
        out.append(float(mse_i.item()))
    return out


# =========================
# Precoder update: build P,Q and solve mu by safeguarded Newton
# =========================
@torch.no_grad()
def build_precoder_mats(users: List[User], decs: List[Decoder]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build:
      Q = sum_i w_i M_i^T M_i (r,r),   M_i = Gy_i^T H_i_eff (d_i, r)
      P = sum_i w_i C_T_i M_i (n,r),   C_T_i = V_i - (U_i Sigma_i) Gw_i (n, d_i)
    """
    device = users[0].V.device
    dtype = users[0].V.dtype
    n = users[0].V.shape[0]
    r = users[0].r

    P = torch.zeros((n, r), device=device, dtype=dtype)
    Q = torch.zeros((r, r), device=device, dtype=dtype)

    for u, dec in zip(users, decs):
        assert u.H is not None and u.side is not None
        H = u.H
        V = u.V

        Gy = dec.Gy
        Gw = dec.Gw
        w = u.weight

        M = Gy.T @ H  # (d, r)
        Q = Q + (w * (M.T @ M))

        VprimeGw = _compute_VprimeGw(u, Gw)
        C_T = V - VprimeGw
        P = P + (w * (C_T @ M))

    Q = 0.5 * (Q + Q.T)
    return P, Q


@torch.no_grad()
def solve_mu_safeguarded_newton(
    P: torch.Tensor,
    Q: torch.Tensor,
    power: float,
    cfg: SolverConfig
) -> Tuple[torch.Tensor, float]:
    """
    Solve mu >= 0 such that ||E(mu)||_F^2 = power, where
      E(mu) = P (Q + mu I)^(-1)

    Using EVD Q=U diag(lam) U^T:
      phi(mu)  = sum_j a_j/(lam_j+mu)^2,  a_j=|| (P U)_(:,j) ||^2
      phi'(mu) = -2 sum_j a_j/(lam_j+mu)^3
    Safeguarded Newton within bracket [lo, hi].

    Numerics:
      - We clamp eigenvalues for phi(0) test to avoid misclassifying the active constraint
        when Q has near-zero eigenvalues.
    """
    device = P.device
    dtype = P.dtype
    r = Q.shape[0]
    if r == 0:
        return torch.zeros_like(P[:, :0]), 0.0

    lam, Uq = torch.linalg.eigh(Q)
    lam = torch.clamp(lam, min=0.0)

    PU = P @ Uq
    a = torch.sum(PU * PU, dim=0)

    power_t = torch.tensor(power, device=device, dtype=dtype)

    # phi(0) test (use lam_safe to avoid underestimating when lam ~ 0)
    lam_safe0 = torch.clamp(lam, min=cfg.eig_tol)
    phi0 = torch.sum(a / (lam_safe0 * lam_safe0))

    # Inactive constraint => mu=0
    if phi0 <= power_t * (1.0 + cfg.mu_tol_rel):
        inv_lam = torch.where(lam > cfg.eig_tol, 1.0 / lam, torch.zeros_like(lam))
        E = (PU * inv_lam.unsqueeze(0)) @ Uq.T
        return E, 0.0

    def phi(mu: torch.Tensor) -> torch.Tensor:
        den = lam + mu
        den = torch.clamp(den, min=cfg.eig_tol)
        return torch.sum(a / (den * den))

    def dphi(mu: torch.Tensor) -> torch.Tensor:
        den = lam + mu
        den = torch.clamp(den, min=cfg.eig_tol)
        return -2.0 * torch.sum(a / (den * den * den))

    # Bracket search for hi
    sum_a = torch.sum(a)
    mu_hi = torch.maximum(torch.tensor(1.0, device=device, dtype=dtype), torch.sqrt(sum_a + 1e-30))
    mu_lo = torch.tensor(0.0, device=device, dtype=dtype)

    for _ in range(60):
        if phi(mu_hi) <= power_t:
            break
        mu_hi = 2.0 * mu_hi

    mu = 0.5 * (mu_lo + mu_hi)

    for _ in range(cfg.newton_max_iters):
        val = phi(mu) - power_t
        if torch.abs(val) <= (cfg.mu_tol_rel * power_t):
            break

        der = dphi(mu)  # negative
        mu_new = mu - val / der

        # Safeguard
        if (not torch.isfinite(mu_new)) or (mu_new <= mu_lo) or (mu_new >= mu_hi):
            mu_new = 0.5 * (mu_lo + mu_hi)

        if phi(mu_new) > power_t:
            mu_lo = mu_new
        else:
            mu_hi = mu_new

        mu = mu_new

    inv = 1.0 / torch.clamp(lam + mu, min=cfg.eig_tol)
    E = (PU * inv.unsqueeze(0)) @ Uq.T
    return E, float(mu.item())


@torch.no_grad()
def update_precoder(users: List[User], decs: List[Decoder], cfg: SolverConfig) -> Tuple[torch.Tensor, float]:
    P, Q = build_precoder_mats(users, decs)
    E, mu = solve_mu_safeguarded_newton(P, Q, power=cfg.power, cfg=cfg)
    return E, mu


# =========================
# AO Solver
# =========================
@torch.no_grad()
def init_precoder(n: int, r: int, cfg: SolverConfig, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device(cfg.device)
    dtype = cfg.dtype

    E0 = torch.randn((n, r), device=dev, dtype=dtype)
    if n >= r and r > 0:
        E0 = qr_orthonormal_columns(E0)

    norm = torch.linalg.norm(E0)
    if norm > 0:
        E0 = E0 * math.sqrt(cfg.power) / norm
    return E0


@torch.no_grad()
def solve_wsmse_ao_preprocessed(
    users: List[User],
    cfg: SolverConfig,
    seed: Optional[int] = None,
    E0: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """
    AO solver assuming users have already been preprocessed by preprocess_side_info(users, cfg).

    Use this for fast sweeps (e.g., SNR sweep) so you don't repeat SVD every point.
    Optionally warm-start by supplying E0 (e.g., previous SNR's solution).
    """
    # sanity check
    for i, u in enumerate(users):
        if u.H is None or u.side is None:
            raise ValueError(f"User {i} is not preprocessed. Call preprocess_side_info(users, cfg) first.")

    n = users[0].V.shape[0]
    r = users[0].r

    # init / warm-start
    if E0 is None:
        E = init_precoder(n=n, r=r, cfg=cfg, seed=seed)
    else:
        E = E0.to(device=torch.device(cfg.device), dtype=cfg.dtype)
        p0 = float(fro_norm_sq(E).item())
        if p0 > 0.0:
            E = E * math.sqrt(cfg.power / p0)
        else:
            E = init_precoder(n=n, r=r, cfg=cfg, seed=seed)

    # initial decoders/objective
    decs = update_decoders(users, E, cfg)
    F = compute_wsmse(users, decs, E)

    hist: Dict[str, object] = {"F": [float(F.item())], "mu": []}
    if cfg.track_per_user:
        hist["mse_per_user"] = [compute_user_mse(users, decs, E)]

    if cfg.verbose:
        print(f"[iter 0] F={hist['F'][-1]:.6e}, ||E||_F^2={float(fro_norm_sq(E).item()):.6e}")

    eps_rel = torch.tensor(1e-30, device=E.device, dtype=E.dtype)

    for t in range(1, cfg.max_iters + 1):
        # Step 1: decoder update (optimal for current E)
        decs = update_decoders(users, E, cfg)

        # Step 2: precoder update (optimal for current decoders)
        E, mu = update_precoder(users, decs, cfg)

        # Step 3: objective using (decs, E) -- decs corresponds to previous E,
        # but core AO logic remains intact; we optionally align outputs at the end.
        F_new = compute_wsmse(users, decs, E)

        rel = torch.abs(F_new - F) / torch.maximum(torch.abs(F), eps_rel)

        hist["F"].append(float(F_new.item()))
        hist["mu"].append(mu)
        if cfg.track_per_user:
            hist["mse_per_user"].append(compute_user_mse(users, decs, E))

        if cfg.verbose:
            print(
                f"[iter {t}] F={hist['F'][-1]:.6e}, rel={float(rel.item()):.3e}, "
                f"mu={mu:.3e}, ||E||_F^2={float(fro_norm_sq(E).item()):.6e}"
            )

        if float(rel.item()) <= cfg.rel_tol:
            if cfg.verbose:
                print(f"[stop] relative improvement <= {cfg.rel_tol}")
            break

        F = F_new

    # Optional: align returned decoders with final E (no algorithm change, just output consistency)
    if cfg.align_final_decoders:
        decs = update_decoders(users, E, cfg)
        F_aligned = compute_wsmse(users, decs, E)
        # overwrite final history value with the aligned one (monotone improvement)
        hist["F"][-1] = float(F_aligned.item())
        if cfg.track_per_user:
            hist["mse_per_user"][-1] = compute_user_mse(users, decs, E)

        if cfg.verbose:
            print(f"[final-align] F_aligned={hist['F'][-1]:.6e}")

    return {"E": E, "decoders": decs, "history": hist}


@torch.no_grad()
def solve_wsmse_ao(users: List[User], cfg: SolverConfig, seed: Optional[int] = None) -> Dict[str, object]:
    """
    Standard entry: preprocess + AO solve.
    For sweeps, prefer:
      preprocess_side_info(users, cfg) once,
      then call solve_wsmse_ao_preprocessed(..., power=...)
    """
    preprocess_side_info(users, cfg)
    return solve_wsmse_ao_preprocessed(users, cfg, seed=seed, E0=None)


# =========================
# Plotting
# =========================
def plot_mse_curve(
    hist: Dict[str, object],
    save_path: str = "mse_curve.png",
    logy: bool = True,
    show: Optional[bool] = None,
    title: str = "WSMSE Descent",
) -> None:
    """
    Plot and save the global objective curve (hist["F"]).
    If track_per_user=True, also plots per-user MSE curves (unweighted) in additional figure.
    """
    if show is None:
        show = _running_in_notebook()

    # In script/headless mode, use a non-interactive backend
    if not show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    F = hist["F"]
    plt.figure()
    plt.plot(range(len(F)), F, marker="o", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("WSMSE")
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()

    if "mse_per_user" in hist:
        mse_per_user = hist["mse_per_user"]  # list over iterations, each is list over users
        K = len(mse_per_user[0]) if len(mse_per_user) > 0 else 0
        plt.figure()
        for i in range(K):
            curve = [m[i] for m in mse_per_user]
            plt.plot(range(len(curve)), curve)
        plt.xlabel("Iteration")
        plt.ylabel("Per-user MSE (unweighted)")
        plt.title("Per-user MSE Descent")
        if logy:
            plt.yscale("log")
        plt.grid(True, which="both", linestyle="--", linewidth=0.6)
        plt.tight_layout()
        per_user_path = save_path.replace(".png", "_per_user.png")
        plt.savefig(per_user_path, dpi=200)
        if show:
            plt.show()
        plt.close()


# =========================
# Random instance generator (for quick tests / scalability)
# =========================
@torch.no_grad()
def random_orthonormal_V(n: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    A = torch.randn((n, d), device=device, dtype=dtype)
    if n >= d and d > 0:
        return qr_orthonormal_columns(A)
    return A / (torch.linalg.norm(A) + 1e-30)


@torch.no_grad()
def generate_random_instance(
    n: int,
    r: int,
    K: int,
    m_list: List[int],
    d_list: List[int],
    dprime_list: List[int],
    rankprime_list: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
    channel_noise_std_list: Optional[List[float]] = None,
    side_noise_std_list: Optional[List[float]] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
) -> List[User]:
    """
    Generate a random test instance.

    channel_noise_std_list[i] sets sigma_ch_i
    side_noise_std_list[i]    sets sigma_side_i

    Vprime_raw is generated low-rank: A(n,rp) @ B(rp,dp).
    """
    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device(device)

    if rankprime_list is None:
        rankprime_list = [min(n, dp) for dp in dprime_list]
    if weights is None:
        weights = [1.0 for _ in range(K)]
    if channel_noise_std_list is None:
        channel_noise_std_list = [1.0 for _ in range(K)]
    if side_noise_std_list is None:
        side_noise_std_list = [1.0 for _ in range(K)]

    users: List[User] = []
    for i in range(K):
        m_i = m_list[i]
        d_i = d_list[i]
        dp_i = dprime_list[i]
        rp_i = rankprime_list[i]

        H = torch.randn((m_i, r), device=dev, dtype=dtype) / math.sqrt(max(r, 1))
        V = random_orthonormal_V(n, d_i, dev, dtype)

        if dp_i == 0 or rp_i == 0:
            Vp = torch.zeros((n, dp_i), device=dev, dtype=dtype)
        else:
            A = torch.randn((n, rp_i), device=dev, dtype=dtype) / math.sqrt(max(rp_i, 1))
            B = torch.randn((rp_i, dp_i), device=dev, dtype=dtype) / math.sqrt(max(dp_i, 1))
            Vp = A @ B

        users.append(
            User(
                H_raw=H,
                V=V,
                Vprime_raw=Vp,
                weight=float(weights[i]),
                channel_noise_std=float(channel_noise_std_list[i]),
                side_noise_std=float(side_noise_std_list[i]),
            )
        )

    return users


# =========================
# CLI (Notebook-safe)
# =========================
def main(argv: Optional[List[str]] = None) -> None:
    """
    - CLI: python lcbc_wsmse_solver.py --K 3 --ch-noise-list 0.5,1,2 --side-noise-list 0.2,0.2,1
    - Notebook: import this file as module, call main([...]) OR call solve_wsmse_ao(...) directly.

    Notebook-safe behavior:
      - If argv is None and running in notebook, ignores ipykernel's injected args (e.g., '-f kernel.json').
      - If argv is provided explicitly, remains strict on unknown args.
    """
    parser = argparse.ArgumentParser("OA-LCBC WSMSE AO Solver (PyTorch) [Optimized]")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--K", type=int, default=10)

    parser.add_argument("--m", type=int, default=16, help="default m_i (homogeneous)")
    parser.add_argument("--d", type=int, default=8, help="default d_i (homogeneous)")
    parser.add_argument("--dprime", type=int, default=32, help="default raw d_i' (homogeneous)")
    parser.add_argument("--rankprime", type=int, default=16, help="default rank(V_i') for low-rank generation (homogeneous)")

    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    parser.add_argument("--power", type=float, default=1.0)

    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")

    # Per-user noise controls (either scalar broadcast, or comma-separated list)
    parser.add_argument("--ch-noise", type=float, default=0.01, help="scalar sigma_ch for all users (if --ch-noise-list not set)")
    parser.add_argument("--side-noise", type=float, default=0.01, help="scalar sigma_side for all users (if --side-noise-list not set)")
    parser.add_argument("--ch-noise-list", type=str, default="", help="comma-separated sigma_ch list of length K, e.g. '0.5,1,2'")
    parser.add_argument("--side-noise-list", type=str, default="", help="comma-separated sigma_side list of length K, e.g. '1,1,0.2'")

    # Weights
    parser.add_argument("--weights", type=str, default="", help="comma-separated omega_i list of length K; default all ones")

    # Side-info storage/speed knobs
    parser.add_argument("--no-store-T", action="store_true", help="do not store side matrix T (saves memory; only needed to map Gw back to raw w')")
    parser.add_argument("--precompute-u-sigma", action="store_true", help="precompute U*sigma (uses extra memory, can speed up VprimeGw)")

    # Plot
    parser.add_argument("--save-fig", type=str, default="mse_curve.png", help="path to save WSMSE curve")
    parser.add_argument("--no-plot", action="store_true", help="disable plotting")
    parser.add_argument("--logy", action="store_true", help="use log-scale y-axis in plots")
    parser.add_argument("--track-per-user", action="store_true", help="track per-user MSE curves (extra compute)")

    args, unknown = parser.parse_known_args(argv)
    if unknown and (argv is not None or (not _running_in_notebook())):
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    cfg = SolverConfig(
        max_iters=args.max_iters,
        rel_tol=args.rel_tol,
        power=args.power,
        dtype=dtype,
        device=args.device,
        verbose=(not args.quiet),
        mu_tol_rel=1e-12 if dtype == torch.float64 else 1e-8,
        chol_jitter=1e-12 if dtype == torch.float64 else 1e-8,
        track_per_user=bool(args.track_per_user),
        store_side_T=(not args.no_store_T),
        precompute_u_sigma=bool(args.precompute_u_sigma),
    )

    K = args.K

    # Parse lists
    ch_list = _parse_csv_floats(args.ch_noise_list)
    side_list = _parse_csv_floats(args.side_noise_list)
    w_list = _parse_csv_floats(args.weights)

    channel_noise_std_list = _broadcast_param_list(ch_list, K, args.ch_noise)
    side_noise_std_list = _broadcast_param_list(side_list, K, args.side_noise)
    weights = _broadcast_param_list(w_list, K, 1.0)

    # Homogeneous sizes (edit here for heterogeneous)
    m_list = [args.m for _ in range(K)]
    d_list = [args.d for _ in range(K)]
    dprime_list = [args.dprime for _ in range(K)]
    rankprime_list = [min(args.rankprime, args.n, args.dprime) for _ in range(K)]

    users = generate_random_instance(
        n=args.n,
        r=args.r,
        K=K,
        m_list=m_list,
        d_list=d_list,
        dprime_list=dprime_list,
        rankprime_list=rankprime_list,
        weights=weights,
        channel_noise_std_list=channel_noise_std_list,
        side_noise_std_list=side_noise_std_list,
        device=cfg.device,
        dtype=cfg.dtype,
        seed=args.seed,
    )

    out = solve_wsmse_ao(users, cfg, seed=args.seed)
    E = out["E"]
    hist = out["history"]

    # Plot/save curve
    if not args.no_plot:
        plot_mse_curve(
            hist=hist,
            save_path=args.save_fig,
            logy=bool(args.logy),
            show=None,
            title="WSMSE Descent (AO/BCD)",
        )
        if not cfg.verbose:
            print(f"Saved curve to: {args.save_fig}")

    if not cfg.verbose:
        print(
            f"Final F={hist['F'][-1]:.6e}, ||E||_F^2={float(fro_norm_sq(E).item()):.6e}, "
            f"iters={len(hist['F'])-1}"
        )


if __name__ == "__main__":
    main()
