"""Coulomb 摩擦則 — return mapping + consistent tangent.

旧 xkep_cae_deprecated/contact/law_friction.py の完全書き直し。
mutable な ContactPair への副作用を排除し、純粋関数 + frozen dataclass で実装。

摩擦の return mapping:
    1. q_trial = z_t_old + k_t × Δu_t       (弾性予測)
    2. Coulomb 条件: ||q_trial|| ≤ μ × p_n
    3. stick: q = q_trial
    4. slip:  q = μ × p_n × q_trial / ||q_trial||

slip consistent tangent:
    D_t = (μ×p_n / ||q_trial||) × k_t × (I₂ - q̂⊗q̂)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.core import ProcessMeta, SolverProcess

# ── データ型 ──────────────────────────────────────────────


@dataclass(frozen=True)
class ReturnMappingInput:
    """Coulomb return mapping の入力."""

    z_t_old: np.ndarray  # (2,) 接線履歴ベクトル
    delta_ut: np.ndarray  # (2,) 接線相対変位増分
    k_t: float  # 接線ペナルティ剛性
    p_n: float  # 法線力 (>0 で接触中)
    mu: float  # 摩擦係数


@dataclass(frozen=True)
class ReturnMappingResult:
    """Coulomb return mapping の出力."""

    q: np.ndarray  # (2,) 摩擦力
    is_stick: bool  # stick 状態フラグ
    q_trial_norm: float  # trial 接線力ノルム (slip consistent tangent 用)
    dissipation: float  # 散逸増分


@dataclass(frozen=True)
class TangentInput:
    """摩擦接線剛性の入力."""

    k_t: float
    p_n: float
    mu: float
    z_t: np.ndarray  # (2,) 現在の接線履歴
    q_trial_norm: float
    is_stick: bool


@dataclass(frozen=True)
class TangentResult:
    """摩擦接線剛性の出力."""

    D_t: np.ndarray  # (2, 2) 摩擦接線剛性


@dataclass(frozen=True)
class RotateHistoryInput:
    """摩擦履歴フレーム回転の入力."""

    z_t: np.ndarray  # (2,) 旧フレームでの摩擦履歴
    t1_old: np.ndarray  # (3,) 旧接線基底1
    t2_old: np.ndarray  # (3,) 旧接線基底2
    t1_new: np.ndarray  # (3,) 新接線基底1
    t2_new: np.ndarray  # (3,) 新接線基底2


@dataclass(frozen=True)
class MuRampInput:
    """μ ランプの入力."""

    mu_target: float
    ramp_counter: int
    mu_ramp_steps: int


# ── 純粋関数 ─────────────────────────────────────────────


def _return_mapping_core(
    z_t_old: np.ndarray,
    delta_ut: np.ndarray,
    k_t: float,
    p_n: float,
    mu: float,
) -> tuple[np.ndarray, bool, float, float]:
    """Coulomb return mapping の純粋関数.

    Args:
        z_t_old: (2,) 接線履歴ベクトル
        delta_ut: (2,) 接線相対変位増分
        k_t: 接線ペナルティ剛性
        p_n: 法線力 (>0 で接触中)
        mu: 摩擦係数

    Returns:
        (q, is_stick, q_trial_norm, dissipation)
    """
    if p_n <= 0.0 or mu <= 0.0:
        return np.zeros(2), True, 0.0, 0.0

    q_trial = z_t_old + k_t * delta_ut
    q_trial_norm = float(np.linalg.norm(q_trial))
    f_yield = mu * p_n

    if q_trial_norm <= f_yield:
        q = q_trial.copy()
        is_stick = True
    else:
        q = f_yield * q_trial / q_trial_norm
        is_stick = False

    dissipation = float(np.dot(q, delta_ut))
    return q, is_stick, q_trial_norm, dissipation


def _tangent_2x2_core(
    k_t: float,
    p_n: float,
    mu: float,
    z_t: np.ndarray,
    q_trial_norm: float,
    is_stick: bool,
) -> np.ndarray:
    """摩擦接線剛性 (2x2) の純粋関数.

    stick: D_t = k_t × I₂
    slip:  D_t = (μ×p_n / ||q_trial||) × k_t × (I₂ - q̂⊗q̂)

    Returns:
        D_t: (2, 2) 摩擦接線剛性
    """
    if p_n <= 0.0 or mu <= 0.0:
        return np.zeros((2, 2))

    if is_stick:
        return k_t * np.eye(2)

    z_norm = float(np.linalg.norm(z_t))
    if z_norm < 1e-30 or q_trial_norm < 1e-30:
        return k_t * np.eye(2)
    q_hat = z_t / z_norm
    ratio = (mu * p_n) / q_trial_norm
    return ratio * k_t * (np.eye(2) - np.outer(q_hat, q_hat))


def _rotate_friction_history(
    z_t: np.ndarray,
    t1_old: np.ndarray,
    t2_old: np.ndarray,
    t1_new: np.ndarray,
    t2_new: np.ndarray,
) -> np.ndarray:
    """摩擦履歴ベクトルを旧フレームから新フレームに回転.

    変換行列 R_{2×2}:
        R[i,j] = t_i_new · t_j_old  (i,j = 1,2)

    Returns:
        z_t_new: (2,) 新フレームでの摩擦履歴
    """
    if float(np.linalg.norm(z_t)) < 1e-30:
        return z_t.copy()

    R = np.array(
        [
            [float(t1_new @ t1_old), float(t1_new @ t2_old)],
            [float(t2_new @ t1_old), float(t2_new @ t2_old)],
        ]
    )
    return R @ z_t


def _compute_mu_effective(
    mu_target: float,
    ramp_counter: int,
    mu_ramp_steps: int,
) -> float:
    """μ ランプを適用した有効摩擦係数.

    μ_eff = μ_target × min(1, ramp_counter / mu_ramp_steps)
    """
    if mu_ramp_steps <= 0:
        return mu_target
    ratio = min(1.0, ramp_counter / mu_ramp_steps)
    return mu_target * ratio


# ── Process ラップ ──────────────────────────────────────────


class ReturnMappingProcess(SolverProcess[ReturnMappingInput, ReturnMappingResult]):
    """Coulomb return mapping Process."""

    meta = ProcessMeta(
        name="ReturnMapping",
        module="solve",
        version="1.0.0",
        document_path="docs/friction.md",
    )

    def process(self, input_data: ReturnMappingInput) -> ReturnMappingResult:
        q, is_stick, q_trial_norm, dissipation = _return_mapping_core(
            input_data.z_t_old,
            input_data.delta_ut,
            input_data.k_t,
            input_data.p_n,
            input_data.mu,
        )
        return ReturnMappingResult(
            q=q, is_stick=is_stick, q_trial_norm=q_trial_norm, dissipation=dissipation
        )


class FrictionTangentProcess(SolverProcess[TangentInput, TangentResult]):
    """摩擦接線剛性 (2x2) Process."""

    meta = ProcessMeta(
        name="FrictionTangent",
        module="solve",
        version="1.0.0",
        document_path="docs/friction.md",
    )

    def process(self, input_data: TangentInput) -> TangentResult:
        D_t = _tangent_2x2_core(
            input_data.k_t,
            input_data.p_n,
            input_data.mu,
            input_data.z_t,
            input_data.q_trial_norm,
            input_data.is_stick,
        )
        return TangentResult(D_t=D_t)
