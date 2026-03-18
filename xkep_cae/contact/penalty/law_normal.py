"""法線接触力則 — AL + Smooth Penalty.

純粋関数 + frozen dataclass で実装。

接触反力（AL）:
    p_n = max(0, λ_n + k_pen × (-g))

接触反力（Smooth Penalty, softplus）:
    p_n = k_pen × softplus(-g + λ_n/k_pen, δ)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from xkep_cae.core import ProcessMeta, SolverProcess

# ── データ型 ──────────────────────────────────────────────


@dataclass(frozen=True)
class NormalForceInput:
    """法線力評価の入力."""

    gap: float
    lambda_n: float
    k_pen: float
    is_active: bool = True


@dataclass(frozen=True)
class NormalForceResult:
    """法線力評価の出力."""

    p_n: float
    dp_dg: float


@dataclass(frozen=True)
class SmoothNormalForceInput:
    """スムースペナルティ法線力の入力."""

    gap: float
    k_pen: float
    lambda_n: float = 0.0
    delta: float = 1e-4


@dataclass(frozen=True)
class VectorizedSmoothInput:
    """スムースペナルティ法線力のベクトル版入力."""

    gaps: np.ndarray
    k_pen: float
    lambdas: np.ndarray
    delta: float = 1e-4


@dataclass(frozen=True)
class VectorizedNormalForceResult:
    """ベクトル版法線力の出力."""

    p_n: np.ndarray
    dp_dg: np.ndarray


# ── 純粋関数 ─────────────────────────────────────────────


def _softplus(x: float, delta: float) -> tuple[float, float]:
    """Softplus 関数とその導関数（数値安定版）.

    softplus(x, δ) = δ × ln(1 + exp(x/δ))
    sigmoid(x, δ) = 1 / (1 + exp(-x/δ))

    Args:
        x: 入力値
        delta: 平滑化幅 (> 0)

    Returns:
        (softplus値, sigmoid値)
    """
    z = x / delta
    if z > 30.0:
        return x, 1.0
    if z < -30.0:
        return 0.0, 0.0
    exp_z = math.exp(z)
    return delta * math.log1p(exp_z), exp_z / (1.0 + exp_z)


def _evaluate_al_normal_force(
    gap: float, lambda_n: float, k_pen: float, *, is_active: bool = True
) -> tuple[float, float]:
    """AL 法線接触力を評価.

    p_n = max(0, λ_n + k_pen × (-g))
    dp_dg = -k_pen  (接触中), 0 (非接触)

    Returns:
        (p_n, dp_dg)
    """
    if not is_active:
        return 0.0, 0.0
    p_n = max(0.0, lambda_n + k_pen * (-gap))
    dp_dg = -k_pen if p_n > 0.0 else 0.0
    return p_n, dp_dg


def _evaluate_smooth_normal_force(
    gap: float, k_pen: float, lambda_n: float = 0.0, *, delta: float = 1e-4
) -> tuple[float, float]:
    """スムースペナルティ法線力（C∞連続）.

    p_n = k_pen × softplus(-g + λ_n/k_pen, δ)
    dp_dg = -k_pen × sigmoid((-g + λ_n/k_pen) / δ)

    Returns:
        (p_n, dp_dg)
    """
    x = -gap + lambda_n / k_pen if k_pen > 0.0 else -gap
    sp, sig = _softplus(x, delta)
    return k_pen * sp, -k_pen * sig


def _evaluate_smooth_normal_force_vectorized(
    gaps: np.ndarray,
    k_pen: float,
    lambdas: np.ndarray,
    *,
    delta: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """スムースペナルティ法線力のベクトル版（全ペア一括）.

    Returns:
        (p_n_array, dp_dg_array)
    """
    x = -gaps + lambdas / k_pen if k_pen > 0.0 else -gaps
    z = x / delta
    z_clip = np.clip(z, -30.0, 30.0)
    exp_z = np.exp(z_clip)

    sp = np.where(z > 30.0, x, np.where(z < -30.0, 0.0, delta * np.log1p(exp_z)))
    sig = np.where(z > 30.0, 1.0, np.where(z < -30.0, 0.0, exp_z / (1.0 + exp_z)))

    return k_pen * sp, -k_pen * sig


def _auto_beam_penalty_stiffness(
    E: float,
    I: float,  # noqa: E741
    L_elem: float,
    *,
    n_contact_pairs: int = 1,
    scale: float = 0.1,
    scaling: str = "linear",
) -> float:
    """梁曲げ剛性 EI/L³ ベースのペナルティ剛性自動推定.

    k_pen = scale × 12EI / L³ / max(1, f(n_pairs))
    """
    if L_elem <= 0.0:
        raise ValueError(f"L_elem は正の値が必要: {L_elem}")
    k_bend = 12.0 * E * I / L_elem**3
    n_eff = max(1, n_contact_pairs)
    if scaling == "sqrt":
        return scale * k_bend / max(1.0, math.sqrt(n_eff))
    return scale * k_bend / n_eff


# ── Process ラップ ──────────────────────────────────────────


class ALNormalForceProcess(SolverProcess[NormalForceInput, NormalForceResult]):
    """AL 法線接触力 Process."""

    meta = ProcessMeta(
        name="ALNormalForce",
        module="solve",
        version="1.0.0",
        document_path="docs/penalty.md",
    )

    def process(self, input_data: NormalForceInput) -> NormalForceResult:
        p_n, dp_dg = _evaluate_al_normal_force(
            input_data.gap, input_data.lambda_n, input_data.k_pen, is_active=input_data.is_active
        )
        return NormalForceResult(p_n=p_n, dp_dg=dp_dg)


class SmoothNormalForceProcess(SolverProcess[SmoothNormalForceInput, NormalForceResult]):
    """スムースペナルティ法線力 Process."""

    meta = ProcessMeta(
        name="SmoothNormalForce",
        module="solve",
        version="1.0.0",
        document_path="docs/penalty.md",
    )

    def process(self, input_data: SmoothNormalForceInput) -> NormalForceResult:
        p_n, dp_dg = _evaluate_smooth_normal_force(
            input_data.gap, input_data.k_pen, input_data.lambda_n, delta=input_data.delta
        )
        return NormalForceResult(p_n=p_n, dp_dg=dp_dg)
