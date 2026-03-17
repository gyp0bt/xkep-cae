"""法線接触力則（Augmented Lagrangian + Smooth Penalty）.

Phase C2: 法線方向の接触力計算と AL 乗数更新。
Phase C7: スムースペナルティ (softplus) による区分連続接触力。

接触反力（従来 AL）:
    p_n = max(0, lambda_n + k_pen * (-g))

接触反力（スムースペナルティ）:
    p_n = k_pen * softplus(-g + lambda_n/k_pen, delta)
    softplus(x, δ) = δ * ln(1 + exp(x/δ))

乗数更新（Uzawa, Outer loop 終了時）:
    lambda_n <- max(0, lambda_n + k_pen * (-g))

"""

from __future__ import annotations

import math

import numpy as np

from __xkep_cae_deprecated.contact.pair import ContactPair, ContactStatus


def evaluate_normal_force(pair: ContactPair) -> float:
    """法線接触反力 p_n を計算し、ContactState に書き込む.

    AL (Augmented Lagrangian) の反力:
        p_n = max(0, lambda_n + k_pen * (-g))

    ACTIVE でないペアは p_n = 0 を返す。

    Args:
        pair: 接触ペア（state.gap, state.lambda_n, state.k_pen が設定済み）

    Returns:
        p_n: 法線接触反力（>= 0）
    """
    if pair.state.status == ContactStatus.INACTIVE:
        pair.state.p_n = 0.0
        return 0.0

    g = pair.state.gap
    lam = pair.state.lambda_n
    k = pair.state.k_pen

    # AL 反力: max(0, lambda_n + k_pen * (-g))
    p_n = max(0.0, lam + k * (-g))
    pair.state.p_n = p_n
    return p_n


def _softplus(x: float, delta: float) -> tuple[float, float]:
    """Softplus 関数とその導関数（数値安定版）.

    softplus(x, δ) = δ * ln(1 + exp(x/δ))
    sigmoid(x, δ) = 1 / (1 + exp(-x/δ))

    オーバーフロー回避:
        x/δ > 30: softplus ≈ x, sigmoid ≈ 1
        x/δ < -30: softplus ≈ 0, sigmoid ≈ 0

    Args:
        x: 入力値
        delta: 平滑化幅 (> 0)

    Returns:
        sp: softplus(x, δ)
        sig: sigmoid(x, δ) = d(softplus)/dx
    """
    z = x / delta
    if z > 30.0:
        return x, 1.0
    elif z < -30.0:
        return 0.0, 0.0
    else:
        exp_z = math.exp(z)
        sp = delta * math.log1p(exp_z)
        sig = exp_z / (1.0 + exp_z)
        return sp, sig


def smooth_normal_force(
    gap: float,
    k_pen: float,
    lambda_n: float = 0.0,
    *,
    delta: float = 1e-4,
) -> tuple[float, float]:
    """スムースペナルティによる法線接触力（C∞連続）.

    softplus 平滑化により max(0, ·) を連続近似する。
    Active/inactive の二値判定を排除し、全ペアから連続的な力を返す。

    p_n = k_pen * softplus(-g + λ_n/k_pen, δ)

    導関数:
        dp_n/dg = -k_pen * sigmoid((-g + λ_n/k_pen) / δ)

    |g| >> δ の領域では従来の max(0, λ + k_pen*(-g)) と一致。
    遷移領域 |g| ~ δ で C∞ 平滑化される。

    Args:
        gap: ギャップ値 g（g > 0: 非接触, g < 0: 貫入）
        k_pen: ペナルティ剛性
        lambda_n: ラグランジュ乗数（Uzawa で更新、初期値 0）
        delta: 平滑化幅。梁半径の 1% 程度を推奨。

    Returns:
        p_n: 法線接触力 (>= 0)
        dp_dg: dp_n/dg（接線剛性への寄与、<= 0）
    """
    # x = -g + λ/k
    x = -gap + lambda_n / k_pen if k_pen > 0.0 else -gap
    sp, sig = _softplus(x, delta)
    p_n = k_pen * sp
    dp_dg = -k_pen * sig  # dp/dg = k_pen * d(softplus)/dx * dx/dg = k_pen * sig * (-1)
    return p_n, dp_dg


def smooth_normal_force_vectorized(
    gaps: np.ndarray,
    k_pen: float,
    lambdas: np.ndarray,
    *,
    delta: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """スムースペナルティのベクトル版（全ペア一括計算）.

    Args:
        gaps: (n,) ギャップ配列
        k_pen: ペナルティ剛性
        lambdas: (n,) ラグランジュ乗数配列
        delta: 平滑化幅

    Returns:
        p_n: (n,) 法線力配列
        dp_dg: (n,) 接線剛性配列
    """
    x = -gaps + lambdas / k_pen if k_pen > 0.0 else -gaps
    z = x / delta

    # 数値安定化: クリップして exp オーバーフロー回避
    z_clip = np.clip(z, -30.0, 30.0)
    exp_z = np.exp(z_clip)

    # softplus と sigmoid
    sp = np.where(z > 30.0, x, np.where(z < -30.0, 0.0, delta * np.log1p(exp_z)))
    sig = np.where(z > 30.0, 1.0, np.where(z < -30.0, 0.0, exp_z / (1.0 + exp_z)))

    p_n = k_pen * sp
    dp_dg = -k_pen * sig
    return p_n, dp_dg


def normal_force_linearization(pair: ContactPair) -> float:
    """法線接触反力の変分（接線剛性の主項係数）を返す.

    d(p_n)/d(gap) の評価。ペナルティ/AL主項:
        - ACTIVE かつ p_n > 0 の場合: k_pen
        - それ以外: 0

    幾何微分（法線 n の変化に伴う項）は v0.2 で追加予定。

    Args:
        pair: 接触ペア

    Returns:
        dp_dg: d(p_n)/d(-gap) = k_pen（接触中）or 0
    """
    if pair.state.status == ContactStatus.INACTIVE:
        return 0.0
    if pair.state.p_n <= 0.0:
        return 0.0
    return pair.state.k_pen


def initialize_penalty_stiffness(
    pair: ContactPair,
    k_pen: float,
    k_t_ratio: float = 0.5,
) -> None:
    """ペナルティ剛性を初期化する.

    Args:
        pair: 接触ペア
        k_pen: 法線ペナルティ剛性
        k_t_ratio: 接線/法線ペナルティ比
    """
    pair.state.k_pen = k_pen
    pair.state.k_t = k_t_ratio * k_pen


def auto_penalty_stiffness(
    E: float,
    A: float,
    L: float,
    scale: float = 1.0,
) -> float:
    """EA/L ベースのペナルティ剛性を自動推定する.

    Args:
        E: ヤング率
        A: 断面積
        L: 代表要素長さ
        scale: スケーリング係数（ContactConfig.k_pen_scale）

    Returns:
        k_pen: ペナルティ剛性
    """
    return scale * E * A / L


def auto_beam_penalty_stiffness(
    E: float,
    I: float,  # noqa: E741
    L_elem: float,
    *,
    n_contact_pairs: int = 1,
    scale: float = 0.1,
    scaling: str = "linear",
) -> float:
    """梁要素の曲げ剛性 EI/L³ ベースのペナルティ剛性自動推定.

    梁–梁接触では曲げ剛性 k_bend = 12EI/L³ が支配的であり、
    軸剛性 EA/L は過大評価となる。本関数は k_bend を基準とし、
    接触ペア数が多い場合にスケールダウンして条件数悪化を抑制する。

    推定式:
        linear: k_pen = scale * 12 * E * I / L³ / max(1, n_contact_pairs)
        sqrt:   k_pen = scale * 12 * E * I / L³ / max(1, sqrt(n_contact_pairs))

    sqrt スケーリングは多ペア時（>10）に k_pen が過度に小さくなることを
    防ぎ、貫入抑制と条件数のバランスを改善する。

    Args:
        E: ヤング率 [Pa]
        I: 代表断面二次モーメント [m⁴]（Iy と Iz のうち小さい方を推奨）
        L_elem: 代表要素長さ [m]
        n_contact_pairs: 予想される同時アクティブ接触ペア数（デフォルト1）
        scale: 基本スケーリング係数（デフォルト0.1）
        scaling: ペア数スケーリング方式 ("linear" | "sqrt")

    Returns:
        k_pen: 推定ペナルティ剛性 [N/m]

    Raises:
        ValueError: L_elem <= 0 の場合

    Examples:
        >>> k = auto_beam_penalty_stiffness(200e9, 1e-12, 0.01, n_contact_pairs=6)
        >>> k_bend = 12.0 * 200e9 * 1e-12 / 0.01**3
        >>> expected = 0.1 * k_bend / 6
        >>> abs(k - expected) < 1e-6
        True
    """
    if L_elem <= 0.0:
        raise ValueError(f"L_elem は正の値が必要: {L_elem}")

    k_bend = 12.0 * E * I / L_elem**3
    n_eff = max(1, n_contact_pairs)

    if scaling == "sqrt":
        import math

        return scale * k_bend / max(1.0, math.sqrt(n_eff))
    else:
        return scale * k_bend / n_eff
