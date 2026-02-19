"""法線接触力則（Augmented Lagrangian）.

Phase C2: 法線方向の接触力計算と AL 乗数更新。

接触反力:
    p_n = max(0, lambda_n + k_pen * (-g))

乗数更新（Outer loop 終了時）:
    lambda_n <- p_n

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md §4
"""

from __future__ import annotations

from xkep_cae.contact.pair import ContactPair, ContactStatus


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


def update_al_multiplier(pair: ContactPair) -> None:
    """AL 乗数 lambda_n を更新する（Outer loop 終了時に呼ぶ）.

    lambda_n <- p_n  （現在の反力を乗数に反映）

    INACTIVE ペアは lambda_n = 0 にリセット。

    Args:
        pair: 接触ペア
    """
    if pair.state.status == ContactStatus.INACTIVE:
        pair.state.lambda_n = 0.0
    else:
        pair.state.lambda_n = pair.state.p_n


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
