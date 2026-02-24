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


def auto_beam_penalty_stiffness(
    E: float,
    I: float,  # noqa: E741
    L_elem: float,
    *,
    n_contact_pairs: int = 1,
    scale: float = 0.1,
) -> float:
    """梁要素の曲げ剛性 EI/L³ ベースのペナルティ剛性自動推定.

    梁–梁接触では曲げ剛性 k_bend = 12EI/L³ が支配的であり、
    軸剛性 EA/L は過大評価となる。本関数は k_bend を基準とし、
    接触ペア数が多い場合にスケールダウンして条件数悪化を抑制する。

    推定式:
        k_pen = scale * 12 * E * I / L³ / max(1, n_contact_pairs)

    n_contact_pairs による線形スケーリングの根拠:
        多点接触で全ペアに同じ k_pen を与えると、等価的な接触剛性が
        n_pairs 倍になり条件数が悪化する。n で線形除算して全体剛性を抑制。

    Args:
        E: ヤング率 [Pa]
        I: 代表断面二次モーメント [m⁴]（Iy と Iz のうち小さい方を推奨）
        L_elem: 代表要素長さ [m]
        n_contact_pairs: 予想される同時アクティブ接触ペア数（デフォルト1）
        scale: 基本スケーリング係数（デフォルト0.1）

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

    return scale * k_bend / n_eff
