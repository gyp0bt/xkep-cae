"""Strategy 互換性マトリクス.

検証済み組み合わせ（ホワイトリスト）と非互換組み合わせ（ブラックリスト）を管理する。

"""

from __future__ import annotations

# 検証済み組み合わせ（ホワイトリスト）
VERIFIED_COMBINATIONS: list[dict[str, str]] = [
    # 基軸構成（7本撚線曲げ収束実績）
    {
        "contact_force": "SmoothPenaltyContactForceProcess",
        "friction": "SmoothPenaltyFrictionProcess",
        "time_integration": "QuasiStaticProcess",
        "contact_geometry": "LineToLineGaussProcess",
        "penalty": "AutoBeamEIProcess",
    },
    # NCP法線 + 摩擦なし（基本テスト用）
    {
        "contact_force": "NCPContactForceProcess",
        "friction": "NoFrictionProcess",
        "time_integration": "QuasiStaticProcess",
        "contact_geometry": "PointToPointProcess",
        "penalty": "AutoBeamEIProcess",
    },
    # 動的解析（Generalized-α）
    {
        "contact_force": "SmoothPenaltyContactForceProcess",
        "friction": "SmoothPenaltyFrictionProcess",
        "time_integration": "GeneralizedAlphaProcess",
        "contact_geometry": "LineToLineGaussProcess",
        "penalty": "AutoBeamEIProcess",
    },
]

# 既知の非互換組み合わせ（ブラックリスト）
INCOMPATIBLE_COMBINATIONS: list[dict[str, str]] = [
    # status-147: NCP鞍点系 + Coulomb摩擦 → 摩擦接線剛性の符号問題で発散
    {
        "contact_force": "NCPContactForceProcess",
        "friction": "CoulombReturnMappingProcess",
        "reason": "摩擦接線剛性の符号問題で鞍点系が不定値化（status-147）",
    },
]


def validate_strategy_combination(strategy_names: dict[str, str]) -> list[str]:
    """Strategy組み合わせの互換性チェック.

    Args:
        strategy_names: {軸名: クラス名} の辞書

    Returns:
        警告メッセージのリスト。空ならOK。
    """
    warnings_list: list[str] = []

    # ブラックリストチェック
    for incompat in INCOMPATIBLE_COMBINATIONS:
        match = all(strategy_names.get(k) == v for k, v in incompat.items() if k != "reason")
        if match:
            warnings_list.append(f"非互換: {incompat['reason']}")

    # ホワイトリストチェック
    is_verified = any(
        all(strategy_names.get(k) == v for k, v in verified.items())
        for verified in VERIFIED_COMBINATIONS
    )
    if not is_verified:
        warnings_list.append(
            f"未検証の組み合わせ: {strategy_names}。"
            "VERIFIED_COMBINATIONS に追加する前に収束テストを実行してください。"
        )

    return warnings_list
