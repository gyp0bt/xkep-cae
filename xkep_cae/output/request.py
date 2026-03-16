"""出力要求（Output Request）データモデル.

Abaqus の *OUTPUT, HISTORY / *OUTPUT, FIELD キーワードに相当する
出力要求を定義する。

HistoryOutputRequest: 時系列プロファイル出力（高頻度・特定節点）
FieldOutputRequest: 空間分布スナップショット出力（低頻度・全節点）
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 対応する出力変数
NODAL_VARIABLES = {"U", "V", "A", "RF", "CF"}
ENERGY_VARIABLES = {"ALLIE", "ALLKE"}
ALL_VARIABLES = NODAL_VARIABLES | ENERGY_VARIABLES


@dataclass
class HistoryOutputRequest:
    """ヒストリ出力要求.

    特定の節点集合に対して、指定時間間隔で変数の時系列を記録する。
    Abaqus の ``*OUTPUT, HISTORY`` + ``*NODE OUTPUT, NSET=...`` に相当。

    Attributes:
        dt: 出力時間間隔 [s]
        variables: 出力変数リスト (例: ["RF", "U", "ALLIE", "ALLKE"])
        node_sets: 節点集合 {名前: 節点インデックス配列}。
            エネルギー変数（ALLIE, ALLKE）は node_set に依存しないグローバル量。
    """

    dt: float
    variables: list[str] = field(default_factory=lambda: ["U"])
    node_sets: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt は正値: {self.dt}")
        for var in self.variables:
            if var not in ALL_VARIABLES:
                raise ValueError(f"未対応の出力変数: {var}（対応: {ALL_VARIABLES}）")


@dataclass
class FieldOutputRequest:
    """フィールド出力要求.

    ステップ全体を等分割してスナップショットを記録する。
    Abaqus の ``*OUTPUT, FIELD, NUMBER INTERVAL=N`` に相当。

    Attributes:
        num: ステップ内のフレーム数（等分割）
        variables: 出力変数リスト (例: ["U", "V", "A"])
        node_sets: 出力対象の節点集合（None = 全節点）
    """

    num: int
    variables: list[str] = field(default_factory=lambda: ["U"])
    node_sets: dict[str, list[int]] | None = None

    def __post_init__(self) -> None:
        if self.num < 1:
            raise ValueError(f"num は1以上: {self.num}")
        for var in self.variables:
            if var not in ALL_VARIABLES:
                raise ValueError(f"未対応の出力変数: {var}（対応: {ALL_VARIABLES}）")


__all__ = [
    "HistoryOutputRequest",
    "FieldOutputRequest",
    "NODAL_VARIABLES",
    "ENERGY_VARIABLES",
    "ALL_VARIABLES",
]
