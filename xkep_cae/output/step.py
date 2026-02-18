"""Step / Increment / Frame データモデル.

Abaqus に準じた解析ステップの階層構造を定義する。

階層:
    Step: 時間と境界条件を与えて解く1つの計算単位
    Increment: Newton-Raphson 反復の1収束分（時間増分の最小単位）
    Frame: フィールド出力のためのインクリメントスナップショット
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Step:
    """解析ステップの定義.

    Attributes:
        name: ステップ名
        total_time: ステップ全体の時間 [s]
        dt: 時間増分 [s]
        history_output: ヒストリ出力要求（None = 出力なし）
        field_output: フィールド出力要求（None = 出力なし）
    """

    name: str
    total_time: float
    dt: float
    history_output: HistoryOutputRequest | None = None
    field_output: FieldOutputRequest | None = None

    def __post_init__(self) -> None:
        if self.total_time <= 0:
            raise ValueError(f"total_time は正値: {self.total_time}")
        if self.dt <= 0:
            raise ValueError(f"dt は正値: {self.dt}")
        if self.dt > self.total_time:
            raise ValueError(f"dt ({self.dt}) は total_time ({self.total_time}) 以下")

    @property
    def n_increments(self) -> int:
        """インクリメント数（切り上げ）."""
        return int(np.ceil(self.total_time / self.dt))


@dataclass
class IncrementResult:
    """1インクリメントの計算結果.

    Attributes:
        increment_index: インクリメント番号（0始まり）
        time: 現在時刻 [s]（ステップ開始からの相対時刻）
        dt: 実際に使用した時間増分 [s]
        displacement: (ndof,) 変位ベクトル
        velocity: (ndof,) 速度ベクトル
        acceleration: (ndof,) 加速度ベクトル
        converged: 収束したか
        iterations: NR 反復回数
    """

    increment_index: int
    time: float
    dt: float
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    converged: bool = True
    iterations: int = 1


@dataclass
class Frame:
    """フィールド出力フレーム（スナップショット）.

    Attributes:
        frame_index: フレーム番号（0始まり）
        time: 時刻 [s]
        displacement: (ndof,) 変位ベクトル
        velocity: (ndof,) or None 速度ベクトル
        acceleration: (ndof,) or None 加速度ベクトル
        element_data: 要素データ辞書 {名前: ndarray}。
            スカラー: (n_elements,) ベクトル/テンソル: (n_elements, n_components)。
            例: {"stress_xx": (100,), "stress": (100, 3)}
    """

    frame_index: int
    time: float
    displacement: np.ndarray
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    element_data: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class StepResult:
    """ステップ全体の計算結果.

    Attributes:
        step: ステップ定義
        step_index: ステップ番号（0始まり）
        start_time: ステップの絶対開始時刻（全ステップ通しの時刻）
        increments: インクリメント結果のリスト
        frames: フィールド出力フレームのリスト
        history: ヒストリ出力データ {nset名: {変数名: ndarray}}
        history_times: ヒストリ出力の時刻配列
        converged: 全インクリメントが収束したか
    """

    step: Step
    step_index: int
    start_time: float = 0.0
    increments: list[IncrementResult] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    history: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    history_times: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: bool = True


# 循環インポート回避のため遅延インポート用
from xkep_cae.output.request import FieldOutputRequest, HistoryOutputRequest  # noqa: E402

__all__ = [
    "Step",
    "IncrementResult",
    "Frame",
    "StepResult",
]
