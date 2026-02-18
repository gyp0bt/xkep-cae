"""接触ペア・接触状態のデータ構造.

Phase C0: ContactPair / ContactState と solver_hooks の骨格。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ContactStatus(Enum):
    """接触状態."""

    INACTIVE = 0  # 非接触
    ACTIVE = 1  # 法線接触（摩擦なし or stick）
    SLIDING = 2  # 滑り（slip）


@dataclass
class ContactState:
    """1接触点の状態変数.

    Attributes:
        s: セグメントA上の最近接パラメータ ∈ [0,1]
        t: セグメントB上の最近接パラメータ ∈ [0,1]
        gap: 法線方向ギャップ（g >= 0: 離間, g < 0: 貫通）
        normal: 法線ベクトル (3,)
        tangent1: 接線基底1 (3,)
        tangent2: 接線基底2 (3,)
        lambda_n: 法線方向 AL 乗数
        k_pen: ペナルティ剛性（法線）
        k_t: 接線ペナルティ剛性
        p_n: 法線反力（≥ 0）
        z_t: 接線履歴ベクトル (2,)（摩擦用）
        status: 接触状態
        stick: stick 状態フラグ
        dissipation: 散逸エネルギー増分
    """

    s: float = 0.0
    t: float = 0.0
    gap: float = 0.0
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tangent1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tangent2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lambda_n: float = 0.0
    k_pen: float = 0.0
    k_t: float = 0.0
    p_n: float = 0.0
    z_t: np.ndarray = field(default_factory=lambda: np.zeros(2))
    status: ContactStatus = ContactStatus.INACTIVE
    stick: bool = True
    dissipation: float = 0.0

    def copy(self) -> ContactState:
        """深いコピーを返す."""
        return ContactState(
            s=self.s,
            t=self.t,
            gap=self.gap,
            normal=self.normal.copy(),
            tangent1=self.tangent1.copy(),
            tangent2=self.tangent2.copy(),
            lambda_n=self.lambda_n,
            k_pen=self.k_pen,
            k_t=self.k_t,
            p_n=self.p_n,
            z_t=self.z_t.copy(),
            status=self.status,
            stick=self.stick,
            dissipation=self.dissipation,
        )


@dataclass
class ContactPair:
    """接触ペアの定義.

    セグメントA（梁要素A）とセグメントB（梁要素B）の接触ペア。

    Attributes:
        elem_a: 要素Aのインデックス
        elem_b: 要素Bのインデックス
        nodes_a: 要素Aの節点インデックス (2,)
        nodes_b: 要素Bの節点インデックス (2,)
        state: 接触状態
        radius_a: 要素Aの断面半径（接触検知用）
        radius_b: 要素Bの断面半径（接触検知用）
    """

    elem_a: int
    elem_b: int
    nodes_a: np.ndarray
    nodes_b: np.ndarray
    state: ContactState = field(default_factory=ContactState)
    radius_a: float = 0.0
    radius_b: float = 0.0

    @property
    def search_radius(self) -> float:
        """探索半径: 断面半径の和."""
        return self.radius_a + self.radius_b

    def is_active(self) -> bool:
        """接触が有効か."""
        return self.state.status != ContactStatus.INACTIVE


@dataclass
class ContactConfig:
    """接触解析の設定.

    Attributes:
        k_pen_scale: ペナルティ剛性のスケール（EA/L 基準）
        k_t_ratio: 接線ペナルティ / 法線ペナルティ比
        mu: 摩擦係数
        g_on: 接触活性化ギャップ閾値
        g_off: 接触非活性化ギャップ閾値（ヒステリシス: g_off > g_on）
        n_outer_max: Outer loop 最大反復回数
        tol_geometry: 最近接点の収束判定値
        use_friction: 摩擦の有無
        mu_ramp_steps: μランプのステップ数（0 = ランプなし）
    """

    k_pen_scale: float = 1.0
    k_t_ratio: float = 0.5
    mu: float = 0.3
    g_on: float = 0.0
    g_off: float = 1e-6
    n_outer_max: int = 5
    tol_geometry: float = 1e-6
    use_friction: bool = False
    mu_ramp_steps: int = 0


@dataclass
class ContactManager:
    """接触ペアの管理.

    全接触ペアの生成・探索・更新を管理する。

    Attributes:
        pairs: 接触ペアのリスト
        config: 接触設定
    """

    pairs: list[ContactPair] = field(default_factory=list)
    config: ContactConfig = field(default_factory=ContactConfig)

    @property
    def n_pairs(self) -> int:
        """ペア数."""
        return len(self.pairs)

    @property
    def n_active(self) -> int:
        """有効な接触ペア数."""
        return sum(1 for p in self.pairs if p.is_active())

    def add_pair(
        self,
        elem_a: int,
        elem_b: int,
        nodes_a: np.ndarray,
        nodes_b: np.ndarray,
        radius_a: float = 0.0,
        radius_b: float = 0.0,
    ) -> ContactPair:
        """接触ペアを追加する."""
        pair = ContactPair(
            elem_a=elem_a,
            elem_b=elem_b,
            nodes_a=np.asarray(nodes_a, dtype=int),
            nodes_b=np.asarray(nodes_b, dtype=int),
            radius_a=radius_a,
            radius_b=radius_b,
        )
        self.pairs.append(pair)
        return pair

    def reset_all(self) -> None:
        """全ペアの状態をリセットする."""
        for pair in self.pairs:
            pair.state = ContactState()

    def get_active_pairs(self) -> list[ContactPair]:
        """有効な接触ペアのリストを返す."""
        return [p for p in self.pairs if p.is_active()]
