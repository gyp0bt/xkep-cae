"""接触ペア・接触状態のデータ構造.

Phase C0: ContactPair / ContactState と solver_hooks の骨格。
Phase C1: broadphase候補探索 + 幾何更新 + Active-setヒステリシス。
Phase C2: 法線AL接触力評価 + 乗数更新 + ペナルティ初期化。
Phase C5: q_trial_norm 追加 + use_geometric_stiffness / use_pdas 設定追加。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from xkep_cae.contact.broadphase import broadphase_aabb
from xkep_cae.contact.geometry import (
    build_contact_frame,
    closest_point_segments,
    compute_gap,
)


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
        q_trial_norm: 摩擦 trial force ノルム（slip consistent tangent 用）
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
    q_trial_norm: float = 0.0
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
            q_trial_norm=self.q_trial_norm,
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
        use_line_search: merit line search の有効化
        line_search_max_steps: backtracking の最大縮小回数
        merit_alpha: merit function の貫通ペナルティ重み
        merit_beta: merit function の散逸ペナルティ重み
        use_geometric_stiffness: 幾何微分込み一貫接線の有効化
        use_pdas: PDAS Active-set 更新の有効化（実験的）
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
    use_line_search: bool = False
    line_search_max_steps: int = 5
    merit_alpha: float = 1.0
    merit_beta: float = 1.0
    use_geometric_stiffness: bool = True
    use_pdas: bool = False


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

    # -- Phase C1: broadphase + 幾何更新 + Active-set ---------

    def detect_candidates(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        radii: np.ndarray | float = 0.0,
        *,
        margin: float = 0.0,
        cell_size: float | None = None,
    ) -> list[tuple[int, int]]:
        """Broadphase で接触候補ペアを検出し pairs を更新する.

        既存ペアのうち候補に含まれないものは INACTIVE にする。
        新規候補は ContactPair として追加する。

        Args:
            node_coords: 節点座標 (n_nodes, 3)
            connectivity: 要素接続 (n_elems, 2) — 各行が [node_i, node_j]
            radii: 断面半径。スカラーなら全要素共通。配列なら要素ごと
            margin: 探索マージン
            cell_size: 格子セルサイズ（None で自動推定）

        Returns:
            候補ペアのリスト (elem_a, elem_b)
        """
        conn = np.asarray(connectivity, dtype=int)
        coords = np.asarray(node_coords, dtype=float)
        n_elems = len(conn)

        # radii ベクトル化
        if np.isscalar(radii):
            r_arr = np.full(n_elems, float(radii))
        else:
            r_arr = np.asarray(radii, dtype=float)

        # セグメントリスト構築
        segments = []
        for e in range(n_elems):
            ni, nj = conn[e]
            segments.append((coords[ni], coords[nj]))

        # broadphase 実行
        candidates = broadphase_aabb(
            segments,
            r_arr,
            margin=margin,
            cell_size=cell_size,
        )

        # 既存ペアをマップ化（(elem_a, elem_b) → index）
        existing: dict[tuple[int, int], int] = {}
        for idx, p in enumerate(self.pairs):
            key = (min(p.elem_a, p.elem_b), max(p.elem_a, p.elem_b))
            existing[key] = idx

        # 候補に含まれない既存ペアを INACTIVE にする
        candidate_set = set(candidates)
        for key, idx in existing.items():
            if key not in candidate_set:
                self.pairs[idx].state.status = ContactStatus.INACTIVE

        # 新規候補を追加
        for i, j in candidates:
            key = (min(i, j), max(i, j))
            if key not in existing:
                self.add_pair(
                    elem_a=i,
                    elem_b=j,
                    nodes_a=conn[i],
                    nodes_b=conn[j],
                    radius_a=float(r_arr[i]),
                    radius_b=float(r_arr[j]),
                )

        return candidates

    def update_geometry(
        self,
        node_coords: np.ndarray,
    ) -> None:
        """全ペアの幾何情報を更新する（Narrowphase）.

        各ペアについて最近接点・ギャップ・接触フレームを再計算し、
        Active-set をヒステリシス付きで更新する。

        Args:
            node_coords: 節点座標 (n_nodes, 3)
        """
        coords = np.asarray(node_coords, dtype=float)

        for pair in self.pairs:
            # セグメント端点を取得
            xA0 = coords[pair.nodes_a[0]]
            xA1 = coords[pair.nodes_a[1]]
            xB0 = coords[pair.nodes_b[0]]
            xB1 = coords[pair.nodes_b[1]]

            # 最近接点計算
            result = closest_point_segments(xA0, xA1, xB0, xB1)

            # ギャップ計算
            gap = compute_gap(result.distance, pair.radius_a, pair.radius_b)

            # 接触フレーム更新（法線履歴 + 平行輸送で連続性を保持）
            prev_t1 = pair.state.tangent1
            prev_n = pair.state.normal
            has_prev = float(np.linalg.norm(prev_t1)) > 1e-10
            has_prev_n = float(np.linalg.norm(prev_n)) > 1e-10
            n, t1, t2 = build_contact_frame(
                result.normal,
                prev_tangent1=prev_t1 if has_prev else None,
                prev_normal=prev_n if has_prev_n else None,
            )

            # 状態を更新
            pair.state.s = result.s
            pair.state.t = result.t
            pair.state.gap = gap
            pair.state.normal = n
            pair.state.tangent1 = t1
            pair.state.tangent2 = t2

            # Active-set ヒステリシス更新
            self._update_active_set(pair)

    def _update_active_set(self, pair: ContactPair) -> None:
        """Active-set をヒステリシス付きで更新する.

        g_on / g_off のヒステリシスバンドにより、
        接触状態のチャタリングを防止する。

        - 非活性 → 活性: gap <= g_on
        - 活性 → 非活性: gap >= g_off  (g_off > g_on)
        """
        gap = pair.state.gap
        g_on = self.config.g_on
        g_off = self.config.g_off

        if pair.state.status == ContactStatus.INACTIVE:
            # 非活性 → 活性化判定
            if gap <= g_on:
                pair.state.status = ContactStatus.ACTIVE
        else:
            # 活性 → 非活性化判定
            if gap >= g_off:
                pair.state.status = ContactStatus.INACTIVE

    # -- Phase C2: 法線AL接触力 + 乗数更新 + ペナルティ初期化 ----

    def evaluate_contact_forces(self) -> None:
        """全 ACTIVE ペアの法線接触反力 p_n を評価する.

        AL: p_n = max(0, lambda_n + k_pen * (-g))
        """
        from xkep_cae.contact.law_normal import evaluate_normal_force

        for pair in self.pairs:
            evaluate_normal_force(pair)

    def update_al_multipliers(self) -> None:
        """全ペアの AL 乗数を更新する（Outer loop 終了時）."""
        from xkep_cae.contact.law_normal import update_al_multiplier

        for pair in self.pairs:
            update_al_multiplier(pair)

    def initialize_penalty(self, k_pen: float, k_t_ratio: float | None = None) -> None:
        """全 ACTIVE ペアのペナルティ剛性を初期化する.

        Args:
            k_pen: 法線ペナルティ剛性
            k_t_ratio: 接線/法線比（None なら config から取得）
        """
        from xkep_cae.contact.law_normal import initialize_penalty_stiffness

        ratio = k_t_ratio if k_t_ratio is not None else self.config.k_t_ratio
        for pair in self.pairs:
            if pair.state.status != ContactStatus.INACTIVE and pair.state.k_pen <= 0.0:
                initialize_penalty_stiffness(pair, k_pen, ratio)
