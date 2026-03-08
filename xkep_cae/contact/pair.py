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
    build_contact_frame_batch,
    closest_point_segments,
    closest_point_segments_batch,
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
    coating_compression: float = 0.0  # 被膜圧縮量 [m]（被膜接触モデル用）
    coating_compression_prev: float = 0.0  # 前ステップの被膜圧縮量 [m]（粘性項用）
    # Gauss point friction states (for line contact friction, Phase C6-L1b)
    gp_z_t: list[np.ndarray] | None = None
    gp_stick: list[bool] | None = None
    gp_q_trial_norm: list[float] | None = None

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
            coating_compression=self.coating_compression,
            coating_compression_prev=self.coating_compression_prev,
            gp_z_t=[z.copy() for z in self.gp_z_t] if self.gp_z_t is not None else None,
            gp_stick=list(self.gp_stick) if self.gp_stick is not None else None,
            gp_q_trial_norm=list(self.gp_q_trial_norm)
            if self.gp_q_trial_norm is not None
            else None,
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
    core_radius_a: float = 0.0  # 芯線半径（被膜なし）
    core_radius_b: float = 0.0  # 芯線半径（被膜なし）

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
        tol_penetration_ratio: 貫入許容比（search_radius = r_a + r_b 基準）。
            0.01 は 1% 許容を意味する。Outer loop で貫入がこの閾値を超えると
            ペナルティ剛性を自動増大する。0 で無効。
        penalty_growth_factor: 貫入超過時のペナルティ成長係数（> 1）
        k_pen_max: ペナルティ剛性の上限（条件数悪化防止）
        contact_tangent_mode: 接触接線剛性のシステム行列への組込み方式。
            - "full": K_total = K_T + K_c（標準。二次収束）
            - "structural_only": K_total = K_T（Uzawa型。接触力は残差にのみ反映）
            - "diagonal": K_total = K_T + diag(K_c)（対角近似。条件数改善）
            - "scaled": K_total = K_T + α·K_c（α = contact_tangent_scale）
        contact_tangent_scale: "scaled" モード時の K_c スケール係数 α ∈ (0,1]。
    """

    k_pen_scale: float = 0.1  # k_pen無次元スケール係数（材料剛性に対する比率）
    k_pen_mode: str = "beam_ei"  # "beam_ei" | "ea_l" （材料ベース自動推定モード）
    beam_E: float = 0.0  # 梁ヤング率 [応力単位]（k_pen自動推定に必須）
    beam_I: float = 0.0  # 梁代表断面二次モーメント [長さ⁴]（beam_ei モード用）
    beam_A: float = 0.0  # 梁断面積 [長さ²]（ea_l モード用）
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
    tol_penetration_ratio: float = 0.01
    penalty_growth_factor: float = 2.0
    k_pen_max: float = 1e12
    staged_activation_steps: int = 0  # 段階的アクティベーション（0=無効）
    elem_layer_map: dict[int, int] | None = None  # 要素→層インデックスのマッピング
    use_modified_newton: bool = False  # Modified Newton法（構造剛性再利用）
    modified_newton_refresh: int = 5  # K_T再計算間隔（反復数）
    contact_damping: float = 1.0  # 接触力under-relaxation係数（1.0=無緩和）
    k_pen_scaling: str = "linear"  # k_penのn_pairsスケーリング: "linear" | "sqrt"
    contact_tangent_mode: str = (
        "full"  # 接触接線モード: "full" | "structural_only" | "diagonal" | "scaled"
    )
    contact_tangent_scale: float = 1.0  # "scaled" モード時のK_cスケール係数（0 < α ≤ 1）
    al_relaxation: float = 1.0  # AL乗数更新の緩和係数 ω ∈ (0,1]。1.0で従来動作
    adaptive_omega: bool = False  # 適応的ωスケジュール（Outer loop内でωを段階的に増大）
    omega_min: float = 0.01  # adaptive_omega時の初期ω
    omega_max: float = 0.3  # adaptive_omega時の上限ω
    omega_growth: float = 2.0  # adaptive_omega時のOuter反復ごとの成長係数
    preserve_inactive_lambda: bool = False  # INACTIVEペアのlambda_n保持（sticky contact）
    linear_solver: str = "auto"  # 線形ソルバー: "direct" | "iterative" | "auto"
    iterative_tol: float = 1e-10  # GMRES収束判定（iterativeモード用）
    ilu_drop_tol: float = 1e-4  # ILU前処理のdrop tolerance
    gmres_dof_threshold: int = 2000  # DOFがこの閾値以上で自動的にGMRES（反復法）を使用
    no_deactivation_within_step: bool = (
        False  # ステップ内でのペア非活性化を禁止（活性セットチャタリング防止）
    )
    monolithic_geometry: bool = (
        False  # Inner NR内で幾何(s,t,normal)を毎反復更新（Outer/Inner分離を廃止）
    )
    line_contact: bool = False  # Line-to-line Gauss 積分の有効化（Phase C6-L1）
    n_gauss: int = 3  # Line contact の Gauss 積分点数（2-5）
    n_gauss_auto: bool = False  # セグメント角度に基づく Gauss 点数自動選択
    consistent_st_tangent: bool = False  # ∂(s,t)/∂u 一貫接線の有効化（Phase C6-L2）
    use_ncp: bool = False  # NCP Semi-smooth Newton の有効化（Phase C6-L3）
    ncp_type: str = "fb"  # NCP 関数の種類: "fb" (Fischer-Burmeister) | "min"
    ncp_reg: float = 1e-12  # FB 関数の正則化パラメータ
    ncp_block_preconditioner: bool = False  # NCP 鞍点系のブロック前処理付き GMRES（Phase C6-L4）
    exclude_same_layer: bool = False  # 同層素線間の接触ペアを除外（Phase S1: ~80%ペア削減）
    use_mortar: bool = False  # Mortar 離散化の有効化（Phase C6-L5）
    midpoint_prescreening: bool = True  # 中点距離プリスクリーニングの有効化
    prescreening_margin: float = 0.0  # プリスクリーニング追加マージン（0=自動推定）
    lambda_n_max_factor: float = (
        0.0  # λ_n上限: lambda_n_max_factor * k_pen * search_radius（0=無効）
    )
    # --- S3: 大規模NCP収束改善パラメータ ---
    augmented_threshold: int = 20  # n_active がこの閾値超で拡大系直接法
    saddle_regularization: float = 0.0  # 鞍点系(2,2)ブロックの正則化δ（0=自動推定）
    ncp_active_threshold: float = 0.0  # NCP活性セットのヒステリシス閾値
    lambda_relaxation: float = 1.0  # λ更新の under-relaxation 係数
    lambda_warmstart_neighbor: bool = False  # 近傍ペアからλ初期値を外挿（S3改良4）
    chattering_window: int = 0  # Active set 履歴ウィンドウ幅（0=無効、S3改良5）
    # --- S3改良6: 適応時間増分制御 ---
    adaptive_timestepping: bool = False  # 適応Δt制御の有効化
    dt_grow_factor: float = 1.5  # 収束良好時のステップ拡大係数
    dt_shrink_factor: float = 0.5  # 収束不良時のステップ縮小係数
    dt_grow_iter_threshold: int = 5  # この反復数以下で収束 → ステップ拡大
    dt_shrink_iter_threshold: int = 15  # この反復数以上で収束 → ステップ縮小
    dt_contact_change_threshold: float = 0.3  # active set変化率がこの閾値超 → ステップ縮小
    dt_min_fraction: float = 0.0  # 最小ステップ分率（0=自動: 1/(n_load_steps*16)）
    dt_max_fraction: float = 0.0  # 最大ステップ分率（0=自動: 4/n_load_steps）
    # --- S3改良7: AMG前処理 ---
    use_amg_preconditioner: bool = False  # PyAMG SA前処理の有効化（ILU代替）
    # --- S3改良8: k_pen continuation ---
    k_pen_continuation: bool = False  # k_penを段階的に増大（初期は低いk_penで開始）
    k_pen_continuation_start: float = 0.1  # 初期k_penスケール（k_pen * この値）
    k_pen_continuation_steps: int = 3  # 何ステップで目標k_penに到達するか
    # --- S3改良10: 残差スケーリング（対角スケーリング前処理） ---
    residual_scaling: bool = False  # 鞍点系の対角スケーリング前処理の有効化
    # --- S3改良11: 接触力ランプ（ステップ内の段階的接触力増大） ---
    contact_force_ramp: bool = False  # Newton反復初期の接触力ランプ有効化
    contact_force_ramp_iters: int = 5  # ランプが1.0に達するまでの反復数
    # --- S3改良12: 初期貫入処理 ---
    adjust_initial_penetration: bool = True  # True: 初期貫入を補正、False: エラー
    position_tolerance: float = 0.0  # 初期貫入の許容量 [m]。0=無制限（adjust時のみ有効）
    # --- 被膜接触モデル（status-137: gap_offset廃止 → 被膜層を陽にモデル化） ---
    coating_stiffness: float = 0.0  # 被膜接触剛性 [Pa/m]。0=被膜モデル無効（従来互換）
    # --- 被膜粘性減衰（status-140: Kelvin-Voigt粘性項） ---
    coating_damping: float = 0.0  # 被膜粘性減衰係数 [Pa·s/m]。0=減衰なし


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
        core_radius_a: float = 0.0,
        core_radius_b: float = 0.0,
    ) -> ContactPair:
        """接触ペアを追加する."""
        pair = ContactPair(
            elem_a=elem_a,
            elem_b=elem_b,
            nodes_a=np.asarray(nodes_a, dtype=int),
            nodes_b=np.asarray(nodes_b, dtype=int),
            radius_a=radius_a,
            radius_b=radius_b,
            core_radius_a=core_radius_a,
            core_radius_b=core_radius_b,
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
        core_radii: np.ndarray | float | None = None,
    ) -> list[tuple[int, int]]:
        """Broadphase で接触候補ペアを検出し pairs を更新する.

        既存ペアのうち候補に含まれないものは INACTIVE にする。
        新規候補は ContactPair として追加する。

        Args:
            node_coords: 節点座標 (n_nodes, 3)
            connectivity: 要素接続 (n_elems, 2) — 各行が [node_i, node_j]
            radii: 断面半径（被膜込み）。スカラーなら全要素共通。配列なら要素ごと
            margin: 探索マージン
            cell_size: 格子セルサイズ（None で自動推定）
            core_radii: 芯線半径（被膜なし）。None の場合 radii と同一（被膜なし）

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

        # core_radii ベクトル化
        if core_radii is None:
            cr_arr = r_arr.copy()
        elif np.isscalar(core_radii):
            cr_arr = np.full(n_elems, float(core_radii))
        else:
            cr_arr = np.asarray(core_radii, dtype=float)

        # セグメントリスト構築
        segments = []
        for e in range(n_elems):
            ni, nj = conn[e]
            segments.append((coords[ni], coords[nj]))

        # broadphase 実行
        raw_candidates = broadphase_aabb(
            segments,
            r_arr,
            margin=margin,
            cell_size=cell_size,
        )

        # 共有節点フィルタ: 同一梁内の隣接セグメントを除外
        # 2セグメントが節点を共有する場合は接触候補から除外する
        candidates = []
        lm = self.config.elem_layer_map
        exclude_same = self.config.exclude_same_layer and lm is not None
        for i, j in raw_candidates:
            nodes_i = set(int(n) for n in conn[i])
            nodes_j = set(int(n) for n in conn[j])
            if nodes_i & nodes_j:
                continue  # 共有節点あり → 同一梁の隣接セグメント
            # 同層除外フィルタ: 同じ層に属する要素ペアを除外
            if exclude_same:
                layer_i = lm.get(i, -1)
                layer_j = lm.get(j, -1)
                if layer_i == layer_j and layer_i >= 0:
                    continue
            candidates.append((i, j))

        # 中点距離プリスクリーニング（ベクトル化）
        # Broadphase AABB は軸整列で粗いため、3D 中点間距離でタイトに絞る
        if self.config.midpoint_prescreening and candidates:
            cand_arr = np.array(candidates, dtype=int)
            n0 = conn[cand_arr[:, 0], 0]  # 要素A 始点ノード
            n1 = conn[cand_arr[:, 0], 1]  # 要素A 終点ノード
            m0 = conn[cand_arr[:, 1], 0]  # 要素B 始点ノード
            m1 = conn[cand_arr[:, 1], 1]  # 要素B 終点ノード
            mid_a = 0.5 * (coords[n0] + coords[n1])  # (m, 3)
            mid_b = 0.5 * (coords[m0] + coords[m1])  # (m, 3)
            half_len_a = 0.5 * np.linalg.norm(coords[n1] - coords[n0], axis=1)
            half_len_b = 0.5 * np.linalg.norm(coords[m1] - coords[m0], axis=1)
            mid_dist = np.linalg.norm(mid_a - mid_b, axis=1)  # (m,)
            r_a = r_arr[cand_arr[:, 0]]
            r_b = r_arr[cand_arr[:, 1]]
            # 最小可能距離: 中点間距離 - 半長の和（セグメント端点が最も近い場合）
            min_possible_dist = np.maximum(0.0, mid_dist - half_len_a - half_len_b)
            # 接触カットオフ: 半径の和 + マージン
            extra_margin = self.config.prescreening_margin
            if extra_margin <= 0.0:
                extra_margin = float(np.mean(r_a + r_b)) * 0.5
            cutoff = r_a + r_b + extra_margin
            keep = min_possible_dist <= cutoff
            candidates = [candidates[k] for k in range(len(candidates)) if keep[k]]

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
                    core_radius_a=float(cr_arr[i]),
                    core_radius_b=float(cr_arr[j]),
                )

        return candidates

    def update_geometry(
        self,
        node_coords: np.ndarray,
        *,
        allow_deactivation: bool = True,
        freeze_active_set: bool = False,
    ) -> None:
        """全ペアの幾何情報を更新する（Narrowphase）.

        各ペアについて最近接点・ギャップ・接触フレームを再計算し、
        Active-set をヒステリシス付きで更新する。

        Args:
            node_coords: 節点座標 (n_nodes, 3)
            allow_deactivation: False でペアの非活性化を禁止（活性化のみ許可）。
                ステップ内でのチャタリング防止に使用。
            freeze_active_set: True で Active-set 更新を完全にスキップ。
                Inner NR 内のモノリシック幾何更新用。
        """
        coords = np.asarray(node_coords, dtype=float)
        n_pairs = len(self.pairs)

        if n_pairs == 0:
            return

        # --- バッチ版: 全ペアの端点を一括取得 ---
        nodes_a0 = np.array([p.nodes_a[0] for p in self.pairs], dtype=int)
        nodes_a1 = np.array([p.nodes_a[1] for p in self.pairs], dtype=int)
        nodes_b0 = np.array([p.nodes_b[0] for p in self.pairs], dtype=int)
        nodes_b1 = np.array([p.nodes_b[1] for p in self.pairs], dtype=int)

        xA0 = coords[nodes_a0]  # (N, 3)
        xA1 = coords[nodes_a1]  # (N, 3)
        xB0 = coords[nodes_b0]  # (N, 3)
        xB1 = coords[nodes_b1]  # (N, 3)

        # --- バッチ最近接点計算 ---
        s_all, t_all, _, _, dist_all, normal_all, _ = closest_point_segments_batch(
            xA0, xA1, xB0, xB1
        )

        # --- ギャップ計算 ---
        _use_coating = self.config.coating_stiffness > 0.0
        if _use_coating:
            # 被膜接触モデル: 芯線半径ベースのギャップ
            core_a = np.array([p.core_radius_a for p in self.pairs])
            core_b = np.array([p.core_radius_b for p in self.pairs])
            radii_a = np.array([p.radius_a for p in self.pairs])
            radii_b = np.array([p.radius_b for p in self.pairs])
            gap_core = dist_all - (core_a + core_b)
            coat_total = (radii_a - core_a) + (radii_b - core_b)
            # 被膜圧縮量: max(0, t_coat_total - gap_core)
            coat_comp = np.maximum(0.0, coat_total - gap_core)
            # NCPに渡すギャップ: gap = gap_core（芯線間距離）
            # NCP complementarity は gap_core ≥ 0 を制約
            gap_all = gap_core
        else:
            # 従来モデル: 被膜込み半径（被膜なし or 後方互換）
            radii_a = np.array([p.radius_a for p in self.pairs])
            radii_b = np.array([p.radius_b for p in self.pairs])
            gap_all = dist_all - (radii_a + radii_b)

        # --- バッチ接触フレーム計算 ---
        prev_t1_all = np.array([p.state.tangent1 for p in self.pairs])  # (N, 3)
        prev_n_all = np.array([p.state.normal for p in self.pairs])  # (N, 3)
        has_prev = np.sqrt(np.einsum("ij,ij->i", prev_t1_all, prev_t1_all)) > 1e-10
        has_prev_n = np.sqrt(np.einsum("ij,ij->i", prev_n_all, prev_n_all)) > 1e-10

        n_all, t1_all, t2_all = build_contact_frame_batch(
            normal_all,
            prev_tangent1s=prev_t1_all,
            prev_normals=prev_n_all,
            has_prev_mask=has_prev,
            has_prev_n_mask=has_prev_n,
        )

        # --- 結果を各ペアに書き戻し ---
        for i, pair in enumerate(self.pairs):
            pair.state.s = float(s_all[i])
            pair.state.t = float(t_all[i])
            pair.state.gap = float(gap_all[i])
            pair.state.normal = n_all[i]
            pair.state.tangent1 = t1_all[i]
            pair.state.tangent2 = t2_all[i]
            if _use_coating:
                pair.state.coating_compression = float(coat_comp[i])

            if not freeze_active_set:
                self._update_active_set(pair, allow_deactivation=allow_deactivation)

    def _update_active_set(
        self,
        pair: ContactPair,
        *,
        allow_deactivation: bool = True,
    ) -> None:
        """Active-set をヒステリシス付きで更新する.

        g_on / g_off のヒステリシスバンドにより、
        接触状態のチャタリングを防止する。

        - 非活性 → 活性: gap <= g_on（または被膜圧縮 > 0）
        - 活性 → 非活性: gap >= g_off  (g_off > g_on)

        allow_deactivation=False の場合、ACTIVE→INACTIVE 遷移を禁止し、
        ステップ内でのチャタリングを防止する。活性化のみ許可。
        """
        gap = pair.state.gap
        g_on = self.config.g_on
        g_off = self.config.g_off

        # 被膜モデル: 被膜圧縮があれば接触活性化
        _coat_active = self.config.coating_stiffness > 0.0 and pair.state.coating_compression > 0.0

        if pair.state.status == ContactStatus.INACTIVE:
            # 非活性 → 活性化判定
            if gap <= g_on or _coat_active:
                pair.state.status = ContactStatus.ACTIVE
        else:
            # 活性 → 非活性化判定
            if allow_deactivation and gap >= g_off and not _coat_active:
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

    def max_layer(self) -> int:
        """elem_layer_map の最大層番号を返す（未設定なら 0）."""
        lm = self.config.elem_layer_map
        if not lm:
            return 0
        return max(lm.values()) if lm else 0

    def compute_active_layer_for_step(self, step: int, n_load_steps: int) -> int:
        """段階的アクティベーション: 現在ステップで許容する最大層番号を計算.

        staged_activation_steps ステップをかけて層を段階的にオンにする。
        例: 3層構造で staged_activation_steps=6, n_load_steps=20 の場合:
          - step 1-2: layer 0 のみ（中心素線どうし）
          - step 3-4: layer 0-1
          - step 5-6: layer 0-2（全層）
          - step 7+: 全層

        Args:
            step: 現在の荷重ステップ（1-indexed）
            n_load_steps: 全荷重ステップ数

        Returns:
            最大許容層番号
        """
        n_staged = self.config.staged_activation_steps
        if n_staged <= 0:
            return self.max_layer()

        max_lay = self.max_layer()
        if max_lay <= 0:
            return 0

        # n_staged ステップで全層をオンにする
        # ステップ per 層 = n_staged / (max_lay + 1)
        steps_per_layer = max(1, n_staged // (max_lay + 1))
        current_max_layer = min(max_lay, (step - 1) // steps_per_layer)
        return current_max_layer

    def filter_pairs_by_layer(self, max_layer: int) -> None:
        """許容層より上の接触ペアを INACTIVE にする.

        elem_layer_map が設定されている場合、両方の要素の層番号が
        max_layer 以下であるペアのみをアクティブに維持する。

        Args:
            max_layer: 許容する最大層番号
        """
        lm = self.config.elem_layer_map
        if not lm:
            return

        for pair in self.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            layer_a = lm.get(pair.elem_a, 0)
            layer_b = lm.get(pair.elem_b, 0)
            # 接触は層間のペア。両方の層が max_layer 以内であること
            if layer_a > max_layer or layer_b > max_layer:
                pair.state.status = ContactStatus.INACTIVE

    def count_same_layer_pairs(self) -> int:
        """同層ペア数を返す（除外効果の事前評価用）.

        elem_layer_map が設定されている場合、両要素が同じ層に属する
        ACTIVE ペアの数を返す。

        Returns:
            同層ペア数
        """
        lm = self.config.elem_layer_map
        if not lm:
            return 0
        count = 0
        for pair in self.pairs:
            if pair.state.status == ContactStatus.INACTIVE:
                continue
            layer_a = lm.get(pair.elem_a, -1)
            layer_b = lm.get(pair.elem_b, -1)
            if layer_a == layer_b and layer_a >= 0:
                count += 1
        return count

    def check_initial_penetration(self, node_coords: np.ndarray) -> int:
        """初期貫入をチェックする.

        被膜モデル有効時: 芯線半径ベースでギャップを計算。
        被膜モデル無効時: 被膜込み半径（total radius）でギャップを計算。

        Args:
            node_coords: 初期節点座標 (n_nodes, 3)

        Returns:
            初期貫入ペア数
        """
        coords = np.asarray(node_coords, dtype=float)
        n_pen = 0
        _use_coating = self.config.coating_stiffness > 0.0

        for pair in self.pairs:
            xA0 = coords[pair.nodes_a[0]]
            xA1 = coords[pair.nodes_a[1]]
            xB0 = coords[pair.nodes_b[0]]
            xB1 = coords[pair.nodes_b[1]]

            result = closest_point_segments(xA0, xA1, xB0, xB1)
            if _use_coating:
                # 被膜モデル: 芯線半径ベースのギャップ
                gap = compute_gap(result.distance, pair.core_radius_a, pair.core_radius_b)
            else:
                # 従来モデル: 被膜込み半径ベース
                gap = compute_gap(result.distance, pair.radius_a, pair.radius_b)

            if gap < 0.0:
                n_pen += 1

        return n_pen

    def adjust_initial_positions(
        self,
        node_coords: np.ndarray,
        position_tolerance: float = 0.0,
        *,
        max_iterations: int = 10,
    ) -> tuple[np.ndarray, int, int]:
        """初期節点座標を調整して貫入解消・ギャップ閉鎖を行う.

        各接触ペアのギャップを検査し:
          - gap < 0（貫入）: 節点を法線方向に離してgap = 0にする
          - 0 <= gap < position_tolerance: 節点を法線方向に近づけてgap = 0にする

        調整は反復的に行い、ペア間の干渉を解消する。

        Args:
            node_coords: 初期節点座標 (n_nodes, 3)。**破壊的に変更される**。
            position_tolerance: ギャップ閉鎖の閾値 [m]。
                gap < tolerance のペアをgap=0に調整する。
            max_iterations: 調整反復の上限

        Returns:
            (adjusted_coords, n_penetration_fixed, n_gap_closed)
        """
        coords = np.asarray(node_coords, dtype=float)
        n_pen_fixed = 0
        n_gap_closed = 0

        for _iteration in range(max_iterations):
            # 各ノードの補正ベクトルを蓄積
            corrections = np.zeros_like(coords)
            correction_count = np.zeros(len(coords), dtype=int)
            any_adjusted = False

            for pair in self.pairs:
                nA0, nA1 = int(pair.nodes_a[0]), int(pair.nodes_a[1])
                nB0, nB1 = int(pair.nodes_b[0]), int(pair.nodes_b[1])
                xA0, xA1 = coords[nA0], coords[nA1]
                xB0, xB1 = coords[nB0], coords[nB1]

                result = closest_point_segments(xA0, xA1, xB0, xB1)
                gap = compute_gap(result.distance, pair.radius_a, pair.radius_b)

                if gap >= position_tolerance:
                    continue  # 調整不要

                # 調整量: gap = 0 にするための移動量
                # gap < 0: 貫入 → 離す（+方向に移動）
                # 0 <= gap < tol: 近接 → 近づける（-方向に移動）
                adjust_dist = -gap  # 正=離す、負=近づける
                half_adjust = adjust_dist / 2.0

                # 法線方向（AからBへ向かう方向）
                normal = result.normal
                if float(np.linalg.norm(normal)) < 1e-15:
                    continue

                # セグメントAを+法線方向に、Bを-法線方向に移動
                # 各セグメントの2節点に均等配分
                for nid in [nA0, nA1]:
                    corrections[nid] -= half_adjust * normal
                    correction_count[nid] += 1
                for nid in [nB0, nB1]:
                    corrections[nid] += half_adjust * normal
                    correction_count[nid] += 1

                any_adjusted = True
                if gap < 0:
                    n_pen_fixed += 1
                else:
                    n_gap_closed += 1

            if not any_adjusted:
                break

            # 補正を適用（複数ペアから補正を受けるノードは平均化）
            mask = correction_count > 0
            coords[mask] += corrections[mask] / correction_count[mask, np.newaxis]

        return coords, n_pen_fixed, n_gap_closed

    def compute_coating_forces(
        self,
        node_coords: np.ndarray,
        dt: float = 1.0,
    ) -> np.ndarray:
        """被膜圧縮による接触力ベクトルを計算する.

        Kelvin-Voigt被膜モデル: 弾性スプリング + 粘性ダッシュポット。
        被膜圧縮量 δ = max(0, t_coat_total - gap_core) に対し
        f_coat = k * δ + c * δ̇ の法線力を生成する。

        δ̇ ≈ (δ - δ_prev) / dt （後退差分近似）

        Args:
            node_coords: 節点座標 (n_nodes, 3)
            dt: 時間増分 [s]（粘性項の計算に使用）

        Returns:
            f_coat: 節点力ベクトル (ndof,)
        """
        k_coat = self.config.coating_stiffness
        c_coat = self.config.coating_damping
        coords = np.asarray(node_coords, dtype=float)
        n_nodes = len(coords)
        ndof_per_node = self.config.ndof_per_node if hasattr(self.config, "ndof_per_node") else 6
        f_coat = np.zeros(n_nodes * ndof_per_node)

        for pair in self.pairs:
            cc = pair.state.coating_compression
            if cc <= 0.0 and pair.state.coating_compression_prev <= 0.0:
                continue

            # Kelvin-Voigt: f = k * δ + c * δ̇
            f_n = k_coat * cc
            if c_coat > 0.0 and dt > 0.0:
                delta_dot = (cc - pair.state.coating_compression_prev) / dt
                f_n += c_coat * delta_dot

            if abs(f_n) < 1e-30:
                continue

            n_vec = pair.state.normal
            s = pair.state.s
            t = pair.state.t

            nA0 = int(pair.nodes_a[0])
            nA1 = int(pair.nodes_a[1])
            nB0 = int(pair.nodes_b[0])
            nB1 = int(pair.nodes_b[1])

            fA = -f_n * n_vec
            fB = f_n * n_vec

            dpn = ndof_per_node
            f_coat[nA0 * dpn : nA0 * dpn + 3] += (1.0 - s) * fA
            f_coat[nA1 * dpn : nA1 * dpn + 3] += s * fA
            f_coat[nB0 * dpn : nB0 * dpn + 3] += (1.0 - t) * fB
            f_coat[nB1 * dpn : nB1 * dpn + 3] += t * fB

        return f_coat

    def compute_coating_stiffness(
        self,
        node_coords: np.ndarray,
        ndof_total: int,
        dt: float = 1.0,
    ) -> np.ndarray:
        """被膜Kelvin-Voigtモデルの接線剛性行列を計算する.

        K_coat = (k_coat + c_coat/dt) * N^T (n ⊗ n) N

        弾性項 k_coat と粘性項 c_coat/dt の合算が接線剛性となる。
        後退Euler離散化により c*δ̇ ≈ c*(δ-δ_prev)/dt の δ に対する微分が c/dt。

        Args:
            node_coords: 節点座標 (n_nodes, 3)
            ndof_total: 全体自由度数
            dt: 時間増分 [s]（粘性項の接線剛性 c/dt に使用）

        Returns:
            K_coat: (ndof_total, ndof_total) CSR形式被膜剛性行列
        """
        import scipy.sparse as sp

        k_coat = self.config.coating_stiffness
        c_coat = self.config.coating_damping
        # 実効接線剛性: 弾性 + 粘性（後退Euler）
        k_eff = k_coat
        if c_coat > 0.0 and dt > 0.0:
            k_eff += c_coat / dt
        ndof_per_node = self.config.ndof_per_node if hasattr(self.config, "ndof_per_node") else 6

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for pair in self.pairs:
            cc = pair.state.coating_compression
            if cc <= 1e-15:
                continue

            n_vec = pair.state.normal
            s = pair.state.s
            t = pair.state.t

            nA0 = int(pair.nodes_a[0])
            nA1 = int(pair.nodes_a[1])
            nB0 = int(pair.nodes_b[0])
            nB1 = int(pair.nodes_b[1])

            weights = [-(1.0 - s), -s, (1.0 - t), t]
            node_ids = [nA0, nA1, nB0, nB1]

            gdofs = []
            g_n = np.zeros(12)
            for idx, (w, nid) in enumerate(zip(weights, node_ids, strict=True)):
                base = nid * ndof_per_node
                gdofs.extend([base, base + 1, base + 2])
                g_n[3 * idx : 3 * idx + 3] = w * n_vec

            K_local = k_eff * np.outer(g_n, g_n)

            for i_loc in range(12):
                for j_loc in range(12):
                    val = K_local[i_loc, j_loc]
                    if abs(val) > 1e-30:
                        rows.append(gdofs[i_loc])
                        cols.append(gdofs[j_loc])
                        data.append(val)

        if not data:
            return sp.csr_matrix((ndof_total, ndof_total))

        return sp.coo_matrix((data, (rows, cols)), shape=(ndof_total, ndof_total)).tocsr()

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
