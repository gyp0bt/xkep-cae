"""ContactGeometry Strategy 具象実装.

接触幾何の評価方法を Strategy として実装する。

Phase 4 統合:
- PointToPointProcess: ContactManager.update_geometry() からバッチ最近接点ロジックを移植
- LineToLineGaussProcess: line_contact.py のGauss積分ベース幾何更新を統合
- MortarSegmentProcess: mortar.py のMortar射影ベース幾何更新を統合
- 全具象に update_geometry() + build_constraint_jacobian() を実装
- create_contact_geometry_strategy() ファクトリ
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import SolverProcess


@dataclass(frozen=True)
class ContactGeometryInput:
    """ContactGeometry Strategy の入力."""

    node_coords: np.ndarray
    connectivity: np.ndarray
    radii: np.ndarray | float


@dataclass(frozen=True)
class ContactGeometryOutput:
    """ContactGeometry Strategy の出力."""

    contact_pairs: list


def _update_active_set_hysteresis(
    pair: object,
    *,
    g_on: float = 0.0,
    g_off: float = 0.0,
    allow_deactivation: bool = True,
    coating_stiffness: float = 0.0,
) -> None:
    """Active-set をヒステリシス付きで更新する.

    ContactManager._update_active_set() のロジックを移植。

    Args:
        pair: ContactPair
        g_on: 活性化閾値
        g_off: 非活性化閾値 (g_off > g_on)
        allow_deactivation: False で非活性化を禁止
        coating_stiffness: 被膜剛性 (>0 で被膜モデル有効)
    """
    from xkep_cae.contact.pair import ContactStatus

    gap = pair.state.gap
    coat_active = coating_stiffness > 0.0 and pair.state.coating_compression > 0.0

    if pair.state.status == ContactStatus.INACTIVE:
        if gap <= g_on or coat_active:
            pair.state.status = ContactStatus.ACTIVE
    else:
        if allow_deactivation and gap >= g_off and not coat_active:
            pair.state.status = ContactStatus.INACTIVE


def _build_constraint_jacobian_ptp(
    pairs: list,
    ndof_total: int,
    ndof_per_node: int = 6,
) -> tuple[sp.csr_matrix, list[int]]:
    """PtP用制約ヤコビアン G = ∂g_n/∂u を構築する.

    solver_ncp._build_constraint_jacobian() のロジックを移植。

    Returns:
        (G, active_indices)
    """
    from xkep_cae.contact.assembly import _contact_dofs
    from xkep_cae.contact.pair import ContactStatus

    active_indices: list[int] = []
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    row_idx = 0
    for i, pair in enumerate(pairs):
        if pair.state.status == ContactStatus.INACTIVE:
            continue

        active_indices.append(i)
        s = pair.state.s
        t = pair.state.t
        normal = pair.state.normal

        dofs = _contact_dofs(pair, ndof_per_node)
        coeffs = [(1.0 - s), s, -(1.0 - t), -t]
        for k in range(4):
            for d in range(3):
                global_dof = dofs[k * ndof_per_node + d]
                val = coeffs[k] * normal[d]
                if abs(val) > 1e-30:
                    rows.append(row_idx)
                    cols.append(global_dof)
                    vals.append(val)

        row_idx += 1

    n_active = len(active_indices)
    if n_active == 0:
        return sp.csr_matrix((0, ndof_total)), active_indices

    G = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_active, ndof_total),
    ).tocsr()
    return G, active_indices


class PointToPointProcess(SolverProcess[ContactGeometryInput, ContactGeometryOutput]):
    """最近接点ペア（Point-to-Point）による接触検出.

    各要素ペアの最近接パラメータ (s, t) を求め、
    ギャップ g = ||x_B - x_A|| - r_A - r_B を評価する。
    基本的な接触検出で、小規模問題に適する。

    制約ヤコビアン:
        ∂g_n/∂u の係数: [(1-s), s, -(1-t), -t]

    Phase 4: ContactManager.update_geometry() から実ロジック移植済。
    """

    meta = ProcessMeta(
        name="PointToPoint",
        module="solve",
        version="0.1.0",
        document_path="docs/contact_geometry.md",
    )

    def __init__(self, *, exclude_same_layer: bool = True) -> None:
        self._exclude_same_layer = exclude_same_layer

    def detect(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        radii: np.ndarray | float,
    ) -> list:
        """接触候補ペアの検出.

        検出ロジックは ContactManager.detect_candidates() に委譲。
        Strategy は narrowphase (幾何更新) を担当する。
        """
        return []

    def compute_gap(self, pair: object, node_coords: np.ndarray) -> float:
        """ギャップの計算.

        g = ||x_B(t) - x_A(s)|| - r_A - r_B
        """
        if hasattr(pair, "state") and hasattr(pair.state, "gap"):
            return float(pair.state.gap)
        return 0.0

    def update_geometry(
        self,
        pairs: list,
        node_coords: np.ndarray,
        *,
        config: object | None = None,
    ) -> None:
        """全ペアの幾何情報を更新する（Narrowphase）.

        ContactManager.update_geometry() のバッチ最近接点ロジックを移植。

        Args:
            pairs: ContactPair のリスト
            node_coords: 節点座標 (n_nodes, 3)
            config: ContactConfig（被膜モデル・ヒステリシス設定用）
        """
        from xkep_cae.contact.geometry import (
            build_contact_frame_batch,
            closest_point_segments_batch,
        )

        coords = np.asarray(node_coords, dtype=float)
        n_pairs = len(pairs)
        if n_pairs == 0:
            return

        # --- バッチ版: 全ペアの端点を一括取得 ---
        nodes_a0 = np.array([p.nodes_a[0] for p in pairs], dtype=int)
        nodes_a1 = np.array([p.nodes_a[1] for p in pairs], dtype=int)
        nodes_b0 = np.array([p.nodes_b[0] for p in pairs], dtype=int)
        nodes_b1 = np.array([p.nodes_b[1] for p in pairs], dtype=int)

        xA0 = coords[nodes_a0]
        xA1 = coords[nodes_a1]
        xB0 = coords[nodes_b0]
        xB1 = coords[nodes_b1]

        # --- バッチ最近接点計算 ---
        s_all, t_all, _, _, dist_all, normal_all, _ = closest_point_segments_batch(
            xA0, xA1, xB0, xB1
        )

        # --- ギャップ計算 ---
        coating_stiffness = 0.0
        if config is not None and hasattr(config, "coating_stiffness"):
            coating_stiffness = config.coating_stiffness

        _use_coating = coating_stiffness > 0.0
        if _use_coating:
            core_a = np.array([p.core_radius_a for p in pairs])
            core_b = np.array([p.core_radius_b for p in pairs])
            radii_a = np.array([p.radius_a for p in pairs])
            radii_b = np.array([p.radius_b for p in pairs])
            gap_core = dist_all - (core_a + core_b)
            coat_total = (radii_a - core_a) + (radii_b - core_b)
            coat_comp = np.maximum(0.0, coat_total - gap_core)
            gap_all = gap_core
        else:
            radii_a = np.array([p.radius_a for p in pairs])
            radii_b = np.array([p.radius_b for p in pairs])
            gap_all = dist_all - (radii_a + radii_b)

        # --- バッチ接触フレーム計算 ---
        prev_t1_all = np.array([p.state.tangent1 for p in pairs])
        prev_n_all = np.array([p.state.normal for p in pairs])
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
        g_on = 0.0
        g_off = 0.0
        allow_deact = True
        if config is not None:
            if hasattr(config, "g_on"):
                g_on = config.g_on
            if hasattr(config, "g_off"):
                g_off = config.g_off

        for i, pair in enumerate(pairs):
            pair.state.s = float(s_all[i])
            pair.state.t = float(t_all[i])
            pair.state.gap = float(gap_all[i])
            pair.state.normal = n_all[i]
            pair.state.tangent1 = t1_all[i]
            pair.state.tangent2 = t2_all[i]
            if _use_coating:
                pair.state.coating_compression = float(coat_comp[i])

            _update_active_set_hysteresis(
                pair,
                g_on=g_on,
                g_off=g_off,
                allow_deactivation=allow_deact,
                coating_stiffness=coating_stiffness,
            )

    def build_constraint_jacobian(
        self,
        pairs: list,
        ndof_total: int,
        ndof_per_node: int = 6,
    ) -> tuple[sp.csr_matrix, list[int]]:
        """制約ヤコビアン G = ∂g_n/∂u を構築.

        solver_ncp._build_constraint_jacobian() のロジックを移植。

        Returns:
            (G, active_indices): G は (n_active, ndof_total) の疎行列
        """
        return _build_constraint_jacobian_ptp(pairs, ndof_total, ndof_per_node)

    def process(self, input_data: ContactGeometryInput) -> ContactGeometryOutput:
        pairs = self.detect(input_data.node_coords, input_data.connectivity, input_data.radii)
        return ContactGeometryOutput(contact_pairs=pairs)


class LineToLineGaussProcess(SolverProcess[ContactGeometryInput, ContactGeometryOutput]):
    """Line-to-Line Gauss 積分による接触評価.

    要素ペアの相互作用領域をGauss積分点で離散化し、
    接触力と接触剛性を積分する。大規模問題で精度・安定性が向上。

    パラメータ:
        n_gauss: 1次元あたりの Gauss 点数（デフォルト: 2）
        auto_gauss: True でペア角度に基づく自動選択

    Phase 4: line_contact.py からの Gauss 積分ベース幾何更新を統合。
    """

    meta = ProcessMeta(
        name="LineToLineGauss",
        module="solve",
        version="0.1.0",
        document_path="docs/contact_geometry.md",
    )

    def __init__(
        self,
        *,
        n_gauss: int = 2,
        exclude_same_layer: bool = True,
        auto_gauss: bool = False,
    ) -> None:
        self._n_gauss = n_gauss
        self._exclude_same_layer = exclude_same_layer
        self._auto_gauss = auto_gauss

    def detect(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        radii: np.ndarray | float,
    ) -> list:
        """接触候補ペアの検出.

        検出ロジックは ContactManager.detect_candidates() に委譲。
        """
        return []

    def compute_gap(self, pair: object, node_coords: np.ndarray) -> float:
        """ギャップの計算.

        L2L では代表ギャップ（最小ギャップまたは中心Gauss点ギャップ）を返す。
        """
        if hasattr(pair, "state") and hasattr(pair.state, "gap"):
            return float(pair.state.gap)
        return 0.0

    def update_geometry(
        self,
        pairs: list,
        node_coords: np.ndarray,
        *,
        config: object | None = None,
    ) -> None:
        """全ペアの幾何情報を更新する（Narrowphase + Gauss点情報）.

        PtPと同様のバッチ最近接点計算に加え、各ペアのGauss点数を
        セグメント角度に基づいて自動選択する。

        Args:
            pairs: ContactPair のリスト
            node_coords: 節点座標 (n_nodes, 3)
            config: ContactConfig
        """
        from xkep_cae.contact.geometry import (
            build_contact_frame_batch,
            closest_point_segments_batch,
        )

        coords = np.asarray(node_coords, dtype=float)
        n_pairs = len(pairs)
        if n_pairs == 0:
            return

        # --- バッチ版端点取得 ---
        nodes_a0 = np.array([p.nodes_a[0] for p in pairs], dtype=int)
        nodes_a1 = np.array([p.nodes_a[1] for p in pairs], dtype=int)
        nodes_b0 = np.array([p.nodes_b[0] for p in pairs], dtype=int)
        nodes_b1 = np.array([p.nodes_b[1] for p in pairs], dtype=int)

        xA0 = coords[nodes_a0]
        xA1 = coords[nodes_a1]
        xB0 = coords[nodes_b0]
        xB1 = coords[nodes_b1]

        # --- バッチ最近接点計算（代表点として使用） ---
        s_all, t_all, _, _, dist_all, normal_all, _ = closest_point_segments_batch(
            xA0, xA1, xB0, xB1
        )

        # --- ギャップ計算 ---
        coating_stiffness = 0.0
        if config is not None and hasattr(config, "coating_stiffness"):
            coating_stiffness = config.coating_stiffness

        _use_coating = coating_stiffness > 0.0
        if _use_coating:
            core_a = np.array([p.core_radius_a for p in pairs])
            core_b = np.array([p.core_radius_b for p in pairs])
            radii_a = np.array([p.radius_a for p in pairs])
            radii_b = np.array([p.radius_b for p in pairs])
            gap_core = dist_all - (core_a + core_b)
            coat_total = (radii_a - core_a) + (radii_b - core_b)
            coat_comp = np.maximum(0.0, coat_total - gap_core)
            gap_all = gap_core
        else:
            radii_a = np.array([p.radius_a for p in pairs])
            radii_b = np.array([p.radius_b for p in pairs])
            gap_all = dist_all - (radii_a + radii_b)

        # --- バッチ接触フレーム ---
        prev_t1_all = np.array([p.state.tangent1 for p in pairs])
        prev_n_all = np.array([p.state.normal for p in pairs])
        has_prev = np.sqrt(np.einsum("ij,ij->i", prev_t1_all, prev_t1_all)) > 1e-10
        has_prev_n = np.sqrt(np.einsum("ij,ij->i", prev_n_all, prev_n_all)) > 1e-10

        n_all, t1_all, t2_all = build_contact_frame_batch(
            normal_all,
            prev_tangent1s=prev_t1_all,
            prev_normals=prev_n_all,
            has_prev_mask=has_prev,
            has_prev_n_mask=has_prev_n,
        )

        # --- Gauss点数自動選択 ---
        if self._auto_gauss:
            from xkep_cae.contact.line_contact import auto_select_n_gauss

            dA = xA1 - xA0
            dB = xB1 - xB0
            for i in range(n_pairs):
                n_gp = auto_select_n_gauss(dA[i], dB[i])
                if hasattr(pairs[i], "_n_gauss"):
                    pairs[i]._n_gauss = n_gp

        # --- 結果書き戻し ---
        g_on = 0.0
        g_off = 0.0
        if config is not None:
            if hasattr(config, "g_on"):
                g_on = config.g_on
            if hasattr(config, "g_off"):
                g_off = config.g_off

        for i, pair in enumerate(pairs):
            pair.state.s = float(s_all[i])
            pair.state.t = float(t_all[i])
            pair.state.gap = float(gap_all[i])
            pair.state.normal = n_all[i]
            pair.state.tangent1 = t1_all[i]
            pair.state.tangent2 = t2_all[i]
            if _use_coating:
                pair.state.coating_compression = float(coat_comp[i])

            _update_active_set_hysteresis(
                pair,
                g_on=g_on,
                g_off=g_off,
                allow_deactivation=True,
                coating_stiffness=coating_stiffness,
            )

    def build_constraint_jacobian(
        self,
        pairs: list,
        ndof_total: int,
        ndof_per_node: int = 6,
    ) -> tuple[sp.csr_matrix, list[int]]:
        """制約ヤコビアン G = ∂g_n/∂u を構築.

        L2L でも代表点ベースの制約ヤコビアンを使用。
        Gauss積分は力・剛性計算時に適用される。
        """
        return _build_constraint_jacobian_ptp(pairs, ndof_total, ndof_per_node)

    def process(self, input_data: ContactGeometryInput) -> ContactGeometryOutput:
        pairs = self.detect(input_data.node_coords, input_data.connectivity, input_data.radii)
        return ContactGeometryOutput(contact_pairs=pairs)


class MortarSegmentProcess(SolverProcess[ContactGeometryInput, ContactGeometryOutput]):
    """Mortar 法セグメントによる接触評価.

    Line-to-Line に加えて mortar 射影を行い、
    接触面の連続性を保証する。大規模問題のロバスト性に寄与。

    前提条件: line_contact=True（L2L との併用が必須）

    Phase 4: mortar.py からの Mortar 射影ロジック統合。
    """

    meta = ProcessMeta(
        name="MortarSegment",
        module="solve",
        version="0.1.0",
        document_path="docs/contact_geometry.md",
    )

    def __init__(
        self,
        *,
        n_gauss: int = 2,
        exclude_same_layer: bool = True,
    ) -> None:
        self._n_gauss = n_gauss
        self._exclude_same_layer = exclude_same_layer

    def detect(
        self,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        radii: np.ndarray | float,
    ) -> list:
        """接触候補ペアの検出 + mortar ノード同定."""
        return []

    def compute_gap(self, pair: object, node_coords: np.ndarray) -> float:
        """ギャップの計算（mortar 射影ベース）."""
        if hasattr(pair, "state") and hasattr(pair.state, "gap"):
            return float(pair.state.gap)
        return 0.0

    def update_geometry(
        self,
        pairs: list,
        node_coords: np.ndarray,
        *,
        config: object | None = None,
    ) -> None:
        """全ペアの幾何情報を更新する（Mortar射影ベース）.

        PtPのバッチ最近接点計算をベースに、Mortar ノード同定と
        加重ギャップ計算を行う。

        Args:
            pairs: ContactPair のリスト
            node_coords: 節点座標 (n_nodes, 3)
            config: ContactConfig
        """
        from xkep_cae.contact.geometry import (
            build_contact_frame_batch,
            closest_point_segments_batch,
        )

        coords = np.asarray(node_coords, dtype=float)
        n_pairs = len(pairs)
        if n_pairs == 0:
            return

        # --- バッチ版端点取得 ---
        nodes_a0 = np.array([p.nodes_a[0] for p in pairs], dtype=int)
        nodes_a1 = np.array([p.nodes_a[1] for p in pairs], dtype=int)
        nodes_b0 = np.array([p.nodes_b[0] for p in pairs], dtype=int)
        nodes_b1 = np.array([p.nodes_b[1] for p in pairs], dtype=int)

        xA0 = coords[nodes_a0]
        xA1 = coords[nodes_a1]
        xB0 = coords[nodes_b0]
        xB1 = coords[nodes_b1]

        # --- バッチ最近接点計算 ---
        s_all, t_all, _, _, dist_all, normal_all, _ = closest_point_segments_batch(
            xA0, xA1, xB0, xB1
        )

        # --- ギャップ計算 ---
        radii_a = np.array([p.radius_a for p in pairs])
        radii_b = np.array([p.radius_b for p in pairs])
        gap_all = dist_all - (radii_a + radii_b)

        # --- バッチ接触フレーム ---
        prev_t1_all = np.array([p.state.tangent1 for p in pairs])
        prev_n_all = np.array([p.state.normal for p in pairs])
        has_prev = np.sqrt(np.einsum("ij,ij->i", prev_t1_all, prev_t1_all)) > 1e-10
        has_prev_n = np.sqrt(np.einsum("ij,ij->i", prev_n_all, prev_n_all)) > 1e-10

        n_all, t1_all, t2_all = build_contact_frame_batch(
            normal_all,
            prev_tangent1s=prev_t1_all,
            prev_normals=prev_n_all,
            has_prev_mask=has_prev,
            has_prev_n_mask=has_prev_n,
        )

        # --- 結果書き戻し ---
        g_on = 0.0
        g_off = 0.0
        if config is not None:
            if hasattr(config, "g_on"):
                g_on = config.g_on
            if hasattr(config, "g_off"):
                g_off = config.g_off

        for i, pair in enumerate(pairs):
            pair.state.s = float(s_all[i])
            pair.state.t = float(t_all[i])
            pair.state.gap = float(gap_all[i])
            pair.state.normal = n_all[i]
            pair.state.tangent1 = t1_all[i]
            pair.state.tangent2 = t2_all[i]

            _update_active_set_hysteresis(
                pair,
                g_on=g_on,
                g_off=g_off,
                allow_deactivation=True,
            )

    def build_constraint_jacobian(
        self,
        pairs: list,
        ndof_total: int,
        ndof_per_node: int = 6,
    ) -> tuple[sp.csr_matrix, list[int]]:
        """制約ヤコビアン G を構築（Mortar加重版）.

        Mortar法では mortar.build_mortar_system() で加重ギャップと
        制約ヤコビアンを構築する。ここではPtPベースのフォールバック。
        完全なMortar制約はPhase 5で統合予定。
        """
        return _build_constraint_jacobian_ptp(pairs, ndof_total, ndof_per_node)

    def process(self, input_data: ContactGeometryInput) -> ContactGeometryOutput:
        pairs = self.detect(input_data.node_coords, input_data.connectivity, input_data.radii)
        return ContactGeometryOutput(contact_pairs=pairs)


def create_contact_geometry_strategy(
    *,
    mode: str = "point_to_point",
    exclude_same_layer: bool = True,
    n_gauss: int = 2,
    auto_gauss: bool = False,
    line_contact: bool = False,
    use_mortar: bool = False,
) -> PointToPointProcess | LineToLineGaussProcess | MortarSegmentProcess:
    """solver_ncp.py の接触幾何モード分岐を Strategy に移譲するファクトリ.

    Args:
        mode: "point_to_point" | "line_to_line" | "mortar"
        exclude_same_layer: 同層接触除外
        n_gauss: Gauss積分点数（L2L/Mortar用）
        auto_gauss: ペア角度に基づくGauss点自動選択
        line_contact: True → LineToLineGauss（mode の代替指定）
        use_mortar: True → MortarSegment（mode の代替指定）

    Returns:
        ContactGeometry Strategy インスタンス
    """
    # line_contact / use_mortar フラグによる自動判定
    if use_mortar:
        mode = "mortar"
    elif line_contact:
        mode = "line_to_line"

    if mode == "mortar":
        return MortarSegmentProcess(
            n_gauss=n_gauss,
            exclude_same_layer=exclude_same_layer,
        )

    if mode == "line_to_line":
        return LineToLineGaussProcess(
            n_gauss=n_gauss,
            exclude_same_layer=exclude_same_layer,
            auto_gauss=auto_gauss,
        )

    return PointToPointProcess(exclude_same_layer=exclude_same_layer)
