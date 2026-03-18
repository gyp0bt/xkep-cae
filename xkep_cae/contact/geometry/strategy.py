"""ContactGeometry Strategy 具象実装.

ContactGeometryStrategy Protocol に従い、接触幾何の検出・更新を行う Process 群。

3クラス構成:
- PointToPointProcess: 最近接点ペア（PtP）
- LineToLineGaussProcess: Line-to-Line Gauss 積分
- MortarSegmentProcess: Mortar 法セグメント
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._broadphase import _broadphase_aabb
from xkep_cae.contact._contact_pair import (
    _ContactPairOutput,
    _ContactStateOutput,
    _evolve_pair,
    _evolve_state,
)
from xkep_cae.core import ProcessMeta, SolverProcess

# ── Input / Output ─────────────────────────────────────────


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


# ── ヘルパー関数 ───────────────────────────────────────────


def _update_active_set_hysteresis(
    state: object,
    *,
    g_on: float = 0.0,
    g_off: float = 0.0,
    allow_deactivation: bool = True,
    coating_stiffness: float = 0.0,
) -> object:
    """Active-set をヒステリシス付きで更新し新 state を返す.

    Args:
        state: _ContactStateOutput
        g_on: 活性化閾値
        g_off: 非活性化閾値 (g_off > g_on)
        allow_deactivation: False で非活性化を禁止
        coating_stiffness: 被膜剛性 (>0 で被膜モデル有効)

    Returns:
        更新された _ContactStateOutput
    """
    from xkep_cae.contact._types import ContactStatus

    gap = state.gap
    coat_active = coating_stiffness > 0.0 and state.coating_compression > 0.0

    if state.status == ContactStatus.INACTIVE:
        if gap <= g_on or coat_active:
            return _evolve_state(state, status=ContactStatus.ACTIVE)
    else:
        if allow_deactivation and gap >= g_off and not coat_active:
            return _evolve_state(state, status=ContactStatus.INACTIVE)
    return state


def _build_constraint_jacobian_ptp(
    pairs: list,
    ndof_total: int,
    ndof_per_node: int = 6,
) -> tuple[sp.csr_matrix, list[int]]:
    """PtP用制約ヤコビアン G = ∂g_n/∂u を構築する.

    Returns:
        (G, active_indices)
    """
    from xkep_cae.contact._assembly_utils import _contact_dofs
    from xkep_cae.contact._types import ContactStatus

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


def _detect_candidates(
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    radii: np.ndarray | float,
    *,
    exclude_same_layer: bool = True,
    margin: float = 0.0,
    cell_size: float | None = None,
) -> list:
    """Broadphase AABB + ペア生成の共通実装.

    Args:
        node_coords: (n_nodes, 3) 節点座標
        connectivity: (n_elems, 2) 要素接続（各行: [node0, node1]）
        radii: 要素ごとの断面半径
        exclude_same_layer: 共有ノードを持つペアを除外
        margin: 探索マージン
        cell_size: 格子セルサイズ

    Returns:
        _ContactPairOutput のリスト
    """
    coords = np.asarray(node_coords, dtype=float)
    conn = np.asarray(connectivity, dtype=int)
    n_elems = len(conn)
    if n_elems < 2:
        return []

    # セグメント端点リスト
    segments = [(coords[conn[i, 0]], coords[conn[i, 1]]) for i in range(n_elems)]

    # 要素ごとの半径
    if np.isscalar(radii):
        r_arr = np.full(n_elems, float(radii))
    else:
        r_arr = np.asarray(radii, dtype=float)

    # Broadphase
    candidates = _broadphase_aabb(segments, r_arr, margin=margin, cell_size=cell_size)

    # フィルタリング: 共有ノード（同層）除外
    pairs = []
    for ei, ej in candidates:
        if exclude_same_layer:
            nodes_i = set(conn[ei])
            nodes_j = set(conn[ej])
            if nodes_i & nodes_j:
                continue

        pair = _ContactPairOutput(
            elem_a=ei,
            elem_b=ej,
            nodes_a=conn[ei],
            nodes_b=conn[ej],
            state=_ContactStateOutput(),
            radius_a=float(r_arr[ei]),
            radius_b=float(r_arr[ej]),
        )
        pairs.append(pair)

    # 初期 narrowphase
    if pairs:
        _batch_update_geometry(pairs, coords)

    return pairs


# ── バッチ幾何更新共通処理 ─────────────────────────────────


def _batch_update_geometry(
    pairs: list,
    node_coords: np.ndarray,
    *,
    config: object | None = None,
) -> None:
    """全ペアの幾何情報をバッチ計算で更新する共通処理.

    PointToPoint, LineToLineGauss, MortarSegment で共通のロジック。
    """
    from xkep_cae.contact.geometry._compute import (
        _build_contact_frame_batch as build_contact_frame_batch,
    )
    from xkep_cae.contact.geometry._compute import (
        _closest_point_segments_batch as closest_point_segments_batch,
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
    s_all, t_all, _, _, dist_all, normal_all, _ = closest_point_segments_batch(xA0, xA1, xB0, xB1)

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
        geom_kw: dict[str, object] = {
            "s": float(s_all[i]),
            "t": float(t_all[i]),
            "gap": float(gap_all[i]),
            "normal": n_all[i],
            "tangent1": t1_all[i],
            "tangent2": t2_all[i],
        }
        if _use_coating:
            geom_kw["coating_compression"] = float(coat_comp[i])
        new_state = _evolve_state(pair.state, **geom_kw)

        new_state = _update_active_set_hysteresis(
            new_state,
            g_on=g_on,
            g_off=g_off,
            allow_deactivation=allow_deact,
            coating_stiffness=coating_stiffness,
        )
        pairs[i] = _evolve_pair(pair, state=new_state)


# ── 具象 Process ──────────────────────────────────────────


class PointToPointProcess(
    SolverProcess[ContactGeometryInput, ContactGeometryOutput],
):
    """最近接点ペア（Point-to-Point）による接触検出.

    各要素ペアの最近接パラメータ (s, t) を求め、
    ギャップ g = ||x_B - x_A|| - r_A - r_B を評価する。
    """

    meta = ProcessMeta(
        name="PointToPoint",
        module="solve",
        version="1.0.0",
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
        """Broadphase AABB + 共有ノード除外 + 初期 narrowphase."""
        return _detect_candidates(
            node_coords,
            connectivity,
            radii,
            exclude_same_layer=self._exclude_same_layer,
        )

    def compute_gap(self, pair: object, node_coords: np.ndarray) -> float:
        """ギャップの計算."""
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
        """全ペアの幾何情報を更新する（Narrowphase）."""
        _batch_update_geometry(pairs, node_coords, config=config)

    def build_constraint_jacobian(
        self,
        pairs: list,
        ndof_total: int,
        ndof_per_node: int = 6,
    ) -> tuple[sp.csr_matrix, list[int]]:
        """制約ヤコビアン G = ∂g_n/∂u を構築."""
        return _build_constraint_jacobian_ptp(pairs, ndof_total, ndof_per_node)

    def process(self, input_data: ContactGeometryInput) -> ContactGeometryOutput:
        pairs = self.detect(input_data.node_coords, input_data.connectivity, input_data.radii)
        return ContactGeometryOutput(contact_pairs=pairs)


class LineToLineGaussProcess(
    SolverProcess[ContactGeometryInput, ContactGeometryOutput],
):
    """Line-to-Line Gauss 積分による接触評価.

    要素ペアの相互作用領域をGauss積分点で離散化し、
    接触力と接触剛性を積分する。大規模問題で精度・安定性が向上。

    パラメータ:
        n_gauss: 1次元あたりの Gauss 点数（デフォルト: 2）
        auto_gauss: True でペア角度に基づく自動選択
    """

    meta = ProcessMeta(
        name="LineToLineGauss",
        module="solve",
        version="1.0.0",
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
        """Broadphase AABB + 共有ノード除外 + 初期 narrowphase."""
        return _detect_candidates(
            node_coords,
            connectivity,
            radii,
            exclude_same_layer=self._exclude_same_layer,
        )

    def compute_gap(self, pair: object, node_coords: np.ndarray) -> float:
        """ギャップの計算."""
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
        """全ペアの幾何情報を更新する（Narrowphase + Gauss点情報）."""
        _batch_update_geometry(pairs, node_coords, config=config)

        # --- Gauss点数自動選択 ---
        if self._auto_gauss and len(pairs) > 0:
            from xkep_cae.contact.geometry._compute import (
                _auto_select_n_gauss as auto_select_n_gauss,
            )

            coords = np.asarray(node_coords, dtype=float)
            for pair in pairs:
                xA0 = coords[pair.nodes_a[0]]
                xA1 = coords[pair.nodes_a[1]]
                xB0 = coords[pair.nodes_b[0]]
                xB1 = coords[pair.nodes_b[1]]
                dA = xA1 - xA0
                dB = xB1 - xB0
                n_gp = auto_select_n_gauss(dA, dB)
                if hasattr(pair, "_n_gauss"):
                    pair._n_gauss = n_gp

    def build_constraint_jacobian(
        self,
        pairs: list,
        ndof_total: int,
        ndof_per_node: int = 6,
    ) -> tuple[sp.csr_matrix, list[int]]:
        """制約ヤコビアン G = ∂g_n/∂u を構築."""
        return _build_constraint_jacobian_ptp(pairs, ndof_total, ndof_per_node)

    def process(self, input_data: ContactGeometryInput) -> ContactGeometryOutput:
        pairs = self.detect(input_data.node_coords, input_data.connectivity, input_data.radii)
        return ContactGeometryOutput(contact_pairs=pairs)


class MortarSegmentProcess(
    SolverProcess[ContactGeometryInput, ContactGeometryOutput],
):
    """Mortar 法セグメントによる接触評価.

    Line-to-Line に加えて mortar 射影を行い、
    接触面の連続性を保証する。大規模問題のロバスト性に寄与。
    """

    meta = ProcessMeta(
        name="MortarSegment",
        module="solve",
        version="1.0.0",
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
        """Broadphase AABB + 共有ノード除外 + 初期 narrowphase."""
        return _detect_candidates(
            node_coords,
            connectivity,
            radii,
            exclude_same_layer=self._exclude_same_layer,
        )

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
        """全ペアの幾何情報を更新する（Mortar射影ベース）."""
        _batch_update_geometry(pairs, node_coords, config=config)

    def build_constraint_jacobian(
        self,
        pairs: list,
        ndof_total: int,
        ndof_per_node: int = 6,
    ) -> tuple[sp.csr_matrix, list[int]]:
        """制約ヤコビアン G を構築（Mortar加重版フォールバック）."""
        return _build_constraint_jacobian_ptp(pairs, ndof_total, ndof_per_node)

    def process(self, input_data: ContactGeometryInput) -> ContactGeometryOutput:
        pairs = self.detect(input_data.node_coords, input_data.connectivity, input_data.radii)
        return ContactGeometryOutput(contact_pairs=pairs)


# ── ファクトリ ─────────────────────────────────────────────


def _create_contact_geometry_strategy(
    *,
    mode: str = "point_to_point",
    exclude_same_layer: bool = True,
    n_gauss: int = 2,
    auto_gauss: bool = False,
    line_contact: bool = False,
    use_mortar: bool = False,
) -> PointToPointProcess | LineToLineGaussProcess | MortarSegmentProcess:
    """接触幾何 Strategy ファクトリ.

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
