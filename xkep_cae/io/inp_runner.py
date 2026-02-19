"""Abaqus .inp ファイルからの梁解析モデル構築ユーティリティ.

AbaqusMesh パース結果をソルバー用のデータ構造に変換する。
要素タイプ（B21/B31）・断面タイプ（RECT/CIRC/PIPE）を自動判定し、
適切な要素・断面・材料オブジェクトを生成する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from xkep_cae.assembly import assemble_global_stiffness
from xkep_cae.bc import apply_dirichlet
from xkep_cae.elements.beam_eb2d import EulerBernoulliBeam2D
from xkep_cae.elements.beam_timo2d import TimoshenkoBeam2D
from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D
from xkep_cae.io.abaqus_inp import AbaqusMesh
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection, BeamSection2D
from xkep_cae.solver import LinearSolveResult, solve_displacement


@dataclass
class BeamModel:
    """パース済み .inp から構築された梁解析モデル.

    Attributes:
        nodes: (n_nodes, ndim) 節点座標配列（0始まりインデックス）
        element_groups: [(element, connectivity), ...] アセンブリ用
        material: BeamElastic1D 材料オブジェクト
        sections: 断面オブジェクトのリスト
        fixed_dofs: 拘束 DOF 配列
        ndof_per_node: 1節点あたりの DOF 数（3 or 6）
        ndof_total: 全 DOF 数
        label_to_index: ノードラベル → 0基底インデックスの辞書
        nsets: ノードセット（インデックスベース）
        is_3d: True なら 3D 梁（6DOF/node）
        rho: 密度（質量行列用）。None なら未定義
        mesh: 元の AbaqusMesh
    """

    nodes: np.ndarray
    element_groups: list[tuple]
    material: BeamElastic1D
    sections: list
    fixed_dofs: np.ndarray
    ndof_per_node: int
    ndof_total: int
    label_to_index: dict[int, int]
    nsets: dict[str, np.ndarray] = field(default_factory=dict)
    is_3d: bool = False
    rho: float | None = None
    mesh: AbaqusMesh | None = None


def build_beam_model_from_inp(
    mesh: AbaqusMesh,
    *,
    beam_formulation: str = "timoshenko",
    kappa: str | float = "cowper",
) -> BeamModel:
    """AbaqusMesh から梁解析モデルを構築する.

    Args:
        mesh: read_abaqus_inp() の戻り値
        beam_formulation: "timoshenko" or "euler-bernoulli"
        kappa: せん断補正係数。"cowper" で Cowper(1966) 自動計算

    Returns:
        BeamModel オブジェクト
    """
    # 1. 節点座標とラベルマッピング
    label_to_index: dict[int, int] = {}
    node_coords_list: list[list[float]] = []

    # 梁要素タイプから次元を判定
    is_3d = _detect_3d(mesh)

    for i, node in enumerate(mesh.nodes):
        label_to_index[node.label] = i
        if is_3d:
            node_coords_list.append([node.x, node.y, node.z])
        else:
            node_coords_list.append([node.x, node.y])

    nodes = np.array(node_coords_list, dtype=float)
    n_nodes = len(nodes)
    ndof_per_node = 6 if is_3d else 3
    ndof_total = ndof_per_node * n_nodes

    # 2. 材料
    mat_obj = _build_material(mesh)
    rho = None
    if mesh.materials:
        rho = mesh.materials[0].density

    # 3. 断面 + 要素 → element_groups
    element_groups = []
    sections = []
    for bsec in mesh.beam_sections:
        section = _build_section(bsec, is_3d)
        sections.append(section)

        # 対応する要素グループの接続配列を取得
        conn = _get_connectivity_for_elset(mesh, bsec.elset, label_to_index)

        # 要素オブジェクト生成
        elem = _build_element(section, is_3d, beam_formulation, kappa, bsec.direction)
        element_groups.append((elem, conn))

    # 4. 境界条件 → fixed_dofs
    fixed_dofs = _build_fixed_dofs(mesh, label_to_index, ndof_per_node)

    # 5. ノードセット（インデックスベース）
    nsets_idx: dict[str, np.ndarray] = {}
    for name, labels in mesh.nsets.items():
        indices = [label_to_index[lbl] for lbl in labels if lbl in label_to_index]
        nsets_idx[name.upper()] = np.array(indices, dtype=int)

    return BeamModel(
        nodes=nodes,
        element_groups=element_groups,
        material=mat_obj,
        sections=sections,
        fixed_dofs=fixed_dofs,
        ndof_per_node=ndof_per_node,
        ndof_total=ndof_total,
        label_to_index=label_to_index,
        nsets=nsets_idx,
        is_3d=is_3d,
        rho=rho,
        mesh=mesh,
    )


def solve_beam_static(
    model: BeamModel,
    f_ext: np.ndarray,
    *,
    show_progress: bool = False,
) -> LinearSolveResult:
    """梁モデルの線形静解析を実行する.

    Args:
        model: build_beam_model_from_inp() の戻り値
        f_ext: 外力ベクトル (ndof_total,)
        show_progress: 進捗表示

    Returns:
        LinearSolveResult: (u, info)
    """
    K = assemble_global_stiffness(
        model.nodes,
        model.element_groups,
        model.material,
        show_progress=show_progress,
    )
    K_sp = sp.csr_matrix(K)
    result = apply_dirichlet(K_sp, f_ext, model.fixed_dofs)
    return solve_displacement(result.K, result.f, show_progress=show_progress)


def node_dof(model: BeamModel, node_label: int, local_dof: int) -> int:
    """ノードラベルとローカルDOF番号(0始まり)からグローバルDOFインデックスを返す."""
    idx = model.label_to_index[node_label]
    return idx * model.ndof_per_node + local_dof


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------


def _detect_3d(mesh: AbaqusMesh) -> bool:
    """要素タイプから 3D かどうかを判定する."""
    for group in mesh.element_groups:
        etype = group.elem_type.upper()
        if etype in ("B31", "B32"):
            return True
        if etype in ("B21", "B22"):
            return False
    # 梁要素が見つからない場合はノード座標から推定
    has_z = any(abs(n.z) > 1e-12 for n in mesh.nodes)
    return has_z


def _build_material(mesh: AbaqusMesh) -> BeamElastic1D:
    """AbaqusMesh から BeamElastic1D を生成する."""
    if not mesh.materials:
        raise ValueError("材料定義が見つかりません。*MATERIAL + *ELASTIC が必要です。")
    mat = mesh.materials[0]
    if mat.elastic is None:
        raise ValueError(f"材料 '{mat.name}' に *ELASTIC が定義されていません。")
    E, nu = mat.elastic
    return BeamElastic1D(E=E, nu=nu)


def _build_section(bsec, is_3d: bool):
    """AbaqusBeamSection から BeamSection / BeamSection2D を生成する."""
    stype = bsec.section_type.upper()
    dims = bsec.dimensions

    if stype == "RECT":
        b, h = dims[0], dims[1]
        return BeamSection.rectangle(b, h) if is_3d else BeamSection2D.rectangle(b, h)
    elif stype == "CIRC":
        r = dims[0]
        d = 2.0 * r
        return BeamSection.circle(d) if is_3d else BeamSection2D.circle(d)
    elif stype == "PIPE":
        r_outer = dims[0]
        t_wall = dims[1]
        d_outer = 2.0 * r_outer
        d_inner = 2.0 * (r_outer - t_wall)
        if is_3d:
            return BeamSection.pipe(d_outer, d_inner)
        else:
            raise ValueError("2D梁にPIPE断面は非対応です。")
    else:
        raise ValueError(f"未対応の断面タイプ: {stype}")


def _build_element(section, is_3d: bool, formulation: str, kappa, direction):
    """断面と設定から要素オブジェクトを生成する."""
    v_ref = None
    if direction is not None:
        v_ref = np.array(direction, dtype=float)

    if is_3d:
        return TimoshenkoBeam3D(
            section=section,
            kappa_y=kappa,
            kappa_z=kappa,
            v_ref=v_ref,
        )
    else:
        if formulation == "euler-bernoulli":
            return EulerBernoulliBeam2D(section=section)
        else:
            return TimoshenkoBeam2D(section=section, kappa=kappa)


def _get_connectivity_for_elset(
    mesh: AbaqusMesh,
    elset_name: str,
    label_to_index: dict[int, int],
) -> np.ndarray:
    """指定 ELSET に属する要素の接続配列（0基底インデックス）を返す."""
    key = elset_name.upper()
    conn_list = []

    for group in mesh.element_groups:
        if group.elset and group.elset.upper() == key:
            for _label, node_labels in group.elements:
                row = [label_to_index[nl] for nl in node_labels]
                conn_list.append(row)

    if not conn_list:
        raise ValueError(f"要素セット '{elset_name}' が見つかりません。")

    return np.array(conn_list, dtype=int)


def _build_fixed_dofs(
    mesh: AbaqusMesh,
    label_to_index: dict[int, int],
    ndof_per_node: int,
) -> np.ndarray:
    """AbaqusBoundary リストから拘束 DOF 配列を構築する."""
    fixed = set()
    for bc in mesh.boundaries:
        if bc.node_label not in label_to_index:
            continue
        idx = label_to_index[bc.node_label]
        # Abaqus DOF は 1始まり → 0始まりに変換
        first = bc.first_dof - 1
        last = bc.last_dof - 1
        # 3D梁: DOF 1-6, 2D梁: DOF 1-3
        for dof in range(first, last + 1):
            if dof < ndof_per_node:
                fixed.add(idx * ndof_per_node + dof)
    return np.array(sorted(fixed), dtype=int)
