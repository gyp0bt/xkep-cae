"""VtkExportProcess — ParaView 用 VTK XML 出力の PostProcess.

設計仕様: docs/vtk_export.md

`SolverResultData` + `MeshData` を受け取り、梁要素（VTK_LINE）の
変形形状とフィールド値を VTK XML 形式（.vtu / .pvd）で書き出す。
ParaView で `.pvd` を開くと時系列アニメーションになる。

`tube_n_segments > 0` で円筒断面相当の **solid pipe mesh** (VTK_QUAD) を
別途 `<prefix>_pipe.pvd` + `<prefix>_pipe_NNNNN.vtu` として並行出力する。
ParaView の Tube filter に依存せず、断面半径 `MeshData.radii` を反映した
擬似ソリッド表示が可能になる。

依存追加なし — 生 XML を直接書く実装。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from xkep_cae.core import MeshData, PostProcess, ProcessMeta, SolverResultData

_VTK_LINE = 3  # VTK cell type for 2-node line
_VTK_QUAD = 9  # VTK cell type for 4-node quadrilateral


@dataclass(frozen=True)
class VtkExportConfig:
    """VTK 出力設定."""

    solver_result: SolverResultData
    mesh: MeshData
    output_dir: str = "output/vtk"
    prefix: str = "result"
    ndof_per_node: int = 6
    write_time_series: bool = True
    # PointData / CellData フィールドの ON/OFF
    include_rotations: bool = True
    include_axial_strain: bool = True
    include_axial_stress: bool = True
    include_curvature_vector: bool = True
    include_contact_force: bool = True
    # 梁断面の応力合力 (円形断面前提、`mesh.radii` を使用)
    include_moment_vector: bool = True  # (M_torsion, M_bend_y, M_bend_z)
    include_max_bending_stress: bool = True  # |M_bend| · r / I
    include_torsion_shear_stress: bool = True  # |M_x| · r / J
    include_von_mises_stress: bool = True  # 最危険繊維での換算応力
    # 物性
    young_modulus: float = 130.0e3  # MPa
    poisson_ratio: float = 0.3  # G = E / (2(1+ν))
    # 円筒パイプメッシュ (擬似ソリッド表示)
    tube_n_segments: int = 0  # 0 = line のみ、≥3 で pipe mesh も出力
    tube_radius_override: float | None = None  # None なら mesh.radii を使用


@dataclass(frozen=True)
class VtkExportResult:
    """VTK 出力結果."""

    vtu_paths: tuple[str, ...] = field(default_factory=tuple)
    pvd_path: str | None = None
    pipe_vtu_paths: tuple[str, ...] = field(default_factory=tuple)
    pipe_pvd_path: str | None = None
    n_timesteps: int = 0
    n_points: int = 0
    n_cells: int = 0
    pipe_n_points: int = 0
    pipe_n_cells: int = 0


# =====================================================================
# 低レベル XML ヘルパー
# =====================================================================


def _format_float_array(arr: np.ndarray) -> str:
    return " ".join(f"{v:.10e}" for v in np.asarray(arr, dtype=float).ravel())


def _format_int_array(arr: np.ndarray) -> str:
    return " ".join(str(int(v)) for v in np.asarray(arr).ravel())


def _build_data_array(
    name: str,
    values: np.ndarray,
    *,
    dtype: str = "Float64",
    n_components: int = 1,
) -> str:
    body = _format_float_array(values) if dtype.startswith("Float") else _format_int_array(values)
    return (
        f'      <DataArray type="{dtype}" Name="{escape(name)}" '
        f'NumberOfComponents="{n_components}" format="ascii">\n'
        f"        {body}\n"
        f"      </DataArray>\n"
    )


def _build_vtu_xml(
    points: np.ndarray,
    connectivity: np.ndarray,
    cell_type: int,
    point_data: dict[str, tuple[np.ndarray, int]],
    cell_data: dict[str, tuple[np.ndarray, int]],
) -> str:
    """完全な .vtu XML 文字列を組み立てる.

    line cell (type 3, 2 nodes/cell) と quad cell (type 9, 4 nodes/cell) の
    両方に対応する。connectivity は (n_cells, nodes_per_cell) 形状を期待する。
    """
    n_points = int(points.shape[0])
    n_cells = int(connectivity.shape[0])
    n_pts_per_cell = int(connectivity.shape[1])
    offsets = np.arange(1, n_cells + 1, dtype=int) * n_pts_per_cell
    types = np.full(n_cells, cell_type, dtype=int)

    parts: list[str] = ['<?xml version="1.0"?>\n']
    parts.append('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
    parts.append("  <UnstructuredGrid>\n")
    parts.append(f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">\n')
    parts.append("      <Points>\n")
    parts.append(_build_data_array("Points", points, n_components=3))
    parts.append("      </Points>\n")
    parts.append("      <Cells>\n")
    parts.append(_build_data_array("connectivity", connectivity, dtype="Int32"))
    parts.append(_build_data_array("offsets", offsets, dtype="Int32"))
    parts.append(_build_data_array("types", types, dtype="UInt8"))
    parts.append("      </Cells>\n")
    if point_data:
        parts.append("      <PointData>\n")
        for name, (arr, ncomp) in point_data.items():
            parts.append(_build_data_array(name, arr, n_components=ncomp))
        parts.append("      </PointData>\n")
    if cell_data:
        parts.append("      <CellData>\n")
        for name, (arr, ncomp) in cell_data.items():
            parts.append(_build_data_array(name, arr, n_components=ncomp))
        parts.append("      </CellData>\n")
    parts.append("    </Piece>\n")
    parts.append("  </UnstructuredGrid>\n")
    parts.append("</VTKFile>\n")
    return "".join(parts)


def _build_pvd_xml(vtu_files: list[str], timesteps: list[float]) -> str:
    parts: list[str] = ['<?xml version="1.0"?>\n']
    parts.append('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    parts.append("  <Collection>\n")
    for t, f in zip(timesteps, vtu_files, strict=True):
        parts.append(f'    <DataSet timestep="{t:.10e}" group="" part="0" file="{escape(f)}"/>\n')
    parts.append("  </Collection>\n")
    parts.append("</VTKFile>\n")
    return "".join(parts)


# =====================================================================
# 物理量計算ヘルパー
# =====================================================================


def _extract_translation_rotation(
    u: np.ndarray, n_nodes: int, ndof_per_node: int
) -> tuple[np.ndarray, np.ndarray | None]:
    u_reshaped = np.asarray(u, dtype=float).reshape(n_nodes, ndof_per_node)
    trans = u_reshaped[:, :3]
    rot = u_reshaped[:, 3:6] if ndof_per_node >= 6 else None
    return trans, rot


def _compute_axial_strain(
    deformed_coords: np.ndarray, connectivity: np.ndarray, ref_lengths: np.ndarray
) -> np.ndarray:
    p0 = deformed_coords[connectivity[:, 0]]
    p1 = deformed_coords[connectivity[:, 1]]
    l_def = np.linalg.norm(p1 - p0, axis=1)
    return (l_def - ref_lengths) / np.where(ref_lengths > 0.0, ref_lengths, 1.0)


def _compute_curvature_vector(
    rot: np.ndarray, connectivity: np.ndarray, ref_lengths: np.ndarray
) -> np.ndarray:
    """各要素の曲率ベクトル κ_e ≈ (θ_j − θ_i) / L_ref (3 成分)."""
    theta_i = rot[connectivity[:, 0]]
    theta_j = rot[connectivity[:, 1]]
    return (theta_j - theta_i) / np.where(ref_lengths > 0.0, ref_lengths, 1.0)[:, None]


def _aggregate_contact_force(snapshot_entries: tuple, n_cells: int) -> np.ndarray:
    """`contact_pair_history` のスナップショットから要素別 |p_n| 集計を作る.

    各ペアの |p_n| を両端要素 (elem_a / elem_b) に加算する。
    """
    p_n_per_elem = np.zeros(n_cells, dtype=float)
    for entry in snapshot_entries:
        a = int(getattr(entry, "elem_a", -1))
        b = int(getattr(entry, "elem_b", -1))
        p = abs(float(getattr(entry, "p_n", 0.0)))
        if 0 <= a < n_cells:
            p_n_per_elem[a] += p
        if 0 <= b < n_cells:
            p_n_per_elem[b] += p
    return p_n_per_elem


def _compute_beam_stress_resultants(
    curvature: np.ndarray,
    elem_radii: np.ndarray,
    young: float,
    poisson: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """円形断面梁の応力合力 (moment_vector, σ_b_max, τ_torsion) を返す.

    曲率ベクトル `(κ_x, κ_y, κ_z)` （ねじり率 + 2 軸曲げ）から:

        I = π r⁴ / 4    (二次モーメント)
        J = π r⁴ / 2    (極二次モーメント)
        G = E / (2(1+ν))

        M_torsion = G · J · κ_x
        M_bend_y  = E · I · κ_y
        M_bend_z  = E · I · κ_z

        σ_b_max   = √(M_bend_y² + M_bend_z²) · r / I
        τ_torsion = |M_x| · r / J
    """
    r = elem_radii
    safe = r > 0.0
    I_sec = np.where(safe, np.pi * r**4 / 4.0, 1.0)
    J_pol = np.where(safe, np.pi * r**4 / 2.0, 1.0)
    G_mod = young / (2.0 * (1.0 + poisson))
    M_torsion = G_mod * J_pol * curvature[:, 0]
    M_bend_y = young * I_sec * curvature[:, 1]
    M_bend_z = young * I_sec * curvature[:, 2]
    moment_vec = np.column_stack([M_torsion, M_bend_y, M_bend_z])
    M_bend_norm = np.sqrt(M_bend_y**2 + M_bend_z**2)
    sigma_b_max = M_bend_norm * r / I_sec
    tau_torsion = np.abs(M_torsion) * r / J_pol
    # 退化要素 (r=0) は応力 0 にしておく
    sigma_b_max = np.where(safe, sigma_b_max, 0.0)
    tau_torsion = np.where(safe, tau_torsion, 0.0)
    return moment_vec, sigma_b_max, tau_torsion


def _compute_von_mises_max(
    sigma_axial: np.ndarray, sigma_b_max: np.ndarray, tau_torsion: np.ndarray
) -> np.ndarray:
    """最危険繊維 (axial + bending fiber surface + torsion surface) での von Mises.

    σ_vM = √( (σ_axial + σ_b_max)² + 3 τ_torsion² )
    """
    sigma_normal = sigma_axial + sigma_b_max
    return np.sqrt(sigma_normal**2 + 3.0 * tau_torsion**2)


def _resolve_element_radii(radii: Any, connectivity: np.ndarray, n_cells: int) -> np.ndarray:
    """要素半径配列を解決（スカラー / 要素長 / ノード長いずれも対応）."""
    if np.isscalar(radii):
        return np.full(n_cells, float(radii))
    arr = np.asarray(radii, dtype=float)
    if arr.shape[0] == n_cells:
        return arr
    r_elem = np.full(n_cells, float(arr.mean()))
    for e in range(n_cells):
        n0 = int(connectivity[e, 0])
        n1 = int(connectivity[e, 1])
        if n0 < arr.shape[0] and n1 < arr.shape[0]:
            r_elem[e] = 0.5 * (arr[n0] + arr[n1])
    return r_elem


# =====================================================================
# パイプメッシュ生成 (擬似ソリッド表示)
# =====================================================================


def _cross_section_frame(t_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """軸方向 t_axis に直交する 2 ベクトル (n, b) を返す."""
    e_z = np.array([0.0, 0.0, 1.0])
    e_x = np.array([1.0, 0.0, 0.0])
    ref = e_z if abs(float(np.dot(t_axis, e_z))) < 0.99 else e_x
    n = np.cross(t_axis, ref)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        n = np.cross(t_axis, e_x)
        n_norm = float(np.linalg.norm(n))
    n /= max(n_norm, 1e-30)
    b = np.cross(t_axis, n)
    return n, b


def _build_pipe_mesh(
    deformed_coords: np.ndarray,
    connectivity: np.ndarray,
    elem_radii: np.ndarray,
    n_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """各 line 要素を `n_segments` 角形パイプ（VTK_QUAD 群）に展開する.

    返り値:
        pipe_points: (n_elems * 2 * n_segments, 3)
        pipe_quads: (n_elems * n_segments, 4) — VTK_QUAD connectivity
        node_expand_idx: (n_elems * 2 * n_segments,) — pipe point → 元 node index
        elem_expand_idx: (n_elems * n_segments,) — pipe quad → 元 element index
    """
    n_elems = int(connectivity.shape[0])
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    pipe_points = np.zeros((n_elems * 2 * n_segments, 3), dtype=float)
    pipe_quads = np.zeros((n_elems * n_segments, 4), dtype=int)
    node_expand = np.zeros(n_elems * 2 * n_segments, dtype=int)
    elem_expand = np.zeros(n_elems * n_segments, dtype=int)

    for e_idx in range(n_elems):
        i, j = int(connectivity[e_idx, 0]), int(connectivity[e_idx, 1])
        p0, p1 = deformed_coords[i], deformed_coords[j]
        axis = p1 - p0
        L = float(np.linalg.norm(axis))
        if L < 1e-12:
            # 退化要素は z 軸を仮置き（quad は潰れる）
            axis_unit = np.array([0.0, 0.0, 1.0])
        else:
            axis_unit = axis / L
        n_vec, b_vec = _cross_section_frame(axis_unit)
        r = float(elem_radii[e_idx])
        offsets = r * (cos_a[:, None] * n_vec[None, :] + sin_a[:, None] * b_vec[None, :])

        pt_base = e_idx * 2 * n_segments
        pipe_points[pt_base : pt_base + n_segments] = p0[None, :] + offsets
        pipe_points[pt_base + n_segments : pt_base + 2 * n_segments] = p1[None, :] + offsets
        node_expand[pt_base : pt_base + n_segments] = i
        node_expand[pt_base + n_segments : pt_base + 2 * n_segments] = j

        cell_base = e_idx * n_segments
        for k in range(n_segments):
            k_next = (k + 1) % n_segments
            pipe_quads[cell_base + k] = (
                pt_base + k,
                pt_base + k_next,
                pt_base + n_segments + k_next,
                pt_base + n_segments + k,
            )
            elem_expand[cell_base + k] = e_idx

    return pipe_points, pipe_quads, node_expand, elem_expand


# =====================================================================
# Process 本体
# =====================================================================


def _build_payload_for_step(
    u: np.ndarray,
    *,
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    ref_lengths: np.ndarray,
    elem_radii: np.ndarray | None,
    cfg: VtkExportConfig,
    pair_entries: tuple | None,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, int]], dict[str, tuple[np.ndarray, int]]]:
    """1 timestep 分の deformed 座標 + PointData/CellData を組み立てる."""
    n_nodes = int(node_coords.shape[0])
    n_cells = int(connectivity.shape[0])
    trans, rot = _extract_translation_rotation(u, n_nodes, cfg.ndof_per_node)
    deformed = node_coords + trans

    point_data: dict[str, tuple[np.ndarray, int]] = {"displacement": (trans, 3)}
    if cfg.include_rotations and rot is not None:
        point_data["rotation"] = (rot, 3)

    cell_data: dict[str, tuple[np.ndarray, int]] = {}
    eps = _compute_axial_strain(deformed, connectivity, ref_lengths)
    sigma_axial = eps * cfg.young_modulus
    if cfg.include_axial_strain:
        cell_data["axial_strain"] = (eps, 1)
    if cfg.include_axial_stress:
        cell_data["axial_stress"] = (sigma_axial, 1)

    # 曲率ベクトル + 梁断面応力合力 (rot と elem_radii が揃っているときのみ)
    needs_resultants = (
        rot is not None
        and elem_radii is not None
        and (
            cfg.include_moment_vector
            or cfg.include_max_bending_stress
            or cfg.include_torsion_shear_stress
            or cfg.include_von_mises_stress
        )
    )
    kappa = None
    if rot is not None and (cfg.include_curvature_vector or needs_resultants):
        kappa = _compute_curvature_vector(rot, connectivity, ref_lengths)
        if cfg.include_curvature_vector:
            cell_data["curvature_vector"] = (kappa, 3)
    if needs_resultants and kappa is not None:
        moment_vec, sigma_b_max, tau_torsion = _compute_beam_stress_resultants(
            kappa, elem_radii, cfg.young_modulus, cfg.poisson_ratio
        )
        if cfg.include_moment_vector:
            cell_data["moment_vector"] = (moment_vec, 3)
        if cfg.include_max_bending_stress:
            cell_data["max_bending_stress"] = (sigma_b_max, 1)
        if cfg.include_torsion_shear_stress:
            cell_data["torsion_shear_stress"] = (tau_torsion, 1)
        if cfg.include_von_mises_stress:
            cell_data["von_mises_stress"] = (
                _compute_von_mises_max(sigma_axial, sigma_b_max, tau_torsion),
                1,
            )

    if cfg.include_contact_force and pair_entries is not None:
        cell_data["contact_force"] = (_aggregate_contact_force(pair_entries, n_cells), 1)

    return deformed, point_data, cell_data


def _expand_to_pipe(
    point_data: dict[str, tuple[np.ndarray, int]],
    cell_data: dict[str, tuple[np.ndarray, int]],
    node_expand_idx: np.ndarray,
    elem_expand_idx: np.ndarray,
) -> tuple[dict[str, tuple[np.ndarray, int]], dict[str, tuple[np.ndarray, int]]]:
    """元 mesh の PointData / CellData をパイプメッシュ用に複製する."""
    pipe_point_data: dict[str, tuple[np.ndarray, int]] = {}
    for name, (arr, ncomp) in point_data.items():
        pipe_point_data[name] = (arr[node_expand_idx], ncomp)
    pipe_cell_data: dict[str, tuple[np.ndarray, int]] = {}
    for name, (arr, ncomp) in cell_data.items():
        pipe_cell_data[name] = (arr[elem_expand_idx], ncomp)
    return pipe_point_data, pipe_cell_data


class VtkExportProcess(PostProcess[VtkExportConfig, VtkExportResult]):
    """VTK XML エクスポートプロセス.

    `displacement_history` を時系列として `.vtu` 群 + `.pvd` を書き出す。
    `write_time_series=False` の場合は最終 state のみ単一 `.vtu` を出力する。

    PointData: displacement, rotation
    CellData: axial_strain, axial_stress, curvature_vector, contact_force

    `tube_n_segments > 0` で円筒断面相当の擬似ソリッド (VTK_QUAD ring) も
    `<prefix>_pipe.pvd` として並行出力する.
    """

    meta = ProcessMeta(
        name="VtkExport",
        module="post",
        version="1.1.0",
        document_path="docs/vtk_export.md",
    )

    def process(self, input_data: VtkExportConfig) -> VtkExportResult:
        cfg = input_data
        mesh = cfg.mesh
        result = cfg.solver_result
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        node_coords = np.asarray(mesh.node_coords, dtype=float)
        connectivity = np.asarray(mesh.connectivity, dtype=int)
        n_nodes = int(node_coords.shape[0])
        n_cells = int(connectivity.shape[0])
        p0_ref = node_coords[connectivity[:, 0]]
        p1_ref = node_coords[connectivity[:, 1]]
        ref_lengths = np.linalg.norm(p1_ref - p0_ref, axis=1)

        # 時系列フレーム集約
        if cfg.write_time_series and result.displacement_history:
            u_states = tuple(np.asarray(u, dtype=float) for u in result.displacement_history)
        else:
            u_states = (np.asarray(result.u, dtype=float),)
        n_steps = len(u_states)

        # timestep 値
        load_steps = result.load_history
        if load_steps and len(load_steps) == n_steps:
            timesteps = [float(t) for t in load_steps]
        else:
            timesteps = [float(i) for i in range(n_steps)]

        # 接触ペア履歴の整列
        pair_history = result.contact_pair_history if cfg.include_contact_force else ()
        pair_aligned: list[tuple | None] = [None] * n_steps
        if pair_history and len(pair_history) == n_steps:
            for i, item in enumerate(pair_history):
                pair_aligned[i] = item[1] if isinstance(item, tuple) and len(item) == 2 else None

        # 要素半径 (パイプメッシュ + 梁断面応力合力で使用、一度だけ計算)
        needs_radii = cfg.tube_n_segments >= 3 or (
            cfg.include_moment_vector
            or cfg.include_max_bending_stress
            or cfg.include_torsion_shear_stress
            or cfg.include_von_mises_stress
        )
        elem_radii: np.ndarray | None = None
        if needs_radii:
            if cfg.tube_radius_override is not None:
                elem_radii = np.full(n_cells, float(cfg.tube_radius_override))
            else:
                elem_radii = _resolve_element_radii(mesh.radii, connectivity, n_cells)
        pipe_enabled = cfg.tube_n_segments >= 3

        vtu_files: list[str] = []
        pipe_vtu_files: list[str] = []
        pipe_n_points = 0
        pipe_n_cells = 0

        for i, u in enumerate(u_states):
            deformed, point_data, cell_data = _build_payload_for_step(
                u,
                node_coords=node_coords,
                connectivity=connectivity,
                ref_lengths=ref_lengths,
                elem_radii=elem_radii,
                cfg=cfg,
                pair_entries=pair_aligned[i],
            )

            # line cell
            xml = _build_vtu_xml(deformed, connectivity, _VTK_LINE, point_data, cell_data)
            fname = f"{cfg.prefix}.vtu" if n_steps == 1 else f"{cfg.prefix}_{i:05d}.vtu"
            (out_dir / fname).write_text(xml, encoding="utf-8")
            vtu_files.append(fname)

            # pipe (solid) cell
            if pipe_enabled:
                pts, quads, node_idx, elem_idx = _build_pipe_mesh(
                    deformed, connectivity, elem_radii, cfg.tube_n_segments
                )
                pipe_pd, pipe_cd = _expand_to_pipe(point_data, cell_data, node_idx, elem_idx)
                pipe_xml = _build_vtu_xml(pts, quads, _VTK_QUAD, pipe_pd, pipe_cd)
                pipe_fname = (
                    f"{cfg.prefix}_pipe.vtu" if n_steps == 1 else f"{cfg.prefix}_pipe_{i:05d}.vtu"
                )
                (out_dir / pipe_fname).write_text(pipe_xml, encoding="utf-8")
                pipe_vtu_files.append(pipe_fname)
                pipe_n_points = int(pts.shape[0])
                pipe_n_cells = int(quads.shape[0])

        pvd_path: str | None = None
        if n_steps > 1:
            pvd_xml = _build_pvd_xml(vtu_files, timesteps)
            pvd_full = out_dir / f"{cfg.prefix}.pvd"
            pvd_full.write_text(pvd_xml, encoding="utf-8")
            pvd_path = str(pvd_full)

        pipe_pvd_path: str | None = None
        if pipe_enabled and n_steps > 1:
            pipe_pvd_xml = _build_pvd_xml(pipe_vtu_files, timesteps)
            pipe_pvd_full = out_dir / f"{cfg.prefix}_pipe.pvd"
            pipe_pvd_full.write_text(pipe_pvd_xml, encoding="utf-8")
            pipe_pvd_path = str(pipe_pvd_full)

        return VtkExportResult(
            vtu_paths=tuple(str(out_dir / f) for f in vtu_files),
            pvd_path=pvd_path,
            pipe_vtu_paths=tuple(str(out_dir / f) for f in pipe_vtu_files),
            pipe_pvd_path=pipe_pvd_path,
            n_timesteps=n_steps,
            n_points=n_nodes,
            n_cells=n_cells,
            pipe_n_points=pipe_n_points,
            pipe_n_cells=pipe_n_cells,
        )
