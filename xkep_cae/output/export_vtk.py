"""VTK/VTU エクスポート（ParaView 対応）.

OutputDatabase のフレームデータを VTK XML 形式でエクスポートする。
外部ライブラリに依存せず、VTK XML を直接生成する。

出力ファイル:
    - .vtu (VTK XML Unstructured Grid): 各フレームに1ファイル
    - .pvd (ParaView Data): タイムステップを束ねるインデックスファイル

出力形式:
    - ascii: テキスト形式（可読性と移植性を優先）
    - binary: Base64エンコード形式（ファイルサイズと読み込み速度を優先）
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import numpy as np

if TYPE_CHECKING:
    from xkep_cae.output.database import OutputDatabase
    from xkep_cae.output.step import Frame

# VTK Cell Type ID マッピング
VTK_VERTEX = 1
VTK_LINE = 3
VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_QUADRATIC_TRIANGLE = 22


def export_vtk(
    db: OutputDatabase,
    output_dir: str | Path,
    *,
    prefix: str = "result",
    binary: bool = False,
) -> str:
    """OutputDatabase のフレームデータを VTK ファイルに出力する.

    各フレームを .vtu ファイルとして出力し、.pvd でタイムシリーズを束ねる。

    Args:
        db: 出力データベース
        output_dir: 出力ディレクトリ
        prefix: ファイル名プレフィックス
        binary: True の場合 Base64 エンコードバイナリ形式で出力。
            False（デフォルト）の場合は ASCII テキスト形式。

    Returns:
        .pvd ファイルのパス
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if db.node_coords is None:
        raise ValueError("node_coords が設定されていない（VTK 出力に必須）")

    pvd_entries: list[tuple[float, str]] = []

    for sr in db.step_results:
        for frame in sr.frames:
            vtu_name = f"{prefix}_{sr.step.name}_f{frame.frame_index:04d}.vtu"
            vtu_path = out / vtu_name
            _write_vtu(vtu_path, frame, db, binary=binary)
            pvd_entries.append((frame.time, vtu_name))

    # .pvd ファイルの生成
    pvd_path = out / f"{prefix}.pvd"
    _write_pvd(pvd_path, pvd_entries)

    return str(pvd_path)


def _write_vtu(
    filepath: Path,
    frame: Frame,
    db: OutputDatabase,
    *,
    binary: bool = False,
) -> None:
    """1フレームの VTU (VTK XML Unstructured Grid) ファイルを書き出す.

    Args:
        filepath: 出力ファイルパス
        frame: フレームデータ
        db: 出力データベース
        binary: True の場合 Base64 エンコード形式で出力
    """
    n_nodes = db.n_nodes
    ndpn = db.ndof_per_node
    ndim = db.ndim

    # 座標を 3D に拡張（VTK は常に 3D）
    coords_3d = np.zeros((n_nodes, 3), dtype=np.float64)
    coords_3d[:, :ndim] = db.node_coords

    add_array = _add_data_array_binary if binary else _add_data_array

    # XML 構造の構築
    root = ET.Element("VTKFile")
    root.set("type", "UnstructuredGrid")
    root.set("version", "0.1")
    root.set("byte_order", "LittleEndian")

    ugrid = ET.SubElement(root, "UnstructuredGrid")
    n_cells = _count_cells(db)
    piece = ET.SubElement(ugrid, "Piece")
    piece.set("NumberOfPoints", str(n_nodes))
    piece.set("NumberOfCells", str(n_cells))

    # --- Points ---
    points = ET.SubElement(piece, "Points")
    add_array(points, "Points", coords_3d.ravel(), n_components=3)

    # --- Cells ---
    cells_el = ET.SubElement(piece, "Cells")
    conn_arr, offsets_arr, types_arr = _build_cell_arrays(db)
    add_array(cells_el, "connectivity", conn_arr, dtype_str="Int32")
    add_array(cells_el, "offsets", offsets_arr, dtype_str="Int32")
    add_array(cells_el, "types", types_arr, dtype_str="UInt8")

    # --- PointData ---
    point_data = ET.SubElement(piece, "PointData")

    # 変位（常に出力）
    u_3d = _reshape_to_3d_vectors(frame.displacement, n_nodes, ndpn)
    add_array(point_data, "U", u_3d.ravel(), n_components=3)

    # 速度
    if frame.velocity is not None:
        v_3d = _reshape_to_3d_vectors(frame.velocity, n_nodes, ndpn)
        add_array(point_data, "V", v_3d.ravel(), n_components=3)

    # 加速度
    if frame.acceleration is not None:
        a_3d = _reshape_to_3d_vectors(frame.acceleration, n_nodes, ndpn)
        add_array(point_data, "A", a_3d.ravel(), n_components=3)

    # 変位の大きさ（スカラー）
    u_mag = np.sqrt(np.sum(u_3d**2, axis=1))
    add_array(point_data, "U_magnitude", u_mag)

    # --- CellData（要素データ）---
    if hasattr(frame, "element_data") and frame.element_data:
        cell_data_el = ET.SubElement(piece, "CellData")
        for name, values in frame.element_data.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 1:
                add_array(cell_data_el, name, arr, n_components=1)
            else:
                add_array(cell_data_el, name, arr.ravel(), n_components=arr.shape[1])

    # XML をファイルに書き出し
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(filepath, encoding="unicode", xml_declaration=True)


def _write_pvd(
    filepath: Path,
    entries: list[tuple[float, str]],
) -> None:
    """.pvd (ParaView Data) ファイルを書き出す."""
    root = ET.Element("VTKFile")
    root.set("type", "Collection")
    root.set("version", "0.1")

    collection = ET.SubElement(root, "Collection")
    for time_val, vtu_name in entries:
        dataset = ET.SubElement(collection, "DataSet")
        dataset.set("timestep", f"{time_val:.10g}")
        dataset.set("file", vtu_name)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(filepath, encoding="unicode", xml_declaration=True)


def _add_data_array(
    parent: ET.Element,
    name: str,
    data: np.ndarray,
    *,
    n_components: int = 1,
    dtype_str: str | None = None,
) -> None:
    """DataArray 要素を追加する（ASCII フォーマット）."""
    arr = np.asarray(data)

    if dtype_str is None:
        if arr.dtype.kind == "f":
            dtype_str = "Float64"
        elif arr.dtype.kind in ("i", "u"):
            dtype_str = "Int32"
        else:
            dtype_str = "Float64"

    da = ET.SubElement(parent, "DataArray")
    da.set("type", dtype_str)
    da.set("Name", name)
    da.set("NumberOfComponents", str(n_components))
    da.set("format", "ascii")

    # データを文字列化
    if dtype_str in ("Int32", "Int64", "UInt8"):
        da.text = " ".join(str(int(v)) for v in arr.ravel())
    else:
        da.text = " ".join(f"{float(v):.10g}" for v in arr.ravel())


# VTK dtype 文字列と numpy dtype の対応
_VTK_DTYPE_MAP: dict[str, np.dtype] = {
    "Float64": np.dtype("<f8"),
    "Float32": np.dtype("<f4"),
    "Int32": np.dtype("<i4"),
    "Int64": np.dtype("<i8"),
    "UInt8": np.dtype("<u1"),
}


def _add_data_array_binary(
    parent: ET.Element,
    name: str,
    data: np.ndarray,
    *,
    n_components: int = 1,
    dtype_str: str | None = None,
) -> None:
    """DataArray 要素を追加する（Base64 バイナリフォーマット）.

    VTK XML の "binary" 形式: データは base64 エンコードされ、
    先頭に 4 バイト（UInt32）のデータ長ヘッダが付く。
    """
    arr = np.asarray(data)

    if dtype_str is None:
        if arr.dtype.kind == "f":
            dtype_str = "Float64"
        elif arr.dtype.kind in ("i", "u"):
            dtype_str = "Int32"
        else:
            dtype_str = "Float64"

    np_dtype = _VTK_DTYPE_MAP.get(dtype_str, np.dtype("<f8"))
    arr_typed = arr.ravel().astype(np_dtype)

    # VTK binary format: UInt32 header (data size in bytes) + raw data
    raw_bytes = arr_typed.tobytes()
    header = np.array([len(raw_bytes)], dtype=np.dtype("<u4")).tobytes()
    encoded = base64.b64encode(header + raw_bytes).decode("ascii")

    da = ET.SubElement(parent, "DataArray")
    da.set("type", dtype_str)
    da.set("Name", name)
    da.set("NumberOfComponents", str(n_components))
    da.set("format", "binary")
    da.text = encoded


def _count_cells(db: OutputDatabase) -> int:
    """セル数を返す."""
    if db.connectivity is None:
        return 0
    return sum(nodes.shape[0] for _, nodes in db.connectivity)


def _build_cell_arrays(
    db: OutputDatabase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VTK のセル配列を構築する.

    Returns:
        (connectivity, offsets, types) の各配列
    """
    if db.connectivity is None or len(db.connectivity) == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.uint8),
        )

    conn_list: list[int] = []
    offsets_list: list[int] = []
    types_list: list[int] = []
    offset = 0

    for vtk_type, node_array in db.connectivity:
        for row in node_array:
            for n in row:
                conn_list.append(int(n))
            offset += len(row)
            offsets_list.append(offset)
            types_list.append(vtk_type)

    return (
        np.array(conn_list, dtype=np.int32),
        np.array(offsets_list, dtype=np.int32),
        np.array(types_list, dtype=np.uint8),
    )


def _reshape_to_3d_vectors(
    dof_vector: np.ndarray,
    n_nodes: int,
    ndof_per_node: int,
) -> np.ndarray:
    """DOF ベクトルを (n_nodes, 3) の 3D ベクトルに変換する.

    梁要素（ndof=3 or 6）の場合、並進成分のみ抽出する。
    """
    result = np.zeros((n_nodes, 3), dtype=np.float64)

    if ndof_per_node <= 3:
        # 2D/3D並進 DOF のみ
        n_trans = min(ndof_per_node, 3)
        for i in range(n_nodes):
            for d in range(n_trans):
                result[i, d] = dof_vector[i * ndof_per_node + d]
    else:
        # 梁要素（6 DOF: ux, uy, uz, θx, θy, θz）
        # 並進成分（最初の3つ）のみ抽出
        n_trans = min(3, ndof_per_node)
        for i in range(n_nodes):
            for d in range(n_trans):
                result[i, d] = dof_vector[i * ndof_per_node + d]

    return result


__all__ = [
    "export_vtk",
    "VTK_VERTEX",
    "VTK_LINE",
    "VTK_TRIANGLE",
    "VTK_QUAD",
    "VTK_QUADRATIC_TRIANGLE",
]
