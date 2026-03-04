"""Abaqus .inp ファイルのパーサー.

pymesh 代替として、Abaqus入力ファイルの以下のセクションを読み込む:
  - *NODE: 節点座標
  - *ELEMENT: 要素接続配列
  - *NSET: ノードセット
  - *ELSET: 要素セット
  - *BEAM SECTION: 梁断面定義（SECTION, ELSET, MATERIAL, 寸法）
  - *TRANSVERSE SHEAR STIFFNESS: 横せん断剛性（K11, K22, K12）
  - *BOUNDARY: 境界条件（節点拘束）
  - *MATERIAL: 材料定義（NAME指定）
  - *ELASTIC: 弾性定数（E, nu）
  - *DENSITY: 密度
  - *PLASTIC: 塑性データ（降伏応力-塑性ひずみテーブル）
  - *OUTPUT, FIELD ANIMATION: アニメーション出力（独自拡張）

対応要素タイプ:
  - CPS3, CPE3: 3節点三角形（TRI3）
  - CPS4, CPS4R, CPE4, CPE4R: 4節点四角形（Q4）
  - CPS6, CPE6: 6節点三角形（TRI6）
  - B21, B22: 2節点/3節点梁要素（2D）
  - B31, B32: 2節点/3節点梁要素（3D）

制限事項:
  - *INCLUDE はサポートしない
  - パーツ/インスタンス構造（*PART, *INSTANCE）はサポートしない
  - 継続行（末尾カンマ）は要素行のみ対応
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AbaqusNode:
    """Abaqus節点データ."""

    label: int
    x: float
    y: float
    z: float = 0.0


@dataclass
class AbaqusBeamSection:
    """Abaqus梁断面定義（*BEAM SECTION）.

    Attributes:
        section_type: 断面タイプ（"RECT", "CIRC", "PIPE" 等）
        elset: 要素セット名
        material: 材料名
        dimensions: 断面寸法リスト（断面タイプに依存）
        direction: 断面方向ベクトル（オプション）
        transverse_shear: 横せん断剛性 (K11, K22, K12)。
            *TRANSVERSE SHEAR STIFFNESS で指定された場合に設定。
    """

    section_type: str
    elset: str
    material: str
    dimensions: list[float] = field(default_factory=list)
    direction: list[float] | None = None
    transverse_shear: tuple[float, float, float] | None = None


@dataclass
class AbaqusElementGroup:
    """同一タイプの要素グループ.

    Attributes:
        elem_type: 要素タイプ文字列（例: "CPS4R", "CPE3"）
        elset: 要素セット名（オプション）
        elements: [(label, [node1, node2, ...]), ...]
    """

    elem_type: str
    elset: str | None = None
    elements: list[tuple[int, list[int]]] = field(default_factory=list)


@dataclass
class AbaqusBoundary:
    """Abaqus境界条件（*BOUNDARY）.

    Attributes:
        node_label: 節点ラベル
        first_dof: 拘束開始DOF番号（1始まり）
        last_dof: 拘束終了DOF番号（1始まり）。first_dofと同じ場合は単一DOF拘束
        value: 規定変位値（デフォルト0.0）
    """

    node_label: int
    first_dof: int
    last_dof: int
    value: float = 0.0


@dataclass
class AbaqusFieldAnimation:
    """アニメーション出力設定（*OUTPUT, FIELD ANIMATION、独自拡張）.

    Abaqusにはないxkep-cae独自キーワード。
    梁要素のx,y,z軸方向からの二次元プロットを生成する。

    Attributes:
        output_dir: 出力ディレクトリ（デフォルト "animation"）
        views: 描画するビュー方向リスト（デフォルト ["xy", "xz", "yz"]）
    """

    output_dir: str = "animation"
    views: list[str] = field(default_factory=lambda: ["xy", "xz", "yz"])


@dataclass
class AbaqusMaterial:
    """Abaqus材料定義（*MATERIAL + サブキーワード）.

    *MATERIAL, NAME=... で定義された材料ブロック内の
    *ELASTIC, *DENSITY, *PLASTIC の情報を保持する。

    Attributes:
        name: 材料名（*MATERIAL の NAME= オプション）
        elastic: (E, nu) タプル。*ELASTIC で定義。None なら未定義
        density: 密度スカラー。*DENSITY で定義。None なら未定義
        plastic: 降伏応力-塑性ひずみのテーブル [(sigma_y, eps_p), ...]。
                 *PLASTIC で定義。None なら未定義
        plastic_hardening: *PLASTIC の HARDENING= オプション
                          ("ISOTROPIC", "KINEMATIC", "COMBINED")
    """

    name: str
    elastic: tuple[float, float] | None = None
    density: float | None = None
    plastic: list[tuple[float, float]] | None = None
    plastic_hardening: str = "ISOTROPIC"


@dataclass
class AbaqusMesh:
    """パース結果を保持するデータクラス.

    pymesh互換のインタフェースを提供する。
    """

    nodes: list[AbaqusNode] = field(default_factory=list)
    element_groups: list[AbaqusElementGroup] = field(default_factory=list)
    nsets: dict[str, list[int]] = field(default_factory=dict)
    elsets: dict[str, list[int]] = field(default_factory=dict)
    beam_sections: list[AbaqusBeamSection] = field(default_factory=list)
    boundaries: list[AbaqusBoundary] = field(default_factory=list)
    materials: list[AbaqusMaterial] = field(default_factory=list)
    field_animation: AbaqusFieldAnimation | None = None

    def get_node_coord_array(self) -> list[dict[str, float]]:
        """節点座標をpymesh互換の辞書リストで返す.

        Returns:
            [{"label": int, "x": float, "y": float, "z": float}, ...]
        """
        return [{"label": n.label, "x": n.x, "y": n.y, "z": n.z} for n in self.nodes]

    def get_node_labels_with_nset(self, nset_name: str) -> list[int]:
        """指定ノードセットのラベルリストを返す.

        Args:
            nset_name: ノードセット名（大文字小文字区別なし）

        Returns:
            ノードラベルのリスト

        Raises:
            KeyError: ノードセットが見つからない場合
        """
        key = nset_name.upper()
        for name, labels in self.nsets.items():
            if name.upper() == key:
                return labels
        raise KeyError(
            f"ノードセット '{nset_name}' が見つかりません。利用可能: {list(self.nsets.keys())}"
        )

    def get_element_labels_with_elset(self, elset_name: str) -> list[int]:
        """指定要素セットのラベルリストを返す.

        *ELSET で明示的に定義されたセットと、*ELEMENT の ELSET= で
        暗黙的に定義されたセットの両方を検索する。

        Args:
            elset_name: 要素セット名（大文字小文字区別なし）

        Returns:
            要素ラベルのリスト

        Raises:
            KeyError: 要素セットが見つからない場合
        """
        key = elset_name.upper()

        # 明示的な *ELSET を検索
        for name, labels in self.elsets.items():
            if name.upper() == key:
                return labels

        # *ELEMENT の ELSET= で暗黙的に定義されたものを検索
        for group in self.element_groups:
            if group.elset and group.elset.upper() == key:
                return [label for label, _ in group.elements]

        raise KeyError(
            f"要素セット '{elset_name}' が見つかりません。利用可能: {list(self.elsets.keys())}"
        )

    def get_material(self, name: str) -> AbaqusMaterial:
        """指定名の材料定義を返す.

        Args:
            name: 材料名（大文字小文字区別なし）

        Returns:
            AbaqusMaterial オブジェクト

        Raises:
            KeyError: 材料が見つからない場合
        """
        key = name.upper()
        for mat in self.materials:
            if mat.name.upper() == key:
                return mat
        available = [m.name for m in self.materials]
        raise KeyError(f"材料 '{name}' が見つかりません。利用可能: {available}")

    def get_element_array(
        self,
        allow_polymorphism: bool = False,
        invalid_node: int = 0,
    ) -> list[list[int]]:
        """要素接続配列をpymesh互換の形式で返す.

        Args:
            allow_polymorphism: Trueの場合、異なる節点数の要素を混在可能にする。
                短い要素はinvalid_nodeでパディングされる。
            invalid_node: パディングに使用する無効節点値

        Returns:
            [[label, n1, n2, ...], ...] の2次元リスト
        """
        all_elements: list[tuple[int, list[int]]] = []
        for group in self.element_groups:
            all_elements.extend(group.elements)

        if not all_elements:
            return []

        max_nodes = max(len(nodes) for _, nodes in all_elements)

        result: list[list[int]] = []
        for label, nodes in all_elements:
            row = [label] + nodes
            if allow_polymorphism and len(nodes) < max_nodes:
                row.extend([invalid_node] * (max_nodes - len(nodes)))
            result.append(row)

        return result


def _parse_keyword_options(line: str) -> dict[str, str]:
    """キーワード行のオプションを辞書にパースする.

    例: "*ELEMENT, TYPE=CPS4R, ELSET=solid" →
        {"TYPE": "CPS4R", "ELSET": "solid"}

    Args:
        line: キーワード行（*で始まる）

    Returns:
        オプション辞書（キーは大文字正規化）
    """
    parts = line.split(",")
    options: dict[str, str] = {}
    for part in parts[1:]:
        part = part.strip()
        if "=" in part:
            key, val = part.split("=", 1)
            options[key.strip().upper()] = val.strip()
        elif part:
            # 値なしオプション（例: GENERATE）
            options[part.upper()] = ""
    return options


def read_abaqus_inp(filepath: str | Path) -> AbaqusMesh:
    """Abaqus .inp ファイルを読み込む.

    Args:
        filepath: .inp ファイルのパス

    Returns:
        AbaqusMesh オブジェクト
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")

    mesh = AbaqusMesh()

    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    idx = 0
    n_lines = len(lines)

    while idx < n_lines:
        line = lines[idx].strip()

        # 空行・コメント行をスキップ
        if not line or line.startswith("**"):
            idx += 1
            continue

        # キーワード行
        if line.startswith("*"):
            keyword = line.split(",")[0].strip().upper()

            if keyword == "*NODE":
                idx = _parse_node_section(lines, idx + 1, mesh)
            elif keyword == "*ELEMENT":
                opts = _parse_keyword_options(line)
                idx = _parse_element_section(lines, idx + 1, opts, mesh)
            elif keyword == "*NSET":
                opts = _parse_keyword_options(line)
                idx = _parse_nset_section(lines, idx + 1, opts, mesh)
            elif keyword == "*ELSET":
                opts = _parse_keyword_options(line)
                idx = _parse_elset_section(lines, idx + 1, opts, mesh)
            elif keyword in ("*BEAM SECTION", "*BEAMSECTION"):
                opts = _parse_keyword_options(line)
                idx = _parse_beam_section(lines, idx + 1, opts, mesh)
            elif keyword in (
                "*TRANSVERSE SHEAR STIFFNESS",
                "*TRANSVERSESHEARSTIFFNESS",
            ):
                idx = _parse_transverse_shear_stiffness(lines, idx + 1, mesh)
            elif keyword == "*BOUNDARY":
                idx = _parse_boundary_section(lines, idx + 1, mesh)
            elif keyword == "*MATERIAL":
                opts = _parse_keyword_options(line)
                idx = _parse_material_section(lines, idx + 1, opts, mesh)
            elif keyword == "*ELASTIC":
                idx = _parse_elastic_section(lines, idx + 1, mesh)
            elif keyword == "*DENSITY":
                idx = _parse_density_section(lines, idx + 1, mesh)
            elif keyword == "*PLASTIC":
                opts = _parse_keyword_options(line)
                idx = _parse_plastic_section(lines, idx + 1, opts, mesh)
            elif keyword == "*OUTPUT":
                opts = _parse_keyword_options(line)
                if "FIELD ANIMATION" in opts or "FIELD" in opts:
                    idx = _parse_field_animation(lines, idx + 1, opts, mesh)
                else:
                    idx += 1
            else:
                idx += 1
        else:
            idx += 1

    return mesh


def _parse_node_section(
    lines: list[str],
    start_idx: int,
    mesh: AbaqusMesh,
) -> int:
    """*NODE セクションをパースする.

    形式: label, x, y [, z]

    Returns:
        次のキーワード行のインデックス
    """
    idx = start_idx
    n_lines = len(lines)

    while idx < n_lines:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        parts = line.split(",")
        label = int(parts[0].strip())
        x = float(parts[1].strip())
        y = float(parts[2].strip()) if len(parts) > 2 else 0.0
        z = float(parts[3].strip()) if len(parts) > 3 else 0.0

        mesh.nodes.append(AbaqusNode(label=label, x=x, y=y, z=z))
        idx += 1

    return idx


def _parse_element_section(
    lines: list[str],
    start_idx: int,
    opts: dict[str, str],
    mesh: AbaqusMesh,
) -> int:
    """*ELEMENT セクションをパースする.

    形式: label, n1, n2, ...
    継続行（前の行が末尾カンマ）にも対応。

    Returns:
        次のキーワード行のインデックス
    """
    elem_type = opts.get("TYPE", "UNKNOWN")
    elset = opts.get("ELSET")

    group = AbaqusElementGroup(elem_type=elem_type, elset=elset)
    idx = start_idx
    n_lines = len(lines)

    accumulated = ""

    while idx < n_lines:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        # 継続行の処理
        accumulated += line
        if accumulated.endswith(","):
            # まだ続きがある
            idx += 1
            continue

        # 完全な行をパース
        parts = [p.strip() for p in accumulated.split(",") if p.strip()]
        label = int(parts[0])
        node_ids = [int(p) for p in parts[1:]]
        group.elements.append((label, node_ids))
        accumulated = ""
        idx += 1

    # 最後の累積行がある場合
    if accumulated.strip():
        parts = [p.strip() for p in accumulated.split(",") if p.strip()]
        if parts:
            label = int(parts[0])
            node_ids = [int(p) for p in parts[1:]]
            group.elements.append((label, node_ids))

    mesh.element_groups.append(group)
    return idx


def _parse_nset_section(
    lines: list[str],
    start_idx: int,
    opts: dict[str, str],
    mesh: AbaqusMesh,
) -> int:
    """*NSET セクションをパースする.

    2つの形式に対応:
    1. 通常: ノードラベルのカンマ区切りリスト
    2. GENERATE: start, end, step

    Returns:
        次のキーワード行のインデックス
    """
    nset_name = opts.get("NSET", "UNNAMED")
    is_generate = "GENERATE" in opts

    labels: list[int] = mesh.nsets.get(nset_name, [])
    idx = start_idx
    n_lines = len(lines)

    while idx < n_lines:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        parts = [p.strip() for p in line.split(",") if p.strip()]

        if is_generate:
            # GENERATE: start, end [, step]
            start = int(parts[0])
            end = int(parts[1])
            step = int(parts[2]) if len(parts) > 2 else 1
            labels.extend(range(start, end + 1, step))
        else:
            # 通常: ラベルのリスト
            labels.extend(int(p) for p in parts)

        idx += 1

    mesh.nsets[nset_name] = labels
    return idx


def _parse_beam_section(
    lines: list[str],
    start_idx: int,
    opts: dict[str, str],
    mesh: AbaqusMesh,
) -> int:
    """*BEAM SECTION セクションをパースする.

    形式:
        *BEAM SECTION, SECTION=RECT, ELSET=beams, MATERIAL=steel
        dim1, dim2, ...       (断面寸法)
        nx, ny, nz            (断面方向ベクトル、オプション)

    Returns:
        次のキーワード行のインデックス
    """
    section_type = opts.get("SECTION", "UNKNOWN")
    elset = opts.get("ELSET", "UNNAMED")
    material = opts.get("MATERIAL", "UNNAMED")

    beam_sec = AbaqusBeamSection(
        section_type=section_type,
        elset=elset,
        material=material,
    )

    idx = start_idx
    n_lines = len(lines)
    data_line_count = 0

    while idx < n_lines:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        parts = [p.strip() for p in line.split(",") if p.strip()]

        if data_line_count == 0:
            # 1行目: 断面寸法
            beam_sec.dimensions = [float(p) for p in parts]
        elif data_line_count == 1:
            # 2行目: 断面方向ベクトル（オプション）
            beam_sec.direction = [float(p) for p in parts]

        data_line_count += 1
        idx += 1

    mesh.beam_sections.append(beam_sec)
    return idx


def _parse_transverse_shear_stiffness(
    lines: list[str],
    start_idx: int,
    mesh: AbaqusMesh,
) -> int:
    """*TRANSVERSE SHEAR STIFFNESS セクションをパースする.

    形式:
        *TRANSVERSE SHEAR STIFFNESS
        K11, K22 [, K12]

    直前の *BEAM SECTION に横せん断剛性を関連付ける。

    Returns:
        次のキーワード行のインデックス
    """
    idx = start_idx
    n_lines = len(lines)

    while idx < n_lines:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        parts = [p.strip() for p in line.split(",") if p.strip()]
        k11 = float(parts[0])
        k22 = float(parts[1]) if len(parts) > 1 else k11
        k12 = float(parts[2]) if len(parts) > 2 else 0.0

        # 直前の beam section に関連付け
        if mesh.beam_sections:
            mesh.beam_sections[-1].transverse_shear = (k11, k22, k12)

        idx += 1
        break  # データは1行のみ

    return idx


def _parse_elset_section(
    lines: list[str],
    start_idx: int,
    opts: dict[str, str],
    mesh: AbaqusMesh,
) -> int:
    """*ELSET セクションをパースする.

    2つの形式に対応:
    1. 通常: 要素ラベルのカンマ区切りリスト
    2. GENERATE: start, end, step

    Returns:
        次のキーワード行のインデックス
    """
    elset_name = opts.get("ELSET", "UNNAMED")
    is_generate = "GENERATE" in opts

    labels: list[int] = mesh.elsets.get(elset_name, [])
    idx = start_idx
    n_lines = len(lines)

    while idx < n_lines:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        parts = [p.strip() for p in line.split(",") if p.strip()]

        if is_generate:
            # GENERATE: start, end [, step]
            start = int(parts[0])
            end = int(parts[1])
            step = int(parts[2]) if len(parts) > 2 else 1
            labels.extend(range(start, end + 1, step))
        else:
            # 通常: ラベルのリスト
            labels.extend(int(p) for p in parts)

        idx += 1

    mesh.elsets[elset_name] = labels
    return idx


def _parse_boundary_section(
    lines: list[str],
    start_idx: int,
    mesh: AbaqusMesh,
) -> int:
    """*BOUNDARY セクションをパースする.

    Abaqus形式:
        node_label, first_dof, last_dof [, value]
    または:
        node_label, dof  （単一DOF拘束、value=0.0）

    Returns:
        次のキーワード行のインデックス
    """
    idx = start_idx
    n_lines = len(lines)

    while idx < n_lines:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        parts = [p.strip() for p in line.split(",") if p.strip()]
        node_label = int(parts[0])

        if len(parts) == 2:
            # node_label, dof
            dof = int(parts[1])
            mesh.boundaries.append(
                AbaqusBoundary(node_label=node_label, first_dof=dof, last_dof=dof)
            )
        elif len(parts) == 3:
            # node_label, first_dof, last_dof
            first_dof = int(parts[1])
            last_dof = int(parts[2])
            mesh.boundaries.append(
                AbaqusBoundary(node_label=node_label, first_dof=first_dof, last_dof=last_dof)
            )
        elif len(parts) >= 4:
            # node_label, first_dof, last_dof, value
            first_dof = int(parts[1])
            last_dof = int(parts[2])
            value = float(parts[3])
            mesh.boundaries.append(
                AbaqusBoundary(
                    node_label=node_label,
                    first_dof=first_dof,
                    last_dof=last_dof,
                    value=value,
                )
            )

        idx += 1

    return idx


def _parse_field_animation(
    lines: list[str],
    start_idx: int,
    opts: dict[str, str],
    mesh: AbaqusMesh,
) -> int:
    """*OUTPUT, FIELD ANIMATION セクションをパースする.

    xkep-cae独自拡張。梁要素のアニメーション出力設定。

    オプション:
        DIR=<出力ディレクトリ>  （デフォルト: "animation"）

    データ行（オプション）:
        ビュー方向のカンマ区切り（例: xy, xz, yz）

    Returns:
        次のキーワード行のインデックス
    """
    output_dir = opts.get("DIR", "animation")
    anim = AbaqusFieldAnimation(output_dir=output_dir)

    idx = start_idx
    n_lines = len(lines)

    while idx < n_lines:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        # データ行: ビュー方向リスト
        parts = [p.strip().lower() for p in line.split(",") if p.strip()]
        if parts:
            anim.views = parts

        idx += 1
        break  # データは1行のみ

    mesh.field_animation = anim
    return idx


def _parse_material_section(
    lines: list[str],
    start_idx: int,
    opts: dict[str, str],
    mesh: AbaqusMesh,
) -> int:
    """*MATERIAL セクションをパースする.

    *MATERIAL は材料定義の開始を示すヘッダーキーワード。
    NAME= オプションで材料名を指定する。後続の *ELASTIC, *DENSITY, *PLASTIC
    サブキーワードがこの材料に紐づく。

    Returns:
        次のキーワード行のインデックス（*MATERIAL 直後の行）
    """
    name = opts.get("NAME", "UNNAMED")
    mat = AbaqusMaterial(name=name)
    mesh.materials.append(mat)
    # *MATERIAL 自体にはデータ行がないので、次の行をそのまま返す
    return start_idx


def _parse_elastic_section(
    lines: list[str],
    start_idx: int,
    mesh: AbaqusMesh,
) -> int:
    """*ELASTIC セクションをパースする.

    形式:
        *ELASTIC [, TYPE=ISOTROPIC]
        E, nu

    直前の *MATERIAL に弾性定数を関連付ける。

    Returns:
        次のキーワード行のインデックス
    """
    idx = start_idx
    n_lines = len(lines)

    while idx < n_lines:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        parts = [p.strip() for p in line.split(",") if p.strip()]
        E = float(parts[0])
        nu = float(parts[1]) if len(parts) > 1 else 0.0

        if mesh.materials:
            mesh.materials[-1].elastic = (E, nu)

        idx += 1
        break  # データは1行のみ

    return idx


def _parse_density_section(
    lines: list[str],
    start_idx: int,
    mesh: AbaqusMesh,
) -> int:
    """*DENSITY セクションをパースする.

    形式:
        *DENSITY
        rho

    直前の *MATERIAL に密度を関連付ける。

    Returns:
        次のキーワード行のインデックス
    """
    idx = start_idx
    n_lines = len(lines)

    while idx < n_lines:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        parts = [p.strip() for p in line.split(",") if p.strip()]
        rho = float(parts[0])

        if mesh.materials:
            mesh.materials[-1].density = rho

        idx += 1
        break  # データは1行のみ

    return idx


def _parse_plastic_section(
    lines: list[str],
    start_idx: int,
    opts: dict[str, str],
    mesh: AbaqusMesh,
) -> int:
    """*PLASTIC セクションをパースする.

    形式:
        *PLASTIC [, HARDENING=ISOTROPIC|KINEMATIC|COMBINED]
        sigma_y1, eps_p1
        sigma_y2, eps_p2
        ...

    降伏応力-塑性ひずみの表データを複数行で定義する。
    直前の *MATERIAL に塑性データを関連付ける。

    Returns:
        次のキーワード行のインデックス
    """
    hardening = opts.get("HARDENING", "ISOTROPIC")
    table: list[tuple[float, float]] = []

    idx = start_idx
    n_lines = len(lines)

    while idx < n_lines:
        line = lines[idx].strip()
        if not line or line.startswith("**"):
            idx += 1
            continue
        if line.startswith("*"):
            break

        parts = [p.strip() for p in line.split(",") if p.strip()]
        sigma_y = float(parts[0])
        eps_p = float(parts[1]) if len(parts) > 1 else 0.0
        table.append((sigma_y, eps_p))
        idx += 1

    if mesh.materials:
        mesh.materials[-1].plastic = table
        mesh.materials[-1].plastic_hardening = hardening

    return idx


# ====================================================================
# .inp 書き出し — データ構造
# ====================================================================


@dataclass
class InpInitialCondition:
    """初期条件（*INITIAL CONDITIONS）.

    Attributes:
        type: 条件タイプ（"VELOCITY", "TEMPERATURE" 等）
        data: [(node_or_nset, dof_or_component, value), ...]
    """

    type: str = "VELOCITY"
    data: list[tuple[str | int, int, float]] = field(default_factory=list)


@dataclass
class InpOutputRequest:
    """出力要求（*OUTPUT）.

    Attributes:
        domain: "FIELD" or "HISTORY"
        frequency: 出力頻度（ステップ数単位、0=最終のみ）
        variables: 変数リスト ["U", "S", "E", "RF", "COORD", ...]
        nset: ノードセット名（HISTORY のみ）
        elset: 要素セット名（HISTORY のみ）
    """

    domain: str = "FIELD"
    frequency: int = 1
    variables: list[str] = field(default_factory=lambda: ["U", "S", "RF"])
    nset: str | None = None
    elset: str | None = None


@dataclass
class InpAnimationRequest:
    """アニメーション出力要求（*ANIMATION、独自拡張）.

    Attributes:
        output_dir: 出力ディレクトリ
        views: ビュー方向リスト
        frequency: 出力頻度
    """

    output_dir: str = "animation"
    views: list[str] = field(default_factory=lambda: ["xy", "xz", "yz"])
    frequency: int = 1


@dataclass
class InpSurfaceDef:
    """サーフェス定義（*SURFACE）.

    Abaqus の *SURFACE キーワードに対応。要素セットベースで
    接触サーフェスを定義する。

    Attributes:
        name: サーフェス名
        type: サーフェスタイプ ("ELEMENT" or "NODE")
        elset: 要素セット名（TYPE=ELEMENT の場合）
        nset: ノードセット名（TYPE=NODE の場合）
    """

    name: str = ""
    type: str = "ELEMENT"
    elset: str | None = None
    nset: str | None = None


@dataclass
class InpSurfaceInteraction:
    """サーフェスインタラクション定義（*SURFACE INTERACTION + *SURFACE BEHAVIOR）.

    Abaqus の *SURFACE INTERACTION キーワードと、そのサブキーワード
    *SURFACE BEHAVIOR に対応。接触挙動（ペナルティ法/ハード接触）を定義する。

    Attributes:
        name: インタラクション名
        pressure_overclosure: 圧力-貫入関係 ("HARD", "LINEAR", "EXPONENTIAL", "TABULAR")
        k_pen: ペナルティ剛性（pressure_overclosure="LINEAR" 時の勾配、0=デフォルト）
        friction: 摩擦係数（0=摩擦なし）
        algorithm: 接触アルゴリズム ("NCP" or "AL"、xkep-cae 独自拡張)
        mortar: Mortar 積分の有効化（xkep-cae 独自拡張）
        options: その他のキー=値オプション
    """

    name: str = "CONTACT_PROP"
    pressure_overclosure: str = "HARD"
    k_pen: float = 0.0
    friction: float = 0.0
    algorithm: str = "NCP"
    mortar: bool = True
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class InpContactDef:
    """接触定義（General Contact）.

    Abaqus の General Contact に対応。ステップレベルで
    *CONTACT / *CONTACT INCLUSIONS / *CONTACT PROPERTY ASSIGNMENT を出力する。
    サーフェスとインタラクションはモデルレベルで事前定義される。

    Attributes:
        interaction: サーフェスインタラクション参照名
        inclusions: 接触ペアリスト [("surf1", "surf2"), ...]
            空リスト=全自己接触 (ALLEXT, ALLEXT)
        exclude_same_layer: 同層除外（xkep-cae 独自拡張）
        surfaces: サーフェス定義リスト（モデルレベルで出力される）
        surface_interactions: インタラクション定義リスト（モデルレベルで出力される）
    """

    interaction: str = "CONTACT_PROP"
    inclusions: list[tuple[str, str]] = field(default_factory=list)
    exclude_same_layer: bool = True
    surfaces: list[InpSurfaceDef] = field(default_factory=list)
    surface_interactions: list[InpSurfaceInteraction] = field(default_factory=list)


@dataclass
class InpStep:
    """解析ステップ定義.

    Abaqus の *STEP ～ *END STEP ブロックに対応。

    Attributes:
        name: ステップ名（None = 自動番号付け）
        procedure: プロシージャ ("STATIC" or "DYNAMIC")
        nlgeom: 幾何学非線形
        unsymm: 非対称行列ソルバー
        inc: 最大インクリメント数
        time_params: プロシージャ依存の時間パラメータ
            STATIC: (initial_inc, total_time, min_inc, max_inc)
            DYNAMIC: (initial_dt, step_time, min_dt, max_dt)
        boundaries: ステップ内の境界条件
            [(node_or_nset, first_dof, last_dof, value), ...]
        boundary_type: "DISPLACEMENT" or "VELOCITY"
        output_requests: 出力要求リスト
        animation: アニメーション出力（独自拡張）
        contact: 接触定義（独自拡張）
        cloads: 集中荷重 [(node_or_nset, dof, magnitude), ...]
        dloads: 分布荷重 [(elset, load_type, magnitude), ...]
    """

    name: str | None = None
    procedure: str = "STATIC"
    nlgeom: bool = True
    unsymm: bool = False
    inc: int = 1000
    time_params: tuple[float, ...] | None = None
    boundaries: list[tuple[int | str, int, int, float]] = field(default_factory=list)
    boundary_type: str = "DISPLACEMENT"
    output_requests: list[InpOutputRequest] = field(default_factory=list)
    animation: InpAnimationRequest | None = None
    contact: InpContactDef | None = None
    cloads: list[tuple[int | str, int, float]] = field(default_factory=list)
    dloads: list[tuple[str, str, float]] = field(default_factory=list)


# ====================================================================
# .inp 書き出し — レガシー API（後方互換）
# ====================================================================


def write_abaqus_inp(
    filepath: str | Path,
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    *,
    elem_type: str = "B31",
    title: str = "xkep-cae export",
    nsets: dict[str, list[int]] | None = None,
    elsets: dict[str, list[int]] | None = None,
    boundaries: list[tuple[int, int, int, float]] | None = None,
    material_name: str = "STEEL",
    E: float = 0.0,
    nu: float = 0.0,
    density: float = 0.0,
    beam_section_type: str = "CIRC",
    beam_section_dims: list[float] | None = None,
    beam_section_direction: list[float] | None = None,
) -> Path:
    """撚線メッシュ等を Abaqus .inp 形式で書き出す.

    Args:
        filepath: 出力ファイルパス
        node_coords: (n_nodes, 2 or 3) 節点座標
        connectivity: (n_elems, nodes_per_elem) 要素接続（0始まり）
        elem_type: 要素タイプ文字列（"B31", "CPS4R" 等）
        title: ヘッダータイトル
        nsets: ノードセット {名前: [node_label, ...]}（1始まり）
        elsets: 要素セット {名前: [elem_label, ...]}（1始まり）
        boundaries: 境界条件 [(node_label, first_dof, last_dof, value), ...]
        material_name: 材料名
        E: ヤング率（0 の場合 *MATERIAL セクションを省略）
        nu: ポアソン比
        density: 密度（0 の場合 *DENSITY を省略）
        beam_section_type: 梁断面タイプ
        beam_section_dims: 梁断面寸法リスト
        beam_section_direction: 梁断面方向ベクトル

    Returns:
        書き出したファイルの Path
    """
    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # --- Header ---
    lines.append("*HEADING")
    lines.append(f"** {title}")
    lines.append("** Generated by xkep-cae")
    lines.append("**")

    # --- Nodes (1-based labels) ---
    lines.append("*NODE")
    ndim = node_coords.shape[1]
    for i in range(node_coords.shape[0]):
        label = i + 1
        if ndim == 2:
            lines.append(f"{label}, {node_coords[i, 0]:.10g}, {node_coords[i, 1]:.10g}")
        else:
            lines.append(
                f"{label}, {node_coords[i, 0]:.10g}, "
                f"{node_coords[i, 1]:.10g}, {node_coords[i, 2]:.10g}"
            )

    # --- Elements (1-based labels) ---
    elset_name = "ALL_ELEMS"
    lines.append(f"*ELEMENT, TYPE={elem_type}, ELSET={elset_name}")
    for i in range(connectivity.shape[0]):
        label = i + 1
        node_labels = ", ".join(str(int(n) + 1) for n in connectivity[i])
        lines.append(f"{label}, {node_labels}")

    # --- Node sets ---
    if nsets:
        for name, labels in nsets.items():
            lines.append(f"*NSET, NSET={name}")
            # 8ラベルごとに改行
            for j in range(0, len(labels), 8):
                chunk = labels[j : j + 8]
                lines.append(", ".join(str(lb) for lb in chunk))

    # --- Element sets ---
    if elsets:
        for name, labels in elsets.items():
            lines.append(f"*ELSET, ELSET={name}")
            for j in range(0, len(labels), 8):
                chunk = labels[j : j + 8]
                lines.append(", ".join(str(lb) for lb in chunk))

    # --- Material ---
    if E > 0:
        lines.append(f"*MATERIAL, NAME={material_name}")
        lines.append("*ELASTIC")
        lines.append(f"{E:.6g}, {nu:.6g}")
        if density > 0:
            lines.append("*DENSITY")
            lines.append(f"{density:.6g}")

    # --- Beam section ---
    if beam_section_dims is not None:
        lines.append(
            f"*BEAM SECTION, SECTION={beam_section_type}, "
            f"ELSET={elset_name}, MATERIAL={material_name}"
        )
        lines.append(", ".join(f"{d:.6g}" for d in beam_section_dims))
        if beam_section_direction is not None:
            lines.append(", ".join(f"{d:.6g}" for d in beam_section_direction))

    # --- Boundaries ---
    if boundaries:
        lines.append("*BOUNDARY")
        for node_label, first_dof, last_dof, value in boundaries:
            if value == 0.0:
                lines.append(f"{node_label}, {first_dof}, {last_dof}")
            else:
                lines.append(f"{node_label}, {first_dof}, {last_dof}, {value:.10g}")

    # --- End ---
    lines.append("*END STEP")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ====================================================================
# .inp 書き出し — Abaqus互換 API（model/step 分離）
# ====================================================================


def _write_step_block(step: InpStep, step_index: int) -> list[str]:
    """1ステップ分の *STEP ～ *END STEP ブロックを生成する.

    Args:
        step: ステップ定義
        step_index: ステップ番号（1始まり、ステップ名自動生成用）

    Returns:
        行のリスト
    """
    lines: list[str] = []

    # --- *STEP キーワード ---
    step_opts: list[str] = []
    step_opts.append(f"INC={step.inc}")
    if step.nlgeom:
        step_opts.append("NLGEOM=YES")
    if step.unsymm:
        step_opts.append("UNSYMM=YES")
    lines.append("*STEP, " + ", ".join(step_opts))

    # ステップ名（*STEP の直後の行）
    step_name = step.name if step.name else f"Step-{step_index}"
    lines.append(step_name)

    # --- プロシージャ ---
    proc = step.procedure.upper()
    if proc == "STATIC":
        lines.append("*STATIC")
        if step.time_params:
            lines.append(", ".join(f"{v:.6g}" for v in step.time_params))
    elif proc == "DYNAMIC":
        lines.append("*DYNAMIC")
        if step.time_params:
            lines.append(", ".join(f"{v:.6g}" for v in step.time_params))
        else:
            # デフォルト: 0.01, 1.0, 1e-8, 1.0
            lines.append("0.01, 1., 1e-08, 1.")
    else:
        lines.append(f"*{proc}")
        if step.time_params:
            lines.append(", ".join(f"{v:.6g}" for v in step.time_params))

    # --- 接触定義（ステップ内差分指定のみ）---
    # General Contact の主定義はモデルレベル（write_abaqus_model）で出力済み。
    # ステップ内で接触ドメインを変更する場合のみここで差分出力する。
    # （現在は差分指定未サポート — 将来拡張用のプレースホルダ）

    # --- 境界条件 ---
    if step.boundaries:
        bc_type = step.boundary_type.upper()
        if bc_type == "DISPLACEMENT":
            lines.append("*BOUNDARY, TYPE=DISPLACEMENT")
        elif bc_type == "VELOCITY":
            lines.append("*BOUNDARY, TYPE=VELOCITY")
        else:
            lines.append("*BOUNDARY")
        for entry in step.boundaries:
            node_or_nset, first_dof, last_dof, value = entry
            if value == 0.0:
                lines.append(f"{node_or_nset}, {first_dof}, {last_dof}")
            else:
                lines.append(f"{node_or_nset}, {first_dof}, {last_dof}, {value:.10g}")

    # --- 集中荷重 ---
    if step.cloads:
        lines.append("*CLOAD")
        for node_or_nset, dof, magnitude in step.cloads:
            lines.append(f"{node_or_nset}, {dof}, {magnitude:.10g}")

    # --- 分布荷重 ---
    if step.dloads:
        lines.append("*DLOAD")
        for elset, load_type, magnitude in step.dloads:
            lines.append(f"{elset}, {load_type}, {magnitude:.10g}")

    # --- 出力要求 ---
    for out_req in step.output_requests:
        domain = out_req.domain.upper()
        if domain == "FIELD":
            lines.append(f"*OUTPUT, FIELD, FREQUENCY={out_req.frequency}")
            if out_req.variables:
                # ノード変数とエネルギー変数を分類
                node_vars = [
                    v for v in out_req.variables if v in ("U", "V", "A", "RF", "CF", "COORD")
                ]
                elem_vars = [
                    v for v in out_req.variables if v in ("S", "E", "PE", "SE", "SK", "SF")
                ]
                energy_vars = [
                    v
                    for v in out_req.variables
                    if v in ("ETOTAL", "ALLKE", "ALLIE", "ALLSE", "ALLPD", "ALLAE")
                ]
                other_vars = [
                    v for v in out_req.variables if v not in node_vars + elem_vars + energy_vars
                ]
                if node_vars:
                    lines.append("*NODE OUTPUT")
                    lines.append(", ".join(node_vars))
                if elem_vars:
                    lines.append("*ELEMENT OUTPUT")
                    lines.append(", ".join(elem_vars))
                if energy_vars:
                    lines.append("*ENERGY OUTPUT")
                    lines.append(", ".join(energy_vars))
                if other_vars:
                    lines.append("*NODE OUTPUT")
                    lines.append(", ".join(other_vars))
        elif domain == "HISTORY":
            nset_part = f", NSET={out_req.nset}" if out_req.nset else ""
            elset_part = f", ELSET={out_req.elset}" if out_req.elset else ""
            lines.append(f"*OUTPUT, HISTORY, FREQUENCY={out_req.frequency}{nset_part}{elset_part}")
            if out_req.variables:
                lines.append(", ".join(out_req.variables))

    # --- アニメーション（独自拡張）---
    if step.animation is not None:
        anim = step.animation
        lines.append(f"*OUTPUT, FIELD ANIMATION, DIR={anim.output_dir}, FREQUENCY={anim.frequency}")
        lines.append(", ".join(anim.views))

    # --- *END STEP ---
    lines.append("*END STEP")
    return lines


def write_abaqus_model(
    filepath: str | Path,
    node_coords: np.ndarray,
    connectivity: np.ndarray,
    *,
    elem_type: str = "B31",
    title: str = "xkep-cae export",
    nsets: dict[str, list[int]] | None = None,
    elsets: dict[str, list[int]] | None = None,
    material_name: str = "STEEL",
    E: float = 0.0,
    nu: float = 0.0,
    density: float = 0.0,
    beam_section_type: str = "CIRC",
    beam_section_dims: list[float] | None = None,
    beam_section_direction: list[float] | None = None,
    initial_conditions: list[InpInitialCondition] | None = None,
    steps: list[InpStep] | None = None,
) -> Path:
    """Abaqus互換 .inp ファイルを書き出す（model/step 分離版）.

    Abaqus のモデルレベル/ステップレベルを正しく分離して出力する。

    モデルレベル（*STEP の前）:
      *HEADING, *NODE, *ELEMENT, *NSET, *ELSET,
      *MATERIAL (*ELASTIC, *DENSITY, *PLASTIC),
      *BEAM SECTION, *INITIAL CONDITIONS

    ステップレベル（*STEP ～ *END STEP の間）:
      *STATIC / *DYNAMIC（プロシージャ）,
      *BOUNDARY, *CLOAD, *DLOAD,
      *OUTPUT, *ANIMATION, *CONTACT

    Args:
        filepath: 出力ファイルパス
        node_coords: (n_nodes, 2 or 3) 節点座標
        connectivity: (n_elems, nodes_per_elem) 要素接続（0始まり）
        elem_type: 要素タイプ文字列
        title: ヘッダータイトル
        nsets: ノードセット {名前: [node_label, ...]}（1始まり）
        elsets: 要素セット {名前: [elem_label, ...]}（1始まり）
        material_name: 材料名
        E: ヤング率（0 の場合 *MATERIAL を省略）
        nu: ポアソン比
        density: 密度（0 の場合 *DENSITY を省略）
        beam_section_type: 梁断面タイプ
        beam_section_dims: 梁断面寸法リスト
        beam_section_direction: 梁断面方向ベクトル
        initial_conditions: 初期条件リスト
        steps: ステップ定義リスト

    Returns:
        書き出したファイルの Path
    """
    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # ============================================================
    # MODEL LEVEL
    # ============================================================

    # --- Header ---
    lines.append("*HEADING")
    lines.append(f"** {title}")
    lines.append("** Generated by xkep-cae")
    lines.append("**")

    # --- Nodes (1-based labels) ---
    lines.append("*NODE")
    ndim = node_coords.shape[1]
    for i in range(node_coords.shape[0]):
        label = i + 1
        if ndim == 2:
            lines.append(f"{label}, {node_coords[i, 0]:.10g}, {node_coords[i, 1]:.10g}")
        else:
            lines.append(
                f"{label}, {node_coords[i, 0]:.10g}, "
                f"{node_coords[i, 1]:.10g}, {node_coords[i, 2]:.10g}"
            )

    # --- Elements (1-based labels) ---
    elset_all = "ALL_ELEMS"
    lines.append(f"*ELEMENT, TYPE={elem_type}, ELSET={elset_all}")
    for i in range(connectivity.shape[0]):
        label = i + 1
        node_labels = ", ".join(str(int(n) + 1) for n in connectivity[i])
        lines.append(f"{label}, {node_labels}")

    # --- Node sets ---
    if nsets:
        for name, labels in nsets.items():
            lines.append(f"*NSET, NSET={name}")
            for j in range(0, len(labels), 8):
                chunk = labels[j : j + 8]
                lines.append(", ".join(str(lb) for lb in chunk))

    # --- Element sets ---
    if elsets:
        for name, labels in elsets.items():
            lines.append(f"*ELSET, ELSET={name}")
            for j in range(0, len(labels), 8):
                chunk = labels[j : j + 8]
                lines.append(", ".join(str(lb) for lb in chunk))

    # --- Material ---
    if E > 0:
        lines.append(f"*MATERIAL, NAME={material_name}")
        lines.append("*ELASTIC")
        lines.append(f"{E:.6g}, {nu:.6g}")
        if density > 0:
            lines.append("*DENSITY")
            lines.append(f"{density:.6g}")

    # --- Beam section ---
    if beam_section_dims is not None:
        lines.append(
            f"*BEAM SECTION, SECTION={beam_section_type}, "
            f"ELSET={elset_all}, MATERIAL={material_name}"
        )
        lines.append(", ".join(f"{d:.6g}" for d in beam_section_dims))
        if beam_section_direction is not None:
            lines.append(", ".join(f"{d:.6g}" for d in beam_section_direction))

    # --- Surface / Surface Interaction（接触定義のモデルレベル部分）---
    # ステップの contact から surface/interaction 定義を収集
    _written_surfaces: set[str] = set()
    _written_interactions: set[str] = set()
    if steps:
        for step in steps:
            if step.contact is not None:
                for surf in step.contact.surfaces:
                    if surf.name and surf.name not in _written_surfaces:
                        _written_surfaces.add(surf.name)
                        surf_type = surf.type.upper()
                        lines.append(f"*SURFACE, TYPE={surf_type}, NAME={surf.name}")
                        if surf_type == "ELEMENT" and surf.elset:
                            lines.append(f"{surf.elset},")
                        elif surf_type == "NODE" and surf.nset:
                            lines.append(f"{surf.nset},")
                for si in step.contact.surface_interactions:
                    if si.name and si.name not in _written_interactions:
                        _written_interactions.add(si.name)
                        lines.append(f"*SURFACE INTERACTION, NAME={si.name}")
                        # *SURFACE BEHAVIOR (PRESSURE-OVERCLOSURE)
                        po = si.pressure_overclosure.upper()
                        if po == "LINEAR" and si.k_pen > 0:
                            lines.append(f"*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE={po}")
                            lines.append(f"{si.k_pen:.6g}, 0.")
                        else:
                            lines.append(f"*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE={po}")
                        # 摩擦
                        if si.friction > 0:
                            lines.append("*FRICTION")
                            lines.append(f"{si.friction:.6g}")
                        # xkep-cae 独自拡張: アルゴリズム・Mortar
                        ext_opts: list[str] = []
                        if si.algorithm:
                            ext_opts.append(f"ALGORITHM={si.algorithm}")
                        if si.mortar:
                            ext_opts.append("MORTAR=YES")
                        for k, v in si.options.items():
                            ext_opts.append(f"{k}={v}")
                        if ext_opts:
                            lines.append("** XKEP-CAE: " + ", ".join(ext_opts))

    # --- General Contact 宣言（モデルレベル、Abaqus/Standard 互換）---
    _contact_written = False
    if steps:
        for step in steps:
            if step.contact is not None and not _contact_written:
                _contact_written = True
                c = step.contact
                lines.append("*CONTACT")
                # *CONTACT INCLUSIONS
                lines.append("*CONTACT INCLUSIONS")
                if c.inclusions:
                    for surf1, surf2 in c.inclusions:
                        lines.append(f"{surf1}, {surf2}")
                else:
                    lines.append("ALLEXT, ALLEXT")
                # *CONTACT PROPERTY ASSIGNMENT
                lines.append("*CONTACT PROPERTY ASSIGNMENT")
                if c.inclusions:
                    for surf1, surf2 in c.inclusions:
                        lines.append(f"{surf1}, {surf2}, {c.interaction}")
                else:
                    lines.append(f", , {c.interaction}")
                # xkep-cae 独自拡張: 同層除外
                if c.exclude_same_layer:
                    lines.append("** XKEP-CAE: EXCLUDE_SAME_LAYER=YES")
                break

    # --- Initial conditions ---
    if initial_conditions:
        for ic in initial_conditions:
            lines.append(f"*INITIAL CONDITIONS, TYPE={ic.type}")
            for node_or_nset, dof, value in ic.data:
                lines.append(f"{node_or_nset}, {dof}, {value:.10g}")

    # ============================================================
    # STEP LEVEL
    # ============================================================

    if steps:
        for idx, step in enumerate(steps, start=1):
            lines.extend(_write_step_block(step, idx))
    else:
        # ステップ未定義の場合: 空のステップを1つ生成（後方互換）
        lines.append("*STEP, INC=1000, NLGEOM=YES")
        lines.append("Step-1")
        lines.append("*STATIC")
        lines.append("*END STEP")

    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
