"""Abaqus .inp ファイルのパーサー.

pymesh 代替として、Abaqus入力ファイルの以下のセクションを読み込む:
  - *NODE: 節点座標
  - *ELEMENT: 要素接続配列
  - *NSET: ノードセット
  - *ELSET: 要素セット
  - *BEAM SECTION: 梁断面定義（SECTION, ELSET, MATERIAL, 寸法）
  - *TRANSVERSE SHEAR STIFFNESS: 横せん断剛性（K11, K22, K12）
  - *BOUNDARY: 境界条件（節点拘束）
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
