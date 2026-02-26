"""撚線（twisted wire）メッシュ生成ファクトリ.

理想的な撚線幾何（ヘリカル配置）に基づく梁メッシュを生成する。
撚ピッチと素線配置パターンから、各素線の中心線座標を計算し、
CR梁/Timo3D接触解析に必要な node_coords, connectivity, radii を返す。

== 撚線構造パターン ==

1+6型（7本）: 中心1本 + 第1層6本
1+6+12型（19本）: 中心1本 + 第1層6本 + 第2層12本
3本:  中心なし、3本が互いに撚り合う（3本対称配置）

== ヘリックス幾何 ==

素線中心線は中心軸まわりのヘリックス:
  x_i(s) = R_layer * cos(2π·s/pitch + φ_i)
  y_i(s) = R_layer * sin(2π·s/pitch + φ_i)
  z_i(s) = s

ここで:
  R_layer: 層の配置半径（中心線から素線中心までの距離）
  pitch:   撚ピッチ（1周の軸方向長さ）
  φ_i:    素線iの初期位相角

参考文献:
  Costello, G.A. "Theory of Wire Rope"
  Cardou, A. & Jolicoeur, C. (1997) "Mechanical Models of Helical Strands"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class StrandInfo:
    """1素線の情報.

    Attributes:
        strand_id: 素線のID
        layer: 層番号（0=中心, 1=第1層, ...）
        angle_offset: 初期位相角 [rad]
        lay_radius: 配置半径（中心軸から素線中心まで）
        wire_radius: 素線の断面半径
        lay_direction: 撚り方向 (+1 = S撚り(右), -1 = Z撚り(左))
    """

    strand_id: int
    layer: int
    angle_offset: float
    lay_radius: float
    wire_radius: float
    lay_direction: int = 1  # +1=S撚り, -1=Z撚り


@dataclass
class TwistedWireMesh:
    """撚線メッシュ.

    Attributes:
        node_coords: (n_nodes, 3) 節点座標
        connectivity: (n_elems, 2) 要素接続（節点インデックス）
        strand_node_ranges: 各素線の節点範囲 [(start, end), ...]
        strand_elem_ranges: 各素線の要素範囲 [(start, end), ...]
        strand_infos: 各素線の情報
        n_strands: 素線本数
        wire_radius: 素線断面半径（均一の場合）
        pitch: 撚ピッチ
        length: モデル長さ（軸方向）
        n_elems_per_strand: 1素線あたりの要素数
    """

    node_coords: np.ndarray
    connectivity: np.ndarray
    strand_node_ranges: list[tuple[int, int]]
    strand_elem_ranges: list[tuple[int, int]]
    strand_infos: list[StrandInfo]
    n_strands: int
    wire_radius: float
    pitch: float
    length: float
    n_elems_per_strand: int

    @property
    def n_nodes(self) -> int:
        """総節点数."""
        return self.node_coords.shape[0]

    @property
    def n_elems(self) -> int:
        """総要素数."""
        return self.connectivity.shape[0]

    @property
    def radii(self) -> np.ndarray:
        """要素ごとの断面半径ベクトル."""
        return np.full(self.n_elems, self.wire_radius)

    def strand_nodes(self, strand_id: int) -> np.ndarray:
        """指定素線の節点インデックスを返す."""
        start, end = self.strand_node_ranges[strand_id]
        return np.arange(start, end)

    def strand_elems(self, strand_id: int) -> np.ndarray:
        """指定素線の要素インデックスを返す."""
        start, end = self.strand_elem_ranges[strand_id]
        return np.arange(start, end)

    def build_elem_layer_map(self) -> dict[int, int]:
        """要素インデックス→層番号のマッピングを構築する.

        各素線の層情報（StrandInfo.layer）を全要素に展開する。
        接触ペアのフィルタリング（段階的アクティベーション）に使用。

        Returns:
            {elem_index: layer_number} の辞書
        """
        lmap: dict[int, int] = {}
        for info in self.strand_infos:
            elems = self.strand_elems(info.strand_id)
            for e in elems:
                lmap[int(e)] = info.layer
        return lmap


def _helix_points(
    n_points: int,
    lay_radius: float,
    pitch: float,
    length: float,
    angle_offset: float = 0.0,
    lay_direction: int = 1,
) -> np.ndarray:
    """ヘリカル中心線の離散点を生成する.

    Args:
        n_points: 離散点数
        lay_radius: 配置半径
        pitch: 撚ピッチ
        length: 軸方向長さ
        angle_offset: 初期位相角 [rad]
        lay_direction: +1=S撚り, -1=Z撚り

    Returns:
        coords: (n_points, 3) 座標配列。z軸が撚線軸方向。
    """
    coords = np.zeros((n_points, 3))
    for i in range(n_points):
        z = i * length / (n_points - 1)
        theta = lay_direction * 2.0 * math.pi * z / pitch + angle_offset
        coords[i, 0] = lay_radius * math.cos(theta)
        coords[i, 1] = lay_radius * math.sin(theta)
        coords[i, 2] = z
    return coords


def _straight_points(n_points: int, length: float) -> np.ndarray:
    """直線（中心素線）の離散点を生成する.

    Args:
        n_points: 離散点数
        length: 長さ

    Returns:
        coords: (n_points, 3) 座標配列
    """
    coords = np.zeros((n_points, 3))
    coords[:, 2] = np.linspace(0.0, length, n_points)
    return coords


def make_strand_layout(
    n_strands: int,
    wire_radius: float,
    *,
    gap: float = 0.0,
    lay_direction: int = 1,
) -> list[StrandInfo]:
    """撚線素線配置を生成する.

    標準的な撚線パターン:
      3本:  中心なし、120°配置（三つ撚り）
      7本:  中心1本 + 第1層6本（1+6型）
      19本: 中心1本 + 第1層6本 + 第2層12本（1+6+12型）
      37本: 中心1本 + 第1層6本 + 第2層12本 + 第3層18本

    一般 n 本: n = 1 + 6 + 12 + 18 + ... = 1 + Σ 6k  (k=1,2,3,...)
    特殊: 3本は中心なし三つ撚り

    Args:
        n_strands: 素線本数 (3, 7, 19, 37, ...)
        wire_radius: 素線断面半径
        gap: 素線間の初期ギャップ（0 = 密着）
        lay_direction: 撚り方向 (+1=S, -1=Z)

    Returns:
        素線情報のリスト
    """
    d = 2.0 * wire_radius
    infos: list[StrandInfo] = []
    sid = 0

    if n_strands == 3:
        # 三つ撚り: 中心なし、3本が120°配置
        # 配置半径 = d/√3（三角形の外接円半径、各素線が互いに接する）
        r_lay = (d + gap) / math.sqrt(3.0)
        for k in range(3):
            angle = 2.0 * math.pi * k / 3.0
            infos.append(
                StrandInfo(
                    strand_id=sid,
                    layer=1,
                    angle_offset=angle,
                    lay_radius=r_lay,
                    wire_radius=wire_radius,
                    lay_direction=lay_direction,
                )
            )
            sid += 1
        return infos

    # 一般パターン: 中心1本 + 同心円層
    # 層 k の素線数 = 6k, 配置半径 = k * (d + gap)
    if n_strands >= 1:
        # 中心素線（直線）
        infos.append(
            StrandInfo(
                strand_id=sid,
                layer=0,
                angle_offset=0.0,
                lay_radius=0.0,
                wire_radius=wire_radius,
                lay_direction=0,
            )
        )
        sid += 1

    remaining = n_strands - 1
    layer = 1
    while remaining > 0:
        n_in_layer = 6 * layer
        actual = min(n_in_layer, remaining)
        r_lay = layer * (d + gap)
        # 交互撚り: 奇数層=lay_direction, 偶数層=-lay_direction
        layer_dir = lay_direction if (layer % 2 == 1) else -lay_direction
        for k in range(actual):
            angle = 2.0 * math.pi * k / n_in_layer
            infos.append(
                StrandInfo(
                    strand_id=sid,
                    layer=layer,
                    angle_offset=angle,
                    lay_radius=r_lay,
                    wire_radius=wire_radius,
                    lay_direction=layer_dir,
                )
            )
            sid += 1
        remaining -= actual
        layer += 1

    return infos


def make_twisted_wire_mesh(
    n_strands: int,
    wire_diameter: float,
    pitch: float,
    length: float,
    n_elems_per_strand: int,
    *,
    gap: float = 0.0,
    lay_direction: int = 1,
    n_pitches: float | None = None,
) -> TwistedWireMesh:
    """撚線メッシュを生成するファクトリ関数.

    理想的なヘリカル配置に基づき、各素線の中心線を離散化して
    梁要素メッシュを構築する。

    Args:
        n_strands: 素線本数 (3, 7, 19, 37, ...)
        wire_diameter: 素線直径 [m]
        pitch: 撚ピッチ [m]（1回転あたりの軸方向長さ）
        length: モデル長さ [m]（軸方向）。n_pitchesが指定された場合は無視。
        n_elems_per_strand: 1素線あたりの要素分割数
        gap: 素線間初期ギャップ [m]（0=密着）
        lay_direction: 撚り方向 (+1=S撚り, -1=Z撚り)
        n_pitches: モデル長さをピッチ数で指定（length を上書き）

    Returns:
        TwistedWireMesh インスタンス
    """
    wire_radius = wire_diameter / 2.0

    if n_pitches is not None:
        length = n_pitches * pitch

    # 素線配置を決定
    layout = make_strand_layout(
        n_strands,
        wire_radius,
        gap=gap,
        lay_direction=lay_direction,
    )

    n_nodes_per_strand = n_elems_per_strand + 1
    total_nodes = n_strands * n_nodes_per_strand
    total_elems = n_strands * n_elems_per_strand

    node_coords = np.zeros((total_nodes, 3))
    connectivity = np.zeros((total_elems, 2), dtype=int)
    strand_node_ranges: list[tuple[int, int]] = []
    strand_elem_ranges: list[tuple[int, int]] = []

    for i, info in enumerate(layout):
        node_start = i * n_nodes_per_strand
        node_end = node_start + n_nodes_per_strand
        elem_start = i * n_elems_per_strand
        elem_end = elem_start + n_elems_per_strand

        # 節点座標を生成
        if info.lay_radius < 1e-15:
            # 中心素線（直線）
            pts = _straight_points(n_nodes_per_strand, length)
        else:
            # ヘリカル素線
            pts = _helix_points(
                n_nodes_per_strand,
                info.lay_radius,
                pitch,
                length,
                angle_offset=info.angle_offset,
                lay_direction=info.lay_direction,
            )
        node_coords[node_start:node_end] = pts

        # 要素接続
        for j in range(n_elems_per_strand):
            connectivity[elem_start + j, 0] = node_start + j
            connectivity[elem_start + j, 1] = node_start + j + 1

        strand_node_ranges.append((node_start, node_end))
        strand_elem_ranges.append((elem_start, elem_end))

    return TwistedWireMesh(
        node_coords=node_coords,
        connectivity=connectivity,
        strand_node_ranges=strand_node_ranges,
        strand_elem_ranges=strand_elem_ranges,
        strand_infos=layout,
        n_strands=n_strands,
        wire_radius=wire_radius,
        pitch=pitch,
        length=length,
        n_elems_per_strand=n_elems_per_strand,
    )


def compute_helix_angle(lay_radius: float, pitch: float) -> float:
    """ヘリックス角を計算する.

    α = arctan(2π·R / pitch)

    Args:
        lay_radius: 配置半径
        pitch: 撚ピッチ

    Returns:
        ヘリックス角 [rad]
    """
    if pitch <= 0:
        raise ValueError(f"撚ピッチは正でなければなりません: {pitch}")
    return math.atan2(2.0 * math.pi * lay_radius, pitch)


def compute_strand_length_per_pitch(lay_radius: float, pitch: float) -> float:
    """1ピッチあたりの素線長（ヘリックス弧長）を計算する.

    L_strand = √((2πR)² + pitch²)

    Args:
        lay_radius: 配置半径
        pitch: 撚ピッチ

    Returns:
        1ピッチあたりの素線弧長 [m]
    """
    circumference = 2.0 * math.pi * lay_radius
    return math.sqrt(circumference**2 + pitch**2)


# ====================================================================
# 被膜モデル（CoatingModel）
# ====================================================================


@dataclass
class CoatingModel:
    """被膜（絶縁被覆・シース）モデル.

    撚線の各素線を包む被膜層の材料特性を定義する。
    被膜は以下の2つの役割を持つ:

    1. **剛性寄与**: 被膜層の環状断面が梁の曲げ/ねじり/引張剛性に寄与
    2. **摩擦制御**: 被膜表面の摩擦係数が素線間の接触摩擦を決定

    被膜は理想化弾性体として扱い、温度・損傷は対象外。

    Attributes:
        thickness: 被膜厚さ [m]
        E: 被膜のヤング率 [Pa]
        nu: 被膜のポアソン比
        mu: 被膜表面の摩擦係数
    """

    thickness: float
    E: float
    nu: float
    mu: float = 0.3

    def __post_init__(self) -> None:
        """入力値の検証."""
        if self.thickness <= 0:
            raise ValueError(f"被膜厚さは正でなければなりません: {self.thickness}")
        if self.E <= 0:
            raise ValueError(f"ヤング率は正でなければなりません: {self.E}")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(f"ポアソン比は (-1, 0.5) の範囲: {self.nu}")
        if self.mu < 0:
            raise ValueError(f"摩擦係数は非負でなければなりません: {self.mu}")

    @property
    def G(self) -> float:
        """せん断弾性係数 [Pa]."""
        return self.E / (2.0 * (1.0 + self.nu))


def coating_section_properties(
    wire_radius: float,
    coating: CoatingModel,
) -> dict[str, float]:
    """被膜層の断面特性（環状断面）を計算する.

    被膜を素線周囲の薄肉環状断面としてモデル化し、
    断面積・断面二次モーメント・ねじり定数を返す。

    Args:
        wire_radius: 素線断面半径 [m]
        coating: 被膜モデル

    Returns:
        断面特性の辞書 {"A", "Iy", "Iz", "J"}
    """
    r_in = wire_radius
    r_out = wire_radius + coating.thickness

    A = math.pi * (r_out**2 - r_in**2)
    Iy = math.pi / 4.0 * (r_out**4 - r_in**4)  # Iy = Iz（円環対称）
    J = math.pi / 2.0 * (r_out**4 - r_in**4)

    return {"A": A, "Iy": Iy, "Iz": Iy, "J": J}


def coated_beam_section(
    wire_radius: float,
    wire_E: float,
    wire_nu: float,
    coating: CoatingModel,
) -> dict[str, float]:
    """被膜込みの等価断面剛性を計算する.

    素線（芯線）と被膜（環状層）の複合断面として、
    軸・曲げ・ねじりの各剛性を足し合わせる。

    EA_total = E_wire * A_wire + E_coat * A_coat
    EI_total = E_wire * I_wire + E_coat * I_coat
    GJ_total = G_wire * J_wire + G_coat * J_coat

    Args:
        wire_radius: 素線断面半径 [m]
        wire_E: 素線のヤング率 [Pa]
        wire_nu: 素線のポアソン比
        coating: 被膜モデル

    Returns:
        等価断面剛性の辞書:
            - EA: 軸剛性 [N]
            - EIy, EIz: 曲げ剛性 [N·m²]
            - GJ: ねじり剛性 [N·m²]
            - A_wire, A_coat: 断面積 [m²]
            - n_axial: 軸剛性ヤング率比 E_coat/E_wire
            - n_torsion: ねじり剛性せん断弾性係数比 G_coat/G_wire
    """
    cp = coating_section_properties(wire_radius, coating)

    # 素線の断面特性（円形断面）
    A_wire = math.pi * wire_radius**2
    I_wire = math.pi / 4.0 * wire_radius**4
    J_wire = math.pi / 2.0 * wire_radius**4

    # せん断弾性係数
    G_wire = wire_E / (2.0 * (1.0 + wire_nu))
    G_coat = coating.G

    # 複合断面剛性
    EA = wire_E * A_wire + coating.E * cp["A"]
    EIy = wire_E * I_wire + coating.E * cp["Iy"]
    EIz = wire_E * I_wire + coating.E * cp["Iz"]
    GJ = G_wire * J_wire + G_coat * cp["J"]

    return {
        "EA": EA,
        "EIy": EIy,
        "EIz": EIz,
        "GJ": GJ,
        "A_wire": A_wire,
        "A_coat": cp["A"],
        "n_axial": coating.E / wire_E,
        "n_torsion": G_coat / G_wire,
    }


def coated_contact_radius(wire_radius: float, coating: CoatingModel) -> float:
    """被膜込みの接触半径を返す.

    接触検出時に使う有効半径。被膜厚さ分だけ大きくなる。

    Args:
        wire_radius: 素線断面半径 [m]
        coating: 被膜モデル

    Returns:
        被膜込み半径 [m]
    """
    return wire_radius + coating.thickness


def coated_radii(mesh: TwistedWireMesh, coating: CoatingModel) -> np.ndarray:
    """被膜込みの要素ごと接触半径ベクトルを返す.

    Args:
        mesh: 撚線メッシュ
        coating: 被膜モデル

    Returns:
        (n_elems,) の接触半径配列
    """
    return np.full(mesh.n_elems, mesh.wire_radius + coating.thickness)


# ====================================================================
# シース（外被）モデル（SheathModel）
# ====================================================================


@dataclass
class SheathModel:
    """撚線全体を覆う円筒シース（外被）モデル.

    撚線束の外周を包む円筒シースの材料特性を定義する。
    CoatingModel（素線個別被膜）とは異なり、束全体への拘束として作用する。

    シースの役割:
    1. **径方向拘束**: 最外層素線がシース内面に押し付けられる外圧効果
    2. **剛性寄与**: 円筒シェルとしての等価梁剛性（EA/EI/GJ）
    3. **摩擦制御**: シース内面と最外層素線間の摩擦

    Attributes:
        thickness: シース肉厚 [m]
        E: シースのヤング率 [Pa]
        nu: シースのポアソン比
        mu: シース内面の摩擦係数
        clearance: シース内面と最外層素線外表面の初期クリアランス [m]（0=密着）
    """

    thickness: float
    E: float
    nu: float
    mu: float = 0.3
    clearance: float = 0.0

    def __post_init__(self) -> None:
        """入力値の検証."""
        if self.thickness <= 0:
            raise ValueError(f"シース肉厚は正でなければなりません: {self.thickness}")
        if self.E <= 0:
            raise ValueError(f"ヤング率は正でなければなりません: {self.E}")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(f"ポアソン比は (-1, 0.5) の範囲: {self.nu}")
        if self.mu < 0:
            raise ValueError(f"摩擦係数は非負でなければなりません: {self.mu}")
        if self.clearance < 0:
            raise ValueError(f"クリアランスは非負でなければなりません: {self.clearance}")

    @property
    def G(self) -> float:
        """せん断弾性係数 [Pa]."""
        return self.E / (2.0 * (1.0 + self.nu))


def compute_envelope_radius(
    mesh: TwistedWireMesh,
    *,
    coating: CoatingModel | None = None,
) -> float:
    """撚線束の外接円半径（エンベロープ半径）を計算する.

    最外層素線の配置半径 + 素線半径（+ 被膜厚さ）で決定。

    Args:
        mesh: 撚線メッシュ
        coating: 被膜モデル（素線被膜がある場合）

    Returns:
        外接円半径 [m]
    """
    max_lay_radius = max(info.lay_radius for info in mesh.strand_infos)
    effective_wire_radius = mesh.wire_radius
    if coating is not None:
        effective_wire_radius += coating.thickness
    return max_lay_radius + effective_wire_radius


def sheath_inner_radius(
    mesh: TwistedWireMesh,
    sheath: SheathModel,
    *,
    coating: CoatingModel | None = None,
) -> float:
    """シース内径を計算する.

    シース内径 = エンベロープ半径 + クリアランス

    Args:
        mesh: 撚線メッシュ
        sheath: シースモデル
        coating: 被膜モデル（素線被膜がある場合）

    Returns:
        シース内径 [m]
    """
    return compute_envelope_radius(mesh, coating=coating) + sheath.clearance


def sheath_section_properties(
    mesh: TwistedWireMesh,
    sheath: SheathModel,
    *,
    coating: CoatingModel | None = None,
) -> dict[str, float]:
    """シースの断面特性（円筒管断面）を計算する.

    シースを薄肉〜厚肉円筒管としてモデル化し、
    断面積・断面二次モーメント・ねじり定数を返す。

    Args:
        mesh: 撚線メッシュ
        sheath: シースモデル
        coating: 被膜モデル（素線被膜がある場合）

    Returns:
        断面特性の辞書 {"A", "Iy", "Iz", "J", "r_inner", "r_outer"}
    """
    r_in = sheath_inner_radius(mesh, sheath, coating=coating)
    r_out = r_in + sheath.thickness

    A = math.pi * (r_out**2 - r_in**2)
    Iy = math.pi / 4.0 * (r_out**4 - r_in**4)
    J = math.pi / 2.0 * (r_out**4 - r_in**4)

    return {
        "A": A,
        "Iy": Iy,
        "Iz": Iy,
        "J": J,
        "r_inner": r_in,
        "r_outer": r_out,
    }


def sheath_equivalent_stiffness(
    mesh: TwistedWireMesh,
    sheath: SheathModel,
    *,
    coating: CoatingModel | None = None,
) -> dict[str, float]:
    """シースの等価梁剛性を計算する.

    シースを等価梁としてモデル化した場合の EA/EI/GJ を返す。

    Args:
        mesh: 撚線メッシュ
        sheath: シースモデル
        coating: 被膜モデル（素線被膜がある場合）

    Returns:
        等価梁剛性の辞書:
            - EA: 軸剛性 [N]
            - EIy, EIz: 曲げ剛性 [N·m²]
            - GJ: ねじり剛性 [N·m²]
            - r_inner, r_outer: 内外径 [m]
    """
    sp = sheath_section_properties(mesh, sheath, coating=coating)
    G = sheath.G
    return {
        "EA": sheath.E * sp["A"],
        "EIy": sheath.E * sp["Iy"],
        "EIz": sheath.E * sp["Iz"],
        "GJ": G * sp["J"],
        "r_inner": sp["r_inner"],
        "r_outer": sp["r_outer"],
    }


def outermost_layer(mesh: TwistedWireMesh) -> int:
    """最外層の層番号を返す.

    Args:
        mesh: 撚線メッシュ

    Returns:
        最外層の層番号
    """
    return max(info.layer for info in mesh.strand_infos)


def outermost_strand_ids(mesh: TwistedWireMesh) -> list[int]:
    """最外層に属する素線IDのリストを返す.

    Args:
        mesh: 撚線メッシュ

    Returns:
        最外層素線のIDリスト
    """
    outer = outermost_layer(mesh)
    return [info.strand_id for info in mesh.strand_infos if info.layer == outer]


def outermost_strand_node_indices(mesh: TwistedWireMesh) -> np.ndarray:
    """最外層素線の全節点インデックスを返す.

    シース-素線接触ペア生成時に最外層節点を特定するのに使用。

    Args:
        mesh: 撚線メッシュ

    Returns:
        最外層素線の節点インデックス配列
    """
    ids = outermost_strand_ids(mesh)
    nodes = []
    for sid in ids:
        nodes.append(mesh.strand_nodes(sid))
    return np.concatenate(nodes)


def sheath_radial_gap(
    mesh: TwistedWireMesh,
    sheath: SheathModel,
    *,
    coating: CoatingModel | None = None,
) -> np.ndarray:
    """最外層素線節点とシース内面の径方向ギャップを計算する.

    ギャップ = シース内径 - (節点の径方向位置 + 素線有効半径)
    正値 = 非接触（ギャップ開）、負値 = 貫入（接触）

    初期配置では clearance > 0 なら全て正（非接触）、
    clearance = 0 なら最外層中央で 0（密着）となる。

    Args:
        mesh: 撚線メッシュ
        sheath: シースモデル
        coating: 被膜モデル（素線被膜がある場合）

    Returns:
        (n_outer_nodes,) の径方向ギャップ配列 [m]
    """
    r_inner = sheath_inner_radius(mesh, sheath, coating=coating)
    effective_wire_radius = mesh.wire_radius
    if coating is not None:
        effective_wire_radius += coating.thickness

    outer_nodes = outermost_strand_node_indices(mesh)
    coords = mesh.node_coords[outer_nodes]

    # 径方向位置 = √(x² + y²)
    r_nodal = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

    return r_inner - (r_nodal + effective_wire_radius)


# ====================================================================
# Stage S2: シース内面プロファイル + コンプライアンス行列
# ====================================================================


def compute_inner_surface_profile(
    mesh: TwistedWireMesh,
    *,
    n_theta: int = 360,
    coating: CoatingModel | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """素線配置からシース内面形状プロファイルを計算する（Stage S2）.

    z=0 切断面で、最外層素線の断面外縁を包絡するシース内面の
    半径方向プロファイル r_inner(θ) を計算する。

    各角度 θ において中心から放射方向に引いた線分が最外層素線の
    断面外縁と交差する最遠点を r_inner(θ) とする:

      r_inner(θ) = max_i { R_i cos(θ−φ_i) + √(r_w² − R_i² sin²(θ−φ_i)) }

    素線と交差しない角度では隣接値から線形補間する。

    Parameters
    ----------
    mesh : TwistedWireMesh
        撚線メッシュ
    n_theta : int
        角度の離散点数（デフォルト 360）
    coating : CoatingModel | None
        被膜モデル（被膜がある場合、有効素線半径が増大）

    Returns
    -------
    theta : (n_theta,) ndarray
        角度配列 [0, 2π)  [rad]
    r_inner : (n_theta,) ndarray
        シース内面半径 [m]
    """
    if n_theta < 3:
        raise ValueError(f"n_theta={n_theta} は 3 以上が必要")

    outer = outermost_layer(mesh)
    outer_infos = [info for info in mesh.strand_infos if info.layer == outer]

    r_w = mesh.wire_radius
    if coating is not None:
        r_w += coating.thickness

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    r_profile = np.zeros(n_theta)

    tol = r_w * 1e-10  # 浮動小数点境界処理用トレランス
    for k in range(n_theta):
        th = theta[k]
        max_r = 0.0
        for info in outer_infos:
            R_lay = info.lay_radius
            phi = info.angle_offset
            d_perp = R_lay * math.sin(th - phi)
            if abs(d_perp) <= r_w + tol:
                d_perp_clamped = min(abs(d_perp), r_w)
                d_radial = R_lay * math.cos(th - phi) + math.sqrt(r_w**2 - d_perp_clamped**2)
                if d_radial > max_r:
                    max_r = d_radial
        r_profile[k] = max_r

    # 素線と交差しない角度（gap）がある場合は線形補間
    zero_mask = r_profile == 0.0
    if np.any(zero_mask):
        nonzero_idx = np.where(~zero_mask)[0]
        if len(nonzero_idx) > 0:
            # 周期的な補間のため前後に延長
            angles_good = theta[nonzero_idx]
            values_good = r_profile[nonzero_idx]
            angles_ext = np.concatenate(
                [angles_good - 2.0 * math.pi, angles_good, angles_good + 2.0 * math.pi]
            )
            values_ext = np.concatenate([values_good, values_good, values_good])
            r_profile = np.interp(theta, angles_ext, values_ext)

    return theta, r_profile


def sheath_compliance_matrix(
    mesh: TwistedWireMesh,
    sheath: SheathModel,
    *,
    coating: CoatingModel | None = None,
    n_theta: int = 360,
    n_modes: int | None = None,
    plane: str = "strain",
) -> np.ndarray:
    """撚線+シースモデルから膜厚分布考慮のコンプライアンス行列を構築する（Stage S2）.

    最外層素線の配置からシース内面プロファイルを計算し、
    各接触点での局所膜厚を反映した修正コンプライアンス行列を返す。

    均一厚みの場合は Stage S1 の ``build_ring_compliance_matrix`` と
    実質的に一致する。

    Parameters
    ----------
    mesh : TwistedWireMesh
        撚線メッシュ
    sheath : SheathModel
        シースモデル
    coating : CoatingModel | None
        被膜モデル
    n_theta : int
        内面プロファイルの角度サンプル数
    n_modes : int | None
        Fourier 級数の打ち切りモード数
    plane : str
        "strain"（平面ひずみ）or "stress"（平面応力）

    Returns
    -------
    C : (N, N) ndarray
        修正コンプライアンス行列 [m/N]。N = 最外層素線本数。
    """
    from xkep_cae.mesh.ring_compliance import build_variable_thickness_compliance_matrix

    # 内面プロファイル計算
    theta_profile, r_inner_profile = compute_inner_surface_profile(
        mesh, n_theta=n_theta, coating=coating
    )

    # 接触点の角度と内面半径を抽出
    outer = outermost_layer(mesh)
    outer_infos = [info for info in mesh.strand_infos if info.layer == outer]
    N = len(outer_infos)
    contact_angles = np.array([info.angle_offset for info in outer_infos])

    # 各接触点での内面半径（プロファイルから補間）
    # 周期的補間のため前後に拡張
    theta_ext = np.concatenate(
        [theta_profile - 2.0 * math.pi, theta_profile, theta_profile + 2.0 * math.pi]
    )
    r_ext = np.concatenate([r_inner_profile, r_inner_profile, r_inner_profile])
    r_inner_at_contacts = np.interp(contact_angles, theta_ext, r_ext)

    # クリアランスを加算してシース内面半径とする
    r_inner_at_contacts = r_inner_at_contacts + sheath.clearance

    # シース外径
    r_outer = np.max(r_inner_at_contacts) + sheath.thickness

    return build_variable_thickness_compliance_matrix(
        N,
        contact_angles,
        r_inner_at_contacts,
        r_outer,
        sheath.E,
        sheath.nu,
        n_modes=n_modes,
        plane=plane,
    )
