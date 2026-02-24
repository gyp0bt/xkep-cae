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
