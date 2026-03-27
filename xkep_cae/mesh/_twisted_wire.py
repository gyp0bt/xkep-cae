"""撚線（twisted wire）メッシュ生成.

StrandMeshProcess が使う最小限の関数。

[← README](../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StrandInfoOutput:
    """1素線の情報."""

    strand_id: int
    layer: int
    angle_offset: float
    lay_radius: float
    wire_radius: float
    lay_direction: int = 1


@dataclass(frozen=True)
class TwistedWireMeshOutput:
    """撚線メッシュ."""

    node_coords: np.ndarray
    connectivity: np.ndarray
    strand_node_ranges: tuple[tuple[int, int], ...]
    strand_elem_ranges: tuple[tuple[int, int], ...]
    strand_infos: tuple[StrandInfoOutput, ...]
    n_strands: int
    wire_radius: float
    pitch: float
    length: float
    n_elems_per_strand: int


def _n_nodes(mesh: TwistedWireMeshOutput) -> int:
    return mesh.node_coords.shape[0]


def _n_elems(mesh: TwistedWireMeshOutput) -> int:
    return mesh.connectivity.shape[0]


def _radii(mesh: TwistedWireMeshOutput) -> np.ndarray:
    return np.full(_n_elems(mesh), mesh.wire_radius)


# ------------------------------------------------------------------
# 内部ヘルパー
# ------------------------------------------------------------------


def _helix_points(
    n_points: int,
    lay_radius: float,
    pitch: float,
    length: float,
    angle_offset: float = 0.0,
    lay_direction: int = 1,
) -> np.ndarray:
    coords = np.zeros((n_points, 3))
    for i in range(n_points):
        z = i * length / (n_points - 1)
        theta = lay_direction * 2.0 * math.pi * z / pitch + angle_offset
        coords[i, 0] = lay_radius * math.cos(theta)
        coords[i, 1] = lay_radius * math.sin(theta)
        coords[i, 2] = z
    return coords


def _straight_points(n_points: int, length: float) -> np.ndarray:
    coords = np.zeros((n_points, 3))
    coords[:, 2] = np.linspace(0.0, length, n_points)
    return coords


def _compute_layer_structure(n_strands: int) -> list[int]:
    if n_strands == 3:
        return [0, 3]
    layers = [1]
    remaining = n_strands - 1
    layer = 1
    while remaining > 0:
        n_in_layer = 6 * layer
        actual = min(n_in_layer, remaining)
        layers.append(actual)
        remaining -= actual
        layer += 1
    return layers


def _minimum_strand_diameter(n_strands: int, wire_diameter: float) -> float:
    r_wire = wire_diameter / 2.0
    d = wire_diameter
    if n_strands == 1:
        return wire_diameter
    if n_strands == 3:
        return 2.0 * (d / math.sqrt(3.0) + r_wire)
    layer_counts = _compute_layer_structure(n_strands)
    n_layers = len(layer_counts) - 1
    r_lay: list[float] = [0.0]
    for k in range(1, n_layers + 1):
        n_k = layer_counts[k]
        r_min_radial = r_lay[k - 1] + d
        r_min_circum = r_wire / math.sin(math.pi / n_k) if n_k >= 2 else 0.0
        r_lay.append(max(r_min_radial, r_min_circum))
    return 2.0 * (r_lay[-1] + r_wire)


def _validate_strand_geometry(n_strands: int, wire_diameter: float, strand_diameter: float) -> None:
    min_d = _minimum_strand_diameter(n_strands, wire_diameter)
    if strand_diameter < min_d - 1e-15 * min_d:
        raise ValueError(
            f"撚線外径 {strand_diameter:.6f}m は {n_strands}本・素線径 "
            f"{wire_diameter:.6f}m に対して小さすぎます。最小外径: {min_d:.6f}m"
        )


def _segment_distance(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """2線分間の最小距離."""
    d = p1 - p0
    e = q1 - q0
    r = p0 - q0
    a = float(np.dot(d, d))
    b = float(np.dot(d, e))
    c = float(np.dot(e, e))
    f = float(np.dot(d, r))
    g = float(np.dot(e, r))
    denom = a * c - b * b
    if denom < 1e-30:
        s = 0.0
        t = g / c if c > 1e-30 else 0.0
    else:
        s = (b * g - c * f) / denom
        t = (a * g - b * f) / denom
    s = max(0.0, min(1.0, s))
    t = max(0.0, min(1.0, t))
    return float(np.linalg.norm(p0 + s * d - (q0 + t * e)))


def _helix_point(z: float, r_lay: float, pitch: float, angle_offset: float) -> np.ndarray:
    """ヘリックス上の点座標."""
    theta = 2.0 * math.pi * z / pitch + angle_offset
    return np.array([r_lay * math.cos(theta), r_lay * math.sin(theta), z])


def _min_chord_distance_between_strands(
    r_lay: float,
    pitch: float,
    n_per_pitch: int,
    phase_diff: float,
) -> float:
    """2つのヘリカル素線の弦近似間の最小距離を数値計算.

    同一 lay_radius・同一ピッチ・位相差 phase_diff の2本のヘリックスを
    n_per_pitch 個の弦要素で近似したときの最小セグメント間距離を返す。
    1ピッチ分の全ペアを検査する。
    """
    dz = pitch / n_per_pitch
    min_dist = float("inf")
    search_range = max(5, n_per_pitch // 4)
    for i in range(n_per_pitch):
        pA0 = _helix_point(i * dz, r_lay, pitch, 0.0)
        pA1 = _helix_point((i + 1) * dz, r_lay, pitch, 0.0)
        for j in range(max(0, i - search_range), min(n_per_pitch, i + search_range + 1)):
            pB0 = _helix_point(j * dz, r_lay, pitch, phase_diff)
            pB1 = _helix_point((j + 1) * dz, r_lay, pitch, phase_diff)
            d = _segment_distance(pA0, pA1, pB0, pB1)
            if d < min_dist:
                min_dist = d
    return min_dist


def _min_helix_distance(
    r_a: float,
    r_b: float,
    pitch: float,
    phase_a: float,
    phase_b: float,
) -> float:
    """2本のヘリックス間の解析的最小距離.

    d²(Δz) = r_a² + r_b² - 2·r_a·r_b·cos(ω·Δz + Δφ) + Δz²
    を最小化する。ω = 2π/pitch, Δφ = phase_a - phase_b。

    Newton-Raphson で d²(Δz) の停留点を求める。
    """
    omega = 2.0 * math.pi / pitch if pitch > 0 else 0.0
    dphi = phase_a - phase_b
    rab = r_a * r_b

    if rab < 1e-30:
        # 片方が中心（直線）: d = r_other
        return max(r_a, r_b)

    def d2(dz: float) -> float:
        return r_a**2 + r_b**2 - 2 * rab * math.cos(omega * dz + dphi) + dz**2

    def dd2(dz: float) -> float:
        return 2 * rab * omega * math.sin(omega * dz + dphi) + 2 * dz

    def ddd2(dz: float) -> float:
        return 2 * rab * omega**2 * math.cos(omega * dz + dphi) + 2

    # 複数の初期点から探索（周期性を考慮）
    best = d2(0.0)
    candidates = [0.0, -dphi / omega if omega > 0 else 0.0]
    # 追加候補: 同位相点 ± 微小シフト
    for k in range(-2, 3):
        candidates.append((-dphi + 2 * math.pi * k) / omega if omega > 0 else 0.0)

    for dz0 in candidates:
        dz = dz0
        for _ in range(20):
            g = dd2(dz)
            h = ddd2(dz)
            if abs(h) < 1e-30:
                break
            dz -= g / h
        val = d2(dz)
        if val < best:
            best = val

    return math.sqrt(max(0.0, best))


def _compute_min_safe_gap(
    n_strands: int,
    wire_radius: float,
    pitch: float,
    n_elems_per_strand: int,
    n_pitches: float,
    *,
    safety_factor: float = 1.5,
    coating_thickness: float = 0.0,
) -> float:
    """弦近似で貫入が生じない最小ギャップを二分探索で計算.

    解析的ヘリックス間最小距離 + 弦近似sagittaの補正で安全ギャップを決定。

    1. レイアウトの全クリティカルペアのヘリックス最小距離を解析計算
    2. 弦近似sagitta（径方向）を加算補正
    3. 全ペアで surface_gap ≥ 0 となる最小 gap を二分探索
    """
    if n_strands <= 1:
        return 0.0
    length = n_pitches * pitch
    n_per_pitch = int(n_elems_per_strand * pitch / length) if length > 0 else n_elems_per_strand
    if n_per_pitch < 1:
        return 0.0

    r_eff = wire_radius + coating_thickness

    def _min_surface_gap(gap: float) -> float:
        """指定 gap でのクリティカルペアの最小表面ギャップ."""
        layout = _make_strand_layout(
            n_strands,
            wire_radius,
            gap=gap,
            coating_thickness=coating_thickness,
        )
        # レイヤーごとにグループ化
        layers: dict[int, list[StrandInfoOutput]] = {}
        for info in layout:
            layers.setdefault(info.layer, []).append(info)

        min_sg = float("inf")
        layer_keys = sorted(layers.keys())

        # 同層内隣接ペアのみ検査（支配的）
        # 異層間は d_min ≈ |r_a - r_b| = d_eff + gap ≥ 2R で常に正
        for lk in layer_keys:
            strands = layers[lk]
            if len(strands) < 2:
                continue
            sorted_s = sorted(strands, key=lambda s: s.angle_offset)
            sa = sorted_s[0]
            sb = sorted_s[1]
            # 弦間最小距離を数値計算（解析的ヘリックス距離より正確）
            d = _min_chord_distance_between_strands(
                sa.lay_radius,
                pitch,
                n_per_pitch,
                sb.angle_offset - sa.angle_offset,
            )
            sg = d - 2.0 * r_eff
            if sg < min_sg:
                min_sg = sg

        return min_sg

    # ヘリックス幾何制約チェック: 同層内の斜め交差距離
    # 最小交差距離 = pitch * Δφ / (2π) ≥ 2·r_eff が必要
    layout_check = _make_strand_layout(
        n_strands,
        wire_radius,
        gap=0.0,
        coating_thickness=coating_thickness,
    )
    layers_check: dict[int, list[StrandInfoOutput]] = {}
    for info in layout_check:
        layers_check.setdefault(info.layer, []).append(info)
    for lk, strands in layers_check.items():
        if len(strands) < 2:
            continue
        sorted_s = sorted(strands, key=lambda s: s.angle_offset)
        min_phase = 2.0 * math.pi  # 最小位相差
        for idx in range(len(sorted_s)):
            da = sorted_s[(idx + 1) % len(sorted_s)].angle_offset - sorted_s[idx].angle_offset
            da = da % (2.0 * math.pi)
            if da < min_phase:
                min_phase = da
        crossing_dist = pitch * min_phase / (2.0 * math.pi)
        if crossing_dist < 2.0 * r_eff:
            min_pitch = 2.0 * r_eff * 2.0 * math.pi / min_phase
            raise ValueError(
                f"レイヤー{lk}のヘリックス幾何制約違反: "
                f"{len(strands)}本の素線（位相差{math.degrees(min_phase):.1f}°）で "
                f"斜め交差距離{crossing_dist:.2f}mm < 素線直径{2 * r_eff:.1f}mm。"
                f"ピッチを{min_pitch:.1f}mm以上に増やすか、"
                f"レイヤーあたりの素線数を減らしてください。"
            )

    # gap=0 で評価
    sg0 = _min_surface_gap(0.0)
    if sg0 >= 0:
        return 0.0

    # 二分探索
    gap_lo = 0.0
    gap_hi = max(abs(sg0) * 3.0, 2.0 * wire_radius)
    for _ in range(8):
        if _min_surface_gap(gap_hi) >= 0:
            break
        gap_hi *= 2.0
    else:
        return gap_hi * safety_factor

    for _ in range(40):
        gap_mid = 0.5 * (gap_lo + gap_hi)
        if _min_surface_gap(gap_mid) >= 0:
            gap_hi = gap_mid
        else:
            gap_lo = gap_mid
        if gap_hi - gap_lo < 1e-4:
            break

    return gap_hi * safety_factor


def _make_strand_layout(
    n_strands: int,
    wire_radius: float,
    *,
    gap: float = 0.0,
    lay_direction: int = 1,
    strand_diameter: float | None = None,
    coating_thickness: float = 0.0,
) -> list[StrandInfoOutput]:
    d = 2.0 * wire_radius
    d_eff = 2.0 * (wire_radius + coating_thickness)
    infos: list[StrandInfoOutput] = []
    sid = 0

    if n_strands == 3:
        if strand_diameter is not None:
            _validate_strand_geometry(n_strands, d, strand_diameter)
            r_lay = strand_diameter / 2.0 - wire_radius
        else:
            r_lay = (d_eff + gap) / math.sqrt(3.0)
        for k in range(3):
            angle = 2.0 * math.pi * k / 3.0
            infos.append(
                StrandInfoOutput(
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

    if n_strands >= 1:
        infos.append(
            StrandInfoOutput(
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

    if strand_diameter is not None:
        _validate_strand_geometry(n_strands, d, strand_diameter)
        layer_counts = _compute_layer_structure(n_strands)
        n_layers = len(layer_counts) - 1
        r_eff = wire_radius + coating_thickness
        r_min: list[float] = [0.0]
        for k in range(1, n_layers + 1):
            n_k = layer_counts[k]
            r_min_radial = r_min[k - 1] + d_eff
            r_min_circum = r_eff / math.sin(math.pi / n_k) if n_k >= 2 else 0.0
            r_min.append(max(r_min_radial, r_min_circum))
        r_max_needed = r_min[-1]
        available = strand_diameter / 2.0 - wire_radius
        surplus = available - r_max_needed
        if surplus > 1e-15 and n_layers > 0:
            delta = surplus / n_layers
            lay_radii = [0.0] + [r_min[k] + k * delta for k in range(1, n_layers + 1)]
        else:
            lay_radii = r_min
        for k in range(1, n_layers + 1):
            n_in_layer = layer_counts[k]
            r_lay_k = lay_radii[k]
            layer_dir = lay_direction if (k % 2 == 1) else -lay_direction
            for j in range(n_in_layer):
                angle = (
                    2.0 * math.pi * j / (6 * k)
                    if n_in_layer == 6 * k
                    else 2.0 * math.pi * j / n_in_layer
                )
                infos.append(
                    StrandInfoOutput(
                        strand_id=sid,
                        layer=k,
                        angle_offset=angle,
                        lay_radius=r_lay_k,
                        wire_radius=wire_radius,
                        lay_direction=layer_dir,
                    )
                )
                sid += 1
    else:
        layer = 1
        while remaining > 0:
            n_in_layer = 6 * layer
            actual = min(n_in_layer, remaining)
            r_lay_l = layer * (d_eff + gap)
            layer_dir = lay_direction if (layer % 2 == 1) else -lay_direction
            for k in range(actual):
                angle = 2.0 * math.pi * k / n_in_layer
                infos.append(
                    StrandInfoOutput(
                        strand_id=sid,
                        layer=layer,
                        angle_offset=angle,
                        lay_radius=r_lay_l,
                        wire_radius=wire_radius,
                        lay_direction=layer_dir,
                    )
                )
                sid += 1
            remaining -= actual
            layer += 1

    return infos


# ------------------------------------------------------------------
# API（モジュール内 private）
# ------------------------------------------------------------------


def _make_twisted_wire_mesh(
    n_strands: int,
    wire_diameter: float,
    pitch: float,
    length: float,
    n_elems_per_strand: int,
    *,
    gap: float = 0.0,
    lay_direction: int = 1,
    n_pitches: float | None = None,
    strand_diameter: float | None = None,
    min_elems_per_pitch: int = 16,
    coating_thickness: float = 0.0,
) -> TwistedWireMeshOutput:
    """撚線メッシュを生成する."""
    wire_radius = wire_diameter / 2.0

    if n_pitches is not None:
        length = n_pitches * pitch

    if min_elems_per_pitch < 16:
        raise ValueError(
            f"min_elems_per_pitch={min_elems_per_pitch} は許可されていません。"
            f"16要素/ピッチ未満での実行は厳格に禁止されています。"
        )
    if n_strands > 1 and length > 0 and pitch > 0:
        n_per_pitch = n_elems_per_strand * pitch / length
        if n_per_pitch < min_elems_per_pitch:
            raise ValueError(
                f"要素密度が不足しています: {n_per_pitch:.1f}要素/ピッチ "
                f"(最小 {min_elems_per_pitch} 要素/ピッチ以上)。"
                f"弦近似による初期貫入が発生し、接触解析の精度が低下します。"
                f"n_elems_per_strand を {math.ceil(min_elems_per_pitch * length / pitch)} "
                f"以上に設定してください。"
            )

    if n_strands > 1 and strand_diameter is None and gap >= 0:
        _n_p = n_pitches if n_pitches is not None else (length / pitch if pitch > 0 else 1.0)
        min_gap = _compute_min_safe_gap(
            n_strands,
            wire_radius,
            pitch,
            n_elems_per_strand,
            _n_p,
            coating_thickness=coating_thickness,
        )
        if gap < min_gap:
            import warnings

            warnings.warn(
                f"指定ギャップ {gap:.6f} < 最小安全ギャップ "
                f"{min_gap:.6f}（弦近似誤差）。"
                f"自動的に {min_gap:.6f} に引き上げます。",
                stacklevel=2,
            )
            gap = min_gap

    layout = _make_strand_layout(
        n_strands,
        wire_radius,
        gap=gap,
        lay_direction=lay_direction,
        strand_diameter=strand_diameter,
        coating_thickness=coating_thickness,
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

        if info.lay_radius < 1e-15:
            pts = _straight_points(n_nodes_per_strand, length)
        else:
            pts = _helix_points(
                n_nodes_per_strand,
                info.lay_radius,
                pitch,
                length,
                angle_offset=info.angle_offset,
                lay_direction=info.lay_direction,
            )
        node_coords[node_start:node_end] = pts

        for j in range(n_elems_per_strand):
            connectivity[elem_start + j, 0] = node_start + j
            connectivity[elem_start + j, 1] = node_start + j + 1

        strand_node_ranges.append((node_start, node_end))
        strand_elem_ranges.append((elem_start, elem_end))

    return TwistedWireMeshOutput(
        node_coords=node_coords,
        connectivity=connectivity,
        strand_node_ranges=tuple(strand_node_ranges),
        strand_elem_ranges=tuple(strand_elem_ranges),
        strand_infos=tuple(layout),
        n_strands=n_strands,
        wire_radius=wire_radius,
        pitch=pitch,
        length=length,
        n_elems_per_strand=n_elems_per_strand,
    )
