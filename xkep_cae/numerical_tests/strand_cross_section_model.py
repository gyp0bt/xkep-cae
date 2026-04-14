"""StrandCrossSectionModel — 断面接触点統計モデル.

撚線ケーブル断面の各素線の幾何学的配置から、曲げ・捻り・引張の
複合荷重下でのスティック-スリップ遷移を解析的に記述する。

物理モデル:
  各素線 i は (R_i, θ_i, α_i) で特徴づけられる:
    R_i: ヘリックス半径（芯線中心からの距離）
    θ_i: 周方向位置
    α_i: ヘリックス角 = atan(2π R_i / p)

  複合荷重 (κ, τ, T) のもとで:
    曲げ κ: 素線軸方向の相対滑り u_bend,i = κ × y_i × sin(α_i) × cos(α_i) × p/(2π)
    捻り τ: 素線軸方向の相対滑り u_torsion,i = τ × R_i × cos(α_i) × p/(2π)
    引張 T: 接触法線力を変化させ、摩擦容量を変える

  スティック-スリップ遷移:
    法線接触力:  N_i = (T/n) × tan(α_i) / R_i × (2π R_i)  [素線張力のヘリックス効果]
    摩擦容量:    f_i = μ × N_i
    相対滑り量:  u_i = |u_bend,i + u_torsion,i|
    臨界曲率:    κ_cr,i = f_i / (E×A × |y_i| × sin(α_i) × cos(α_i))

  散逸エネルギー（1サイクル、閉形式）:
    W_diss = 4 × Σ_{slipped} f_i × Δu_i
    ここで Δu_i = u_i(κ_max) - u_i(κ_cr,i)

参考文献:
  - Papailiou (1997): Analytical model for bending of stranded conductors
  - Foti & Martinelli (2016): Mechanical modeling of metallic strands

status-332: 断面接触点統計モデル。

[← README](../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ====================================================================
# ケーブル断面幾何
# ====================================================================


@dataclass(frozen=True)
class WireGeometry:
    """単一素線の幾何パラメータ.

    Attributes:
        layer: 層番号（0=芯線, 1=第1層, 2=第2層, ...）
        index_in_layer: 層内インデックス
        R_helix: ヘリックス半径 [mm]
        theta: 周方向位置 [rad]
        alpha: ヘリックス角 [rad]
        y: 中立軸からの y 座標 = R_helix × cos(θ)
        z: 中立軸からの z 座標 = R_helix × sin(θ)
    """

    layer: int
    index_in_layer: int
    R_helix: float
    theta: float
    alpha: float
    y: float
    z: float


@dataclass(frozen=True)
class CableSection:
    """撚線ケーブル断面の完全記述.

    Attributes:
        n_strands: 素線総数
        wire_radius: 素線半径 [mm]
        pitch_length: ピッチ長 [mm]
        wires: 各素線の幾何（芯線を含む）
        E: ヤング率 [MPa]
        A_wire: 素線断面積 [mm²]
        I_wire: 素線断面2次モーメント [mm⁴]
    """

    n_strands: int
    wire_radius: float
    pitch_length: float
    wires: tuple[WireGeometry, ...]
    E: float
    A_wire: float
    I_wire: float


def make_cable_section(
    n_strands: int = 7,
    wire_radius: float = 0.5,
    pitch_length: float = 100.0,
    E: float = 130.0e3,
) -> CableSection:
    """撚線ケーブル断面を生成.

    層構成:
      n=1:  芯線のみ
      n=2-7:  1+n_outer（第1層）
      n=8-19: 1+6+n2（第1層+第2層）
      n=20-37: 1+6+12+n3（3層）

    Args:
        n_strands: 素線総数
        wire_radius: 素線半径 [mm]
        pitch_length: ピッチ長 [mm]
        E: ヤング率 [MPa]
    """
    R = wire_radius
    A_wire = math.pi * R**2
    I_wire = math.pi / 4.0 * R**4

    wires: list[WireGeometry] = []

    # 芯線（layer=0, R_helix=0）
    wires.append(
        WireGeometry(
            layer=0,
            index_in_layer=0,
            R_helix=0.0,
            theta=0.0,
            alpha=0.0,
            y=0.0,
            z=0.0,
        )
    )

    if n_strands <= 1:
        return CableSection(
            n_strands=1,
            wire_radius=R,
            pitch_length=pitch_length,
            wires=tuple(wires),
            E=E,
            A_wire=A_wire,
            I_wire=I_wire,
        )

    # 層構成の決定
    layers: list[tuple[int, float]] = []  # (n_wires, R_helix)

    remaining = n_strands - 1  # 芯線を除く
    layer_idx = 1
    R_current = 2.0 * R  # 第1層: 芯線と接触

    while remaining > 0:
        # 各層の最大素線数: floor(2π R_layer / (2R))
        n_max = max(1, int(2.0 * math.pi * R_current / (2.0 * R)))
        n_layer = min(remaining, n_max)
        layers.append((n_layer, R_current))
        remaining -= n_layer
        layer_idx += 1
        R_current += 2.0 * R  # 次の層

    # 各層の素線を配置
    for layer_num, (n_layer, R_h) in enumerate(layers, start=1):
        alpha = math.atan(2.0 * math.pi * R_h / pitch_length)
        for j in range(n_layer):
            theta = 2.0 * math.pi * j / n_layer
            y = R_h * math.cos(theta)
            z = R_h * math.sin(theta)
            wires.append(
                WireGeometry(
                    layer=layer_num,
                    index_in_layer=j,
                    R_helix=R_h,
                    theta=theta,
                    alpha=alpha,
                    y=y,
                    z=z,
                )
            )

    return CableSection(
        n_strands=n_strands,
        wire_radius=R,
        pitch_length=pitch_length,
        wires=tuple(wires),
        E=E,
        A_wire=A_wire,
        I_wire=I_wire,
    )


# ====================================================================
# 複合荷重下の解析モデル
# ====================================================================


@dataclass(frozen=True)
class WireState:
    """単一素線のスティック-スリップ状態.

    Attributes:
        wire: 素線幾何
        u_relative: 相対滑り量 [mm]（正=滑り方向）
        f_friction: 摩擦容量 [N]
        is_slipping: スリップ中かどうか
        kappa_cr: 臨界曲率 [1/mm]（曲げモードのみ）
    """

    wire: WireGeometry
    u_relative: float
    f_friction: float
    is_slipping: bool
    kappa_cr: float


@dataclass(frozen=True)
class SectionResponse:
    """断面応答（複合荷重）.

    Attributes:
        M_y: 曲げモーメント（y軸回り）[N·mm]
        M_z: 曲げモーメント（z軸回り）[N·mm]
        T_torque: 捻りモーメント [N·mm]
        N_axial: 軸力 [N]
        n_slipping: スリップ中の素線数
        n_total: 全素線数（芯線除く）
        wire_states: 各素線の状態
        EI_effective: 有効曲げ剛性 [N·mm²]
    """

    M_y: float
    M_z: float
    T_torque: float
    N_axial: float
    n_slipping: int
    n_total: int
    wire_states: tuple[WireState, ...]
    EI_effective: float


def compute_section_response(
    section: CableSection,
    kappa_y: float = 0.0,
    kappa_z: float = 0.0,
    tau: float = 0.0,
    axial_strain: float = 0.0,
    mu: float = 0.15,
    T_pretension: float = 0.0,
) -> SectionResponse:
    """複合荷重下の断面応答を解析的に計算.

    各素線について:
    1. 相対滑り量を計算（曲げ + 捻り寄与）
    2. 摩擦容量と比較してスティック/スリップ判定
    3. モーメント寄与を積算

    Args:
        section: ケーブル断面
        kappa_y: y軸回り曲率 [1/mm]（x-z面内曲げ）
        kappa_z: z軸回り曲率 [1/mm]（x-y面内曲げ）
        tau: 捻率 [1/mm]
        axial_strain: 軸ひずみ [-]
        mu: 摩擦係数
        T_pretension: 初期張力 [N]（全ケーブル）
    """
    E = section.E
    A = section.A_wire
    p = section.pitch_length
    n = section.n_strands

    # 素線あたりの軸力
    T_per_wire = T_pretension / n if n > 0 else 0.0

    wire_states: list[WireState] = []
    M_y = 0.0
    M_z = 0.0
    T_torque = 0.0
    N_axial = 0.0
    n_slipping = 0
    n_outer = 0
    EI_sum = 0.0

    for w in section.wires:
        if w.layer == 0:
            # 芯線: 常にスティック（自身の EI のみ寄与）
            M_z += -E * section.I_wire * kappa_z
            M_y += -E * section.I_wire * kappa_y
            EI_sum += E * section.I_wire
            continue

        n_outer += 1
        alpha = w.alpha
        R_h = w.R_helix
        y = w.y
        z = w.z
        sa = math.sin(alpha)
        ca = math.cos(alpha)

        # === 相対滑り量（ヘリックス軸方向投影）===

        # 曲げによる相対滑り: 素線軸方向ひずみと芯線ひずみの差
        # Papailiou (1997): Δε_bend = κ × y × sin(α) × cos(α)
        # 1ピッチあたりの滑り量: u_bend = Δε_bend × p
        u_bend_y = kappa_z * y * sa * ca * p / (2.0 * math.pi)  # κ_z による y方向曲げ
        u_bend_z = kappa_y * z * sa * ca * p / (2.0 * math.pi)  # κ_y による z方向曲げ

        # 捻りによる相対滑り
        # τ による周方向変位の軸方向投影
        u_torsion = tau * R_h * ca * p / (2.0 * math.pi)

        # 合計相対滑り量
        u_total = math.sqrt(
            (u_bend_y + u_bend_z) ** 2 + u_torsion**2
        )

        # === 接触法線力と摩擦容量 ===
        # 初期張力のみから摩擦力を計算（Papailiou標準定式化）
        f_friction, kappa_cr = _wire_friction_and_kappa_cr(
            w, section, T_per_wire, mu
        )

        # === スティック/スリップ判定 ===
        # 弾性域: u < f/k_wire, k_wire = E×A×cos(α)/p
        k_wire = E * A * ca / p if p > 1e-10 else 1e30
        u_elastic = f_friction / k_wire if k_wire > 1e-10 else 0.0

        is_slipping = u_total > u_elastic

        if is_slipping:
            n_slipping += 1

        # === モーメント寄与 ===

        if is_slipping:
            # スリップ時: 摩擦力プラトー
            # モーメント寄与 = f_friction × sign(滑り) × lever_arm
            sign_y = math.copysign(1.0, kappa_z * y) if abs(kappa_z * y) > 1e-20 else 0.0
            sign_z = math.copysign(1.0, kappa_y * z) if abs(kappa_y * z) > 1e-20 else 0.0

            # 曲げモーメント（摩擦プラトー寄与）
            M_z += -sign_y * f_friction * abs(y) / (p / ca) * p  # 正規化
            M_y += -sign_z * f_friction * abs(z) / (p / ca) * p

            # 有効剛性は EI_min 寄与のみ
            EI_sum += E * section.I_wire
        else:
            # スティック時: 弾性（Steiner 定理の寄与あり）
            M_z += -E * (section.I_wire + A * y**2) * kappa_z
            M_y += -E * (section.I_wire + A * z**2) * kappa_y

            EI_sum += E * (section.I_wire + A * y**2)

        # 捻りモーメント寄与
        if is_slipping and u_torsion != 0.0:
            sign_tau = math.copysign(1.0, tau) if abs(tau) > 1e-20 else 0.0
            T_torque += sign_tau * f_friction * R_h
        else:
            # スティック時の捻り剛性: GJ_wire + G×A×R²
            G = E / 2.6  # 概算
            T_torque += -G * (2.0 * section.I_wire + A * R_h**2) * tau

        # 軸力
        N_axial += E * A * axial_strain

        wire_states.append(
            WireState(
                wire=w,
                u_relative=u_total,
                f_friction=f_friction,
                is_slipping=is_slipping,
                kappa_cr=kappa_cr,
            )
        )

    return SectionResponse(
        M_y=M_y,
        M_z=M_z,
        T_torque=T_torque,
        N_axial=N_axial,
        n_slipping=n_slipping,
        n_total=n_outer,
        wire_states=tuple(wire_states),
        EI_effective=EI_sum,
    )


# ====================================================================
# 散逸エネルギー解析式（閉形式）
# ====================================================================


def _wire_friction_and_kappa_cr(
    w: WireGeometry,
    section: CableSection,
    T_per_wire: float,
    mu: float,
) -> tuple[float, float]:
    """素線の摩擦容量と臨界曲率を計算.

    摩擦力源は初期張力のみ（曲げ誘起張力は含めない）。
    Papailiou (1997) の標準定式化に従う。

    Returns:
        (f_friction, kappa_cr)
    """
    E = section.E
    A = section.A_wire
    p = section.pitch_length
    R_h = w.R_helix
    y = w.y
    sa = math.sin(w.alpha)
    ca = math.cos(w.alpha)

    if abs(y) < 1e-10 or abs(sa * ca) < 1e-10 or R_h < 1e-10:
        return 0.0, float("inf")

    if abs(T_per_wire) < 1e-15:
        return 0.0, float("inf")

    # 法線接触力: ヘリックス張力の径方向成分
    # q_n = T_wire × sin(α) / R_helix  [N/mm]
    q_normal = abs(T_per_wire) * sa / R_h

    # 1ピッチあたりの接触長 = p / cos(α)
    contact_length = p / ca

    # 摩擦容量（1ピッチあたり）
    f_friction = mu * q_normal * contact_length

    # 臨界曲率: u(κ_cr) = f / k_wire
    # u = κ × |y| × sin(α) × cos(α) × p/(2π)
    # k_wire = E × A × cos(α) / p  (ワイヤ軸方向剛性/ピッチ長)
    # κ_cr = f × 2π / (E × A × |y| × sin(α) × cos²(α) × p)
    #       = f / (E × A × |y| × sin(α) × cos²(α) × p / (2π))
    kappa_cr = f_friction * 2.0 * math.pi / (
        E * A * abs(y) * sa * ca**2 * p
    )

    return f_friction, kappa_cr


def dissipation_energy_bending(
    section: CableSection,
    kappa_max: float,
    mu: float = 0.15,
    T_pretension: float = 0.0,
) -> dict[str, float]:
    """曲げ1サイクルの散逸エネルギーを閉形式で計算.

    Papailiou (1997) モデル: 初期張力による素線間摩擦が散逸源。
    W_diss = 4 × Σ_{κ_cr,i < κ_max} f_i × (u_i(κ_max) - u_i(κ_cr,i))

    Note:
        T_pretension=0 では摩擦力ゼロのため散逸もゼロ。
        数値モデルとの比較には calibrate_pretension() でキャリブレーション。

    Args:
        section: ケーブル断面
        kappa_max: 最大曲率 [1/mm]
        mu: 摩擦係数
        T_pretension: 初期張力 [N]

    Returns:
        dict with dissipation_energy, n_slipping, kappa_cr_min/max, EI_min, EI_max
    """
    E = section.E
    A = section.A_wire
    p = section.pitch_length
    n = section.n_strands

    T_per_wire = T_pretension / n if n > 0 else 0.0

    W_total = 0.0
    n_slipping = 0
    n_outer = 0
    EI_min = 0.0
    EI_max = 0.0
    kappa_crs: list[float] = []

    for w in section.wires:
        if w.layer == 0:
            EI_min += E * section.I_wire
            EI_max += E * section.I_wire
            continue

        n_outer += 1
        sa = math.sin(w.alpha)
        ca = math.cos(w.alpha)
        y = w.y

        # EI 寄与
        EI_min += E * section.I_wire
        EI_max += E * (section.I_wire + A * y**2)

        f_friction, kappa_cr = _wire_friction_and_kappa_cr(
            w, section, T_per_wire, mu
        )

        if kappa_cr == float("inf"):
            continue

        kappa_crs.append(kappa_cr)

        # 滑り量（1ピッチあたり）
        u_at_kappa_max = abs(kappa_max * y * sa * ca * p / (2.0 * math.pi))
        u_at_kappa_cr = abs(kappa_cr * y * sa * ca * p / (2.0 * math.pi))

        if kappa_max > kappa_cr:
            n_slipping += 1
            delta_u = u_at_kappa_max - u_at_kappa_cr
            W_total += 4.0 * f_friction * delta_u

    return {
        "dissipation_energy": W_total,
        "n_slipping": n_slipping,
        "n_total": n_outer,
        "slip_ratio": n_slipping / n_outer if n_outer > 0 else 0.0,
        "kappa_cr_min": min(kappa_crs) if kappa_crs else float("inf"),
        "kappa_cr_max": max(kappa_crs) if kappa_crs else float("inf"),
        "EI_min": EI_min,
        "EI_max": EI_max,
        "EI_ratio": EI_max / EI_min if EI_min > 0 else 1.0,
    }


def calibrate_pretension(
    section: CableSection,
    kappa_ref: float,
    W_target: float,
    mu: float = 0.15,
    T_range: tuple[float, float] = (0.1, 1e4),
    tol: float = 1e-3,
    max_iter: int = 50,
) -> float:
    """数値モデルの散逸エネルギーにマッチする初期張力を二分法で求める.

    散逸 W(T) は T に対して単峰（T→0 で W→0, T→∞ で全スティック→W→0）。
    W_target が達成可能な範囲にある場合、2つの解が存在するが
    低張力側（物理的に妥当な側）を返す。

    Args:
        section: ケーブル断面
        kappa_ref: キャリブレーション用の最大曲率 [1/mm]
        W_target: 目標散逸エネルギー [N·mm²]
        mu: 摩擦係数
        T_range: 探索範囲 [N]
        tol: 相対許容誤差
        max_iter: 最大反復数

    Returns:
        キャリブレーションされた初期張力 T [N]
    """
    # まず T の範囲で W(T) を評価し、ピークを見つける
    n_scan = 40
    T_scan = np.logspace(
        np.log10(max(T_range[0], 0.01)), np.log10(T_range[1]), n_scan
    )
    W_scan = np.zeros(n_scan)
    for i, T in enumerate(T_scan):
        result = dissipation_energy_bending(
            section, kappa_max=kappa_ref, mu=mu, T_pretension=T
        )
        W_scan[i] = result["dissipation_energy"]

    W_peak = W_scan.max()
    if W_target > W_peak * 1.05:
        # ピーク付近の T を返す（達成不可能）
        return float(T_scan[np.argmax(W_scan)])

    # 低張力側で W = W_target となる T を二分法で探索
    i_peak = int(np.argmax(W_scan))
    T_lo = float(T_scan[0])
    T_hi = float(T_scan[min(i_peak, n_scan - 1)])

    for _ in range(max_iter):
        T_mid = (T_lo + T_hi) / 2.0
        result = dissipation_energy_bending(
            section, kappa_max=kappa_ref, mu=mu, T_pretension=T_mid
        )
        W_mid = result["dissipation_energy"]

        if abs(W_mid - W_target) / max(W_target, 1e-20) < tol:
            return T_mid

        if W_mid < W_target:
            T_lo = T_mid
        else:
            T_hi = T_mid

    return (T_lo + T_hi) / 2.0


def dissipation_energy_bending_distributed(
    section: CableSection,
    kappa_max: float,
    mu: float = 0.15,
    T_pretension: float = 0.0,
    kappa_cr_max: float = 0.01,
    threshold_exponent: float = 0.85,
    n_sub: int = 50,
) -> dict[str, float]:
    """多層摩擦閾値分布モデルによる曲げ散逸エネルギー.

    各素線の摩擦容量を n_sub 個のサブコンタクトに分配。
    閾値分布 F(κ) = (κ/K)^α により、漸進的スリップを記述。

    κ << K のとき: W ∝ κ^(α+1)
    κ >> K のとき: W ∝ κ（全スリップ、Papailiou古典解と一致）

    閉形式:
      W_wire = 4 × f × c × α/(α+1) × κ^(α+1) / K^α   (κ ≤ K)
      W_wire = 4 × f × c × [κ - K/(α+1)]                (κ > K)
    ここで c = |y| × sin(α) × cos(α) × p/(2π)

    Args:
        section: ケーブル断面
        kappa_max: 最大曲率 [1/mm]
        mu: 摩擦係数
        T_pretension: 初期張力 [N]
        kappa_cr_max: 閾値分布の上限 K [1/mm]
        threshold_exponent: 分布指数 α (κ^(α+1) 冪を制御)
        n_sub: サブコンタクト数（検証用、閉形式では不使用）

    Returns:
        dict with dissipation_energy, kappa_power_law, etc.
    """
    E = section.E
    A = section.A_wire
    p = section.pitch_length
    n = section.n_strands
    K = kappa_cr_max
    alpha = threshold_exponent

    T_per_wire = T_pretension / n if n > 0 else 0.0

    W_total = 0.0
    n_outer = 0
    EI_min = 0.0
    EI_max = 0.0

    for w in section.wires:
        if w.layer == 0:
            EI_min += E * section.I_wire
            EI_max += E * section.I_wire
            continue

        n_outer += 1
        y = w.y
        sa = math.sin(w.alpha)
        ca = math.cos(w.alpha)

        EI_min += E * section.I_wire
        EI_max += E * (section.I_wire + A * y**2)

        # 摩擦容量（初期張力由来）
        f_friction, _ = _wire_friction_and_kappa_cr(
            w, section, T_per_wire, mu
        )
        if f_friction < 1e-15:
            continue

        # 滑り係数 c = |y| × sin(α) × cos(α) × p/(2π)
        c = abs(y) * sa * ca * p / (2.0 * math.pi)
        if c < 1e-20:
            continue

        # 閉形式散逸（分布閾値モデル）
        if kappa_max <= 0:
            continue
        elif kappa_max <= K:
            W_wire = 4.0 * f_friction * c * alpha / (alpha + 1.0) * (
                kappa_max ** (alpha + 1.0) / K**alpha
            )
        else:
            W_wire = 4.0 * f_friction * c * (
                kappa_max - K / (alpha + 1.0)
            )

        W_total += W_wire

    return {
        "dissipation_energy": W_total,
        "n_total": n_outer,
        "kappa_power_law": alpha + 1.0,
        "kappa_cr_max": K,
        "threshold_exponent": alpha,
        "EI_min": EI_min,
        "EI_max": EI_max,
    }


def calibrate_distributed_model(
    section: CableSection,
    kappa_W_pairs: list[tuple[float, float]],
    mu: float = 0.15,
    T_pretension: float = 10.0,
) -> dict[str, float]:
    """分布閾値モデルのパラメータを最小二乗法でキャリブレーション.

    κ-W データ点から (T_pretension, kappa_cr_max, alpha) を推定。

    Args:
        section: ケーブル断面
        kappa_W_pairs: [(κ_1, W_1), (κ_2, W_2), ...] 数値モデルの結果
        mu: 摩擦係数
        T_pretension: 初期推定の張力 [N]

    Returns:
        dict with T_pretension, kappa_cr_max, threshold_exponent
    """
    kappas = np.array([p[0] for p in kappa_W_pairs])
    W_targets = np.array([p[1] for p in kappa_W_pairs])

    # κ冪指数を log-log 回帰で推定
    mask = (kappas > 0) & (W_targets > 0)
    if mask.sum() < 2:
        return {
            "T_pretension": T_pretension,
            "kappa_cr_max": 0.01,
            "threshold_exponent": 0.85,
        }

    coeffs = np.polyfit(np.log(kappas[mask]), np.log(W_targets[mask]), 1)
    power_law = coeffs[0]
    alpha = max(0.1, power_law - 1.0)  # W ∝ κ^(α+1) → α = power - 1

    # K と T をグリッド探索で最適化
    best_err = float("inf")
    best_T = T_pretension
    best_K = 0.01

    for K in np.logspace(-3, -1, 30):
        for T in np.logspace(-1, 3, 30):
            W_pred = np.array([
                dissipation_energy_bending_distributed(
                    section, kappa_max=k, mu=mu, T_pretension=T,
                    kappa_cr_max=K, threshold_exponent=alpha,
                )["dissipation_energy"]
                for k in kappas
            ])
            err = np.sum(((W_pred - W_targets) / np.maximum(W_targets, 1e-20)) ** 2)
            if err < best_err:
                best_err = err
                best_T = T
                best_K = K

    return {
        "T_pretension": float(best_T),
        "kappa_cr_max": float(best_K),
        "threshold_exponent": float(alpha),
        "kappa_power_law": float(alpha + 1.0),
        "fit_error_rms": float(np.sqrt(best_err / len(kappas))),
    }


def dissipation_energy_combined(
    section: CableSection,
    kappa_max: float = 0.0,
    tau_max: float = 0.0,
    mu: float = 0.15,
    T_pretension: float = 0.0,
) -> dict[str, float]:
    """曲げ+捻り複合1サイクルの散逸エネルギーを閉形式で計算.

    曲げと捻りの相対滑りをベクトル合成し、
    合計滑り量が摩擦容量を超える素線で散逸が発生。
    摩擦力源は初期張力のみ（Papailiou標準定式化）。

    Args:
        section: ケーブル断面
        kappa_max: 最大曲率 [1/mm]
        tau_max: 最大捻率 [1/mm]
        mu: 摩擦係数
        T_pretension: 初期張力 [N]

    Returns:
        dict with dissipation_energy, W_bending, W_torsion, W_coupling, etc.
    """
    E = section.E
    A = section.A_wire
    p = section.pitch_length
    n = section.n_strands

    T_per_wire = T_pretension / n if n > 0 else 0.0

    W_total = 0.0
    W_bend_only = 0.0
    W_torsion_only = 0.0
    n_slipping = 0
    n_outer = 0

    for w in section.wires:
        if w.layer == 0:
            continue

        n_outer += 1
        R_h = w.R_helix
        y = w.y
        sa = math.sin(w.alpha)
        ca = math.cos(w.alpha)

        if R_h < 1e-10 or abs(ca) < 1e-10:
            continue

        # 摩擦容量（初期張力のみ）
        f_friction, _ = _wire_friction_and_kappa_cr(
            w, section, T_per_wire, mu
        )

        if f_friction < 1e-15:
            continue

        # 相対滑り量（ベクトル合成）
        u_bend = abs(kappa_max * y * sa * ca * p / (2.0 * math.pi))
        u_torsion = abs(tau_max * R_h * ca * p / (2.0 * math.pi))
        u_combined = math.sqrt(u_bend**2 + u_torsion**2)

        # 弾性域: u_cr = f / k_wire, k_wire = E×A×cos(α)/p
        k_wire = E * A * ca / p
        u_elastic = f_friction / k_wire if k_wire > 1e-10 else 0.0

        if u_combined > u_elastic:
            n_slipping += 1
            delta_u = u_combined - u_elastic
            W_wire = 4.0 * f_friction * delta_u
            W_total += W_wire

            # 成分分解（方向比率で按分）
            if u_combined > 1e-20:
                ratio_bend = u_bend / u_combined
                ratio_torsion = u_torsion / u_combined
                W_bend_only += W_wire * ratio_bend
                W_torsion_only += W_wire * ratio_torsion

    W_coupling = W_total - W_bend_only - W_torsion_only

    return {
        "dissipation_energy": W_total,
        "W_bending": W_bend_only,
        "W_torsion": W_torsion_only,
        "W_coupling": W_coupling,
        "n_slipping": n_slipping,
        "n_total": n_outer,
        "slip_ratio": n_slipping / n_outer if n_outer > 0 else 0.0,
    }


# ====================================================================
# M-κ 曲線の解析的構成
# ====================================================================


def mk_curve_analytical(
    section: CableSection,
    kappa_max: float,
    n_points: int = 200,
    mu: float = 0.15,
    T_pretension: float = 0.0,
) -> tuple[NDArray, NDArray, NDArray]:
    """M-κ ヒステリシスループを解析的に構成.

    Returns:
        (kappas, M_loading, M_unloading) の3配列
    """
    kappas = np.linspace(0, kappa_max, n_points)
    M_loading = np.zeros(n_points)
    M_unloading = np.zeros(n_points)

    for idx, kappa in enumerate(kappas):
        resp_load = compute_section_response(
            section, kappa_z=kappa, mu=mu, T_pretension=T_pretension
        )
        M_loading[idx] = -resp_load.M_z  # 符号調整

        # 除荷: κ_max → κ（逆方向からの弾性復帰）
        # 除荷時の M = M(κ_max) - ΔM_elastic(κ_max - κ)
        # 全素線がスティックに戻る（弾性除荷）ので EI_max で除荷
        resp_max = compute_section_response(
            section, kappa_z=kappa_max, mu=mu, T_pretension=T_pretension
        )
        M_at_max = -resp_max.M_z
        # resp_max.EI_effective: 除荷時は全スティック → EI_max に近い

        # 除荷パス: 弾性的に EI_max で戻る
        # ただし 2×κ_cr 分進むと再び逆方向にスリップ開始
        # 簡略化: 除荷は EI_max の傾きで線形
        delta_kappa = kappa_max - kappa

        # 全素線の EI_max
        EI_max_full = sum(
            section.E * (section.I_wire + section.A_wire * w.y**2)
            for w in section.wires
        )
        M_unloading[idx] = M_at_max - EI_max_full * delta_kappa

    return kappas, M_loading, M_unloading
