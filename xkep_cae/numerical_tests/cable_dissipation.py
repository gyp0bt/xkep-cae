"""CableDissipationProcess — ケーブル曲げ散逸エネルギー計算 Process.

ファイバー梁モデルによるケーブルの曲げヒステリシス散逸エネルギーを計算する。
接触モデルとの比較（Phase F5 検証）およびパラメータ感度分析に使用。

物理的背景:
  撚線ケーブルの曲げでは素線間摩擦により M-κ ヒステリシスループが発生。
  ループ面積 = 1サイクルあたりの散逸エネルギー [N·mm²]。
  散逸は撚線本数・撚構成・ピッチ・曲率振幅に対して非線形に依存。

散逸エネルギーの非線形依存性:
  - 撚線本数 n: 接触ペア数 ∝ n → 散逸 ∝ n（ただし外層の lever arm 効果で超線形）
  - ピッチ p: ヘリックス角 α = atan(2πR/p) → 短ピッチで滑り増大 → 散逸 ∝ 1/p
  - 曲率 κ: 低κでは弾性（散逸≈0）、高κでは飽和 → S字型の非線形応答
  - モード: 曲げ vs ねじり vs 複合。曲げ散逸が支配的（法線力×接線滑り）

status-331: Phase F5 散逸エネルギー検証。

[← README](../../README.md)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from xkep_cae.core import BatchProcess, ProcessMeta, SolverResultData
from xkep_cae.elements.fiber import (
    MultiLayerFrictionDegrading1D,
)
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)

# ====================================================================
# 純粋関数: M-κ ループ面積計算
# ====================================================================


def _compute_mk_loop_area(mk_history: tuple | list) -> float:
    """M-κ 履歴からヒステリシスループ面積（散逸エネルギー）を計算.

    台形則による数値積分: ∮ M dκ（符号付き面積）。
    閉ループの場合、正の面積 = 散逸エネルギー。

    Args:
        mk_history: ((κ_0, M_0), (κ_1, M_1), ...) の系列

    Returns:
        ループ面積 [N·mm²]（正 = 散逸あり）
    """
    if len(mk_history) < 3:
        return 0.0

    kappas = np.array([p[0] for p in mk_history])
    moments = np.array([p[1] for p in mk_history])

    # Shoelace formula（符号付き面積）
    area = 0.0
    n = len(kappas)
    for i in range(n - 1):
        area += (kappas[i + 1] - kappas[i]) * (moments[i + 1] + moments[i]) / 2.0

    return abs(area)


def _compute_mk_metrics(
    mk_history: tuple | list,
) -> dict[str, float]:
    """M-κ 履歴からヒステリシス指標を計算.

    Returns:
        dict with keys:
          - loop_area: ループ面積 [N·mm²]
          - peak_moment: 最大モーメント [N·mm]
          - peak_curvature: 最大曲率 [1/mm]
          - EI_secant: 割線剛性 M_peak / κ_peak [N·mm²]
          - EI_initial: 初期接線剛性 [N·mm²]
          - loading_work: 負荷時の仕事 [N·mm²]
          - unloading_work: 除荷時の仕事 [N·mm²]
          - dissipation_ratio: 散逸/負荷仕事比
    """
    if len(mk_history) < 3:
        return {
            "loop_area": 0.0,
            "peak_moment": 0.0,
            "peak_curvature": 0.0,
            "EI_secant": 0.0,
            "EI_initial": 0.0,
            "loading_work": 0.0,
            "unloading_work": 0.0,
            "dissipation_ratio": 0.0,
        }

    kappas = np.array([p[0] for p in mk_history])
    moments = np.array([p[1] for p in mk_history])

    peak_idx = int(np.argmax(np.abs(kappas)))
    peak_kappa = float(kappas[peak_idx])
    peak_moment = float(moments[peak_idx])

    # 負荷パス: κ 増加方向
    # 除荷パス: κ 減少方向
    loading_work = 0.0
    unloading_work = 0.0
    for i in range(len(kappas) - 1):
        dk = kappas[i + 1] - kappas[i]
        dm_avg = (moments[i + 1] + moments[i]) / 2.0
        work = dk * dm_avg
        if dk > 0:
            loading_work += work
        else:
            unloading_work += abs(work)

    loop_area = abs(loading_work - unloading_work)

    # 割線剛性
    EI_secant = abs(peak_moment / peak_kappa) if abs(peak_kappa) > 1e-30 else 0.0

    # 初期接線剛性（最初の2点から）
    if len(kappas) >= 2 and abs(kappas[1] - kappas[0]) > 1e-30:
        EI_initial = abs((moments[1] - moments[0]) / (kappas[1] - kappas[0]))
    else:
        EI_initial = EI_secant

    dissipation_ratio = loop_area / loading_work if loading_work > 1e-30 else 0.0

    return {
        "loop_area": loop_area,
        "peak_moment": abs(peak_moment),
        "peak_curvature": abs(peak_kappa),
        "EI_secant": EI_secant,
        "EI_initial": EI_initial,
        "loading_work": loading_work,
        "unloading_work": unloading_work,
        "dissipation_ratio": dissipation_ratio,
    }


# ====================================================================
# ケーブル断面幾何計算
# ====================================================================


def _cable_geometry(
    n_strands: int,
    wire_radius: float,
    pitch_length: float,
) -> dict[str, float]:
    """撚線ケーブルの幾何パラメータを計算.

    1+6 配置（n=7）を基準に、一般 n の概算も提供。

    Returns:
        dict with keys:
          - cable_diameter: ケーブル外径 [mm]
          - helix_radius: ヘリックス半径 [mm]
          - helix_angle: ヘリックス角 [rad]
          - EI_min: 全滑り時の曲げ剛性比（E×I_min / E）[mm⁴]
          - EI_max: 全ロック時の曲げ剛性比（E×I_max / E）[mm⁴]
          - EI_ratio: EI_max / EI_min
    """
    R = wire_radius
    A_wire = math.pi * R**2
    I_wire = math.pi / 4.0 * R**4

    if n_strands <= 1:
        return {
            "cable_diameter": 2.0 * R,
            "helix_radius": 0.0,
            "helix_angle": 0.0,
            "EI_min": I_wire,
            "EI_max": I_wire,
            "EI_ratio": 1.0,
        }

    # 1+6 配置: 6外層素線のヘリックス半径 = 2R（芯線と外層が接触）
    if n_strands == 7:
        n_outer = 6
        R_helix = 2.0 * R
    elif n_strands == 19:
        # 1+6+12: 2層目は R_helix2 = 4R
        n_outer = 6 + 12
        R_helix = 2.0 * R  # 第1層（代表値）
    elif n_strands == 2:
        n_outer = 1
        R_helix = 2.0 * R
    else:
        # 概算: n_outer ≈ n-1, R_helix ≈ 2R
        n_outer = n_strands - 1
        R_helix = 2.0 * R

    cable_diameter = 2.0 * (R_helix + R)
    alpha = math.atan(2.0 * math.pi * R_helix / pitch_length)

    # EI_min: 各素線の自己回りの曲げ剛性の和
    EI_min = float(n_strands) * I_wire

    # EI_max: 全素線がロックした場合の断面2次モーメント
    # = Σ I_wire + Σ A_wire × R_i² (Steiner の平行軸定理)
    if n_strands == 7:
        EI_max = n_strands * I_wire + 6 * A_wire * R_helix**2
    elif n_strands == 19:
        R_helix2 = 4.0 * R
        EI_max = n_strands * I_wire + 6 * A_wire * R_helix**2 + 12 * A_wire * R_helix2**2
    else:
        EI_max = n_strands * I_wire + n_outer * A_wire * R_helix**2

    EI_ratio = EI_max / EI_min if EI_min > 0 else 1.0

    return {
        "cable_diameter": cable_diameter,
        "helix_radius": R_helix,
        "helix_angle": alpha,
        "EI_min": EI_min,
        "EI_max": EI_max,
        "EI_ratio": EI_ratio,
    }


# ====================================================================
# 自動材料パラメータ生成
# ====================================================================


def _make_cable_material(
    n_strands: int = 7,
    wire_radius: float = 0.5,
    pitch_length: float = 100.0,
    E: float = 130.0e3,
    mu: float = 0.15,
    n_layers: int = 30,
    degrade_ratio: float = 0.25,
) -> tuple[MultiLayerFrictionDegrading1D, dict[str, float]]:
    """ケーブル幾何からファイバー梁用 MultiLayerFriction 材料を自動生成.

    Foti & Martinelli (2016) の物理モデルに基づき、
    ケーブル断面パラメータから等価材料パラメータを推定。

    パラメータ推定の根拠:
      - E_base = E × EI_min / I_fiber (全滑り時の等価剛性)
      - k_total = E × (EI_max - EI_min) / I_fiber (摩擦ロック時の追加剛性)
      - f_y: 対数間隔の降伏閾値（漸進的スリップ → 滑らかなティアドロップ）
      - β: 接触剛性劣化比（初回スリップ後の剛性低下）

    Args:
        n_strands: 素線本数
        wire_radius: 素線半径 [mm]
        pitch_length: ピッチ長 [mm]
        E: ヤング率 [MPa]
        mu: 摩擦係数
        n_layers: 摩擦層数
        degrade_ratio: 接触剛性劣化比 β

    Returns:
        (material, info_dict)
    """
    geom = _cable_geometry(n_strands, wire_radius, pitch_length)
    d_cable = geom["cable_diameter"]
    I_fiber = math.pi / 64.0 * d_cable**4

    EI_min = geom["EI_min"]
    EI_max = geom["EI_max"]

    # 等価材料パラメータ
    E_base = E * EI_min / I_fiber
    k_contact_total = E * (EI_max - EI_min) / I_fiber

    # 段階的剛性: 外層ソフト・内層スティフ（05_smooth_teardrop.py 方式）
    k_weights = np.linspace(0.5, 1.5, n_layers)
    k_weights *= n_layers / k_weights.sum()
    k_each = k_contact_total / n_layers
    k_virgin = k_each * k_weights

    # 対数間隔の降伏ひずみ → f_y = k × eps_y
    # 基準ひずみスケール: mu に比例（摩擦力が大きいほど降伏が遅い）
    eps_min = 1e-4
    eps_max = 0.01 * (1.0 + mu * 10.0)
    eps_y = np.logspace(np.log10(eps_min), np.log10(eps_max), n_layers)
    f_y = k_virgin * eps_y

    k_degraded = k_virgin * degrade_ratio

    material = MultiLayerFrictionDegrading1D(
        E_base=E_base,
        k_virgin=k_virgin,
        k_degraded=k_degraded,
        f_y=f_y,
    )

    info = {
        "E_base": E_base,
        "k_contact_total": k_contact_total,
        "cable_diameter": d_cable,
        "I_fiber": I_fiber,
        "EI_min": EI_min,
        "EI_max": EI_max,
        "EI_ratio": geom["EI_ratio"],
        "helix_angle_deg": math.degrees(geom["helix_angle"]),
        "n_layers": n_layers,
        "degrade_ratio": degrade_ratio,
        "eps_y_min": float(eps_y[0]),
        "eps_y_max": float(eps_y[-1]),
    }

    return material, info


# ====================================================================
# 入出力データ
# ====================================================================


@dataclass(frozen=True)
class CableDissipationConfig:
    """ケーブル散逸エネルギー計算の構成.

    Attributes:
        n_strands: 素線本数
        wire_radius: 素線半径 [mm]
        pitch_length: ピッチ長 [mm]
        n_elements_per_pitch: ピッチあたり要素数
        n_pitches: ピッチ数
        E: ヤング率 [MPa]
        nu: ポアソン比
        bending_curvature: 曲げ曲率 κ_max [1/mm]
        n_increments_per_half: 半サイクルあたりインクリメント数
        n_half_cycles: 半サイクル数（2=負荷+除荷で1ループ）
        mu: 摩擦係数（自動材料生成時）
        n_layers: 摩擦層数
        degrade_ratio: 接触剛性劣化比 β
        fiber_material: 外部指定材料（None=自動生成）
        fiber_n_fiber: ファイバー数
        max_nr_attempts: NR最大反復数
        tol_force: 力収束判定
        show_progress: 進捗表示
    """

    n_strands: int = 7
    wire_radius: float = 0.5
    pitch_length: float = 100.0
    n_elements_per_pitch: int = 16
    n_pitches: float = 1.0
    E: float = 130.0e3
    nu: float = 0.3
    bending_curvature: float = 0.001
    n_increments_per_half: int = 20
    n_half_cycles: int = 2  # 2 = 1 full loop (load + unload)
    mu: float = 0.15
    n_layers: int = 30
    degrade_ratio: float = 0.25
    fiber_material: object | None = None
    fiber_n_fiber: int = 60
    max_nr_attempts: int = 50
    tol_force: float = 1e-8
    show_progress: bool = True


@dataclass(frozen=True)
class CableDissipationResult:
    """ケーブル散逸エネルギー計算の結果.

    Attributes:
        solver_result: ソルバー結果
        dissipation_energy: 散逸エネルギー [N·mm²]
        mk_metrics: M-κ ヒステリシス指標 dict
        cable_info: ケーブル幾何情報 dict
        material_info: 材料パラメータ情報 dict
    """

    solver_result: SolverResultData
    dissipation_energy: float
    mk_metrics: dict
    cable_info: dict
    material_info: dict


# ====================================================================
# Process
# ====================================================================


class CableDissipationProcess(
    BatchProcess[CableDissipationConfig, CableDissipationResult],
):
    """ケーブル曲げ散逸エネルギー計算 Process.

    ファイバー梁モデルでケーブルの M-κ ヒステリシスループを計算し、
    ループ面積（= 散逸エネルギー）を返す。

    パイプライン:
    1. cable_geometry() でケーブル断面幾何を計算
    2. make_cable_material() で等価材料を自動生成（or 外部指定）
    3. StrandBendingOscillationProcess (use_fiber_beam=True) で実行
    4. compute_mk_metrics() で M-κ 指標を計算

    status-331: Phase F5 散逸エネルギー検証。
    """

    meta = ProcessMeta(
        name="CableDissipation",
        module="batch",
        version="1.0.0",
        document_path="docs/cable_dissipation.md",
    )
    uses = [StrandBendingOscillationProcess]

    def process(self, input_data: CableDissipationConfig) -> CableDissipationResult:
        """ケーブル散逸エネルギーを計算."""
        cfg = input_data

        # ── 1. ケーブル断面幾何 ──
        cable_info = _cable_geometry(cfg.n_strands, cfg.wire_radius, cfg.pitch_length)

        # ── 2. 材料生成 ──
        if cfg.fiber_material is not None:
            material = cfg.fiber_material
            material_info = {"source": "external"}
        else:
            material, material_info = _make_cable_material(
                n_strands=cfg.n_strands,
                wire_radius=cfg.wire_radius,
                pitch_length=cfg.pitch_length,
                E=cfg.E,
                mu=cfg.mu,
                n_layers=cfg.n_layers,
                degrade_ratio=cfg.degrade_ratio,
            )
            material_info["source"] = "auto"

        # ── 3. ファイバー梁ソルバー実行 ──
        # ケーブル外径に対応する wire_radius を設定
        # fiber beam の diameter = 2 * wire_radius なので
        # cable_diameter = 2 * effective_wire_radius
        effective_wire_radius = cable_info["cable_diameter"] / 2.0

        osc_cfg = StrandBendingOscillationConfig(
            n_strands=cfg.n_strands,
            wire_radius=effective_wire_radius,
            pitch_length=cfg.pitch_length,
            n_elements_per_pitch=cfg.n_elements_per_pitch,
            n_pitches=cfg.n_pitches,
            E=cfg.E,
            nu=cfg.nu,
            rho=8.96e-9,
            bending_curvature=cfg.bending_curvature,
            n_cycles=cfg.n_half_cycles,
            n_increments_per_cycle=cfg.n_increments_per_half,
            use_fiber_beam=True,
            fiber_material=material,
            fiber_n_fiber=cfg.fiber_n_fiber,
            fiber_section_type="strip",
            max_nr_attempts=cfg.max_nr_attempts,
            tol_force=cfg.tol_force,
        )
        result = StrandBendingOscillationProcess().process(osc_cfg)

        # ── 4. M-κ 指標計算 ──
        mk_hist = result.solver_result.moment_curvature_history
        mk_metrics = _compute_mk_metrics(mk_hist)

        dissipation = mk_metrics["loop_area"]

        if cfg.show_progress:
            print("\n=== ケーブル散逸エネルギー結果 ===")
            print(f"  n_strands: {cfg.n_strands}")
            print(f"  cable_diameter: {cable_info['cable_diameter']:.2f} mm")
            print(f"  EI_ratio: {cable_info['EI_ratio']:.1f}")
            print(f"  κ_max: {cfg.bending_curvature:.4f} 1/mm")
            print(f"  M_peak: {mk_metrics['peak_moment']:.4e} N·mm")
            print(f"  EI_secant: {mk_metrics['EI_secant']:.4e} N·mm²")
            print(f"  EI_initial: {mk_metrics['EI_initial']:.4e} N·mm²")
            print(f"  散逸エネルギー: {dissipation:.4e} N·mm²")
            print(f"  散逸/負荷比: {mk_metrics['dissipation_ratio']:.4f}")

        return CableDissipationResult(
            solver_result=result.solver_result,
            dissipation_energy=dissipation,
            mk_metrics=mk_metrics,
            cable_info=cable_info,
            material_info=material_info,
        )
