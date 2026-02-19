#!/usr/bin/env python3
"""xkep-cae サンプル .inp ファイルの解析実行スクリプト.

各サンプルの .inp ファイルを読み込み、線形静解析を実行して結果を表示する。
解析解が存在する場合は比較結果も出力する。

Usage:
    python examples/run_examples.py                 # 全サンプル実行
    python examples/run_examples.py cantilever      # 片持ち梁のみ
    python examples/run_examples.py three_point     # 3点曲げのみ
    python examples/run_examples.py portal          # 門型フレームのみ
    python examples/run_examples.py l_frame         # L型フレームのみ
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# プロジェクトルートを PYTHONPATH に追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xkep_cae.io import build_beam_model_from_inp, node_dof, read_abaqus_inp, solve_beam_static

EXAMPLES_DIR = Path(__file__).resolve().parent


def run_cantilever_beam_3d():
    """3D 片持ち梁: 先端集中荷重による静解析."""
    print("=" * 60)
    print("片持ち梁（3D Timoshenko, 円形断面）")
    print("=" * 60)

    mesh = read_abaqus_inp(EXAMPLES_DIR / "cantilever_beam_3d.inp")
    model = build_beam_model_from_inp(mesh)

    # 荷重: 先端（node 11）に y 方向 -1000 N
    P = -1000.0
    f = np.zeros(model.ndof_total)
    f[node_dof(model, 11, 1)] = P  # DOF 1 = uy

    result = solve_beam_static(model, f, show_progress=False)
    u = result.u

    # 結果
    tip_uy = u[node_dof(model, 11, 1)]
    tip_theta_z = u[node_dof(model, 11, 5)]

    # 解析解（Timoshenko）
    mat = model.material
    sec = model.sections[0]
    L = 1.0  # 1m
    E, nu = mat.E, mat.nu
    G = E / (2.0 * (1.0 + nu))
    I = sec.Iz
    A = sec.A
    kappa = sec.cowper_kappa_z(nu)
    delta_eb = abs(P) * L**3 / (3.0 * E * I)
    delta_shear = abs(P) * L / (kappa * G * A)
    delta_analytical = delta_eb + delta_shear

    print(f"  荷重: P = {P:.0f} N（y方向先端集中荷重）")
    print(f"  材料: E = {E:.0f} MPa, nu = {nu}")
    print(f"  断面: 円形 R=10mm, A={A:.4f}, Iz={I:.6e}")
    print(f"  先端たわみ (FEM):    {tip_uy:.6e} m")
    print(f"  先端たわみ (解析解): {-delta_analytical:.6e} m")
    error = abs(abs(tip_uy) - delta_analytical) / delta_analytical * 100
    print(f"  相対誤差: {error:.4f}%")
    print(f"  先端回転角: {tip_theta_z:.6e} rad")
    print()
    return error


def run_three_point_bending():
    """3 点曲げ試験: 両端ピン支持、中央集中荷重."""
    print("=" * 60)
    print("3点曲げ試験（2D Timoshenko, 矩形断面）")
    print("=" * 60)

    mesh = read_abaqus_inp(EXAMPLES_DIR / "three_point_bending.inp")
    model = build_beam_model_from_inp(mesh)

    # 荷重: 中央（node 11）に y 方向 -1000 N
    P = -1000.0
    f = np.zeros(model.ndof_total)
    f[node_dof(model, 11, 1)] = P  # DOF 1 = uy

    result = solve_beam_static(model, f, show_progress=False)
    u = result.u

    mid_uy = u[node_dof(model, 11, 1)]

    # 解析解（Timoshenko 3点曲げ）
    mat = model.material
    sec = model.sections[0]
    L = 0.5  # 0.5m
    E, nu = mat.E, mat.nu
    G = E / (2.0 * (1.0 + nu))
    I = sec.I
    A = sec.A
    kappa = sec.cowper_kappa(nu)
    delta_bend = abs(P) * L**3 / (48.0 * E * I)
    delta_shear = abs(P) * L / (4.0 * kappa * G * A)
    delta_analytical = delta_bend + delta_shear

    print(f"  荷重: P = {P:.0f} N（y方向中央集中荷重）")
    print(f"  材料: E = {E:.0f} MPa, nu = {nu}")
    print(f"  断面: 矩形 10mm x 10mm, A={A:.4f}, I={I:.6e}")
    print(f"  スパン: L = {L} m")
    print(f"  中央たわみ (FEM):    {mid_uy:.6e} m")
    print(f"  中央たわみ (解析解): {-delta_analytical:.6e} m")
    error = abs(abs(mid_uy) - delta_analytical) / delta_analytical * 100
    print(f"  相対誤差: {error:.4f}%")
    print()
    return error


def run_portal_frame():
    """門型フレーム: 水平荷重による静解析."""
    print("=" * 60)
    print("門型フレーム（3D Timoshenko, 矩形断面）")
    print("=" * 60)

    mesh = read_abaqus_inp(EXAMPLES_DIR / "portal_frame.inp")
    model = build_beam_model_from_inp(mesh)

    # 荷重: 左柱上端（node 7）に x 方向 500 N
    P = 500.0
    f = np.zeros(model.ndof_total)
    f[node_dof(model, 7, 0)] = P  # DOF 0 = ux

    result = solve_beam_static(model, f, show_progress=False)
    u = result.u

    load_ux = u[node_dof(model, 7, 0)]
    load_uy = u[node_dof(model, 7, 1)]

    print(f"  荷重: P = {P:.0f} N（x方向、左柱上端）")
    print(f"  両脚固定、左柱+梁+右柱（合計 22 要素）")
    print(f"  柱断面: 矩形 0.3m x 0.3m")
    print(f"  梁断面: 矩形 0.3m x 0.5m")
    print(f"  荷重点 x 変位: {load_ux:.6e} m")
    print(f"  荷重点 y 変位: {load_uy:.6e} m")
    print()


def run_l_frame():
    """L 型フレーム: 先端鉛直荷重による静解析."""
    print("=" * 60)
    print("L型フレーム（3D Timoshenko, パイプ断面）")
    print("=" * 60)

    mesh = read_abaqus_inp(EXAMPLES_DIR / "l_frame_3d.inp")
    model = build_beam_model_from_inp(mesh)

    # 荷重: 先端（node 11）に y 方向 -100 N
    P = -100.0
    f = np.zeros(model.ndof_total)
    f[node_dof(model, 11, 1)] = P  # DOF 1 = uy

    result = solve_beam_static(model, f, show_progress=False)
    u = result.u

    tip_ux = u[node_dof(model, 11, 0)]
    tip_uy = u[node_dof(model, 11, 1)]
    tip_uz = u[node_dof(model, 11, 2)]

    print(f"  荷重: P = {P:.0f} N（y方向、水平梁先端）")
    print(f"  基部固定、垂直+水平（合計 10 要素）")
    print(f"  パイプ断面: 外径 50mm, 肉厚 3mm")
    print(f"  先端 x 変位: {tip_ux:.6e} m")
    print(f"  先端 y 変位: {tip_uy:.6e} m")
    print(f"  先端 z 変位: {tip_uz:.6e} m")
    print()


def main():
    """メイン実行."""
    print("=" * 60)
    print("xkep-cae サンプル解析実行")
    print("=" * 60)
    print()

    # 引数でフィルタ
    filter_key = sys.argv[1].lower() if len(sys.argv) > 1 else None

    examples = {
        "cantilever": run_cantilever_beam_3d,
        "three_point": run_three_point_bending,
        "portal": run_portal_frame,
        "l_frame": run_l_frame,
    }

    errors = []
    for name, func in examples.items():
        if filter_key is None or filter_key in name:
            err = func()
            if err is not None:
                errors.append((name, err))

    if errors:
        print("-" * 60)
        print("解析解比較まとめ:")
        for name, err in errors:
            status = "PASS" if err < 1.0 else "CHECK"
            print(f"  {name}: 誤差 {err:.4f}% [{status}]")
        print()


if __name__ == "__main__":
    main()
