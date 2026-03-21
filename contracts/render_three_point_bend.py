"""動的3点曲げの3Dレンダリング結果出力.

接触なし版の動的三点曲げを実行し、変形メッシュの3Dビューを
docs/verification/ に出力する。

[← README](../README.md)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# プロジェクトルートを追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendJigConfig,
    DynamicThreePointBendJigProcess,
)
from xkep_cae.output.stress_contour import (
    ContourFieldInput,
    StressContour3DConfig,
    StressContour3DProcess,
)


def main() -> None:
    """動的3点曲げの3Dレンダリングを実行."""
    output_dir = Path("docs/verification/three_point_bend")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  動的3点曲げ 3Dレンダリング")
    print("=" * 60)

    # 1. 動的三点曲げ実行（接触なし、変位制御）
    cfg = DynamicThreePointBendJigConfig(
        wire_length=100.0,
        wire_diameter=2.0,
        n_elems_wire=40,
        E=200e3,  # MPa (鉄鋼)
        nu=0.3,
        rho=7.85e-9,  # ton/mm³
        jig_push=1.0,  # mm
        n_periods=2.0,
        rho_inf=0.9,
    )
    print(f"\n  設定: L={cfg.wire_length}mm, d={cfg.wire_diameter}mm, "
          f"n_elem={cfg.n_elems_wire}, push={cfg.jig_push}mm")
    print(f"  材料: E={cfg.E:.0f}MPa, ν={cfg.nu}, ρ={cfg.rho:.2e} ton/mm³")

    proc = DynamicThreePointBendJigProcess()
    result = proc.process(cfg)

    sr = result.solver_result
    print(f"\n  結果: converged={sr.converged}, "
          f"n_incr={sr.n_increments}, "
          f"elapsed={sr.elapsed_seconds:.1f}s")
    print(f"  最大変位: {result.wire_midpoint_deflection:.4f} mm")
    print(f"  解析解変位: {result.analytical_deflection_static:.4f} mm")
    print(f"  解析周波数: {result.analytical_frequency_hz:.1f} Hz")

    if not sr.converged:
        print("  WARNING: 収束しなかった。利用可能なデータでレンダリングを実行。")

    # 2. ひずみフィールド計算
    n_elems = cfg.n_elems_wire
    disp_hist = sr.displacement_history
    n_frames = min(6, len(disp_hist))
    if n_frames == 0:
        print("  ERROR: 変位履歴なし。レンダリングスキップ。")
        return

    # 等間隔フレーム選択
    indices = np.linspace(0, len(disp_hist) - 1, n_frames, dtype=int)
    selected_disp = [disp_hist[i] for i in indices]

    # 要素曲げひずみ計算（EB近似: ε = κ * r）
    from xkep_cae.numerical_tests.beam_oscillation import (
        ElementBendingStrainInput,
        ElementBendingStrainProcess,
    )
    from xkep_cae.core import MeshData

    # メッシュデータ構築
    L = cfg.wire_length
    n_nodes = n_elems + 1
    node_coords = np.zeros((n_nodes, 3))
    node_coords[:, 0] = np.linspace(0, L, n_nodes)
    connectivity = np.array([[i, i + 1] for i in range(n_elems)])
    wire_radius = cfg.wire_diameter / 2.0
    mesh = MeshData(
        node_coords=node_coords,
        connectivity=connectivity,
        radii=wire_radius,
        n_strands=1,
    )

    # 全フレームのひずみ計算
    strain_proc = ElementBendingStrainProcess()
    s11_snaps = []
    le11_snaps = []
    sk1_snaps = []

    for u in selected_disp:
        strain_out = strain_proc.process(
            ElementBendingStrainInput(
                node_coords=node_coords,
                connectivity=connectivity,
                u=u,
                wire_radius=wire_radius,
            )
        )
        # S11 = E * ε, LE11 = ε, SK1 = κ
        le11_snaps.append(strain_out.element_strain)
        s11_snaps.append(strain_out.element_strain * cfg.E)
        sk1_snaps.append(strain_out.element_curvature)

    # 3. 時刻値
    load_hist = sr.load_history
    f1 = result.analytical_frequency_hz
    T1 = 1.0 / f1 if f1 > 0 else 1.0
    t_total = cfg.n_periods * T1
    if len(load_hist) >= len(disp_hist):
        time_vals = np.array([load_hist[i] * t_total for i in indices])
    else:
        time_vals = np.linspace(0, t_total, n_frames)

    # 4. 3Dレンダリング
    contour_fields = [
        ContourFieldInput(name="S11", snapshots=s11_snaps),
        ContourFieldInput(name="LE11", snapshots=le11_snaps),
        ContourFieldInput(name="SK1", snapshots=sk1_snaps),
    ]

    render_cfg = StressContour3DConfig(
        mesh=mesh,
        node_coords_initial=node_coords,
        displacement_snapshots=selected_disp,
        contour_fields=contour_fields,
        time_values=time_vals,
        wire_radius=wire_radius,
        output_dir=str(output_dir),
        prefix="dynamic_three_point_bend",
        n_render_frames=n_frames,
    )

    print(f"\n  3Dレンダリング: {n_frames} フレーム × 3 フィールド")
    render_result = StressContour3DProcess().process(render_cfg)

    print(f"  出力画像: {len(render_result.image_paths)} 枚")
    for path in render_result.image_paths[:5]:
        print(f"    {path}")
    if len(render_result.image_paths) > 5:
        print(f"    ... 他 {len(render_result.image_paths) - 5} 枚")

    print(f"\n  フィールド最大値:")
    for name, val in render_result.field_max_values.items():
        print(f"    {name}: {val:.4e}")

    # 5. インクリメント診断サマリ
    if sr.increment_diagnostics:
        print(f"\n  インクリメント診断: {len(sr.increment_diagnostics)} 件")
        print(f"  {'step':>5} {'frac':>8} {'att':>4} {'res':>10} {'rate':>8} "
              f"{'du_norm':>10} {'n_act':>6}")
        for d in sr.increment_diagnostics[:10]:
            print(f"  {d.step:5d} {d.load_frac:8.4f} {d.n_attempts:4d} "
                  f"{d.final_residual:10.3e} {d.convergence_rate:8.4f} "
                  f"{d.du_norm:10.3e} {d.n_active:6d}")
        if len(sr.increment_diagnostics) > 10:
            print(f"  ... 他 {len(sr.increment_diagnostics) - 10} 件")

    print("\n" + "=" * 60)
    print(f"  完了。出力先: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
