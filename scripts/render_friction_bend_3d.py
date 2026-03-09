"""摩擦あり曲げ揺動の3Dパイプレンダリング出力スクリプト.

初期状態・計算途中・最終状態の3D断面付きレンダリング画像を生成する。

[← README](../../README.md)
"""

import sys
import time
from pathlib import Path

import numpy as np

from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh
from xkep_cae.numerical_tests.wire_bending_benchmark import run_bending_oscillation
from xkep_cae.output.render_beam_3d import render_twisted_wire_3d


def _common_params() -> dict:
    """全ケース共通パラメータ."""
    return dict(
        n_elems_per_strand=16,
        n_pitches=0.5,
        max_iter=50,
        tol_force=1e-4,
        show_progress=True,
        use_ncp=True,
        use_mortar=True,
        adaptive_timestepping=True,
        exclude_same_layer=True,
        midpoint_prescreening=True,
        use_line_search=False,
        g_on=0.0005,
        g_off=0.001,
        use_updated_lagrangian=True,
        use_friction=True,
        mu=0.1,
    )


def _deformed_coords(mesh, u_snap: np.ndarray) -> np.ndarray:
    """変位スナップショットから変形後座標を計算."""
    coords = mesh.node_coords.copy()
    for i in range(mesh.n_nodes):
        coords[i, 0] += u_snap[6 * i]
        coords[i, 1] += u_snap[6 * i + 1]
        coords[i, 2] += u_snap[6 * i + 2]
    return coords


def _save_render(mesh, coords, title: str, filepath: Path, elev=25.0, azim=-60.0):
    """3Dレンダリングを保存."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = render_twisted_wire_3d(
        mesh,
        node_coords=coords,
        elev=elev,
        azim=azim,
        title=title,
        figsize=(12, 10),
        dpi=150,
        n_circ=12,
    )
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {filepath}")


def main():
    output_dir = Path("docs/verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    # メッシュ生成（レンダリング用）
    mesh = make_twisted_wire_mesh(
        7, 2.0, 40.0, length=0.0, n_elems_per_strand=16, n_pitches=0.5
    )

    # --- 初期状態レンダリング ---
    print("\n=== 初期状態 3Dレンダリング ===")
    _save_render(
        mesh,
        mesh.node_coords,
        "7-strand Initial (μ=0.1)",
        output_dir / "friction_bend_initial_iso.png",
    )
    _save_render(
        mesh,
        mesh.node_coords,
        "7-strand Initial — Side (XZ)",
        output_dir / "friction_bend_initial_xz.png",
        elev=0.0,
        azim=0.0,
    )

    # --- Case 1: 45度曲げ ---
    print("\n=== Case 1: 45度曲げ 計算中 ===")
    t0 = time.perf_counter()
    r1 = run_bending_oscillation(
        n_strands=7,
        bend_angle_deg=45.0,
        oscillation_amplitude_mm=0.0,
        n_cycles=0,
        n_steps_per_quarter=1,
        **_common_params(),
    )
    print(f"  Case 1: converged={r1.phase1_converged}, {time.perf_counter() - t0:.1f}s")

    if r1.displacement_snapshots:
        coords_45 = _deformed_coords(mesh, r1.displacement_snapshots[-1])
        _save_render(
            mesh,
            coords_45,
            "Case 1: 45° bend (μ=0.1)",
            output_dir / "friction_bend_case1_45deg_iso.png",
        )
        _save_render(
            mesh,
            coords_45,
            "Case 1: 45° bend — Side (XZ)",
            output_dir / "friction_bend_case1_45deg_xz.png",
            elev=0.0,
            azim=0.0,
        )

    # --- Case 2: 90度曲げ ---
    print("\n=== Case 2: 90度曲げ 計算中 ===")
    t0 = time.perf_counter()
    r2 = run_bending_oscillation(
        n_strands=7,
        bend_angle_deg=90.0,
        oscillation_amplitude_mm=0.0,
        n_cycles=0,
        n_steps_per_quarter=1,
        **_common_params(),
    )
    print(f"  Case 2: converged={r2.phase1_converged}, {time.perf_counter() - t0:.1f}s")

    if r2.displacement_snapshots:
        coords_90 = _deformed_coords(mesh, r2.displacement_snapshots[-1])
        _save_render(
            mesh,
            coords_90,
            "Case 2: 90° bend (μ=0.1)",
            output_dir / "friction_bend_case2_90deg_iso.png",
        )
        _save_render(
            mesh,
            coords_90,
            "Case 2: 90° bend — Side (XZ)",
            output_dir / "friction_bend_case2_90deg_xz.png",
            elev=0.0,
            azim=0.0,
        )

    # --- Case 3: 90度曲げ + 揺動 ---
    if r2.phase1_converged:
        print("\n=== Case 3: 90度曲げ + 揺動1周期 計算中 ===")
        t0 = time.perf_counter()
        r3 = run_bending_oscillation(
            n_strands=7,
            bend_angle_deg=90.0,
            oscillation_amplitude_mm=2.0,
            n_cycles=1,
            n_steps_per_quarter=3,
            **_common_params(),
        )
        print(
            f"  Case 3: phase1={r3.phase1_converged}, phase2={r3.phase2_converged}, "
            f"{time.perf_counter() - t0:.1f}s"
        )

        if r3.displacement_snapshots:
            # 途中（中間スナップショット）
            mid_idx = len(r3.displacement_snapshots) // 2
            coords_mid = _deformed_coords(mesh, r3.displacement_snapshots[mid_idx])
            mid_label = (
                r3.snapshot_labels[mid_idx]
                if mid_idx < len(r3.snapshot_labels)
                else f"Step {mid_idx}"
            )
            _save_render(
                mesh,
                coords_mid,
                f"Case 3: Mid ({mid_label})",
                output_dir / "friction_bend_case3_mid_iso.png",
            )

            # 最終状態
            coords_final = _deformed_coords(mesh, r3.displacement_snapshots[-1])
            _save_render(
                mesh,
                coords_final,
                "Case 3: Final (90° + osc, μ=0.1)",
                output_dir / "friction_bend_case3_final_iso.png",
            )
            _save_render(
                mesh,
                coords_final,
                "Case 3: Final — Side (XZ)",
                output_dir / "friction_bend_case3_final_xz.png",
                elev=0.0,
                azim=0.0,
            )
    else:
        print("\n  Case 2 未収束 → Case 3 スキップ")

    print("\n=== 完了 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
