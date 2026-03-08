"""7本撚線 曲げ揺動の収束検証スクリプト.

新機能の検証フロー:
  1. 曲げ（Phase1）+ 揺動（Phase2）の収束確認
  2. 収束ログ（カットバック、接触状態、エネルギー）をteeで保存
  3. 変形メッシュの2D投影スナップショットで物理的妥当性を確認

Usage:
  python scripts/verify_7strand_bending_oscillation.py 2>&1 | tee /tmp/verify_7strand.log

[← README](../README.md)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# --- ログ tee設定 ---
log_path = f"/tmp/verify_7strand_{int(time.time())}.log"


class TeeWriter:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


tee = TeeWriter(log_path)
sys.stdout = tee

print("=== 7本撚線 曲げ揺動 収束検証 ===")
print(f"ログ出力先: {log_path}")
print(f"日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

from xkep_cae.numerical_tests.wire_bending_benchmark import run_bending_oscillation  # noqa: E402

# ==================================================================
# 共通パラメータ
# ==================================================================
# mesh_gap=0: メッシュ生成時にcompute_min_safe_gapで自動計算（status-145）
# exclude_same_layer=False: 密な撚線では同層接触が物理的に発生する
# use_mortar=False: Point contact を使用（Mortarは接触チャタリング問題あり）
_COMMON_PARAMS = dict(
    use_ncp=True,
    use_mortar=False,  # Point contact（Mortar接触チャタリング未解決）
    adaptive_timestepping=True,
    use_updated_lagrangian=True,
    exclude_same_layer=False,  # 同層接触を含める（密な撚線で物理的に必要）
    midpoint_prescreening=True,
    use_line_search=True,
    g_on=0.0005,
    g_off=0.001,
    mesh_gap=0.0,  # 自動計算（make_twisted_wire_meshで安全ギャップに引き上げ）
    show_progress=True,
    # 接触安定化（status-145: Phase2チャタリング対策）
    chattering_window=5,
    contact_stabilization=0.3,
    lambda_decay=0.5,
    # Newton安定化: 鞍点系悪条件化時の過大ステップ防止
    du_norm_cap=0.5,
)

# ==================================================================
# 1. 45度曲げ（Phase1のみ）
# ==================================================================
print("=" * 70)
print("  テスト1: 7本撚線 45度曲げ（UL+NCP, point contact, adaptive）")
print("=" * 70)

t0 = time.perf_counter()
result_45 = run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=45,
    max_iter=30,
    tol_force=1e-4,
    n_cycles=0,
    **_COMMON_PARAMS,
)
t_45 = time.perf_counter() - t0
print(f"\n結果: converged={result_45.phase1_converged}, 時間={t_45:.2f}s")
print(f"  NR反復: {result_45.phase1_result.total_newton_iterations}")
print(f"  最大貫入比: {result_45.max_penetration_ratio:.6f}")
print(f"  活性接触ペア: {result_45.n_active_contacts}")

# ==================================================================
# 2. 90度曲げ + 揺動1周期
# ==================================================================
print()
print("=" * 70)
print("  テスト2: 7本撚線 90度曲げ + 揺動1周期（UL+NCP, point contact）")
print("=" * 70)

t0 = time.perf_counter()
result_90 = run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    oscillation_amplitude_mm=2.0,
    n_cycles=1,
    n_steps_per_quarter=3,
    max_iter=50,
    tol_force=1e-4,
    **_COMMON_PARAMS,
)
t_90 = time.perf_counter() - t0
print("\n結果:")
print(f"  Phase1(曲げ): converged={result_90.phase1_converged}")
print(f"  Phase2(揺動): converged={result_90.phase2_converged}")
print(f"  総計算時間: {t_90:.2f}s")
print(f"  活性接触ペア: {result_90.n_active_contacts}")
print(f"  最大貫入比: {result_90.max_penetration_ratio:.6f}")
print(f"  Phase2ステップ数: {len(result_90.phase2_results)}")

# ==================================================================
# 3. 摩擦あり（μ=0.1）45度曲げ
# ==================================================================
print()
print("=" * 70)
print("  テスト3: 7本撚線 摩擦あり（μ=0.1）45度曲げ")
print("=" * 70)

t0 = time.perf_counter()
result_friction = run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=45,
    max_iter=50,
    tol_force=1e-4,
    n_cycles=0,
    use_friction=True,
    mu=0.1,
    **_COMMON_PARAMS,
)
t_friction = time.perf_counter() - t0
print(f"\n結果: converged={result_friction.phase1_converged}, 時間={t_friction:.2f}s")
print(f"  活性接触ペア: {result_friction.n_active_contacts}")

# ==================================================================
# 4. 3Dチューブレンダリングスナップショット（物理的妥当性確認）
# ==================================================================
print()
print("=" * 70)
print("  3Dレンダリングスナップショット生成")
print("=" * 70)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh
    from xkep_cae.output.render_beam_3d import render_twisted_wire_3d

    output_dir = Path("docs/verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = make_twisted_wire_mesh(
        7, 2.0, 40.0, length=0.0, n_elems_per_strand=8, n_pitches=0.5, gap=0.15
    )

    def plot_3d_snapshots(mesh_obj, snapshots, labels, filename, title=""):
        """メッシュの3Dチューブレンダリングスナップショットを生成."""
        n_snaps = min(len(snapshots), 8)
        n_cols = min(n_snaps, 4)
        n_rows = (n_snaps + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows), dpi=120)

        for idx in range(n_snaps):
            u_snap = snapshots[idx]
            coords = mesh_obj.node_coords.copy()
            for i in range(mesh_obj.n_nodes):
                coords[i, 0] += u_snap[6 * i]
                coords[i, 1] += u_snap[6 * i + 1]
                coords[i, 2] += u_snap[6 * i + 2]

            fig_tmp, _ax_tmp = render_twisted_wire_3d(
                mesh_obj,
                node_coords=coords,
                elev=25.0,
                azim=-60.0,
                title=labels[idx] if idx < len(labels) else f"Step {idx}",
                figsize=(8, 7),
                dpi=80,
                n_circ=10,
            )
            plt.close(fig_tmp)

            # マルチパネルに再描画
            from xkep_cae.output.render_beam_3d import (
                _STRAND_COLORS,
                _make_tube_mesh,
                _set_equal_aspect_3d,
            )

            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
            coords_mm = coords * 1000.0
            r_mm = mesh_obj.wire_radius * 1000.0
            for sid in range(mesh_obj.n_strands):
                color = _STRAND_COLORS[sid % len(_STRAND_COLORS)]
                ns, ne = mesh_obj.strand_node_ranges[sid]
                for eidx in range(len(mesh_obj.connectivity)):
                    n0, n1 = mesh_obj.connectivity[eidx]
                    if ns <= n0 < ne and ns <= n1 < ne:
                        X, Y, Z = _make_tube_mesh(coords_mm[n0], coords_mm[n1], r_mm, 8)
                        ax.plot_surface(
                            X,
                            Y,
                            Z,
                            color=color,
                            alpha=0.85,
                            shade=True,
                            linewidth=0,
                            antialiased=True,
                        )
            ax.view_init(elev=25.0, azim=-60.0)
            snap_title = labels[idx] if idx < len(labels) else f"Step {idx}"
            ax.set_title(snap_title, fontsize=9)
            ax.set_xlabel("X [mm]", fontsize=7)
            ax.set_ylabel("Y [mm]", fontsize=7)
            ax.set_zlabel("Z [mm]", fontsize=7)
            ax.tick_params(labelsize=5)
            _set_equal_aspect_3d(ax, coords_mm)

        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  保存: {filename}")

    if result_90.displacement_snapshots:
        plot_3d_snapshots(
            mesh,
            result_90.displacement_snapshots,
            result_90.snapshot_labels,
            str(output_dir / "7strand_bending_oscillation_3d.png"),
            title="7-strand 90deg bending + oscillation (3D)",
        )
    else:
        print("  スナップショットなし（GIF出力が無効）")

except ImportError:
    print("  matplotlib が利用不可能。3Dレンダリングスキップ。")

# ==================================================================
# 5. サマリー
# ==================================================================
print()
print("=" * 70)
print("  検証サマリー")
print("=" * 70)
print(
    f"  45度曲げ（接触あり）:     {'PASS' if result_45.phase1_converged else 'FAIL'} ({t_45:.1f}s)"
)
print(f"  90度曲げ:                 {'PASS' if result_90.phase1_converged else 'FAIL'}")
print(
    f"  90度揺動:                 {'PASS' if result_90.phase2_converged else 'FAIL'} ({t_90:.1f}s)"
)
print(
    f"  摩擦あり45度（μ=0.1）:   {'PASS' if result_friction.phase1_converged else 'FAIL'} ({t_friction:.1f}s)"
)
print(f"  ログ:                     {log_path}")

# 45度曲げとPhase1は必須、Phase2とfrictionはベストエフォート
must_pass = result_45.phase1_converged and result_90.phase1_converged
nice_to_have = result_90.phase2_converged and result_friction.phase1_converged
all_pass = must_pass and nice_to_have
print(f"\n  必須項目: {'PASS' if must_pass else 'FAIL'}")
print(f"  推奨項目: {'PASS' if nice_to_have else 'FAIL'}")
print(f"  総合:     {'ALL PASS' if all_pass else 'PARTIAL' if must_pass else 'FAIL'}")

sys.stdout = tee.terminal
tee.close()
print(f"\n検証完了。ログ: {log_path}")

if not must_pass:
    sys.exit(1)
