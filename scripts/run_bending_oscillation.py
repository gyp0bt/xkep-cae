#!/usr/bin/env python3
"""撚線曲げ揺動 — 分析用実行スクリプト（7〜91本）.

各素線数で曲げ揺動を実行し、以下のフックで結果を出力する:
  - VTK (.vtu + .pvd): ParaView で変形・変位を確認
  - GIF: 曲げ揺動アニメーション（yz / xz ビュー）
  - 接触グラフ GIF: 接触ペアの活性推移

使い方:
  python scripts/run_bending_oscillation.py [--strands 7,19] [--outdir results/bending]

出力ディレクトリ構成:
  {outdir}/
    {N}strand/
      bending_oscillation_{N}strand_yz.gif
      bending_oscillation_{N}strand_xz.gif
      contact_graph_{N}strand.gif
      vtk/
        result.pvd
        result_*.vtu
      summary.txt

[← README](../README.md)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from xkep_cae.contact.graph import save_contact_graph_gif
from xkep_cae.numerical_tests.wire_bending_benchmark import (
    BendingOscillationResult,
    export_bending_oscillation_gif,
    run_bending_oscillation,
)
from xkep_cae.output.database import OutputDatabase
from xkep_cae.output.export_vtk import VTK_LINE, export_vtk
from xkep_cae.output.step import Frame, Step, StepResult

# ====================================================================
# 共通パラメータ
# ====================================================================

# 軽量パラメータ（分析確認用 — 本番では値を上げる）
ANALYSIS_PARAMS = {
    "n_elems_per_strand": 4,
    "n_pitches": 0.5,
    "bend_angle_deg": 45.0,
    "n_bending_steps": 5,
    "oscillation_amplitude_mm": 2.0,
    "n_cycles": 1,
    "n_steps_per_quarter": 2,
    "max_iter": 20,
    "n_outer_max": 3,
    "tol_force": 1e-4,
    "auto_kpen": True,
    "use_friction": False,
    "show_progress": True,
    "gif_snapshot_interval": 1,
}

STRAND_COUNTS_DEFAULT = [7, 19, 37, 61, 91]


# ====================================================================
# エクスポートフック
# ====================================================================


def hook_export_vtk(
    result: BendingOscillationResult,
    mesh,
    out_dir: Path,
) -> str | None:
    """VTK (.vtu + .pvd) エクスポート."""
    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(parents=True, exist_ok=True)

    # OutputDatabase を組み立て
    db = OutputDatabase(
        node_coords=mesh.node_coords,
        connectivity=[(VTK_LINE, mesh.connectivity)],
        ndof_per_node=6,
    )

    # 全スナップショットを Frame にまとめて1つの Step に入れる
    frames = []
    for i, (u, _label) in enumerate(
        zip(result.displacement_snapshots, result.snapshot_labels, strict=True)
    ):
        frames.append(
            Frame(
                frame_index=i,
                time=float(i),
                displacement=u,
            )
        )

    n_frames = len(frames)
    if n_frames == 0:
        print("    [VTK] スナップショットなし — スキップ")
        return None

    step = Step(name="bending_oscillation", total_time=float(n_frames), dt=1.0)
    sr = StepResult(step=step, step_index=0, frames=frames)
    db.step_results.append(sr)

    pvd_path = export_vtk(db, vtk_dir, prefix="result")
    print(f"    [VTK] {pvd_path} ({n_frames} frames)")
    return pvd_path


def hook_export_gif(
    result: BendingOscillationResult,
    mesh,
    out_dir: Path,
) -> list[Path]:
    """変位 GIF エクスポート."""
    if not result.displacement_snapshots:
        print("    [GIF] スナップショットなし — スキップ")
        return []

    try:
        gif_paths = export_bending_oscillation_gif(
            mesh,
            result.displacement_snapshots,
            result.snapshot_labels,
            out_dir,
            prefix=f"bending_oscillation_{result.n_strands}strand",
            views=["yz", "xz"],
            figsize=(10.0, 8.0),
            dpi=80,
            duration=300,
        )
        for p in gif_paths:
            print(f"    [GIF] {p}")
        return gif_paths
    except ImportError:
        print("    [GIF] matplotlib/Pillow 未インストール — スキップ")
        return []


def hook_export_contact_graph(
    result: BendingOscillationResult,
    mesh,
    out_dir: Path,
) -> Path | None:
    """接触グラフ GIF エクスポート."""
    graph_history = None

    # Phase 1 の結果から取得
    if hasattr(result.phase1_result, "graph_history"):
        graph_history = result.phase1_result.graph_history

    # Phase 2 の結果を追加
    for r2 in result.phase2_results:
        if hasattr(r2, "graph_history") and r2.graph_history is not None:
            if graph_history is None:
                graph_history = r2.graph_history
            else:
                for snap in r2.graph_history.snapshots:
                    graph_history.add_snapshot(snap)

    if graph_history is None or graph_history.n_steps == 0:
        print("    [接触グラフ] データなし — スキップ")
        return None

    gif_path = out_dir / f"contact_graph_{result.n_strands}strand.gif"
    try:
        save_contact_graph_gif(
            graph_history,
            str(gif_path),
            fps=2,
            figsize=(8, 6),
            dpi=80,
        )
        print(f"    [接触グラフ] {gif_path} ({graph_history.n_steps} snapshots)")
        return gif_path
    except ImportError:
        print("    [接触グラフ] matplotlib/Pillow 未インストール — スキップ")
        return None
    except Exception as e:
        print(f"    [接触グラフ] エラー: {e}")
        return None


def hook_export_summary(
    result: BendingOscillationResult,
    out_dir: Path,
) -> Path:
    """テキストサマリー出力."""
    summary_path = out_dir / "summary.txt"
    lines = [
        f"撚線曲げ揺動結果サマリー: {result.n_strands}本",
        f"{'=' * 50}",
        f"要素数:       {result.n_elems}",
        f"節点数:       {result.n_nodes}",
        f"自由度数:     {result.ndof}",
        f"モデル長さ:   {result.mesh_length * 1000:.1f} mm",
        "",
        "Phase 1（曲げ）:",
        f"  収束:       {result.phase1_converged}",
        f"  NR反復:     {result.phase1_result.total_newton_iterations}",
        "",
        "Phase 2（揺動）:",
        f"  収束:       {result.phase2_converged}",
        f"  ステップ数: {len(result.phase2_results)}",
        "",
        f"先端変位:     ({result.tip_displacement_final[0] * 1000:.3f}, "
        f"{result.tip_displacement_final[1] * 1000:.3f}, "
        f"{result.tip_displacement_final[2] * 1000:.3f}) mm",
        f"最大貫入比:   {result.max_penetration_ratio:.6f}",
        f"活性接触ペア: {result.n_active_contacts}",
        f"計算時間:     {result.total_time_s:.1f} s",
        "",
        f"スナップショット数: {len(result.displacement_snapshots)}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    [Summary] {summary_path}")
    return summary_path


# ====================================================================
# メイン実行
# ====================================================================


def run_single(n_strands: int, out_root: Path) -> BendingOscillationResult | None:
    """単一素線数の曲げ揺動を実行し全フックを発火."""
    out_dir = out_root / f"{n_strands}strand"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# {n_strands}本撚線 — 曲げ揺動実行")
    print(f"{'#' * 70}")

    t0 = time.perf_counter()
    try:
        result = run_bending_oscillation(
            n_strands=n_strands,
            gif_output_dir=None,  # GIF は独自フックで出力
            **ANALYSIS_PARAMS,
        )
    except Exception as e:
        print(f"\n  [ERROR] {n_strands}本: {e}")
        import traceback

        traceback.print_exc()
        return None

    elapsed = time.perf_counter() - t0
    print(f"\n  計算完了: {elapsed:.1f}s")
    print(f"  Phase1 収束={result.phase1_converged}, Phase2 収束={result.phase2_converged}")

    # --- メッシュを再構築（run_bending_oscillation が内部で作るため）---
    from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

    mesh = make_twisted_wire_mesh(
        n_strands,
        0.002,
        0.040,
        length=0.0,
        n_elems_per_strand=ANALYSIS_PARAMS["n_elems_per_strand"],
        n_pitches=ANALYSIS_PARAMS["n_pitches"],
    )

    # --- フック発火 ---
    print("\n  エクスポート:")
    hook_export_summary(result, out_dir)
    hook_export_vtk(result, mesh, out_dir)
    hook_export_gif(result, mesh, out_dir)
    hook_export_contact_graph(result, mesh, out_dir)

    return result


def main():
    parser = argparse.ArgumentParser(description="撚線曲げ揺動 分析用実行スクリプト")
    parser.add_argument(
        "--strands",
        type=str,
        default=",".join(str(s) for s in STRAND_COUNTS_DEFAULT),
        help="素線数リスト（カンマ区切り、例: 7,19,37）",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/bending",
        help="出力ルートディレクトリ",
    )
    args = parser.parse_args()

    strand_counts = [int(s.strip()) for s in args.strands.split(",")]
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("撚線曲げ揺動 分析用実行")
    print(f"  素線数: {strand_counts}")
    print(f"  出力先: {out_root.resolve()}")

    results: list[tuple[int, BendingOscillationResult | None]] = []
    for n in strand_counts:
        r = run_single(n, out_root)
        results.append((n, r))

    # --- 全体サマリー ---
    print(f"\n{'=' * 70}")
    print("  全体サマリー")
    print(f"{'=' * 70}")
    print(f"{'素線数':>8} {'要素':>6} {'DOF':>8} {'Phase1':>8} {'Phase2':>8} {'時間[s]':>10}")
    print(f"{'-' * 60}")
    for n, r in results:
        if r is None:
            print(f"{n:>8} {'FAILED':>6}")
            continue
        p1 = "OK" if r.phase1_converged else "NG"
        p2 = "OK" if r.phase2_converged else "NG"
        print(f"{n:>8} {r.n_elems:>6} {r.ndof:>8} {p1:>8} {p2:>8} {r.total_time_s:>10.1f}")


if __name__ == "__main__":
    main()
