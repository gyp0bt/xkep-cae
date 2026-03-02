"""撚線規模別 曲げ揺動計算時間計測スクリプト.

撚線本数を段階的に上げながら曲げ揺動計算を実施し、
各工程の計算時間とトータルの計算時間を取得して文書に残す。

使い方:
  python scripts/measure_wire_calculation_time.py

出力:
  docs/verification/wire_calculation_timing.md

[← README](../README.md)
"""

from __future__ import annotations

import datetime
import platform
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xkep_cae.numerical_tests.wire_bending_benchmark import (
    BendingOscillationResult,
    print_benchmark_report,
    run_bending_oscillation,
)

# ====================================================================
# 計測パラメータ
# ====================================================================

# 計測する素線本数リスト
STRAND_COUNTS = [7, 19, 37, 61, 91]

# 全規模で統一するパラメータ
COMMON_PARAMS = {
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
}

# 出力先
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "verification"


def run_measurement() -> list[BendingOscillationResult]:
    """撚線規模別の計測を実行."""
    results: list[BendingOscillationResult] = []

    for n_strands in STRAND_COUNTS:
        print(f"\n{'#' * 80}")
        print(f"# 計測開始: {n_strands}本撚線")
        print(f"{'#' * 80}")

        try:
            result = run_bending_oscillation(
                n_strands=n_strands,
                **COMMON_PARAMS,
            )
            results.append(result)

            # 途中経過をコンソールに出力
            report = print_benchmark_report(result)
            print(report)

        except Exception as e:
            print(f"\n  [ERROR] {n_strands}本撚線で失敗: {e}")
            # 失敗した場合はスキップして次に進む
            import traceback

            traceback.print_exc()
            continue

    return results


def generate_report(results: list[BendingOscillationResult]) -> str:
    """計測結果を Markdown レポートに整形."""
    lines: list[str] = []

    lines.append("# 撚線規模別 曲げ揺動計算時間計測レポート")
    lines.append("")
    lines.append(f"計測日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"プラットフォーム: {platform.platform()}")
    lines.append(f"Python: {platform.python_version()}")
    lines.append("")
    lines.append("[← README](../../README.md)")
    lines.append("")

    # ------------------------------------------------------------------
    # 計測条件
    # ------------------------------------------------------------------
    lines.append("## 計測条件")
    lines.append("")
    lines.append("全規模で統一パラメータを使用。")
    lines.append("")
    lines.append("| パラメータ | 値 |")
    lines.append("|---|---|")
    lines.append(f"| 素線あたり要素数 | {COMMON_PARAMS['n_elems_per_strand']} |")
    lines.append(f"| ピッチ数 | {COMMON_PARAMS['n_pitches']} |")
    lines.append(f"| 目標曲げ角度 | {COMMON_PARAMS['bend_angle_deg']}° |")
    lines.append(f"| 曲げステップ数 | {COMMON_PARAMS['n_bending_steps']} |")
    lines.append(f"| 揺動振幅 | ±{COMMON_PARAMS['oscillation_amplitude_mm']} mm |")
    lines.append(f"| 揺動サイクル数 | {COMMON_PARAMS['n_cycles']} |")
    lines.append(f"| 1/4周期ステップ数 | {COMMON_PARAMS['n_steps_per_quarter']} |")
    lines.append(f"| NR最大反復数 | {COMMON_PARAMS['max_iter']} |")
    lines.append(f"| Outer loop 最大数 | {COMMON_PARAMS['n_outer_max']} |")
    lines.append(f"| 力残差収束判定 | {COMMON_PARAMS['tol_force']} |")
    lines.append(f"| 摩擦 | {'あり' if COMMON_PARAMS['use_friction'] else 'なし'} |")
    lines.append("| ペナルティ剛性 | 自動推定 |")
    lines.append("")

    # ------------------------------------------------------------------
    # サマリ表
    # ------------------------------------------------------------------
    lines.append("## サマリ")
    lines.append("")
    lines.append(
        "| 素線数 | 要素数 | 節点数 | 自由度数 | P1収束 | P2収束 | "
        "活性接触 | 最大貫入比 | 計算時間(s) |"
    )
    lines.append("|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.n_strands} | {r.n_elems} | {r.n_nodes} | {r.ndof} | "
            f"{'○' if r.phase1_converged else '×'} | "
            f"{'○' if r.phase2_converged else '×'} | "
            f"{r.n_active_contacts} | {r.max_penetration_ratio:.6f} | "
            f"{r.total_time_s:.2f} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # 工程別タイミング詳細
    # ------------------------------------------------------------------
    lines.append("## 工程別計算時間")
    lines.append("")

    # 全結果から工程名を収集
    all_phases: list[str] = []
    for r in results:
        if r.timing is not None:
            for phase in r.timing.phase_totals():
                if phase not in all_phases:
                    all_phases.append(phase)

    # 工程別の時間を大きい順にソート（最初の結果基準）
    if results and results[0].timing is not None:
        ref_totals = results[0].timing.phase_totals()
        all_phases.sort(key=lambda p: -ref_totals.get(p, 0.0))

    # ヘッダ行
    header_cols = ["工程"]
    for r in results:
        header_cols.append(f"{r.n_strands}本(s)")
        header_cols.append(f"{r.n_strands}本(%)")
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|---" + "|---:" * (len(header_cols) - 1) + "|")

    # 各工程の行
    for phase in all_phases:
        cols = [phase]
        for r in results:
            if r.timing is not None:
                totals = r.timing.phase_totals()
                total = r.timing.total_time()
                t = totals.get(phase, 0.0)
                pct = (t / total * 100) if total > 0 else 0.0
                cols.append(f"{t:.3f}")
                cols.append(f"{pct:.1f}")
            else:
                cols.append("-")
                cols.append("-")
        lines.append("| " + " | ".join(cols) + " |")

    # TOTAL行
    cols = ["**TOTAL**"]
    for r in results:
        if r.timing is not None:
            total = r.timing.total_time()
            cols.append(f"**{total:.3f}**")
            cols.append("**100.0**")
        else:
            cols.append("-")
            cols.append("-")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("")

    # ------------------------------------------------------------------
    # スケーリング分析
    # ------------------------------------------------------------------
    if len(results) >= 2:
        lines.append("## スケーリング分析")
        lines.append("")
        lines.append("| 素線数 | 自由度数 | 計算時間(s) | 対7本比(自由度) | 対7本比(時間) | 効率 |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        base = results[0]
        for r in results:
            dof_ratio = r.ndof / base.ndof if base.ndof > 0 else 1.0
            time_ratio = r.total_time_s / base.total_time_s if base.total_time_s > 0 else 1.0
            efficiency = dof_ratio / time_ratio if time_ratio > 0 else 0.0
            lines.append(
                f"| {r.n_strands} | {r.ndof} | {r.total_time_s:.2f} | "
                f"{dof_ratio:.2f} | {time_ratio:.2f} | {efficiency:.2f} |"
            )
        lines.append("")
        lines.append(
            "> 効率 = 自由度比 / 計算時間比。1.0 なら線形スケーリング、"
            "1.0 未満なら超線形（計算時間が自由度以上に増加）。"
        )
        lines.append("")

    # ------------------------------------------------------------------
    # 各規模の詳細レポート
    # ------------------------------------------------------------------
    lines.append("## 各規模の詳細レポート")
    lines.append("")
    for r in results:
        lines.append(f"### {r.n_strands}本撚線")
        lines.append("")
        lines.append("```")
        lines.append(print_benchmark_report(r))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main():
    """メイン処理."""
    print("=" * 80)
    print("  撚線規模別 曲げ揺動計算時間計測")
    print(f"  計測対象: {STRAND_COUNTS}本")
    print(f"  開始時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    t_start = time.perf_counter()
    results = run_measurement()
    t_elapsed = time.perf_counter() - t_start

    if not results:
        print("\n[ERROR] 全ての計測が失敗しました。")
        sys.exit(1)

    print(f"\n全計測完了: {t_elapsed:.2f} s")

    # レポート生成
    report = generate_report(results)

    # ファイル出力
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "wire_calculation_timing.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"\nレポート出力: {output_path}")

    # コンソールにもサマリを表示
    print("\n" + "=" * 80)
    print("  サマリ")
    print("=" * 80)
    for r in results:
        timing_total = r.timing.total_time() if r.timing else 0.0
        print(
            f"  {r.n_strands:>3}本: {r.n_elems:>5}要素, "
            f"{r.ndof:>6}DOF, "
            f"P1={'○' if r.phase1_converged else '×'}, "
            f"P2={'○' if r.phase2_converged else '×'}, "
            f"計測時間={timing_total:.2f}s, "
            f"壁時計={r.total_time_s:.2f}s"
        )


if __name__ == "__main__":
    main()
