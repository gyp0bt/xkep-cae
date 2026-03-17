"""摩擦あり曲げ揺動の収束検証スクリプト（smooth penalty + Uzawa）.

smooth penalty + Uzawa 外部ループで摩擦曲げ揺動を実行する。
NCP 鞍点系の摩擦接線剛性符号問題（status-147）を回避するため、
摩擦ありの場合は必ず contact_mode="smooth_penalty" を使用すること。

段階的検証:
  Case 1: 7本 45度曲げ（揺動なし）摩擦あり smooth penalty
  Case 2: 7本 90度曲げ（揺動なし）摩擦あり smooth penalty
  Case 3: 7本 90度曲げ + 揺動1周期 摩擦あり smooth penalty

[← README](../../README.md)
"""

import sys
import time
from datetime import datetime

from xkep_cae.numerical_tests.wire_bending_benchmark import (
    BendingOscillationResult,
    _run_bending_oscillation,
)


def _header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _common_params() -> dict:
    """全ケース共通パラメータ（smooth penalty + Uzawa）.

    注意: 摩擦ありの場合は contact_mode="smooth_penalty" が必須。
    NCP 鞍点系（デフォルト）では摩擦接線剛性の符号問題で発散する。
    詳細は status-147 を参照。
    """
    return dict(
        n_elems_per_strand=16,
        n_pitches=0.5,
        max_iter=50,
        tol_force=1e-4,
        show_progress=True,
        # smooth penalty + Uzawa（NCP鞍点系は摩擦接線剛性符号問題あり）
        contact_mode="smooth_penalty",
        use_ncp=True,
        adaptive_timestepping=True,
        # 接触パラメータ
        exclude_same_layer=True,
        midpoint_prescreening=True,
        use_line_search=False,
        g_on=0.0005,
        g_off=0.001,
        # Updated Lagrangian
        use_updated_lagrangian=True,
        # 摩擦パラメータ
        use_friction=True,
        mu=0.1,
    )


def run_case1() -> BendingOscillationResult:
    """Case 1: 7本 45度曲げ（揺動なし）smooth penalty 摩擦あり."""
    _header("Case 1: 7本 45度曲げ（揺動なし）smooth penalty μ=0.1")
    t0 = time.perf_counter()
    result = _run_bending_oscillation(
        n_strands=7,
        bend_angle_deg=45.0,
        oscillation_amplitude_mm=0.0,
        n_cycles=0,
        n_steps_per_quarter=1,
        **_common_params(),
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  結果: converged={result.phase1_converged}")
    print(f"  計算時間: {elapsed:.1f}s")
    print(f"  active contacts: {result.n_active_contacts}")
    return result


def run_case2() -> BendingOscillationResult:
    """Case 2: 7本 90度曲げ（揺動なし）smooth penalty 摩擦あり."""
    _header("Case 2: 7本 90度曲げ（揺動なし）smooth penalty μ=0.1")
    t0 = time.perf_counter()
    result = _run_bending_oscillation(
        n_strands=7,
        bend_angle_deg=90.0,
        oscillation_amplitude_mm=0.0,
        n_cycles=0,
        n_steps_per_quarter=1,
        **_common_params(),
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  結果: converged={result.phase1_converged}")
    print(f"  計算時間: {elapsed:.1f}s")
    print(f"  active contacts: {result.n_active_contacts}")
    return result


def run_case3() -> BendingOscillationResult:
    """Case 3: 7本 90度曲げ + 揺動1周期 smooth penalty 摩擦あり."""
    _header("Case 3: 7本 90度曲げ + 揺動1周期 smooth penalty μ=0.1")
    t0 = time.perf_counter()
    result = _run_bending_oscillation(
        n_strands=7,
        bend_angle_deg=90.0,
        oscillation_amplitude_mm=2.0,
        n_cycles=1,
        n_steps_per_quarter=3,
        **_common_params(),
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  結果: phase1={result.phase1_converged}, phase2={result.phase2_converged}")
    print(f"  計算時間: {elapsed:.1f}s")
    print(f"  active contacts: {result.n_active_contacts}")
    if result.phase2_results:
        print(f"  揺動ステップ数: {len(result.phase2_results)}")
    return result


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"摩擦あり曲げ揺動 収束検証 — {timestamp}")
    print(f"Python {sys.version}")

    results = {}

    # Case 1: 45度曲げ
    r1 = run_case1()
    results["case1_45deg"] = r1

    # Case 2: 90度曲げ
    r2 = run_case2()
    results["case2_90deg"] = r2

    # Case 3: 90度曲げ + 揺動 (Case 2が収束した場合のみ)
    if r2.phase1_converged:
        r3 = run_case3()
        results["case3_oscillation"] = r3
    else:
        print("\n  Case 2 未収束のため Case 3 スキップ")

    # サマリ
    print(f"\n{'=' * 70}")
    print("  サマリ")
    print(f"{'=' * 70}")
    for name, r in results.items():
        p1 = r.phase1_converged
        p2 = r.phase2_converged if hasattr(r, "phase2_converged") else "N/A"
        print(f"  {name}: phase1={p1}, phase2={p2}, active={r.n_active_contacts}")

    # 全ケース収束判定
    all_converged = all(r.phase1_converged for r in results.values())
    if "case3_oscillation" in results:
        all_converged = all_converged and results["case3_oscillation"].phase2_converged

    if all_converged:
        print("\n  *** 全ケース収束 ***")
    else:
        print("\n  *** 未収束あり ***")

    return 0 if all_converged else 1


if __name__ == "__main__":
    sys.exit(main())
