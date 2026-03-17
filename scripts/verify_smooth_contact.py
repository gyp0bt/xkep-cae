"""スムースペナルティ接触の収束検証スクリプト（Phase C7）.

NCP鞍点系 vs スムースペナルティ+Uzawa の比較検証。
Active setチャタリングが解消され、摩擦ありでもNewtonが安定収束することを確認する。

段階的検証:
  Case 1: 7本 45度曲げ（摩擦なし）— 基本動作確認
  Case 2: 7本 45度曲げ（μ=0.15）— 摩擦あり収束比較
  Case 3: 7本 90度曲げ（μ=0.25）— 高摩擦チャタリング領域

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
    """全ケース共通パラメータ."""
    return dict(
        n_elems_per_strand=16,
        n_pitches=0.5,
        max_iter=50,
        tol_force=1e-4,
        show_progress=True,
        use_ncp=True,
        use_mortar=False,
        adaptive_timestepping=True,
        exclude_same_layer=True,
        midpoint_prescreening=True,
        use_line_search=True,
        g_on=0.0005,
        g_off=0.001,
        use_updated_lagrangian=True,
    )


def run_case(
    label: str,
    n_strands: int,
    bend_angle_deg: float,
    mu: float,
    contact_mode: str,
) -> BendingOscillationResult:
    """1ケースを実行."""
    _header(f"{label}: {n_strands}本 {bend_angle_deg}°曲げ μ={mu} [{contact_mode}]")
    t0 = time.perf_counter()
    result = _run_bending_oscillation(
        n_strands=n_strands,
        bend_angle_deg=bend_angle_deg,
        oscillation_amplitude_mm=0.0,
        n_cycles=0,
        n_steps_per_quarter=1,
        use_friction=mu > 0.0,
        mu=mu,
        contact_mode=contact_mode,
        **_common_params(),
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  結果: converged={result.phase1_converged}")
    print(f"  計算時間: {elapsed:.1f}s")
    print(f"  active contacts: {result.n_active_contacts}")
    return result


def main() -> None:
    print("\n  スムースペナルティ接触 収束検証")
    print(f"  実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results: dict[str, bool] = {}

    # --- Case 1: 基本動作（摩擦なし）---
    r1_smooth = run_case("Case 1a", 7, 45.0, 0.0, "smooth_penalty")
    results["1a_smooth_nofric"] = r1_smooth.phase1_converged

    r1_ncp = run_case("Case 1b", 7, 45.0, 0.0, "ncp")
    results["1b_ncp_nofric"] = r1_ncp.phase1_converged

    # --- Case 2: 摩擦あり（μ=0.15）---
    r2_smooth = run_case("Case 2a", 7, 45.0, 0.15, "smooth_penalty")
    results["2a_smooth_mu015"] = r2_smooth.phase1_converged

    r2_ncp = run_case("Case 2b", 7, 45.0, 0.15, "ncp")
    results["2b_ncp_mu015"] = r2_ncp.phase1_converged

    # --- Case 3: 高摩擦（μ=0.25、90度）---
    r3_smooth = run_case("Case 3a", 7, 90.0, 0.25, "smooth_penalty")
    results["3a_smooth_mu025"] = r3_smooth.phase1_converged

    r3_ncp = run_case("Case 3b", 7, 90.0, 0.25, "ncp")
    results["3b_ncp_mu025"] = r3_ncp.phase1_converged

    # --- 結果サマリー ---
    print(f"\n{'=' * 70}")
    print("  収束結果サマリー")
    print(f"{'=' * 70}")
    for key, conv in results.items():
        status = "PASS" if conv else "FAIL"
        print(f"  {key:30s} : {status}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  合計: {n_pass}/{n_total} passed")

    # smooth_penalty が NCP より改善されているか確認
    smooth_cases = [v for k, v in results.items() if "smooth" in k]
    ncp_cases = [v for k, v in results.items() if "ncp" in k]
    print(f"  Smooth: {sum(smooth_cases)}/{len(smooth_cases)} passed")
    print(f"  NCP:    {sum(ncp_cases)}/{len(ncp_cases)} passed")

    if not all(smooth_cases):
        print("\n  WARNING: スムースペナルティで収束失敗あり")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
