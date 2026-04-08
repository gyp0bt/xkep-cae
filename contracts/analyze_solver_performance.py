"""7本撚線ソルバー性能分析スクリプト.

status-301: カットバック率分析 + プロセス別プロファイリング。
被膜なし/被膜付きの2ケースを実行し、改善余地を定量評価する。

Usage:
    python contracts/analyze_solver_performance.py 2>&1 | tee /tmp/log-perf-$(date +%s).log

[← README](../README.md)
"""

import math
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.core.base import ProcessMetaclass  # noqa: E402
from xkep_cae.numerical_tests.strand_bending_oscillation import (  # noqa: E402
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _print_separator(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _analyze_increment_diagnostics(result) -> None:
    """インクリメント診断データからカットバック内訳を分析."""
    sr = result.solver_result
    diags = sr.increment_diagnostics
    if not diags:
        print("  increment_diagnostics なし（詳細分析不可）")
        return

    # NR反復数の統計
    attempts = [d.n_attempts for d in diags]
    if attempts:
        arr = np.array(attempts)
        print("  NR反復数統計:")
        print(f"    平均     = {arr.mean():.1f}")
        print(f"    中央値   = {np.median(arr):.1f}")
        print(f"    最大     = {arr.max()}")
        print(f"    95%tile  = {np.percentile(arr, 95):.0f}")
        print(f"    合計     = {arr.sum()}")

    # 収束率の統計
    rates = [d.convergence_rate for d in diags if d.converged]
    if rates:
        arr = np.array(rates)
        print("  収束率統計（収束したステップのみ）:")
        print(f"    平均     = {arr.mean():.4f}")
        print(f"    中央値   = {np.median(arr):.4f}")
        print(f"    >0.9     = {(arr > 0.9).sum()} / {len(arr)}")

    # 不収束ステップの分析
    failed = [d for d in diags if not d.converged]
    if failed:
        fracs = [d.load_frac for d in failed]
        print(f"  不収束ステップ: {len(failed)} 回")
        if fracs:
            print(f"    frac範囲  = [{min(fracs):.4f}, {max(fracs):.4f}]")
            # frac区間別の不収束数
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            counts, _ = np.histogram(fracs, bins=bins)
            print("    frac区間別カットバック分布:")
            for i, c in enumerate(counts):
                if c > 0:
                    print(f"      [{bins[i]:.1f}, {bins[i + 1]:.1f}): {c}")

    # 接触ペア数の推移
    active_pairs = [d.n_active_pairs for d in diags if d.converged]
    if active_pairs:
        arr = np.array(active_pairs)
        print("  接触ペア数（収束ステップ）:")
        print(f"    平均     = {arr.mean():.1f}")
        print(f"    最大     = {arr.max()}")
        print(f"    最小     = {arr.min()}")


def _print_profile_report() -> None:
    """ProcessMetaclass プロファイルレポートを出力."""
    _print_separator("プロセス別プロファイル")
    data = ProcessMetaclass._profile_data
    if not data:
        print("  プロファイルデータなし")
        return

    # 合計時間でソート（降順）
    items = []
    for name, times in data.items():
        n = len(times)
        total = sum(times)
        avg = total / n if n > 0 else 0
        items.append((name, n, total, avg))
    items.sort(key=lambda x: x[2], reverse=True)

    grand_total = sum(x[2] for x in items)
    print(f"  {'プロセス名':<45s} {'呼出':>6s} {'合計[s]':>10s} {'平均[s]':>10s} {'割合':>6s}")
    print(f"  {'-' * 45} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 6}")
    for name, n, total, avg in items[:20]:  # 上位20件
        pct = 100.0 * total / grand_total if grand_total > 0 else 0
        print(f"  {name:<45s} {n:>6d} {total:>10.3f} {avg:>10.5f} {pct:>5.1f}%")
    print(f"  {'合計':<45s} {'':>6s} {grand_total:>10.3f}")


def run_case(
    label: str,
    coating_stiffness: float = 0.0,
    coating_damping: float = 0.0,
    coating_mu: float = 0.0,
    coating_thickness: float = 0.0,
) -> tuple[float, int, int, float]:
    """1ケースの実行 + 分析."""
    kappa_90 = math.pi / 2.0 / 100.0

    _print_separator(f"ケース: {label}")
    cfg = StrandBendingOscillationConfig(
        n_strands=7,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=kappa_90,
        n_cycles=1,
        n_increments_per_cycle=40,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=True,
        penalty_exponent=1.5,
        n_oscillation_cycles=2,
        oscillation_amplitude=48.0,
        # 被膜パラメータ
        coating_stiffness=coating_stiffness,
        coating_damping=coating_damping,
        coating_mu=coating_mu,
        coating_thickness=coating_thickness,
    )

    print(f"  penalty_exponent   = {cfg.penalty_exponent}")
    print(f"  coating_stiffness  = {cfg.coating_stiffness}")
    print(f"  coating_damping    = {cfg.coating_damping}")
    print(f"  coating_mu         = {cfg.coating_mu}")
    print()

    ProcessMetaclass.reset_profile()
    t0 = time.time()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.time() - t0

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0

    _print_separator(f"結果サマリ: {label}")
    print(f"  frac          = {frac:.4f}")
    print(f"  n_increments  = {sr.n_increments}")
    print(f"  n_cutbacks    = {sr.n_cutbacks}")
    print(f"  total_attempts= {sr.total_attempts}")
    print(f"  elapsed       = {elapsed:.1f} sec")
    if sr.n_increments > 0:
        cutback_rate = sr.n_cutbacks / (sr.n_increments + sr.n_cutbacks)
        print(f"  cutback率     = {cutback_rate:.1%}")
        print(f"  NR/incr平均   = {sr.total_attempts / sr.n_increments:.1f}")

    _analyze_increment_diagnostics(result)
    _print_profile_report()

    return frac, sr.n_increments, sr.n_cutbacks, elapsed


def main() -> None:
    """メイン: 被膜なし→被膜付きの順で実行."""
    _print_separator("7本撚線ソルバー性能分析 (status-301)")
    print("目的: カットバック率分析 + プロセス別プロファイリング")
    print("ベースライン (status-299): incr=1900, cutback=72, 1504s")

    # ── ケース1: 被膜なし（ベースライン再現） ──
    frac1, incr1, cb1, t1 = run_case("被膜なし（ベースライン再現）")

    # ── ケース2: 被膜付き（physics test準拠） ──
    # coating_thickness=0.05mm (wire_radius=0.5mmの10%)
    # k=1e6 N/mm, c=1e4 N·s/mm, μ=0.3 (physics test値)
    frac2, incr2, cb2, t2 = run_case(
        "被膜付き（k=1e6, c=1e4, μ=0.3, t=0.05mm）",
        coating_stiffness=1e6,
        coating_damping=1e4,
        coating_mu=0.3,
        coating_thickness=0.05,
    )

    # ── 比較サマリ ──
    _print_separator("比較サマリ")
    print(
        f"  {'ケース':<30s} {'frac':>6s} {'incr':>6s} {'cutback':>7s} {'elapsed':>8s} {'cb率':>6s}"
    )
    print(f"  {'-' * 30} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 8} {'-' * 6}")
    for label, frac, incr, cb, t in [
        ("被膜なし", frac1, incr1, cb1, t1),
        ("被膜付き", frac2, incr2, cb2, t2),
    ]:
        rate = cb / (incr + cb) if (incr + cb) > 0 else 0
        print(f"  {label:<30s} {frac:>6.4f} {incr:>6d} {cb:>7d} {t:>7.1f}s {rate:>5.1%}")

    print()
    print("分析完了")


if __name__ == "__main__":
    main()
