"""three_point_bend_jig huber_delta_h スイープ（剛体-梁接触での最適値探索）.

status-262 TODO #2: 剛体-梁接触で delta_h の効果を検証。
条件: E=25MPa, n_periods=30, jig_push=30mm, wire_diameter=17mm。
ベースライン(status-234): frac=1.0, incr=1592, cutback=2477。

注意: status-263 で E=25 n_periods=30 の回帰バグが発見されている（frac=0.0003）。
回帰原因コミット: cc6f465。回帰修正後に本スクリプトを再実行すること。

Usage:
    python contracts/bench_three_point_bend_delta_h.py 2>&1 | tee /tmp/log-tpb-delta-h.log

[← README](../README.md)
"""

from __future__ import annotations

import sys
import time
import warnings

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)

warnings.filterwarnings("ignore", category=UserWarning)


def _run_case(label: str, cfg: DynamicThreePointBendContactJigConfig) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {label} (huber_delta_h={cfg.huber_delta_h})")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    proc = DynamicThreePointBendContactJigProcess()
    t0 = time.perf_counter()
    result = proc.process(cfg)
    elapsed = time.perf_counter() - t0

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0
    nr_attempts = [d.n_attempts for d in sr.increment_diagnostics]
    avg_nr = sum(nr_attempts) / len(nr_attempts) if nr_attempts else 0
    max_nr = max(nr_attempts) if nr_attempts else 0

    info = {
        "label": label,
        "huber_delta_h": cfg.huber_delta_h,
        "frac": frac,
        "converged": sr.converged,
        "n_increments": sr.n_increments,
        "n_cutbacks": sr.n_cutbacks,
        "total_attempts": sr.total_attempts,
        "avg_nr": avg_nr,
        "max_nr": max_nr,
        "elapsed": elapsed,
    }

    print(
        f"  frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}, "
        f"NR_avg={avg_nr:.1f}, NR_max={max_nr}, time={elapsed:.1f}s"
    )
    sys.stdout.flush()

    return info


def main() -> None:
    # DynamicThreePointBendContactJig: 剛体円柱ジグ + 梁ワイヤ接触
    # E=25MPa, n_periods=30（status-234 ベースライン条件）
    base_cfg = dict(
        E=25.0,
        n_periods=30.0,
        jig_push=30.0,
        max_increments=10000,
        tol_force=1e-6,
        tol_disp=1e-8,
        max_nr_attempts=30,
    )

    # delta_h スイープ: ベースライン + 代表的な値
    # wire_radius=8.5mm, auto δ=5000/8.5≈588
    cases = [
        (
            "baseline (auto δ)",
            DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.0),
        ),
        (
            "delta_h=0.025",
            DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.025),
        ),
        (
            "delta_h=0.100",
            DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.100),
        ),
        (
            "delta_h=0.425",
            DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.425),
        ),
    ]

    results = []
    for label, cfg in cases:
        try:
            info = _run_case(label, cfg)
            results.append(info)
        except Exception as e:
            print(f"\n!!! {label} failed: {e}")
            results.append({"label": label, "error": str(e)})

    print("\n\n" + "=" * 80)
    print("  huber_delta_h スイープ結果（three_point_bend_jig, 剛体円柱-梁接触）")
    print("  条件: E=25MPa, n_periods=30, jig_push=30mm, wire_diameter=17mm")
    print("  ベースライン (status-234): frac=1.0 incr=1592 cutback=2477")
    print("=" * 80)
    header = (
        f"{'label':25s} {'delta_h':>10s} {'frac':>8s} {'incr':>6s} "
        f"{'cutback':>8s} {'NR_avg':>8s} {'NR_max':>8s} {'time':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['label']:25s}  ERROR: {r['error']}")
            continue
        print(
            f"{r['label']:25s} {r['huber_delta_h']:10.3f} {r['frac']:8.4f} "
            f"{r['n_increments']:6d} {r['n_cutbacks']:8d} "
            f"{r['avg_nr']:8.1f} {r['max_nr']:8d} {r['elapsed']:8.1f}s"
        )


if __name__ == "__main__":
    main()
