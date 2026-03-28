"""three_point_bend_jig huber_delta_h スイープ（剛体-梁接触での最適値探索）.

status-262 TODO #2: 剛体-梁接触で delta_h=0.025 が有効か検証。
梁-梁接触（strand_bending, wire_radius=0.5mm）では delta_h=0.025 が最速完走だが、
three_point_bend_jig（wire_radius=8.5mm）は k_pen スケールが異なるため有効範囲が変わる。

スケーリング考察:
  strand_bending: wire_radius=0.5mm, delta_h/r=0.05 (最適)
  three_point_bend: wire_radius=8.5mm → delta_h ≈ 0.05*8.5 = 0.425 が期待値
  → delta_h=0.1〜1.0 の広い範囲でスイープ

第1ラウンド結果: delta_h=0.0, 0.005, 0.010 は全て同一結果(frac=0.87, 500incr)
→ 小さい delta_h は効果なし。ワイヤ径スケールに合わせた広範囲スイープに変更。

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
    # n_periods=3 で高速テスト（準静的版は遅いため）
    base_cfg = dict(
        E=25.0,  # 低い E で梁変形大
        n_periods=3.0,
        jig_push=30.0,
        max_increments=200,
        tol_force=1e-6,
        tol_disp=1e-8,
        max_nr_attempts=30,
    )

    # delta_h スイープ: wire_radius=8.5mm スケール
    # strand_bending（r=0.5mm）最適 delta_h/r=0.05 → ここでは delta_h≈0.425 が期待値
    # auto δ=5000/8.5≈588 → delta_h_auto = k_pen/588
    # 小さい値(0.005-0.010)は効果なし（第1ラウンド確認済み）→ 広範囲スイープ
    cases = [
        (
            "delta_h=0 (auto δ)",
            DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.0),
        ),
        ("delta_h=0.025", DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.025)),
        ("delta_h=0.050", DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.050)),
        ("delta_h=0.100", DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.100)),
        ("delta_h=0.200", DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.200)),
        ("delta_h=0.300", DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.300)),
        ("delta_h=0.425", DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.425)),
        ("delta_h=0.500", DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.500)),
        ("delta_h=0.750", DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=0.750)),
        ("delta_h=1.000", DynamicThreePointBendContactJigConfig(**base_cfg, huber_delta_h=1.000)),
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
