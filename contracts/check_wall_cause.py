"""frac=0.60 壁の原因切り分け.

仮説:
1. 摩擦タンジェントが非正定値の原因 → μ=0 でテスト
2. k_pen 不足で接触幾何剛性が支配的 → k_pen 明示指定でテスト

Usage:
    python contracts/check_wall_cause.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)


def _run(label: str, **kwargs) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}\n")

    defaults = dict(
        E=25.0,
        n_periods=30.0,
        jig_push=30.0,
        max_increments=200,
        tol_force=1e-6,
        tol_disp=1e-8,
        max_nr_attempts=30,
    )
    defaults.update(kwargs)
    cfg = DynamicThreePointBendContactJigConfig(**defaults)
    r = DynamicThreePointBendContactJigProcess().process(cfg)

    sr = r.solver_result
    mid = 6 * r.wire_mid_node + 1
    final_frac = sr.load_history[-1] if sr.load_history else 0.0
    fc_last = sr.contact_force_history[-1] if sr.contact_force_history else 0.0

    print(f"\n--- {label} ---")
    print(
        f"  frac={final_frac:.4f} fc={fc_last:.2f}N incr={sr.n_increments} "
        f"cutback={sr.n_cutbacks} time={sr.elapsed_seconds:.1f}s"
    )

    return {
        "label": label,
        "final_frac": final_frac,
        "fc_last": fc_last,
        "n_increments": sr.n_increments,
        "elapsed": sr.elapsed_seconds,
        "n_cutbacks": sr.n_cutbacks,
    }


results = []

# 1. ベースライン
results.append(_run("Baseline (E=25, mu=0.15)"))

# 2. 摩擦なし
results.append(_run("No friction (mu=0)", mu=0.0))

# 3. 高 k_pen (明示指定)
# E=25, I=4098.7, L=100 → 48EI/L³ = 4.918 N/mm
# 現在の自動推定は c0*M*scale ≈ 数百〜数千
# 倍率 10x で試す
results.append(_run("High k_pen (5000)", k_pen=5000.0))

# 4. 低 k_pen (ソフト接触)
results.append(_run("Low k_pen (100)", k_pen=100.0))

# 比較サマリ
print("\n" + "=" * 70)
print("  比較サマリ")
print("=" * 70)
print(f"{'ケース':<30} {'frac':>8} {'fc(N)':>10} {'incr':>5} {'cutback':>7} {'time(s)':>8}")
for r in results:
    print(
        f"{r['label']:<30} {r['final_frac']:8.4f} {r['fc_last']:10.2f} "
        f"{r['n_increments']:5d} {r['n_cutbacks']:7d} {r['elapsed']:8.1f}"
    )
