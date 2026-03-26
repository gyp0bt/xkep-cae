"""K_st + LM + freeze_geometry_in_nr=False の効果検証.

freeze_geometry_in_nr と K_st の相互排他性を検証。
正しい組み合わせ: K_st=True + freeze=False + LM=ON

Usage:
    python contracts/check_kst_lm_unfrozen.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.numerical_tests.three_point_bend_jig import (  # noqa: E402
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)


def _run_config(label, **overrides):
    """指定構成で三点曲げを実行し結果を表示."""
    cfg = DynamicThreePointBendContactJigConfig(
        E=25.0,
        n_periods=3.0,
        jig_push=30.0,
        max_increments=50,
        tol_force=1e-6,
        tol_disp=1e-8,
        max_nr_attempts=30,
        **overrides,
    )
    print(f"\n{'=' * 60}")
    print(f"構成: {label}")
    print(f"  freeze_geometry_in_nr={cfg.freeze_geometry_in_nr}")
    print(f"  consistent_st_tangent={cfg.consistent_st_tangent}")
    print(f"  lm_lambda_init={cfg.lm_lambda_init}")
    print("=" * 60)
    sys.stdout.flush()

    t0 = time.time()
    try:
        r = DynamicThreePointBendContactJigProcess().process(cfg)
        elapsed = time.time() - t0

        sr = r.solver_result
        final_frac = sr.load_history[-1] if sr.load_history else 0.0
        fc_last = sr.contact_force_history[-1] if sr.contact_force_history else 0.0

        n_total = sr.n_increments + sr.n_cutbacks
        cutback_rate = sr.n_cutbacks / n_total if n_total > 0 else 0.0

        print(f"\nRESULT: frac={final_frac:.4f} fc={fc_last:.2f}N")
        print(f"  incr={sr.n_increments} cutback={sr.n_cutbacks} cutback_rate={cutback_rate:.1%}")
        print(f"  wall_time={elapsed:.1f}s")
        return {
            "label": label,
            "frac": final_frac,
            "fc": fc_last,
            "incr": sr.n_increments,
            "cutback": sr.n_cutbacks,
            "cutback_rate": cutback_rate,
            "time": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\nFAILED: {e} (wall_time={elapsed:.1f}s)")
        return {
            "label": label,
            "frac": 0.0,
            "fc": 0.0,
            "incr": 0,
            "cutback": 0,
            "cutback_rate": 1.0,
            "time": elapsed,
        }


if __name__ == "__main__":
    results = []

    # 1. 現行デフォルト: freeze=True, K_st=OFF
    results.append(
        _run_config(
            "freeze=T, K_st=OFF (default)",
            freeze_geometry_in_nr=True,
            consistent_st_tangent=False,
            lm_lambda_init=0.0,
            lm_adaptive=False,
        )
    )

    # 2. freeze=False, K_st=OFF（幾何更新あり、接線不整合）
    results.append(
        _run_config(
            "freeze=F, K_st=OFF",
            freeze_geometry_in_nr=False,
            consistent_st_tangent=False,
            lm_lambda_init=0.0,
            lm_adaptive=False,
        )
    )

    # 3. freeze=False, K_st=ON, LM=ON（正しい組合せ）
    results.append(
        _run_config(
            "freeze=F, K_st=ON, LM=ON",
            freeze_geometry_in_nr=False,
            consistent_st_tangent=True,
            lm_lambda_init=1e-4,
            lm_adaptive=True,
        )
    )

    # サマリ
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    fmt = f"{'Label':<35} {'frac':>6} {'fc':>8} {'incr':>6} {'cb':>6} {'cb%':>6} {'time':>6}"
    print(fmt)
    print("-" * 80)
    for r in results:
        print(
            f"{r['label']:<35} {r['frac']:>6.3f} {r['fc']:>8.1f} "
            f"{r['incr']:>6} {r['cutback']:>6} {r['cutback_rate']:>5.1%} {r['time']:>6.1f}"
        )
