"""E=200e3 鉄鋼での K_st + LM 簡易検証（n_periods=1, max_incr=30）.

status-240 用の簡易版。短時間で効果の方向性を把握。

Usage:
    python contracts/check_steel_kst_lm_quick.py 2>&1 | tee /tmp/log-$(date +%s).log
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
        E=200e3,
        n_periods=1.0,
        jig_push=5.0,
        max_increments=30,
        tol_force=1e-6,
        tol_disp=1e-8,
        max_nr_attempts=30,
        **overrides,
    )
    print(f"\n{'=' * 60}")
    print(f"構成: {label}")
    print(f"  E={cfg.E}, freeze={cfg.freeze_geometry_in_nr}")
    print(f"  K_st={cfg.consistent_st_tangent}, LM_init={cfg.lm_lambda_init}")
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

        # 力収束/変位収束インクリメント分類
        n_force = 0
        n_disp = 0
        for d in sr.diagnostics_history:
            if hasattr(d, "res_history") and d.res_history:
                last_ratio = d.res_history[-1]
                if last_ratio < cfg.tol_force:
                    n_force += 1
                else:
                    n_disp += 1

        print(f"\nRESULT: conv={sr.converged} frac={final_frac:.4f} fc={fc_last:.2f}N")
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
            "converged": sr.converged,
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\nFAILED: {e} (wall_time={elapsed:.1f}s)")
        return {
            "label": label,
            "frac": 0.0,
            "incr": 0,
            "cutback": 0,
            "cutback_rate": 1.0,
            "time": elapsed,
            "converged": False,
            "fc": 0.0,
        }


if __name__ == "__main__":
    results = []

    # 1. ベースライン: freeze=T, K_st=OFF
    results.append(
        _run_config(
            "baseline (freeze=T)",
            freeze_geometry_in_nr=True,
            consistent_st_tangent=False,
            lm_lambda_init=0.0,
            lm_adaptive=False,
        )
    )

    # 2. freeze=F, K_st=ON, LM=1e-4
    results.append(
        _run_config(
            "freeze=F, K_st=ON, LM=1e-4",
            freeze_geometry_in_nr=False,
            consistent_st_tangent=True,
            lm_lambda_init=1e-4,
            lm_adaptive=True,
        )
    )

    # 3. freeze=F, K_st=OFF（幾何更新のみ）
    results.append(
        _run_config(
            "freeze=F, K_st=OFF",
            freeze_geometry_in_nr=False,
            consistent_st_tangent=False,
            lm_lambda_init=0.0,
            lm_adaptive=False,
        )
    )

    # サマリ
    print("\n" + "=" * 80)
    print("SUMMARY — E=200e3 鉄鋼 (n_periods=1, jig_push=5mm, max_incr=30)")
    print("=" * 80)
    print(f"{'Label':<35} {'frac':>6} {'fc':>8} {'incr':>6} {'cb':>6} {'cb%':>6} {'time':>6}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['label']:<35} {r['frac']:>6.3f} {r['fc']:>8.1f} "
            f"{r['incr']:>6} {r['cutback']:>6} {r['cutback_rate']:>5.1%} {r['time']:>6.1f}"
        )
