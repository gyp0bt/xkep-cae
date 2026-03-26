"""dof_scale_rot パラメータスイープ — 三点曲げで最適値調査.

status-241 TODO: dof_scale_rot の最適値を 0.3〜0.8 で調査。
回転 DOF の NR 更新を減衰させ、並進/回転残差の逆相関を緩和する。

Usage:
    python contracts/check_dof_scale_rot_sweep.py 2>&1 | tee /tmp/log-$(date +%s).log
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
    cfg = DynamicThreePointBendContactJigConfig(**overrides)
    print(f"\n{'=' * 60}")
    print(f"構成: {label}")
    print(f"  dof_scale_rot={cfg.dof_scale_rot}, lm_auto_lambda={cfg.lm_auto_lambda}")
    print(f"  E={cfg.E}, freeze={cfg.freeze_geometry_in_nr}")
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

        print(f"\nRESULT: conv={sr.converged} frac={final_frac:.4f} fc={fc_last:.2f}N")
        print(
            f"  incr={sr.n_increments} cutback={sr.n_cutbacks} "
            f"cutback_rate={cutback_rate:.1%}"
        )
        print(f"  wall_time={elapsed:.1f}s")
        return {
            "label": label,
            "dof_scale_rot": cfg.dof_scale_rot,
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
            "dof_scale_rot": overrides.get("dof_scale_rot", 1.0),
            "frac": 0.0,
            "incr": 0,
            "cutback": 0,
            "cutback_rate": 1.0,
            "time": elapsed,
            "converged": False,
            "fc": 0.0,
        }


# 共通パラメータ: E=200e3 鉄鋼, K_st+LM自動推定
COMMON = {
    "E": 200e3,
    "nu": 0.3,
    "rho": 7.85e-9,
    "n_periods": 1.0,
    "jig_push": 5.0,
    "max_increments": 30,
    "tol_force": 1e-6,
    "tol_disp": 1e-8,
    "max_nr_attempts": 30,
    "freeze_geometry_in_nr": False,
    "consistent_st_tangent": True,
    "lm_auto_lambda": True,
    "lm_adaptive": True,
}

SCALE_VALUES = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


if __name__ == "__main__":
    results = []

    for scale in SCALE_VALUES:
        label = f"dof_scale_rot={scale:.1f}"
        results.append(
            _run_config(label, **COMMON, dof_scale_rot=scale)
        )

    # サマリ
    print("\n" + "=" * 90)
    print("SUMMARY — dof_scale_rot スイープ (E=200e3, K_st+LM auto, n_periods=1)")
    print("=" * 90)
    print(
        f"{'scale':>6} {'frac':>6} {'fc':>8} {'incr':>6} "
        f"{'cb':>6} {'cb%':>6} {'time':>6}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['dof_scale_rot']:>6.1f} {r['frac']:>6.3f} {r['fc']:>8.1f} "
            f"{r['incr']:>6} {r['cutback']:>6} {r['cutback_rate']:>5.1%} "
            f"{r['time']:>6.1f}"
        )

    # 最適値判定
    best = max(results, key=lambda r: (r["frac"], -r["cutback_rate"]))
    print(f"\n最適: dof_scale_rot={best['dof_scale_rot']:.1f} "
          f"(frac={best['frac']:.3f}, cutback={best['cutback_rate']:.1%})")
