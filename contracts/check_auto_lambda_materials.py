"""λ自動推定の異材料検証 — 銅(E=120e3)・アルミ(E=70e3)・鉄鋼(E=200e3).

status-241 TODO: lm_auto_lambda=True の c=20 定数が異材料で有効か検証。
定数 c=20 の根拠: status-240 で E=200e3, λ=1e-4 → c = λ × E = 20。

検証方法:
  各材料で (1) ベースライン(LM無し) (2) lm_auto_lambda=True を比較。
  frac 改善率と cutback 比率で効果を判定。

Usage:
    python contracts/check_auto_lambda_materials.py 2>&1 | tee /tmp/log-$(date +%s).log
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
    print(f"  E={cfg.E}, lm_auto_lambda={cfg.lm_auto_lambda}")
    print(f"  lm_lambda_init={cfg.lm_lambda_init}, freeze={cfg.freeze_geometry_in_nr}")
    print(f"  K_st={cfg.consistent_st_tangent}")
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


# 材料パラメータ (mm-ton-MPa 単位系)
MATERIALS = {
    "steel": {"E": 200e3, "nu": 0.3, "rho": 7.85e-9},
    "copper": {"E": 120e3, "nu": 0.34, "rho": 8.96e-9},
    "aluminum": {"E": 70e3, "nu": 0.33, "rho": 2.70e-9},
}

# 共通ソルバーパラメータ
COMMON = {
    "n_periods": 1.0,
    "jig_push": 5.0,
    "max_increments": 30,
    "tol_force": 1e-6,
    "tol_disp": 1e-8,
    "max_nr_attempts": 30,
    "freeze_geometry_in_nr": False,
    "consistent_st_tangent": True,
}


if __name__ == "__main__":
    results = []

    for mat_name, mat_params in MATERIALS.items():
        # ベースライン: LM無し
        results.append(
            _run_config(
                f"{mat_name} baseline (no LM)",
                **mat_params,
                **COMMON,
                lm_lambda_init=0.0,
                lm_adaptive=False,
                lm_auto_lambda=False,
            )
        )

        # lm_auto_lambda=True (c=20/E)
        results.append(
            _run_config(
                f"{mat_name} auto_lambda (c=20/E)",
                **mat_params,
                **COMMON,
                lm_lambda_init=0.0,
                lm_adaptive=True,
                lm_auto_lambda=True,
            )
        )

    # サマリ
    print("\n" + "=" * 90)
    print("SUMMARY — λ自動推定の異材料検証 (n_periods=1, jig_push=5mm)")
    print("=" * 90)
    print(
        f"{'Label':<35} {'frac':>6} {'fc':>8} {'incr':>6} "
        f"{'cb':>6} {'cb%':>6} {'time':>6}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['label']:<35} {r['frac']:>6.3f} {r['fc']:>8.1f} "
            f"{r['incr']:>6} {r['cutback']:>6} {r['cutback_rate']:>5.1%} "
            f"{r['time']:>6.1f}"
        )

    # 材料別改善判定
    print("\n--- 材料別 frac 改善 ---")
    for mat_name in MATERIALS:
        bl = next(r for r in results if r["label"].startswith(mat_name) and "baseline" in r["label"])
        al = next(r for r in results if r["label"].startswith(mat_name) and "auto_lambda" in r["label"])
        delta = al["frac"] - bl["frac"]
        direction = "改善" if delta > 0 else ("悪化" if delta < 0 else "変化なし")
        expected_lambda = 20.0 / MATERIALS[mat_name]["E"]
        print(
            f"  {mat_name}: baseline={bl['frac']:.3f} → auto={al['frac']:.3f} "
            f"(Δ={delta:+.3f} {direction}) λ={expected_lambda:.2e}"
        )
