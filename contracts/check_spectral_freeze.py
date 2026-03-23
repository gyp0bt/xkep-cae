"""条件数スペクトル分析 + NR内幾何凍結の検証.

1. E=25 三点曲げで frac=0.60 付近の K_T 条件数・固有値を計測
2. freeze_geometry_in_nr=True で同じ計算を実行し収束改善を評価

Usage:
    python contracts/check_spectral_freeze.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

import warnings
from unittest.mock import patch

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.contact._contact_pair import _ContactConfigInput
from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)


def _patched_config_init(*, freeze_geometry_in_nr_override=False, **kwargs):
    """_ContactConfigInput のコンストラクタに freeze_geometry_in_nr を注入."""
    kwargs["freeze_geometry_in_nr"] = freeze_geometry_in_nr_override
    return _ContactConfigInput(**kwargs)


def _run_bending(
    *,
    label: str,
    freeze_geometry_in_nr: bool = False,
    compute_cond: bool = False,
    E: float = 25.0,
    max_increments: int = 200,
    n_periods: float = 30.0,
    jig_push: float = 30.0,
) -> dict:
    """三点曲げを実行し結果を返す."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  E={E}, freeze_st_nr={freeze_geometry_in_nr}, cond={compute_cond}")
    print(f"{'=' * 60}\n")

    cfg = DynamicThreePointBendContactJigConfig(
        E=E,
        n_periods=n_periods,
        jig_push=jig_push,
        max_increments=max_increments,
        tol_force=1e-6,
        tol_disp=1e-8,
        max_nr_attempts=30,
    )

    # freeze_geometry_in_nr / compute_condition_number を注入するため、
    # 内部の _ContactConfigInput 構築と NewtonDynamicInput 構築を monkey-patch する。
    import xkep_cae.numerical_tests.three_point_bend_jig as _tpb_mod
    import xkep_cae.contact.solver.process as _sp_mod

    # 1. freeze_geometry_in_nr 注入: _ContactConfigInput のインスタンス化をフック
    _orig_cci = _tpb_mod._ContactConfigInput

    def _patched_cci(**kwargs):
        kwargs["freeze_geometry_in_nr"] = freeze_geometry_in_nr
        return _orig_cci(**kwargs)

    # 2. compute_condition_number 注入: NewtonDynamicInput をフック
    from xkep_cae.contact.solver._newton_dynamic import NewtonDynamicInput as _orig_ndi

    def _patched_ndi(**kwargs):
        if compute_cond:
            kwargs["compute_condition_number"] = True
        return _orig_ndi(**kwargs)

    _tpb_mod._ContactConfigInput = _patched_cci
    _sp_mod.NewtonDynamicInput = _patched_ndi
    try:
        r = DynamicThreePointBendContactJigProcess().process(cfg)
    finally:
        _tpb_mod._ContactConfigInput = _orig_cci
        _sp_mod.NewtonDynamicInput = _orig_ndi

    sr = r.solver_result
    mid = 6 * r.wire_mid_node + 1

    # 結果サマリ
    final_frac = sr.load_history[-1] if sr.load_history else 0.0
    fc_last = sr.contact_force_history[-1] if sr.contact_force_history else 0.0
    print(f"\n--- {label} 結果 ---")
    print(f"  conv={sr.converged} n_incr={sr.n_increments} frac={final_frac:.4f}")
    print(f"  wire_y={sr.u[mid]:+.6f}")
    print(f"  contact_force={fc_last:.2f} N")
    print(f"  cutbacks={sr.n_cutbacks}")
    print(f"  elapsed={sr.elapsed_seconds:.1f}s")

    # インクリメント履歴
    print(f"\n{'incr':>4} {'frac':>8} {'att':>4} {'res':>10} {'rate':>8} {'fc':>10}")
    for d in sr.increment_diagnostics:
        print(
            f"{d.step:4d} {d.load_frac:8.4f} {d.n_attempts:4d} "
            f"{d.final_residual:10.2e} {d.convergence_rate:8.4f} "
            f"{d.contact_force_norm:10.2e}"
        )

    # 条件数履歴（最終インクリメントの診断から）
    if sr.diagnostics and sr.diagnostics.condition_number_history:
        print("\n  [Spectral Analysis — 最終インクリメント]")
        for i, (cond, eig_min, eig_max) in enumerate(
            zip(
                sr.diagnostics.condition_number_history,
                sr.diagnostics.min_eigenvalue_history,
                sr.diagnostics.max_eigenvalue_history,
            )
        ):
            print(f"    att={i}: cond={cond:.2e}, λ_min={eig_min:.2e}, λ_max={eig_max:.2e}")

    return {
        "converged": sr.converged,
        "final_frac": final_frac,
        "fc_last": fc_last,
        "n_increments": sr.n_increments,
        "elapsed": sr.elapsed_seconds,
        "n_cutbacks": sr.n_cutbacks,
    }


# ================================================================
# ケース 1: ベースライン（条件数分析付き）— 最大 20 インクリメント
# ================================================================
print("\n" + "=" * 70)
print("  ケース 1: ベースライン + 条件数分析 (E=25)")
print("=" * 70)
r1 = _run_bending(
    label="Baseline + spectral",
    freeze_geometry_in_nr=False,
    compute_cond=True,
    max_increments=20,
)

# ================================================================
# ケース 2: NR内幾何凍結（条件数分析付き）
# ================================================================
print("\n" + "=" * 70)
print("  ケース 2: NR内幾何凍結 + 条件数分析 (E=25)")
print("=" * 70)
r2 = _run_bending(
    label="freeze_st_nr + spectral",
    freeze_geometry_in_nr=True,
    compute_cond=True,
    max_increments=20,
)

# ================================================================
# ケース 3: ベースライン長時間実行 — frac 到達確認
# ================================================================
print("\n" + "=" * 70)
print("  ケース 3: ベースライン長時間実行 (E=25, max_incr=200)")
print("=" * 70)
r3 = _run_bending(
    label="Baseline long run",
    freeze_geometry_in_nr=False,
    compute_cond=False,
    max_increments=200,
)

# ================================================================
# ケース 4: NR内幾何凍結 長時間実行
# ================================================================
print("\n" + "=" * 70)
print("  ケース 4: NR内幾何凍結 長時間実行 (E=25, max_incr=200)")
print("=" * 70)
r4 = _run_bending(
    label="freeze_st_nr long run",
    freeze_geometry_in_nr=True,
    compute_cond=False,
    max_increments=200,
)

# ================================================================
# 比較サマリ
# ================================================================
print("\n" + "=" * 70)
print("  比較サマリ")
print("=" * 70)
print(f"{'ケース':<40} {'frac':>8} {'fc(N)':>10} {'incr':>5} {'cutback':>7} {'time(s)':>8}")
for lbl, r in [
    ("1. Baseline+spectral (20)", r1),
    ("2. freeze_st_nr+spectral (20)", r2),
    ("3. Baseline (200)", r3),
    ("4. freeze_st_nr (200)", r4),
]:
    print(
        f"{lbl:<40} {r['final_frac']:8.4f} {r['fc_last']:10.2f} "
        f"{r['n_increments']:5d} {r['n_cutbacks']:7d} {r['elapsed']:8.1f}"
    )
