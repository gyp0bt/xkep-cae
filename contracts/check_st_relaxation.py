"""s,t under-relaxation の効果検証.

frac=0.60 の 2-cycle 残差振動を s,t 更新の under-relaxation で解消する。

Usage:
    python contracts/check_st_relaxation.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

import xkep_cae.contact.solver.process as _sp_mod
import xkep_cae.numerical_tests.three_point_bend_jig as _tpb_mod
from xkep_cae.contact._contact_pair import _ContactConfigInput
from xkep_cae.contact.solver._newton_dynamic import NewtonDynamicInput as _orig_ndi
from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)


def _run(label: str, st_relaxation: float, max_increments: int = 200) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {label} (st_relax={st_relaxation})")
    print(f"{'=' * 60}\n")

    cfg = DynamicThreePointBendContactJigConfig(
        E=25.0,
        n_periods=30.0,
        jig_push=30.0,
        max_increments=max_increments,
        tol_force=1e-6,
        tol_disp=1e-8,
        max_nr_attempts=30,
    )

    _orig_cci = _tpb_mod._ContactConfigInput

    def _patched_cci(**kwargs):
        kwargs["st_relaxation"] = st_relaxation
        return _orig_cci(**kwargs)

    _tpb_mod._ContactConfigInput = _patched_cci
    try:
        r = DynamicThreePointBendContactJigProcess().process(cfg)
    finally:
        _tpb_mod._ContactConfigInput = _orig_cci

    sr = r.solver_result
    mid = 6 * r.wire_mid_node + 1
    final_frac = sr.load_history[-1] if sr.load_history else 0.0
    fc_last = sr.contact_force_history[-1] if sr.contact_force_history else 0.0

    print(f"\n--- {label} ---")
    print(
        f"  frac={final_frac:.4f} fc={fc_last:.2f}N incr={sr.n_increments} "
        f"cutback={sr.n_cutbacks} time={sr.elapsed_seconds:.1f}s"
    )

    # インクリメント履歴（要約）
    print(f"  {'incr':>4} {'frac':>8} {'att':>4} {'res':>10} {'rate':>8} {'fc':>10}")
    for d in sr.increment_diagnostics[-10:]:  # 最後10件のみ
        print(
            f"  {d.step:4d} {d.load_frac:8.4f} {d.n_attempts:4d} "
            f"{d.final_residual:10.2e} {d.convergence_rate:8.4f} "
            f"{d.contact_force_norm:10.2e}"
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

# ベースライン (α=1.0)
results.append(_run("Baseline (α=1.0)", st_relaxation=1.0))

# α=0.5
results.append(_run("Relaxation α=0.5", st_relaxation=0.5))

# α=0.3
results.append(_run("Relaxation α=0.3", st_relaxation=0.3))

# ================================================================
# 比較サマリ
# ================================================================
print("\n" + "=" * 70)
print("  比較サマリ")
print("=" * 70)
print(f"{'ケース':<30} {'frac':>8} {'fc(N)':>10} {'incr':>5} {'cutback':>7} {'time(s)':>8}")
for r in results:
    print(
        f"{r['label']:<30} {r['final_frac']:8.4f} {r['fc_last']:10.2f} "
        f"{r['n_increments']:5d} {r['n_cutbacks']:7d} {r['elapsed']:8.1f}"
    )
