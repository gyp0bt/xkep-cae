"""動的接触三点曲げの序盤収束チェック.

自動推定δで序盤の収束速度を確認する。
被膜あり/なしの切替: coating_stiffness > 0 で有効化。

Usage:
    python contracts/check_dynamic_contact.py          # 被膜なし
    python contracts/check_dynamic_contact.py --coat   # 被膜あり
"""

import sys
import warnings

from xkep_cae.numerical_tests.three_point_bend_jig import (  # noqa: E402
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)

warnings.filterwarnings("ignore", category=UserWarning)

use_coating = "--coat" in sys.argv
max_incr = 10

# 被膜パラメータ（Kelvin-Voigt: k=100 N/mm, c=0.01 N·s/mm）
coat_kwargs = {}
if use_coating:
    coat_kwargs = {
        "coating_stiffness": 100.0,
        "coating_damping": 0.01,
        "coating_mu": 0.1,
    }

cfg = DynamicThreePointBendContactJigConfig(
    jig_push=0.05,
    n_periods=2.0,
    n_elems_wire=20,
    max_increments=max_incr,
    **coat_kwargs,
)

label = "被膜あり" if use_coating else "被膜なし"
print(f"[{label}] delta=auto, k_pen=auto, max_incr={max_incr}", flush=True)
if use_coating:
    print(
        f"  coating: k={cfg.coating_stiffness}, c={cfg.coating_damping}, mu={cfg.coating_mu}",
        flush=True,
    )
print("solving...", flush=True)

r = DynamicThreePointBendContactJigProcess().process(cfg)
sr = r.solver_result
mid = 6 * r.wire_mid_node + 1

print(f"\nRESULT: conv={sr.converged} n={sr.n_increments}")
print(f"  wire_y={sr.u[mid]:+.6f} jig_y={sr.u[6 * r.n_wire_nodes + 1]:+.6f}")

print(f"\n{'incr':>4} {'frac':>8} {'att':>4} {'res':>10} {'rate':>8} {'fc':>10}")
for d in sr.increment_diagnostics:
    print(
        f"{d.step:4d} {d.load_frac:8.4f} {d.n_attempts:4d} "
        f"{d.final_residual:10.2e} {d.convergence_rate:8.4f} "
        f"{d.contact_force_norm:10.2e}"
    )
