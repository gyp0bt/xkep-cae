"""動的接触三点曲げの序盤収束チェック.

自動推定δで序盤10 incrementの収束速度を確認する。
"""

import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)

cfg = DynamicThreePointBendContactJigConfig(
    jig_push=0.05,
    n_periods=2.0,
    n_elems_wire=20,
    max_increments=10,
)

print(f"delta=auto(5000/r), k_pen=auto, n_periods={cfg.n_periods}", flush=True)
print("solving...", flush=True)

r = DynamicThreePointBendContactJigProcess().process(cfg)
sr = r.solver_result
mid = 6 * r.wire_mid_node + 1

print(f"\nRESULT: conv={sr.converged} n={sr.n_increments}")
print(f"  wire_y={sr.u[mid]:+.6f} jig_y={sr.u[6*r.n_wire_nodes+1]:+.6f}")

print(f"\n{'incr':>4} {'frac':>8} {'att':>4} {'res':>10} {'rate':>8} {'fc':>10}")
for d in sr.increment_diagnostics:
    print(
        f"{d.step:4d} {d.load_frac:8.4f} {d.n_attempts:4d} "
        f"{d.final_residual:10.2e} {d.convergence_rate:8.4f} {d.contact_force_norm:10.2e}"
    )
