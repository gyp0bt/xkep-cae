"""status-224 Run 06 の再現テスト.

status-224 の成功条件: E=25, 動的k_pen=4.56, status_filter=あり, max_incr=500

Usage:
    python contracts/check_reproduce_224.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)

# status-224 Run 06 と同等条件
cfg = DynamicThreePointBendContactJigConfig(
    E=25.0,
    n_periods=30.0,
    jig_push=30.0,
    max_increments=500,
    tol_force=1e-6,
    tol_disp=1e-8,
    max_nr_attempts=30,
    # k_pen=0 → 自動（動的推定）
)

print(f"E={cfg.E}, push={cfg.jig_push}, n_periods={cfg.n_periods}")
print(f"max_incr={cfg.max_increments}, tol_f={cfg.tol_force}, tol_d={cfg.tol_disp}")
print("solving...\n")

r = DynamicThreePointBendContactJigProcess().process(cfg)
sr = r.solver_result
mid = 6 * r.wire_mid_node + 1
final_frac = sr.load_history[-1] if sr.load_history else 0.0
fc_last = sr.contact_force_history[-1] if sr.contact_force_history else 0.0

print(f"\n{'=' * 60}")
print(f"RESULT: conv={sr.converged} frac={final_frac:.4f} fc={fc_last:.2f}N")
print(f"  incr={sr.n_increments} cutback={sr.n_cutbacks} time={sr.elapsed_seconds:.1f}s")
print(f"  wire_y={sr.u[mid]:+.6f}")

# 到達 push
push_reached = final_frac * cfg.jig_push
print(f"  push_reached={push_reached:.2f}mm / {cfg.jig_push}mm")

# 解析比較
import numpy as np

I = np.pi * (cfg.wire_diameter / 2) ** 4 / 4
k_eb = 48.0 * cfg.E * I / cfg.wire_length**3
P_eb = k_eb * push_reached
print(f"  k_EB={k_eb:.2f} N/mm, P_EB({push_reached:.1f}mm)={P_eb:.1f}N")
