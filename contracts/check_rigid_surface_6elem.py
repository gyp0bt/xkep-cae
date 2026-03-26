"""status-238 追加: n_elems_wire=6 での n_periods=30 収束検証.

n_elems_wire=4 (L_elem=25mm) vs n_elems_wire=6 (L_elem=16.7mm) の比較。
wire_diameter=17mm に対して L_elem≈diameter の境界ケース。

Usage:
    python contracts/check_rigid_surface_6elem.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)

cfg = DynamicThreePointBendContactJigConfig(
    E=25.0,
    n_periods=30.0,
    jig_push=30.0,
    max_increments=10000,
    tol_force=1e-6,
    tol_disp=1e-8,
    max_nr_attempts=30,
    n_elems_wire=6,  # 4→6 に変更（L_elem=16.7mm ≈ wire_diameter=17mm）
)

print("=" * 60)
print("n_elems_wire=6 収束検証: 解析的剛体表面")
print("=" * 60)
print(f"  E={cfg.E} MPa, push={cfg.jig_push} mm, n_periods={cfg.n_periods}")
print(f"  n_elems_wire={cfg.n_elems_wire}, L_elem={cfg.wire_length / cfg.n_elems_wire:.1f}mm")
print(
    f"  wire_diameter={cfg.wire_diameter}mm, L_elem/diameter={cfg.wire_length / cfg.n_elems_wire / cfg.wire_diameter:.2f}"
)
print(f"  use_rigid_surface={cfg.use_rigid_surface}")
print(f"  max_incr={cfg.max_increments}, tol_f={cfg.tol_force}, tol_d={cfg.tol_disp}")
print()
print("ベースライン:")
print("  n_elems=4:  frac=1.0 fc=216.96N incr=707 cutback=400 rate=36.1% time=476s")
print("  n_elems=20: frac=1.0 fc=208.6N  incr=1592 cutback=2477 rate=60.9% time=4403s")
print()
print("solving...")
sys.stdout.flush()

t0 = time.time()
r = DynamicThreePointBendContactJigProcess().process(cfg)
elapsed = time.time() - t0

sr = r.solver_result
mid = 6 * r.wire_mid_node + 1
final_frac = sr.load_history[-1] if sr.load_history else 0.0
fc_last = sr.contact_force_history[-1] if sr.contact_force_history else 0.0

print(f"\n{'=' * 60}")
print(f"RESULT: conv={sr.converged} frac={final_frac:.4f} fc={fc_last:.2f}N")
print(f"  incr={sr.n_increments} cutback={sr.n_cutbacks} wall_time={elapsed:.1f}s")
print(f"  wire_y={sr.u[mid]:+.6f}")

push_reached = final_frac * cfg.jig_push
print(f"  push_reached={push_reached:.2f}mm / {cfg.jig_push}mm")

import numpy as np

I = np.pi * (cfg.wire_diameter / 2) ** 4 / 4
k_eb = 48.0 * cfg.E * I / cfg.wire_length**3
P_eb = k_eb * push_reached
print(f"  k_EB={k_eb:.2f} N/mm, P_EB({push_reached:.1f}mm)={P_eb:.1f}N")

if sr.n_increments > 0:
    cutback_rate = sr.n_cutbacks / (sr.n_increments + sr.n_cutbacks)
    print(f"  cutback_rate={cutback_rate:.1%}")

print()
print("比較:")
print("  n_elems=4:  frac=1.0 fc=216.96N incr=707  cutback=400  rate=36.1% time=476s")
print("  n_elems=20: frac=1.0 fc=208.6N  incr=1592 cutback=2477 rate=60.9% time=4403s")
print(
    f"  n_elems=6:  frac={final_frac:.4f} fc={fc_last:.2f}N incr={sr.n_increments} "
    f"cutback={sr.n_cutbacks}",
    end="",
)
if sr.n_increments > 0:
    print(f" rate={cutback_rate:.1%} time={elapsed:.0f}s")
else:
    print()
print("=" * 60)
