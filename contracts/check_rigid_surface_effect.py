"""status-237 TODO: 解析的剛体表面 + 粗メッシュでの n_periods=30 収束検証.

status-234 ベースライン: n_periods=30 frac=1.0 208.6N incr=1592 cutback=2477
status-237 変更点: n_elems_wire=4(粗メッシュ) + use_rigid_surface=True(解析的円柱)

Usage:
    python contracts/check_rigid_surface_effect.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
)

# --- ベースライン比較用パラメータ（status-234 と同等条件）---
cfg = DynamicThreePointBendContactJigConfig(
    E=25.0,
    n_periods=30.0,
    jig_push=30.0,
    max_increments=10000,  # 制限なし（500→10000）
    tol_force=1e-6,
    tol_disp=1e-8,
    max_nr_attempts=30,
    # status-237 デフォルト:
    # n_elems_wire=4 (粗メッシュ)
    # use_rigid_surface=True (解析的剛体円柱)
)

print("=" * 60)
print("status-237 収束検証: 解析的剛体表面 + 粗メッシュ")
print("=" * 60)
print(f"  E={cfg.E} MPa, push={cfg.jig_push} mm, n_periods={cfg.n_periods}")
print(f"  n_elems_wire={cfg.n_elems_wire}, use_rigid_surface={cfg.use_rigid_surface}")
print(f"  max_incr={cfg.max_increments}, tol_f={cfg.tol_force}, tol_d={cfg.tol_disp}")
print(f"  max_nr={cfg.max_nr_attempts}")
print()
print("ベースライン (status-234): frac=1.0 208.6N incr=1592 cutback=2477")
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

# 到達 push
push_reached = final_frac * cfg.jig_push
print(f"  push_reached={push_reached:.2f}mm / {cfg.jig_push}mm")

# 解析比較
import numpy as np

I = np.pi * (cfg.wire_diameter / 2) ** 4 / 4
k_eb = 48.0 * cfg.E * I / cfg.wire_length**3
P_eb = k_eb * push_reached
print(f"  k_EB={k_eb:.2f} N/mm, P_EB({push_reached:.1f}mm)={P_eb:.1f}N")

# カットバック率
if sr.n_increments > 0:
    cutback_rate = sr.n_cutbacks / (sr.n_increments + sr.n_cutbacks)
    print(f"  cutback_rate={cutback_rate:.1%}")

print()
print("比較:")
print("  status-234: frac=1.0 fc=208.6N incr=1592 cutback=2477 cutback_rate=60.9%")
print(
    f"  status-237: frac={final_frac:.4f} fc={fc_last:.2f}N incr={sr.n_increments} cutback={sr.n_cutbacks}",
    end="",
)
if sr.n_increments > 0:
    print(f" cutback_rate={cutback_rate:.1%}")
else:
    print()
print("=" * 60)
