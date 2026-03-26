"""status-237 クイックテスト: n_periods=3 で解析的剛体表面の動作確認.

Usage:
    python contracts/check_rigid_surface_quick.py 2>&1 | tee /tmp/log-$(date +%s).log
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
    n_periods=3.0,
    jig_push=30.0,
    max_increments=500,
    tol_force=1e-6,
    tol_disp=1e-8,
    max_nr_attempts=30,
)

print("=" * 60)
print("クイックテスト: n_periods=3, 解析的剛体表面+粗メッシュ")
print("=" * 60)
print(f"  n_elems_wire={cfg.n_elems_wire}, use_rigid_surface={cfg.use_rigid_surface}")
print(f"  max_incr={cfg.max_increments}")
print("solving...")
sys.stdout.flush()

t0 = time.time()
r = DynamicThreePointBendContactJigProcess().process(cfg)
elapsed = time.time() - t0

sr = r.solver_result
mid = 6 * r.wire_mid_node + 1
final_frac = sr.load_history[-1] if sr.load_history else 0.0
fc_last = sr.contact_force_history[-1] if sr.contact_force_history else 0.0

print(f"\nRESULT: conv={sr.converged} frac={final_frac:.4f} fc={fc_last:.2f}N")
print(f"  incr={sr.n_increments} cutback={sr.n_cutbacks} wall_time={elapsed:.1f}s")
if sr.n_increments > 0:
    cutback_rate = sr.n_cutbacks / (sr.n_increments + sr.n_cutbacks)
    print(f"  cutback_rate={cutback_rate:.1%}")
print(f"  wire_y={sr.u[mid]:+.6f}")
