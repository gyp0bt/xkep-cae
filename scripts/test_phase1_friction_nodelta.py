#!/usr/bin/env python3
"""Phase1摩擦テスト: δ=0 vs δ=auto の比較."""
import sys
import time

sys.path.insert(0, ".")
from xkep_cae.numerical_tests.wire_bending_benchmark import run_bending_oscillation

log_path = f"/tmp/test_p1_nodelta_{int(time.time())}.log"

class TeeWriter:
    def __init__(self, fp):
        self.terminal = sys.stdout
        self.file = open(fp, "w")
    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def close(self):
        self.file.close()

tee = TeeWriter(log_path)
sys.stdout = tee

_COMMON = dict(
    use_ncp=True,
    use_mortar=False,
    adaptive_timestepping=True,
    use_updated_lagrangian=True,
    exclude_same_layer=False,
    midpoint_prescreening=True,
    use_line_search=True,
    g_on=0.0005,
    g_off=0.001,
    mesh_gap=0.0,
    show_progress=True,
    chattering_window=5,
    contact_stabilization=0.3,
    lambda_decay=0.5,
    du_norm_cap=0.5,
)

# テスト1: δ=0 (正則化なし)
print("=" * 70)
print("  テスト1: μ=0.15 45° δ=0 (正則化なし)")
print("=" * 70)
t0 = time.perf_counter()
r1 = run_bending_oscillation(
    n_strands=7, n_pitches=0.5, bend_angle_deg=45.0,
    max_iter=50, tol_force=1e-4, n_cycles=0,
    use_friction=True, mu=0.15,
    contact_compliance=0.0,
    **_COMMON,
)
t1 = time.perf_counter() - t0
print(f"\n結果: converged={r1.phase1_converged}, 時間={t1:.2f}s, active={r1.n_active_contacts}")

# テスト2: μ=0.1 δ=0 (前回収束した設定)
print()
print("=" * 70)
print("  テスト2: μ=0.1 45° δ=0")
print("=" * 70)
t0 = time.perf_counter()
r2 = run_bending_oscillation(
    n_strands=7, n_pitches=0.5, bend_angle_deg=45.0,
    max_iter=50, tol_force=1e-4, n_cycles=0,
    use_friction=True, mu=0.1,
    contact_compliance=0.0,
    **_COMMON,
)
t2 = time.perf_counter() - t0
print(f"\n結果: converged={r2.phase1_converged}, 時間={t2:.2f}s, active={r2.n_active_contacts}")

print()
print("=" * 70)
print("  サマリー")
print("=" * 70)
print(f"  μ=0.15 δ=0: {'PASS' if r1.phase1_converged else 'FAIL'} ({t1:.1f}s)")
print(f"  μ=0.1  δ=0: {'PASS' if r2.phase1_converged else 'FAIL'} ({t2:.1f}s)")

sys.stdout = tee.terminal
tee.close()
print(f"完了。ログ: {log_path}")
