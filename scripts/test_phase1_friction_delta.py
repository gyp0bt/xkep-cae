#!/usr/bin/env python3
"""Phase1摩擦+δ正則化の収束テスト.

δ正則化による active set 急変抑制の効果を検証する。
"""

import sys
import time

sys.path.insert(0, ".")

from xkep_cae.numerical_tests.wire_bending_benchmark import _run_bending_oscillation

log_path = f"/tmp/test_phase1_fric_delta_{int(time.time())}.log"


class TeeWriter:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, "w")

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

print("=== Phase1 摩擦+δ正則化 収束テスト ===")
print(f"ログ: {log_path}")
print(f"日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

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
    contact_compliance=0.0,  # δ=0（単調active setのみ有効）
)

# テスト1: μ=0.15 45度
print("=" * 70)
print("  テスト1: 7本 μ=0.15 45度曲げ（δ正則化あり）")
print("=" * 70)

t0 = time.perf_counter()
result_45 = _run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=45.0,
    max_iter=50,
    tol_force=1e-4,
    n_cycles=0,
    use_friction=True,
    mu=0.15,
    **_COMMON,
)
t_45 = time.perf_counter() - t0
print(f"\n結果: converged={result_45.phase1_converged}, 時間={t_45:.2f}s")
print(f"  NR反復: {result_45.phase1_result.total_newton_iterations}")
print(f"  活性接触ペア: {result_45.n_active_contacts}")

# テスト2: μ=0.15 90度
print()
print("=" * 70)
print("  テスト2: 7本 μ=0.15 90度曲げ（δ正則化あり）")
print("=" * 70)

t0 = time.perf_counter()
result_90 = _run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    max_iter=50,
    tol_force=1e-4,
    n_cycles=0,
    use_friction=True,
    mu=0.15,
    **_COMMON,
)
t_90 = time.perf_counter() - t0
print(f"\n結果: converged={result_90.phase1_converged}, 時間={t_90:.2f}s")
print(f"  NR反復: {result_90.phase1_result.total_newton_iterations}")
print(f"  活性接触ペア: {result_90.n_active_contacts}")

# サマリー
print()
print("=" * 70)
print("  サマリー")
print("=" * 70)
print(f"  45° μ=0.15: {'PASS' if result_45.phase1_converged else 'FAIL'} ({t_45:.1f}s)")
print(f"  90° μ=0.15: {'PASS' if result_90.phase1_converged else 'FAIL'} ({t_90:.1f}s)")
print(f"  ログ: {log_path}")

sys.stdout = tee.terminal
tee.close()
print(f"\n完了。ログ: {log_path}")
