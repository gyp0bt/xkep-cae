"""7本撚線 90度曲げの収束検証スクリプト.

SVD截断Schur complementの効果を確認する。

Usage:
  python scripts/verify_90deg_bend.py 2>&1 | tee /tmp/verify_90deg.log

[← README](../README.md)
"""

from __future__ import annotations

import sys
import time

# --- ログ tee設定 ---
log_path = f"/tmp/verify_90deg_{int(time.time())}.log"


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

print("=== 7本撚線 90度曲げ 収束検証 ===")
print(f"ログ出力先: {log_path}")
print(f"日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

from xkep_cae.numerical_tests.wire_bending_benchmark import run_bending_oscillation  # noqa: E402

_COMMON_PARAMS = dict(
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

print("=" * 70)
print("  7本撚線 90度曲げ（UL+NCP, point contact, adaptive, SVD截断）")
print("=" * 70)

t0 = time.perf_counter()
result = run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    max_iter=50,
    tol_force=1e-4,
    n_cycles=0,
    **_COMMON_PARAMS,
)
elapsed = time.perf_counter() - t0

print(f"\n結果: converged={result.phase1_converged}, 時間={elapsed:.2f}s")
print(f"  NR反復: {result.phase1_result.total_newton_iterations}")
print(f"  最大貫入比: {result.max_penetration_ratio:.6f}")
print(f"  活性接触ペア: {result.n_active_contacts}")

sys.stdout = tee.terminal
tee.close()
print(f"\n検証完了。ログ: {log_path}")

if not result.phase1_converged:
    sys.exit(1)
