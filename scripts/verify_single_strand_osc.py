"""単線（1本）90度曲げ+揺動テスト.

接触の影響を排除してUL+揺動フレームワーク自体の動作確認。

Usage:
  python scripts/verify_single_strand_osc.py 2>&1 | tee /tmp/verify_single_osc.log

[← README](../README.md)
"""

from __future__ import annotations

import sys
import time

log_path = f"/tmp/verify_single_osc_{int(time.time())}.log"


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

print("=== 単線 90度曲げ+揺動（接触なし）===")
print(f"ログ出力先: {log_path}")
print(f"日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

from xkep_cae.numerical_tests.wire_bending_benchmark import _run_bending_oscillation  # noqa: E402

# 1本撚線 = 接触なし
print("=" * 70)
print("  単線 90度曲げ+揺動（UL+NCP, adaptive, 接触なし）")
print("=" * 70)

t0 = time.perf_counter()
result = _run_bending_oscillation(
    n_strands=1,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    oscillation_amplitude_mm=2.0,
    n_cycles=1,
    n_steps_per_quarter=3,
    max_iter=50,
    tol_force=1e-4,
    use_ncp=True,
    use_mortar=False,
    adaptive_timestepping=True,
    use_updated_lagrangian=True,
    exclude_same_layer=False,
    midpoint_prescreening=True,
    use_line_search=True,
    mesh_gap=0.0,
    show_progress=True,
    du_norm_cap=0.5,
)
elapsed = time.perf_counter() - t0

print(f"\n結果:")
print(f"  Phase1(曲げ): converged={result.phase1_converged}")
print(f"  Phase2(揺動): converged={result.phase2_converged}")
print(f"  総計算時間: {elapsed:.2f}s")
print(f"  Phase2ステップ数: {len(result.phase2_results)}")

# 7本で90度曲げのみ（接触あり、Phase2なし）: 既に確認済
print()
print("=" * 70)
print("  7本撚線 90度曲げ+揺動（接触あり、NCP、SVD截断）")
print("  n_steps_per_quarter=6（より細かいステップ）")
print("=" * 70)

t0 = time.perf_counter()
result7 = _run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    oscillation_amplitude_mm=2.0,
    n_cycles=1,
    n_steps_per_quarter=6,
    max_iter=50,
    tol_force=1e-4,
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
    broadphase_margin_phase2=1.0,
)
elapsed7 = time.perf_counter() - t0

print(f"\n結果:")
print(f"  Phase1(曲げ): converged={result7.phase1_converged}")
print(f"  Phase2(揺動): converged={result7.phase2_converged}")
print(f"  総計算時間: {elapsed7:.2f}s")
print(f"  Phase2ステップ数: {len(result7.phase2_results)}")
print(f"  活性接触ペア: {result7.n_active_contacts}")

sys.stdout = tee.terminal
tee.close()
print(f"\n検証完了。ログ: {log_path}")
