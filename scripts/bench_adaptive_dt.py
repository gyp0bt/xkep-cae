"""適応時間増分の n_load_steps=1 ベンチマーク.

n_load_steps=1 + dt_initial_fraction で 7本撚線90度曲げの性能を計測。
ハードコード版(n_load_steps=30)の26秒を上回ることを確認する。

Usage:
  python scripts/bench_adaptive_dt.py 2>&1 | tee /tmp/bench_adaptive_dt.log
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

log_path = f"/tmp/bench_adaptive_dt_{int(time.time())}.log"


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

print(f"=== 適応時間増分 n_load_steps=1 ベンチマーク ===")
print(f"ログ: {log_path}")
print(f"日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

from xkep_cae.numerical_tests.wire_bending_benchmark import _run_bending_oscillation

# ==================================================================
# テスト1: n_load_steps=1 + adaptive (dt_initial_fraction自動)
# ==================================================================
print("=" * 70)
print("  テスト1: 7本 90度曲げ n_load_steps=1 + adaptive")
print("=" * 70)

t0 = time.perf_counter()
result = _run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    n_bending_steps=1,  # n_load_steps=1 固定
    max_angle_per_step_deg=3.0,  # 初期dt = 1/30
    n_cycles=0,
    max_iter=40,
    tol_force=1e-4,
    show_progress=True,
    use_ncp=True,
    use_mortar=True,
    adaptive_timestepping=True,
    exclude_same_layer=True,
    midpoint_prescreening=True,
    use_line_search=False,
    g_on=0.0005,
    g_off=0.001,
    use_updated_lagrangian=True,
)
t1 = time.perf_counter() - t0

print(f"\n結果: converged={result.phase1_converged}, 時間={t1:.2f}s")
print(f"  NR反復: {result.phase1_result.total_newton_iterations}")
print(f"  n_load_steps(実効): {result.phase1_result.n_load_steps}")
print(f"  最大貫入比: {result.max_penetration_ratio:.6f}")

# ==================================================================
# テスト2: ハードコード版（n_load_steps=30, adaptive無し）
# ==================================================================
print()
print("=" * 70)
print("  テスト2: 7本 90度曲げ n_load_steps=30 (均等分割, 比較用)")
print("=" * 70)

t0 = time.perf_counter()
result_hc = _run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    n_bending_steps=30,
    n_cycles=0,
    max_iter=40,
    tol_force=1e-4,
    show_progress=True,
    use_ncp=True,
    use_mortar=True,
    adaptive_timestepping=False,
    exclude_same_layer=True,
    midpoint_prescreening=True,
    use_line_search=False,
    g_on=0.0005,
    g_off=0.001,
    use_updated_lagrangian=True,
)
t_hc = time.perf_counter() - t0

print(f"\n結果: converged={result_hc.phase1_converged}, 時間={t_hc:.2f}s")
print(f"  NR反復: {result_hc.phase1_result.total_newton_iterations}")

# ==================================================================
# サマリー
# ==================================================================
print()
print("=" * 70)
print("  サマリー")
print("=" * 70)
print(f"  adaptive (n_steps=1): {t1:.2f}s, conv={result.phase1_converged}")
print(f"  hardcoded (n_steps=30): {t_hc:.2f}s, conv={result_hc.phase1_converged}")
if result.phase1_converged and result_hc.phase1_converged:
    speedup = t_hc / t1 if t1 > 0 else float("inf")
    print(f"  高速化比: {speedup:.2f}x")
    if t1 < t_hc:
        print(f"  => adaptive版が {t_hc - t1:.2f}s 高速！")
    else:
        print(f"  => hardcoded版が {t1 - t_hc:.2f}s 高速（改善必要）")

# ==================================================================
# テスト3: n_load_steps=1 + adaptive + 揺動1周期
# ==================================================================
print()
print("=" * 70)
print("  テスト3: 7本 90度曲げ + 揺動1周期 n_load_steps=1 + adaptive")
print("=" * 70)

t0 = time.perf_counter()
result_full = _run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    n_bending_steps=1,
    max_angle_per_step_deg=3.0,
    oscillation_amplitude_mm=2.0,
    n_cycles=1,
    n_steps_per_quarter=3,
    max_iter=40,
    tol_force=1e-4,
    show_progress=True,
    use_ncp=True,
    use_mortar=True,
    adaptive_timestepping=True,
    exclude_same_layer=True,
    midpoint_prescreening=True,
    use_line_search=False,
    g_on=0.0005,
    g_off=0.001,
    use_updated_lagrangian=True,
)
t_full = time.perf_counter() - t0

print(f"\n結果:")
print(f"  Phase1(曲げ): converged={result_full.phase1_converged}")
print(f"  Phase2(揺動): converged={result_full.phase2_converged}")
print(f"  総計算時間: {t_full:.2f}s")
print(f"  最大貫入比: {result_full.max_penetration_ratio:.6f}")
print(f"  ターゲット: 26秒以下 → {'PASS' if t_full < 26 else 'FAIL'}")

# ==================================================================
# 最終サマリー
# ==================================================================
print()
print("=" * 70)
print("  最終サマリー")
print("=" * 70)
print(f"  90度曲げのみ:")
print(f"    adaptive (n_steps=1): {t1:.2f}s")
print(f"    hardcoded (n_steps=30): {t_hc:.2f}s")
if result.phase1_converged and result_hc.phase1_converged:
    print(f"    比: {t_hc / t1:.2f}x")
print(f"  90度曲げ + 揺動1周期:")
print(f"    adaptive (n_steps=1): {t_full:.2f}s")
print(f"    目標: 26秒以下 → {'PASS' if t_full < 26 else 'FAIL'}")

sys.stdout = tee.terminal
tee.close()
print(f"\nログ: {log_path}")
