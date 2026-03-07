"""7本撚線NCP曲げのプロファイリングスクリプト.

ボトルネック特定のため cProfile + カスタム計時で詳細分析する。
ログは /tmp にtee出力する。

[← README](../README.md)
"""

import cProfile
import pstats
import sys
import time

# ログファイルのパスを先に確保
log_path = f"/tmp/profile_ncp_7strand_{int(time.time())}.log"


class TeeWriter:
    """標準出力とファイルの両方に出力."""

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

print(f"=== NCP 7本撚線プロファイリング ===")
print(f"ログ出力先: {log_path}")
print()

from xkep_cae.numerical_tests.wire_bending_benchmark import run_bending_oscillation

# --- 45度曲げのプロファイリング ---
print("--- 7本撚線 45度 UL+NCP曲げ プロファイリング ---")

profiler = cProfile.Profile()
profiler.enable()

t0 = time.perf_counter()
result = run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=45,
    use_ncp=True,
    use_mortar=True,
    n_gauss=2,
    max_iter=30,
    tol_force=1e-6,
    adaptive_timestepping=True,
    use_updated_lagrangian=True,
    show_progress=True,
    n_cycles=0,  # 曲げのみ
)
elapsed = time.perf_counter() - t0

profiler.disable()

print(f"\n=== 結果 ===")
print(f"  converged: {result.phase1_converged}")
print(f"  総計算時間: {elapsed:.2f}s")
print(f"  NR反復: {result.phase1_result.total_newton_iterations}")
print()

# --- プロファイル結果の出力 ---
print("=" * 70)
print("  cProfile 結果 (cumulative time 上位40)")
print("=" * 70)
stats = pstats.Stats(profiler, stream=sys.stdout)
stats.sort_stats("cumulative")
stats.print_stats(40)

print()
print("=" * 70)
print("  cProfile 結果 (tottime 上位40)")
print("=" * 70)
stats.sort_stats("tottime")
stats.print_stats(40)

# --- 関数別の呼び出し回数・時間の詳細 ---
print()
print("=" * 70)
print("  関数呼び出し詳細 (solver_ncp / beam_timo3d / contact)")
print("=" * 70)
stats.sort_stats("tottime")
stats.print_stats("solver_ncp|beam_timo3d|pair|mortar|line_contact|assembly", 30)

sys.stdout = tee.terminal
tee.close()
print(f"\nプロファイリング完了。ログ: {log_path}")
