"""ベースライン計測: n_bending_steps=30, adaptive=True（status-132相当）."""

from __future__ import annotations

import sys
import time

log_path = f"/tmp/bench_baseline_{int(time.time())}.log"


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

print(f"=== ベースライン計測 (status-132相当) ===")
print(f"ログ: {log_path}")
print()

from xkep_cae.numerical_tests.wire_bending_benchmark import _run_bending_oscillation

t0 = time.perf_counter()
result = _run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    n_bending_steps=30,  # ハードコード30ステップ
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
t_total = time.perf_counter() - t0

print(f"\n=== ベースライン結果 ===")
print(f"  Phase1: converged={result.phase1_converged}")
print(f"  Phase2: converged={result.phase2_converged}")
print(f"  総計算時間: {t_total:.2f}s")
print(f"  目標: 26秒以下")

sys.stdout = tee.terminal
tee.close()
print(f"ログ: {log_path}")
