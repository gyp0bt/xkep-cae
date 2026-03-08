"""7本撚線 90度曲げ+揺動（接触なし）の収束検証.

接触を切って揺動が動くか確認する。

Usage:
  python scripts/verify_90deg_no_contact.py 2>&1 | tee /tmp/verify_90deg_nocontact.log

[← README](../README.md)
"""

from __future__ import annotations

import sys
import time

log_path = f"/tmp/verify_90deg_nocontact_{int(time.time())}.log"


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

print("=== 7本撚線 90度曲げ+揺動（接触なし）===")
print(f"ログ出力先: {log_path}")
print(f"日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

from xkep_cae.numerical_tests.wire_bending_benchmark import run_bending_oscillation  # noqa: E402

print("=" * 70)
print("  7本撚線 90度曲げ+揺動（接触なし、UL、adaptive）")
print("=" * 70)

t0 = time.perf_counter()
result = run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    oscillation_amplitude_mm=2.0,
    n_cycles=1,
    n_steps_per_quarter=3,
    max_iter=50,
    tol_force=1e-4,
    use_ncp=False,  # 接触なし
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
)
elapsed = time.perf_counter() - t0

print(f"\n結果:")
print(f"  Phase1(曲げ): converged={result.phase1_converged}")
print(f"  Phase2(揺動): converged={result.phase2_converged}")
print(f"  総計算時間: {elapsed:.2f}s")

sys.stdout = tee.terminal
tee.close()
print(f"\n検証完了。ログ: {log_path}")
