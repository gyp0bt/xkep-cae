"""被膜接触モデルの収束検証スクリプト.

status-142 TODO: 被膜接触+被膜摩擦（μ=0.25）の検証。
7本撚線45度曲げで被膜モデルの各構成を検証する。

ケース:
  1. 被膜なし（ベースライン, point contact）
  2. 被膜あり（coating_thickness=0.1mm, k_coat=E/t自動導出）
  3. 被膜あり + 摩擦（μ=0.25）
  4. 被膜あり + 粘性減衰 + 摩擦（μ=0.25）

Usage:
  python scripts/verify_coating_contact_convergence.py 2>&1 | tee /tmp/verify_coating_contact.log

[← README](../README.md)
"""

from __future__ import annotations

import sys
import time

# --- ログ tee設定 ---
log_path = f"/tmp/verify_coating_contact_{int(time.time())}.log"


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

print("=" * 70)
print("  被膜接触モデル 収束検証（status-142 TODO）")
print("=" * 70)
print(f"ログ出力先: {log_path}")
print(f"日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

from xkep_cae.numerical_tests.wire_bending_benchmark import run_bending_oscillation  # noqa: E402

# 共通パラメータ（point contact, mesh_gap で初期貫入防止）
COMMON_PARAMS = dict(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=45,
    use_ncp=True,
    use_mortar=False,  # Point contact（Mortar接触チャタリング問題あり）
    adaptive_timestepping=True,
    use_updated_lagrangian=True,
    show_progress=True,
    n_cycles=0,
    exclude_same_layer=True,
    midpoint_prescreening=True,
    use_line_search=False,
    g_on=0.0005,
    g_off=0.001,
    mesh_gap=0.15,
    max_iter=50,
    tol_force=1e-4,
)

COAT_T = 0.1  # mm（100μm被膜厚）
results = {}


def run_case(name, **extra_params):
    """ケースを実行してサマリーを表示."""
    print()
    print("=" * 70)
    print(f"  {name}")
    print("=" * 70)
    t0 = time.perf_counter()
    result = run_bending_oscillation(**COMMON_PARAMS, **extra_params)
    elapsed = time.perf_counter() - t0
    print(f"\n結果: converged={result.phase1_converged}, 時間={elapsed:.2f}s")
    print(f"  活性接触ペア: {result.n_active_contacts}")
    print(f"  最大貫入比: {result.max_penetration_ratio:.6f}")
    return result, elapsed


# ==================================================================
# ケース1: 被膜なし（ベースライン）
# ==================================================================
results["bare"] = run_case("ケース1: 被膜なし（ベースライン）")

# ==================================================================
# ケース2: 被膜あり（k_coat=E/t自動導出）
# ==================================================================
results["coated"] = run_case(
    "ケース2: 被膜あり（t=0.1mm, k_coat=E/t自動）",
    coating_thickness=COAT_T,
)

# ==================================================================
# ケース3: 被膜あり + 摩擦（μ=0.25）
# ==================================================================
results["coated_friction"] = run_case(
    "ケース3: 被膜あり + 被膜摩擦（μ=0.25）",
    coating_thickness=COAT_T,
    use_friction=True,
    mu=0.25,
)

# ==================================================================
# ケース4: 被膜あり + 粘性減衰 + 摩擦（μ=0.25）
# ==================================================================
results["coated_damped_friction"] = run_case(
    "ケース4: 被膜あり + 粘性減衰 + 摩擦（μ=0.25）",
    coating_thickness=COAT_T,
    coating_damping=100.0,  # MPa·s/mm
    use_friction=True,
    mu=0.25,
)

# ==================================================================
# サマリー
# ==================================================================
print()
print("=" * 70)
print("  検証サマリー")
print("=" * 70)

all_pass = True
for key, label in [
    ("bare", "被膜なし（ベースライン）"),
    ("coated", "被膜あり"),
    ("coated_friction", "被膜+摩擦(μ=0.25)"),
    ("coated_damped_friction", "被膜+減衰+摩擦"),
]:
    result, t_elapsed = results[key]
    status = "PASS" if result.phase1_converged else "FAIL"
    if not result.phase1_converged:
        all_pass = False
    print(
        f"  {label:25s}: {status}  "
        f"active={result.n_active_contacts:3d}  "
        f"pen_ratio={result.max_penetration_ratio:.6f}  "
        f"time={t_elapsed:.1f}s"
    )

print(f"\n  総合: {'ALL PASS' if all_pass else 'FAIL'}")
print(f"  ログ: {log_path}")

sys.stdout = tee.terminal
tee.close()
print(f"\n検証完了。ログ: {log_path}")

if not all_pass:
    sys.exit(1)
