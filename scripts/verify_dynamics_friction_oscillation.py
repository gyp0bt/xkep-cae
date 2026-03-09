#!/usr/bin/env python3
"""動的解析+摩擦あり曲げ揺動の収束検証.

目的: Generalized-α動的解析 + Coulomb摩擦（μ=0.15）で
7本撚線の曲げ揺動（Phase1 + Phase2）を収束させる。

修正内容:
  1. C_t正規化: ||C_t||/(k_t * n_active) でスケール統一
  2. mu_ramp_steps: 摩擦を段階的に導入（0→μ）
  3. 動的解析Phase1: 慣性効果で摩擦遷移を正則化

使い方:
    python scripts/verify_dynamics_friction_oscillation.py 2>&1 | tee /tmp/log-dyn-fric-$(date +%s).log

[← README](../README.md)
"""

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from xkep_cae.numerical_tests.wire_bending_benchmark import run_bending_oscillation

# --- ログ tee設定 ---
log_path = f"/tmp/verify_dyn_fric_osc_{int(time.time())}.log"


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

print("=== 動的解析+摩擦あり 曲げ揺動 収束検証 ===")
print(f"ログ出力先: {log_path}")
print(f"日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ==================================================================
# 共通パラメータ
# ==================================================================
_COMMON_PARAMS = dict(
    use_ncp=True,
    use_mortar=False,  # Point contact（チャタリング抑制）
    adaptive_timestepping=True,
    use_updated_lagrangian=True,
    exclude_same_layer=False,
    midpoint_prescreening=True,
    use_line_search=True,
    g_on=0.0005,
    g_off=0.001,
    mesh_gap=0.0,  # 自動計算
    show_progress=True,
    chattering_window=5,
    contact_stabilization=0.3,
    lambda_decay=0.5,
    du_norm_cap=0.5,
)

# Rayleigh減衰パラメータ
xi = 0.002  # 鋼の減衰比
f_osc = 1.0  # Hz
omega_1 = 2.0 * np.pi * f_osc
beta_R = 2.0 * xi / omega_1  # ≈ 6.4e-4

_DYNAMICS_PARAMS = dict(
    dynamics=True,
    dynamics_phase1=True,  # Phase1にも動的解析
    oscillation_frequency_hz=f_osc,
    bending_time=1.0,  # Phase1曲げ物理時間1s
    rho=7.85e-9,  # ton/mm³（鋼）
    rayleigh_alpha=0.0,
    rayleigh_beta=beta_R,
    rho_inf=0.9,  # わずかな数値減衰
)

# ==================================================================
# 1. 摩擦あり45度曲げ（動的解析 + mu_ramp）
# ==================================================================
print("=" * 70)
print("  テスト1: 7本 摩擦あり（μ=0.15）45度曲げ（動的+mu_ramp=5）")
print("=" * 70)

t0 = time.perf_counter()
result_45 = run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=45.0,
    max_iter=50,
    tol_force=1e-4,
    n_cycles=0,
    use_friction=True,
    mu=0.15,
    mu_ramp_steps=5,
    **_COMMON_PARAMS,
    **_DYNAMICS_PARAMS,
)
t_45 = time.perf_counter() - t0
print(f"\n結果: converged={result_45.phase1_converged}, 時間={t_45:.2f}s")
print(f"  NR反復: {result_45.phase1_result.total_newton_iterations}")
print(f"  活性接触ペア: {result_45.n_active_contacts}")

# ==================================================================
# 2. 摩擦あり45度曲げ+揺動1周期（動的解析 + mu_ramp）
# ==================================================================
print()
print("=" * 70)
print("  テスト2: 7本 摩擦あり（μ=0.15）45度曲げ+揺動（動的+mu_ramp=5）")
print(f"  Rayleigh: β_R={beta_R:.4e}, ρ∞={_DYNAMICS_PARAMS['rho_inf']:.2f}")
print("=" * 70)

t0 = time.perf_counter()
result_45_osc = run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=45.0,
    oscillation_amplitude_mm=2.0,
    n_cycles=1,
    n_steps_per_quarter=3,
    max_iter=50,
    tol_force=1e-4,
    use_friction=True,
    mu=0.15,
    mu_ramp_steps=5,
    **_COMMON_PARAMS,
    **_DYNAMICS_PARAMS,
)
t_45_osc = time.perf_counter() - t0
print(f"\n結果:")
print(f"  Phase1(曲げ): converged={result_45_osc.phase1_converged}")
print(f"  Phase2(揺動): converged={result_45_osc.phase2_converged}")
print(f"  計算時間: {t_45_osc:.2f}s")
if result_45_osc.phase2_results:
    p2_nr = sum(r.total_newton_iterations for r in result_45_osc.phase2_results)
    p2_conv = sum(1 for r in result_45_osc.phase2_results if r.converged)
    print(f"  Phase2 NR反復: {p2_nr}")
    print(f"  Phase2 収束ステップ: {p2_conv}/{len(result_45_osc.phase2_results)}")

# ==================================================================
# 3. 摩擦あり90度曲げ+揺動1周期（動的解析 + mu_ramp）
# ==================================================================
print()
print("=" * 70)
print("  テスト3: 7本 摩擦あり（μ=0.15）90度曲げ+揺動（動的+mu_ramp=5）")
print("=" * 70)

t0 = time.perf_counter()
result_90_osc = run_bending_oscillation(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    oscillation_amplitude_mm=2.0,
    n_cycles=1,
    n_steps_per_quarter=3,
    max_iter=50,
    tol_force=1e-4,
    use_friction=True,
    mu=0.15,
    mu_ramp_steps=5,
    **_COMMON_PARAMS,
    **_DYNAMICS_PARAMS,
)
t_90_osc = time.perf_counter() - t0
print(f"\n結果:")
print(f"  Phase1(曲げ): converged={result_90_osc.phase1_converged}")
print(f"  Phase2(揺動): converged={result_90_osc.phase2_converged}")
print(f"  計算時間: {t_90_osc:.2f}s")
if result_90_osc.phase2_results:
    p2_nr = sum(r.total_newton_iterations for r in result_90_osc.phase2_results)
    p2_conv = sum(1 for r in result_90_osc.phase2_results if r.converged)
    print(f"  Phase2 NR反復: {p2_nr}")
    print(f"  Phase2 収束ステップ: {p2_conv}/{len(result_90_osc.phase2_results)}")

# ==================================================================
# 4. サマリー
# ==================================================================
print()
print("=" * 70)
print("  検証サマリー")
print("=" * 70)
print(f"{'テスト':40s} {'結果':>10s} {'時間':>10s}")
print("-" * 65)
print(
    f"{'1. μ=0.15 45°曲げ（動的+ramp）':40s} "
    f"{'PASS' if result_45.phase1_converged else 'FAIL':>10s} "
    f"{t_45:>8.1f}s"
)
print(
    f"{'2. μ=0.15 45°曲げ+揺動（動的+ramp）':40s} "
    f"{'PASS' if result_45_osc.phase2_converged else 'FAIL':>10s} "
    f"{t_45_osc:>8.1f}s"
)
print(
    f"{'3. μ=0.15 90°曲げ+揺動（動的+ramp）':40s} "
    f"{'PASS' if result_90_osc.phase2_converged else 'FAIL':>10s} "
    f"{t_90_osc:>8.1f}s"
)

all_pass = (
    result_45.phase1_converged
    and result_45_osc.phase2_converged
    and result_90_osc.phase2_converged
)
print(f"\n  総合: {'ALL PASS' if all_pass else 'PARTIAL'}")
print(f"  ログ: {log_path}")

sys.stdout = tee.terminal
tee.close()
print(f"\n検証完了。ログ: {log_path}")
