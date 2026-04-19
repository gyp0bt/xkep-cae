"""7本撚線 90度曲げ（接触あり）回帰確認スクリプト.

status-298 / status-353 baseline の回帰確認用:
  - κ = π/2/L = 0.015708 [1/mm]（bending_angle = π/2 = 90°）
  - Hertz 型ペナルティ (α=1.5)
  - contact_enabled=True, free_end_mode=True
  - max_nr_attempts=200, tol_force=1e-8

status-298 baseline:
  frac=1.0000, incr=535, cutback=45, elapsed=752s

status-353 で数理台帳訂正（K_mat,ndir ≡ K_geo、コード数値挙動変更なし）
後の重量回帰を status-354（本確認）で実施。

Usage:
    python contracts/verify_7strand_90deg_contact.py 2>&1 | tee /tmp/log-$(date +%s).log

[← README](../README.md)
"""

from __future__ import annotations

import math
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def run_90deg_bending() -> tuple[float, int, int, float]:
    kappa_90 = math.pi / 2.0 / 100.0  # κ = π/(2·L), θ = κ·L = π/2
    cfg = StrandBendingOscillationConfig(
        n_strands=7,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=kappa_90,
        n_cycles=1,
        n_increments_per_cycle=40,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=True,
        penalty_exponent=1.5,
    )

    print("=" * 72)
    print("7本撚線 90度曲げ（接触あり）回帰確認 — status-354")
    print("=" * 72)
    print(f"  κ = {kappa_90:.6e} [1/mm]")
    print(f"  θ_target = {math.degrees(kappa_90 * 100.0):.2f}°")
    print(f"  n_strands = {cfg.n_strands}")
    print(f"  n_elements_per_pitch = {cfg.n_elements_per_pitch}")
    print(f"  penalty_exponent = {cfg.penalty_exponent}")
    print(f"  contact_enabled = {cfg.contact_enabled}")
    print(f"  free_end_mode = {cfg.free_end_mode}")
    print(f"  mu = {cfg.mu}")
    print(f"  max_nr_attempts = {cfg.max_nr_attempts}")
    print(f"  tol_force = {cfg.tol_force}")
    print()

    t0 = time.time()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.time() - t0

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0

    print()
    print("=" * 72)
    print("結果サマリ")
    print("=" * 72)
    print(f"  frac          = {frac:.4f}")
    print(f"  converged     = {sr.converged}")
    print(f"  n_increments  = {sr.n_increments}")
    print(f"  n_cutbacks    = {sr.n_cutbacks}")
    print(f"  elapsed       = {elapsed:.1f} sec")
    if sr.contact_force_history:
        print(f"  contact hist  = {len(sr.contact_force_history)} entries")
        print(f"  max contact F = {max(sr.contact_force_history):.6e}")
    print()
    print("--- status-298 baseline 比較 ---")
    print("  baseline: frac=1.0000, incr=535, cutback=45, 752s")
    print(f"  current : frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}, {elapsed:.1f}s")
    print()

    return frac, sr.n_increments, sr.n_cutbacks, elapsed


if __name__ == "__main__":
    run_90deg_bending()
