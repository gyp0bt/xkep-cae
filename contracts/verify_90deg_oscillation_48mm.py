"""90度曲げ + 先端横変位±48mm揺動の検証スクリプト.

status-299: 90度曲げ完走後に先端u_z横変位±48mmで2サイクル揺動。
Hertz型ペナルティ(α=1.5) + 接触あり。

Usage:
    python contracts/verify_90deg_oscillation_48mm.py 2>&1 | tee /tmp/log-$(date +%s).log

[← README](../README.md)
"""

import math
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.numerical_tests.strand_bending_oscillation import (  # noqa: E402
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def run_90deg_oscillation(amplitude: float = 48.0):
    """90度曲げ + 先端横変位揺動."""
    kappa_90 = math.pi / 2.0 / 100.0  # κ = π/(2*L), θ = κ*L = π/2
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
        penalty_exponent=1.5,  # Hertz型
        # 揺動パラメータ（status-299）
        n_oscillation_cycles=2,  # 2サイクル
        oscillation_amplitude=amplitude,  # ±amplitude mm
    )

    print("=" * 70)
    print(f"90度曲げ + 先端横変位±{amplitude:.1f}mm揺動（2サイクル）")
    print("=" * 70)
    print(f"  κ = {kappa_90:.6f} [1/mm]")
    print(f"  θ_target = {math.degrees(kappa_90 * 100):.1f}°")
    print(f"  n_strands = {cfg.n_strands}")
    print(f"  n_elements_per_pitch = {cfg.n_elements_per_pitch}")
    print(f"  penalty_exponent = {cfg.penalty_exponent}")
    print(f"  contact_enabled = {cfg.contact_enabled}")
    print(f"  n_oscillation_cycles = {cfg.n_oscillation_cycles}")
    print(f"  oscillation_amplitude = ±{cfg.oscillation_amplitude:.1f} mm")
    print(f"  max_nr_attempts = {cfg.max_nr_attempts}")
    print(f"  max_increments = {cfg.max_increments}")
    print()

    t0 = time.time()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.time() - t0

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0

    print()
    print("=" * 70)
    print("結果サマリ")
    print("=" * 70)
    print(f"  frac          = {frac:.4f}")
    print(f"  n_increments  = {sr.n_increments}")
    print(f"  n_cutbacks    = {sr.n_cutbacks}")
    print(f"  elapsed       = {elapsed:.1f} sec")
    print()
    print("--- ベースライン比較 ---")
    print("  曲げのみ (status-298): frac=1.0, incr=535, cutback=45, 752s")
    print()
    print(f"  今回 (曲げ+揺動±{amplitude:.0f}mm×2cyc):")
    print(f"    frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}")
    print()
    print("検証完了")
    return frac, sr.n_increments, sr.n_cutbacks


if __name__ == "__main__":
    import sys

    amp = float(sys.argv[1]) if len(sys.argv) > 1 else 48.0
    run_90deg_oscillation(amplitude=amp)
