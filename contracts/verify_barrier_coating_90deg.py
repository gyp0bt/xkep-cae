"""被膜バリア関数の90度曲げ収束性検証.

status-303 TODO: 被膜付き90度曲げでバリア関数の収束性検証（status-298ベースライン比較）。

比較対象:
  - status-298: 被膜なし (frac=1.0, incr=535, cutback=45, 752s)
  - status-301: 線形被膜 (90度曲げ単体ではなく曲げ+揺動で incr=965, cutback=31, 555s)
  - 今回: バリア関数被膜 (coating_barrier=True)

Usage:
    python contracts/verify_barrier_coating_90deg.py 2>&1 | tee /tmp/log-barrier-$(date +%s).log

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


def run_barrier_coating_90deg():
    """バリア関数被膜付き 90度曲げ."""
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
        # 被膜パラメータ（status-301と同一）
        coating_stiffness=1_000_000.0,  # 1e6 N/mm
        coating_damping=10_000.0,  # 1e4 N·s/mm
        coating_mu=0.3,
        coating_thickness=0.05,  # 0.05 mm
        coating_barrier=True,  # status-303: バリア関数
    )

    print("=" * 70)
    print("90度曲げ検証: バリア関数被膜 + Hertz型ペナルティ(α=1.5)")
    print("=" * 70)
    print(f"  κ = {kappa_90:.6f} [1/mm]")
    print(f"  θ_target = {math.degrees(kappa_90 * 100):.1f}°")
    print(f"  n_strands = {cfg.n_strands}")
    print(f"  n_elements_per_pitch = {cfg.n_elements_per_pitch}")
    print(f"  penalty_exponent = {cfg.penalty_exponent}")
    print(f"  contact_enabled = {cfg.contact_enabled}")
    print(f"  coating_stiffness = {cfg.coating_stiffness}")
    print(f"  coating_thickness = {cfg.coating_thickness}")
    print(f"  coating_barrier = {cfg.coating_barrier}")
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
    print()
    print("  status-298 (被膜なし, Hertz型):")
    print("    frac=1.0000, incr=535, cutback=45, elapsed=752s")
    print()
    print("  今回 (バリア関数被膜 + Hertz型):")
    print(
        f"    frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}, "
        f"elapsed={elapsed:.1f}s"
    )
    print()

    # status-298との比較
    base_incr, base_cb, base_time = 535, 45, 752.0
    if frac >= 0.999:
        print("  → frac: 完走 ✓")
    else:
        print(f"  → frac: {frac:.4f} (未完走)")

    if sr.n_increments < base_incr:
        pct = (base_incr - sr.n_increments) / base_incr * 100
        print(f"  → incr改善: {base_incr} → {sr.n_increments} ({pct:.0f}%削減)")
    else:
        pct = (sr.n_increments - base_incr) / base_incr * 100
        print(f"  → incr変化: {base_incr} → {sr.n_increments} (+{pct:.0f}%)")

    if sr.n_cutbacks < base_cb:
        pct = (base_cb - sr.n_cutbacks) / base_cb * 100
        print(f"  → cutback改善: {base_cb} → {sr.n_cutbacks} ({pct:.0f}%削減)")
    else:
        pct = (sr.n_cutbacks - base_cb) / base_cb * 100
        print(f"  → cutback変化: {base_cb} → {sr.n_cutbacks} (+{pct:.0f}%)")

    if elapsed < base_time:
        pct = (base_time - elapsed) / base_time * 100
        print(f"  → 実行時間改善: {base_time:.0f}s → {elapsed:.1f}s ({pct:.0f}%短縮)")
    else:
        pct = (elapsed - base_time) / base_time * 100
        print(f"  → 実行時間変化: {base_time:.0f}s → {elapsed:.1f}s (+{pct:.0f}%)")

    print()
    print("検証完了")
    return frac, sr.n_increments, sr.n_cutbacks, elapsed


if __name__ == "__main__":
    run_barrier_coating_90deg()
