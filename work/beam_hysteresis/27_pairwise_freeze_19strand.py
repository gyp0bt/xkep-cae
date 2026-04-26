"""work/beam_hysteresis/27_pairwise_freeze_19strand.py — 候補 (g3) Phase 2 19本実機検証.

[← README](README.md) | [← project README](../../README.md)

status-374 Phase 1 で実装した `PairwiseFreezingProcess` を status-375 Phase 2 で
NR ループに配線した上で、19 本撚線 90° 曲げに適用し、`flip_threshold ∈ {2,3,5}`
を掃引する。

**ベースライン（status-357 19 本 K_c FD 再計測時）**:

- frac=0.3739 / mat_only rel_err mean=0.508 / Type D+E:67%, E:28%

**Gate 基準（status-374 §引継ぎ）**: `frac ≥ 0.6`（baseline 0.3739 比 +60%）.

実行:
    uv run python work/beam_hysteresis/27_pairwise_freeze_19strand.py 3 2>&1 \\
        | tee /tmp/pairwise_freeze_19strand_$(date +%s).log
"""

from __future__ import annotations

import sys
import time

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    flip_threshold = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print("=" * 72)
    print(f"候補 (g3) pair-wise freeze 19本撚線 90° 曲げ — flip_threshold={flip_threshold}")
    print("=" * 72)
    print("ベースライン (status-357): frac=0.3739, Type D+E:67%, E:28%")

    cfg = StrandBendingOscillationConfig(
        n_strands=19,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.015,
        n_cycles=1,
        n_increments_per_cycle=20,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        penalty_exponent=1.5,
        smoothing_delta=1000.0,
        # ★ 候補 (g3) pair-wise freeze
        pairwise_freeze_enabled=True,
        pairwise_freeze_flip_threshold=flip_threshold,
        pairwise_freeze_skip_type_d=True,
    )

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = sr.load_history[-1] if sr.load_history else 0.0
    n_incr = int(sr.n_increments)
    n_cb = int(sr.n_cutbacks)
    converged = bool(sr.converged)

    print()
    print("─" * 72)
    print(
        f"flip_threshold={flip_threshold:>2}: frac={frac:.4f}, "
        f"incr={n_incr}, cb={n_cb}, elapsed={elapsed:.2f}s, converged={converged}"
    )
    base_frac = 0.3739
    delta_pct = (frac - base_frac) / base_frac * 100.0
    print(f"  ベースライン比: {delta_pct:+.1f}% (baseline frac=0.3739)")
    gate_passed = frac >= 0.6
    print(f"  Gate frac >= 0.6: {'PASS' if gate_passed else 'FAIL'}")
    print("=" * 72)

    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
