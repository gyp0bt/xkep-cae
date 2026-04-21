"""work/beam_hysteresis/15_hypothesis_c_7strand.py — 仮説 C 候補 (a) smoothing_delta 拡大検証.

[← README](README.md) | [← project README](../../README.md)

status-358 で立案した仮説 C（active 集合振動対策）候補 (a)「smoothing_delta
遷移帯広げ」を 7本撚線 90°曲げで実測検証する。

**ベースライン（status-358 実測、09_kcr_measurement_7strand.py）**:

- smoothing_delta 自動 = 1000/r = 2000（default）→ δ_h = k_pen / 2000
- frac=1.0000, incr=524, cutback=57, 452s
- 166 チャタリング/Type D 事象

**本スクリプト（候補 (a)）**:

- smoothing_delta=500（手動指定、4x 拡大）→ δ_h = k_pen / 500 = 4x 広い
- Huber 遷移帯が 4 倍広くなり active 集合振動領域で ∂f_c/∂g の勾配が緩やか。
- active flip に伴う接線不連続が低減し Type D stall の発火頻度が下がる見込み。

**合否判定**: cutback / elapsed / chattering 事象数の **10% 以上**改善で採択。
10% 未満は誤差扱いで revert（ユーザー指示）。

実行:
    uv run python work/beam_hysteresis/15_hypothesis_c_7strand.py 2>&1 \\
        | tee /tmp/hypothesis_c_a_7strand_$(date +%s).log
"""

from __future__ import annotations

import sys
import time

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    """7本撚線 90° 曲げ、仮説 C (a) smoothing_delta 拡大."""
    print("=" * 70)
    print("仮説 C 候補 (a): smoothing_delta 拡大（7本撚線 90°曲げ）")
    print("=" * 70)

    cfg = StrandBendingOscillationConfig(
        n_strands=7,
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
        # ★ 仮説 C 候補 (a): smoothing_delta = 500（default 2000 の 1/4、δ_h 4x 拡大）
        smoothing_delta=500.0,
        track_contact_mk=False,
        track_contact_pairs=False,
    )
    print(
        f"cfg: n_strands={cfg.n_strands}, n_pitches={cfg.n_pitches}, "
        f"κ_max={cfg.bending_curvature}, mu={cfg.mu}, "
        f"smoothing_delta={cfg.smoothing_delta} (baseline=2000), "
        f"n_incr_per_cycle={cfg.n_increments_per_cycle}"
    )
    print()

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = sr.load_history[-1] if sr.load_history else 0.0
    print()
    print("=" * 70)
    print("ソルバー結果（仮説 C (a)）")
    print("=" * 70)
    print(f"  frac_completed: {frac:.4f}")
    print(f"  converged:      {sr.converged}")
    print(f"  n_increments:   {sr.n_increments}")
    print(f"  n_cutbacks:     {sr.n_cutbacks}")
    print(f"  elapsed:        {elapsed:.2f} s")

    print()
    print("=" * 70)
    print("ベースライン比較（status-358 実測値）")
    print("=" * 70)
    print("  baseline: frac=1.0000, incr=524, cb=57, 452.02s, chatter_events=166")
    print(
        f"  current : frac={frac:.4f}, incr={sr.n_increments}, cb={sr.n_cutbacks}, {elapsed:.2f}s"
    )
    # 10% 判定
    base_cb = 57
    base_time = 452.02
    cb_delta = (base_cb - sr.n_cutbacks) / base_cb * 100.0
    time_delta = (base_time - elapsed) / base_time * 100.0
    print(f"  cutback 改善: {cb_delta:+.1f}%  (10% 以上で採択)")
    print(f"  elapsed 改善: {time_delta:+.1f}%  (10% 以上で採択)")
    print()
    print("=" * 70)
    return 0 if sr.converged else 1


if __name__ == "__main__":
    sys.exit(main())
