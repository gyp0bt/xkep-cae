"""work/beam_hysteresis/15_hypothesis_c_7strand.py — 仮説 C 候補 (a') smoothing_delta 拡大検証.

[← README](README.md) | [← project README](../../README.md)

status-358 で立案した仮説 C（active 集合振動対策）の中間値再試行（status-359 実施）。
候補 (a) `smoothing_delta=500`（4x 拡大）は status-358 で frac=0.9241 未完走で却下。
本スクリプト（候補 (a')）は **2x 拡大（中間値）** に切替えて再試行。

**ベースライン（status-358 実測、09_kcr_measurement_7strand.py）**:

- smoothing_delta 自動 = 1000/r = 2000（default）→ δ_h = k_pen / 2000
- frac=1.0000, incr=524, cutback=57, 452.02s

**本スクリプト（候補 (a'）、status-359 実測結果）**:

- smoothing_delta=1000（手動指定、2x 拡大）→ δ_h = k_pen / 1000 = 2x 広い
- **frac=1.0000, incr=475, cutback=53, 259.92s**
- 比較: cutback -7.0%（10% 未満）/ elapsed **-42.5%（1.74x 高速化）**
- 判定: **採択方向（frac=1.0 完走 + elapsed 大幅改善）**。
  cutback は 10% 未達だが、elapsed の半減近い改善は active flip 抑制で
  各 increment の NR 反復数が減った効果で、対策効果としては十分。

**default 化の留保**: 7本撚線のみの検証のため、`StrandBendingOscillationConfig
.smoothing_delta` の default 値変更（2000→1000）は status-359 では実施せず、
19 本撚線（Type D stall 本体）での再検証を次セッションへ委ねる。
本スクリプトは **成功実験記録**として残置（status-358 の (a) 失敗実験と対称）。

実行:
    uv run python work/beam_hysteresis/15_hypothesis_c_7strand.py 2>&1 \\
        | tee /tmp/hypothesis_c_aprime_7strand_$(date +%s).log
"""

from __future__ import annotations

import sys
import time

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    """7本撚線 90° 曲げ、仮説 C (a') smoothing_delta 中間値再試行."""
    print("=" * 70)
    print("仮説 C 候補 (a'): smoothing_delta=1000 (2x 拡大、7本撚線 90°曲げ)")
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
        # ★ 仮説 C 候補 (a'): smoothing_delta = 1000（default 2000 の 1/2、δ_h 2x 拡大）
        smoothing_delta=1000.0,
        track_contact_mk=False,
        track_contact_pairs=False,
    )
    print(
        f"cfg: n_strands={cfg.n_strands}, n_pitches={cfg.n_pitches}, "
        f"κ_max={cfg.bending_curvature}, mu={cfg.mu}, "
        f"smoothing_delta={cfg.smoothing_delta} (baseline=2000, candidate a was 500), "
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
