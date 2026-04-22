"""work/beam_hysteresis/19_hypothesis_c_backtracking_7strand.py

[← README](README.md) | [← project README](../../README.md)

status-362 仮説 C 候補 (c) line search 強化（接触残差 / active flip
backtracking）の 7 本撚線 90° 曲げ回帰チェック。

**ベースライン（status-359 採択設定）**:

- smoothing_delta=1000（2x 拡大）
- frac=1.0000, n_incr=475, n_cb=53, elapsed=259.92s

**本スクリプト（候補 (c)）**:

- 上記に `contact_backtracking_enabled=True` を追加
- mixed (C+D) 検知時のみ backtracking 発動（既定）
- 合否基準: **frac=1.0000 回帰なし + elapsed 増加 ≤ 20%**

7 本撚線は status-361 Type 分布で mixed=1.2% と稀（他は C/D 単独）なので、
backtracking は殆ど発動しない想定。発動頻度 / 採択 α のログで確認。

実行:
    uv run --extra dev python work/beam_hysteresis/19_hypothesis_c_backtracking_7strand.py \\
        2>&1 | tee /tmp/hypothesis_c_bt_7strand_$(date +%s).log
"""

from __future__ import annotations

import sys
import time

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    print("=" * 70)
    print("仮説 C 候補 (c): contact_backtracking_enabled=True (7本撚線 90°)")
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
        smoothing_delta=1000.0,  # status-359 採択設定
        # ★ 仮説 C 候補 (c): backtracking 有効化
        contact_backtracking_enabled=True,
        contact_backtracking_mixed_only=True,
        contact_backtracking_active_flip_threshold=3,
        contact_backtracking_residual_ratio=2.0,
        contact_backtracking_max_steps=4,
    )
    print(
        f"cfg: n_strands={cfg.n_strands}, κ_max={cfg.bending_curvature}, "
        f"smoothing_delta={cfg.smoothing_delta}, "
        f"backtracking={cfg.contact_backtracking_enabled} "
        f"(mixed_only={cfg.contact_backtracking_mixed_only}, "
        f"flip_th={cfg.contact_backtracking_active_flip_threshold}, "
        f"res_ratio={cfg.contact_backtracking_residual_ratio})"
    )
    print()

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0

    print()
    print("=" * 70)
    print("結果（仮説 C 候補 (c) 7本撚線）")
    print("=" * 70)
    print(f"  frac_completed: {frac:.4f}")
    print(f"  converged:      {sr.converged}")
    print(f"  n_increments:   {sr.n_increments}")
    print(f"  n_cutbacks:     {sr.n_cutbacks}")
    print(f"  elapsed:        {elapsed:.2f} s")
    print(f"  max |u|:        {float(np.max(np.abs(sr.u))):.6e}")

    print()
    print("=" * 70)
    print("status-359 ベースライン比較（smoothing_delta=1000 のみ）")
    print("=" * 70)
    base_frac, base_incr, base_cb, base_elapsed = 1.0000, 475, 53, 259.92
    print(f"  {'項目':<16s} {'baseline':>14s} {'候補(c)':>14s} {'Δ%':>10s}")
    print(f"  {'frac':<16s} {base_frac:>14.4f} {frac:>14.4f} {(frac - base_frac) * 100:>9.2f}%")
    print(
        f"  {'n_increments':<16s} {base_incr:>14d} {sr.n_increments:>14d} "
        f"{(sr.n_increments - base_incr) / base_incr * 100:>9.1f}%"
    )
    print(
        f"  {'n_cutbacks':<16s} {base_cb:>14d} {sr.n_cutbacks:>14d} "
        f"{(sr.n_cutbacks - base_cb) / base_cb * 100:>9.1f}%"
    )
    print(
        f"  {'elapsed [s]':<16s} {base_elapsed:>14.2f} {elapsed:>14.2f} "
        f"{(elapsed - base_elapsed) / base_elapsed * 100:>9.1f}%"
    )

    ok = frac >= 0.9999
    slow_cap = 1.20
    fast_ok = elapsed <= base_elapsed * slow_cap
    print()
    print(
        f"判定: frac=1.0 完走 {'OK' if ok else 'NG'}, "
        f"elapsed <= {slow_cap:.2f}x {'OK' if fast_ok else 'NG (許容超過)'}"
    )

    return 0 if sr.converged and ok else 1


if __name__ == "__main__":
    sys.exit(main())
