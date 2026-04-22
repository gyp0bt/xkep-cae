"""work/beam_hysteresis/18_light_load_19strand.py — 19本撚線軽荷重 Type 分布切り分け.

[← README](README.md) | [← project README](../../README.md)

status-361: 7 本/19 本の挙動反転を数値的に切り分けるため、19 本撚線に軽荷重
（`bending_curvature=0.005`、90° ではなく 30° 相当）を与えて接触活性化密度を
下げた状態で Type 分布を取得する。

仮説:
- 19 本 κ=0.015（90°）で Type D 系 44% 優位 → 接触密度が増えると K_c x/z
  カップリング不整合（status-344, mat_only rel_err 44%）が前面化
- κ=0.005（30°）なら接触ペア数・active 集合が小さくなり、Type D 発火が
  抑制されて frac=1.0 完走できるはず
- 完走 → 19 本の「挙動反転」は**接触密度に依存する数値的問題**（幾何モデル
  自体は正常）
- 未完走 → 幾何モデル自体（多層 pitch 同一 / 非等ヘリックス角設計）が
  寄与している可能性を更に追跡

実行:
    uv run python work/beam_hysteresis/18_light_load_19strand.py 2>&1 \\
        | tee /tmp/light_load_19strand_$(date +%s).log
"""

from __future__ import annotations

import sys
import time

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    """19本撚線軽荷重 (κ=0.005) で Type 分布観測."""
    print("=" * 70)
    print("19本撚線 軽荷重 (κ=0.005 / 30° 相当、default smoothing_delta)")
    print("=" * 70)

    cfg = StrandBendingOscillationConfig(
        n_strands=19,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.005,  # ★ 軽荷重 (κ=0.015 の 1/3)
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
        track_contact_mk=False,
        track_contact_pairs=False,
    )
    print(
        f"cfg: n_strands={cfg.n_strands}, κ_max={cfg.bending_curvature} (軽荷重), "
        f"mu={cfg.mu}, smoothing_delta auto=2000"
    )
    print()

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = sr.load_history[-1] if sr.load_history else 0.0
    print()
    print("=" * 70)
    print("ソルバー結果（19本軽荷重）")
    print("=" * 70)
    print(f"  frac_completed: {frac:.4f}")
    print(f"  converged:      {sr.converged}")
    print(f"  n_increments:   {sr.n_increments}")
    print(f"  n_cutbacks:     {sr.n_cutbacks}")
    print(f"  elapsed:        {elapsed:.2f} s")
    print()
    print("=" * 70)
    print("判定指針（status-361）")
    print("=" * 70)
    print("  frac=1.0 完走 → 接触密度依存の数値問題（幾何モデル自体は正常）")
    print("  未完走 → 19 本幾何モデル自体（多層同一 pitch / 非等ヘリックス角）が影響")
    return 0 if sr.converged else 1


if __name__ == "__main__":
    sys.exit(main())
