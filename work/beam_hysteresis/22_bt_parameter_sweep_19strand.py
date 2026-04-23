"""work/beam_hysteresis/22_bt_parameter_sweep_19strand.py

[← README](README.md) | [← project README](../../README.md)

status-363 仮説 C 候補 (c) ContactBacktrackingLineSearchProcess の 19 本撚線
90° 曲げ **パラメータ感度掃引**（status-362 引継ぎ 1.）。

status-362 で backtracking default 設定（rate_threshold=0.85 /
active_flip_ratio=0.3 / mixed_only=True）が 19 本撚線で frac=0.5153 止まり
（+6.5% 改善、未完走）だったことを受け、以下 3 パラメータの単独・組み合わせ
掃引で frac=1.0 完走の working point を探索する。

**掃引軸**:

| case | rate_threshold | active_flip_ratio | mixed_only | 意図 |
|------|---|---|---|---|
| A: relaxed_trigger       | 0.70 | 0.30 | True  | D.slow 領域も BT 発動、発動数増加 |
| B: strict_flip           | 0.85 | 0.15 | True  | flip 許容量半減、mixed 強抑制 |
| C: always_on             | 0.85 | 0.30 | False | 全 NR 反復で BT チェック、コスト 2x |
| D: combined (A+B+C)      | 0.70 | 0.15 | False | 3 パラメータ全部緩和/強化 |

全ケースで `smoothing_delta` は default（自動 2000）、
`contact_backtracking_enabled=True`、`max_increments=1500`（safe cap）。

**ベースライン**:
- status-339: backtracking なし → frac=0.4839, incr=271, cb=39, 534.68s
- status-362: backtracking default → frac=0.5153, incr=318, cb=38, 729.36s

**合否基準**: frac=1.0 完走で MCDD 凍結解除条件達成。elapsed は許容上限
なし（参考指標）。

実行:
    uv run --extra dev python work/beam_hysteresis/22_bt_parameter_sweep_19strand.py \\
        2>&1 | tee /tmp/bt_sweep_19strand_$(date +%s).log
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


@dataclass(frozen=True)
class SweepCase:
    """1 ケースのパラメータ + 結果."""

    label: str
    description: str
    rate_threshold: float
    active_flip_ratio: float
    mixed_only: bool


@dataclass
class SweepResult:
    case: SweepCase
    frac: float
    converged: bool
    n_increments: int
    n_cutbacks: int
    elapsed: float


def _run_case(case: SweepCase) -> SweepResult:
    print("=" * 70)
    print(f"Case {case.label}: {case.description}")
    print(
        f"  rate_threshold={case.rate_threshold}, "
        f"active_flip_ratio={case.active_flip_ratio}, "
        f"mixed_only={case.mixed_only}"
    )
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
        bending_curvature=0.015,
        n_cycles=1,
        n_increments_per_cycle=20,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=1500,
        exclude_same_strand=True,
        free_end_mode=True,
        penalty_exponent=1.5,
        # ★ BT sweep axes
        contact_backtracking_enabled=True,
        contact_backtracking_rate_threshold=case.rate_threshold,
        contact_backtracking_active_flip_ratio=case.active_flip_ratio,
        contact_backtracking_mixed_only=case.mixed_only,
        contact_backtracking_active_flip_threshold=3,
        contact_backtracking_residual_ratio=2.0,
        contact_backtracking_max_steps=4,
        track_contact_pairs=False,
    )
    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0

    print(
        f"  → frac={frac:.4f} converged={sr.converged} "
        f"incr={sr.n_increments} cb={sr.n_cutbacks} elapsed={elapsed:.2f}s "
        f"max|u|={float(np.max(np.abs(sr.u))):.4e}"
    )
    print()

    return SweepResult(
        case=case,
        frac=frac,
        converged=sr.converged,
        n_increments=sr.n_increments,
        n_cutbacks=sr.n_cutbacks,
        elapsed=elapsed,
    )


def main() -> int:
    print("=" * 70)
    print("仮説 C 候補 (c) パラメータ感度掃引 (19本撚線 90°)")
    print("=" * 70)
    print()

    cases = (
        SweepCase(
            label="A",
            description="relaxed_trigger (rate_threshold=0.70)",
            rate_threshold=0.70,
            active_flip_ratio=0.30,
            mixed_only=True,
        ),
        SweepCase(
            label="B",
            description="strict_flip (active_flip_ratio=0.15)",
            rate_threshold=0.85,
            active_flip_ratio=0.15,
            mixed_only=True,
        ),
        SweepCase(
            label="C",
            description="always_on (mixed_only=False)",
            rate_threshold=0.85,
            active_flip_ratio=0.30,
            mixed_only=False,
        ),
        SweepCase(
            label="D",
            description="combined (A+B+C)",
            rate_threshold=0.70,
            active_flip_ratio=0.15,
            mixed_only=False,
        ),
    )

    results: list[SweepResult] = []
    for case in cases:
        r = _run_case(case)
        results.append(r)

    # サマリ表
    print("=" * 70)
    print("掃引結果サマリ（baseline 参考値: status-339/362）")
    print("=" * 70)
    base_frac, base_incr, base_cb, base_elapsed = 0.4839, 271, 39, 534.68
    bt_frac, bt_incr, bt_cb, bt_elapsed = 0.5153, 318, 38, 729.36
    print(
        f"  {'case':<6s} {'frac':>8s} {'incr':>6s} {'cb':>4s} "
        f"{'elapsed[s]':>12s} {'Δfrac%base':>12s} {'Δfrac%bt':>10s}"
    )
    print(
        f"  {'noBT':<6s} {base_frac:>8.4f} {base_incr:>6d} {base_cb:>4d} "
        f"{base_elapsed:>12.2f} {'--':>12s} {'--':>10s}"
    )
    print(
        f"  {'BTdef':<6s} {bt_frac:>8.4f} {bt_incr:>6d} {bt_cb:>4d} "
        f"{bt_elapsed:>12.2f} "
        f"{(bt_frac - base_frac) / base_frac * 100:>11.1f}% {'--':>10s}"
    )
    for r in results:
        d_base = (r.frac - base_frac) / base_frac * 100
        d_bt = (r.frac - bt_frac) / bt_frac * 100
        print(
            f"  {r.case.label:<6s} {r.frac:>8.4f} {r.n_increments:>6d} {r.n_cutbacks:>4d} "
            f"{r.elapsed:>12.2f} {d_base:>11.1f}% {d_bt:>9.1f}%"
        )
    print()

    winners = [r for r in results if r.frac >= 0.9999]
    if winners:
        best = min(winners, key=lambda r: r.elapsed)
        print(
            f"判定: **frac=1.0 完走** 達成ケースあり。最速 = Case {best.case.label} "
            f"({best.case.description}), elapsed={best.elapsed:.2f}s"
        )
        print("→ MCDD 凍結解除条件達成。次 status で default 変更議論。")
        return 0

    best = max(results, key=lambda r: r.frac)
    print(
        f"判定: 全ケース frac<1.0 未完走。best = Case {best.case.label} "
        f"(frac={best.frac:.4f}, elapsed={best.elapsed:.2f}s)。"
    )
    print(
        "→ MCDD 凍結解除条件 **未達**。次は候補 (d) 接触凍結モード 19 本適用 or "
        "候補 (e) 接触減衰 escape hatch を検討。"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
