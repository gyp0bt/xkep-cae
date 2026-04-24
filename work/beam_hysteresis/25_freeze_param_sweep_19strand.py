"""work/beam_hysteresis/25_freeze_param_sweep_19strand.py

[← README](README.md) | [← project README](../../README.md)

status-368 候補 (d) **接触凍結モード 19 本撚線再評価**（status-367 引継ぎ 1.）。

status-284 で 7 本撚線にて frac 0.40→0.70 を達成したチャタリング検知→
接触凍結モード（`chattering_freeze_*`）の既定パラメータは 7 本向けの調整。
19 本 Type D stall 本体（status-339: frac=0.4839 / status-362: BT=0.5153）
に対し、3 パラメータの感度掃引で MCDD 凍結解除条件（frac=1.0 完走）達成可否
を実測する。

**掃引軸**:

| case | max_cycles | nr_max | tol_factor | 意図 |
|------|---|---|---|---|
| default | 5  | 15 | 10.0  | status-284 既定（7本用、比較基準） |
| A: more_cycles | 10 | 15 | 10.0  | 凍結サイクル数倍増、収束待ち拡張 |
| B: longer_nr   | 5  | 30 | 10.0  | 凍結中NR上限倍増、内部収束深掘り |
| C: loose_tol   | 5  | 15 | 100.0 | 凍結tol 10x緩和、phys解即抜け出し |
| D: combined    | 10 | 30 | 100.0 | A+B+C 全緩和 |
| E: disabled    | -  | -  | -     | chattering_freeze_enabled=False（比較） |

全ケースで `smoothing_delta` は自動（1000/r=2000）、`contact_backtracking`
は default OFF、`max_increments=1500`（safe cap）。

**ベースライン**:
- status-339: backtracking なし + freeze default → frac=0.4839, 271 incr, 39 cb, 535s
- status-362: backtracking default + freeze default → frac=0.5153, 318 incr, 38 cb, 729s

**合否基準**: frac=1.0 完走で MCDD 凍結解除条件達成。elapsed は参考指標。

実行:
    uv run --extra dev python work/beam_hysteresis/25_freeze_param_sweep_19strand.py \\
        2>&1 | tee /tmp/freeze_sweep_19strand_$(date +%s).log
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
    """1 ケースのパラメータ."""

    label: str
    description: str
    enabled: bool
    max_cycles: int
    nr_max: int
    tol_factor: float


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
        f"  enabled={case.enabled}, max_cycles={case.max_cycles}, "
        f"nr_max={case.nr_max}, tol_factor={case.tol_factor}"
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
        # ★ freeze sweep axes
        chattering_freeze_enabled=case.enabled,
        chattering_freeze_max_cycles=case.max_cycles,
        chattering_freeze_nr_max=case.nr_max,
        chattering_freeze_tol_factor=case.tol_factor,
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
    print("候補 (d) chattering_freeze パラメータ感度掃引 (19本撚線 90°)")
    print("=" * 70)
    print()

    cases = (
        SweepCase(
            label="default",
            description="status-284 既定（7本用）",
            enabled=True,
            max_cycles=5,
            nr_max=15,
            tol_factor=10.0,
        ),
        SweepCase(
            label="A",
            description="more_cycles (max_cycles=10)",
            enabled=True,
            max_cycles=10,
            nr_max=15,
            tol_factor=10.0,
        ),
        SweepCase(
            label="B",
            description="longer_nr (nr_max=30)",
            enabled=True,
            max_cycles=5,
            nr_max=30,
            tol_factor=10.0,
        ),
        SweepCase(
            label="C",
            description="loose_tol (tol_factor=100)",
            enabled=True,
            max_cycles=5,
            nr_max=15,
            tol_factor=100.0,
        ),
        SweepCase(
            label="D",
            description="combined (max=10, nr=30, tol=100)",
            enabled=True,
            max_cycles=10,
            nr_max=30,
            tol_factor=100.0,
        ),
        SweepCase(
            label="E",
            description="disabled (chattering_freeze_enabled=False)",
            enabled=False,
            max_cycles=5,
            nr_max=15,
            tol_factor=10.0,
        ),
    )

    results: list[SweepResult] = []
    for case in cases:
        r = _run_case(case)
        results.append(r)

    # サマリ表
    print("=" * 70)
    print("掃引結果サマリ（baseline 参考値: status-339）")
    print("=" * 70)
    base_frac, base_incr, base_cb, base_elapsed = 0.4839, 271, 39, 534.68
    print(
        f"  {'case':<10s} {'frac':>8s} {'incr':>6s} {'cb':>4s} "
        f"{'elapsed[s]':>12s} {'Δfrac%base':>12s}"
    )
    print(
        f"  {'status-339':<10s} {base_frac:>8.4f} {base_incr:>6d} {base_cb:>4d} "
        f"{base_elapsed:>12.2f} {'--':>12s}"
    )
    for r in results:
        d_base = (r.frac - base_frac) / base_frac * 100
        print(
            f"  {r.case.label:<10s} {r.frac:>8.4f} {r.n_increments:>6d} {r.n_cutbacks:>4d} "
            f"{r.elapsed:>12.2f} {d_base:>11.1f}%"
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
        "→ MCDD 凍結解除条件 **未達**。候補 (d) クローズ、次候補 (f) Phase C-3' "
        "s-tracking 19 本再評価を検討。"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
