"""status-319: 初期ギャップ固定 + 大曲率でのバイアス補正スイープ.

status-318 の実測条件は以下のバイアスを抱えていた:

1. **gap 自動補正の n_strands 依存**: `gap=0.0`（デフォルト）指定時に
   `_compute_min_safe_gap` が n_strands ごとに異なる安全ギャップへ自動補正。
   実測値:

     n=7..61: 0.031769 mm (gap/r=0.0635)
     n=91:    0.035248 mm (gap/r=0.0705)
     n=127:   0.069946 mm (gap/r=0.1399)  ← **約 2.2 倍**

   これにより n=127 では初期クリアランスが 2 倍以上となり、接触活性化の
   度合いが他ケースと等価ではない。

2. **曲率が小さすぎる**: status-318 では `bending_curvature=0.0005 /mm`。
   L=25mm のビームで曲げ角 = 0.7162°。90° ターゲットの **0.8%** しか
   曲げていない。これだと接触活性ペアがほとんどなく、Type D stall + NR
   stall の系外評価となり、**コアの接触アセンブリが dominant にならない**。

3. **n_increments_per_cycle=4**: カットバック余裕ゼロ。収束パスが不安定。

本 status の目的:

- **gap=0.07 mm**（全ケース固定）: n=127 の min_safe_gap 0.069946 を
  わずかに上回る値で統一。レイアウト次元（≒ 斜め交差距離）の影響を
  排除し、**初期クリアランスを n_strands 非依存**にする。
- **bending_curvature=0.005 /mm**: L=25mm で曲げ角 7.16°（90° の 8%）。
  status-318 の 10 倍、work/beam_hysteresis と同じ値。最大たわみ
  L²/8 × κ = 0.391 mm で gap=0.07 の 5.6 倍 → **接触は確実に活性化**する。
- **n_increments_per_cycle=10**: status-318 の 2.5 倍でカットバック余裕を
  確保。
- **同じ 6 ケース（7/19/37/61/91/127）**: status-318 と直接比較可能。

[← README](../../README.md) | [← status-318](../../docs/status/status-318.md) | [← status-319](../../docs/status/status-319.md)

使い方::

    PYTHONPATH=. python work/strand_profiling/status319_corrected_sweep.py \\
        2>&1 | tee /tmp/log-status319-$(date +%s).log
"""

from __future__ import annotations

import time

from xkep_cae.numerical_tests.parameter_sweep_benchmark import (
    ParameterSweepBenchmarkInput,
    ParameterSweepBenchmarkProcess,
)
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)

# --------------------------------------------------------------------
# 設定（status-318 との差分を明示）
# --------------------------------------------------------------------
# status-318: gap=0.0 (auto), bending_curvature=0.0005, n_inc=4
# status-319: gap=0.07 (fixed), bending_curvature=0.005, n_inc=10
# --------------------------------------------------------------------

BASE = StrandBendingOscillationConfig(
    n_strands=7,  # ダミー（sweep で上書き）
    n_pitches=0.25,
    n_elements_per_pitch=16,
    gap=0.07,  # ★status-319: 全ケース固定（n=127 の min_safe_gap 0.069946 > 0.07 にならないことは事前検証済み）
    n_increments_per_cycle=10,  # ★status-319: 4 → 10
    bending_curvature=0.005,  # ★status-319: 0.0005 → 0.005（10x、曲げ角 0.72°→7.16°）
    max_increments=400,  # カットバック余裕を増やす
    contact_enabled=True,
    coating_stiffness=0.0,
    coating_barrier=False,
)

SWEEP_VALUES: tuple[int, ...] = (7, 19, 37, 61, 91, 127)


def main() -> None:
    sweep_input = ParameterSweepBenchmarkInput(
        target_process=StrandBendingOscillationProcess(),
        base_config=BASE,
        param_name="n_strands",
        param_values=SWEEP_VALUES,
        result_extractors={
            "n_increments": lambda r: r.solver_result.n_increments,
            "converged": lambda r: bool(r.solver_result.converged),
            "total_ndof": lambda r: int(r.total_ndof),
        },
        status_file="docs/status/status-319.md",
        output_dir="docs/benchmarks",
        profile_top_n=10,
    )

    t0 = time.time()
    result = ParameterSweepBenchmarkProcess().process(sweep_input)
    elapsed_total = time.time() - t0

    print("=" * 100)
    print(f"ParameterSweepBenchmark 完了: 総実行時間 = {elapsed_total:.2f} s")
    print(f"集約サマリ YAML: {result.summary_yaml_path}")
    print("-" * 100)
    print(
        f"{'n_strands':>10} {'ndof':>6} {'inc':>4} {'elapsed[s]':>12} "
        f"{'dominant_leaf_process':<36} {'leaf_pct[%]':>11} {'leaf_total[s]':>13}"
    )
    print("-" * 100)
    for row, case in zip(result.summary_rows, result.cases, strict=True):
        extras = case.manifest.results_summary
        ndof = extras.get("total_ndof", "?")
        n_inc = extras.get("n_increments", "?")
        print(
            f"{row['value']:>10} {ndof:>6} {n_inc:>4} "
            f"{row['elapsed_seconds']:>12.2f} "
            f"{row['dominant_leaf_process'][:36]:<36} "
            f"{row['dominant_leaf_pct']:>10.2f} "
            f"{row['dominant_leaf_total']:>13.3f}"
        )
    print("-" * 100)

    for value, case in zip(SWEEP_VALUES, result.cases, strict=True):
        print(f"\n[n_strands={value}] profile_breakdown (top 7):")
        for i, row in enumerate(case.manifest.profile_breakdown[:7], start=1):
            print(
                f"  {i}. {row.get('name', '?'):<44} "
                f"total={row.get('total', 0):.3f}s "
                f"n={row.get('n', 0):>6} "
                f"pct={row.get('pct', 0):.2f}%"
            )


if __name__ == "__main__":
    main()
