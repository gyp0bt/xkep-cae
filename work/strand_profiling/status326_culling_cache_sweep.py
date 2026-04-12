"""status-326: culling + cache 効果定量計測スイープ.

status-319 と同一条件で n=7/19/37 掃引を実行し、
status-324 の K_st distance culling と status-325 の symbolic factorization
reuse の per-call 時間への効果を計測する。

status-319 ベースライン（culling/cache なし）:
  n=7:  TangentAssembly=110.5ms, ContactForceSt=29.4ms, FrictionSt=31.3ms
  n=19: TangentAssembly=170.9ms, ContactForceSt=51.7ms, FrictionSt=51.3ms
  n=37: TangentAssembly=512.7ms, ContactForceSt=204.9ms, FrictionSt=199.9ms

注意: pypardiso 未インストール時は cache 効果は scipy fallback で計測不可。
culling 効果のみ計測可能。

[← README](../../README.md) | [← status-319](../../docs/status/status-319.md)

使い方::

    PYTHONPATH=. python work/strand_profiling/status326_culling_cache_sweep.py \
        2>&1 | tee /tmp/log-status326-$(date +%s).log
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
# 設定（status-319 と完全同一条件）
# --------------------------------------------------------------------
BASE = StrandBendingOscillationConfig(
    n_strands=7,  # ダミー（sweep で上書き）
    n_pitches=0.25,
    n_elements_per_pitch=16,
    gap=0.07,  # status-319: 全ケース固定
    n_increments_per_cycle=10,  # status-319: 10
    bending_curvature=0.005,  # status-319: 0.005（曲げ角 7.16°）
    max_increments=400,
    contact_enabled=True,
    coating_stiffness=0.0,
    coating_barrier=False,
)

# status-319 で実測できた 3 ケースのみ（n=61+ は Type D stall で中断）
SWEEP_VALUES: tuple[int, ...] = (7, 19, 37)

# status-319 ベースライン（比較用）
BASELINE_319 = {
    7: {
        "TangentAssembly": 110.5,
        "ContactForceStStiffness": 29.4,
        "FrictionStStiffness": 31.3,
    },
    19: {
        "TangentAssembly": 170.9,
        "ContactForceStStiffness": 51.7,
        "FrictionStStiffness": 51.3,
    },
    37: {
        "TangentAssembly": 512.7,
        "ContactForceStStiffness": 204.9,
        "FrictionStStiffness": 199.9,
    },
}


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
        status_file="docs/status/status-326.md",
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

    # per-call 時間をプロファイルから抽出して status-319 と比較
    print("\n" + "=" * 100)
    print("status-319 ベースラインとの比較（per-call avg [ms]）")
    print("=" * 100)
    target_procs = [
        "TangentAssembly",
        "ContactForceStStiffness",
        "FrictionStStiffness",
    ]
    print(
        f"{'n_strands':>10} {'Process':<30} "
        f"{'s319[ms]':>10} {'s326[ms]':>10} {'ratio':>8} {'改善':>8}"
    )
    print("-" * 100)
    for value, case in zip(SWEEP_VALUES, result.cases, strict=True):
        for proc_name in target_procs:
            # profile_breakdown からプロセス名で検索
            s326_avg = None
            for row in case.manifest.profile_breakdown:
                name = row.get("name", "")
                if proc_name in name:
                    total = row.get("total", 0)
                    n_calls = row.get("n", 1)
                    s326_avg = total / max(n_calls, 1) * 1000  # ms
                    break
            s319_avg = BASELINE_319.get(value, {}).get(proc_name)
            if s326_avg is not None and s319_avg is not None:
                ratio = s326_avg / s319_avg
                improvement = (1 - ratio) * 100
                print(
                    f"{value:>10} {proc_name:<30} "
                    f"{s319_avg:>10.1f} {s326_avg:>10.1f} "
                    f"{ratio:>7.2f}x {improvement:>+7.1f}%"
                )
            elif s326_avg is not None:
                print(
                    f"{value:>10} {proc_name:<30} "
                    f"{'N/A':>10} {s326_avg:>10.1f} {'N/A':>8} {'N/A':>8}"
                )

    # 個別プロファイル
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
