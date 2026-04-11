"""status-316/318: n_strands 掃引プロファイリング実測スクリプト.

status-315 で整備した `ParameterSweepBenchmarkProcess` ×
`StrandBendingOscillationProcess` を使い、n_strands を六角充填の系列
(7, 19, 37, 61, 91, 127) で掃引し、dominant Process（status-317 で追加した
`dominant_leaf_process`）の推移を実測する。

status-318 で `SWEEP_VALUES` を 3 ケース → 6 ケースに拡張し、
**TangentAssembly が LinearSolve を抜く転換点**の所在を明らかにする。
status-317 の `dominant_leaf_process` フィールドにより、wrapper（
`StrandBendingOscillationProcess` 等）でなく葉プロセスを直接読める。

[← README](../../README.md) | [← status-316](../../docs/status/status-316.md) | [← status-318](../../docs/status/status-318.md)

使い方::

    PYTHONPATH=. python work/strand_profiling/status316_nstrands_sweep.py \
        2>&1 | tee /tmp/log-$(date +%s).log

生成物::

    docs/benchmarks/ParameterSweepBenchmark_<timestamp>.yaml   # 集約サマリ
    docs/benchmarks/BenchmarkRun_<timestamp>[_NN].yaml         # 各ケース
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
# 設定
# --------------------------------------------------------------------
# 以下は実測時間と代表性のバランスを取った軽量構成:
#   - n_pitches = 0.25 で要素数をピッチあたり 16 に抑制
#   - n_increments_per_cycle = 4 で NR の代表的なワークロードを取得
#   - bending_curvature を小さめにして確実に収束
#   - 接触を有効化（dominant Process 測定の主目的）
#   - 被膜を無効化（status-302 で k_coat=1e6 は数値正則化と判明。
#     dominant process 測定には素線接触コアワークロードを優先）
# 100-1000 本の本格実測には計算リソース確保後、n_pitches や
# n_increments を拡張するだけで再現できる。
# --------------------------------------------------------------------

BASE = StrandBendingOscillationConfig(
    n_strands=7,  # ダミー（sweep で上書き）
    n_pitches=0.25,
    n_elements_per_pitch=16,
    n_increments_per_cycle=4,
    bending_curvature=0.0005,
    max_increments=200,
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
        status_file="docs/status/status-316.md",
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
        f"{'dominant_leaf_process':<32} {'leaf_pct[%]':>11} {'leaf_total[s]':>13}"
    )
    print("-" * 100)
    for row, case in zip(result.summary_rows, result.cases, strict=True):
        extras = case.manifest.results_summary
        ndof = extras.get("total_ndof", "?")
        n_inc = extras.get("n_increments", "?")
        print(
            f"{row['value']:>10} {ndof:>6} {n_inc:>4} "
            f"{row['elapsed_seconds']:>12.2f} "
            f"{row['dominant_leaf_process'][:32]:<32} "
            f"{row['dominant_leaf_pct']:>10.2f} "
            f"{row['dominant_leaf_total']:>13.3f}"
        )
    print("-" * 100)

    # 各ケースの profile_breakdown 上位 5 件を表示
    for value, case in zip(SWEEP_VALUES, result.cases, strict=True):
        print(f"\n[n_strands={value}] profile_breakdown (top 5):")
        for i, row in enumerate(case.manifest.profile_breakdown[:5], start=1):
            print(
                f"  {i}. {row.get('name', '?'):<40} "
                f"total={row.get('total', 0):.3f}s "
                f"n={row.get('n', 0):>5} "
                f"pct={row.get('pct', 0):.2f}%"
            )


if __name__ == "__main__":
    main()
