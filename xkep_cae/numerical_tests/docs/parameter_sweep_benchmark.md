# パラメータスイープベンチマーク

[← README](../../../README.md)

任意の frozen dataclass の1フィールドを掃引し、`BenchmarkRunnerProcess`
でそれぞれを実行して `profile_breakdown` を収集する汎用 Process。

1000 本撚線プロファイリング（status-314 TODO）の実測エントリポイントとし
て設計されている。

## ParameterSweepBenchmarkProcess

- **カテゴリ**: `BatchProcess`
- **入力**: `ParameterSweepBenchmarkInput`
- **出力**: `ParameterSweepBenchmarkResult`
- **uses**: `BenchmarkRunnerProcess`
- **document_path**: `docs/parameter_sweep_benchmark.md`

### 役割

`BenchmarkRunnerProcess` は 1 ケースの自動マニフェスト記録に責務を限定
している。本 Process はその上層に位置し、ひとつの `base_config` に対し
指定フィールド (`param_name`) を掃引値 (`param_values`) で差し替えて
繰り返し実行する。各走査の `profile_breakdown` を `ProcessMetaclass.
snapshot_profile()` のデルタとして単独に取得するため、ケース間で
プロファイル情報が混ざらない。

### パイプライン

1. 各 `value in param_values` について
   1. `dataclasses.replace(base_config, **{param_name: value})` で
      掃引後 config を生成
   2. `BenchmarkRunnerProcess` にラップして実行（プロファイルスナップ
      ショット付き）
   3. `BenchmarkRunResult` をそのまま蓄積
   4. サマリー行に `param_name / value / elapsed / dominant_process /
      dominant_pct / manifest_path` を追記
2. すべての掃引完了後、サマリー YAML を `output_dir` に保存
   (`ParameterSweepBenchmark_{timestamp}.yaml`)

### 入力 (`ParameterSweepBenchmarkInput`)

| フィールド | 型 | 意味 |
|------------|-----|------|
| `target_process` | `Any`（`AbstractProcess`）| 掃引対象のプロセス |
| `base_config` | `Any`（frozen dataclass）| ベースとなる config |
| `param_name` | `str` | 差し替え対象フィールド名 |
| `param_values` | `tuple[Any, ...]` | 差し替え値の列 |
| `result_extractors` | `dict[str, Callable]` | `BenchmarkRunInput` へ渡す結果抽出関数 |
| `output_dir` | `str \| None` | 個別/集約マニフェストの保存先 |
| `status_file` | `str \| None` | status ドキュメントリンク |
| `profile_sort_by` | `str` | `BenchmarkRunInput.profile_sort_by` と同じ |
| `profile_top_n` | `int \| None` | `BenchmarkRunInput.profile_top_n` と同じ |

### 出力 (`ParameterSweepBenchmarkResult`)

| フィールド | 型 | 意味 |
|------------|-----|------|
| `param_name` | `str` | 掃引フィールド名 |
| `param_values` | `tuple[Any, ...]` | 掃引した値（入力のコピー） |
| `cases` | `tuple[BenchmarkRunResult, ...]` | 1 ケース 1 エントリ |
| `summary_rows` | `tuple[dict[str, Any], ...]` | 集約サマリー（YAML 保存に使用） |
| `summary_yaml_path` | `str \| None` | 集約 YAML の保存先 |

### サマリー YAML 例

```yaml
process: ParameterSweepBenchmarkProcess
param_name: n_strands
param_values:
  - 7
  - 19
cases:
  -
    n_strands: 7
    elapsed_seconds: 215.3
    dominant_process: ContactFrictionProcess
    dominant_pct: 81.4
    manifest_path: docs/benchmarks/StrandBendingOscillationProcess_20260410T....yaml
  -
    n_strands: 19
    elapsed_seconds: 1104.8
    dominant_process: ContactFrictionProcess
    dominant_pct: 86.2
    manifest_path: docs/benchmarks/StrandBendingOscillationProcess_20260410T....yaml
```

### 使い方（1000 本撚線プロファイリング）

```python
from xkep_cae.numerical_tests.parameter_sweep_benchmark import (
    ParameterSweepBenchmarkInput,
    ParameterSweepBenchmarkProcess,
)
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)

base = StrandBendingOscillationConfig(
    n_pitches=1.0,
    n_increments_per_cycle=20,
    bending_curvature=0.001,
    coating_stiffness=1.0e6,
    coating_barrier=True,
)

sweep = ParameterSweepBenchmarkProcess().process(
    ParameterSweepBenchmarkInput(
        target_process=StrandBendingOscillationProcess(),
        base_config=base,
        param_name="n_strands",
        param_values=(7, 19, 37),
        result_extractors={
            "n_incr": lambda r: r.solver_result.n_increments,
            "converged": lambda r: r.solver_result.converged,
        },
        status_file="docs/status/status-315.md",
        profile_top_n=10,
    )
)

for row in sweep.summary_rows:
    print(row)
```

### status-314 連携

本 Process はその実装に `BenchmarkRunnerProcess`（status-314 で
`profile_breakdown` を自動記録するように強化済み）しか使用しない。
既存の一発実行テストをそのまま掃引化できる薄い層として設計されている。

### status-315

ParameterSweepBenchmarkProcess 新規実装。
