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

### サマリー行のキー

各 `summary_rows` エントリは次のキーを持つ（`ParameterSweepBenchmarkResult.summary_rows`
と集約 YAML の両方で同じスキーマ）:

| キー | 内容 |
|------|------|
| `param_name` | 掃引フィールド名 |
| `value` | そのケースの差し替え値（`_scalarize` でプリミティブ化） |
| `elapsed_seconds` | BenchmarkRunnerProcess が計測したケース全体秒 |
| `dominant_process` | profile_breakdown 先頭（inclusive 時間最大。wrapper 込み） |
| `dominant_pct` | `dominant_process` の pct |
| `dominant_leaf_process` | **status-317 追加** — `uses` が空の葉プロセスの先頭。真のボトルネック |
| `dominant_leaf_pct` | `dominant_leaf_process` の pct |
| `dominant_leaf_total` | `dominant_leaf_process` の total 秒 |
| `manifest_path` | そのケースの個別 manifest YAML パス |

### `dominant_process` と `dominant_leaf_process` の違い（status-317）

`ProcessMetaclass._profile_data` は各 Process の *inclusive* 時間（ネストした
子プロセスの時間も含む壁時計）を記録する。このため `StrandBendingOscillationProcess`
→ `ContactFrictionProcess` → `NewtonDynamicProcess` のように wrapper が
1:1 で子を呼び出す階層では、各層が同じ elapsed を記録して breakdown 先頭を
占めてしまう（status-316 n=37 ケースで 3 wrapper が ~25% ずつ並んだ現象）。

`dominant_leaf_process` は `target_process` の `uses` グラフを再帰走査し、
`uses` が空のクラスを「葉」として先頭から抽出することで、wrapper 占有を
読み飛ばして本当にコストを使っている Process を指す。レジストリに依存せず
static に判定するため、`_skip_registry=True` のテストフィクスチャでも機能する。

### サマリー YAML 例

```yaml
process: ParameterSweepBenchmarkProcess
param_name: n_strands
param_values:
  - 7
  - 19
cases:
  -
    param_name: n_strands
    value: 7
    elapsed_seconds: 22.39
    dominant_process: StrandBendingOscillationProcess
    dominant_pct: 25.1
    dominant_leaf_process: LinearSolve
    dominant_leaf_pct: 22.5
    dominant_leaf_total: 20.08
    manifest_path: docs/benchmarks/StrandBendingOscillationProcess_20260410T....yaml
  -
    param_name: n_strands
    value: 19
    elapsed_seconds: 53.85
    dominant_process: StrandBendingOscillationProcess
    dominant_pct: 25.0
    dominant_leaf_process: LinearSolve
    dominant_leaf_pct: 21.0
    dominant_leaf_total: 45.15
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

### status-317

- `summary_rows` に `dominant_leaf_process` / `dominant_leaf_pct` /
  `dominant_leaf_total` を追加。`uses` グラフ再帰走査で静的に葉判定。
- `parameter_sweep_benchmark.py` module docstring に `BenchmarkRunResult`
  の正しい属性参照サンプル（`case.manifest.results_summary`）を追記。
