# status-315: ParameterSweepBenchmarkProcess 新設 + manifest 連番衝突回避

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-10
- **ブランチ**: `claude/execute-status-todos-bVcT1`
- **テスト数**: 11 件追加（`TestParameterSweepBenchmarkProcessAPI` 10 + `TestBenchmarkRunnerProcessAPI::test_manifest_filename_collision_avoided` 1）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

---

## 概要

status-314 の TODO **「1000 本撚線プロファイリング実測（profile_breakdown 活用）」** の実測エントリポイントを整備した。

1 回の `BenchmarkRunnerProcess.process()` は 1 ケース分のマニフェスト＋`profile_breakdown` を YAML に書き出すところまでで止まっている。これを複数ケースに掃引するための汎用 BatchProcess `ParameterSweepBenchmarkProcess` を新設し、`StrandBendingOscillationProcess` を含む任意の frozen dataclass 系プロセスをそのまま n_strands／wire_radius／n_elements 等のパラメータで掃引できるようにした。

掃引実装中に、`BenchmarkRunnerProcess._save_manifest` が同一秒内の複数呼び出しで manifest YAML を **上書き** してしまう bug を発見したため、同一ファイル名が存在する場合に `_01`, `_02` ... の連番を付与するフォールバックを追加した。これにより 1000 本撚線スイープでも各ケースの manifest が確実に残る。

---

## 実施内容

### 1. `ParameterSweepBenchmarkProcess` 新設（`xkep_cae/numerical_tests/parameter_sweep_benchmark.py`）

任意 Process × 任意 frozen dataclass のパラメータ掃引を BatchProcess として実装。`uses = [BenchmarkRunnerProcess]` により既存インフラを全面的に再利用し、掃引ループ＋集約サマリ YAML 生成だけを薄く乗せる設計。

#### 入力 `ParameterSweepBenchmarkInput`

| フィールド | 型 | 意味 |
|-----------|-----|-----|
| `target_process` | `AbstractProcess` | 掃引対象のプロセス |
| `base_config` | frozen dataclass | 掃引元となる config |
| `param_name` | `str` | 差し替え対象フィールド名 |
| `param_values` | `tuple[Any, ...]` | 差し替え値の列 |
| `result_extractors` | `dict[str, Callable]` | `BenchmarkRunInput` に転送される結果抽出関数 |
| `output_dir` | `str \| None` | 個別／集約マニフェスト保存先 |
| `status_file` | `str \| None` | status ドキュメントリンク |
| `profile_sort_by` | `str` | `BenchmarkRunInput.profile_sort_by` 転送 |
| `profile_top_n` | `int \| None` | `BenchmarkRunInput.profile_top_n` 転送 |

#### 出力 `ParameterSweepBenchmarkResult`

| フィールド | 型 | 意味 |
|-----------|-----|-----|
| `param_name` | `str` | 入力のコピー |
| `param_values` | `tuple[Any, ...]` | 入力のコピー |
| `cases` | `tuple[BenchmarkRunResult, ...]` | 1 ケース 1 エントリ（各 YAML マニフェスト付き）|
| `summary_rows` | `tuple[dict, ...]` | 集約サマリ行：`param_name`, `value`, `elapsed_seconds`, `dominant_process`, `dominant_pct`, `manifest_path` |
| `summary_yaml_path` | `str \| None` | 集約 YAML の保存先 |

#### パイプライン

1. `dataclasses.is_dataclass(base_config)` + `param_name in fields` を事前検証（違反は `TypeError` / `ValueError` で即時失敗）
2. `param_values` の各要素について:
   1. `dataclasses.replace(base_config, **{param_name: value})`
   2. `BenchmarkRunnerProcess().process(BenchmarkRunInput(..., capture_profile=True))` を 1 回呼ぶ（プロファイルスナップショットは BenchmarkRunnerProcess 側が責任を持つので本 Process は介入しない）
   3. `manifest.profile_breakdown[0]` を dominant Process として summary_rows に追記
3. 集約サマリ YAML を `ParameterSweepBenchmark_{timestamp}.yaml` で保存

### 2. `BenchmarkRunnerProcess._save_manifest` 衝突回避（`xkep_cae/core/benchmark.py`）

同一秒内に複数ケースが完走すると `timestamp[:15]` が衝突して manifest が上書きされていた。連番フォールバックを追加:

```python
ts = manifest.environment.timestamp.replace(":", "").replace("-", "")[:15]
filename = f"{name}_{ts}.yaml"
path = base / filename
counter = 1
while path.exists():
    filename = f"{name}_{ts}_{counter:02d}.yaml"
    path = base / filename
    counter += 1
```

既存テストの YAML ファイル名 assertion は無いので非破壊。

### 3. テスト追加（11 件）

| ファイル | クラス | テスト数 | 内容 |
|----------|--------|---------|------|
| `xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py` | `TestParameterSweepBenchmarkProcessAPI` | 10 | meta 紐付け / uses 宣言 / 掃引値反映 / summary_rows の dominant_process / 集約 YAML 保存 / ケース毎 manifest ユニーク性 / result_extractors 転送 / 空 param_values 例外 / 不正 param_name 例外 / 非 dataclass 例外 |
| `tests/test_benchmark_runner.py` | `TestBenchmarkRunnerProcessAPI::test_manifest_filename_collision_avoided` | 1 | 同一秒内 3 連続呼び出しで 3 つの異なるファイルが存在することを検証 |

`_SweepTargetProcess` は `_skip_registry=True` + `time.sleep(1ms*n)` の軽量ダミーで、実ソルバー不要で 10 件すべて 0.6s 以下で完走する。

---

## 変更ファイル

### 新規
- `xkep_cae/numerical_tests/parameter_sweep_benchmark.py` (+217 行)
- `xkep_cae/numerical_tests/docs/parameter_sweep_benchmark.md` (新規、設計仕様)
- `xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py` (+223 行)

### 更新
- `xkep_cae/core/benchmark.py`: `_save_manifest` 連番フォールバック（+7 行）
- `tests/test_benchmark_runner.py`: 衝突回避テスト追加（+27 行）
- `README.md`: テスト数 459+13 → 459+23、status-315 追記
- `docs/status/status-index.md`: status-315 行 + footer 履歴
- `docs/roadmap.md`: status-315 行 + 「次」更新
- `CLAUDE.md`: テスト数更新 + TODO 次項修正

---

## 再現手順

```bash
# ブランチ
git checkout claude/execute-status-todos-bVcT1

# 新規テスト単体
python -m pytest xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py -v

# Benchmark 衝突回避テスト
python -m pytest tests/test_benchmark_runner.py::TestBenchmarkRunnerProcessAPI::test_manifest_filename_collision_avoided -v

# 関連テスト全体（status-314 との互換性確認）
python -m pytest xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py \
                 xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py \
                 tests/test_benchmark_runner.py \
                 tests/test_profile_stats.py -q -m "not slow" 2>&1 | tee /tmp/log-$(date +%s).log

# xkep_cae/ 全体（slow 除外）
python -m pytest xkep_cae/ -q -m "not slow" --ignore=xkep_cae/post 2>&1 | tee /tmp/log-$(date +%s).log

# tests/ 全体
python -m pytest tests/ -q -m "not slow" 2>&1 | tee /tmp/log-$(date +%s).log

# lint / format
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/

# 契約チェック
python contracts/validate_process_contracts.py
```

### テスト結果

```
xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py  10 passed
tests/test_benchmark_runner.py                                   15 passed (status-314: 14 + 衝突回避 1)
tests/test_profile_stats.py                                       9 passed
xkep_cae/ (-m "not slow")                                       530 passed, 10 skipped, 14 deselected, 1 xfailed, 1 pre-existing FAIL (stress_contour, status-312 既知)
tests/ (-m "not slow")                                          202 passed, 10 skipped, 59 deselected
```

契約違反 0 件、条例違反 0 件、ruff check/format 全通過。

---

## 1000 本撚線プロファイリング使用例

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
        param_values=(7, 19, 37),  # 100-1000 拡張は計算資源確保後
        result_extractors={
            "n_increments": lambda r: r.solver_result.n_increments,
            "converged": lambda r: r.solver_result.converged,
        },
        status_file="docs/status/status-315.md",
        profile_top_n=10,
    )
)

for row in sweep.summary_rows:
    print(row)  # {param_name, value, elapsed_seconds, dominant_process, dominant_pct, manifest_path}
```

集約 YAML (`ParameterSweepBenchmark_{ts}.yaml`) に:
- `target_process: StrandBendingOscillationProcess`
- `base_config_type: StrandBendingOscillationConfig`
- `param_name: n_strands`
- `param_values: [7, 19, 37]`
- `cases:` の各行に per-case の `dominant_process` / `dominant_pct` / `manifest_path`

が記録され、各ケースの詳細 profile_breakdown は per-case の `BenchmarkRunnerProcess` YAML（衝突回避済み）に残る。

---

## TODO

- [ ] **実測実施**: 上記コード例を 7 → 19 → 37 本で実行し、dominant Process の推移をデータとして記録する（本 status ではインフラ整備までに留め、実測は計算リソース確保後に別 status で実施）
- [ ] 被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装
- [ ] シース-素線接触統合（旧 SheathModel/HEX8 の Process 化）
- [ ] リスタート解析方式への移行 — `(u, v, a, 接触ペア)` I/O 整理
- [ ] `ParameterSweepBenchmarkProcess` の並列実行モード（現状は直列）— ケース間で `_profile_data` が競合しないよう snapshot 差分管理を工夫する必要あり

---

## 次の担当者向け

### 重要ポイント

1. **汎用掃引 Process**: `ParameterSweepBenchmarkProcess` は `StrandBendingOscillationProcess` 専用ではない。任意の frozen dataclass × AbstractProcess の組み合わせで 1 フィールド掃引ができる。例えば `DynamicThreePointBendContactJigConfig.E` を 25 → 50 → 100 MPa で掃引して剛性依存の dominant Process 推移を見るといった使い方も可能。

2. **掃引値は `dataclasses.replace` で反映**: `base_config` のその他フィールドは全てそのまま維持される。複数フィールド同時掃引が必要なら、現状は掃引値を `base_config` ごと入れ替える上位ラッパーを別途作る（本 Process は 1 フィールド限定で設計を単純化した）。

3. **manifest 衝突回避**: `BenchmarkRunnerProcess._save_manifest` は同一秒内の複数呼び出しに対して `_01`, `_02`, ... の連番を付与する。集約 YAML (`ParameterSweepBenchmark_*.yaml`) には各ケースの実際のパス（`manifest_path`）が記録されるので、衝突しても紛失しない。

4. **`profile_breakdown` のデルタ集計**: `BenchmarkRunnerProcess` が `ProcessMetaclass.snapshot_profile()` のスナップショットを取ってから `get_profile_stats(since=...)` で差分を計算しているので、ケース間で以前の profile が混ざることはない（status-314 の設計がそのまま活きている）。

5. **`_skip_registry=True` ダミー**: テストで使う `_SweepTargetProcess` は `_skip_registry=True` にしないと `@binds_to(ParameterSweepBenchmarkProcess)` の C3 契約チェックが過剰に反応する。他の BenchmarkRunner テストの `_DummyProcess` と同じ流儀。

### 開発運用で発見した点

- **効果的**: BenchmarkRunnerProcess（status-314）の設計が十分に薄かったので、その上に掃引 Process を被せる際に既存 API を一切変更せずに済んだ。Process Architecture の合成容易性が素直に活きた例。
- **注意（bug fix 相当）**: `_save_manifest` の timestamp 衝突は、単発ベンチマーク中心だった status-314 時点では表面化しなかったが、掃引用途では即座に顕在化した。「新機能実装時に旧機能の前提条件をチェックする」運用が機能した例。
- **非効果的**: 最初、テスト用ダミー Process の `document_path` を `tests/test_benchmark_runner.py` のもの (`"../xkep_cae/core/docs/benchmark_runner.md"`) をそのまま流用しようとしたが、`numerical_tests/tests/` からの相対パスとしては不正だった。`document_path` は常にソースファイル相対であることを再認識した。

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: 実測 benchmark はインフラ整備に留め、テストは軽量ダミーでのみ実行
- [x] **再現手順記載**: 上記「再現手順」セクション
- [x] **ベースライン維持**: `xkep_cae/` 530 passed + 10 skipped + 1 xfailed + 1 pre-existing FAIL（stress_contour、status-312/314 既知）
- [x] **回帰なし**: 契約違反 0 件、lint 全通過、BenchmarkRunnerProcess 既存テスト 14 件すべて成功（`_save_manifest` 連番フォールバック追加後も）
