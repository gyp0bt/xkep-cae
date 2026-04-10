# status-313: プロファイル統計 API 強化 + BenchmarkRunner プロファイル自動キャプチャ

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-10
- **ブランチ**: `claude/check-status-todos-CtBHe`
- **テスト数**: 13 件追加（`TestProfileStatsAPI` 6 + `TestProfileReportAPI` 3 + `TestBenchmarkRunnerProcessAPI` 4）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

---

## 概要

status-312 の TODO のうち **「1000 本撚線プロファイリング」** の準備基盤として、`ProcessMetaclass._profile_data` を活用した構造化プロファイル統計 API を整備し、`BenchmarkRunnerProcess` の走査マニフェストに `profile_breakdown` を自動記録するようにした。

これにより「ベンチマーク 1 回の YAML マニフェストから、どの Process が合計時間の何 % を占めたか」を即座に読み取れるようになり、大規模ケースの熱い経路特定が手動計測無しで完結する。

---

## 実施内容

### 1. `ProcessMetaclass` 構造化プロファイル API（`xkep_cae/core/base.py`）

既存の `_profile_data: dict[str, list[float]]` を活かしたまま、以下のクラスメソッドを追加/強化:

| メソッド | 役割 |
|----------|------|
| `snapshot_profile()` | 現時点の呼び出し回数 dict を返す（非破壊的スナップショット） |
| `get_profile_stats(since, sort_by)` | 構造化統計 `[{name, n, total, avg, min, max, median, pct}, ...]` |
| `get_profile_report(since, sort_by, top_n)` | ソート済みテキストレポート（デフォルト: total 降順、全件、%付き） |

**デルタ集計の原理**: `_profile_data` はグローバルアキュムレータのため、ベンチマーク前後を切り出すには呼び出し回数をスナップショットしておき、後で `times[snapshot_count:]` に対して統計を取る。これにより既存データを破壊せずベンチマーク単位の集計が可能。

**ソート軸**:
- `"total"`（デフォルト）: 合計時間降順 — ボトルネック特定用
- `"avg"`: 平均時間降順 — 1 呼び出しが重い処理の検出
- `"n"`: 呼び出し回数降順 — 内側ループ候補の検出
- `"name"`: 名前昇順 — 差分比較用

### 2. `BenchmarkRunnerProcess` へのプロファイル統合（`xkep_cae/core/benchmark.py`）

| 項目 | 旧実装 | 新実装 |
|------|--------|--------|
| `BenchmarkRunInput` | `capture_profile` 無し | `capture_profile=True`（デフォルト ON）+ `profile_sort_by`/`profile_top_n` |
| `RunManifest` | `profile_breakdown` 無し | `profile_breakdown: tuple[dict, ...]` 追加（YAML 出力にも反映） |
| `BenchmarkRunnerProcess.process()` | プロファイル未捕捉 | 実行前後で `snapshot_profile()` → `get_profile_stats(since=...)` を実行して delta を記録 |

**YAML 出力例**（抜粋）:
```yaml
profile_breakdown:
  -
    name: ContactFrictionProcess
    n: 1
    total: 612.543
    avg: 612.543
    min: 612.543
    max: 612.543
    median: 612.543
    pct: 81.4
  -
    name: ComputeStJacobianProcess
    n: 5840
    total: 72.188
    avg: 0.012
    ...
    pct: 9.6
```

### 3. テスト追加（13 件）

| ファイル | クラス | テスト数 | 内容 |
|----------|--------|---------|------|
| `tests/test_profile_stats.py` | `TestProfileStatsAPI` | 6 | snapshot/stats の基本 API、`since` フィルタ、ソート軸、pct 正規化 |
| `tests/test_profile_stats.py` | `TestProfileReportAPI` | 3 | テキストレポートのヘッダ、`top_n`、`since` 絞り込み |
| `tests/test_benchmark_runner.py` | `TestBenchmarkRunnerProcessAPI` | 4 | `profile_breakdown` の記録、`capture_profile=False`、`profile_top_n`、YAML 出力 |

`@binds_to` は 1:1 制約のため、プロファイル API 用ダミープロセスは `_FastProcess`（Stats 用）と `_SlowProcess`（Report 用）の 2 つに分離。

---

## 変更ファイル

- `xkep_cae/core/base.py`: `snapshot_profile` / `get_profile_stats` / `get_profile_report` 強化（+73 行）
- `xkep_cae/core/benchmark.py`: `BenchmarkRunInput` 3 フィールド追加 + `RunManifest.profile_breakdown` + `process()` へのプロファイル統合（+40 行）
- `tests/test_profile_stats.py`: 新規作成（9 テスト）
- `tests/test_benchmark_runner.py`: 4 テスト追加

---

## 再現手順

```bash
# ブランチ
git checkout claude/check-status-todos-CtBHe

# 新規テスト単体
python -m pytest tests/test_profile_stats.py -v

# Benchmark 統合テスト
python -m pytest tests/test_benchmark_runner.py::TestBenchmarkRunnerProcessAPI -v

# 全 xkep_cae/ サブテスト（pre-existing stress_contour 1 件のみ FAIL）
python -m pytest xkep_cae/ -q 2>&1 | tee /tmp/log-$(date +%s).log

# lint / format
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/

# 契約チェック
python contracts/validate_process_contracts.py
```

---

## 使い方（大規模ベンチマーク向け）

```python
from xkep_cae.core.benchmark import BenchmarkRunInput, BenchmarkRunnerProcess
from xkep_cae.core.base import ProcessMetaclass

# ベンチマーク対象 Process を BenchmarkRunner で包むだけ
run_input = BenchmarkRunInput(
    process=my_process,
    config=my_config,
    status_file="docs/status/status-314.md",
    output_dir="docs/benchmarks/",
    capture_profile=True,      # ← デフォルト ON
    profile_sort_by="total",    # ← 合計時間降順
    profile_top_n=20,           # ← 上位 20 件のみ保存
)
run_result = BenchmarkRunnerProcess().process(run_input)

# YAML マニフェストに profile_breakdown が自動記録される
print(run_result.manifest_path)

# 手動でテキストレポートも取得可能
print(ProcessMetaclass.get_profile_report(top_n=20))
```

---

## TODO

- [ ] **1000 本撚線ベンチマーク実測** — 今回の `profile_breakdown` を使って実際に 100〜1000 本ケースでどの Process が支配的かを定量測定し、次の最適化対象を特定
- [ ] 被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装
- [ ] シース-素線接触統合（旧 SheathModel/HEX8 の Process 化）
- [ ] リスタート解析方式への移行 — `(u, v, a, 接触ペア)` I/O 整理
- [ ] `get_profile_stats` の階層集計（親子プロセスのネスト時間計上） — 現状はフラットな classname 単位。`ProcessMetaclass._call_stack` を活かせば cProfile 風のツリーも可能

---

## 次の担当者向け

### 重要ポイント

1. **`BenchmarkRunInput.capture_profile` はデフォルト ON** — 既存ベンチマークコードを変更せずに `run_result.manifest.profile_breakdown` でボトルネック内訳を取得できる
2. **delta 集計**: プロファイルはグローバル累積なのでベンチマーク前後の `snapshot_profile()` を取り、`get_profile_stats(since=...)` で差分だけ集計している（既存データは破壊しない）
3. **ソート軸**: bottleneck 特定には `sort_by="total"`、内側ループ候補には `sort_by="n"` を使い分ける
4. **`@binds_to` 1:1 制約**: Stats/Report で同じダミープロセスを使えないため `_FastProcess` と `_SlowProcess` に分離している点に注意
5. **既存の `reset_profile()` は温存** — CI 並列実行や個別テストで分離したい場合は引き続きリセット可能

### 開発運用で発見した点

- **効果的**: `ProcessMetaclass._profile_data` は status-265 で既に全 Process を自動計測済み。今回の改修は「集計の読み方」を整備しただけで、計測の仕掛けは不要だった。過去の基盤投資が活きた例。
- **注意**: `_profile_data` はクラス変数でグローバル累積のため、並列テストや長時間セッションでは肥大化する。必要に応じて `reset_profile()` を呼ぶこと。

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: テスト結果をそのまま記録（13 件追加、全合格）
- [x] **再現手順記載**: 上記「再現手順」セクション
- [x] **ベースライン維持**: `xkep_cae/` 534 passed + 10 skipped + 1 xfailed + 1 pre-existing FAIL（stress_contour、status-312 でも既知）
- [x] **回帰なし**: 契約違反 0 件、lint 全通過
