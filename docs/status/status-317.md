# status-317: ParameterSweepBenchmark `dominant_leaf_process` — wrapper 占有を読み飛ばす真のボトルネック抽出

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-10
- **ブランチ**: `claude/check-status-todos-NIgfb`
- **テスト数**: 459+13+22（status-316 から +11）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

---

## 概要

status-316 の分析で判明した **`dominant_process` フィールドがネスト wrapper
を指してしまう問題**（status-316 知見 D）を解消する。`ProcessMetaclass.
_profile_data` は各 Process の *inclusive* 時間（ネストした子の時間を含む
壁時計）を記録するため、`StrandBendingOscillationProcess` →
`ContactFrictionProcess` → `NewtonDynamicProcess` のように wrapper が
1:1 で子を呼び出す階層では、複数 wrapper 層が同じ elapsed で並び、breakdown
先頭を wrapper が占有して真のボトルネックが見えなくなる（n=37 ケースで
3 wrapper が ~25% ずつ並んだ現象）。

本 status では `ParameterSweepBenchmarkProcess` の `summary_rows` に
`dominant_leaf_process` / `dominant_leaf_pct` / `dominant_leaf_total` を
追加し、`uses` グラフを再帰走査して葉プロセス（`uses=[]`）を先頭から
抽出できるようにする。

---

## 実施内容

### 1. ヘルパ関数 3 種を `parameter_sweep_benchmark.py` に追加

| 関数 | 役割 |
|------|------|
| `_collect_uses_graph(root_cls)` | `root_cls` の `uses` を再帰走査し `{class_name: class}` を返す |
| `_is_leaf_process(name, known_classes)` | `known_classes` に載っていて `uses` が空なら葉。未知は保守的に葉扱い |
| `_first_leaf_breakdown_entry(breakdown, known_classes)` | `profile_breakdown`（total 降順）の先頭葉エントリを抽出 |

**registry 非依存**で設計。`target_process` クラスから静的に `uses` を
辿るだけなので、テスト用 `_skip_registry=True` のダミープロセスでも
正しく葉判定できる。`ProcessRegistry.default()` を触らないため
グローバル状態の副作用もない。

### 2. `ParameterSweepBenchmarkProcess.process()` の拡張

- ループ開始前に `known_classes = _collect_uses_graph(type(target_process))`
  を 1 回だけ構築
- 各ケースの `breakdown` から `_first_leaf_breakdown_entry()` で葉を抽出
- `summary_rows` に以下 3 キーを追加:
  - `dominant_leaf_process`: 葉プロセスのクラス名
  - `dominant_leaf_pct`: pct（小数 3 桁丸め）
  - `dominant_leaf_total`: total 秒（小数 6 桁丸め）

既存の `dominant_process` / `dominant_pct` は breakdown 先頭をそのまま
記録（inclusive 時間ベース）し、**後方互換を保持**。新旧両方を summary
YAML に書き出すことで wrapper と葉の違いが一目で分かるようにする。

### 3. `parameter_sweep_benchmark.py` モジュール docstring に使い方追記

status-316 の「非効果的 / 注意」で指摘された **`case.extracted` 参照誤り**
を再発させないため、`BenchmarkRunResult` の正しい属性参照サンプルを追加:

```python
# 集約 summary 表示（dominant_leaf_process が真のボトルネックを示す）
for row in sweep.summary_rows:
    print(
        f"{row['param_name']}={row['value']}: "
        f"elapsed={row['elapsed_seconds']:.2f}s  "
        f"dominant={row['dominant_process']}({row['dominant_pct']:.1f}%)  "
        f"leaf={row['dominant_leaf_process']}"
        f"({row['dominant_leaf_pct']:.1f}%, "
        f"{row['dominant_leaf_total']:.2f}s)"
    )

# ケース個別の詳細（extractors 値は case.manifest.results_summary）
for case in sweep.cases:
    print(case.manifest_path, case.manifest.results_summary)
```

### 4. テスト追加（+11 テスト）

**`TestParameterSweepBenchmarkProcessAPI` への追加 — 3 テスト**:
- `test_summary_rows_include_dominant_leaf_fields` — 新フィールド存在確認
- `test_dominant_leaf_skips_wrapper` — `_SweepWrapperProcess` target で
  `dominant_process=_SweepWrapperProcess` / `dominant_leaf_process=_SweepTargetProcess`
  の差を検証（wrapper 占有をスキップ）
- `test_summary_yaml_contains_leaf_fields` — 集約 YAML に書き出される

**`TestDominantLeafHelpers` 新設 — 8 テスト**:
- `_collect_uses_graph`: 葉単独 / wrapper+leaf の 2 ケース
- `_is_leaf_process`: empty uses=葉 / non-empty uses=wrapper / unknown=保守的に葉
- `_first_leaf_breakdown_entry`: wrapper をスキップ / 空 breakdown / 葉未検出（全 wrapper）

**fixture**: `_SweepWrapperProcess` を追加。`uses=[_SweepTargetProcess]` と
宣言し `process()` 内で `_inner.process()` を呼ぶ（wrapper → leaf の
ネスト構造を再現）。`_skip_registry=True` を維持したままでも新ヘルパが
機能することを確認。

### 5. 設計文書更新

`xkep_cae/numerical_tests/docs/parameter_sweep_benchmark.md`:
- 「サマリー行のキー」表を新設し 9 キーを明記
- 「`dominant_process` と `dominant_leaf_process` の違い（status-317）」節
  を新設し inclusive 時間の重複計上を説明
- サマリー YAML 例を `dominant_leaf_*` 含む形式に更新
- 末尾の status 履歴に `### status-317` 追記

---

## 変更ファイル

### 更新
- `xkep_cae/numerical_tests/parameter_sweep_benchmark.py`
  - モジュール docstring 拡張（status-317 + 使い方サンプル）
  - `ParameterSweepBenchmarkResult.summary_rows` キー追加ドキュメント
  - `process()` に `known_classes` 構築 + leaf 抽出ロジック
  - `_collect_uses_graph` / `_is_leaf_process` / `_first_leaf_breakdown_entry` ヘルパ追加
- `xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py`
  - `_SweepWrapperProcess` fixture 追加
  - `TestParameterSweepBenchmarkProcessAPI` に 3 テスト追加
  - `TestDominantLeafHelpers` 新設（8 テスト）
- `xkep_cae/numerical_tests/docs/parameter_sweep_benchmark.md`
  - サマリー行キー表 + dominant_leaf 説明節 + YAML 例更新 + status-317 履歴
- `README.md`: 状態行（459+13+22）+ status-317 リンク
- `docs/status/status-index.md`: status-317 行 + テスト数推移 footer
- `docs/roadmap.md`: 完了テーブルに葉プロセス抽出行追加
- `CLAUDE.md`: テスト数 + TODO 消化マーク

### 新規
- `docs/status/status-317.md`（本ファイル）

---

## 検証手順（再現手順）

```bash
git checkout claude/check-status-todos-NIgfb

# 1. 契約チェック
uv run python contracts/validate_process_contracts.py

# 2. テスト実行（status-314/315/316/317 合計 44 テスト）
uv run --with pytest --with pytest-timeout pytest \
    xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py \
    tests/test_benchmark_runner.py -v \
    2>&1 | tee /tmp/log-status317-$(date +%s).log

# 3. lint / format
uv run ruff check xkep_cae/ tests/
uv run ruff format --check xkep_cae/ tests/
```

### 実行結果

- テスト: **44 passed**（`test_parameter_sweep_benchmark.py` 21件 +
  `test_benchmark_runner.py` 23件）
- 契約違反 0 件、条例違反 0 件
- ruff check/format: All checks passed

### 実測環境

- Linux 4.4 / Python 3.11.15 / uv 0.8.17
- NumPy 2.4.4 / pytest 9.0.3 / ruff 0.14.3

---

## 判断の根拠

### なぜ `uses` グラフを静的に走査するのか

3 つの代替案を比較した:

| 案 | 長所 | 短所 |
|----|------|------|
| **A. ProcessRegistry 参照** | シンプル | `_skip_registry=True` のテスト fixture で機能しない |
| **B. 動的 exclusive 時間計算** | 最も正確（self-time） | `ProcessMetaclass._profile_data` の構造改変が必要。status-314 のAPI と競合 |
| **C. `uses` グラフ再帰走査**（本実装） | registry 非依存、静的に完結、テスト fixture でも動作、API 無改変 | `uses` 宣言漏れがある process だと誤判定の可能性 |

C を採用。`uses` 宣言は `AbstractProcess.__init_subclass__` で自動 `_used_by`
連結の起点でもあり、既に全 Process が厳密に維持している。契約違反検証
（`contracts/validate_process_contracts.py`）で 0 件維持されている現状で
信頼できる情報源となる。

### 後方互換の維持方針

既存の `dominant_process` / `dominant_pct` フィールドは **意図的に残す**。
wrapper を先頭に置く breakdown は inclusive 時間集計の自然な表現であり、
呼び出し階層の可視化にも使えるため削除しない。新フィールドは **追加**
のみで、既存 YAML / 既存コードを壊さない。

### wrapper しか含まれない場合のフォールバック

全エントリが非葉の場合 `dominant_leaf_process` は空文字を返す。breakdown
の先頭をフォールバックで入れる案もあったが、**「葉が見つからなかったこと」
を明示する**ほうが診断価値が高いため空返りを採用。実運用では registry の
`uses` グラフに 1 つでも葉（線形ソルバ等）があれば必ず見つかる。

---

## TODO（次担当者向け）

### 直近

- [ ] **100 / 200 / 500 本への掃引拡張** — status-316 から継続。新 `dominant_leaf_process`
  フィールドで **TangentAssembly が LinearSolve を抜く転換点**を特定する
  ことが目的。同じ `status316_nstrands_sweep.py` を `SWEEP_VALUES = (7, 19, 37, 61, 91, 127)`
  で再実行するだけで、本 status の新フィールドが集約 YAML に自動的に載る。
- [ ] **被膜 ON でのプロファイル取得** — `coating_barrier=True` で同スイープ
  実施（status-316 から継続）。
- [ ] **被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装** — status-304 で FD 誤差 67%
  の主因と判明した項。

### 中期

- [ ] **リスタート解析方式への移行** — `(u, v, a, 接触ペア)` I/O 化
- [ ] **シース-素線接触統合** — 旧 SheathModel/HEX8 の Process 化
- [ ] **`ParameterSweepBenchmarkProcess` 並列実行モード**
- [ ] **動的 exclusive 時間記録** — `ProcessMetaclass` に親子関係を記録して
  self-time を直接計測する API を追加すれば、静的 `uses` 走査よりも正確に
  leaf/wrapper を区別できる。現状は `uses` 宣言の信頼性に依存しているが、
  将来的に宣言漏れが発生した場合に検出できるよう、動的計測への移行を
  検討する価値あり（status-314 の profile 統計 API 拡張の一環として）。

### 開発運用メモ

- **効果的**: status-316 で「dominant_process が wrapper を指す問題」を
  現象として明確に記録していたおかげで、本 status の着手判断が 5 分で
  付いた。**status の「TODO」セクションで課題をバックログ化する運用**
  が想定通り機能している。
- **効果的**: `_skip_registry=True` を壊さずに新ヘルパをテストできた。
  `uses` class var は `_skip_registry` に関係なく維持されるので、
  registry を汚さずに wrapper/leaf の関係を検証可能。
- **注意**: `target_process` の `uses` グラフ外のプロセス（テストハーネス
  由来等）が `profile_breakdown` に混ざった場合、`_is_leaf_process` は
  保守的に「葉」として扱う。これは誤って wrapper 扱いして breakdown から
  読み飛ばすリスクを避けるため。代償として、グラフ外の wrapper が混ざると
  それが先頭に来る可能性がある。実運用では `target_process` が十分広い
  グラフを持っているので顕在化しない想定。

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: テスト実行ログは `/tmp/log-status317-*.log` に tee 保存。
- [x] **再現手順記載**: 上記「検証手順（再現手順）」セクション。
- [x] **ベースライン維持**: status-316 の 459+13+11 テストはそのまま通過。
      新規は +11 で合計 459+13+22。
- [x] **変更前計測**: ruff check/format/contract validation は変更前後で
      等しく 0 件を維持。
- [x] **tee ログ出力**: 上記コマンドで明示。
