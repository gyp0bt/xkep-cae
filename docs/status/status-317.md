# status-317: dominant_leaf_process 拡張 — wrapper/leaf 自動分類

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-10
- **ブランチ**: `claude/check-status-todos-t5Ckj`
- **テスト数**: 459+13+11+7（`tests/test_profile_stats.py` に +4、
  `xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py` に +3、計 +7）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

---

## 概要

status-316 の「知見 D」で特定した **集約プロファイルの `dominant_process` が
wrapper（ネスト Process）を指す問題** を改修する。

実測時 `StrandBendingOscillationProcess` / `ContactFrictionProcess` /
`NewtonDynamicProcess` の 3 つが ~25% ずつで並んで先頭に来る現象は、
これらが子 Process（LinearSolveProcess, TangentAssembly, ...）を包含する
wrapper であるため elapsed が入れ子で二重計上されていることが原因だった。
dominant 判定で「本当に計算負荷が支配的な葉 Process」を一目で出せるよう、
wrapper を自動検出して除外する分類機構を追加した。

本 status の TODO 消化対象（status-316 直近 TODO 末尾）:

> **`ParameterSweepBenchmarkProcess.dominant_leaf_process`** — 集約 summary の
> dominant_process が wrapper を指す問題の改善。`profile_breakdown` から
> wrapper/nested を除外した葉 Process の先頭を別フィールドで出す拡張。

---

## 変更内容

### 1. `ProcessMetaclass` に wrapper 追跡を追加 — `xkep_cae/core/base.py`

`traced_process` 起動時、`_call_stack` が非空なら top の Process を
「子を呼び出した履歴あり」として `_wrapper_classes: set[str]` に記録する。

```python
# status-317: 親（スタック top）が居れば、その親は子を抱える
# wrapper として記録する。dominant_leaf_process の葉判定に使う。
if ProcessMetaclass._call_stack:
    ProcessMetaclass._wrapper_classes.add(ProcessMetaclass._call_stack[-1])
ProcessMetaclass._call_stack.append(cls_name)
```

- `_wrapper_classes` は `ClassVar[set[str]]`。全 Process 共有のランタイム集合。
- `reset_profile()` で併せてクリアされる。
- `get_profile_stats()` の各エントリに新フィールド `is_wrapper: bool` を追加。

### 2. `BenchmarkRunnerProcess.profile_breakdown` に `is_wrapper` を含める — `xkep_cae/core/benchmark.py`

`RunManifest.profile_breakdown` の各 dict に `is_wrapper` を書き出す。
既存の YAML serializer は bool を先に処理するためそのまま動く（`_dict_to_yaml`）。

### 3. `ParameterSweepBenchmarkProcess` に `dominant_leaf_process` 追加 — `xkep_cae/numerical_tests/parameter_sweep_benchmark.py`

`summary_rows` の各行に `dominant_leaf_process` / `dominant_leaf_pct` の 2 列を追加。
`profile_breakdown` を先頭から走査し、`is_wrapper=False` の初エントリを選ぶ。
該当なしなら空文字 / 0.0。

サマリ YAML にもそのまま保存される（既存 `_dict_to_yaml` 経由）。

### 4. 単体テスト追加 — `tests/test_profile_stats.py` / `xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py`

| 追加先 | テスト | 内容 |
|--------|--------|------|
| `test_profile_stats.py::TestProfileWrapperClassification` | `test_leaf_only_call_has_no_wrapper` | 葉単独呼び出しで `is_wrapper=False` |
| 〃 | `test_wrapper_calling_leaf_marks_wrapper_true` | 親→子呼び出しで親 `is_wrapper=True`、子 `False` |
| 〃 | `test_reset_profile_clears_wrapper_classes` | `reset_profile` で `_wrapper_classes` もクリア |
| 〃 | `test_wrapper_flag_sticks_across_snapshots` | `since` snapshot を跨いでも wrapper 判定は持続 |
| `test_parameter_sweep_benchmark.py::TestParameterSweepBenchmarkProcessAPI` | `test_leaf_only_target_has_leaf_equal_to_dominant` | 葉 target 直接掃引 → leaf == dominant |
| 〃 | `test_wrapper_target_resolves_to_inner_leaf` | wrapper target 掃引 → leaf は内部葉を指す |
| 〃 | `test_summary_yaml_contains_dominant_leaf_process` | 集約 YAML に `dominant_leaf_process` 出力 |

テスト方針: 実ソルバーを走らせず、`time.sleep` ベースの軽量ダミー Process
（`_LeafWorkerProcess` / `_WrapperBatchProcess` / `_SweepInnerLeafProcess` /
`_SweepWrapperTargetProcess`）で wrapper/leaf の分類ロジックだけを検証。

---

## 実行結果

### テスト

```bash
python -m pytest tests/test_profile_stats.py \
    tests/test_benchmark_runner.py \
    xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py -q
# 49 passed
```

全関連テスト 49 件合格。

```bash
python -m pytest tests/ -m "not slow and not external" -q
# 206 passed, 10 skipped, 59 deselected
```

`tests/` 全体で回帰 0。

```bash
python -m pytest xkep_cae/contact/ xkep_cae/core/batch/ xkep_cae/elements/ -q
# 405 passed, 9 skipped
```

ProcessMetaclass 変更の透過性確認（Process-heavy test で 405 件回帰 0）。

### 契約・lint

```bash
python contracts/validate_process_contracts.py
# 契約違反なし、条例違反なし

ruff check xkep_cae/ tests/ work/strand_profiling/
# All checks passed!

ruff format --check xkep_cae/ tests/ work/strand_profiling/
# 156 files already formatted
```

### 既知の pre-existing 問題（本 status と無関係）

`xkep_cae/` 全体実行で 2 件の pre-existing 失敗を確認したが、いずれも
`git stash` したベースラインでも同様に失敗することを確認済み:

- `xkep_cae/numerical_tests/tests/test_beam_oscillation.py::TestBeamOscillationProcessAPI::test_process_runs`
  — 60s default timeout を超えて stall（slow マーカー漏れの可能性）
- `xkep_cae/output/tests/test_stress_contour.py::TestStressContour3DProcessAPI::test_process_runs`
  — `result.image_paths` が空で assert 失敗（環境依存）

本 status の変更とは無関係。次担当者 TODO に追加。

---

## 変更ファイル

### 更新
- `xkep_cae/core/base.py`: ProcessMetaclass に `_wrapper_classes` 追跡 +
  `is_wrapper` フィールド + `reset_profile` 拡張
- `xkep_cae/core/benchmark.py`: `profile_breakdown` の dict に `is_wrapper` 追加
- `xkep_cae/numerical_tests/parameter_sweep_benchmark.py`: `summary_rows`
  に `dominant_leaf_process` / `dominant_leaf_pct` 追加、dataclass docstring 更新
- `tests/test_profile_stats.py`: `TestProfileWrapperClassification` 追加（+4）
- `xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py`:
  `_SweepInnerLeafProcess` / `_SweepWrapperTargetProcess` ダミー追加 +
  dominant_leaf テスト 3 件（+3）

### 新規
- `docs/status/status-317.md`（本ファイル）

---

## 再現手順

```bash
git checkout claude/check-status-todos-t5Ckj

# 1. 契約チェック
python contracts/validate_process_contracts.py

# 2. 新テスト実行
python -m pytest tests/test_profile_stats.py \
    tests/test_benchmark_runner.py \
    xkep_cae/numerical_tests/tests/test_parameter_sweep_benchmark.py -v \
    2>&1 | tee /tmp/log-status317-$(date +%s).log

# 3. tests/ 回帰確認
python -m pytest tests/ -m "not slow and not external" -q

# 4. lint / format
ruff check xkep_cae/ tests/ work/strand_profiling/
ruff format --check xkep_cae/ tests/ work/strand_profiling/
```

### 実測環境

- Linux 4.4 / Python 3.11.15 / NumPy 2.4.4 / SciPy 1.17.1
- git_commit (baseline): 88f61f8（status-316 完了時点）
- git_branch: claude/check-status-todos-t5Ckj

---

## TODO（次担当者向け）

### 直近

- [ ] **100 / 200 / 500 本への掃引拡張 + `dominant_leaf_process` での検証** —
  status-317 で追加した `dominant_leaf_process` が **転換点**（LinearSolve →
  TangentAssembly へ支配が移る n_strands）を明確に示せるか確認する。
  `SWEEP_VALUES = (7, 19, 37, 61, 91, 127)` 相当で再実行し、summary YAML の
  `dominant_leaf_process` が途中で切り替わる点を特定。
- [ ] **被膜 ON でのプロファイル取得** — status-316 の直近 TODO 継続。
  `coating_barrier=True` で同スイープを実施し `dominant_leaf_process` の推移を
  記録。
- [ ] **被膜幾何接線剛性（∂n/∂u, ∂s/∂u）の実装** — status-304 の FD 誤差 67%
  主因項。フルバリア被膜の正確な接線剛性。
- [ ] **pre-existing 失敗テスト 2 件の原因調査** —
  `test_beam_oscillation::test_process_runs`（60s timeout stall）と
  `test_stress_contour::test_process_runs`（image_paths 空）。本 status 以前
  から失敗していたが、status-316 の「459+13+11 passed」記載とは
  矛盾するため、実行環境や pytest 設定の差を調査すべき。

### 中期

- [ ] **リスタート解析方式への移行** — `(u, v, a, 接触ペア)` I/O 化
  （status-315 から継続）
- [ ] **シース-素線接触統合** — 旧 SheathModel/HEX8 の Process 化
- [ ] **`ParameterSweepBenchmarkProcess` 並列実行モード** —
  `_profile_data` / `_wrapper_classes` はプロセス全体で共有なので
  並列化時はスナップショット差分管理が必要（status-315 から継続）。
  ただし `_wrapper_classes` は**monotonic な集合**（追加のみ）なので
  並列スレッド間のレースは比較的安全。

### 開発運用メモ

- **効果的**: status-316 で課題を明記しておいたおかげで、次の実装タスクを
  即座に特定できた。「次 status でやる」レベルの small TODO を status-316
  TODO 末尾に残しておく方式は有効。
- **効果的**: `_call_stack` による wrapper 検出は、静的な基底クラス判定
  （`BatchProcess` 継承チェック）よりもロバスト。例えば `SolverProcess`
  を継承する `NewtonDynamicProcess` も NR 反復内で子 Process を呼ぶので
  wrapper だが、継承ベース判定ではこれを見逃す。
- **非効果的 / 注意**: 初期案では継承ベースで wrapper 判定しようとしたが、
  上記の理由でボツ。ランタイムの実際の呼び出し関係を使うのが唯一確実。
- **既知の制限**: 動的に wrapper 化/leaf 化する Process は扱えない。
  例えば「接触なしなら子を呼ばず leaf、接触ありなら wrapper」のような
  Process があった場合、一度でも wrapper として実行されると永久に
  wrapper 扱いになる。実装上は保守的扱い（偽陽性）なので問題は起きにくい。

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: wrapper 判定はランタイム観測ベース、外部データなし。
- [x] **再現手順記載**: 上記「再現手順」セクション。
- [x] **ベースライン維持**: status-316 テスト 459+13+11 からの差分は +7（新規）のみ。
  既存 206（tests/）+ 405（contact/batch/elements）の回帰 0 を確認。
- [x] **変更前計測**: `git stash` ベースラインで同じコマンド実行し、
  失敗 2 件が pre-existing であることを確認。
- [x] **契約チェック**: 契約違反 0 件、条例違反 0 件。
