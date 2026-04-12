# status-189: Phase 8 完了 — C14 抜け道修正 + friction/geometry 実装完成

[<- README](../../README.md) | [<- status-index](status-index.md)

## 日付

2026-03-16

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-w2l4E

## 概要

status-188 の TODO 3件を消化。C14 チェッカーの importlib alias 抜け道を修正し、8つの暫定 re-export モジュールの deprecated 依存を除去。friction evaluate()/tangent() と geometry detect() の stub を完全実装に置換。

## 変更内容

### 1. C14 チェッカー強化 — importlib alias 検出

`import importlib as _il` → `_il.import_module("__xkep_cae_deprecated...")` パターンが検出されていなかった:

- `_collect_importlib_aliases(tree)`: AST から importlib のエイリアス名を収集
- `_is_importlib_deprecated_call()`: エイリアス名セットを受け取り検出精度向上
- `check_c14_deprecated_imports()`: ファイルごとにエイリアスを収集して渡す

### 2. 8つの暫定 re-export モジュール除去

以下のモジュールが `import importlib as _il` でまるごと deprecated を re-export していた:

| モジュール | 対応 |
|-----------|------|
| elements, thermal, sections, math, materials | `__init__.py` を空化（新パッケージ未使用） |
| io | テスト/example のインポート先を deprecated に変更、空化 |
| tuning | テストのインポート先を deprecated に変更、空化 |
| mesh | deprecated re-export 除去（移植済み _twisted_wire/process のみ保持） |

### 3. friction evaluate()/tangent() 実装完成

`contact/friction/_assembly.py` 新設 — deprecated assembly.py の摩擦関連を移植:
- `_compute_tangential_displacement()`: 接線相対変位増分
- `_friction_return_mapping_loop()`: return mapping 統合ループ（純粋関数版を使用）
- `_assemble_friction_force()` / `_assemble_friction_tangent_stiffness()`

`CoulombReturnMappingProcess.evaluate()`:
- NCP 法線力（lambdas + k_pen * gap）から p_n を計算
- _friction_return_mapping_loop() で摩擦力・残差・接線剛性を一括評価

`SmoothPenaltyFrictionProcess.evaluate()`:
- pair.state.p_n（smooth penalty 事前計算済み）を使用
- それ以外は Coulomb 版と同じ return mapping ループ

### 4. geometry detect() 実装完成

`_detect_candidates()` 共通ヘルパー関数を追加:
- `_broadphase_aabb()` で候補ペア検出
- 共有ノード除外（同層フィルタ）
- `_ContactPair` 生成 + 初期 narrowphase（`_batch_update_geometry`）

3 Strategy (PtP/L2L/Mortar) の detect() を `return []` → `_detect_candidates()` 呼び出しに変更。

## テスト結果

- 全テスト: **284テスト合格**（変更なし）
- ruff check: 0 error
- ruff format: 0 issue
- 契約違反: **0件**（C14: 0件、C16: 0件）

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| C14: `importlib` のみ検出 | `importlib` + エイリアス検出 | status-189 |
| 8モジュール暫定 re-export | 空化（deprecated 直接参照に移行） | status-189 |
| friction evaluate()/tangent() ゼロ返却 stub | 実装完成（_assembly.py 経由） | status-189 |
| geometry detect() 空リスト stub | _detect_candidates() 経由の broadphase 実装 | status-189 |

## TODO

- [ ] solver 内部プライベートモジュールの Process 化
- [ ] NewtonUzawa を dynamic/static に完全分離
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動の Phase 2 xfail 解消（ソルバー収束問題 — CLAUDE.md 制約でスコープ外）

## 懸念事項・メモ

- **S3 xfail はソルバー収束問題**: CLAUDE.md で「NCP ソルバーの収束ロジック変更」が禁止されているため、Phase 2 揺動の xfail 解消は本セッションのスコープ外。
- **solver 内部 Process 化**: ユーザーから「solver内部のプライベートモジュールを全てprocess化」「NewtonUzawaLoopをdynamicとstaticで2つに完全分離」の指示あり。次の作業として実施予定。
- **friction/geometry の実装は deprecated と同等のロジック**: 新パッケージの純粋関数（law_friction._return_mapping_core 等）を使用。deprecated への参照は完全排除。

---
