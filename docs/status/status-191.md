# status-191: process.py Process API 移行 + ContactManager Process 分割 + 後方互換エイリアス整理

[<- README](../../README.md) | [<- status-index](status-index.md)

## 日付

2026-03-16

## 作業者

Claude Code

## ブランチ

claude/refactor-contact-process-pzpW6

## 概要

ContactFrictionProcess (process.py) 内のプライベート関数直接呼び出しを全て Process API 経由に移行。ContactManager のメソッドを3つの Process クラスに分割。NewtonUzawaDynamic を正統 NewtonUzawaProcess に昇格。315テスト全パス、C14/C16 契約違反ゼロ維持。

## 変更内容

### 1. process.py プライベート関数 → Process API 移行

process.py 内で直接呼ばれていたプライベート関数を全て Process API 経由に置き換え:

| 旧（プライベート関数直接呼び出し） | 新（Process API 経由） |
|---|---|
| `_snapshot_contact_graph()` | `ContactGraphProcess().process(ContactGraphInput(...))` |
| `_format_diagnostics_report()` | `DiagnosticsReportProcess().process(DiagnosticsInput(...))` |
| `_adjust_initial_positions()` / `_check_initial_penetration()` | `InitialPenetrationProcess().process(InitialPenetrationInput(...))` |
| `_deformed_coords()` | `DeformedCoordsProcess().process(DeformedCoordsInput(...))` |
| `manager.detect_candidates()` | `DetectCandidatesProcess().process(DetectCandidatesInput(...))` |
| `manager.update_geometry()` | `UpdateGeometryProcess().process(UpdateGeometryInput(...))` |

`uses` リストも全 Process 依存を明示するよう更新。

### 2. ContactManager Process 分割（新規ファイル）

`xkep_cae/contact/_manager_process.py` を新規作成:

| Process クラス | 元メソッド | Input/Output |
|---|---|---|
| `DetectCandidatesProcess` | `_ContactManager.detect_candidates()` | `DetectCandidatesInput` → `DetectCandidatesOutput` |
| `UpdateGeometryProcess` | `_ContactManager.update_geometry()` | `UpdateGeometryInput` → `UpdateGeometryOutput` |
| `InitializePenaltyProcess` | `_ContactManager.initialize_penalty()` | `InitializePenaltyInput` → `InitializePenaltyOutput` |

全て `SolverProcess` 継承 + `ProcessMeta` + `@binds_to` テスト付き。

### 3. NewtonUzawaDynamic → 正統 NewtonUzawaProcess 昇格

`_newton_uzawa.py` のエイリアスを更新:
- 旧: `NewtonUzawaProcess = NewtonUzawaStaticProcess`
- 新: `NewtonUzawaProcess = NewtonUzawaDynamicProcess`

Static は保存用として維持。Dynamic が標準動作（dt_sub=0 で静的と同等に動作）。

### 4. テスト更新

- `test_dynamic_is_primary_process`: Dynamic が正統 Process であることを検証
- `TestDetectCandidatesProcessAPI`: 候補検出 Process のテスト
- `TestUpdateGeometryProcessAPI`: 幾何更新 Process のテスト
- `TestInitializePenaltyProcessAPI`: ペナルティ初期化 Process のテスト

## テスト結果

- **315 passed** (xkep_cae/ 全体)
- **51 passed** (solver テスト)
- C14/C16 契約違反: **0件**
- ruff check/format: **全パス**

## 互換ヒストリー

| 旧 | 新 | status |
|----|----|----|
| `process.py` プライベート関数5種直接呼び出し | Process API 経由 | status-191 |
| `manager.detect_candidates()` 直接呼び出し | `DetectCandidatesProcess` | status-191 |
| `manager.update_geometry()` 直接呼び出し | `UpdateGeometryProcess` | status-191 |
| `NewtonUzawaProcess = NewtonUzawaStaticProcess` | `NewtonUzawaProcess = NewtonUzawaDynamicProcess` | status-191 |

## TODO

- [ ] friction/geometry Strategy の process.py 内 deprecated Strategy 直接使用 → 新 Strategy 経由に移行（status-190 引き継ぎ）
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動 Phase2 xfail 解消

---
