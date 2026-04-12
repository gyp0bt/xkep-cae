# status-192: Process 内部プライベート関数移行 + Strategy 公開 API 化 + 条例違反検知

[<- README](../../README.md) | [<- status-index](status-index.md)

## 日付

2026-03-16

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-IlXEv

## 概要

status-191 の TODO を実行。Process 内部のプライベート関数直接呼び出しを Process API 経由に移行（HIGH 3箇所）。Strategy クラスのプライベート属性アクセスを公開 setter/property に移行。条例違反検知（O1: テストでの直接関数呼び出し検出）を validation スクリプトに追加。315テスト全パス、C14/C16 契約違反ゼロ、条例違反ゼロ維持。

## 変更内容

### 1. HIGH: _nuzawa_steps.py プライベート関数 → Process API 移行

| Process クラス内 | 旧（直接呼び出し） | 新（Process API 経由） |
|---|---|---|
| `ContactForceAssemblyProcess` | `_deformed_coords()` | `DeformedCoordsProcess().process(DeformedCoordsInput(...))` |
| `UzawaUpdateProcess` | `_deformed_coords()` | `DeformedCoordsProcess().process(DeformedCoordsInput(...))` |
| `LineSearchUpdateProcess` | `_ncp_line_search()` | `NCPLineSearchProcess().process(NCPLineSearchInput(...))` |

`uses` リストも全 Process 依存を明示するよう更新。

### 2. Strategy プライベート属性アクセス → 公開 API 化

| ファイル | 旧（プライベート属性直接アクセス） | 新（公開 setter/property） |
|---|---|---|
| `friction/strategy.py` | `_k_pen`, `_k_t_ratio`, `_mu_ramp_counter` 直接書込み | `set_k_pen()`, `set_k_t_ratio()`, `set_mu_ramp_counter()` |
| `contact_force/strategy.py` | `_ndof` 直接書込み, `_n_uzawa_max`/`_tol_uzawa` getattr | `set_ndof()`, `n_uzawa_max`/`tol_uzawa` プロパティ |
| `solver/process.py` | `hasattr(..., "_k_pen")` + 直接代入 | `hasattr(..., "set_k_pen")` + setter 呼出 |
| `_nuzawa_steps.py` | `_mu_ramp_counter` 直接代入 | `set_mu_ramp_counter()` |
| `_newton_uzawa_static.py` | `getattr(..., "_n_uzawa_max", 5)` | `getattr(..., "n_uzawa_max", 5)` |
| `_newton_uzawa_dynamic.py` | 同上 | 同上 |

### 3. MEDIUM: 同一ファイル内ヘルパー → 設計上維持

以下4箇所のヘルパー関数は Process 内部実装の詳細として維持:

- `_initial_penetration.py`: `_check_initial_penetration()`, `_adjust_initial_positions()`
- `_contact_graph.py`: `_snapshot_contact_graph()`
- `_diagnostics.py`: `_format_diagnostics_report()`

理由: 関数サイズが大きく（25-55行）、インライン化すると可読性が低下。同一ファイル内のプライベート関数として適切。

### 4. O1 条例違反検知の追加

`scripts/validate_process_contracts.py` に O1 チェックを追加:

- Process ラッパーが存在するプライベート関数をテストが直接 import している場合を検出
- 既知の Process ラッパーマッピング（6関数）を管理
- 契約違反（C3-C16）とは別に「条例違反」として報告（警告レベル）

## テスト結果

- **315 passed** (xkep_cae/ 全体)
- C14/C16 契約違反: **0件**
- O1 条例違反: **0件**
- ruff check/format: **全パス**

## 互換ヒストリー

| 旧 | 新 | status |
|----|----|----|
| `_deformed_coords()` 直接呼び出し（Process 内） | `DeformedCoordsProcess` API 経由 | status-192 |
| `_ncp_line_search()` 直接呼び出し（Process 内） | `NCPLineSearchProcess` API 経由 | status-192 |
| `_friction_strategy._k_pen = ...` 直接代入 | `_friction_strategy.set_k_pen(...)` | status-192 |
| `getattr(strategy, "_n_uzawa_max", 5)` | `getattr(strategy, "n_uzawa_max", 5)` | status-192 |

## TODO

- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動 Phase2 xfail 解消
- [ ] LOW: 状態操作ユーティリティ（`_state_set`, `_save_checkpoint` 等）の Process 化検討

---
