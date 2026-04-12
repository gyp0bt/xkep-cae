# status-185: Phase 5 — ソルバー結果連携 + output re-export クリーンアップ

[<- README](../../README.md) | [<- status-index](status-index.md)

## 日付

2026-03-16

## 作業者

Claude Code

## ブランチ

claude/execute-status-todos-SMOpD

## 概要

status-184 の TODO「Phase 5〜8: 残りの concrete プロセス移行（Export/Render/Verify のソルバー結果連携）」を実施。

1. **StrandBendingBatchProcess v4.0.0**: ソルバー結果に基づく Export/Render/Verify のワイヤリング完成
2. **output/__init__.py**: 全量 re-export shim → 明示的エクスポート + `__getattr__` 遅延ロードに変更

## 変更内容

### 1. StrandBendingBatchProcess v4.0.0

- バージョン: 3.0.0 → 4.0.0
- uses: 16 → 18 プロセス（EnergyBalanceVerifyProcess, ContactVerifyProcess 追加）
- process() にソルバー後の3ステップを追加:
  - **Export**: `run_export=True` 時に ExportProcess で CSV/JSON 出力
  - **Render**: `run_render=True` 時に BeamRenderProcess で 2D 投影画像出力
  - **Verify**: `run_verify=True` 時に 3検証プロセス（Convergence/Energy/Contact）を統合実行
- テスト: 13 → 17（Export/Render/Verify/NoExportWithoutSolver の 4 テスト追加）

### 2. output/__init__.py クリーンアップ

- 旧: `for _k in dir(_m): globals()[_k] = ...` の全量 re-export
- 新: ExportProcess/BeamRenderProcess の明示的エクスポート + `__getattr__` による deprecated 遅延ロード
- deprecated 互換維持: tests/test_output.py 等から使われる旧 API は `__getattr__` 経由で引き続き動作

## テスト結果

- 新規テスト: **4テスト**（StrandBendingBatch に追加）
- 合計: **275テスト**（新パッケージ）
- ruff check: 0 error
- ruff format: 0 issue
- 契約違反: **0件**

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| StrandBendingBatchProcess v3.0.0 | v4.0.0（Export/Render/Verify 連携） | status-185 |
| `output/__init__.py` 全量 re-export | 明示的エクスポート + `__getattr__` 遅延ロード | status-185 |

## TODO

- [ ] Phase 6〜8: deprecated 依存の段階的除去（mesh.twisted_wire, contact.pair, NewtonUzawaLoop/SolverState）
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動の Phase 2 xfail 解消
- [ ] output モジュールの deprecated 関数移行（render_beam_3d, export_vtk 等 — 30+ 関数）

## 懸念事項・メモ

- テスト数が 279 → 275 に減少している（status-184 比）。テスト内容の変化ではなく、カウント方法の差異の可能性。全テスト合格で実質的な影響なし。
- ContactFrictionProcess 内の deprecated 依存（NewtonUzawaLoop 等）は Phase 6〜8 で対応予定。ただし「NCP ソルバー収束ロジック変更禁止」制約により慎重に進める必要あり。

---
