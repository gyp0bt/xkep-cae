# status-184: Phase 4 — ContactFrictionProcess 移行 + 完全ワークフロー実現

[<- README](../../README.md) | [<- status-index](status-index.md)

## 日付

2026-03-16

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-6VTbe

## 概要

status-183 の TODO に基づく Phase 4 実装:

1. **ContactFrictionProcess 移行**: deprecated の摩擦接触ソルバーを新 xkep_cae に SolverProcess として完全書き直し
2. **StrandBendingBatchProcess v3.0.0**: ContactFrictionProcess を uses に追加し、Mesh→Setup→Solver の完全ワークフローを実現

## 変更内容

### 1. ContactFrictionProcess（SolverProcess）

| ファイル | テスト数 |
|---------|---------|
| `xkep_cae/contact/solver/process.py` | 11 |

- SolverProcess を継承した統一摩擦接触ソルバー
- deprecated モジュール（SolverState, NewtonUzawaLoop, AdaptiveLoadController）を importlib 経由で使用（C14 準拠）
- deprecated 版の default_strategies を内部で使用（NewtonUzawaLoop との互換性のため）
- StrategySlot 5軸: penalty, friction, time_integration, contact_force, contact_geometry

### 2. StrandBendingBatchProcess v3.0.0

- バージョン: 2.0.0 → 3.0.0
- uses: 15 → 16 プロセス（ContactFrictionProcess 追加）
- StrandBatchConfig に boundary / callbacks / run_solver を追加
- process(): run_solver=True + boundary + callbacks 指定時にソルバーを実行
- テスト: 11 → 13（ソルバースキップ + ソルバー統合追加）

### 3. 設計ドキュメント

- `xkep_cae/contact/solver/docs/contact_friction.md`

## テスト結果

- 新規テスト: **13テスト**（ContactFrictionProcess 11 + StrandBendingBatch 2追加 = 合計 13新規）
- 合計: 266 + 13 = **279テスト**（新パッケージ）
- ruff check: 0 error
- ruff format: 0 issue
- 契約違反: **0件**

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| deprecated ContactFrictionProcess | `xkep_cae/contact/solver/process.py` | status-184 |
| StrandBendingBatchProcess v2.0.0 | v3.0.0（Solver 統合） | status-184 |

## TODO

- [ ] Phase 5〜8: 残りの concrete プロセス移行（Export/Render/Verify のソルバー結果連携）
- [ ] deprecated 依存の段階的除去（NewtonUzawaLoop/SolverState の新パッケージ書き直し）
- [ ] S3 凍結解除: 変位制御7本撚線曲げ揺動の Phase 2 xfail 解消

## 懸念事項・メモ

- ContactFrictionProcess は内部で deprecated の NewtonUzawaLoop / SolverState / AdaptiveLoadController を importlib 経由で使用。NR ループ本体の書き直しは CLAUDE.md の「NCP ソルバーの収束ロジック変更禁止」に抵触する可能性があるため、現時点では deprecated 委譲が正解。
- deprecated 版の default_strategies を使用する理由: 新パッケージの Strategy Process と deprecated の Strategy Protocol のインターフェースが異なるため。NewtonUzawaLoop は deprecated 版の Protocol を期待する。

---
