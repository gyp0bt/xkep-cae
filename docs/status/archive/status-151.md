# status-151: プロセスアーキテクチャ Phase 1 — 基盤 + Strategy Protocol

[← status-index](status-index.md) | [← README](../../README.md)

**日付**: 2026-03-10
**テスト数**: 2310（+39）

## 概要

status-150 で策定した設計仕様（xkep_cae/process/process-architecture.md）に基づき、
プロセスアーキテクチャの Phase 1（基盤 + Strategy Protocol）を実装した。
39テストを追加し、全テスト通過・lint clean。

## 成果物

### 新規ファイル（17ファイル）

| ファイル | 内容 | テスト数 |
|---------|------|---------|
| `xkep_cae/process/__init__.py` | 公開API | — |
| `xkep_cae/process/base.py` | AbstractProcess, ProcessMeta, ProcessMetaclass | 12 |
| `xkep_cae/process/test_base.py` | base.py の 1:1 テスト | — |
| `xkep_cae/process/categories.py` | PreProcess, SolverProcess, PostProcess, VerifyProcess, BatchProcess | 5 |
| `xkep_cae/process/test_categories.py` | categories.py の 1:1 テスト | — |
| `xkep_cae/process/testing.py` | binds_to デコレータ（1:1紐付け） | 2 |
| `xkep_cae/process/test_testing.py` | testing.py の 1:1 テスト | — |
| `xkep_cae/process/data.py` | MeshData, SolverInputData 等 データ契約型 | — |
| `xkep_cae/process/tree.py` | ProcessTree, ProcessNode, NodeType | 6 |
| `xkep_cae/process/test_tree.py` | tree.py の 1:1 テスト | — |
| `xkep_cae/process/strategies/__init__.py` | Strategy公開API | — |
| `xkep_cae/process/strategies/protocols.py` | 5つの Strategy Protocol | 14 |
| `xkep_cae/process/strategies/test_protocols.py` | Protocol準拠 + 契約 + 互換性テスト | — |
| `xkep_cae/process/strategies/compatibility.py` | 互換性マトリクス（WL/BL） | — |
| `xkep_cae/process/concrete/__init__.py` | Phase 3 用スタブ | — |
| `xkep_cae/process/verify/__init__.py` | Phase 4 用スタブ | — |
| `xkep_cae/process/batch/__init__.py` | Phase 5 用スタブ | — |

### 実装された機能

1. **AbstractProcess基底クラス**: メタクラスによる自動トレース + プロファイリング
2. **ProcessMetaclass**: process() 自動ラップ、call stack、profile data
3. **5カテゴリ中間クラス**: Pre/Solver/Post/Verify/Batch
4. **binds_to デコレータ**: テスト-プロセス 1:1紐付け、重複検出
5. **データ契約型**: MeshData, BoundaryData, ContactSetupData, AssembleCallbacks, SolverInputData, SolverResultData, VerifyInput, VerifyResult
6. **5つの Strategy Protocol**: ContactForce, Friction, TimeIntegration, ContactGeometry, Penalty
7. **互換性マトリクス**: 検証済み3構成 + 非互換1構成（status-147）
8. **ProcessTree**: 実行グラフ + 依存バリデーション + Mermaid/Markdown出力

### 設計判断

| 判断 | 選択 | 理由 |
|------|------|------|
| 抽象クラス判定 | `ABC in __bases__` + `__abstractmethods__` | ABCミックスイン時の誤検出回避 |
| 循環検出 | `id(node)` ベースの seen set | ProcessNode 自己参照でも安全 |
| data.py の NCPSolverInput 変換 | Phase 3 で実装 | 既存コードへの影響を最小化 |
| ContactManager の型 | `object`（循環参照回避） | Phase 3 でプロトコル化 |

## TODO（次セッション: Phase 2）

- [ ] Phase 2 開始: Strategy 具象実装（solver_ncp.py からの抽出）
  - strategies/contact_ncp.py — NCPContactForce
  - strategies/contact_smooth.py — SmoothPenaltyContactForce
  - strategies/friction_*.py — 摩擦3バリアント
  - strategies/time_*.py — QuasiStatic + GeneralizedAlpha
  - strategies/geometry_*.py — PtP + L2L
  - strategies/penalty_*.py — 自動推定 + continuation
- [ ] 各strategy の 1:1 テスト
- [ ] U1 判断: Strategy を Process として維持するか Protocol に降格するか（オーバーヘッド計測）

## 懸念事項・確認事項

- **C7 対策（メタクラスのラップ漏れ）**: monkey-patch 対策は Phase 1 では未実装。Phase 2 で execute() 内チェックを追加予定。
- **C8 対策（動的 uses）**: NCPContactSolverProcess の動的 uses は Phase 3 で _runtime_uses で管理予定。
