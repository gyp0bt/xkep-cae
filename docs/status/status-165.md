# status-165: Phase 8 完遂 — ManualPenaltyProcess CompatibilityProcess 移行 + Phase 9 計画

[← README](../../README.md) | [← status-index](status-index.md) | [← status-164](status-164.md)

**日付**: 2026-03-14
**テスト数**: 2477（回帰テスト変更なし）+ 315 process テスト（314→315: +1テスト）

## 概要

status-164 の TODO 3件を全て消化。ManualPenaltyProcess の CompatibilityProcess 移行により
C13 チェックが実効化。機械的ガードの現状評価を実施し、Phase 9 計画を策定。

## 実施内容

### ManualPenaltyProcess → CompatibilityProcess 移行

- `penalty.py`: 基底クラスを `SolverProcess` → `CompatibilityProcess` に変更
- `test_penalty.py`: `test_is_compatibility_process` テスト追加（+1テスト）
- 影響分析: 本番コードで ManualPenaltyProcess を `uses` に宣言しているプロセスは0件
- `validate_process_contracts.py` 実行: **契約違反 0件**を確認

### CI コメント更新

- `ci.yml`: test-process ジョブのテスト数コメントを `~277` → `~315` に修正

### 機械的ガードの現状評価

| チェック | 状態 | 機械的強制力 | 残りの穴 |
|---------|------|------------|---------|
| C3 テスト紐付け | OK | @binds_to 強制 | deprecated はスキップ（意図的） |
| C5 未宣言依存 | OK | AST 解析 | 動的 import/getattr 検出不可（現状未使用） |
| C6 Strategy 意味論 | OK | テスト存在確認 | テストの「質」は検証不可 |
| C7 メタクラスラップ | OK | 自動検出 | — |
| C8 動的依存 | OK | StrategySlot 整合性 | NCPContactSolverProcess のみ対象 |
| C9 frozen 不変性 | OK | execute() チェックサム | ランタイム検証は機能中 |
| C11 推移的依存 | OK | ヒューリスティック | 間接呼び出し検出不可 |
| C12 Batch 順序 | OK | uses 宣言検証 | — |
| C13 Compat uses 禁止 | OK | **今回実効化** | ManualPenaltyProcess 移行完了 |

**最大の穴**: `_import_all_modules()` のモジュールリストがハードコード。新規プロセス追加時に登録忘れるとチェック対象外になる → Phase 9-A で対策。

### Phase 9 計画策定

- **9-A**: `_import_all_modules()` をファイルシステム走査ベースに変更（ハードコード廃止）
- **9-B**: `_runtime_uses` → `StrategySlot` 完全移行（後方互換コード除去）
- **9-C**: S3 凍結解除判断（Phase 8 完了により基盤整備完了）
- **9-D**: BatchProcess パイプライン改善（S3 再開しない場合の代替）

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/process/strategies/penalty.py` | ManualPenaltyProcess 基底クラス変更 |
| `xkep_cae/process/strategies/tests/test_penalty.py` | test_is_compatibility_process 追加 |
| `.github/workflows/ci.yml` | test-process テスト数コメント修正 |
| `docs/roadmap.md` | Phase 9 計画追記 |
| `README.md` | テスト数・状態更新 |
| `CLAUDE.md` | テスト数・フォーカスガード更新 |
| `docs/status/status-165.md` | 本ファイル |
| `docs/status/status-index.md` | インデックス追加 |

## TODO（次セッション）

- [ ] Phase 9-A 実装: `_import_all_modules()` ファイルシステム走査化
- [ ] Phase 9-B 実装: `_runtime_uses` → `StrategySlot` 完全移行
- [ ] S3 凍結解除判断（9-C）

## 運用メモ

- ManualPenaltyProcess の CompatibilityProcess 移行は C13 チェックの「実弾装填」に相当。
  これにより、新規プロセスが deprecated な ManualPenaltyProcess を uses に宣言すると
  CI で即座に検出される。
- 機械的ガード全9項目が全て実効化した状態になった。
  残る弱点は `_import_all_modules()` のハードコードのみ。
