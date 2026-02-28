# Status 088: CI test-slow タイムアウト修正（3並列シャード分割）

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日付**: 2026-02-28
- **ブランチ**: claude/fix-ci-timeout-oWvzs
- **テスト数**: 1822（fast: 1525 / slow: 297）— テスト数変更なし

---

## 概要

CI の `test-slow` ジョブが毎回 45分付近でタイムアウトにより打ち切られていた問題を修正。
295件の slow テストを3並列シャードに分割し、各シャード ~98件（~45分以内）で完了するようにした。

## 原因分析

- `test-slow` ジョブの `timeout-minutes: 45` に対し、slow テストが **295件** 存在
- 全量実行に推定 **~136分** 必要（45分で33%・約97件のみ完了）
- `test_graph_dissipation_series PASSED [ 33%]` 以降でジョブがキャンセルされていた

### テスト分布（295件の内訳）

| ファイル | テスト数 |
|---------|---------|
| test_twisted_wire_contact.py | 77 |
| test_numerical_tests.py | 70 |
| test_dynamics.py | 66 |
| test_real_beam_contact.py | 21 |
| test_coated_wire_integration.py | 20 |
| test_beam_contact_penetration.py | 20 |
| test_large_scale_contact.py | 11 |
| test_abaqus_validation_elastoplastic.py | 5 |
| test_mortar_twisted_wire.py | 5 |

## 修正内容

### `.github/workflows/ci.yml`

**変更前**:
- 単一ジョブ、`timeout-minutes: 45`、全295テストを逐次実行

**変更後**:
- **3並列シャード**（matrix strategy: `shard: [0, 1, 2]`）
- `timeout-minutes: 60`（各シャードに余裕を持たせた）
- `fail-fast: false`（1シャード失敗でも他シャード継続）
- テスト収集 → `awk NR%3==shard` で均等分割 → 各シャード ~98件を実行

**メリット**:
- 新規依存パッケージ不要（pytest-split等は使わない）
- テスト追加時にシャード再定義不要（自動で均等分割される）
- 壁時間: ~136分 → ~60分（並列化効果）

## TODO

- [ ] CI グリーン確認（push後のGitHub Actionsで3シャード全通過を確認）
- [ ] status-087 のTODO残項目（大規模スピードアップ測定、Broadphaseグリッド並列化、Phase S3ベンチマーク）

## 確認事項・懸念

- `awk NR%3==shard` による均等分割は、テストの実行時間ではなくテスト数ベースの分割。テスト時間に偏りがある場合、特定シャードが遅くなる可能性がある。実際にタイムアウトが再発する場合は、シャード数を4に増やすか、テスト時間ベースの分割（pytest-split）導入を検討。
- `fail-fast: false` により1シャード失敗でも他が継続するため、CI全体の失敗通知が遅れる可能性がある。

## 開発運用メモ

- **効果的**: status TODO ベースのタスク管理は引き継ぎに有効
- **非効果的**: CI の timeout が slow テスト増加に追従していなかった。テスト数が大幅に増えた際は CI 設定の見直しを定期的に行うべき

---
