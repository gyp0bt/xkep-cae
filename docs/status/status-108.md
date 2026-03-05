# Status 108: deprecatedテストNCP移行版 + 互換ヒストリー規約 + S3改良テスト追加

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-05
**ブランチ**: `claude/update-coding-standards-Q4P97`
**テスト数**: 2101（fast: 1651+6 / slow: 374 + 5 + 65）— +71テスト

## 概要

status-107で deprecated 化した5つの旧ソルバーテストファイルのNCP移行版を作成。同時にCLAUDE.mdに互換ヒストリー規約を追加し、19本NCP収束テストにS3改良1-5パラメータを追加。

## 実施内容

### CLAUDE.md 互換ヒストリー規約

機能の置き換え・統合を行った場合のルールをCLAUDE.mdに追加:
1. 互換ヒストリーテーブルをstatusファイルに記録
2. 推奨構成が確立されたらデフォルトを切り替え、旧機能はdeprecatedマーカーを付与
3. deprecatedテストは対応するNCP移行版テストを作成
4. deprecated化から2 status以上経過で旧コードの削除を検討

### NCP移行版テスト作成（5ファイル、65テスト）

| 旧テスト（deprecated） | NCP移行版 | テスト数 | 内容 |
|----------------------|----------|---------|------|
| `test_beam_contact_penetration.py`（20テスト） | `test_beam_contact_penetration_ncp.py` | 17 | 接触検出、貫入制限、法線力、摩擦、変位履歴、マルチセグメント、横スライド |
| `test_large_scale_contact.py`（11テスト） | `test_large_scale_contact_ncp.py` | 11 | DOFスケーリング、broadphase効率、16seg収束、スケーラビリティ |
| `test_real_beam_contact.py`（21テスト） | `test_real_beam_contact_ncp.py` | 15 | Timo3D/CR梁接触検出、貫入制限、マルチセグメント、一致性、摩擦、スライド |
| `test_twisted_wire_contact.py`（72テスト） | `test_twisted_wire_contact_ncp.py` | 17 | 3本/7本撚り引張・ねじり・曲げ・横力・摩擦・Line contact |
| `test_coated_wire_integration.py`（20テスト） | `test_coated_wire_integration_ncp.py` | 5 | 被膜付き3本撚り接触（引張・横力・曲げ・摩擦・剛性比較） |

### 19本NCP収束テスト改良追加（6テスト）

| テスト | 内容 |
|-------|------|
| `test_ncp_19strand_s3_improvements` | S3改良1-5全有効化（chattering_window=3, lambda_warmstart_neighbor=True）で19本収束試行 |
| `test_chattering_window_7strand[0]` | chattering_window=0（無効）で7本収束確認 |
| `test_chattering_window_7strand[3]` | chattering_window=3で7本収束確認 |
| `test_chattering_window_7strand[5]` | chattering_window=5で7本収束確認 |
| `test_warmstart_7strand[False]` | lambda_warmstart_neighbor=False で7本収束確認 |
| `test_warmstart_7strand[True]` | lambda_warmstart_neighbor=True で7本収束確認 |

### 旧テストファイルへのマイグレーションコメント追加

5つのdeprecatedテストファイル冒頭に `# DEPRECATED: NCP版は test_xxx_ncp.py を参照` コメントを追加。

## テスト数内訳

- 旧: 2030（fast: 1651 / slow: 374 + 5）
- 新: 2101（NCP移行版65テスト + 改良テスト6テスト = +71）
- deprecated: 144テスト（変更なし、`-m "not deprecated"` で除外可能）

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status | 状態 |
|--------|--------|-----------|------|
| `newton_raphson_with_contact`（ペナルティ/AL） | `newton_raphson_contact_ncp`（NCP） | status-107（deprecated化）→ status-108（NCP移行版テスト作成） | NCP移行版テスト作成完了 |

## 次の課題（TODO）

- [ ] 19本NCP収束テスト: S3改良1-5有効化での収束確認（CIで結果確認）
- [ ] chattering_window の19本以上での最適値チューニング
- [ ] lambda_warmstart_neighbor の19本以上での効果検証
- [ ] 自動安定時間増分: 接触状態変化率に基づくΔt自動制御の設計
- [ ] マルチレベル前処理（AMG）の検討
- [ ] deprecated旧テストの削除判断（status-110以降で検討）

## 確認事項

- NCP移行版テストは全てslowマーカー付き（CIでの実行を想定）
- 旧テストは引き続きdeprecatedマーカーで除外可能
- CLAUDE.mdの互換ヒストリー規約により、今後の機能置き換え時にも追跡可能

## 運用メモ

- `pytest -m "not deprecated"` でNCP版含む1957テストを実行
- `pytest -m "not slow and not deprecated"` でfastテスト1651+6テストを実行
- NCP移行版はslowテストのため、CI環境でのみ実行推奨
