# status-177: ドキュメント再編 — 新 xkep_cae 用にドキュメント全体を再構成

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-15

## 概要

status-176 TODO の消化（C16 純粋関数 `_` prefix 化）と、
脱出ポット計画（status-175）に合わせた全ドキュメントの新 xkep_cae 対応再編。

## 変更内容

### 1. status-176 TODO 消化: C16 純粋関数 `_` prefix 化

`xkep_cae/process/strategies/penalty/law_normal.py` の5関数を private 化:

| 旧名 | 新名 | 理由 |
|------|------|------|
| `softplus()` | `_softplus()` | Process 内部ヘルパー |
| `evaluate_al_normal_force()` | `_evaluate_al_normal_force()` | ALNormalForceProcess が存在 |
| `evaluate_smooth_normal_force()` | `_evaluate_smooth_normal_force()` | SmoothNormalForceProcess が存在 |
| `evaluate_smooth_normal_force_vectorized()` | `_evaluate_smooth_normal_force_vectorized()` | private 化 |
| `auto_beam_penalty_stiffness()` | `_auto_beam_penalty_stiffness()` | Strategy 内部ヘルパー |

- `penalty/__init__.py` の `__all__` から5関数を除去
- テストは `law_normal.py` から直接 `_` prefix 付きでインポート
- C16 契約違反: **0件に解消**

### 2. status ファイル 097〜174 をアーカイブ

- `docs/status/status-097.md` 〜 `docs/status/status-174.md`（78ファイル）を `docs/status/archive/` に移動
- `docs/status/s3-completed.md` を `docs/status/archive/` に移動
- 理由: 旧 xkep_cae（S3/R1 フェーズ）のステータスであり、新 xkep_cae の作業と区別するため

### 3. ドキュメント再編

#### README.md
- パッケージ構成を新 `xkep_cae/` + 旧 `xkep_cae_deprecated/` の2構成に更新
- モジュール別ドキュメントリンクを `xkep_cae_deprecated/` 配下に修正
- 現在の状態テーブルに「脱出ポット計画」を追加

#### docs/design/README.md
- 新 xkep_cae セクション（penalty.md のみ）+ アーカイブセクション（旧文書）に再構成
- 全リンクを `xkep_cae_deprecated/` パスに修正

#### docs/reference/examples.md
- 旧 API であることを明示（冒頭注意書き追加）
- 全インポートパスを `xkep_cae_deprecated.` に修正

#### docs/reference/abaqus-differences.md
- コード例のインポートパスを `xkep_cae_deprecated.` に修正

#### docs/roadmap.md
- 設計仕様書リンクを `xkep_cae_deprecated/` パスに修正
- S3完了済みリンクをアーカイブパスに修正
- モジュール README リンクを `xkep_cae_deprecated/` パスに修正

#### docs/status/status-index.md
- アクティブ status を 175〜 のみに変更（新 xkep_cae 用）
- 097〜174 をアーカイブセクション化（マイルストーン抜粋）
- テスト数推移に「脱出ポット Phase 1」マイルストーン追加

## テスト数

~2260+34p(新)（変更なし）

## 契約違反

5件（C6×4 + C12×1）— 脱出ポット計画 Phase 2 以降で解消予定

## TODO

- [ ] 脱出ポット計画 Phase 2: FrictionStrategy 移行 → docs に新ドキュメント追加
- [ ] 新 xkep_cae モジュール追加時に docs/design/README.md を更新
- [ ] docs/reference/examples.md の新 API 版を作成（移行完了後）

---
