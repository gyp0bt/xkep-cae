# status-198: Phase 14 — S3 xfail テスト Process API 対応版作成

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-17

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-Ok6yi

## 概要

status-197 の TODO「Phase 14: S3 xfail テストの Process API 対応版作成」を実行。
新 xkep_cae パッケージの Process API のみ（deprecated import なし）で
撚線接触テストを構成。循環参照の修正も実施。

## アーキテクチャ

```
tests/contact/
└── test_strand_contact_process.py  # 11テスト（8 passed, 2 xfailed, 1 xpassed）

xkep_cae/contact/geometry/
└── __init__.py                     # 循環参照修正（遅延インポート化）
```

## 変更内容

### 1. 新規テストファイル: test_strand_contact_process.py（11テスト）

| クラス | テスト数 | 内容 |
|--------|----------|------|
| TestStrandMeshProcessAPI | 2 | StrandMeshProcess のメッシュ生成 (7本/19本) |
| TestContactSetupProcessAPI | 1 | ContactSetupProcess の初期化 + 候補検出 |
| TestSevenStrandRadialProcessAPI | 1 | 7本撚線の線形梁径方向圧縮 (CFP 収束) |
| TestSevenStrandBendingProcessAPI | 2 | 7本 UL CR 梁 45°/90° 曲げ (CFP 収束) |
| TestSevenStrandBendingOscillationProcess | 1 | 7本 曲げ + 揺動 (xfail: 活性セット変動) |
| TestStrandFrictionProcess | 1 | 7本 CR梁摩擦曲げ (xfail: 摩擦接線剛性) |
| TestLargeStrandProcessAPI | 1 | 19本径方向圧縮 (xfail: CI timeout) |
| TestStrandBendingPhysicsProcess | 2 | 物理検証（先端変位方向 + 法線力正値） |

### 2. 使用 Process API コンポーネント

| コンポーネント | パス | 役割 |
|---------------|------|------|
| StrandMeshProcess | xkep_cae/mesh/process.py | 撚線メッシュ生成 |
| ContactSetupProcess | xkep_cae/contact/setup/process.py | 接触初期化 |
| ContactFrictionProcess | xkep_cae/contact/solver/process.py | 摩擦接触ソルバー |
| ULCRBeamAssembler | xkep_cae/elements/_beam_assembler.py | UL CR 梁アセンブラ |
| BeamSection | xkep_cae/elements/_beam_section.py | 断面特性 |

### 3. 循環参照修正: geometry/__init__.py

- 問題: `_contact_pair.py` → `geometry._compute` → `geometry/__init__` → `geometry.strategy` → `_contact_pair.py`
- 修正: `geometry/__init__.py` の eager import を `__getattr__` 遅延インポートに変更
- 影響: geometry パッケージからの公開 API アクセスは従来通り動作

### 4. テスト結果

| テスト | 結果 | 備考 |
|--------|------|------|
| 8 passed | ✅ | API + 物理検証テスト |
| 2 xfailed | ✅ | 揺動不収束 + 摩擦接線剛性 (既知の物理問題) |
| 1 xpassed | ⚠️ | 19本径方向が CI 時間内に収束 (strict=False) |
| C14 違反 | **0 件** | deprecated import なし |
| ruff check | エラー 0 件 | (既存の test_dynamics.py I001 は今回無関係) |
| ruff format | 全ファイルフォーマット済み | |

### 5. xfail テストの対応状況

| 旧テスト | 新テスト | 状態 |
|---------|---------|------|
| test_bending_oscillation::7strand_full | TestSevenStrandBendingOscillationProcess | xfail (活性セット変動) |
| test_friction_validation (7件) | TestStrandFrictionProcess | xfail (摩擦接線剛性) |
| test_real_beam_contact::cr_friction | TestStrandFrictionProcess | xfail (摩擦接線剛性) |
| test_convergence_19strand::19strand | TestLargeStrandProcessAPI | xpass (CI timeout) |
| test_bending_oscillation (numerical_tests依存) | 対象外 | numerical_tests 未移行 |
| test_coated_wire_integration | 対象外 | 被膜モデル物理検証が必要 |
| test_ring_compliance_s2 | 対象外 | 既に新パッケージのみ使用 |

## TODO

- [ ] 19本 xpass テストを正式 passing テストに昇格（CI 時間計測後）
- [ ] 被膜モデルの物理検証テスト（CoatingStrategy + ContactFrictionProcess）
- [ ] numerical_tests.wire_bending_benchmark の Process API 版再実装

## 開発運用メモ

- geometry/__init__.py の循環参照は `__getattr__` 遅延インポートで解決。他のパッケージでも同様のパターンが適用可能
- gap=0.15 の曲げテストでは接触が発生しない（弦近似誤差防止のため）。接触が必要な場合は gap=0 を使用
- ContactFrictionProcess は常に smooth_penalty + use_friction=True で動作する。mu=0 で実質無摩擦
- ULCRBeamAssembler + ContactFrictionProcess の統合が初めてテストで検証された

---
