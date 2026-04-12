# status-196: S3 xfail テスト Process API 対応の前提条件調査

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-17

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-ZxOSQ

## 概要

status-195 の TODO「S3 xfail テストの Process API 対応版作成」を調査した結果、
**ビームアセンブラの新 xkep_cae への移植が前提条件**であることが判明。

## 調査結果

### S3 xfail テスト一覧（17件）

| ファイル | テスト数 | 主な原因 |
|---------|----------|---------|
| `tests/contact/test_bending_oscillation.py` | 2 | Phase2 接触活性セット変動不収束 |
| `tests/contact/test_convergence_19strand.py` | 5 | CI タイムアウト (>600s) |
| `tests/contact/test_friction_validation.py` | 7 | NCP 摩擦接線剛性符号問題 |
| `tests/contact/test_real_beam_contact.py` | 1 | CR 梁摩擦接触不収束 |
| `tests/contact/test_coated_wire_integration.py` | 1 | 被膜モデル接触剛性不足 |
| `tests/mesh/test_ring_compliance_s2.py` | 1 | 被膜モデル変更後の非単調半径 |

### Process API 対応に必要なコンポーネント

| コンポーネント | 新 xkep_cae | deprecated | 状態 |
|---------------|-------------|------------|------|
| ContactFrictionProcess | ✅ | — | 移行済み |
| ContactSetupProcess | ✅ | — | 移行済み |
| StrandMeshProcess | ✅ | — | 移行済み |
| AssembleCallbacks | ✅（型定義のみ） | — | — |
| **ULCRBeamAssembler** | **❌ 未移植** | ✅ | **ブロッカー** |
| **assemble_cr_beam3d()** | **❌ 未移植** | ✅ | **ブロッカー** |
| BeamSection | ❌ 未移植 | ✅ | 依存 |

### ブロッカー: ビームアセンブラが新パッケージに存在しない

`ContactFrictionProcess` は `AssembleCallbacks` 経由でアセンブリ関数を受け取るが、
その実体（`ULCRBeamAssembler.assemble_tangent`/`assemble_internal_force`）は
`__xkep_cae_deprecated/elements/beam_timo3d.py` にしか存在しない。

S3 xfail テスト（曲げ揺動）は CR 梁要素の幾何学的非線形問題であり、
UL（Updated Lagrangian）アセンブラなしでは物理的に意味のあるテストが構成できない。

### 移植の規模見積もり

| 対象 | 行数 | 依存 |
|------|------|------|
| `ULCRBeamAssembler` クラス | ~220 | assemble_cr_beam3d, BeamSection |
| `assemble_cr_beam3d()` 関数 | ~190 | CR 要素剛性行列関数群 |
| CR 梁要素関数群 | ~600 | ローカルのみ |
| `BeamSection` データクラス | ~50 | numpy のみ |
| **合計** | **~1060** | — |

## 結論

S3 xfail テストの Process API 対応は、以下の順序で進める必要がある:

1. **Phase 13**: ビームアセンブラ（CR/UL）の新 xkep_cae 移植
   - 移植先: `xkep_cae/elements/beam/` 新パッケージ
   - `BeamSection`, `assemble_cr_beam3d`, `ULCRBeamAssembler` の3点
   - C14/C16 準拠
2. **Phase 14**: S3 xfail テストの Process API 対応版作成
   - `ContactFrictionProcess` + 新ビームアセンブラで曲げ揺動テスト

## TODO

- [ ] Phase 13: ビームアセンブラ（CR/UL）の新 xkep_cae 移植（~1060行）
- [ ] Phase 14: S3 xfail テストの Process API 対応版作成（Phase 13 完了後）

## 開発運用メモ

- deprecated を使わないとパスできないテストは作成しないこと
- テスト作成は、必要なコンポーネントの移植が完了してから行う
- 前提条件の調査を先に行い、ブロッカーを早期に特定する

---
