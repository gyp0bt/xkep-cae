# status-197: Phase 13 完了 — ビームアセンブラ新 xkep_cae 移植

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-17

## 作業者

Claude Code

## ブランチ

claude/execute-status-todos-HWcFa

## 概要

status-196 の TODO「Phase 13: ビームアセンブラ（CR/UL）の新 xkep_cae 移植」を実行。
BeamSection, CR梁要素関数群, assemble_cr_beam3d, ULCRBeamAssembler を
deprecated import なし（C14準拠）で移植。四元数関数もインライン化。

## アーキテクチャ

```
xkep_cae/elements/
├── __init__.py             # 公開API: BeamSection, BeamSection2D, BeamForces3D, ULCRBeamAssembler
├── _beam_section.py        # BeamSection / BeamSection2D (frozen dataclass)
├── _beam_cr.py             # CR梁要素関数群 + 四元数ユーティリティ + バッチ版
├── _beam_assembly.py       # assemble_cr_beam3d + バッチアセンブリ
└── _beam_assembler.py      # ULCRBeamAssembler クラス

tests/elements/
├── __init__.py
└── test_beam_assembler.py  # 32テスト（API + 物理検証）
```

## 変更内容

### 1. 新規ファイル（xkep_cae/elements/）

| ファイル | 行数 | 内容 |
|---------|------|------|
| `_beam_section.py` | ~155 | BeamSection / BeamSection2D（frozen dataclass） |
| `_beam_cr.py` | ~760 | CR梁関数群 + 四元数インライン + バッチ版全関数 |
| `_beam_assembly.py` | ~240 | assemble_cr_beam3d + バッチアセンブリ実装 |
| `_beam_assembler.py` | ~195 | ULCRBeamAssembler クラス |
| `__init__.py` | ~25 | 公開API（クラスのみ re-export、C16準拠） |

### 2. C14 準拠の設計判断

- `_rotvec_to_rotmat` / `_rotmat_to_rotvec` が deprecated の `math.quaternion` に依存していた
- 四元数関数（`_quat_from_rotvec`, `_quat_to_rotmat`, `_rotmat_to_quat`, `_quat_to_rotvec`）を
  `_beam_cr.py` にインライン化して deprecated import を完全排除
- バッチ版関数は元々自己完結していたためそのまま移植

### 3. C16 準拠の設計判断

- `__init__.py` からは frozen dataclass とクラスのみ re-export
- 純粋関数（`assemble_cr_beam3d`, `timo_beam3d_ke_local` 等）は `_` prefix のプライベートモジュール内に配置
- テストはプライベートモジュールから直接 import

### 4. テスト（32件）

| クラス | テスト数 | 内容 |
|--------|----------|------|
| TestBeamSectionAPI | 10 | BeamSection/2D の API・バリデーション |
| TestStiffnessAPI | 3 | 剛性行列の対称性・正定値性 |
| TestMassAPI | 4 | 質量行列（集中・整合） |
| TestCRBeamAPI | 3 | CR内力・接線剛性（解析vs数値） |
| TestAssemblyAPI | 4 | グローバルアセンブリ（sparse/dense/数値） |
| TestULCRBeamAssemblerAPI | 6 | UL アセンブラ API + checkpoint/rollback |
| TestBeamPhysics | 2 | 片持梁先端荷重収束 + 軸伸び解析解 |

## テスト結果

- `tests/elements/test_beam_assembler.py`: **32 passed** in 0.14s
- C14 違反: **0 件**
- C16 違反: **0 件**（elements 関連）
- ruff check: エラー 0 件
- ruff format: 全ファイルフォーマット済み

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `__xkep_cae_deprecated.elements.beam_timo3d.ULCRBeamAssembler` | `xkep_cae.elements._beam_assembler.ULCRBeamAssembler` | status-197 |
| `__xkep_cae_deprecated.elements.beam_timo3d.assemble_cr_beam3d` | `xkep_cae.elements._beam_assembly.assemble_cr_beam3d` | status-197 |
| `__xkep_cae_deprecated.sections.beam.BeamSection` | `xkep_cae.elements._beam_section.BeamSection` | status-197 |
| `__xkep_cae_deprecated.sections.beam.BeamSection2D` | `xkep_cae.elements._beam_section.BeamSection2D` | status-197 |
| `__xkep_cae_deprecated.math.quaternion` 関数群（rotvec↔rotmat用） | `xkep_cae.elements._beam_cr` にインライン化 | status-197 |

## TODO

- [ ] Phase 14: S3 xfail テストの Process API 対応版作成（Phase 13 完了後）
  - 新ビームアセンブラ + ContactFrictionProcess で曲げ揺動テスト
  - deprecated を使わないとパスしないテストは対象外

## 開発運用メモ

- C16 では `__init__.py` から純粋関数を公開 re-export するのは禁止。クラス・frozen dataclass・Enum のみ
- 四元数のインライン化は ~50行増だが、deprecated import 排除のために必要な判断
- バッチ版関数を含めたフルセットの移植により、大規模問題でのパフォーマンスも新パッケージで維持

---
