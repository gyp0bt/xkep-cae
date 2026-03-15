# Status 106: inp パーサー OOP リファクタリング + k_pen EA/L 対応

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-04
**ブランチ**: `claude/inp-parser-refactor-6Q5YJ`
**テスト数**: 2016（fast: 1637 / slow: 374 + 5）— +37テスト

## 概要

.inp パーサーを手続き的実装から **AbstractKeywordParser + `__init_subclass__`** による OOP フレームワークに全面リファクタリング。スケーラブルなキーワード追加と `*INCLUDE` 再帰読み込みに対応。また k_pen の EA/L ベース自動スケーリングを実装。

## 実施内容

### 1. OOP パーサーフレームワーク（`xkep_cae/io/inp_parser.py` 新規作成）

#### 設計原則

- **AbstractKeywordParser**: `__init_subclass__` でサブクラスを自動登録。新キーワード追加はクラスを1つ書くだけ
- **ParseContext**: ブロックコンテキスト管理（Step / Material / SurfaceInteraction / Contact）
- **InpReader**: `*INCLUDE` 再帰展開 + キーワードディスパッチ

#### 登録パーサー（42個: 21キーワード + 21別名）

| カテゴリ | キーワード | 状態 |
|---------|----------|------|
| 基本 | `*HEADING`, `*NODE`, `*ELEMENT`, `*NSET`, `*ELSET` | ✅ |
| 材料 | `*MATERIAL`, `*ELASTIC`, `*DENSITY`, `*PLASTIC` | ✅ |
| 断面 | `*BEAM SECTION`, `*TRANSVERSE SHEAR STIFFNESS` | ✅ |
| 境界 | `*BOUNDARY`（モデル/ステップ両対応） | ✅ |
| ステップ | `*STEP`, `*END STEP`, `*STATIC`, `*DYNAMIC` | ✅ **新規** |
| 荷重 | `*CLOAD`, `*DLOAD` | ✅ **新規** |
| サーフェス | `*SURFACE`, `*SURFACE INTERACTION`, `*SURFACE BEHAVIOR`, `*FRICTION` | ✅ **新規** |
| 接触 | `*CONTACT`, `*CONTACT INCLUSIONS`, `*CONTACT PROPERTY ASSIGNMENT` | ✅ **新規** |
| 初期条件 | `*INITIAL CONDITIONS` | ✅ **新規** |
| 出力 | `*OUTPUT`, `*NODE OUTPUT`, `*ELEMENT OUTPUT`, `*ENERGY OUTPUT`, `*ANIMATION` | ✅ **新規** |
| ファイル | `*INCLUDE`（再帰展開、深度制限10） | ✅ **新規** |

### 2. AbaqusMesh 拡張

`AbaqusMesh` に以下のフィールドを追加:

| フィールド | 型 | 用途 |
|-----------|---|------|
| `heading` | `str` | ヘッダ文字列 |
| `steps` | `list[InpStep]` | ステップ定義 |
| `initial_conditions` | `list[InpInitialCondition]` | 初期条件 |
| `surfaces` | `list[InpSurfaceDef]` | サーフェス定義 |
| `surface_interactions` | `list[InpSurfaceInteraction]` | サーフェスインタラクション |
| `contact_defs` | `list[InpContactDef]` | コンタクト定義 |

### 3. 旧パーサー関数群の削除

`abaqus_inp.py` から `_parse_node_section` 等の旧パーサー関数群（539行）を完全削除。`read_abaqus_inp` は `InpReader` に委譲。

### 4. k_pen EA/L ベース自動スケーリング

- `ContactConfig.beam_A` フィールド追加
- `k_pen_mode="ea_l"` オプション追加
- `solver_hooks.py` の k_pen 初期化2箇所を `ea_l` 対応
- `_build_contact_manager` に `beam_A` 伝播

### テスト追加

| テストファイル | テスト数 | 内容 |
|--------------|---------|------|
| `tests/test_inp_parser_roundtrip.py` | 31 | ラウンドトリップ、INCLUDE、ステップ、接触、IC、出力、後方互換 |
| `tests/contact/test_auto_beam_kpen.py` | +6 | EA/L テスト（計19件） |

## テスト結果

- 新規テスト: 37/37 パス
- 既存テスト: 回帰なし

## 次の課題（TODO）

- [ ] `build_beam_model_from_inp` ベースの接触解析モデル構築経路（メッシュ直接利用）
- [ ] 19本以上 NCP 収束のパラメータ最適化（status-104 引継ぎ、CLI実行待ち）
- [ ] EA/L モードの最適 scale 値の検証（現状は未チューニング）
- [ ] `*SOLID SECTION`, `*SHELL SECTION` パーサー追加（将来）
- [ ] `*AMPLITUDE`, `*CONNECTOR` パーサー追加（将来）

## 確認事項

- `read_abaqus_inp` の後方互換性は100%保持（114テストで検証済み）
- 新フィールド（steps, surfaces 等）は全てデフォルト空値のため既存コードに影響なし
- `*INCLUDE` は深度制限10、存在しないファイルは警告のみ（エラーにしない）
- k_pen EA/L モードは EI/L³ モードと併存。既存のデフォルト動作は変更なし
