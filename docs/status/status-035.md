# status-035: .inpパーサー拡張（*MATERIAL / *ELASTIC / *DENSITY / *PLASTIC）+ pyproject.toml更新

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-18

## 概要

status-034 の TODO を消化。Abaqus .inp パーサーに材料定義キーワード群（*MATERIAL, *ELASTIC, *DENSITY, *PLASTIC）を追加。
pyproject.toml に matplotlib をオプショナル依存として追加。
テスト数 741 → 753（+12テスト）。

## 実施内容

### 1. pyproject.toml 更新

- `[project.optional-dependencies]` に `plot = ["matplotlib>=3.5"]` を追加
- FIELD ANIMATION出力モジュールが matplotlib に依存するため、オプショナル依存として明示

### 2. .inp パーサー材料キーワード追加 (`xkep_cae/io/abaqus_inp.py`)

#### `*MATERIAL` キーワード追加
- `NAME=` オプションで材料名を指定
- 後続の `*ELASTIC`, `*DENSITY`, `*PLASTIC` サブキーワードがこの材料に紐づく
- `AbaqusMaterial` データクラス新設
- `AbaqusMesh.materials` フィールド追加（`list[AbaqusMaterial]`）
- `AbaqusMesh.get_material(name)` メソッド追加（大文字小文字区別なし）

#### `*ELASTIC` キーワード追加
- データ行: `E, nu`（nu省略時はデフォルト0.0）
- 直前の `*MATERIAL` に紐づけ
- `AbaqusMaterial.elastic` に `(E, nu)` タプルとして格納

#### `*DENSITY` キーワード追加
- データ行: `rho`
- 直前の `*MATERIAL` に紐づけ
- `AbaqusMaterial.density` にスカラーとして格納

#### `*PLASTIC` キーワード追加
- データ行: `sigma_y, eps_p` の表データ（複数行）
- `HARDENING=` オプション対応（ISOTROPIC/KINEMATIC/COMBINED、デフォルトISOTROPIC）
- 直前の `*MATERIAL` に紐づけ
- `AbaqusMaterial.plastic` に `[(sigma_y, eps_p), ...]` リストとして格納
- `AbaqusMaterial.plastic_hardening` に硬化タイプ文字列を格納

#### `io/__init__.py` 更新
- `AbaqusMaterial` を公開APIとしてエクスポート

### 3. テスト (`tests/test_abaqus_inp.py`) — +12テスト

`TestMaterialParsing` クラス新設:
- `test_material_elastic_only`: 弾性定数のみの材料定義
- `test_material_with_density`: 弾性+密度
- `test_material_with_plastic`: 弾性+塑性テーブル
- `test_plastic_kinematic_hardening`: HARDENING=KINEMATIC指定
- `test_plastic_combined_hardening`: HARDENING=COMBINED指定
- `test_multiple_materials`: 複数材料定義
- `test_get_material_case_insensitive`: 大文字小文字区別なし
- `test_get_material_not_found`: 存在しない材料名でKeyError
- `test_elastic_without_nu`: ポアソン比省略
- `test_plastic_without_eps_p`: 塑性ひずみ省略
- `test_full_model_with_material`: 完全モデル統合テスト
- `test_no_materials`: 材料未定義の場合

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `pyproject.toml` | `plot = ["matplotlib>=3.5"]` をオプショナル依存に追加 |
| `xkep_cae/io/abaqus_inp.py` | `*MATERIAL`, `*ELASTIC`, `*DENSITY`, `*PLASTIC` パーサー追加。`AbaqusMaterial` データクラス新設。`AbaqusMesh` に `materials` フィールド・`get_material()` メソッド追加 |
| `xkep_cae/io/__init__.py` | `AbaqusMaterial` のエクスポート追加 |
| `tests/test_abaqus_inp.py` | +12テスト（材料キーワードパーサー） |
| `docs/status/status-034.md` | TODO欄にチェック・*PLASTICキーワードTODO追加 |

## テスト数

741 → 753（+12テスト）

## 対応キーワード一覧（.inp パーサー）

| キーワード | 状態 | 備考 |
|-----------|------|------|
| `*NODE` | 既存 | 2D/3D座標 |
| `*ELEMENT` | 既存 | ELSET=対応、継続行対応 |
| `*NSET` | 既存 | GENERATE対応 |
| `*ELSET` | 既存 | GENERATE対応 |
| `*BEAM SECTION` | 既存 | SECTION/ELSET/MATERIAL |
| `*TRANSVERSE SHEAR STIFFNESS` | 既存 | K11/K22/K12 |
| `*BOUNDARY` | 既存 | 単一DOF/範囲/規定変位 |
| `*OUTPUT, FIELD ANIMATION` | 既存 | xkep-cae独自拡張 |
| `*MATERIAL` | **新規** | NAME=指定 |
| `*ELASTIC` | **新規** | E, nu |
| `*DENSITY` | **新規** | rho |
| `*PLASTIC` | **新規** | HARDENING=対応（ISOTROPIC/KINEMATIC/COMBINED） |

## 確認事項・懸念

1. ***PLASTIC テーブルと既存 Plasticity1D/3D の接続**: パーサーは降伏応力-塑性ひずみテーブルを読み込むが、既存の `Plasticity1D`（`IsotropicHardening` は `sigma_y0` + `H_iso` の線形硬化）や `PlaneStrainPlasticity`（`IsotropicHardening3D` は `sigma_y0` + `H_iso` + Voce）とのマッピングは未実装。テーブル補間型の硬化則（piecewise linear）の実装が必要
2. **export_animation テストの失敗**: matplotlib 未インストール環境では `test_export_animation.py` の15テストが失敗する。pytest マーカーでスキップするか、matplotlib の有無をチェックする `skipIf` の追加を検討

## TODO

- [ ] テーブル補間型硬化則の実装（*PLASTIC テーブル → Plasticity1D/3D への変換）
- [ ] GIF/MP4 出力のサポート（ffmpeg連携）— status-034 から継続
- [ ] 連続体要素のメッシュプロット対応（将来）— status-034 から継続
- [ ] matplotlib 未インストール環境でのテストスキップ対応

---
