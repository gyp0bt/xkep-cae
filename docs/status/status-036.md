# status-036: テーブル補間型硬化則 + matplotlib テストスキップ対応

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-18

## 概要

status-035 の TODO を消化。テーブル補間型硬化則（区分線形）を実装し、Abaqus *PLASTIC テーブルから Plasticity1D / PlaneStrainPlasticity への変換パイプラインを構築。matplotlib 未インストール環境でのテストスキップにも対応。
テスト数 753 → 782（+29テスト）。

## 実施内容

### 1. matplotlib 未インストール環境でのテストスキップ対応 (`tests/test_export_animation.py`)

- `pytest.mark.skipif` を使用し、matplotlib 未インストール時に描画系テストをスキップ
- `TestCollectBeamSegments`（6テスト）は matplotlib 不要のため常時実行
- `TestRenderBeamAnimationFrame`（8テスト）、`TestExportFieldAnimation`（5テスト）、`TestIntegrationWithInpParser`（2テスト）計15テストを条件付きスキップ
- `try: import matplotlib` パターンで `_has_matplotlib` フラグを設定

### 2. テーブル補間型硬化則 (`xkep_cae/materials/plasticity_1d.py`)

#### `IsotropicHardening` メソッド追加

- `sigma_y(alpha)` メソッド追加: `sigma_y0 + H_iso * alpha` を返す
- `dR_dalpha(alpha)` メソッド追加: `H_iso` を返す
- 既存の直接属性アクセスとの後方互換性を保持

#### `TabularIsotropicHardening` データクラス新設

- Abaqus *PLASTIC テーブル形式 `[(sigma_y, eps_p), ...]` を直接受け取る
- `sigma_y(alpha)` メソッド: 区分線形補間で降伏応力を返す
- `dR_dalpha(alpha)` メソッド: テーブルの傾きを返す
- テーブル範囲外（alpha > 最終 eps_p）では降伏応力一定・硬化なし（Abaqus 互換）
- `sigma_y0` プロパティ: テーブル最初の降伏応力
- バリデーション: eps_p の単調増加チェック、最低1点チェック

#### `Plasticity1D.return_mapping()` 修正

- 線形硬化（`IsotropicHardening`）の場合は従来のクローズドフォームを維持
- テーブル硬化（`TabularIsotropicHardening`）の場合は Newton 反復で塑性乗数 dg を求解
- consistent tangent は `self.iso.dR_dalpha(alpha_new)` を使用し、テーブルの局所勾配を反映
- 既存テストの完全な後方互換性を確認

### 3. PlaneStrainPlasticity 互換対応 (`xkep_cae/materials/plasticity_3d.py`)

- `_solve_dg()` の閉形式パス判定を `isinstance(self.iso, IsotropicHardening3D)` に変更
- `TabularIsotropicHardening` は Newton 反復パスを通り、`sigma_y()`/`dR_dalpha()` メソッドで正しく動作
- 既存テストの後方互換性を確認

### 4. コンバータ (`xkep_cae/io/material_converter.py` 新規)

- `abaqus_material_to_plasticity_1d()`: AbaqusMaterial → Plasticity1D
  - *ELASTIC の E を使用、*PLASTIC テーブル → TabularIsotropicHardening
  - *PLASTIC 未定義の場合は弾性のみ（降伏しない高い降伏応力）
  - HARDENING=KINEMATIC/COMBINED は ValueError（未対応の旨を明示）
- `abaqus_material_to_plane_strain_plasticity()`: AbaqusMaterial → PlaneStrainPlasticity
  - E, nu を使用、同様のテーブル変換
- `xkep_cae/io/__init__.py` にエクスポート追加

### 5. テスト (`tests/test_tabular_hardening.py`) — +29テスト

#### `TestTabularIsotropicHardening` — 10テスト
- `test_sigma_y0`: 初期降伏応力
- `test_sigma_y_at_table_points`: テーブル上の点
- `test_sigma_y_interpolation`: 区間内の線形補間
- `test_sigma_y_extrapolation_beyond_table`: テーブル範囲外は一定
- `test_dR_dalpha_within_table`: テーブル内の硬化係数
- `test_dR_dalpha_variable_slope`: 区間ごとに異なる勾配
- `test_dR_dalpha_beyond_table`: テーブル範囲外は硬化なし
- `test_single_point_table`: 1点テーブル（完全弾塑性）
- `test_invalid_non_monotonic`: eps_p 非単調で ValueError
- `test_invalid_empty_table`: 空テーブルで ValueError

#### `TestPlasticity1DTabular` — 8テスト
- `test_elastic_range`: 降伏前は弾性応答
- `test_yielding`: 降伏後の応力
- `test_consistent_tangent_positive`: 硬化中の正の接線
- `test_perfect_plasticity`: 完全弾塑性（1点テーブル）
- `test_equivalence_with_linear_hardening`: 線形テーブル ↔ IsotropicHardening の等価性
- `test_incremental_loading`: 増分載荷で応力単調増加
- `test_unloading_elastic`: 除荷は弾性的
- `test_multilinear_stress_path`: 多直線テーブルでの応力経路

#### `TestPlaneStrainPlasticityTabular` — 4テスト
- `test_elastic_range`: 降伏前は弾性
- `test_yielding_uniaxial`: 単軸引張で降伏
- `test_equivalence_linear_hardening_3d`: 線形テーブル ↔ IsotropicHardening3D の等価性
- `test_consistent_tangent_symmetry`: consistent tangent の対称性

#### `TestMaterialConverter` — 7テスト
- `test_to_plasticity_1d_with_table`: テーブル硬化 → Plasticity1D
- `test_to_plasticity_1d_elastic_only`: 弾性のみ → Plasticity1D
- `test_to_plasticity_1d_no_elastic_error`: *ELASTIC 未定義で ValueError
- `test_to_plasticity_1d_kinematic_error`: KINEMATIC 硬化で ValueError
- `test_to_plane_strain_plasticity`: テーブル硬化 → PlaneStrainPlasticity
- `test_to_plane_strain_no_elastic_error`: *ELASTIC 未定義で ValueError
- `test_roundtrip_return_mapping`: パーサー → コンバータ → return mapping ラウンドトリップ

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/materials/plasticity_1d.py` | `IsotropicHardening` に `sigma_y()`/`dR_dalpha()` メソッド追加。`TabularIsotropicHardening` データクラス新設。`Plasticity1D.return_mapping()` をメソッドベースインターフェースに対応 |
| `xkep_cae/materials/plasticity_3d.py` | `_solve_dg()` の閉形式パス判定を `isinstance` チェックに変更 |
| `xkep_cae/io/material_converter.py` | 新規。AbaqusMaterial → Plasticity1D / PlaneStrainPlasticity 変換関数 |
| `xkep_cae/io/__init__.py` | コンバータ関数のエクスポート追加 |
| `tests/test_export_animation.py` | matplotlib 未インストール時のテストスキップ対応 |
| `tests/test_tabular_hardening.py` | 新規。+29テスト（テーブル硬化則 + コンバータ） |

## テスト数

753 → 782（+29テスト、うち15はmatplotlib条件付きスキップ）

## 確認事項・懸念

1. **HARDENING=KINEMATIC/COMBINED の未対応**: `abaqus_material_to_plasticity_1d()` で ValueError を返す。Abaqus の移動硬化テーブルは Armstrong-Frederick パラメータ（C_kin, gamma_kin）への変換が必要で、自動推定は困難。手動でのパラメータ指定を受け付けるインターフェース拡張を検討
2. **テーブル硬化のconsistent tangent精度**: 区間境界（テーブルの折れ点）付近では硬化係数が不連続。NR収束性に注意が必要。大きな増分で折れ点を跨ぐ場合はステップの細分化を推奨

## TODO

- [ ] HARDENING=KINEMATIC テーブル → Armstrong-Frederick パラメータ変換の検討
- [ ] GIF/MP4 出力のサポート（ffmpeg連携）— status-034 から継続
- [ ] 連続体要素のメッシュプロット対応（将来）— status-034 から継続

---
