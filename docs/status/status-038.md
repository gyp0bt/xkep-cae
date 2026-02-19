# status-038: HARDENING=KINEMATIC テーブル → Armstrong-Frederick 変換 + TODO整理

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-19

## 概要

status-037 の TODO を消化。HARDENING=KINEMATIC テーブルから Armstrong-Frederick 移動硬化パラメータ (C_kin, gamma_kin) への自動変換機能を実装。Plasticity1D / PlaneStrainPlasticity の両方で KINEMATIC 硬化対応が完了。
連続体メッシュプロットは凍結、examples の計算スクリプトは接触完了後に対応。
テスト数 789 → 802（+13テスト）。

## 実施内容

### 1. `kinematic_table_to_armstrong_frederick()` 関数新設 (`xkep_cae/io/material_converter.py`)

Abaqus *PLASTIC, HARDENING=KINEMATIC テーブルを Armstrong-Frederick パラメータに変換する関数。

#### 変換ロジック

テーブルの各点 (σ_i, ε_p_i) から背応力 β_i = σ_i - σ_y0 を算出し、
Armstrong-Frederick モデル β(ε_p) = (C/γ)(1 - exp(-γ·ε_p)) でフィッティング。

| テーブル点数 | 変換方法 | 結果 |
|------------|---------|------|
| 1点 | 完全弾塑性 | C_kin=0, gamma_kin=0 |
| 2点 | 線形移動硬化 | C_kin=勾配, gamma_kin=0 |
| 3点以上 | 非線形フィッティング（scipy.optimize.least_squares） | C_kin>0, gamma_kin>0 |

- 3点以上のフィッティングでは `least_squares(method='trf', bounds=([0,0],[inf,inf]))` を使用
- gamma_kin < 1e-4 の場合は線形とみなし、線形最小二乗で C_kin を再推定
- 初期推定: C_init = 初期勾配, gamma_init = C_init / β_sat

#### 技術的注記

- Abaqus の多直線移動硬化（Ziegler）を AF 指数関数で近似するため、近似精度はデータ形状に依存
- 既知の AF パラメータからテーブルを生成した場合、5% 以内の精度でパラメータを復元

### 2. コンバータ更新 (`xkep_cae/io/material_converter.py`)

#### `abaqus_material_to_plasticity_1d()`

- KINEMATIC: `kinematic_table_to_armstrong_frederick()` → `IsotropicHardening(sigma_y0)` + `KinematicHardening(C_kin, gamma_kin)` で `Plasticity1D` を生成
- COMBINED: ValueError（未対応）
- ISOTROPIC: 従来通り `TabularIsotropicHardening`

#### `abaqus_material_to_plane_strain_plasticity()`

- KINEMATIC: 同様に `IsotropicHardening3D` + `KinematicHardening3D` で `PlaneStrainPlasticity` を生成
- COMBINED: ValueError（未対応）

### 3. エクスポート更新

- `xkep_cae/io/__init__.py`: `kinematic_table_to_armstrong_frederick` を追加
- `__all__` リスト更新

### 4. テスト (`tests/test_tabular_hardening.py`) — +13テスト（29→42テスト）

#### `TestKinematicTableToArmstrongFrederick` — 7テスト（新規クラス）

- `test_single_point_perfect_plasticity`: 1点テーブル → C_kin=0, gamma_kin=0
- `test_two_point_linear`: 2点テーブル → 線形移動硬化
- `test_two_point_high_slope`: 高い硬化勾配のテスト
- `test_multipoint_nonlinear_fit`: 既知 AF パラメータの復元検証（C=5000, γ=50）
- `test_multipoint_saturating_curve`: 飽和型背応力曲線の精度（β_sat 5%以内）
- `test_linear_table_returns_zero_gamma`: 線形テーブル（3点以上）→ gamma_kin≈0
- `test_empty_table_raises`: 空テーブルで ValueError

#### `TestKinematicRoundtrip` — 3テスト（新規クラス）

- `test_linear_kinematic_hysteresis`: 載荷→除荷→逆載荷でバウシンガー効果の検証
- `test_kinematic_converter_roundtrip`: AbaqusMaterial(KINEMATIC) → Plasticity1D → 載荷・除荷
- `test_nonlinear_kinematic_converter_roundtrip`: 多点テーブル → AF 変換 → return mapping → 増分載荷

#### `TestMaterialConverter` — 変更（+3テスト）

- `test_to_plasticity_1d_kinematic_linear`: KINEMATIC → Plasticity1D 変換成功（旧 ValueError テスト置換）
- `test_to_plasticity_1d_kinematic_perfect`: 1点テーブル KINEMATIC
- `test_to_plasticity_1d_combined_error`: COMBINED → ValueError
- `test_to_plane_strain_kinematic`: KINEMATIC → PlaneStrainPlasticity 変換

### 5. TODO 整理

- **連続体メッシュプロット**: 凍結（TODOから削除）
- **examples 計算スクリプト**: 接触完了後に対応（Phase C 完了後）

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/io/material_converter.py` | `kinematic_table_to_armstrong_frederick()` 新設、KINEMATIC 対応追加、ドキュメント更新 |
| `xkep_cae/io/__init__.py` | `kinematic_table_to_armstrong_frederick` エクスポート追加 |
| `tests/test_tabular_hardening.py` | +13テスト（KINEMATIC 変換・ラウンドトリップ・コンバータ） |
| `README.md` | テスト数・KINEMATIC 対応の反映 |
| `docs/roadmap.md` | 現在地・テスト数更新 |
| `docs/status/status-index.md` | status-038 行追加 |

## テスト数

789 → 802（+13テスト）

## 確認事項・懸念

1. **HARDENING=COMBINED の未対応**: 等方硬化+移動硬化の複合モデルは未実装。Abaqus の COMBINED は等方硬化テーブル+移動硬化テーブルの組み合わせで、パラメータ分離が必要。需要に応じて対応
2. **多直線→AF 近似の精度**: 3点以上のテーブルで Abaqus のZiegler多直線モデルを AF 指数関数で近似するため、折れ点の多いテーブルでは近似誤差が大きくなる可能性がある
3. **scipy.optimize 依存**: 3点以上のフィッティングで `scipy.optimize.least_squares` を使用。scipy は必須依存なので問題なし

## TODO

- [ ] Phase C2: 法線AL + Active-setヒステリシス + 主項接線（接触モジュール次段階）
- [ ] examples の .inp ファイルを使った実際の解析実行スクリプトの追加（Phase C 完了後）

---
