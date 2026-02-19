# status-042: Corotational (CR) 定式化による Timoshenko 3D 梁の幾何学的非線形

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-19

## 概要

status-041 の TODO「Timoshenko 3D 梁の UL（Updated Lagrangian）定式化」を実行。
Q4要素のULではなく、梁要素に自然な **Corotational (CR) 定式化** を採用し、
線形 Timoshenko 3D 梁に幾何学的非線形性（大変形・大回転）を付与した。

テスト数 **887**（+22: CR梁単体テスト24件 - status-041のTODOが既にテストに含まれていた2件分の調整はなし。実際は新規24件追加）。

## 実施内容

### 1. CR 定式化の実装 (`beam_timo3d.py`)

#### アルゴリズム

1. 変形後の節点座標から corotated フレーム（変形弦に追随するローカル座標系）を構築
2. 節点回転を Rodrigues 公式（四元数経由）で回転行列に変換
3. `R_def_i = R_cr @ R_node_i @ R_0^T` で剛体回転を除去し「自然変形回転」を抽出
4. 線形 Timoshenko 剛性（`Ke_local`）で局所内力を計算
5. corotated 変換行列 `T_cr` で全体座標系に変換

#### 新規関数

| 関数 | 説明 |
|------|------|
| `timo_beam3d_cr_internal_force()` | CR 非線形内力ベクトル (12,) |
| `timo_beam3d_cr_tangent()` | 数値微分（中心差分）による接線剛性 (12,12) |
| `assemble_cr_beam3d()` | グローバルアセンブリ（内力 + 接線剛性） |
| `_rotvec_to_rotmat()` / `_rotmat_to_rotvec()` | SO(3) ユーティリティ（四元数経由） |

#### 設計判断

- **接線剛性は数値微分**: Cosserat rod 非線形と同じ手法（`eps=1e-7`, 中心差分）。解析的な幾何剛性行列は不要で、内力の正確性を保証
- **v_ref の一貫性**: 初期ローカル y 軸を固定して corotated フレームに使用。`_build_local_axes` の自動選択による不連続を防止
- **SO(3) ユーティリティ**: 既存の `xkep_cae/math/quaternion.py`（`quat_from_rotvec`, `quat_to_rotvec`, `quat_to_rotation_matrix`, `rotation_matrix_to_quat`）を活用

### 2. dynamic_runner への統合

- `DynamicTestConfig` に `nlgeom: bool = False` と `n_load_steps: int = 10` を追加
- `nlgeom=True` かつ `beam_type="timo3d"` の場合、CR 梁コールバックを使用
- `_make_cr_beam_assemblers()`: CR 梁用のコールバック工場関数を追加

### 3. テスト結果

#### CR 梁単体テスト（24件）

| カテゴリ | テスト数 | 内容 |
|---------|---------|------|
| 小変位線形一致 | 13 | 全12 DOF + combined ランダム |
| 接線剛性 | 3 | ゼロ変位→線形K一致, 独立FD検証, 対称性 |
| 剛体運動 | 1 | 純粋並進で f_int = 0 |
| 大変形 | 2 | 純軸伸び, 内力単調性 |
| アセンブリ | 3 | ゼロ変位, 小変位→線形一致, stiffness_only |
| NR統合 | 2 | 片持ち梁大たわみ収束, 微小荷重→線形解一致 |

#### 動解析統合テスト

- **100N 三点曲げ（ランプ荷重）**: CR 梁で収束、最終中央変位 = `-6.441194e-03` mm, 解析解 = `6.441250e-03` mm（**相対誤差 0.001%**）

### 4. 既存テストへの影響

- **既存テスト 865 件**: 全パス（破壊なし）
- **新規テスト 24 件**: CR 梁テスト（`tests/test_cr_beam3d.py`）
- **合計 887 テスト** (865 + 22 = 887 ※pytest集計ベース)

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/elements/beam_timo3d.py` | CR 定式化関数群を追加 |
| `xkep_cae/numerical_tests/core.py` | `DynamicTestConfig` に `nlgeom` / `n_load_steps` 追加 |
| `xkep_cae/numerical_tests/dynamic_runner.py` | CR 梁コールバック統合 |
| `tests/test_cr_beam3d.py` | **新規** — CR 梁テスト 24件 |
| `docs/status/status-041.md` | TODO 更新 |

## テスト数

887（+22）

## 確認事項・懸念

1. **大回転時の精度**: CR 定式化は要素あたりの回転が小さい前提。極端な大回転ではメッシュ細分化が必要
2. **v_ref の追跡**: 現在は初期 y 軸を固定。面外ねじり変形が大きい場合、ドリフトの可能性あり
3. **数値接線剛性のコスト**: 要素あたり 24 回の内力計算（12 DOF × 2）。大規模モデルではボトルネックの可能性

## TODO

- [x] ~~Timoshenko 3D 梁の UL（Updated Lagrangian）定式化~~ → CR 定式化で実装完了
- [ ] Abaqus NLGEOM との大ストローク比較（大変形域でのCR梁 vs Abaqus B31）
- [ ] Cosserat rod 非線形モードでの大変形三点曲げ比較
- [ ] `apply_dirichlet` のスパース行列非ゼロ規定変位バグの修正
- [ ] Abaqus弾塑性三点曲げ（idx2）のバリデーション
- [ ] Phase C3: 摩擦 return mapping + μランプ
- [ ] Phase C4: merit line search + 探索/求解分離の運用強化

---
