# status-237: 梁メッシュ粗化 + 解析的剛体円柱表面

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-25
**テスト**: 190+10s+8（+8: 剛体表面射影テスト） | 契約違反 1件（既存C3） | 条例違反 0件

---

## 概要

三点曲げジグの接触安定性改善のため、2つの構造的変更を実施:

1. **梁メッシュ粗化**: L_elem > wire_diameter で梁サーフェスの面連続性を確保
2. **解析的剛体円柱表面**: ジグ離散セグメント → C∞ 連続な解析的円柱表面

これらは Phase B（NR力収束改善）の前提条件となる接触面品質の改善。

---

## 変更内容

### 1. 梁メッシュ粗化（L_elem > wire_diameter）

**問題**: wire_diameter=17mm に対して n_elems=20（L_elem=5mm）では、
要素長が直径の 1/3 以下。隣接要素の円柱表面が角度を持ち、
接触検出時に法線不連続やギャップジャンプの原因に。

**対策**: `n_elems_wire` を 20→4 に変更（L_elem=25mm > 17mm）。
梁サーフェスの円柱面が隣接要素間で連続しやすくなる。

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| n_elems_wire | 20 | 4 |
| L_elem | 5.0mm | 25.0mm |
| L_elem/diameter | 0.29 | 1.47 |

### 2. 解析的剛体円柱表面

**問題**: ジグが 8 セグメントの折れ線で近似されており、
セグメント境界で法線が不連続。接触点がセグメント境界を
越える際にギャップが急変し、NR収束を阻害。

**対策**: `RigidCylinderSurfaceInput` を導入し、ジグ表面を
解析的な円柱関数で定義。離散化誤差ゼロ、C∞ 連続。

**アーキテクチャ**:
- ジグセグメントは broadphase 候補検出用に残存（AABB ベース）
- narrowphase の幾何計算は解析的円柱射影でオーバーライド
- `_ContactConfigInput` に `rigid_surface_*` フィールド追加
- `_batch_update_geometry` で剛体表面ペアを検出し解析的に射影

**射影アルゴリズム**:
- 梁セグメント p(s) = A0 + s*(A1-A0) から円柱軸への垂直距離を最小化
- s = -(w0_perp · dA_perp) / |dA_perp|² で解析的に求解
- gap = |p_perp(s) - center_perp| - R_cyl - r_beam
- 法線 = (p_perp - center_perp) / |p_perp - center_perp|

---

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/geometry/_rigid_surface.py` | **新規**: RigidCylinderSurfaceInput + project_beam_to_cylinder_batch |
| `xkep_cae/contact/_contact_pair.py` | `_ContactConfigInput` に rigid_surface_* フィールド追加 |
| `xkep_cae/contact/geometry/strategy.py` | `_batch_update_geometry` に剛体表面オーバーライド追加 |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | n_elems_wire: 20→4, jig_n_segments: 8→4, use_rigid_surface=True |
| `tests/contact/test_rigid_surface.py` | **新規**: 射影テスト 8件（API 4件 + 物理 4件） |
| `docs/status/status-237.md` | 本ステータス |
| `docs/status/status-index.md` | 更新 |
| `docs/roadmap.md` | 更新 |
| `README.md` | 更新 |

---

## テスト結果

### 新規テスト（8件 all pass）
- `TestRigidCylinderProjectionAPI`: frozen dataclass、直下射影、gap計算、貫入
- `TestRigidCylinderProjectionPhysics`: バッチ射影、斜めビーム、法線連続性、z軸独立性

### 回帰テスト
- 471 passed, 20 skipped, 1 xfailed（既存失敗 2件: HuberForce inactive pair, StressContour matplotlib）
- 契約違反: 1件（既存 C3: ComputeStJacobianProcess 未紐付け）

---

## TODO

- [ ] **Phase B（NR力収束改善）**: 本変更で接触面品質は改善。次は接線剛性整合性の調査
- [ ] **n_periods=30 での収束検証**: 解析的剛体表面 + 粗メッシュでの効果測定
- [ ] **撚線モデルへの剛体表面適用**: 三点曲げ以外のジグ問題への汎用化

---

## 設計上の懸念

1. **st_jacobian との整合**: 剛体表面ペアの t パラメータは形式的に 0.5 を設定。
   ジグ DOF は変位処方で境界条件により剛性行列に寄与しないため、
   t 関連の接線剛性項は自動的にゼロ化される。大規模問題での検証が必要。

2. **broadphase と narrowphase の乖離**: broadphase は離散ジグセグメントの AABB、
   narrowphase は解析的円柱。接触候補の見落とし/誤検出の可能性がある。
   ジグセグメントを十分に広く配置することで対処。

3. **梁要素数 4**: Timoshenko 梁で三点曲げの精度は十分だが、
   大変位時の幾何学的非線形精度は低下する可能性がある。
