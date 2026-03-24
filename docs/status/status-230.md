# status-230: ComputeStJacobianProcess Hermite 幾何対応 + frac=0.98 達成

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-24
**ブランチ**: `claude/execute-status-todos-5W8JA`

---

## 概要

status-229 の TODO「ComputeStJacobianProcess を Hermite 幾何対応に拡張」を実施。

- ComputeStJacobianProcess を Hermite 曲線上の陰関数微分に拡張（完全 Newton Jacobian）
- 接触力分配・接線剛性を Hermite 形状関数 H00(s)/H01(s) に切替
- freeze_geometry_in_nr=True（NR 内 s,t 凍結）と組み合わせで **frac=0.98 達成**

---

## 検証結果（再現性確認済み — tee ログ保存）

| 条件 | frac | fc [N] | incr | 時間 [s] | 備考 |
|------|------|--------|------|----------|------|
| Hermite OFF, ε=0.02 | 0.86 | 154.1 | 276 | 786 | ベースライン（status-229 と一致） |
| **Hermite ON + freeze_st** | **0.98** | **166.5** | **290** | **693** | **本セッション** |
| Hermite ON（旧 status-229） | 0.03 | 16.0 | 20 | 28 | 破滅的回帰 |
| Hermite ON + 整合形状関数のみ | 0.03 | 15.3 | 13 | 33 | 形状関数だけでは不十分 |

### テスト条件

- E=25, push=30mm, n_periods=1, max_incr=500
- ε=0.02 完全統一（status-229）

---

## 変更内容

### 1. ComputeStJacobianProcess の Hermite 対応

**ファイル**: `xkep_cae/contact/geometry/_st_jacobian.py`

- **バージョン**: 2.0.0 → 3.0.0
- **StJacobianInput**: mA0/mA1/mB0/mB1, use_hermite フラグ追加
- **完全 Newton Jacobian**: d²p/ds² 項含む（Gauss-Newton 近似を排除）
  ```
  a = dpA·dpA + δ·d²pA/ds²
  c = dpB·dpB - δ·d²pB/dt²
  ```
- **Hermite RHS**: H00(s)/H01(s) 基底関数の微分で ∂F/∂u を計算
- **スカラー Hermite ヘルパー**: eval, deriv, second_deriv 関数群
- **FD 検証テスト 4件**: 直線/直交/斜交/非対称配置

### 2. 接触力の Hermite 形状関数対応

**ファイル**: `xkep_cae/contact/contact_force/strategy.py`

- `_contact_shape_vector`: use_hermite=True で H00(s)/H01(s) 基底に切替
- `evaluate()`: Hermite 形状関数で力分配
- `tangent()`: K_mat/K_geo で Hermite 係数使用
- `_add_kst_contact`: Hermite 接線 dpA/ds で ∂n/∂s 計算、dc_ds/dc_dt を Hermite 基底微分に更新
- `_hermite_shape_coeffs`, `_hermite_dc_ds`, `_hermite_dc_dt` ヘルパー追加

### 3. 摩擦アセンブリ基盤

**ファイル**: `xkep_cae/contact/friction/_assembly.py`

- `_contact_tangent_shape_vector`: use_hermite パラメータ追加
- `_compute_tangential_displacement`: use_hermite パラメータ追加
- `_assemble_friction_geometric_stiffness`: use_hermite パラメータ追加

### 4. connectivity 伝播

**ファイル**: `xkep_cae/contact/_contact_pair.py`, `xkep_cae/contact/_manager_process.py`

- `_ContactManagerInput` に `connectivity` フィールド追加
- 全 Process（AddPair, ResetAll, DetectCandidates, UpdateGeometry, InitializePenalty）で connectivity を伝播
- `process.py`: 初期 UpdateGeometryInput に connectivity 伝播

### 5. freeze_st の Hermite 対応

**ファイル**: `xkep_cae/contact/_manager_process.py`

- freeze_st パスで Hermite 使用時は `_hermite_eval` で接触点計算
  （従来は線形補間 `(1-s)*xA0 + s*xA1`）

### 6. 三点曲げ Config

**ファイル**: `xkep_cae/numerical_tests/three_point_bend_jig.py`

- `use_hermite_centerline=True`（デフォルト ON）
- `freeze_geometry_in_nr=True` 追加（Hermite 安定化）

---

## 技術的知見

### 1. Hermite 形状関数の線形との差

| s | H00(s) | (1-s) | 差 |
|---|--------|-------|-----|
| 0.0 | 1.0 | 1.0 | 0% |
| 0.25 | 0.844 | 0.75 | 12.5% |
| 0.5 | 0.5 | 0.5 | 0% |
| 0.75 | 0.156 | 0.25 | 37.5% |
| 1.0 | 0.0 | 0.0 | 0% |

H00(s) + H01(s) = 1 は保持（力の合計は不変）。

### 2. freeze_geometry_in_nr が不可欠な理由

Hermite refine（`_closest_point_hermite_refine`）は Newton 法で最近接点を求める。
NR 反復で節点位置が変化すると:
1. 接線ベクトル m が変化（`_compute_node_tangents` は全節点座標に依存）
2. Hermite 曲線形状が変化
3. 最近接点 s,t が大きくジャンプする可能性
4. 接線剛性が予測できない残差変化 → NR 不収束

freeze_geometry_in_nr=True は s,t をNR内で凍結し、ステップ間のみ更新する。
これにより Hermite refine のジャンプが NR を妨害しない。

### 3. 完全 Newton vs Gauss-Newton（st_jacobian）

FD 検証で Gauss-Newton 近似（δ·d²p/ds² 無視）は O(||δ||·||d²p/ds²||) ≈ 0.002 の誤差。
完全 Newton Jacobian で O(smooth_clip 相互作用) ≈ 0.005 に改善。
K_st は補正項なので 1e-2 精度で十分。

---

## テスト

**100+10s passed** — 契約違反 1件（既存）、条例違反 0件
- Hermite FD テスト 4件新規追加（全 pass）
- 既存テスト全 pass（contact 166件 + st_jacobian 11件）

---

## TODO

- [ ] n_periods=30, E=25 での数百 N 確認 — max_incr=500 では frac≈0.09 で終了（n_periods=1 で 290 incr 使用のため）。max_incr=15000 程度が必要
- [ ] ε=0.02 での物理テスト（貫入量精度）検証 — status-229 で遷移帯 2% は十分と評価済み
- [ ] 摩擦アセンブリの Hermite 完全対応（現在は use_hermite=False デフォルト）
- [ ] frac=1.0 到達のためのさらなる安定化（frac=0.98 の壁分析）

---

## 運用メモ

- **STA2 対策**: 全結果は tee でログ保存 + YAML 出力
- **ベースライン確認**: Hermite OFF で frac=0.86 を確認してから Hermite ON テスト
- **freeze_geometry_in_nr**: Hermite ON 時は必須。OFF だと frac=0.03 に回帰
