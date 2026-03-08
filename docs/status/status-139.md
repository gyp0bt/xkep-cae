# status-139: 被膜接触モデル収束検証 — 接線剛性実装 + 6DOFバグ修正 + 3D投影可視化

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-08
**テスト数**: 2271（変更なし）

---

## 概要

status-138 TODO「検証スクリプトの実行と結果記録」「被膜接触モデルの接線剛性行列への寄与」を消化。
被膜接触曲げ揺動の収束検証を実施し、3D梁表面の2D投影画像で物理的妥当性を確認。

---

## 変更内容

### 1. compute_coating_forces の6DOFバグ修正

**ファイル**: `xkep_cae/contact/pair.py:816-851`

問題: `compute_coating_forces` が並進3DOF/nodeの力ベクトル（ndof=n_nodes*3）を返していたが、
ソルバーのf_cは6DOF/node（ndof=n_nodes*6）。結果として `ValueError: operands could not be broadcast together with shapes (378,) (189,)` 発生。

修正: 6DOF/nodeに対応するようインデックス計算を `nA0 * dpn : nA0 * dpn + 3` に変更（`dpn=ndof_per_node=6`）。

### 2. 被膜スプリング接線剛性行列の実装

**ファイル**: `xkep_cae/contact/pair.py:855-931` — `compute_coating_stiffness()` メソッド追加

被膜スプリングの接線剛性を K_T に加算することで Newton 法の収束を改善:

```
K_coat = k_coat * α * g_n ⊗ g_n
```

- `g_n`: ギャップ勾配ベクトル（12×1）= [w_i * n_vec] for 4 nodes
- `α = 1.0`（活性領域）
- COO形式で組み立て → CSR変換

**ファイル**: `xkep_cae/contact/solver_ncp.py:2232-2235`

`K_T = K_T + K_coat` を Newton ループ内で K_line の直後に加算。

### 3. 検証スクリプトの3D梁表面2D投影化

**ファイル**: `scripts/verify_coating_contact_convergence.py:146-312`

既存の線プロット（YZ投影）を3Dパイプ表面の2D投影（四元数回転 + PolyCollection深度ソート）に置換。
`tests/generate_verification_plots.py` の `_beam_surface_polys_2d()`, `_project_3d_to_2d()` と同等のロジックをスクリプト内に自己完結で実装。

---

## 検証結果

### 被膜接触曲げ揺動（7本45°曲げ）

| ケース | coating_thickness | coating_stiffness | 収束 | NR反復 | 貫入比 | 時間 |
|--------|-------------------|-------------------|------|--------|--------|------|
| 被膜なし | 0 | 0 | **PASS** | 64 | 0.1105 | 1.4s |
| 被膜あり(高剛性) | 100μm | 1e8 Pa/m | FAIL | 159 | 0.0047 | 2.2s |
| 被膜あり(低剛性) | 100μm | 1e6 Pa/m | **PASS** | 82 | 0.0151 | 1.4s |

### 分析

- **k=1e6（低剛性被膜）**: 接線剛性の追加により完全収束。NR反復82回で安定。
  - 被膜スプリングが「柔らかいバッファ」として機能し、芯線間の直接接触を防止
  - 貫入比0.015（被膜なし0.11より大幅改善）
- **k=1e8（高剛性被膜）**: アクティブセットチャタリングで不収束。
  - 残差が `1.4e-2 ↔ 6.9` の2値振動パターン
  - k_coat/k_pen ≈ 137 — 構造剛性に対して被膜剛性が過大
  - 根本原因: `max(0, δ)` の非連続性で被膜圧縮δが0を跨いで振動

### 3D梁表面2D投影画像

`docs/verification/coating_contact_convergence.png` に保存。
- 左パネル: 被膜なし（PASS）— 正常な45°曲げ変形
- 中央パネル: 被膜あり高剛性（FAIL）— 不完全変形
- 右パネル: 被膜あり低剛性（PASS）— 正常な45°曲げ変形

---

## TODO

- [ ] k=1e8（高剛性被膜）のチャタリング対策（continuation法 or C1平滑化の再検討）
- [ ] 61本以上のNCP収束テスト
- [ ] 被膜なし＋あり混合ペア接触テスト
- [ ] 被膜モジュール分離リファクタリング

### 設計上の懸念

- 高剛性被膜（k_coat >> k_pen）では条件数悪化 + アクティブセットチャタリングが発生。
  C1平滑化・k_coat continuation・アクティブセット凍結の3手法を試行したが、
  いずれも k=1e6 のPASSを壊すトレードオフあり。
  実用上は k_coat ≤ k_pen の範囲（~1e6）が推奨。

---

## 前回status

- [status-138](status-138.md): CI失敗修正 + 被膜接触モデル収束検証基盤 + 被膜パラメータスタディ再実施

---
