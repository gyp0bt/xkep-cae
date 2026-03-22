# status-225: 摩擦接線幾何剛性の追加 + smooth_clip_01 revert

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-22
**ブランチ**: `claude/verify-dynamic-bending-load-cmPwf`

---

## 概要

摩擦力の接線方向幾何剛性（dt1/du, dt2/du のチェーン微分項）を実装。
status-223 で追加した接触法線回転の幾何剛性に加え、摩擦力方向の回転も
接線剛性行列に反映する。

併せて、smooth_clip_01（最近接点クランプのC1平滑化）が収束を悪化させることを
確認し、np.clip に戻した。

---

## 実装内容

### 摩擦接線幾何剛性

摩擦力 `f_fric = q₁·G_t1 + q₂·G_t2` の接線剛性には2つの寄与がある:

1. **材料項**（既存）: `dq/du ⊗ G_tα` — return mapping の微分
2. **幾何項**（新規）: `q_α · dG_tα/du` — 接線方向の回転

幾何項の3×3行列 M:
```
M_{ij} = -q₁·n_i·t1_j + q₂·ε_{ijk}·t1_k - q₂·t2_i·n_j
```

導出:
- dt1/dn = -n⊗t1（Gram-Schmidt 射影の線形化）
- dt2/dn·P_perp = ε_{ijk}·t1_k - t2_i·n_j（t2 = n×t1 の連鎖微分 × 法線射影）
- dn/du = c_l/dist · (I - n⊗n)

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/friction/_assembly.py` | `_assemble_friction_geometric_stiffness()` 追加、戻り値に `friction_forces_local` 追加 |
| `xkep_cae/contact/friction/strategy.py` | `tangent()` に幾何項を統合、`_friction_forces_local` 保持 |
| `xkep_cae/contact/geometry/_compute.py` | smooth_clip_01 → np.clip に revert（関数定義は残置） |

### smooth_clip_01 revert

`_smooth_clip_01` 関数（C1平滑クランプ）は以下の問題を確認:
- 有効化時にカットバック頻発（frac=0.02 レベルで停滞）
- np.clip に戻すとベースライン（frac=0.60）に回復
- 関数定義は残置（将来の再検討用）、呼び出し箇所のみ np.clip に戻した

---

## 三点曲げ実行結果

| 設定 | frac到達 | push [mm] | 接触力 [N] | 時間 [s] |
|------|---------|-----------|-----------|----------|
| E=25, 幾何剛性なし, np.clip | 0.6011 | 18.03 | 80.8 | 121 |
| E=25, 幾何剛性あり, np.clip | 0.6011 | 18.03 | 80.8 | 107 |
| E=100, 幾何剛性あり, np.clip | 0.6011 | 18.03 | 323.2 | 104 |

### 分析

- 摩擦幾何剛性は frac=0.60 の収束壁に対して直接的な効果なし
  - 幾何剛性の大きさが材料剛性に比べ微小（q/dist << k_t）
  - 大変形域で効果が期待されるが、壁以前に到達できない
- frac=0.60 の壁は Newton 残差比 ≈ 1.001 で停滞（30反復後も同値）
- E=100 では frac=0.60 でも 323N（数百N達成）

### frac=0.60 壁の特徴

- active=15（安定、チャタリングなし）
- dt を dt_min まで縮小しても NR 不収束
- 変位収束（du 基準）では通過するが、力収束（残差基準）では不可能
- status-224 run#6（frac=0.89）との差異は環境依存の可能性

---

## 次のステップ

1. 幾何剛性の有限差分検証テスト追加
2. frac=0.60 壁の条件数・スペクトル分析
3. ラインサーチの改善
4. ∂s/∂u, ∂t/∂u の接線剛性への反映

---

## テスト

**175 passed, 10 skipped** — 契約違反 0件、条例違反 0件
