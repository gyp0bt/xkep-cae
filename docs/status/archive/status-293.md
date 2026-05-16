# status-293: StJacobian smooth遷移帯 + frozen-m内部接触点検証 + MPC調査

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-04
- **ブランチ**: `claude/check-status-todos-OR8dq`
- **テスト数**: 631+ passed（+4: smooth遷移帯FDテスト）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-292のTODO 3件を実行:

1. **TODO 2（実装完了）**: StJacobian 1×1フォールバックの遷移帯をsmooth blendingに改善
2. **TODO 1（検証完了）**: frozen-m内部接触点でのK_st FD検証（交差配置に変更）
3. **TODO 3（調査のみ）**: free_end_mode=False（MPC）での90度曲げ発散原因の仮説整理

---

## 実装内容

### 1. StJacobian smooth遷移帯（TODO 2）

**問題**: status-292で実装した1×1/2×2系の切替がhard threshold（w_t < 1e-10）で急峻。遷移帯 w_t ∈ (1e-10, 0.5) で不連続な接線剛性変化がNR収束を阻害しうる。

**解決**: w_s/w_tによる連続補間（smooth blending）を実装。

```python
# 高速パス: 両方完全有効 → 2×2系のみ
if w_s >= 1.0 - 1e-10 and w_t >= 1.0 - 1e-10:
    ds_du = ds_du_2x2
    dt_du = dt_du_2x2
else:
    # 1×1系と2×2系を両方計算し、w_t/w_sで連続補間
    ds_du = w_t * ds_du_2x2 + (1.0 - w_t) * ds_du_1x1
    dt_du = w_s * dt_du_2x2 + (1.0 - w_s) * dt_du_1x1
```

**利点**:
- w_t=0 → 純粋1×1系（status-292と同一）
- w_t=1 → 純粋2×2系（従来動作）
- 0 < w_t < 1 → 連続補間（C1連続性は smooth_clip_01 の性質から保証）

### IFT幾何のunclamped座標評価

smooth blending実装中にFDテストで発見した追加修正:

**問題**: IFT（陰関数定理）の幾何量（接線ベクトル・差分ベクトル）をclamped座標 s ∈ [0,1] で評価していた。smooth_clip_01の遷移帯では s_unc ≠ s_clamped であり、FD微分とanalytical微分に不整合が発生。

**修正**: IFT幾何評価を unclamped座標 (s_unc, t_unc) で実施するよう変更。

```python
# status-293: IFT 幾何を unclamped (s_unc, t_unc) で評価
s = s_unc  # 旧: s = inp.s (clamped)
t = t_unc  # 旧: t = inp.t (clamped)
```

**影響範囲**: _process_linear と _process_hermite の両方を修正。

### 追加テスト（4件）

| テスト | 検証内容 |
|--------|----------|
| `test_t_transition_zone` | t下端遷移帯（w_t ∈ (0,1)）でのds_du/dt_du FD整合性 |
| `test_s_transition_zone` | s下端遷移帯（w_s ∈ (0,1)）でのds_du/dt_du FD整合性 |
| `test_both_transition_zone` | s+t同時遷移帯での整合性 |
| `test_t_upper_transition_zone` | t上端遷移帯（t ≈ 1付近）での整合性 |

全テスト atol=0.02 以内で合格。

### 2. frozen-m内部接触点検証（TODO 1）

**問題**: status-292の3Dヘリカル配置テストでs_unc=2.886（端部）のため、K_stがゼロだった。内部接触点での検証が未完。

**解決**: helical_z + n_elems>=3 の配置を平行→交差（skew）配置に変更し、内部接触点を生成。

```python
# status-293: 交差配置（skew）+ z方向傾き → 内部接触点
node_coords = np.array([
    [-1.0, 0.0, 0.00], [0.0, 0.0, 0.02], [1.0, 0.0, 0.04], [2.0, 0.0, 0.06],
    [0.5, -1.5, -0.02], [0.5, -0.5+sep/2, 0.0], [0.5, 0.5+sep/2, 0.02], [0.5, 1.5+sep/2, 0.04],
])
```

**結果**: s≈0.5, t≈0.45（内部接触点）、||K_st||=15182（非ゼロ確認）。

**frozen-m制限**: K_st FD相対誤差 ≈ 3x（rel_err < 0.5 でアサート）。これはfrozen-m近似（∂m/∂u=0）の既知制限であり、evaluate()とtangent()の両方で_node_counts=Noneとする意図的な設計（status-278）。

### 3. MPC 90度曲げ発散調査（TODO 3）

free_end_mode=False（MPC使用）でfrac=0.0016で発散する問題を調査。以下の仮説を整理:

| 仮説 | 内容 | 確度 |
|------|------|------|
| A | MPC T行列が最初のUL更新前に再構築されていない | 中 |
| B | _ExtendedULAssemblerWrapperで参照点回転更新が欠落 | 中 |
| C | 拡張系の質量行列とMPC変換の相互作用 | 低 |
| D | 初期ステップ感度（最初のincrementで大変形→即発散） | 低 |

**次セッションへの推奨**: 仮説A/Bの検証を優先。MPC T行列再構築タイミングのデバッグログ追加が有効。

---

## 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/geometry/_st_jacobian.py` | **smooth blending**（1×1/2×2連続補間）+ **unclamped座標IFT評価** |
| `tests/contact/test_st_jacobian.py` | smooth遷移帯FDテスト4件追加 |
| `xkep_cae/contact/contact_force/tests/test_kc_component_fd.py` | helical_z交差配置 + frozen-m制限ドキュメント + アサート追加 |

---

## TODO

- [ ] frozen-m解消（∂m/∂uの正確計算）→ K_st FD不整合3x→0%
- [ ] free_end_mode=False（MPC）での90度曲げ収束改善（仮説A/B検証）
- [ ] z方向DOFカップリングをK_stに追加（frozen-m解消の前提）

---

## 次の担当者向け

### smooth blending の意義

hard thresholdの1×1/2×2切替をsmooth blendingに置換。これにより:
1. 遷移帯でのK_st不連続性が解消
2. NRソルバーの接線剛性がC1連続
3. 接触パラメータが端部付近を移動する際の安定性向上

### unclamped座標IFT評価の意義

smooth_clip_01の遷移帯（ε幅）ではs_unc ≠ s_clampedとなる。IFT幾何量をunclamped座標で評価することで、FD微分との整合性が保証される。

### frozen-m制限の現状

K_st FD相対誤差3xは、frozen-m（∂m/∂u=0）が接線ベクトルのDOF依存性を無視しているため。status-289で特定されたcomp=2(z方向)不整合と同根。解消にはevaluate()/tangent()両方で_node_countsを有効化し、dm補正を整合的に実装する必要がある。

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: FD整合性テスト結果はpytest出力と一致
- [x] **回帰なし**: 465 internal tests passed（既存テスト全合格）
- [x] **ベースライン確認**: status-292の1×1/2×2 hard thresholdが修正前ベースライン

---
