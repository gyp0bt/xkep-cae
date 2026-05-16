# status-294: frozen-m部分解消（dm_A/dm_B有効化 + dm_ext無効化）

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-05
- **ブランチ**: `claude/check-status-todos-WAetC`
- **テスト数**: 631+ passed（既存テスト全合格、新規変更なし）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-293のTODO 1+3を実行: frozen-m近似の部分解消。

1. **TODO 1+3（完了）**: evaluate/tangent/tangent_componentsでdm_A/dm_B有効化（∂m/∂uの正確計算）
2. **dm_ext無効化**: K_st_adjとK_c_adjの二重計上を防止
3. **結果**: K_c FD相対誤差 15.5% → 11.0%（z方向高次効果が残余）
4. **TODO 2（未着手）**: MPC 90度曲げ発散の仮説A/B検証

---

## 実装内容

### 1. frozen-m部分解消の設計

**問題**: status-278でevaluate/tangentの両方で`_node_counts=None`（frozen-m）としていた。
これはdm補正を完全に無効化し、K_stがFDの約3倍となる不整合を引き起こしていた。

**解決策**: dm_A/dm_Bを有効化し、dm_extを無効化する「部分解消」方式を採用。

| dm成分 | 状態 | 理由 |
|--------|------|------|
| dm_A/dm_B（要素内） | **有効** | evaluate/tangent整合性を維持しつつK_st精度向上 |
| dm_ext（隣接ノード） | **無効** | K_c_adjが隣接ノード効果を正確にカバー。K_st_adjを追加すると二重計上 |

### 2. 変更箇所

**strategy.py**:
1. `evaluate()` (line ~857): `_eval_node_counts`を接続情報から計算（旧: None固定）
2. `tangent()` (line ~984): `_node_counts = _adj_node_counts`（旧: None固定）
3. `tangent_components()` (line ~1234): 同上
4. `_add_kst_contact_to_coo()` (line ~277): dm_ext計算を削除（コメントで理由記載）
5. 未使用import `_compute_dm_ext_coeffs` を削除

### 3. 二重計上の発見と解決

FD検証の結果、以下の構成を比較:

| 構成 | K_c rel_err | 備考 |
|------|-------------|------|
| frozen-m（全無効） | 15.5% | status-278ベースライン |
| 全dm有効（dm_ext含む） | 40% | **悪化** — K_st_adj + K_c_adj 二重計上 |
| dm_A/dm_B有効 + dm_ext無効 | **11.0%** | **最良** |

**根本原因**: K_c_adj（tangent内のK_mat+K_geo隣接ノード拡張）は、dm補正係数を使用して
隣接ノードのDOF依存性を正確にカバーしている。dm_extを通じてK_st_adjを追加すると、
同じ効果が二重に計上され、隣接ノードDOFで27倍の過大評価となる。

### 4. テスト修正

| テスト | 変更 |
|--------|------|
| `test_kst_adj_manual_formula` → `test_kst_adj_disabled_status294` | K_st隣接列がゼロであることを検証 |
| `test_kst_adj_nonzero` → `test_kst_adj_zero_status294` | 隣接ノード列ゼロを期待するよう反転 |
| `test_kc_adj_fd` | FD参照関数をdm補正係数使用に更新（evaluate()と整合） |
| `test_helical_3d_hermite` | docstring更新（frozen-m部分解消の記録） |

---

## 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/contact_force/strategy.py` | dm_A/dm_B有効化（evaluate/tangent/tangent_components）、dm_ext削除 |
| `xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py` | K_st_adjテスト3件更新 + K_c_adj FD参照dm整合 |
| `xkep_cae/contact/contact_force/tests/test_kc_component_fd.py` | helical_3d docstring更新 |

---

## 残余11%不整合の分析

z方向が不整合の99.4%を占める:

```
comp別不整合ノルム:
  x: 45.9 (7.6%)
  y: 46.5 (7.7%)
  z: 597.9 (99.4%)
```

z方向の不整合は、K_stの高次幾何効果（法線回転のz成分）がまだ完全には
捕捉されていないため。これはfrozen-m解消とは独立の問題であり、
次ステップで`∂n/∂u`のz方向カップリングを追加する必要がある。

---

## TODO

- [ ] z方向DOFカップリング追加（K_stのz方向不整合11%→0%）
- [ ] free_end_mode=False（MPC）での90度曲げ収束改善（仮説A/B検証）
- [ ] 90度曲げ接触ありでfrozen-m解消の効果検証（NR収束速度改善の確認）

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: FD整合性テスト結果はpytest出力と一致（rel_err=0.1096）
- [x] **回帰なし**: 全テスト合格（test_stress_contourの既存失敗を除く）
- [x] **ベースライン確認**: status-293のfrozen-m（15.5%）がベースライン
