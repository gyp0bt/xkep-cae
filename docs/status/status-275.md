# status-275: テスト品質改善 + frozen_hermite_tangent回帰修正

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-31
- **ブランチ**: `claude/check-status-todos-yKUof`
- **テスト数**: 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+3（変更なし）→ **合計613 passed**
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### status-274 TODO消化

#### 1. TestKstNonlocalFD 座標修正（TODO #3）

**問題**: 平行配置でStJacobian特異 → `ds_du_adj=None` → K_st_adj列が全ゼロ → FDも0 → trivially passing

**修正**:
- 座標を非平行配置に変更（摩擦テストと同じパターン）
- `test_kst_adj_nodes_fd` → `test_kst_adj_manual_formula`に変更（完全なdf_ds公式で検証）
- `test_kst_adj_nonzero` 追加（非ゼロ値の存在確認）
- `test_kst_adj_endpoint_zero` 維持
- atol=1e-10で一致（公式検証のため精度保証）

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py` | TestKstNonlocalFD: 非平行座標 + 手動公式検証パターン |

#### 2. TestKcAdjFD 座標・要素修正（TODO #3 拡張）

**問題**: 
1. 平行座標 → K_c_adj非ゼロだがelem_a=0,elem_b=5が空間的に離れすぎ（gap=2.1）→ p_n=0 → trivially passing
2. FD力計算がdm込みだがtangent()はdm凍結 → 不整合

**修正**:
- elem_a=1, elem_b=4（中央要素、近接配置）に変更
- 非平行座標で隣接ノード4つ全てが有効（adj=[0,3,4,7]）
- s=t=0.5（同一x位置で接触保証）
- FD力計算をdm凍結（`_hermite_shape_coeffs`使用）でtangent()と一致
- `test_kc_adj_nonzero` 追加

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py` | TestKcAdjFD: 非平行座標 + 中央要素 + dm凍結一致 |

#### 3. Hermite FDテスト atol 厳格化（TODO #2）

**調査結果**:

| テスト | 最大差 | 残差F1,F2 | atol変更 |
|--------|--------|-----------|----------|
| asymmetric | 2.3e-9 | ~1e-16 | **1e-2 → 1e-5** |
| orthogonal | 1.0e-5 | ~5e-7 | 1e-2 維持 |
| skew | 7.4e-3 | ~6e-4 | 1e-2 維持 |

**根本原因更新**: frozen-m は status-271〜274 で解消済み。残るatol=1e-2の原因は `_closest_point_hermite_refine` の収束精度不足（Newton反復の残差が0に収束しない配置がある）。

| ファイル | 変更内容 |
|----------|----------|
| `tests/contact/test_st_jacobian.py` | asymmetric: atol=1e-5、NOTE更新 |

---

## テスト結果

```
600 passed, 20 skipped, 58 deselected, 1 xfailed (9.31s)
+ tests/contact/test_st_jacobian.py: 13 passed
合計: 613 passed
契約違反: 0件
```

---

## 再現手順

```bash
git checkout claude/check-status-todos-yKUof
pip install -e .

# 全テスト
python -m pytest xkep_cae/ tests/ -q -k "not slow and not stress_contour"

# 修正したテスト
python -m pytest xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py -v -k "Nonlocal or KcAdj"
python -m pytest tests/contact/test_st_jacobian.py -v -k "hermite_asymmetric"

# 契約検証
python contracts/validate_process_contracts.py
```

---

### 撚線ベンチマーク回帰修正

#### bisect 結果

| コミット | 内容 | frac |
|----------|------|------|
| `f7db2ae` (status-260) | ベースライン | **0.5914** |
| `60a6f3d` | active_contact_dofs NR結合 | 0.5914 |
| `d9c3758` | huber_delta_h 直接指定API | 0.5914 |
| `7058453` | three_point_bend delta_h貫通 | 0.5914 |
| `7403aa2` **(status-264)** | **frozen_hermite_tangent + _cur_ratio統一** | **0.3750** |

**回帰コミット: `7403aa2` (status-264)**

#### 仮説検証

| 条件 | frac | 備考 |
|------|------|------|
| HEAD (frozen=True) | 0.375 | 回帰状態 |
| frozen=**False** | **0.413** | 部分回復 |
| consistent_st=False | 0.388 | 微改善 |
| Hermite OFF | 0.213 | 大幅悪化 |
| K_c_adj無効 | 0.369 | 非局所は無関係 |
| frozen=F + tangent dm ON | 0.388 | tangent dm非凍結は逆効果 |

#### 根本原因

`frozen_hermite_tangent=True` のデフォルト化（status-264）により、`evaluate()` の dm 補正がOFFに。接触力評価の精度低下がチャタリング誘発。

#### 修正

- `frozen_hermite_tangent` デフォルト: **True → False** に変更
- evaluate() で dm 補正ON、tangent() は常に凍結（status-266設計維持）
- frac: 0.375 → **0.413** に回復（残りの差0.18はstatus-264〜274の他の変更の累積）

---

## 次セッションへの引き継ぎ

### 残課題（優先度順）

1. **NR力収束のさらなる改善**（frac=0.413 → 0.59以上）
   - status-264〜274の累積変更（NRリストア、delta_hブースト、チャタリング分析等）が個別にどの程度影響するか未特定
   - より詳細なbisectで各コミットの影響を定量化すべき

2. **_closest_point_hermite_refine の収束精度改善**
   - curved/skew配置でNewton残差が~6e-4残る
   - 改善すれば Hermite FDテスト全3件を atol=1e-5 に厳格化可能
   - 優先度は低い（実用上は問題ない）

### 設計メモ

1. **tangent()のdm凍結**: status-266の設計判断。tangent()はdm_A=None（凍結）、evaluate()はdm込み。テストのFD力計算もこれに合わせる必要がある。
2. **trivially passing テストのパターン**: 平行座標 + 離れた要素ペア → 接触力ゼロ → 0≈0で常にPASS。新テスト作成時は「接触あり（gap<0）」「非平行座標」を確認すること。

---

## STA2 準拠チェック

- [x] **tee ログ保存**: テスト実行結果をstatus内に記録
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: テスト結果を正直に報告
- [x] **コミットハッシュ記録**: push後に記録

---
