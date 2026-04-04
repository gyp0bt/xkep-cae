# status-291: K_c不整合の根本原因特定 + s_unclamped修正

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-04
- **ブランチ**: `claude/check-status-todos-O9uDz`
- **テスト数**: 624+ passed（回帰なし、新規FDテスト5件追加）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-290のTODO（K_mat/K_geo/K_stの個別FD検証）を実行し、K_c不整合の根本原因を特定・修正。

### 最重要発見

**K_cの不整合は100% K_stに帰属。K_matとK_geoは正確。**

K_stが過大な値を持つ原因: **s_unclampedがペア状態に格納されず、StJacobianがクランプ済みsをs_unclampedとして使用 → smooth_clip_derivの重みが不正確。**

### 修正効果

| 構成 | 修正前 rel_err | 修正後 rel_err | 改善率 |
|------|--------------|--------------|--------|
| linear+hermite | 19.9% | **0.0001%** | 10万倍 |
| hertz+hermite | 15.7% | **0.0001%** | 10万倍 |
| linear+linear | 7.85% | 7.85% | - (inherent) |
| hertz+linear | 5.26% | 5.26% | - (inherent) |

線形ケースの残存誤差は smooth_clip 境界効果（s_unc=0.0 で w=0.5）によるもので、接触点が厳密にセグメント端部にある場合の inherent な誤差。

---

## 実装内容

### 1. K_c コンポーネント個別FD検証テスト

`test_kc_component_fd.py` を新規作成。K_c = K_mat - K_geo + K_st の各成分を個別にFDと比較:

- `tangent_components()` メソッドを `HuberContactForceProcess` に追加
- geometry更新 → evaluate → FD の完全パイプラインで検証
- 4構成（linear/hertz × linear/hermite）を網羅

### 2. s_unclamped パイプライン貫通

**根本原因**: Hermite精密化で s_unc > 1.02 の場合、smooth_clip後に s=1.0 として格納。StJacobianは s=1.0 を s_unclamped として使い、smooth_clip_deriv(1.0) = 0.5 を得る。しかし真の s_unclamped > 1.02 では重みは 0.0 であるべき。

**修正**:
1. `_ContactStateOutput` に `s_unclamped`/`t_unclamped` フィールドを追加
2. `_closest_point_hermite_refine` でクランプ前の値を返すように変更
3. `_closest_point_segments_batch` で有効 s_unc/t_unc を返すように変更
4. `UpdateGeometryProcess` で s_unclamped/t_unclamped をペア状態に格納
5. `_add_kst_contact_to_coo` で StJacobianInput に s_unclamped/t_unclamped を渡す

### 3. C17違反修正

`NRIterationSnapshot` → `NRIterationSnapshotOutput` にリネーム（C17命名規約準拠）。

---

## 根本原因の詳細分析

### K_st の DOF レベル不整合パターン（Hermiteケース、修正前）

| DOF | comp | FD | K_c | 不整合源 |
|-----|------|----|-----|---------|
| node 2, x | x | -4.29e+03 | -2.15e+03 | K_geo=-4.29e3(正確), **K_st=+2.14e3(過大)** |
| node 2, y | y | +1.00e+04 | +9.10e+03 | K_mat=+1.0e4(正確), **K_st=-9.00e2(過大)** |
| node 6, x | x | +4.29e+03 | +2.15e+03 | K_geo=+4.29e3(正確), **K_st=-2.14e3(過大)** |

**K_stがK_mat/K_geoの正しい寄与を50%打ち消していた。**

### なぜ Hermite で問題が顕在化するか

- 線形形状関数: s_unc = s → smooth_clip_deriv重みは一致
- Hermite: Newton精密化で s_unc が [0,1] の外に出る → smooth_clip で s=0/1 に格納
- 格納された s=0/1 を s_unc として使うと smooth_clip_deriv(1.0)=0.5 (真の重み0.0)

---

## 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/_contact_pair.py` | s_unclamped/t_unclamped フィールド追加 |
| `xkep_cae/contact/_manager_process.py` | s_unc 格納パイプライン |
| `xkep_cae/contact/geometry/_compute.py` | _closest_point_* で s_unc 返却 |
| `xkep_cae/contact/contact_force/strategy.py` | tangent_components() + s_unc 伝搬 |
| `xkep_cae/contact/solver/_diagnostics.py` | NRIterationSnapshotOutput リネーム |
| `xkep_cae/contact/solver/_newton_dynamic.py` | リネーム追従 |
| `xkep_cae/contact/geometry/strategy.py` | 戻り値更新 |
| `contracts/analyze_chattering_breakdown.py` | リネーム追従 |
| `tests/contact/test_st_jacobian.py` | 戻り値更新 |
| `tests/contact/test_consistent_st_tangent.py` | 戻り値更新 |
| `xkep_cae/contact/contact_force/tests/test_kc_component_fd.py` | **新規** FDテスト |

---

## TODO

- [ ] 線形ケースの7.85%残存誤差: smooth_clip境界効果の改善（端部接触でK_stが不要に大きい問題は残るが、実際のソルバーでは接触点は通常セグメント内部にあるため影響は限定的）
- [ ] 90度曲げでの s_unclamped 修正効果検証（NR 2次収束改善の確認）
- [ ] frozen-m 解消: ∂m/∂u の正確計算 → z方向DOFカップリングをK_stに追加

---

## 次の担当者向け

### s_unclamped 修正の意義

**Hermite構成でのK_stが正しい smooth_clip_deriv 重みで計算されるようになった。** これにより:
1. 端部接触点（s≈0 or s≈1）でのK_st過大問題が解消
2. K_c全体のFD整合度がHermiteで20%→0.0001%に改善
3. NR 2次収束率の改善が期待される（K_T の精度向上）

### 次のステップ

1. **90度曲げでの実機検証**: s_unclamped修正がfrac改善に寄与するか確認
2. **frozen-m 解消**: status-289で特定されたz方向DOFカップリング問題への対処

---
