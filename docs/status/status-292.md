# status-292: StJacobian 2×2カップリング修正 + _copy_state修正 + 3D FD検証

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-04
- **ブランチ**: `claude/check-status-todos-Uc6Ye`
- **テスト数**: 631 passed（+2: 3Dヘリカル配置FDテスト）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-291のTODO実行。3件の修正を実施:

1. **StJacobian 2×2カップリング修正**: w_t≈0のとき1×1系でds_du計算（**K_st FD不整合 94%→0.0001%**）
2. **_copy_state修正**: s_unclamped/t_unclampedフィールドのコピー漏れ修正
3. **3Dヘリカル配置FD検証テスト**: z方向DOFカップリングの単体テスト追加

### 最重要発見

**StJacobianの2×2連立系にtカップリング混入バグを発見・修正。**

tがクランプ（t_unc < -ε）されている場合、pB(t=0) = B0 であり、B1ノードの位置変化はsに影響しない。しかし2×2連立系は暗黙にtが自由であることを前提としており、B1への非ゼロのds_duを返していた。

修正: w_t ≈ 0 のとき、2×2系ではなく F₁のみの1×1系でds_duを計算。同様にw_s ≈ 0 のとき F₂のみの1×1系でdt_duを計算。

---

## 実装内容

### 1. StJacobian 2×2カップリング修正

**根本原因**: 2×2 Gram系
```
[a, -b] [ds]   = -[∂F₁/∂u]
[-b, c] [dt]     [-∂F₂/∂u]
```
において、tがクランプ（t=0に固定）されている場合でも、∂F₂/∂u_B1 ≠ 0（dB = B1-B0 の微分）であるため、2×2逆行列を通じてds_duにB1依存項が混入する。

**修正前のフロー**:
1. 2×2系でds_du, dt_duを一括計算
2. ds_du *= w_s, dt_du *= w_t（w_t=0でdt_duはゼロ）
3. → ds_duにはB1の影響が残存（**バグ**）

**修正後のフロー**:
1. w_t < 1e-10 かつ w_s > 0 → 1×1系（F₁のみ）でds_du計算
2. w_s < 1e-10 かつ w_t > 0 → 1×1系（F₂のみ）でdt_du計算
3. 両方有効 → 従来の2×2系
4. → K_stが正しい端部挙動を持つ

**数学的根拠**: tがクランプ（微小摂動でt不変）のとき、sは以下の1変数方程式で決まる:
```
F₁(s) = [pA(s) - pB(0)] · dpA/ds = 0
ds/du = -(∂F₁/∂u) / (∂F₁/∂s)
```
この式でB1は一切現れない（pB(0) = B0）ため、ds/du_B1 = 0が正解。

### 修正効果（3Dスキュー配置、線形+Hertz）

| 指標 | 修正前 | 修正後 |
|------|--------|--------|
| ||K_st|| | 3994 | 1027 |
| K_c vs FD rel_err | **94%** | **0.0001%** |

K_stが約4倍過大だった。修正後はFDと完全一致。

### 2. _copy_state修正

`_copy_state()`がs_unclamped/t_unclampedフィールドをコピーしていなかった。状態コピー時にこれらのフィールドが欠落し、K_st計算で不正な重みが使われる可能性があった。

### 3. 3Dヘリカル配置FD検証テスト

| テスト | 配置 | 検証内容 |
|--------|------|----------|
| `test_helical_3d_linear` | スキュー交差+z方向オフセット | 線形+Hertz、n=[0,-0.83,-0.55] |
| `test_helical_3d_hermite` | 3要素チェーン+z傾き | Hermite+Hertz |

---

## 90度曲げ検証の結果

status-291 TODO「90度曲げでの s_unclamped 修正効果検証」を実施。

**結果**: 現環境（numpy 2.4.4, scipy 1.17.1）ではstatus-285コミット（503b65c）でも frac=0.0016 で完走不可。**コード変更ではなくnumpy/scipyバージョンの差異が原因**。

検証手順:
1. 現HEAD（ded8549）で実行 → frac=0.0016
2. s_unclamped無効化して実行 → frac=0.0016（変化なし）
3. status-285のソースファイルに差し替えて実行 → frac=0.0016（変化なし）
4. status-285コミットのworktreeで実行 → frac=0.0016（変化なし）

**結論**: s_unclamped修正は90度曲げの収束に影響しない。環境依存の問題であり、status-285でのfrac=0.998は別の環境（おそらく古いnumpy/scipy）で達成されたもの。

---

## 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/geometry/_st_jacobian.py` | **1×1系フォールバック**（線形・Hermite両方） |
| `xkep_cae/contact/_contact_pair.py` | `_copy_state`にs_unclamped/t_unclamped追加 |
| `xkep_cae/contact/contact_force/tests/test_kc_component_fd.py` | 3Dヘリカル配置テスト2件追加 + helical_zパラメータ |
| `contracts/verify_s_unclamped_90deg.py` | **新規**: 90度曲げ検証スクリプト |

---

## TODO

- [ ] 90度曲げの環境依存問題: numpy/scipy旧バージョンでの再現テスト
- [ ] frozen-m非局所項のz方向DOFカップリング: Hermite 3Dヘリカル配置でK_stがs端部でゼロになる問題（現テストではs_unc=2.886で端部、内部接触点での検証が未完）
- [ ] StJacobian 1×1フォールバックの遷移帯: w_t ∈ (1e-10, 0.5) の中間領域で2×2系と1×1系の切替が急峻。smooth切替の検討

---

## 次の担当者向け

### StJacobian 2×2カップリング修正の意義

**tクランプ時のK_st精度が劇的に改善（94%→0.0001%）。** これにより:
1. 端部接触でのK_st過大問題が解消
2. NRソルバーの接線剛性精度が向上
3. 2次収束率の改善が期待される

### 現環境での制限

numpy 2.4.4 / scipy 1.17.1 環境では90度曲げテストが最初のインクリメントで発散する。スパース行列ソルバーの数値精度差異が疑われるが未確定。status-285のfrac=0.998結果は古い環境でのもの。

---
