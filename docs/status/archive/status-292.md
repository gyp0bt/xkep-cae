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

### 初回検証（free_end_mode=False）

`free_end_mode=False`（MPC使用）で実行 → frac=0.0016で発散。status-285コミットでも同一結果。
worktreeテストでstatus-285の正しい設定が`free_end_mode=True`であることが判明。

### 正式検証（free_end_mode=True）— **完走達成**

| 指標 | status-285（修正前） | 今回（修正後） | 改善 |
|------|---------------------|---------------|------|
| frac | 0.9981 | **1.0000** | **完走達成** |
| cutback | 60 | **47** | **22%削減** |
| incr | 551 | 553 | 同等 |
| elapsed | — | 778 sec | — |

**s_unclamped修正 + StJacobian 2×2カップリング修正により、90度曲げが初めて完走（frac=1.0）。**

カットバック22%削減は、K_st精度向上によりNR接線が正確になり、不要なカットバックが減少したことを示す。

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

- [ ] frozen-m非局所項のz方向DOFカップリング: Hermite 3Dヘリカル配置でK_stがs端部でゼロになる問題（現テストではs_unc=2.886で端部、内部接触点での検証が未完）
- [ ] StJacobian 1×1フォールバックの遷移帯: w_t ∈ (1e-10, 0.5) の中間領域で2×2系と1×1系の切替が急峻。smooth切替の検討
- [ ] free_end_mode=False（MPC）での90度曲げ収束改善（現在frac=0.0016で発散）

---

## 次の担当者向け

### StJacobian 2×2カップリング修正の意義

**tクランプ時のK_st精度が劇的に改善（94%→0.0001%）。** これにより:
1. 端部接触でのK_st過大問題が解消
2. NRソルバーの接線剛性精度が向上
3. 2次収束率の改善が期待される

### 90度曲げ完走の意義

status-285では frac=0.9981（99.8%、残り0.2%で停止）だったが、今回の修正で **frac=1.0（完走）** を達成。K_st精度向上によりカットバックが22%削減され、NR収束が安定化した。

### 注意: free_end_mode

status-285の90度曲げベンチマークは `free_end_mode=True`（MPC不使用、直接処方変位）で実行されている。`free_end_mode=False`（MPC使用）では現在frac=0.0016で発散する別問題がある。

---

## STA2 準拠チェック

- [x] **tee ログ保存**: `/tmp/log-s_unclamped-90deg-freeend.log`
- [x] **再現手順**: `python contracts/verify_s_unclamped_90deg.py 2>&1 | tee /tmp/log.log`
- [x] **数値の捏造なし**: frac=1.0, incr=553, cutback=47 は tee ログと一致
- [x] **ベースライン先行取得**: status-285 frac=0.9981（worktreeで再現確認済み）
- [x] **回帰なし**: 631 passed, 0 failed

---
