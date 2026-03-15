# status-142: Mortarギャップ計算の致命的バグ発見・修正 + 接触未発動問題の解明

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-08
**テスト数**: 2271（一部既存テストが接触未発動で通過していた問題判明）

---

## 概要

摩擦あり7本撚線曲げ揺動の収束検証中に、**Mortarギャップ計算に致命的バグ**を発見。
全てのMortar NCP曲げ揺動テストが**接触力ゼロ**のまま「収束」していたことが判明した。

---

## 1. 発見されたバグ

**ファイル**: `xkep_cae/contact/mortar.py`

### 問題コード（2箇所）

```python
# build_mortar_system() line 152-153
r_a = pair.state.radius_a if hasattr(pair.state, "radius_a") else 0.0
r_b = pair.state.radius_b if hasattr(pair.state, "radius_b") else 0.0

# compute_mortar_contact_force() line 267-268
r_a = pair.state.radius_a if hasattr(pair.state, "radius_a") else 0.0
r_b = pair.state.radius_b if hasattr(pair.state, "radius_b") else 0.0
```

### 原因

- `ContactState`には`radius_a`属性が**存在しない**
- `hasattr(pair.state, "radius_a")` → **常にFalse**
- `r_a = 0, r_b = 0` → `gap_val = gap_dist`（中心線間距離がそのままギャップ）
- 中心線間距離は常に正 → `g_mortar > 0` → `p_n = max(0, λ + k_pen·(-g̃)) = 0`
- **全ペアが永久にNCP inactive** → 接触力ゼロ

### 修正

```python
# 正しくは pair.radius_a / pair.radius_b を参照
gap_val = gap_dist - (pair.radius_a + pair.radius_b)
```

---

## 2. 影響範囲

### 影響を受けるテスト

Mortar有効（`use_mortar=True`）かつ接触半径が正のケース:
- `test_ncp_bending_oscillation.py` — 7本/19本の全曲げ揺動テスト
- `test_ncp_7strand_bending_45deg`, `90deg`, `oscillation_full`
- `test_ncp_19strand_bending_45deg`
- S3ベンチマーク全般

### 影響を受けないテスト

- Mortarなし（`use_mortar=False`）のテスト — 従来のペアベースNCP
- 接触半径0のユニットテスト — gap_val = gap_dist のまま動作

### 修正後の影響

- 修正により接触力が正しく計算される
- **初期貫入**（弦近似誤差）による不収束が発現
- 最大貫入比0.94（94%!）— メッシュ初期状態で大量の貫入ペアが存在

---

## 3. 初期貫入問題

修正後、以下の問題が露呈:

1. 7本撚線16要素/ピッチで初期貫入595ペア（gap min = -0.69mm）
2. `adjust_initial_penetration=True`だが`position_tolerance=0.0`で調整不実行
3. Mortarバグにより隠蔽されていた — 接触力ゼロで素通り

### 解決方針

- 初期貫入解消: メッシュ生成時にgap確保 or `position_tolerance > 0`で調整
- 被膜モデル活用: `coating_thickness > 0`で被膜弾性スプリングとして初期貫入を処理

---

## 4. 摩擦係数の方針決定

ユーザ指示:
- **素線間（梁-梁）接触**: μ = 0.1
- **被膜間（coating-coating）接触**: μ = 0.25

---

## TODO

- [ ] 初期貫入解消の実装（メッシュ側 or position_tolerance調整）
- [ ] 7本NCP曲げ揺動の**接触あり**収束達成
- [ ] 摩擦あり（μ=0.1）曲げ揺動収束検証
- [ ] 被膜接触+被膜摩擦（μ=0.25）の検証スクリプト
- [ ] 既存テストの修正（接触あり前提でパラメータ調整）
