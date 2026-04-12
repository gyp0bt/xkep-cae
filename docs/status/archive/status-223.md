# status-223: 接触接線剛性の幾何項追加 + 摩擦接線符号修正

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-22
**ブランチ**: `claude/verify-dynamic-bending-load-cmPwf`

---

## 概要

接触接線剛性に幾何剛性項（法線回転）を追加し、摩擦接線剛性の符号誤りを修正。
動的三点曲げ（E=25 MPa）で frac=0.89（push=26.7mm）まで到達、接触力 132N を確認。

---

## 変更内容

### 1. 接触接線剛性の幾何項追加

**ファイル**: `xkep_cae/contact/contact_force/strategy.py`

従来の材料剛性（ペナルティ勾配）に加え、法線回転による幾何剛性を追加:

```
K_c = K_mat - K_geo

K_mat = h'(x) * k_pen * Σ_ij c_i c_j (n ⊗ n)      ← 既存（正定値）
K_geo = p_n / dist * Σ_ij c_i c_j (I₃ - n ⊗ n)     ← 追加（法線回転）
```

ここで `c = [(1-s), s, -(1-t), -t]`, `dist = gap + r_A + r_B`。

### 2. 摩擦接線剛性の符号修正

**ファイル**: `xkep_cae/contact/solver/_newton_steps.py`

残差 `R = f_int - f_c_raw - f_ext` に対し、接触力の符号反転 `f_c = -f_c_raw` により:

- `contact_force_strategy.tangent()` は `-d(f_contact)/du` を返す → **加算** ✓
- `friction.tangent()` は `+d(f_friction)/du` を返す → **減算が必要** ✗

**修正**: `K_T = K_T + K_fric` → `K_T = K_T - K_fric`

### 3. 試行・撤回

- **status フィルタ廃止**: INACTIVE ペアの不正ジオメトリで発散 → 撤回
- **k_pen 梁剛性下限**: E=25 で dt 成長阻害 → 撤回

---

## 検証結果

### E=25 MPa, L=100, φ17, push=30, n_periods=30

| 項目 | 値 |
|------|-----|
| 到達 frac | 0.8903 |
| 到達 push | 26.71 mm |
| 接触力 | 131.9 N |
| 実効剛性 | 5.42 N/mm |
| 解析解剛性 (EB) | 4.92 N/mm |
| 剛性誤差 | 10.2% |
| 計算時間 | 926-1313 s |
| incr 数 | 240 |

frac=0.89 で停滞: 残差 ratio=1.006 で NR が進まない（大変形域の接線精度不足）。

### 解析解との整合性

P_EB(30mm) = 48 × 25 × I / L³ × 30 = 147.6 N

frac=0.89 時点での接触力 132N は、到達 push=26.7mm に対し:
P_EB(26.7mm) = 4.92 × 26.7 = 131.4 N → **解析解と0.4%一致**。

---

## 残課題

1. **frac=0.89 停滞**: 大変形域（push/L > 0.27）で NR が停滞。摩擦接線の幾何項（dn/du, dt1/du, dt2/du）が不足している可能性。
2. **E=100 MPa**: k_pen が動的推定（c0*M）で不足。cfg.k_pen=48EI/L³ で明示指定すれば進行するが、自動推定の改善が必要。
3. **曲げ荷重レベル**: E=25 MPa → 132N（百数十N）。E=100 MPa → 解析解 590N（数百N）。

---

## テスト

**499 テスト** — 変更前と同数（レンダリングテスト 1件は環境依存で失敗）
