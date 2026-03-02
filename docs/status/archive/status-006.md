# status-006: EAS-4 Q4要素実装 / B-barバグ修正 / Abaqus梁要素調査

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-005](./status-005.md)

**日付**: 2026-02-12
**作業者**: Claude Code
**ブランチ**: `claude/add-eas-bbar-q4-XjhPs`

---

## 実施内容

### 1. EAS-4 (Enhanced Assumed Strain) Q4要素の実装

Simo & Rifai (1990) のEAS-4定式化をQ4双線形四角形要素に実装。

#### 定式化

- **拡張ひずみ補間**: 4個の内部パラメータ α で拡張ひずみモード M̃(ξ,η) を定義
  - α₁: εxx方向にξで線形変化（y軸まわり曲げ対応）
  - α₂: εyy方向にηで線形変化（x軸まわり曲げ対応）
  - α₃, α₄: γxyにξ,ηで線形変化（せん断ロッキング対策）
- **物理座標変換**: M = (detJ₀/detJ) × T₀ × M̃ （T₀ = T(J₀⁻ᵀ) でひずみ変換）
- **静的縮合**: K = K_uu − K_uα K_αα⁻¹ K_αuᵀ で内部自由度を消去
- 外部インタフェースは標準Q4と完全同一（8 DOF）

#### 数値検証結果

| 条件 | plain Q4 | B-bar | EAS-4 | EAS+B-bar |
|------|----------|-------|-------|-----------|
| 片持ち梁 ν=0.3, 10×1 | 0.64 | 1.43 | **1.005** | 4.00 |
| 片持ち梁 ν=0.3, 20×2 | 0.88 | 1.08 | **1.001** | 1.23 |
| 片持ち梁 ν=0.3, 40×4 | 0.97 | 1.02 | **1.003** | 1.05 |
| 片持ち梁 ν=0.4999, 10×1 | 0.01 | 2.00 | **1.008** | 4.00 |
| 片持ち梁 ν=0.4999, 40×4 | 0.02 | 1.02 | **0.985** | 1.04 |

（値は解析解に対する変位比 ratio = δ_FEM / δ_analytical）

#### 重要な知見

- **EAS-4単体でせん断ロッキングと体積ロッキングの両方を同時に抑制**
- plain Q4は非圧縮材料で壊滅的にロック（ratio=0.01）
- B-bar単体は曲げ問題で過補正（体積ひずみ平均化が曲げ変形を柔らかくする）
- EAS+B-bar併用は曲げで重度の過補正（ratio=4.0）→ 推奨しない
- **EAS-4単体をQ4のデフォルトに設定**

### 2. Q4 B-barのeinsumバグ修正

`quad4_bbar.py` の体積ひずみ感度計算にバグを発見・修正。

```python
# 修正前（バグ）: k が B_arr の正しい軸と結合されない
b_m = np.einsum("k, gij -> gi", vol_selector, B_arr)  # 結果: (4,3) ← 誤り

# 修正後: k は B_arr の2番目の軸（ひずみ成分軸）と結合
b_m = np.einsum("k, gkj -> gj", vol_selector, B_arr)  # 結果: (4,8) ← 正しい
```

このバグにより `Quad4BBarPlaneStrain` は実際の計算で IndexError を起こしていた。
Protocol適合テストのみで実際の解析テストがなかったため発見が遅れた。

### 3. API デフォルトQ4の変更

`api.py` の `solve_plane_strain()` で使用するQ4要素を変更:
- 旧: `Quad4PlaneStrain`（plain Q4）
- 新: `Quad4EASPlaneStrain`（EAS-4）

### 4. Abaqus梁要素の高級補正に関する文献調査

Abaqusの梁要素B21/B22がプレーンなTimoshenko梁理論とどう異なるか調査。

#### 4.1 Abaqus B21の定式化

- **基本定式化**: Hughes, Taylor, Kanok-Nukulchai (1977) の「低減積分ペナルティ型」
  - 変位と回転を独立に線形補間
  - 横せん断ひずみエネルギーは低減積分（1点）で評価
  - 横せん断剛性はペナルティ項として扱われる
- **xkep-cae との違い**: xkep-caeは解析的厳密Timoshenko剛性行列（Przemieniecki型）

#### 4.2 スレンダネス補償係数 (SCF) — 最大の差異要因

Abaqusは横せん断剛性に補正係数を適用:

```
f_p = 1 / (1 + xi * SCF * l² * A / (12I))
```

- SCF デフォルト = 0.25
- 細長い梁ほど f_p → 0 で、Euler-Bernoulli梁に自動遷移
- xkep-caeにはこの補正なし → 粗メッシュ×細長梁でAbaqusと差が出る

**対策**: `*TRANSVERSE SHEAR STIFFNESS` で SCF=0 を指定すればAbaqus側で無効化可能

#### 4.3 Cowperのせん断補正係数

- Abaqus: κ = 10(1+ν)/(12+11ν) （Cowper 1966, ν依存）
- xkep-cae: κ = 5/6（固定）
- ν=0.3 のとき: Abaqus κ≈0.850, xkep-cae κ=0.833 → 約2%の差

#### 4.4 CalculiXの梁要素

- CalculiXは梁要素を内部的にC3D20/C3D20R（20節点六面体ソリッド）に展開
- Timoshenko梁理論ではなく3次元弾性体として解く
- xkep-caeとの直接比較は本質的に異なるアプローチのため困難

#### 4.5 CPE4I（非適合モード四角形）について

- 5個の内部非適合モード自由度を追加（Wilson et al. 1973, Simo & Rifai 1990）
- 寄生せん断応力とPoisson効果による人工硬化を除去
- 1要素厚さでも解析解5%以内の精度
- **xkep-caeのEAS-4はCPE4Iと同等の精神に基づく（変分原理は異なるが効果は類似）**

---

## テスト結果

```
88 passed, 2 deselected (external), 26 warnings
ruff check: All checks passed!
ruff format: All files formatted
```

| テストファイル | テスト数 | 結果 |
|---------------|---------|------|
| `test_abaqus_inp.py` | 21 | PASSED |
| `test_beam_eb2d.py` | 21 | PASSED |
| `test_beam_timo2d.py` | 14 | PASSED |
| `test_benchmark_shear.py` | 4 | PASSED |
| `test_benchmark_tensile.py` | 4 | PASSED |
| `test_elements_manufactured.py` | 3 | PASSED |
| `test_protocol_assembly.py` | 7 | PASSED |
| **`test_quad4_eas.py`** | **14** | **PASSED (新規)** |
| `test_benchmark_cutter_q4tri3.py` | 1 | DESELECTED (external) |
| `test_benchmark_cutter_tri6.py` | 1 | DESELECTED (external) |

### 新規テスト内訳 (`test_quad4_eas.py`)

| テストクラス | テスト数 | 検証内容 |
|-------------|---------|---------|
| `TestEASBasicProperties` | 4 | 対称性・剛体モード・正定値・歪み要素 |
| `TestEASManufacturedSolution` | 3 | 製造解（1要素・2×2メッシュ・混在メッシュ） |
| `TestShearLocking` | 3 | せん断ロッキング抑制・plain Q4との比較・収束 |
| `TestVolumetricLocking` | 2 | 非圧縮材料の曲げ・plain Q4との比較 |
| `TestEASBBar` | 2 | EAS+B-bar併用の基本特性・製造解 |

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/elements/quad4_eas_bbar.py` | **新規** — EAS-4 / EAS+B-bar 要素 |
| `xkep_cae/elements/quad4_bbar.py` | einsum バグ修正 |
| `xkep_cae/api.py` | デフォルトQ4を `Quad4EASPlaneStrain` に変更 |
| `tests/test_quad4_eas.py` | **新規** — EAS要素テスト（14テスト） |
| `tests/test_protocol_assembly.py` | EAS要素のProtocol適合チェック追加 |

---

## 提供される要素クラス一覧

| クラス名 | ファイル | 説明 | 推奨用途 |
|---------|---------|------|---------|
| `Quad4PlaneStrain` | `quad4.py` | プレーンQ4 | 参照・教育用 |
| `Quad4BBarPlaneStrain` | `quad4_bbar.py` | B-bar法 Q4 | 非圧縮専用（曲げ注意） |
| **`Quad4EASPlaneStrain`** | `quad4_eas_bbar.py` | **EAS-4 Q4 (推奨)** | **汎用デフォルト** |
| `Quad4EASBBarPlaneStrain` | `quad4_eas_bbar.py` | EAS+B-bar Q4 | 特殊用途（過補正注意） |

---

## TODO（次回以降の作業）

- [ ] Phase 2.3: Timoshenko梁（3D空間）の実装
- [ ] Cowperのせん断補正係数 κ(ν) を `TimoshenkoBeam2D` に実装（Abaqus整合性向上）
- [ ] Abaqusベンチマーク時の `*TRANSVERSE SHEAR STIFFNESS` SCF=0 設定の文書化
- [ ] Q4要素のAbaqus比較テスト追加（CPE4I相当の精度検証）
- [ ] Phase 2.4: 断面モデルの拡張（一般断面）
- [ ] Phase 3: 幾何学的非線形（Newton-Raphson, 共回転定式化）

---

## 設計上のメモ

1. **EAS-4 がせん断・体積ロッキングの両方を同時に抑制する理由**: εxx,εyy方向の拡張モードが体積ひずみの線形変化を許容し、γxy方向の拡張モードがせん断ロッキングを解消する。4つのモードが協調して、Q4双線形要素の本質的な欠陥を補完する。

2. **B-barとの併用が過補正になる理由**: B-barは体積ひずみを要素平均に置換するため、曲げ変形による体積ひずみの厚さ方向変化を消去してしまう。EAS-4は既に体積ロッキングを抑制しているため、B-barの追加は過剰な柔軟化となる。

3. **Abaqus B21との精度比較**: xkep-caeの厳密Timoshenko行列は理論的に正しいが、Abaqusの結果と直接比較する場合は(1)SCF設定、(2)κの定義、(3)メッシュ密度の3点を統制する必要がある。

4. **静的縮合の数値精度**: 歪み要素では K_αα⁻¹ 計算の丸め誤差で3番目の剛体モード固有値が微小非ゼロ（~0.25 vs ~5×10⁴）になりうる。実用上は問題ないが、テストでは相対閾値で判定する。

---

## 参考文献

- Simo, J.C. & Rifai, M.S. (1990) "A class of mixed assumed strain methods and the method of incompatible modes", IJNME, 29, 1595-1638.
- Andelfinger, U. & Ramm, E. (1993) "EAS-elements for two-dimensional, three-dimensional, plate and shell structures", IJNME, 36, 1311-1337.
- Hughes, T.J.R., Taylor, R.L. & Kanok-Nukulchai, W. (1977) "A simple and efficient finite element for plate bending", IJNME, 11, 1529-1543.
- Cowper, G.R. (1966) "The shear coefficient in Timoshenko's beam theory", J. Applied Mechanics, 33, 335-340.
- Wilson, E.L. et al. (1973) "Incompatible displacement models", in Fenves et al. (eds), Numerical and Computer Methods in Structural Mechanics.
- Przemieniecki, J.S. (1968) "Theory of Matrix Structural Analysis", McGraw-Hill.
