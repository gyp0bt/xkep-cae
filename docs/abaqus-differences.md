# xkep-cae と Abaqus の差異

[← README](../README.md) | [ロードマップ](roadmap.md) | [最新status](status/status-009.md)

xkep-cae の要素定式化と Abaqus との既知の差異をまとめる。
Abaqus ベンチマークとの比較時に参照すること。

---

## 1. Q4 要素の対応関係

| xkep-cae | Abaqus | 定式化 | 備考 |
|----------|--------|--------|------|
| `Quad4PlaneStrain` | CPE4 | フル積分Q4 | せん断ロッキング・体積ロッキングあり |
| `Quad4EASPlaneStrain` | **CPE4I** | 非適合モード / EAS | ロッキング抑制。**xkep-caeの推奨デフォルト** |
| `Quad4BBarPlaneStrain` | CPE4H（近似） | B-bar / ハイブリッド | 非圧縮材料に特化。曲げ過補正に注意 |
| `Quad4EASBBarPlaneStrain` | （該当なし） | EAS+B-bar併用 | 実験的。曲げで過補正（非推奨） |

### 1.1 EAS-4 と CPE4I の差異

| 項目 | xkep-cae EAS-4 | Abaqus CPE4I |
|------|----------------|-------------|
| **理論的基盤** | Simo-Rifai (1990) EAS 定式化 | Wilson et al. (1973) 非適合モード + Simo-Rifai |
| **拡張パラメータ数** | 4（εxx, εyy, γxy×2） | 5（2D: 直接変位の非適合モード）|
| **効果** | せん断・体積ロッキング同時抑制 | 寄生せん断・Poisson効果除去 |
| **精度** | 粗メッシュ10×1で解析解比 1.005 | 粗メッシュでも解析解5%以内（文献値）|

両者は異なる変分原理に基づくが、実用的な精度は同等レベル。

### 1.2 Abaqus CPE4R（低減積分）との差異

xkep-cae には CPE4R 相当の要素はない。
CPE4R は1点積分 + hourglass制御であり、EAS-4 とは定式化が根本的に異なる。

---

## 2. 梁要素の差異

### 2.1 要素対応関係

| xkep-cae | Abaqus | 定式化 |
|----------|--------|--------|
| `EulerBernoulliBeam2D` | B23（2D） | Euler-Bernoulli梁（3次Hermite補間） |
| `TimoshenkoBeam2D` | B21（2D） | Timoshenko梁（せん断変形考慮） |
| `TimoshenkoBeam3D` | B31（3D） | 3D Timoshenko梁（12DOF, 二軸曲げ+ねじり） |

### 2.2 せん断補正係数 κ（Cowperの補正）

| 項目 | xkep-cae（デフォルト）| xkep-cae（cowperモード） | Abaqus B21 |
|------|---------------------|-------------------------|------------|
| **矩形断面** | κ = 5/6 ≈ 0.8333 | κ = 10(1+ν)/(12+11ν) | κ = 10(1+ν)/(12+11ν) |
| **円形断面** | κ = 5/6 ≈ 0.8333 | κ = 6(1+ν)/(7+6ν) | κ = 6(1+ν)/(7+6ν) |
| **ν=0.3 矩形** | 0.8333 | 0.8497 | 0.8497 |
| **ν=0.3 円形** | 0.8333 | 0.8864 | 0.8864 |
| **差（矩形, ν=0.3）** | — | 0% | **約2%** |

#### Abaqus 準拠にする方法

```python
from xkep_cae.elements.beam_timo2d import TimoshenkoBeam2D
from xkep_cae.sections.beam import BeamSection2D

sec = BeamSection2D.rectangle(b=10.0, h=10.0)
beam = TimoshenkoBeam2D(section=sec, kappa="cowper")  # Abaqus準拠
```

`kappa="cowper"` を指定すると、材料の ν から断面形状に応じた
Cowper (1966) の補正係数を自動計算する。

参考文献: Cowper, G.R. (1966) "The shear coefficient in Timoshenko's beam theory",
J. Applied Mechanics, 33, 335-340.

### 2.3 スレンダネス補償係数 (SCF)

**これは xkep-cae と Abaqus の最も大きな差異要因である。**

Abaqus B21/B22 は横せん断剛性に「スレンダネス補償係数」を適用する:

```
f_p = 1 / (1 + ξ · SCF · L²A / (12I))
```

- SCF デフォルト = 0.25
- 細長い梁ほど f_p → 0 で、Euler-Bernoulli梁に自動遷移

**xkep-cae では `scf` パラメータでオプション対応**:

```python
beam = TimoshenkoBeam2D(section=sec, scf=0.25)  # Abaqus相当のSCF
beam = TimoshenkoBeam3D(section=sec, scf=0.25)   # 3D版も同様
```

xkep-cae の SCF はせん断パラメータ Φ を直接低減する:
Φ_eff = Φ · f_p → 細長い梁で Φ → 0（EB遷移）。
Abaqus はペナルティ法の横せん断剛性を制限する。
適用メカニズムが異なるため、SCF=0.25 でも数値的に完全一致しない場合がある。

#### Abaqus 比較時の設定

xkep-cae `scf=None`（デフォルト）と Abaqus を直接比較する場合、
Abaqus 側で SCF を無効化する必要がある:

```
*BEAM SECTION, SECTION=RECT, ELSET=beams, MATERIAL=steel
10.0, 10.0
*TRANSVERSE SHEAR STIFFNESS
** K11, K22, K12 をマニュアル指定して SCF を無効化
** K = κ·G·A （κ=5/6 or Cowper, G=E/(2(1+ν))）
** 矩形 10×10, E=200e3, ν=0.3:
** κGA = (5/6) × 76923 × 100 = 6,410,256
6410256.0, 6410256.0
```

あるいは、Section Controls で SCF=0 を指定:

```
*SECTION CONTROLS, NAME=no_scf
** SLENDERNESS_COMPENSATION_FACTOR = 0
, , , , , , , , , 0.0
```

**SCF を無効化しないと、特に粗メッシュ×細長い梁で
xkep-cae と Abaqus の結果に大きな差が出る。**

### 2.4 剛性行列の定式化

| 項目 | xkep-cae | Abaqus B21 |
|------|----------|-----------|
| **定式化** | Przemieniecki型の解析的厳密行列 | Hughes-Taylor-Kanok-Nukulchai (1977) 低減積分ペナルティ型 |
| **せん断ひずみ積分** | 解析的（厳密） | 低減積分（1点） |
| **せん断剛性** | 剛性行列に直接組み込み | ペナルティ項として扱う |
| **精度** | 理論的に厳密 | 実用的に同等 |

両者は異なるアプローチだが、十分なメッシュ分割では同じ結果に収束する。

### 2.5 CalculiX の梁要素

CalculiX は梁要素を内部的に C3D20/C3D20R（20節点六面体ソリッド）に展開する。
Timoshenko梁理論ではなく 3次元弾性体として解くため、
xkep-cae との直接比較は困難。

---

## 3. Abaqus ベンチマーク比較のチェックリスト

xkep-cae と Abaqus の結果を比較する際に確認すべき項目:

### Q4 要素の比較

- [ ] Abaqus の要素タイプ（CPE4, CPE4I, CPE4R, CPE4H）を明確にする
- [ ] xkep-cae の対応要素を選択（上記対応表参照）
- [ ] 節点順序の整合性を確認（反時計回り）
- [ ] 平面ひずみ vs 平面応力の確認

### 梁要素の比較

- [ ] Abaqus 側で `*TRANSVERSE SHEAR STIFFNESS` を指定して SCF を無効化
- [ ] κ の定義を統一（Cowper or 固定 5/6）
- [ ] メッシュ密度の統一
- [ ] 断面定義の数値精度を統一

### 共通

- [ ] 材料定数（E, ν）の完全一致
- [ ] 境界条件の定義方法の一致
- [ ] 荷重条件（集中 vs 分布、等価節点力）の一致

---

## 4. 今後の差異解消予定

| 項目 | 現状 | 計画 |
|------|------|------|
| Cowper κ | `kappa="cowper"` で対応済み | — |
| SCF | `scf` パラメータで対応済み | — |
| CPE4R 相当 | 未実装 | 優先度低（EAS-4で十分） |
| 3D梁要素 | `TimoshenkoBeam3D` 実装済み | — |
| シェル要素 | 未実装 | Phase 8.3 で検討 |

---

## 参考文献

- Cowper, G.R. (1966) "The shear coefficient in Timoshenko's beam theory", J. Applied Mechanics, 33, 335-340.
- Hughes, T.J.R., Taylor, R.L. & Kanok-Nukulchai, W. (1977) "A simple and efficient finite element for plate bending", IJNME, 11, 1529-1543.
- Simo, J.C. & Rifai, M.S. (1990) "A class of mixed assumed strain methods and the method of incompatible modes", IJNME, 29, 1595-1638.
- Wilson, E.L. et al. (1973) "Incompatible displacement models", in Fenves et al. (eds), Numerical and Computer Methods in Structural Mechanics.
- Przemieniecki, J.S. (1968) "Theory of Matrix Structural Analysis", McGraw-Hill.
- Abaqus Theory Manual, Section 3.2.1 (Beam Elements) and 3.6.3 (Incompatible Mode Elements).
