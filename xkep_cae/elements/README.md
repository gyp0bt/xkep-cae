# elements/ — 有限要素ライブラリ

[← README](../../README.md) | [← roadmap](../../docs/roadmap.md)

## 概要

FEM要素の剛性行列・内力・質量行列を実装する。
全要素は `core/` の `ElementProtocol` に準拠。

## 要素一覧

### 平面要素

| ファイル | 要素 | DOF/節点 | 特徴 |
|---------|------|---------|------|
| `quad4.py` | Q4 | 2 | 4節点四辺形、2×2 Gauss積分 |
| `quad4_bbar.py` | Q4_BBAR | 2 | B-bar法（体積ロッキング防止） |
| `quad4_eas_bbar.py` | Q4_EAS | 2 | EAS-4 + B-bar（デフォルト） |
| `tri3.py` | TRI3 | 2 | 3節点三角形、1点積分 |
| `tri6.py` | TRI6 | 2 | 6節点三角形、3点積分 |

### 梁要素

| ファイル | 要素 | DOF/節点 | 特徴 |
|---------|------|---------|------|
| `beam_eb2d.py` | EB2D | 3 | Euler-Bernoulli 2D |
| `beam_timo2d.py` | Timo2D | 3 | Timoshenko 2D（Cowper κ(ν)） |
| `beam_timo3d.py` | Timo3D / CR | 6 | Timoshenko 3D（12DOF）、CR定式化（幾何学的非線形）|

### Cosserat rod

| ファイル | 要素 | DOF/節点 | 特徴 |
|---------|------|---------|------|
| `beam_cosserat.py` | Cosserat | 7 | 四元数回転、B行列、SRI、初期曲率 |

### 3D固体要素

| ファイル | 要素 | DOF/節点 | 特徴 |
|---------|------|---------|------|
| `hex8.py` | C3D8 | 3 | SRI+B-bar（デフォルト）|
| | C3D8B | 3 | B-bar（平均膨張法） |
| | C3D8R | 3 | 低減積分+アワーグラス制御 |
| | C3D8I | 3 | 非適合モード |

### 非線形要素

| ファイル | 内容 |
|---------|------|
| `continuum_nl.py` | TL/UL定式化（Q4要素の幾何学的非線形） |

## 設計仕様書

| 仕様書 | 内容 |
|--------|------|
| [cosserat-design](../../docs/cosserat-design.md) | Cosserat rod 四元数回転・B行列定式化 |
| [abaqus-differences](../../docs/abaqus-differences.md) | xkep-cae と Abaqus の既知の差異 |

## 検証

解析解との比較検証は [docs/verification/validation.md](../../docs/verification/validation.md) を参照（検証図15枚）。
