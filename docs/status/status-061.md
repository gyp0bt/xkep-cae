# status-061: HEX8 要素ファミリ拡充 — C3D8/C3D8R/C3D8I + Abaqus 命名準拠

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-25
**作業者**: Claude Code
**テスト数**: 1399（+5）

## 概要

HEX8（8節点6面体）要素の B-bar 実装を全面的に見直し、3つのバリエーション（C3D8/C3D8R/C3D8I）を実装した。Abaqus 準拠の要素タイプ名 `element_type` を全要素クラスに追加した。

### 旧実装の問題点

旧 `hex8_ke_bbar` は「B-bar法」と称していたが、実際には SRI の vol/dev 分割が逆（偏差=完全積分、体積=低減積分）であり、せん断ロッキング回避効果が不十分だった。

### 新実装（3バリアント）

| 要素名 | クラス | 積分方式 | ランク | 特徴 |
|--------|--------|---------|--------|------|
| **C3D8** | `Hex8SRI` | SRI（偏差=1点、体積=2×2×2） | 12 | 断面4×4以上で曲げ精度良好 |
| **C3D8R** | `Hex8Reduced` | 均一1点 + アワーグラス制御 | 6（制御なし）/ 15（制御あり） | 高速、陽解法向け |
| **C3D8I** | `Hex8Incompatible` | 非適合モード + 静的縮合 | 18 | 断面1×1でも曲げ高精度（< 5%） |

### Abaqus 準拠命名

全要素クラスに `element_type` 属性を追加:

| クラス | element_type | 説明 |
|--------|-------------|------|
| Quad4PlaneStrain | CPE4 | 平面ひずみ4節点 |
| Quad4BBarPlaneStrain | CPE4B | B-bar法 |
| Quad4EASPlaneStrain | CPE4E | EAS法 |
| Quad4EASBBarPlaneStrain | CPE4I | EAS+B-bar法 |
| Tri3PlaneStrain | CPE3 | 平面ひずみ3節点 |
| Tri6PlaneStrain | CPE6 | 平面ひずみ6節点 |
| Hex8SRI | C3D8 | 3D固体SRI |
| Hex8Reduced | C3D8R | 3D固体低減積分 |
| Hex8Incompatible | C3D8I | 3D固体非適合モード |
| EulerBernoulliBeam2D | B21E | 2D Euler-Bernoulli梁 |
| TimoshenkoBeam2D | B21 | 2D Timoshenko梁 |
| TimoshenkoBeam3D | B31 | 3D Timoshenko梁 |
| CosseratRod | B31C | Cosserat rod |

## 実装詳細

### `xkep_cae/elements/hex8.py`（全面書き換え）

**カーネル関数**:
- `_split_D_vol_dev(D)`: D行列を体積(rank 1) + 偏差(rank 5) に分解
- `hex8_ke_sri(node_xyz, D)`: C3D8 — 偏差=1点積分、体積=2×2×2完全積分
- `hex8_ke_reduced(node_xyz, D, *, alpha_hg=0.0)`: C3D8R — 1点積分 + Flanagan-Belytschko アワーグラス制御
- `hex8_ke_incompatible(node_xyz, D)`: C3D8I — Wilson-Taylor 非適合モード + 静的縮合

**クラス**:
- `Hex8SRI`: element_type="C3D8"
- `Hex8Reduced(alpha_hg=0.0)`: element_type="C3D8R", アワーグラス制御パラメータ
- `Hex8Incompatible`: element_type="C3D8I"
- `Hex8BBar = Hex8SRI`（後方互換エイリアス）
- `hex8_ke_bbar = hex8_ke_sri`（後方互換エイリアス）

**アワーグラス制御（Flanagan-Belytschko）**:
- 4つのアワーグラスベースベクトル（ξη, ξζ, ηζ, ξηζ モード）
- 人工剛性: `k_hg = alpha_hg * D_max * V / L_char²`
- 推奨値: alpha_hg = 0.03〜0.05
- alpha_hg=0.05 でランク 6→15（9モード制御）

### `tests/test_hex8.py`（50テスト）

9つのテストクラス:

| クラス | テスト数 | 内容 |
|--------|---------|------|
| TestShapeFunctions | 4 | 分配性、節点値、微分形状 |
| TestStiffnessBasicSRI | 8 | 対称性、ランク12、RBM、PSD、後方互換 |
| TestPatchTest | 3 | SRI/非適合モード パッチテスト |
| TestUniaxialCompression | 2 | 軸変位、ポアソン収縮 |
| TestElastic3D | 6 | D行列性質 |
| TestHex8SRIProtocol | 5 | Protocol適合、後方互換 |
| TestDSplit | 4 | vol/dev分解検証 |
| TestHex8Reduced | 8 | ランク6、アワーグラス制御（ランク増加/対称/PSD/クラス連携） |
| TestHex8Incompatible | 6 | ランク18、RBM、Protocol適合 |
| TestCantileverBending | 4 | 片持ち梁3バリアント比較 |

### `tests/mesh/test_sheath_bending_validation.py`（変更）

シース曲げモードバリデーションテストを `hex8_ke_bbar` → `hex8_ke_incompatible` に変更。C3D8I（非適合モード）は断面1×1要素でも曲げ精度が高い。

### 各要素ファイルへの `element_type` 追加

- `xkep_cae/elements/quad4.py`, `quad4_bbar.py`, `quad4_eas_bbar.py`
- `xkep_cae/elements/tri3.py`, `tri6.py`
- `xkep_cae/elements/beam_eb2d.py`, `beam_timo2d.py`, `beam_timo3d.py`, `beam_cosserat.py`

### `xkep_cae/core/element.py`（docstring更新）

ElementProtocol docstring に Abaqus 要素タイプ名一覧を追加。

## 片持ち梁バリデーション結果

L=10, h=w=1, E=200GPa, ν=0.3, P=1000N の片持ち梁で比較:

| バリアント | メッシュ (断面×長さ) | Timoshenko解に対する誤差 |
|-----------|---------------------|------------------------|
| C3D8I | 1×1×8 | < 5% |
| C3D8 (SRI) | 4×4×8 | < 5% |
| C3D8I > C3D8 | 2×2×8 | C3D8I の方が高精度 |

## ファイル変更

### 新規
- `docs/status/status-061.md`

### 変更
- `xkep_cae/elements/hex8.py` — 3バリアント実装 + Abaqus命名
- `xkep_cae/core/element.py` — docstring更新
- `tests/test_hex8.py` — 50テスト（+25新規、25更新）
- `tests/mesh/test_sheath_bending_validation.py` — C3D8I使用に変更
- `xkep_cae/elements/quad4.py` — element_type追加
- `xkep_cae/elements/quad4_bbar.py` — element_type追加
- `xkep_cae/elements/quad4_eas_bbar.py` — element_type追加
- `xkep_cae/elements/tri3.py` — element_type追加
- `xkep_cae/elements/tri6.py` — element_type追加
- `xkep_cae/elements/beam_eb2d.py` — element_type追加
- `xkep_cae/elements/beam_timo2d.py` — element_type追加
- `xkep_cae/elements/beam_timo3d.py` — element_type追加
- `xkep_cae/elements/beam_cosserat.py` — element_type追加
- `docs/status/status-index.md` — status-061追加
- `docs/roadmap.md` — HEX8ファミリ拡充記述更新
- `README.md` — 現在状態更新

## 設計上の懸念・TODO

- [ ] Stage S2〜S4: シース挙動モデルの後続ステージ
- [ ] アセンブリへの HEX8 3バリアント統合（現在はスタンドアロンカーネル）
- [ ] C3D8R のアワーグラス制御について、alpha_hg のより精密なチューニング指針
- [ ] 7本撚りブロック分解ソルバー

---
