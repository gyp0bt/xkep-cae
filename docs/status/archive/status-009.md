# status-009: Phase 2.3/2.4 — 3D Timoshenko梁 & 断面モデル拡張 & SCF & パーサー拡張

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-008](./status-008.md)

**日付**: 2026-02-13
**作業者**: Claude Code
**ブランチ**: `claude/execute-status-todos-cP5vZ`

---

## 実施内容

status-008 の短期TODO 4項目を全て実装・テスト完了。

### 1. Phase 2.3: Timoshenko梁（3D空間）の実装

3D空間の Timoshenko 梁要素を新規実装。

#### 仕様

| 項目 | 値 |
|------|------|
| ファイル | `xkep_cae/elements/beam_timo3d.py` |
| DOF/node | 6 (ux, uy, uz, θx, θy, θz) |
| DOF/element | 12 |
| 節点数 | 2 |
| 剛性行列サイズ | 12×12 |

#### 定式化

- **軸方向**: EA/L
- **ねじり**: GJ/L
- **xy面曲げ** (v たわみ, θz 回転): EIz ベース, Φz = 12EIz/(κz·G·A·L²)
- **xz面曲げ** (w たわみ, θy 回転): EIy ベース, Φy = 12EIy/(κy·G·A·L²)
- **座標変換**: 12×12 ブロック対角回転行列（3×3 R を4ブロック）
- **局所座標系**: 自動選択 or ユーザー指定参照ベクトル (v_ref)

#### 符号規約

xz面曲げ（w-θy）の符号は xy面曲げ（v-θz）と反転。
右手系の整合性による: θy > 0 は w が減少する方向。
剛性行列のカップリング項 (w, θy) では v-θz とは逆符号。

#### クラス `TimoshenkoBeam3D`

```python
class TimoshenkoBeam3D:
    ndof_per_node = 6
    nnodes = 2
    ndof = 12

    def __init__(self, section: BeamSection,
                 kappa_y=5/6, kappa_z=5/6,
                 v_ref=None, scf=None)
```

- `kappa_y`, `kappa_z`: float or "cowper"（Cowper κ 対応）
- `v_ref`: 局所y軸の参照ベクトル（None で自動選択）
- `scf`: SCF パラメータ（下記参照）

### 2. Phase 2.4: 断面モデルの拡張（`BeamSection`）

3D梁に必要な断面特性を持つ `BeamSection` クラスを追加。

#### `BeamSection` データクラス

```python
@dataclass(frozen=True)
class BeamSection:
    A: float     # 断面積
    Iy: float    # y軸まわり断面二次モーメント
    Iz: float    # z軸まわり断面二次モーメント
    J: float     # ねじり定数（St. Venant）
    shape: str   # "rectangle", "circle", "general"
```

#### ファクトリメソッド

| メソッド | 引数 | 自動計算 |
|---------|------|---------|
| `rectangle(b, h)` | 幅, 高さ | A, Iy, Iz, J（St. Venant 近似） |
| `circle(d)` | 直径 | A, Iy=Iz, J=2I |
| `pipe(d_outer, d_inner)` | 外径, 内径 | A, Iy=Iz, J=2I |

#### その他のメソッド

- `cowper_kappa_y(nu)`, `cowper_kappa_z(nu)`: 方向別 Cowper κ
- `to_2d()`: `BeamSection2D` への変換

### 3. SCF（スレンダネス補償係数）のオプション実装

2D/3D 両方の Timoshenko 梁に SCF パラメータを追加。

#### 数式

```
f_p = 1 / (1 + SCF · L²A/(12I))
Φ_eff = Φ · f_p
```

- 太い梁 (L²A/(12I) ≈ 1): Φ_eff ≈ Φ（通常の Timoshenko）
- 細い梁 (L²A/(12I) >> 1): Φ_eff → 0（EB 梁に遷移）

#### API

```python
# 2D
beam = TimoshenkoBeam2D(section=sec, scf=0.25)

# 3D
beam = TimoshenkoBeam3D(section=sec, scf=0.25)
```

#### Abaqusとの関係

xkep-cae の SCF はせん断パラメータ Φ を直接低減する。
Abaqus の SCF はペナルティ法の横せん断剛性を制限する。
物理的効果は同等（細長い梁で EB 挙動に遷移）だが、
内部の適用メカニズムが異なるため、数値的な完全一致は保証されない。

### 4. Abaqus .inp パーサーへの `*BEAM SECTION` / `*TRANSVERSE SHEAR STIFFNESS` サポート追加

#### 新規対応キーワード

| キーワード | 内容 |
|-----------|------|
| `*BEAM SECTION` | 断面タイプ, ELSET, MATERIAL, 寸法, 方向ベクトル |
| `*TRANSVERSE SHEAR STIFFNESS` | K11, K22, K12（横せん断剛性） |

#### 新規データクラス `AbaqusBeamSection`

```python
@dataclass
class AbaqusBeamSection:
    section_type: str       # "RECT", "CIRC", etc.
    elset: str
    material: str
    dimensions: list[float]
    direction: list[float] | None
    transverse_shear: tuple[float, float, float] | None  # (K11, K22, K12)
```

#### `AbaqusMesh` の拡張

`beam_sections: list[AbaqusBeamSection]` フィールドを追加。

---

## テスト結果

**161 passed, 2 deselected (external)**（46テスト増加）

### 新規テスト

| テストファイル | テスト数 | 内容 |
|--------------|---------|------|
| `test_beam_timo3d.py` | 33 | 3D梁の全検証（対称性, 剛体モード, 軸引張, ねじり, 二軸曲げ, 傾斜梁, SCF, Cowper, 分布荷重, 断面特性, Protocol適合） |
| `test_beam_timo2d.py` (追加) | 6 | 2D梁のSCFテスト |
| `test_abaqus_inp.py` (追加) | 7 | `*BEAM SECTION` / `*TRANSVERSE SHEAR STIFFNESS` パーステスト |

### 解析解検証

| 問題 | 解析解 | FEM結果 | 相対誤差 |
|------|--------|---------|---------|
| 3D片持ち梁 軸引張 (δ = PL/EA) | 厳密 | 一致 | < 1e-10 |
| 3D片持ち梁 ねじり (θ = TL/GJ) | 厳密 | 一致 | < 1e-10 |
| 3D片持ち梁 y方向曲げ (Timoshenko) | 厳密 | 一致 | < 1e-10 |
| 3D片持ち梁 z方向曲げ (Timoshenko) | 厳密 | 一致 | < 1e-10 |
| 3D-2D 整合性（xy面内曲げ） | 2D解と一致 | 一致 | < 1e-10 |
| 非対称矩形断面 (b≠h) 二軸曲げ | 厳密 | 一致 | < 1e-10 |
| Cowper κ 統合テスト | 厳密 | 一致 | < 1e-10 |
| 3D等分布荷重 | 厳密 | 一致 | < 1e-6 |

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/elements/beam_timo3d.py` | **新規** — 3D Timoshenko 梁要素 |
| `xkep_cae/elements/beam_timo2d.py` | 更新 — SCF パラメータ追加 |
| `xkep_cae/sections/beam.py` | 更新 — `BeamSection` クラス追加 |
| `xkep_cae/io/abaqus_inp.py` | 更新 — `*BEAM SECTION` / `*TRANSVERSE SHEAR STIFFNESS` パーサー |
| `tests/test_beam_timo3d.py` | **新規** — 3D梁テスト (33テスト) |
| `tests/test_beam_timo2d.py` | 更新 — SCFテスト追加 (6テスト) |
| `tests/test_abaqus_inp.py` | 更新 — パーサーテスト追加 (7テスト) |
| `docs/status/status-009.md` | **新規** — 本ステータス |
| `docs/roadmap.md` | 更新 — Phase 2.3/2.4 チェックボックス更新 |
| `docs/abaqus-differences.md` | 更新 — 3D梁・SCF情報追加 |
| `README.md` | 更新 — 現在の状態・リンク更新 |

---

## 設計上の考慮事項

### 1. 3D梁の局所座標系自動選択

梁軸に最も直交する座標軸を参照ベクトルとして自動選択。
任意方向の梁でも追加入力なしで動作するが、
意図しない局所座標系の向きになる可能性がある。
ユーザーは `v_ref` で明示指定可能。

### 2. xz面曲げの符号規約

右手系の整合性のため、θy と w のカップリング項の符号が
θz と v のカップリングとは反転する。
これは標準的な定式化（Przemieniecki, Bathe 等）と一致。

### 3. SCF の解釈

xkep-cae の解析的（厳密）定式化では、Abaqus のペナルティ法における
せん断ロッキング問題は存在しない。SCF は Abaqus 比較のための
オプション機能として実装。物理的には Φ の低減（EB 遷移）として機能。

### 4. BeamSection の St. Venant ねじり定数

矩形断面の J は近似公式を使用:
```
J ≈ a·b³ · (1/3 - 0.21·(b/a) · (1 - (b/a)⁴/12))
```
正方形断面で J = 0.1406·a⁴（厳密値 ≈ 0.1406a⁴）。
円形・パイプ断面は厳密解（J = πd⁴/32）。

---

## TODO（次回以降の作業）

### 短期（Phase 2 残り）

- [ ] 3D梁のアセンブリレベルテスト（`test_protocol_assembly.py` に追加）
- [ ] ワーピング（オプション、薄肉断面用）の検討
- [ ] 3D梁の応力・内力ポスト処理の基盤設計

### 中期（Phase 2.5 / Phase 3）

- [ ] Phase 2.5: Cosserat rod の設計仕様書作成
- [ ] SO(3) 回転パラメトライゼーションの選定と実装
- [ ] Cosserat rod の線形化バージョン（テスト用）
- [ ] Phase 3: Newton-Raphson ソルバーフレームワーク

### 長期（Phase 4.6 撚線モデル）

- [ ] θ_i 縮約の具体的手法検討（曲げ主目的に最適化）
- [ ] δ_ij 追跡＋サイクルカウント基盤の設計
- [ ] Level 0: 軸方向素線＋penalty接触＋正則化Coulomb＋δ_ij追跡
- [ ] Level 1: θ_i 未知量化＋被膜弾性ばね（G_c, K_c）
- [ ] Level 2: 素線Cosserat rod化＋接触ペア動的更新

### 残存する不確定事項

- [ ] 接触ペア更新頻度 N_update の適切な値（数値実験で決定）
- [ ] 接触活性化閾値 g_threshold の設定方針
- [ ] 大変形時のペア更新に伴う力の不連続の許容度
- [ ] 疲労評価のサイクルカウント手法（雨流計数法？他？）
