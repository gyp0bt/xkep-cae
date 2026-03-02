# status-063: C3D8 SRI+B-bar 併用デフォルト化 + 撚撚線（被膜付き撚線）統合解析テスト

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-26
**作業者**: Claude Code
**テスト数**: 1478（+33）

## 概要

status-062 の TODO を消化。C3D8 要素の SRI+B-bar 併用をデフォルトに変更（後方互換維持）し、被膜付き撚線の統合解析テストを追加。

## 実装内容

### 1. C3D8 SRI+B-bar 併用カーネル（+13テスト）

偏差=1点SRI（せん断ロッキング回避）+ 体積=B-bar平均膨張法（体積ロッキング回避）+ Flanagan-Belytschko アワーグラス制御を組み合わせた新カーネル `hex8_ke_sri_bbar()` を実装。

**新規カーネル**: `hex8_ke_sri_bbar(node_xyz, D, *, alpha_hg=0.03)`

| フェーズ | 処理 | 目的 |
|---------|------|------|
| Phase 1 | B̄_vol を 2×2×2 積分で平均化 | 体積ロッキング回避 |
| Phase 2 | K_vol = B̄_vol^T D_vol B̄_vol * V_total | 体積剛性 |
| Phase 3 | K_dev = B0_dev^T D_dev B0_dev * V（1点積分） | せん断ロッキング回避 |
| Phase 4 | K_hg = Flanagan-Belytschko HG 制御 | ゼロエネルギーモード安定化 |

**クラス変更**: `Hex8SRI` のデフォルトを変更

```python
# Before
Hex8SRI(alpha_hg=0.0)  # SRI のみ

# After
Hex8SRI(alpha_hg=0.03, *, mode="sri_bbar")  # SRI+B-bar 併用
Hex8SRI(alpha_hg=0.0, mode="sri_only")      # 旧 SRI（後方互換）
```

テスト（`TestSRIBBar`）:

| テスト | 内容 |
|--------|------|
| test_symmetry | 対称性 |
| test_psd | 正半定値 |
| test_rank_without_hg | alpha_hg=0: ランク 12 |
| test_rank_with_default_hg | alpha_hg=0.03: ランク > 12 |
| test_rbm | 6 RBM |
| test_volume_locking_resistance | ν=0.499 で安定 |
| test_patch_test | パッチテスト合格 |
| test_cantilever_bending | 片持ち梁曲げ（解析解 < 5%） |
| test_better_than_bbar_for_bending | B-bar 単体より曲げ精度向上 |
| test_better_than_sri_only_for_volume | SRI 単体より体積ロッキング耐性向上 |
| test_class_default_uses_sri_bbar | デフォルト mode="sri_bbar" |
| test_class_sri_only_mode | mode="sri_only" 後方互換 |
| test_invalid_mode | 不正 mode で ValueError |

### 2. 撚撚線（被膜付き撚線）統合解析テスト（+20テスト）

被膜付きワイヤ（CoatingModel）とシース（SheathModel）の統合テストを新規作成。

**ファイル**: `tests/contact/test_coated_wire_integration.py`

**CRアセンブラ使用**: 接触収束のため CR（Corotational）アセンブラを使用（線形 Timo3D アセンブラでは収束不可）。被膜厚さに基づく初期ギャップ設定（`gap = coating.thickness * 4`）で初期貫入を回避。

#### TestCoatedBeamIntegration（4テスト）
被膜モデル単体の統合確認。

| テスト | 内容 |
|--------|------|
| test_coated_radius | 被膜込み半径の計算 |
| test_radii_array | 複数素線の半径配列 |
| test_stiffness_increase | 被膜による剛性増加 |
| test_coating_ratio | 被膜/ワイヤ面積比 |

#### TestSheathGeometryIntegration（6テスト）
シースモデルの幾何パラメータ統合確認。

| テスト | 内容 |
|--------|------|
| test_envelope_radius | エンベロープ半径 |
| test_sheath_inner_radius | シース内径 |
| test_section_properties | シース断面特性 |
| test_stiffness | シース等価剛性 |
| test_radial_gap | 径方向ギャップ |
| test_outermost_strand_count | 最外層素線数 |

#### TestCoatedThreeStrandContact（5テスト）
3本撚線の被膜付き接触解析。CR アセンブラ使用、16要素/素線。

| テスト | 内容 |
|--------|------|
| test_tension_convergence | 張力下で収束 |
| test_lateral_convergence | 横荷重下で収束 |
| test_bending_convergence | 曲げ荷重下で収束 |
| test_tension_with_friction | 摩擦付き張力 |
| test_coated_vs_bare_stiffness | 被膜あり/なし剛性比較 |

#### TestCoatedSevenStrandIntegration（5テスト）
7本撚線の被膜付きセットアップ検証。

| テスト | 内容 |
|--------|------|
| test_setup_validation | 7本撚線セットアップ検証 |
| test_gap_distribution | ギャップ分布 |
| test_contact_config | 接触設定 |
| test_seven_strand_tension | 7本撚線張力解析（xfail: 36+ペア同時収束） |
| test_seven_strand_bending | 7本撚線曲げ解析（xfail: 36+ペア同時収束） |

**備考**: 7本撚線の接触解析は依然として xfail（36+接触ペアの同時NR収束が線形アセンブラ/現行ソルバーでは困難）。ブロック分解ソルバーが根本解決策。

## ファイル変更

### 新規
- `tests/contact/test_coated_wire_integration.py` — 撚撚線統合テスト（20テスト）
- `docs/status/status-063.md`

### 変更
- `xkep_cae/elements/hex8.py` — `hex8_ke_sri_bbar()` 追加、`Hex8SRI` デフォルト変更
- `tests/test_hex8.py` — `TestSRIBBar` 追加（13テスト）、既存テスト名更新
- `tests/test_protocol_assembly.py` — ランクアサーション更新（SRI+B-bar のランク変更に対応）
- `docs/status/status-index.md` — status-063 追加
- `docs/roadmap.md` — SRI+B-bar / 撚撚線テスト記述更新
- `README.md` — 現在状態更新

## HEX8 要素ファミリ一覧（更新後）

| 要素名 | クラス | 積分方式 | ランク | 特徴 |
|--------|--------|---------|--------|------|
| **C3D8** | `Hex8SRI` | **SRI+B-bar併用** + HG制御 | 12 (HG なし) / 18+ (HG あり) | **せん断+体積ロッキング同時回避** |
| **C3D8B** | `Hex8BBarMean` | B-bar 平均膨張 | 18 | 体積ロッキング回避、非圧縮安定 |
| **C3D8R** | `Hex8Reduced` | 1点 + HG制御 | 6 / 15 | 高速、陽解法向け |
| **C3D8I** | `Hex8Incompatible` | 非適合モード | 18 | 曲げ最高精度 |

## 技術的知見

### SRI+B-bar 併用の利点
- **SRI 単体**: 偏差1点+体積2×2×2 → せん断ロッキング回避だが体積ロッキングは残る
- **B-bar 単体**: 2×2×2完全積分+B̄_vol → 体積ロッキング回避だがせん断ロッキングは残る
- **SRI+B-bar**: 偏差1点+B̄_vol → **両方同時回避**、alpha_hg=0.03 でゼロエネルギーモードも安定化

### 被膜付き接触のポイント
- 線形 Timo3D アセンブラでは接触収束不可 → CR（Corotational）アセンブラが必須
- 被膜厚さにより接触半径が増大 → gap=0 で初期貫入が発生 → 被膜厚の4倍の初期ギャップが必要
- 16要素/素線で安定収束（8要素では不十分）

## 設計上の懸念・TODO

- [ ] Stage S2〜S4: シース挙動モデルの後続ステージ
- [ ] 7本撚りブロック分解ソルバー（Schur補完法 or Uzawa法、36+ペア同時NR収束の根本解決）
- [ ] 撚撚線チェックボックスの完了マーク（roadmap.md — 3本撚線テスト成功で実質完了だが7本は xfail）

---
