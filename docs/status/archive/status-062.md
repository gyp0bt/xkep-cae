# status-062: HEX8 アセンブリ統合 + B-bar 平均膨張法 + SRI アワーグラス制御 + alpha_hg チューニング指針

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-26
**作業者**: Claude Code
**テスト数**: 1445（+46）

## 概要

status-061 の TODO を消化。HEX8 要素ファミリのアセンブリ統合テスト、B-bar 平均膨張法（Hughes 1980）の新規実装、SRI へのアワーグラス制御追加、C3D8R の alpha_hg チューニング指針テストを実施。

## 実装内容

### 1. HEX8 アセンブリ統合テスト（+6テスト）

`tests/test_protocol_assembly.py` に HEX8 要素のアセンブリテストを追加:

| テスト | 内容 |
|--------|------|
| `test_assembly_hex8_sri_single` | C3D8 単一要素（対称性、ランク12） |
| `test_assembly_hex8_incompatible_single` | C3D8I 単一要素（対称性、ランク18、6 RBM） |
| `test_assembly_hex8_reduced_single` | C3D8R 単一要素（ランク6/15、HG制御有無） |
| `test_assembly_hex8_multi_element` | C3D8I 2×1×1 メッシュ（対称性、6 RBM） |
| `test_assembly_hex8_manufactured_solution` | C3D8I 製造解テスト（K·u → ソルブ → 一致） |
| `test_assembly_hex8_cantilever` | C3D8I 片持ち梁（Timoshenko 解析解 < 5%） |

Protocol 適合テスト: `Hex8SRI`, `Hex8Incompatible`, `Hex8BBarMean`, `Hex8Reduced` を追加。

### 2. B-bar 平均膨張法 — C3D8B（+9テスト）

Hughes (1980) の B-bar 法を新規実装。2×2×2 完全積分ベースで体積部 B 行列を要素平均値で置換。

**カーネル関数**: `hex8_ke_bbar_mean(node_xyz, D)`
**クラス**: `Hex8BBarMean` (element_type="C3D8B")
**ヘルパー**: `_extract_B_vol(B)` — B 行列から体積部を射影抽出

**特性**:
- ランク 18 = 24 - 6 RBM（アワーグラスなし）
- 体積ロッキング回避（ν→0.5 でも安定）
- せん断ロッキングは残る（完全積分ベース）
- パッチテスト合格（一様ひずみ正確再現）

テスト:
| テスト | 内容 |
|--------|------|
| test_symmetry | 対称性 |
| test_rank | ランク 18 |
| test_psd | 正半定値 |
| test_rbm | 6 RBM |
| test_protocol | ElementProtocol 適合 |
| test_patch_test | パッチテスト（一様引張） |
| test_b_vol_projection | 体積射影の正確性 |
| test_cantilever_bending_coarse | 片持ち梁（せん断ロッキングの確認） |
| test_volume_locking_resistance | ν=0.499 でランク 18 維持 |

### 3. SRI アワーグラス制御（+7テスト）

`hex8_ke_sri()` に `alpha_hg` パラメータを追加。偏差 1 点積分の 6 個のゼロエネルギーモードを Flanagan-Belytschko で安定化。

**変更点**:
- `hex8_ke_sri(node_xyz, D, *, alpha_hg=0.0)` — alpha_hg 追加
- `Hex8SRI(alpha_hg=0.0)` — コンストラクタ引数追加（後方互換: デフォルト 0.0）
- 偏差成分の代表剛性 D_dev_max を使用してアワーグラス人工剛性を算出

テスト:
| テスト | 内容 |
|--------|------|
| test_alpha_hg_default_zero | デフォルト 0.0 |
| test_rank_without_hg | alpha_hg=0: ランク 12 |
| test_rank_with_hg | alpha_hg=0.03: ランク > 12 |
| test_symmetry_with_hg | HG 制御付き対称性 |
| test_psd_with_hg | HG 制御付き正半定値 |
| test_class_alpha_hg | クラス経由 HG 制御 |
| test_cantilever_improvement | HG 制御で曲げ改善 |

### 4. C3D8R alpha_hg チューニング指針（+4テスト）

C3D8R のアワーグラス制御パラメータの最適範囲を検証テストで文書化。

**発見**:
- alpha_hg=0: アワーグラスモードにより曲げ不安定（誤差 > 50%）
- **alpha_hg=0.03 が最適**（誤差約 3%）
- alpha_hg=0.05: やや過剛性（誤差約 10%）
- alpha_hg=0.50: 過大で過剛性

テスト:
| テスト | 内容 |
|--------|------|
| test_alpha_0_no_bending_stiffness | alpha=0 で大誤差 |
| test_recommended_range | alpha=0.03 で誤差 < 10% |
| test_optimal_alpha_around_003 | 0.03 が最適（0.01 < 0.03 > 0.10） |
| test_excessive_alpha_overstiffens | 過大 alpha で過剛性 |

### 5. HEX8 メッシュ生成ヘルパー

`_make_beam_hex8_mesh(nx, ny, nz, Lx, Ly, Lz)` — nx×ny×nz の HEX8 構造格子メッシュ生成ユーティリティ。アセンブリテストとアプリケーションテストの基盤。

## ファイル変更

### 新規
- `docs/status/status-062.md`

### 変更
- `xkep_cae/elements/hex8.py` — B-bar 平均膨張法追加、SRI アワーグラス制御追加、クラス更新
- `tests/test_hex8.py` — +36テスト（B-bar 9、SRI_HG 7、alpha_hg 4、アセンブリヘルパー含む16追加）
- `tests/test_protocol_assembly.py` — +6テスト（HEX8 アセンブリ統合）+ Protocol 適合テスト更新
- `docs/status/status-index.md` — status-062 追加
- `docs/roadmap.md` — HEX8 B-bar/HG 記述更新
- `README.md` — 現在状態更新

## HEX8 要素ファミリ一覧（更新後）

| 要素名 | クラス | 積分方式 | ランク | 特徴 |
|--------|--------|---------|--------|------|
| **C3D8** | `Hex8SRI` | SRI + HG制御 | 12 (HG なし) / 18+ (HG あり) | せん断ロッキング回避、HG制御で安定化 |
| **C3D8B** | `Hex8BBarMean` | B-bar 平均膨張 | 18 | 体積ロッキング回避、非圧縮安定 |
| **C3D8R** | `Hex8Reduced` | 1点 + HG制御 | 6 / 15 | 高速、陽解法向け |
| **C3D8I** | `Hex8Incompatible` | 非適合モード | 18 | 曲げ最高精度 |

## 設計上の懸念・TODO

- [ ] Stage S2〜S4: シース挙動モデルの後続ステージ
- [ ] 7本撚りブロック分解ソルバー
- [ ] 撚撚線（7本撚線＋被膜の7撚線）統合解析テスト

---
