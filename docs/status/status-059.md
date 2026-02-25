# status-059: HEX8 B-bar 要素 + シース曲げモードバリデーション

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-25
**作業者**: Claude Code
**テスト数**: 1391（+45）

## 概要

status-058 の TODO「シースモデルのテストバリデーションに曲げモードチェックを追加」を実施。現在の平面ひずみベース（z 軸断面内の径方向圧縮/引張/せん断）に加え、長手方向の圧縮/引張/横せん断/曲げモードの検証を追加。

せん断ロッキング回避のため、B-bar 法付き HEX8 レンガ要素を新規実装し、3D チューブメッシュで円筒管を離散化して梁理論の解析解と比較するバリデーションテストを構築。

## 実装内容

### `xkep_cae/elements/hex8.py`（新規）

8 節点 6 面体（レンガ）要素。B-bar 法で体積ロッキングを回避。

| 関数/クラス | 内容 |
|------------|------|
| `_hex8_shape(xi, eta, zeta)` | トリリニア形状関数 N_i (i=0..7) |
| `_hex8_dNdxi(xi, eta, zeta)` | 自然座標微分 dN/dξ, dN/dη, dN/dζ |
| `_build_B(dN_dx)` | 標準 B 行列 (6×24) 構築 |
| `hex8_ke_bbar(node_xyz, D)` | B-bar 法付き HEX8 要素剛性行列 |
| `Hex8BBar` | ElementProtocol 適合クラス (3 DOF/node, 8 nodes, 24 DOF) |

#### B-bar 法の定式化

- 体積ひずみ (εxx + εyy + εzz) を要素平均に置換 → 体積ロッキング回避
- 偏差成分: 2×2×2 フル積分（8 ガウス点）
- 体積成分: 要素平均（detJ 重み付き） → 選択低減積分効果

### `xkep_cae/materials/elastic.py`（拡張）

| 関数/クラス | 内容 |
|------------|------|
| `constitutive_3d(E, nu)` | 3D 等方弾性テンソル D (6×6) — Voigt 表記 |
| `IsotropicElastic3D` | 3D 等方線形弾性構成則 (ConstitutiveProtocol 適合) |

### `xkep_cae/mesh/tube_mesh.py`（新規）

HEX8 要素による円筒管メッシュジェネレータ。

| 関数 | 内容 |
|------|------|
| `make_tube_mesh(r_inner, r_outer, length, n_r, n_theta, n_z)` | 構造格子チューブメッシュ生成 |
| `tube_face_nodes(nodes, face, ...)` | 面別節点インデックス取得 (z0/zL/inner/outer) |

### テスト

#### `tests/test_hex8.py`（新規、25 テスト）

| クラス | テスト数 | 内容 |
|--------|----------|------|
| `TestShapeFunctions` | 4 | 分配性、節点値、微分形状、和ゼロ |
| `TestStiffnessBasic` | 7 | 対称性、ランク18、剛体モード6、正半定値、スケーリング、歪み要素 |
| `TestPatchTest` | 2 | 一様ひずみパッチテスト（2×2×2 要素）、定ひずみ単一要素 |
| `TestUniaxialCompression` | 2 | 軸変位 δ=FL/(EA)、ポアソン収縮 Δa/a=-ν·ε |
| `TestElastic3D` | 6 | 3D 弾性テンソル: 形状、対称性、正定値、一軸ひずみ応力、せん断、クラスIF |
| `TestHex8BBarProtocol` | 4 | ElementProtocol 属性、local_stiffness、dof_indices、入力バリデーション |

#### `tests/mesh/test_sheath_bending_validation.py`（新規、20 テスト）

| クラス | テスト数 | 内容 |
|--------|----------|------|
| `TestTubeMesh` | 8 | 節点数、要素数、径範囲、z 範囲、面節点、ヤコビアン正、入力エラー |
| `TestAxialStiffness` | 3 | 軸引張 δ=FL/(EA) <5%、軸圧縮、肉厚スケーリング |
| `TestBendingStiffness` | 3 | 片持ち曲げ δ=PL³/(3EI)+Timo補正 <10%、純曲げモーメント <15%、肉厚依存性 |
| `TestTransverseShear` | 2 | 短い片持ち梁（せん断支配）Timo解 <15%、せん断ロッキング確認 |
| `TestSheathStiffnessConsistency` | 4 | EA/EI/GJ 整合性（SheathModel vs 解析解）、FEM vs シースモデル EI <15% |

## 検証結果

| 項目 | 結果 |
|------|------|
| HEX8 パッチテスト（一様ひずみ） | 機械精度で一致 (rtol < 1e-10) |
| 一軸引張 FEM vs 解析 δ=FL/(EA) | < 1e-6 (0.0001%) |
| ポアソン収縮 FEM vs 解析 | < 1e-6 |
| 片持ち曲げ FEM vs Timoshenko 解 | < 10% |
| 純曲げモーメント FEM vs 解析 | < 15% |
| 短管横せん断 FEM vs Timoshenko 解 | < 15% |
| SheathModel EA/EI/GJ vs 解析解 | 機械精度一致 |
| FEM 曲げたわみ vs シースモデル EI | < 15% |
| B-bar せん断ロッキング回避 | 粗メッシュでも >50% of EB解 |

## ファイル変更

### 新規
- `xkep_cae/elements/hex8.py` — HEX8 B-bar 要素
- `xkep_cae/mesh/tube_mesh.py` — 3D チューブメッシュジェネレータ
- `tests/test_hex8.py` — HEX8 要素テスト（25件）
- `tests/mesh/test_sheath_bending_validation.py` — シース曲げモードバリデーション（20件）

### 変更
- `xkep_cae/materials/elastic.py` — 3D 弾性テンソル + IsotropicElastic3D 追加
- `docs/status/status-index.md` — status-059 行追加
- `docs/roadmap.md` — HEX8 要素 + 曲げモードバリデーション追記
- `README.md` — 現在状態更新

## TODO

### 次ステップ（実装順）

- [ ] **Stage S2**: 膜厚分布 t(θ) の Fourier 近似 — 素線配置からの内面形状計算、Fourier 係数 aₙ/bₙ 抽出、修正コンプライアンス行列
- [ ] **Stage S3**: シース-素線/被膜 有限滑り — 接触位置 θ_contact の変形追従、C 行列の接触点列再配置、既存 friction_return_mapping 統合
- [ ] **Stage S4**: シース-シース接触 — 円形外面同士のペナルティ接触（既存 ContactPair/梁-梁接触フレームワーク流用）
- [ ] 撚撚線（7本撚線＋被膜の7撚線）: 被膜込み接触半径・摩擦・断面剛性を用いた統合解析テスト
- [ ] 7本撚りブロック分解ソルバー（Schur補完法 or Uzawa法）
- [ ] HEX8 要素のアセンブリ統合（assembly.py への組み込み） — 現在はテスト内でのみ使用

### 確認事項

- HEX8 要素は現時点ではテスト内での直接組立のみ使用。本体のアセンブリモジュール（`assembly.py`）は 2D 要素前提なので、3D 対応は必要に応じて追加。
- 曲げモードバリデーションの許容誤差（10-15%）は、3D FEM と 1D 梁理論の本質的差異（断面変形、ポアソン効果、端部拘束効果）による。メッシュ細分化で改善可能。

---
