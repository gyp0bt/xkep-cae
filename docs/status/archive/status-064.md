# status-064: Stage S2 シース内面Fourier近似 + 接触接線モード（contact_tangent_mode）実装

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-26
**作業者**: Claude Code
**テスト数**: 1516（+38）

## 概要

status-063 の TODO 3項を消化。

1. **Stage S2**: シース内面形状の Fourier 近似 + 膜厚分布考慮の修正コンプライアンス行列（+34テスト）
2. **接触接線モード（contact_tangent_mode）**: 7本撚りブロック分解ソルバーの基盤として、Inner NR の接触剛性組込み方式を選択可能にするインフラ実装（+4テスト）
3. **撚撚線チェックボックス**: roadmap.md で既に [x] であることを確認（変更なし）

## 実装内容

### 1. Stage S2: 内面形状 Fourier + 修正コンプライアンス（+34テスト）

シース内面プロファイル `r_inner(θ)` を素線配置から計算し、Fourier 分解で平滑化。膜厚分布 `t(θ) = r_outer - r_inner(θ)` を考慮した修正コンプライアンス行列を構築。

**新規関数（xkep_cae/mesh/twisted_wire.py）**:

| 関数 | 概要 |
|------|------|
| `compute_inner_surface_profile()` | 素線配置からレイ-円交差で内面形状 r_inner(θ) を計算 |
| `sheath_compliance_matrix()` | 撚線+シースからS2修正コンプライアンス行列を構築（高レベルAPI） |

**新規関数（xkep_cae/mesh/ring_compliance.py）**:

| 関数 | 概要 |
|------|------|
| `fourier_decompose_profile()` | DFTベースの Fourier 係数抽出（a_n, b_n） |
| `evaluate_fourier_profile()` | Fourier 係数からプロファイル再構築 |
| `build_variable_thickness_compliance_matrix()` | 膜厚分布考慮のコンプライアンス行列（有効内径 a_eff = (a_i+a_j)/2 のGreen関数） |

**テスト（tests/mesh/test_ring_compliance_s2.py, 34テスト）**:

| クラス | テスト数 | 主な検証 |
|--------|---------|---------|
| TestInnerSurfaceProfile | 8 | プロファイル形状、6/3/12倍対称、被膜効果、エンベロープ一致 |
| TestFourierDecomposition | 9 | 定数/余弦/正弦入力、ラウンドトリップ、支配モード（7本→a₆, 19本→a₁₂）、収束性 |
| TestVariableThicknessCompliance | 9 | 均一厚でS1一致、対称性、正定値、非巡回性、薄→高コンプライアンス |
| TestSheathComplianceMatrix | 8 | 7/19/3本撚りシースの形状・対称・正定値・被膜/厚さ/クリアランス効果 |

**技術的知見**:
- 浮動小数点境界処理: `abs(d_perp) ≤ r_w + tol` (tol = r_w × 1e-10) + `d_perp_clamped = min(abs(d_perp), r_w)` でsqrt負引数を防止
- 小スケール行列テストで `np.allclose` のデフォルト atol=1e-8 が不適切 → 相対差アサーション使用

### 2. 接触接線モード（contact_tangent_mode）

7本撚りブロック分解ソルバーの基盤として、`ContactConfig` に `contact_tangent_mode` パラメータを追加。Inner NR ループの K_c 組込み方式を選択可能にした。

**新規パラメータ（ContactConfig）**:

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `contact_tangent_mode` | `"full"` | `"full"` / `"structural_only"` / `"diagonal"` / `"scaled"` |
| `contact_tangent_scale` | `1.0` | `"scaled"` モード時の α 係数 |

**各モードの動作**:

| モード | 接線行列 | 特徴 |
|--------|---------|------|
| `full` | K_T + K_c | 標準。二次収束。既存動作と完全互換 |
| `structural_only` | K_T | Uzawa型。K_c をシステム行列に含めない |
| `diagonal` | K_T + diag(K_c) | 対角近似。K_c の対角成分のみ使用 |
| `scaled` | K_T + α·K_c | スケール接触接線。α < 1 で条件数改善 |

**自動安全装置**: `structural_only` モードでは merit-based line search を自動無効化（Newton方向に接触寄与がないためmerit不整合）。

**テスト（TestContactTangentModeBasic, 4テスト）**:

| テスト | 内容 | 状態 |
|--------|------|------|
| test_full_mode_3strand_converges | 3本撚り full モード | PASS |
| test_scaled_mode_3strand_converges | 3本撚り scaled (α=1) モード | PASS |
| test_diagonal_mode_3strand_converges | 3本撚り diagonal モード | xfail |
| test_structural_only_mode_3strand_converges | 3本撚り structural_only モード | xfail |

## 7本撚りブロック分解ソルバー: diagnostic findings

### 問題の根本原因

7本撚り（1+6構成）の接触NR収束困難の原因を詳細に診断した結果:

1. **Inner NR は収束可能**: full tangent (K_T + K_c) で3-5反復で力/エネルギー収束達成
2. **Outer loop が発散**: pen_ratio = 15.5%（閾値2%超過）で、各Outer反復で merit が2倍に増大
3. **ペナルティ増大も無効**: k_pen を増大すると Inner NR の条件数が悪化し、spsolve の精度劣化で収束不可

### 試行結果

| アプローチ | 3本撚り | 7本撚り | 課題 |
|-----------|---------|---------|------|
| full (標準) | ✓ 収束 | ✗ Outer発散 | pen_ratio 15.5% → merit倍増 |
| structural_only (Uzawa) | ✗ K_T特異 | — | 接触安定化不足でK_T特異行列化 |
| diagonal | ✗ 収束遅い | — | 対角近似が不十分、NR方向が悪い |
| scaled (α<1) | ✓ (α=1) | ✗ | α<1 では full と同じ問題 |
| higher k_pen + growth | — | ✗ | K_T+K_c条件数悪化 → spsolve破綻 |

### 根本的解決策（今後の方向性）

ペナルティ法ベースの接触では、多ペア同時活性化と高ペナルティ剛性が矛盾する:
- 低 k_pen → 過大貫入 → Outer loop 発散
- 高 k_pen → 条件数悪化 → Inner NR 発散

根本解決には以下のいずれかが必要:

| 手法 | 概要 | 複雑度 |
|------|------|--------|
| **Augmented Lagrangian** | ペナルティ + 乗数更新で k_pen を低く保てる | 中 |
| **Mortar discretization** | 接触界面の適合離散化で安定性向上 | 高 |
| **反復ソルバー + 前処理** | ILU/AMG前処理で ill-conditioned K を直接扱う | 中 |
| **接触ペア間引き** | 代表ペアのみ活性化（n_active を制限） | 低 |

→ 今回は `contact_tangent_mode` インフラと diagnostic findings をコミット。根本解決は次回以降の課題。

## ファイル変更

### 新規
- `tests/mesh/test_ring_compliance_s2.py` — Stage S2 テスト（34テスト）
- `docs/status/status-064.md`

### 変更
- `xkep_cae/mesh/ring_compliance.py` — `fourier_decompose_profile`, `evaluate_fourier_profile`, `build_variable_thickness_compliance_matrix` 追加
- `xkep_cae/mesh/twisted_wire.py` — `compute_inner_surface_profile`, `sheath_compliance_matrix` 追加
- `xkep_cae/mesh/__init__.py` — 新規関数のエクスポート追加
- `xkep_cae/contact/pair.py` — `ContactConfig` に `contact_tangent_mode`, `contact_tangent_scale` 追加
- `xkep_cae/contact/solver_hooks.py` — Inner NR で `contact_tangent_mode` に基づく K_c 条件分岐
- `tests/contact/test_twisted_wire_contact.py` — `_solve_twisted_wire` / `_make_contact_manager` に mode パラメータ追加、`TestContactTangentModeBasic` 追加（4テスト）
- `docs/status/status-index.md` — status-064 追加
- `docs/roadmap.md` — Stage S2 完了、ブロック分解ソルバー状況更新
- `README.md` — 現在状態更新

## 設計上の懸念・TODO

- [ ] Stage S3: シース-素線/被膜 有限滑り（接触位置θの変形追従、C行列再配置）
- [ ] Stage S4: シース-シース接触（円-円ペナルティ接触、既存ContactPair流用）
- [ ] 7本撚り収束: Augmented Lagrangian 乗数更新の改善 or 接触ペア間引き
- [ ] 反復ソルバー + ILU前処理による ill-conditioned system の直接対処

---
