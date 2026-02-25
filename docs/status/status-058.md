# status-058: Stage S1 — 解析的リングコンプライアンス行列実装

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-25
**作業者**: Claude Code
**テスト数**: 1346（+35）

## 概要

status-057 で策定したシース挙動設計の Stage S1 を実装。厚肉弾性リングの Fourier モード別コンプライアンスの閉形式解 + N×N コンプライアンス行列構築 + FEM リング解との比較テスト。

## 実装内容

### `xkep_cae/mesh/ring_compliance.py`（新規）

厚肉弾性リングのコンプライアンス行列を解析的に構築するモジュール。

| 関数 | 内容 |
|------|------|
| `ring_mode0_compliance(a, b, E, nu, plane)` | モード 0（均等内圧）— Lamé 解 |
| `_build_michell_system(beta, n)` | Michell 解の正規化 4×4 連立方程式（数値安定性のため α₁=Aa^{n-2} 等で正規化） |
| `ring_mode_n_compliance(a, b, E, nu, n, plane)` | モード n≥2 — Michell 応力関数解（変位公式: 2G·U_r = a·[-n·α₁ + n·α₂ + (κ-n-1)·α₃ + (κ+n-1)·α₄]） |
| `build_ring_compliance_matrix(N, a, b, E, nu, *, n_modes, plane)` | N×N コンプライアンス行列（Green 関数から循環行列として構築） |
| `ring_compliance_summary(N, a, b, E, nu, *, n_modes, plane)` | 行列 + 固有値 + 条件数 + モード別コンプライアンスの診断情報 |

### 理論的背景

- **モード 0**: Lamé 解（厚肉円筒の均等内圧問題）
  - 平面ひずみ: `c₀ = a(1+ν)/[E(b²-a²)] · [(1-2ν)a² + b²]`
  - 平面応力: `c₀ = a/[E(b²-a²)] · [(1-ν)a² + (1+ν)b²]`
- **モード n≥2**: Michell 応力関数解
  - `φ_n = (Ar^n + Br^{-n} + Cr^{n+2} + Dr^{-n+2}) cos(nθ)`
  - 4 つの境界条件（σ_r, τ_rθ at r=a,b）から係数決定
  - 正規化座標系で大きな n でも数値安定
- **Green 関数**: `G(θ) = c₀/(2πa) + (1/(πa)) Σ_{n≥2} cₙ cos(nθ)`
- **コンプライアンス行列**: `C_ij = G(θ_i - θ_j)` — N×N 対称正定値循環行列

### テスト（35件）

| クラス | テスト数 | 内容 |
|--------|----------|------|
| `TestMode0Compliance` | 8 | Lamé 解析解一致、厚肉効果、パラメータ依存性、入力バリデーション |
| `TestModeNCompliance` | 9 | 正値性、モード次数依存、平面ひずみ/応力差異、Michell 系のランク、境界条件充足、高次モード安定性 |
| `TestComplianceMatrix` | 10 | 対称性、正定値性、循環行列性、対角優位、均等荷重→モード0応答、Fourier 収束 |
| `TestThinRingLimit` | 3 | モード0薄肉極限（c₀≈R²/(Et)）、モードn非伸張極限（cₙ≈R⁴/(EI(n²-1)²)）、収束性確認 |
| `TestFEMComparison` | 2 | Q4要素FEMリング解との比較（均等内圧: <1%誤差、N点集中荷重: <5%誤差） |
| `TestComplianceSummary` | 3 | 診断情報の返却、固有値正値、条件数有限 |

### FEM 比較テストの実装

テスト内に独立した2D Q4要素FEMリング解メッシュ生成・組立・求解を実装:
- `_make_ring_mesh()`: 全周Q4メッシュ生成（nr_elem×ntheta_elem 分割）
- `_assemble_ring()`: 平面ひずみ弾性体の全体剛性行列組立（2×2 Gauss 積分）
- 境界条件: 接線方向拘束（θ≈0 で u_y=0, θ≈π/2 で u_x=0）— 径方向膨張を妨げない

### その他の変更

- `xkep_cae/mesh/__init__.py` — ring_compliance 関数のインポート・エクスポート追加
- `docs/status/status-055.md`, `status-056.md`, `status-057.md`, `docs/roadmap.md` — 誤字修正（撚線線→撚撚線）

## 検証結果

| 項目 | 結果 |
|------|------|
| 均等内圧 FEM vs 解析解 | 相対誤差 < 1% |
| N点集中荷重 FEM vs 行列解 | 平均相対誤差 < 5% |
| 薄肉極限（t/R→0）収束 | 薄くなるほど誤差減少を確認 |
| エネルギー法による独立検証 | 平面ひずみで変位公式と完全一致（ratio=1.000000） |
| 行列の対称性 | `‖C - Cᵀ‖ < 1e-15 * ‖C‖` |
| 行列の正定値性 | 全固有値 > 0 |
| 循環行列性 | `C[i,j] = f((j-i) mod N)` |

## ファイル変更

### 新規
- `xkep_cae/mesh/ring_compliance.py` — 解析的リングコンプライアンス行列モジュール
- `tests/mesh/test_ring_compliance.py` — 35件のテスト

### 変更
- `xkep_cae/mesh/__init__.py` — ring_compliance 関数のインポート・エクスポート追加
- `docs/roadmap.md` — S1 チェック + 誤字修正（撚線線→撚撚線）
- `docs/status/status-055.md` — 誤字修正（撚線線→撚撚線）
- `docs/status/status-056.md` — 誤字修正（撚線線→撚撚線）
- `docs/status/status-057.md` — 誤字修正（撚線線→撚撚線）
- `docs/status/status-index.md` — status-058 行追加
- `README.md` — Stage S1 実装完了の記載追加

## TODO

### 次ステップ（実装順）

- [ ] **Stage S2**: 膜厚分布 t(θ) の Fourier 近似 — 素線配置からの内面形状計算、Fourier 係数 aₙ/bₙ 抽出、修正コンプライアンス行列
- [ ] **Stage S3**: シース-素線/被膜 有限滑り — 接触位置 θ_contact の変形追従、C 行列の接触点列再配置、既存 friction_return_mapping 統合
- [ ] **Stage S4**: シース-シース接触 — 円形外面同士のペナルティ接触（既存 ContactPair/梁-梁接触フレームワーク流用）
- [ ] 撚撚線（7本撚線＋被膜の7撚線）: 被膜込み接触半径・摩擦・断面剛性を用いた統合解析テスト
- [ ] 7本撚りブロック分解ソルバー（Schur補完法 or Uzawa法）

---
