# 数理台帳（docs/math/）

[← README](../../README.md) | [← roadmap](../roadmap.md) | [← MCDD 設計仕様](../../xkep_cae/mathematics/docs/mathematics.md)

## 概要

本ディレクトリは MCDD（数理契約駆動開発、status-346〜）における**離散化方程式
の単一のソース・オブ・トゥルース**として運用される。

`MathematicalContract.equation_ref`（`xkep_cae/mathematics/contracts.py`）から
`<ファイル名>#<アンカー>` 形式で参照される。アンカーは `## <h2>` 直前に
`<a id="eq-..."></a>` の形で明示的に貼り、status-349 で予定の `equation_index.py`
が機械抽出可能な構造を維持する。

## 章立て

| # | ファイル | 内容 | 状態 |
|---|---------|------|------|
| 01 | [`01_kinematics_beam.md`](01_kinematics_beam.md) | 梁要素の運動学（CR / TL / UL、Hermite） | ✅ status-349 |
| 02 | [`02_contact_geometry.md`](02_contact_geometry.md) | 接触ペア構築・最近接点（s, t）射影 | ✅ status-349 |
| **03** | [`03_huber_contact_penalty.md`](03_huber_contact_penalty.md) | **Huber 法線ペナルティ + Hertz 非線形 + K_c 項展開** | ✅ status-348 |
| 04 | [`04_friction_smooth_penalty.md`](04_friction_smooth_penalty.md) | Coulomb 摩擦 return mapping + K_t 接線剛性 | ✅ status-349 |
| 05 | [`05_coating_barrier.md`](05_coating_barrier.md) | バリア関数被膜モデル | ✅ status-349 |
| 06 | [`06_time_integration.md`](06_time_integration.md) | 動的解析 (Generalized-α / Newmark) と疑似時間 | ✅ status-349 |

## アンカー命名規約

- `eq-<topic>` : 方程式そのもの（例 `eq-kc-full-decomposition`、`eq-pn-huber`）
- `eq-<topic>-fd` : 同方程式の FD 整合性版（`FDConsistencyContract` 用）
- `inv-<topic>` : 不変量・不等式（`InequalityContract` 用、例 `inv-pn-nonneg`）
- `sym-<topic>` : 対称性命題（`SymmetryContract` 用、例 `sym-kmat`）

## 整合性ルール（status-348 で確立）

1. アンカーはファイル全体で**一意**。重複は status-349 で新設した `equation_index.py`
   + C15 拡張が機械検出する（6 章 55 アンカー、重複ゼロ確認済）。
2. 数式は TeX 文字列で記述（GitHub Flavored Markdown の `$$ ... $$` ブロック）。
   sympy 等のシンボリック処理は導入しない（MCDD 設計方針 `mathematics.md`）。
3. 各方程式の右に「→ 実装: `<file>:<func>`」の trace を併記し、
   後述 Phase C / D で項別 Process が実装された際に追記する。
4. **数値は表示しない**: 報告書ではなく台帳。tol/誤差/実測値は status ファイル
   側に置く。台帳は「式そのもの」と「項の網羅性」のみ責任を持つ。

## 関連

- 設計仕様: [`xkep_cae/mathematics/docs/mathematics.md`](../../xkep_cae/mathematics/docs/mathematics.md)
- 計画書: `/root/.claude/plans/deep-wiggling-seal.md` は **永久ロスト**
  （status-352 記録）。以降 `../../CLAUDE.md` と `../roadmap.md` を正とする
- 進捗 status: [status-346](../status/status-346.md) / [status-347](../status/status-347.md) / [status-348](../status/status-348.md) / [status-349](../status/status-349.md)
