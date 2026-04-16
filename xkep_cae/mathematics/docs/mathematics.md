# 数理契約駆動開発（MCDD, Mathematical Contract-Driven Development）

[← README](../../../README.md)

## 概要

Process が満たすべき「方程式」「不変量」「残差・誤差定義」を **frozen dataclass**
として宣言可能にする型システム。既存の Process 契約（C3〜C17: 依存関係・メタ
情報・コード衛生）が「コード管理面」のメタであったのに対し、本契約は
**「数理面」のメタ**を扱う。両者は責務直交。

## 背景

status-289 以来 10 status にわたり K_c の FD 不整合を追跡してきたが、
status-345 で「report の `{:5.2f}` 精度バグによる K_geo=0 誤認」が判明した。
根本原因は「数理契約が型・契約レベルで宣言されていないため、欠落項や表示精度
バグが実測でしか露呈しない」構造的問題にある。

MCDD は以下 3 点を構造的に解決する:

1. `tangent_components()` の「∂n̂/∂u の x/z 成分が抜けている」ような欠落を
   **契約違反として機械検出**
2. 診断 (TangentFD / KcComponentFD) の散在トリガーを **契約違反 → 診断
   Process の自動 dispatch** に一本化
3. 離散化方程式レジストリ (`docs/math/`) と Process の双方向リンクで
   status-289〜345 の調査文脈を永続化

## 契約型

### 階層

```
MathematicalContract (abstract, frozen)
  ├─ IdentityContract        : LHS ≡ RHS（TeX 文字列、人間可読）
  ├─ InequalityContract      : LHS ≥/≤ RHS（不等式不変量）
  ├─ FDConsistencyContract   : K @ du ≈ (f(u+εdu) − f(u))/ε（FD 整合性）
  ├─ SymmetryContract        : K = K^T または K = −K^T
  └─ TermExpansionContract   : K = Σ_k K_term_k（項展開網羅性）★
```

### 共通フィールド

| フィールド | 型 | 意味 |
|---|---|---|
| `name` | `str` | 契約の一意名（例 `"K_c_fd_consistency"`）。重複不可 |
| `equation_ref` | `str` | 対応数式の参照（`docs/math/ファイル名#アンカー`）|
| `severity` | `Literal["hard","soft","nightly"]` | CI での扱い |
| `description` | `str` | 人間可読な説明（省略可）|

### severity の運用

| severity | CI 扱い | 用途 |
|---|---|---|
| `"hard"` | 違反即 fail | 構造的契約（対称性、項網羅性の静的検査）|
| `"soft"` | 警告のみ | 情報提供（TeX ドキュメント等）|
| `"nightly"` | 通常 CI 対象外、nightly/PR ラベル実行 | 重量数値検査（FD 整合の実計算）|

### TermExpansionContract（MCDD の核）

`K = Σ K_term_k` を宣言し、各 `K_term_k` を実装する Process クラス名を
`providers` に列挙する。効果:

1. **静的検査 (C19, Phase E)**: `providers` の全クラスが `ProcessRegistry` に
   実在することを CI で確認。実装抜けを検出
2. **動的検証 (Phase D)**: `DiagnosticDispatcher` が合計 `Σ K_term_k` を FD と
   比較、不整合項を自動で `ContactKcComponentFDDiagnosticProcess` 等に dispatch

**K_mat x/z カップリング問題（status-344）** への適用:
```python
TermExpansionContract(
    name="K_c_term_expansion",
    equation_ref="03_huber_contact_penalty.md#eq-kc-full-decomposition",
    total_name="K_c",
    term_names=("K_mat_nn", "K_mat_ndir", "K_closest", "K_hermite_adj", "K_geo", "K_st"),
    providers=(
        "KcNormalStiffnessProcess",
        "KcNormalDirectionStiffnessProcess",  # ← 現状未実装、x/z 欠落の主犯
        "KcClosestPointStiffnessProcess",
        "KcHermiteNonlocalStiffnessProcess",
        "KcGeoStiffnessProcess",
        "KcStStiffnessProcess",
    ),
    combinator="add_sub",  # K_c = K_mat − K_geo + K_st パターン
    tol_rel=5e-3,
    severity="nightly",
)
```

`KcNormalDirectionStiffnessProcess` を `providers` に明示列挙することで、
Phase C status-352 での実装が契約側から要求される構造になる。

## 使用方法（Phase A-2 status-347 以降）

### Process への契約宣言

Phase A-2（status-347）で `AbstractProcess` にクラス属性
`contracts: ClassVar[tuple[MathematicalContract, ...]]` を追加予定。

```python
class HuberContactForceProcess(SolverProcess[...]):
    meta = ProcessMeta(...)
    uses = [...]
    contracts = (
        SymmetryContract(
            name="K_mat_symmetric",
            equation_ref="03_huber_contact_penalty.md#eq-kmat",
            matrix_name="K_mat",
            kind="symmetric",
            tol=1e-10,
            severity="hard",
        ),
        FDConsistencyContract(
            name="K_c_fd_consistency",
            equation_ref="03_huber_contact_penalty.md#eq-kc-fd",
            vector_name="f_c",
            jacobian_name="K_c",
            tol_rel=5e-3,
            severity="nightly",
        ),
        # ... (TermExpansionContract は Phase C で追加)
    )
```

### 検証 Process のバインディング（Phase A-2）

`@verified_by(contract_name, process_class)` デコレータ（Phase A-2 で追加）で
契約と検証 Process を紐付ける。C18（Phase E）で紐付け有無を静的検査。

## 段階導入ロードマップ

| Phase | status | 内容 |
|---|---|---|
| A-1 | **346 ★本 status** | `contracts.py` 5 種の型実装、単体テスト |
| A-2 | 347 | `@verified_by` + `registry.py` + `ProcessMeta.math_contracts` 拡張 |
| B-1 | 348 | `docs/math/03_huber_contact_penalty.md` 先行整備 |
| B-2 | 349 | 他 5 章 + `equation_index.py` + C15 拡張 |
| C-1 | 350 | `KcNormalStiffness` + `KcGeoStiffness` 抽出 |
| C-2 | 351 | `KcStStiffness` rename + `KcClosestPoint` 分離 |
| C-3 | 352 | **`KcNormalDirectionStiffness` 新設 = x/z 本命修正** |
| C-4 | 353 | `KcHermiteNonlocal` 抽出 + `TermExpansionContract` 完全検査 + 19本回帰 |
| D-1 | 354 | `DiagnosticDispatcherProcess` + `_newton_dynamic.py` 配線 |
| D-2 | 355 | 既存 FD 診断フラグの deprecation |
| E | 356 | C18/C19 契約検査追加 |

全体計画: `/root/.claude/plans/deep-wiggling-seal.md`

## 設計上の注意

### 数式表現の範囲

- **式そのもの** は TeX 文字列（`IdentityContract.lhs/rhs` 等）として保持。
  シンボリック処理（sympy 等）は導入しない（依存増は避ける）。
- **機械検証可能な部分** は `FDConsistencyContract` / `SymmetryContract` /
  `TermExpansionContract` が担当。
- 台帳 `docs/math/*.md`（Phase B）が単一のソース・オブ・トゥルース。

### providers が文字列である理由

`TermExpansionContract.providers: tuple[str, ...]` は Process クラス名の
**文字列** で保持する:

1. 循環 import 回避（`HuberContactForceProcess` が Kc 各項 Process を参照すると
   逆依存が生じうる）
2. 遅延解決（Phase E の C19 検査で `ProcessRegistry.default()` 経由で解決）
3. 未実装項の明示（`providers` に列挙されているが registry 未登録 → 違反検出）

### 既存 Process への非侵襲性

status-346 時点では `MathematicalContract` 型を定義するのみ。
既存の `ProcessMeta` / `AbstractProcess` / `StrategySlot` は一切改変しない。
既存 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12 テストは無影響。

`contracts` クラス属性の導入は status-347（Phase A-2）で行う。

## 関連 status

- status-289: K_c 不整合の初期発見
- status-342: 19本撚線 K_c FD 実測（x 成分 68% 不整合）
- status-343: `ContactKcComponentFDDiagnosticProcess` 新設
- status-344: 仮説 A 決着（K_mat 主導）
- status-345: report 精度バグ修正（K_geo=0 誤認の訂正）
- **status-346: 本 status、MCDD Phase A-1 着手**
