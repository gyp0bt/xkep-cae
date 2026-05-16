# status-346: MCDD Phase A-1 — MathematicalContract 型システム新設

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-16
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+**33**（`xkep_cae/mathematics/tests/test_contracts.py` 33 件追加）

## 概要

status-345 までの K_c FD 不整合追跡で「診断の診断が必要」状態（report 精度バグ
による `K_geo=0` 誤認）に至ったことを構造的に解消するため、**数理契約駆動開発
（MCDD, Mathematical Contract-Driven Development）** を最優先タスクとして導入。
本 status はその Phase A-1（5 契約型の実装 + 単体テスト）を完了した。

設計計画: `/root/.claude/plans/deep-wiggling-seal.md`（v1.0.0 凍結）
— Phase A〜E / status-346〜356 の 11 status 構成。他ロードマップ項目は凍結。

## MCDD とは

- 既存 C3〜C17 契約は「**依存関係・メタ情報・コード衛生**」の静的契約であり、
  「**この Process が解くべき方程式・満たすべき不変量**」は契約外であった。
- MCDD はそれを `MathematicalContract` 型と `docs/math/` 台帳でメタ化し、

  1. `tangent_components()` の「∂n̂/∂u の x/z 成分が抜けている」欠落を
     **契約違反として機械検出**
  2. 診断（TangentFD / KcComponentFD）の散在トリガーを
     **契約違反 → 診断 Process の自動 dispatch** に一本化
  3. 離散化方程式レジストリと Process の双方向リンクで
     status-289〜345 の調査文脈を永続化

  を実現する。

## 成果物（本 status）

### 新規パッケージ `xkep_cae/mathematics/`

| ファイル | 行数 | 内容 |
|---------|-----|------|
| `__init__.py` | 45 | 5 契約型の re-export |
| `contracts.py` | 327 | 5 種の frozen dataclass 契約型（`__post_init__` 検証） |
| `docs/mathematics.md` | 183 | 設計仕様書（階層 / severity / TermExpansion 例 / ロードマップ） |
| `tests/__init__.py` | 1 | パッケージマーカー |
| `tests/test_contracts.py` | 412 | 単体テスト 33 件 |

### 契約型階層

```
MathematicalContract (abstract, frozen dataclass)
  ├─ IdentityContract        : LHS ≡ RHS（TeX 文字列、人間可読）
  ├─ InequalityContract      : LHS {≥,≤,>,<} RHS（不変量）
  ├─ FDConsistencyContract   : K @ du ≈ (f(u+εdu) − f(u))/ε
  ├─ SymmetryContract        : K = K^T or K = −K^T
  └─ TermExpansionContract   : K = Σ_k K_term_k  ★MCDD の核
```

### 共通属性（`MathematicalContract`）

| 属性 | 型 | 意味 |
|---|---|---|
| `name` | `str` | 契約一意名（例 `"K_c_fd_consistency"`）|
| `equation_ref` | `str` | 対応数式の参照（`docs/math/*.md#アンカー`）|
| `severity` | `Literal["hard","soft","nightly"]` | CI での扱い |
| `description` | `str` | 人間可読な説明（省略可）|

### severity 運用

| severity | CI 扱い | 用途 |
|---|---|---|
| `"hard"` | 違反即 fail | 構造的契約（対称性、項網羅性の静的検査）|
| `"soft"` | 警告のみ | 情報提供（TeX ドキュメント等）|
| `"nightly"` | 通常 CI 対象外、nightly/PR ラベル | 重量数値検査（FD 整合の実計算）|

### TermExpansionContract（本契約が MCDD の核）

`K = Σ_k K_term_k` を宣言し、各項を実装する Process クラス名を
`providers: tuple[str, ...]` に文字列で列挙する:

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

**効果**:

1. **静的検査（C19, Phase E）**: `providers` の全クラスが `ProcessRegistry` に
   実在することを CI で確認。K_mat x/z 問題で想定される
   `KcNormalDirectionStiffnessProcess` の実装抜けを契約側から要求。
2. **動的検証（Phase D）**: `DiagnosticDispatcher` が合計 `Σ K_term_k` を FD と
   比較、不整合項を自動で `ContactKcComponentFDDiagnosticProcess` 等に dispatch。

### 脱法実装防止ガードレール

計画 `/root/.claude/plans/deep-wiggling-seal.md` の「🚫 脱法実装パターン 10 項」
のうち、型レベルで構造的に封じ込めるガードを本 status で実装:

- **`TermExpansionContract.providers` 重複検出**: 同じ Process を複数 term に
  二重登録して網羅性を水増しする pattern 2（dummy `@verified_by`）を防ぐ
  → `test_duplicate_providers_rejected`
- **`term_names` と `providers` の長さ一致必須**: term と provider の対応を
  曖昧にして一部項の検証を迂回する pattern 3（wrapper 被せ分解）を防ぐ
  → `test_mismatched_term_provider_length_rejected`
- **frozen dataclass**: C17（private dataclass 衛生）と整合、mutable 置換を
  封じる pattern 9（`tuple → list` 置換による frozen 回避）を防ぐ
  → `test_identity_contract_is_frozen` / `test_symmetry_contract_is_frozen`
- **必須フィールドの空文字列拒否**: ダミー値での検証回避を防ぐ
  → `test_empty_*_rejected` 各種

## 設計上の特徴

### providers が文字列である理由

`TermExpansionContract.providers: tuple[str, ...]` は Process クラス名の
**文字列** で保持する:

1. **循環 import 回避**: `HuberContactForceProcess` が Kc 各項 Process を参照
   すると逆依存が生じうる
2. **遅延解決**: Phase E の C19 検査で `ProcessRegistry.default()` 経由で解決
3. **未実装項の明示**: `providers` に列挙されているが registry 未登録
   → 違反検出可能

### 既存 Process への非侵襲性

本 status 時点では `MathematicalContract` 型を定義するのみ。
既存の `ProcessMeta` / `AbstractProcess` / `StrategySlot` は一切改変しない。
既存 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12 テストは無影響。

`contracts` クラス属性の導入は **status-347（Phase A-2）** で行う。

### 式表現の範囲

- **式そのもの** は TeX 文字列（`IdentityContract.lhs/rhs` 等）として保持。
  シンボリック処理（sympy 等）は導入しない。
- **機械検証可能な部分** のみ `FDConsistencyContract` / `SymmetryContract` /
  `TermExpansionContract` が担当。
- 台帳 `docs/math/*.md`（Phase B status-348〜349）が single source of truth。

## 段階導入ロードマップ（計画書抜粋）

| Phase | status | 内容 |
|---|---|---|
| **A-1** | **346（本 status）** | ✅ `contracts.py` 5 種の型実装、単体テスト |
| A-2 | 347 | `@verified_by` + `registry.py` + `ProcessMeta.math_contracts` 拡張 |
| B-1 | 348 | `docs/math/03_huber_contact_penalty.md` 先行整備 |
| B-2 | 349 | 他 5 章 + `equation_index.py` + C15 拡張 |
| C-1 | 350 | `KcNormalStiffness` + `KcGeoStiffness` 抽出 |
| C-2 | 351 | `KcStStiffness` rename + `KcClosestPoint` 分離 |
| C-3 | 352 | **`KcNormalDirectionStiffness` 新設 = x/z 本命修正** |
| C-4 | 353 | `KcHermiteNonlocal` 抽出 + `TermExpansionContract` 完全検査 + 19本回帰 |
| D-1 | 354 | `DiagnosticDispatcherProcess` + `_newton_dynamic.py` 配線 |
| D-2 | 355 | 既存 FD 診断フラグの deprecation |
| E | 356 | C18（`@verified_by` 紐付け検査）/ C19（`term_processes` 実在検査）追加 |

## 検証・品質確認（4-Gate 全 pass）

計画 `/root/.claude/plans/deep-wiggling-seal.md` の「✅ 各 status で必須となる
『本質対策ゲート』」に従い、全 Gate のログを取得:

### Gate 1: 既存テスト無影響（`/tmp/mcdd-346-pytest.log`）

```
700 passed, 10 skipped, 1 xfailed in 96.53s
(xkep_cae/ excluding test_stress_contour: 667 pre-existing + 33 new contract tests)

249 passed, 10 skipped, 64 deselected in 38.79s (tests/, -m "not slow and not external")
```

既存 skip/xfail 増加 **0**。`test_stress_contour` の 1 failure は status-329 以前
から存在する matplotlib 描画環境依存の pre-existing failure（image_paths=[] の
環境問題で、本 status の変更とは独立）。

### Gate 2: 契約検査（`/tmp/mcdd-346-contract.log`）

```
============================================================
契約違反なし、条例違反なし
```

C3〜C17 + O1〜O3 全 15 検査項目クリア。`xkep_cae/mathematics/` 新設による
契約違反は 0。

### Gate 3: ruff check/format（`/tmp/mcdd-346-ruff.log`）

```
All checks passed!
4 files already formatted
```

### Gate 4: 7本撚線回帰

Phase A-1 は型定義のみで既存ソルバーへの侵襲がなく、7本撚線回帰は省略可能と
判断（計画書で Gate 4 は Phase C 以降のみ必須）。

## コミット予定

```
feat: MCDD Phase A-1 導入 — MathematicalContract 型システム新設 (status-346)

- xkep_cae/mathematics/ パッケージ新設
  - contracts.py: 5 種の frozen dataclass 契約型
    (Identity / Inequality / FDConsistency / Symmetry / TermExpansion)
  - docs/mathematics.md: 設計仕様書
  - tests/test_contracts.py: 33 件単体テスト（全 pass）

- TermExpansionContract で K = Σ K_term_k を宣言可能に
  - providers 重複検出で脱法実装 pattern 2 を封じ込め
  - term_names と providers の長さ一致必須で pattern 3 を防ぐ

- 既存 Process 改変なし（Phase A-2 status-347 で ProcessMeta 拡張予定）

Plan: /root/.claude/plans/deep-wiggling-seal.md (v1.0.0 frozen)
他ロードマップ項目は MCDD 完了まで凍結。
```

## 次セッション引き継ぎ（status-347 向け）

**開始前に必ず読むファイル**（計画「🔍 セッション開始時の必須確認」順守）:

1. `/root/.claude/plans/deep-wiggling-seal.md` を**全文読む**（要約禁止）
2. 本 `status-346.md`（Phase A-1 完了断面）
3. 計画書「🚫 脱法実装パターン 10 項」を読み返し、status-347 で陥りそうな
   項目（特に pattern 2: dummy `@verified_by`）を自己チェック

### status-347 の目標（Phase A-2）

1. `xkep_cae/mathematics/registry.py` 新設
   - `ProcessContractRegistry` クラス
   - `@verified_by(contract_name, verify_process_class)` デコレータ
   - `register_contracts(process_class, contracts)` ヘルパ

2. `xkep_cae/core/base.py` の `ProcessMeta` に `math_contracts` フィールド追加
   - デフォルト空 tuple（後方互換）
   - `AbstractProcess` サブクラスで `contracts: ClassVar[tuple[MathematicalContract, ...]]`
     クラス属性を宣言可能に

3. 単体テスト: `xkep_cae/mathematics/tests/test_registry.py`
   - `@verified_by` で実 `VerifyProcess`（中身のある）を紐付けるテスト
   - dummy VerifyProcess を紐付けようとした場合の検出（C18 前段）

### status-347 の禁止事項（脱法実装 pattern チェックリスト）

- ❌ 中身が空の dummy `VerifyProcess` を `@verified_by` に紐付けて
  「A-2 完了」とする
- ❌ `math_contracts` フィールドを既存 `ProcessMeta` に追加する際、
  default_factory ではなく list 型で追加（pattern 9）
- ❌ Phase を「A-2a / A-2b」に分割して status-347 を骨格だけで締める（pattern 6）

### 成功基準（status-347）

- 既存全テスト pass、skip/xfail 増加 0
- `test_registry.py` 新規テスト pass
- 契約違反 0 件維持
- `ProcessMeta.math_contracts` が既存 Process（例: `HuberContactForceProcess`）で
  デフォルト空 tuple で動作することを確認
