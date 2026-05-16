# status-348: MCDD Phase B-1 — `docs/math/03_huber_contact_penalty.md` 先行整備

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-16
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33（**変動なし**：本 status は Markdown 台帳のみで実装非侵襲）

## 概要

status-347（Phase A-2: `ProcessContractRegistry` + `@verified_by`）に続き、
**MCDD Phase B-1** として接触系の離散化方程式台帳
`docs/math/03_huber_contact_penalty.md` を先行整備した。
`MathematicalContract.equation_ref` から `<file>#<anchor>` 形式で参照される
**単一のソース・オブ・トゥルース**を確立し、Phase C（status-350〜353）の
項別 Process 抽出に向けた数式アンカー基盤を提供する。

設計計画: `/root/.claude/plans/deep-wiggling-seal.md`（v1.0.0 凍結）
— Phase A〜E / status-346〜356 の **3/11 完了**。他ロードマップ項目は凍結。

## 成果物

### 新規ファイル

| ファイル | 行数 | 内容 |
|---------|-----|------|
| `docs/math/README.md` | 38 | 数理台帳の索引、章立て、アンカー命名規約、整合性ルール |
| `docs/math/03_huber_contact_penalty.md` | 322 | 8 節 + 19 アンカー：Huber/Hertz $p_n$、$\boldsymbol{f}_c$、$\boldsymbol{K}_c$ 完全項展開、$\partial \hat{\boldsymbol{n}}/\partial \boldsymbol{u}$、対称性、FD 整合、Hermite 非局所 |

### 変更ファイル

| ファイル | 内容 |
|---------|------|
| `README.md` | 「現在の状態」更新（Phase A-2 → B-1）、ドキュメント表に [`docs/math/`](../docs/math/README.md) 追加 |
| `docs/roadmap.md` | 「現在地」と Phase 進捗（**3/11**）を更新 |
| `docs/status/status-index.md` | 348 行を追加 |

## 設計

### 章立てとアンカー命名規約（`docs/math/README.md`）

`equation_index.py`（status-349 で実装）が機械抽出できるよう、
以下の規約をプロジェクト全体で統一:

| アンカー prefix | 用途 | 契約型 |
|---|---|---|
| `eq-<topic>` | 方程式そのもの | `IdentityContract` / `TermExpansionContract` / `FDConsistencyContract` |
| `eq-<topic>-fd` | 同方程式の FD 整合性版 | `FDConsistencyContract` |
| `inv-<topic>` | 不変量・不等式 | `InequalityContract` |
| `sym-<topic>` | 対称性命題 | `SymmetryContract` |

章番号体系は status-349（Phase B-2）で 6 章まで拡張予定:

| # | ファイル | 状態 |
|---|---------|------|
| 01 | `01_kinematics_beam.md` | status-349 で整備 |
| 02 | `02_contact_geometry.md` | status-349 で整備 |
| **03** | **`03_huber_contact_penalty.md`** | ✅ **本 status** |
| 04 | `04_friction_smooth_penalty.md` | status-349 で整備 |
| 05 | `05_coating_barrier.md` | status-349 で整備 |
| 06 | `06_time_integration.md` | status-349 で整備 |

### `03_huber_contact_penalty.md` の 8 節構成

| 節 | アンカー | 内容 |
|---|---|---|
| 表記 | — | 記号一覧 + 実装 trace（`HuberContactForceProcess`） |
| 1. 法線ペナルティ力 $p_n$ | `eq-pn-huber`, `eq-pn-linear`, `eq-pn-hertz`, `eq-dpn-dx`, `inv-pn-nonneg`, `eq-pn`（alias） | Huber 平滑化 + Hertz $\alpha=1.5$ + 非負不変量 |
| 2. 接触力 $\boldsymbol{f}_c$ | `eq-fc`, `eq-fc-assembly` | $\boldsymbol{f}_c=-p_n\hat{\boldsymbol{n}}$ + Hermite 形状係数組み立て |
| **3. $\boldsymbol{K}_c$ 完全項展開** | **`eq-kc-full-decomposition`**, `eq-kc-pair-block`, `eq-kc`（alias）, `eq-kc-def`（alias） | **6 項分解**（`K_mat_nn` / **`K_mat_ndir`★** / `K_closest` / `K_hermite_adj` / `K_geo` / `K_st`）+ `term_processes` 対応表 |
| 4. 法線方向感度 $\partial \hat{\boldsymbol{n}}/\partial \boldsymbol{u}$ | `sec-ndir`, `eq-dn-du` | **K_mat,ndir 本命**（status-352 修正対象） + $\boldsymbol{K}_{\mathrm{geo}}$ との混同防止表 |
| 5. $\boldsymbol{K}_{\mathrm{mat}}$ 対称性 | `eq-kmat`, `sym-kmat` | `SymmetryContract` 用 |
| 6. FD 整合性 | `eq-kc-fd`, `eq-kc-term-fd` | `FDConsistencyContract` 用 + 項別 FD 拡張 |
| 7. Hermite 非局所 | `eq-hermite-pA` | $\boldsymbol{p}_A(s)$ + frozen-m 部分解消（status-294） |
| 8. 既存実装との trace | — | 各方程式 ↔ コード位置の対応表（**`eq-dn-du` のみ "未実装"** と明記） |

合計 **19 アンカー**、すべて一意（重複なし、`python3 -c "..."` で検証済み）。

### 既存 docstring 例との整合（`xkep_cae/mathematics/contracts.py`）

status-346 で書かれた contracts.py の docstring 例は以下のアンカーを参照:

| 参照元 | アンカー | 03 章での提供形態 |
|---|---|---|
| `MathematicalContract.equation_ref` 説明 | `#eq-kc` | **alias** として `## 3. ...` に併記 |
| `IdentityContract` 例 | `#eq-kc-def` | **alias** として `## 3. ...` に併記 |
| `InequalityContract` 例 | `#eq-pn` | **alias** として `## 1. ...` に併記 |
| `FDConsistencyContract` 例 | `#eq-kc-fd` | 6 節に正規アンカー |
| `SymmetryContract` 例 | `#eq-kmat` | 5 節に正規アンカー |
| `TermExpansionContract` 例 | `#eq-kc-full-decomposition` | 3 節に正規アンカー |

`docs/mathematics.md` 例の参照（`#eq-kc-full-decomposition` / `#eq-kmat` /
`#eq-kc-fd`）も全て解決可能。**status-349 で `equation_index.py` を実装した
時点で全 6 参照が機械的に検証される**。

## 脱法実装防止（計画書「🚫 10 項」）の本 status での自己点検

| Pattern | 本 status での回避策 |
|---|---|
| **6**: 困難の先送り（骨格 status） | 8 節 + 19 アンカーで Phase C で必要となる全項を**先取り宣言**。特に `K_mat_ndir`（未実装）も明示列挙し、status-352 で「契約側から要求される」構造を確立 |
| **7**: 診断 report の精度バグ（`{:5.2f}` 問題） | 台帳には**数値を一切書かない**ルールを README で明文化（`docs/math/README.md` 整合性ルール 4）。実測値は status 側に隔離 |
| **10**: 「TODO として積む」で次回送り | 03 章の trace 表（8 節）で「未実装＝ `eq-dn-du`」の単一項を明示。status-349 以降の Phase B-2 / Phase C のスコープも `docs/math/README.md` 章立てで予告済み |
| **3**: wrapper 被せ分解 | `K_c` の **6 項全て** に `term_name` と Process クラス名を割当て（実装抜け = 契約違反）。`KcNormalDirectionStiffnessProcess` は **未実装でも `term_processes` に登録予定** とし、status-352 の実装抜けを Phase E（C19）で機械検出可能にする |

## 検証・品質確認（4-Gate 全 pass）

計画「✅ 各 status で必須となる『本質対策ゲート』」に従い、本 status は
**Markdown のみの台帳整備**で実装非侵襲のため Gate 1/4 は省略可能と判断:

### Gate 2: 契約検査（`uv run python contracts/validate_process_contracts.py`）

```
============================================================
契約違反なし、条例違反なし
```

C3〜C17 + O1〜O3 全 15 検査項目クリア。`mathematics/` 除外（status-347）と
`docs/math/` 新設は契約検査の対象外。

### Gate 3: ruff check / format

```
All checks passed!
187 files already formatted
```

### Gate 1（テスト）/ Gate 4（7本撚線回帰）: 省略

Phase B-1 は `docs/` 配下の Markdown 追加のみ。`xkep_cae/` ソース変更ゼロ、
`tests/` 変更ゼロのため、既存テスト 459+13+...+33+33 への影響は構造的にあり得ない。
計画書で Gate 4 は Phase C 以降のみ必須。

### アンカー一意性検証

```
total: 19 unique: 19
['eq-dn-du', 'eq-dpn-dx', 'eq-fc', 'eq-fc-assembly', 'eq-hermite-pA',
 'eq-kc', 'eq-kc-def', 'eq-kc-fd', 'eq-kc-full-decomposition',
 'eq-kc-pair-block', 'eq-kc-term-fd', 'eq-kmat', 'eq-pn', 'eq-pn-hertz',
 'eq-pn-huber', 'eq-pn-linear', 'inv-pn-nonneg', 'sec-ndir', 'sym-kmat']
```

重複ゼロ。Phase B-2 status-349 で `equation_index.py` が C15 拡張として
台帳全体（6 章）の一意性を機械検証する。

## 次セッション引き継ぎ（status-349 向け: Phase B-2）

**開始前に必ず読むファイル**:

1. `/root/.claude/plans/deep-wiggling-seal.md` を**全文読む**（要約禁止）
2. 本 `status-348.md` および `status-346.md` / `status-347.md`（Phase A-B 継続断面）
3. 計画書「🚫 脱法実装パターン 10 項」を読み返し、status-349 で陥りそうな
   項目（特に **pattern 6: 困難の先送り**、**pattern 10: TODO 積み**）を自己チェック

### status-349 の目標（Phase B-2）

1. 残り 5 章を `docs/math/` 配下に追加:
   - `01_kinematics_beam.md`（CR / TL / UL、Hermite 補間）
   - `02_contact_geometry.md`（最近接点 $(s,t)$ 射影、StJacobian）
   - `04_friction_smooth_penalty.md`（Coulomb return mapping）
   - `05_coating_barrier.md`（バリア関数）
   - `06_time_integration.md`（HHT-α / Newmark）

2. `xkep_cae/mathematics/equation_index.py` 新設:
   - `docs/math/*.md` を走査して `<a id="(eq-|inv-|sym-|sec-)...">` を抽出
   - `MathematicalContract.equation_ref` の `<file>#<anchor>` を解決
   - **重複アンカー** / **未解決参照** を検出する API

3. `contracts/validate_process_contracts.py` の **C15 拡張**:
   - 全 `MathematicalContract` インスタンスの `equation_ref` を `equation_index`
     で解決し、未解決時に契約違反を計上
   - ただし dummy 例（contracts.py docstring 内）は AST 検査で除外

### status-349 の禁止事項

- ❌ 5 章を **アウトラインだけ** で済ませる（pattern 6）。各章で最低限
  `## 表記 / ## 主要方程式 / ## 既存実装との trace` の 3 節を完成させる
- ❌ `equation_index.py` を骨格だけ作って実 C15 拡張は status-350 送り（pattern 10）
- ❌ アンカー命名規約（`eq-/inv-/sym-/sec-`）から逸脱

### 成功基準（status-349）

- 全 6 章で重複ゼロ + `equation_index.py` で機械検証
- C15 拡張の単体テスト追加（重複検出 + 未解決検出 + 正常解決）
- 既存全テスト pass、skip/xfail 増加 0、契約違反 0 件維持

## コミット予定

```
docs(math): MCDD Phase B-1 — Huber 接触ペナルティ系の離散化方程式台帳 (status-348)

- docs/math/ ディレクトリ新設（数理台帳の単一のソース・オブ・トゥルース）
  - README.md: 索引、章立て、アンカー命名規約、整合性ルール
  - 03_huber_contact_penalty.md: 8 節 / 19 アンカー
    - Huber 平滑化 + Hertz 非線形 (α=1.5) ペナルティ
    - 接触力 f_c = -p_n n̂
    - K_c の 6 項完全分解 (K_mat_nn / K_mat_ndir / K_closest /
      K_hermite_adj / K_geo / K_st)
    - ∂n̂/∂u 法線方向感度（K_mat,ndir 未実装、status-352 本命修正）
    - K_mat 対称性、FD 整合（項別を含む）、Hermite 非局所

- README.md / docs/roadmap.md / docs/status/status-index.md 更新
  - Phase A〜E / status-346〜356 の進捗 3/11 へ

- 既存 contracts.py docstring 例（#eq-kc / #eq-kc-def / #eq-pn）を
  alias アンカーで解決可能に維持

Plan: /root/.claude/plans/deep-wiggling-seal.md (v1.0.0 frozen)
Phase A〜E / status-346〜356 の 3/11 完了。
```
