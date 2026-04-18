# status-351: MCDD Phase C-2 — `KcHermiteNonlocal` / `KcClosestPoint` Process 抽出 + 5 項 TermExpansionContract

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-18
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+**25**（Phase C-2 で +11 追加、14→25）

## 概要

status-350 Phase C-1 の 3 項分解（`K_mat` / `K_geo` / `K_st`）を **5 項** に拡張した。
数理台帳 `docs/math/03_huber_contact_penalty.md#eq-kc-full-decomposition`
の 6 項完全分解（`K_mat_nn` / `K_mat_ndir` / `K_closest` / `K_hermite_adj` /
`K_geo` / `K_st`）のうち、**K_mat_ndir 以外の 5 項** を独立 Process として
確立。`K_mat_ndir`（法線方向感度）は status-352 本命修正で追加予定。

- `KcNormalStiffnessProcess` — **K_mat_nn のみ**（ペア局所 $w_\mathrm{mat}(\hat{n}\otimes\hat{n})$）
  に縮退。旧実装の K_mat_adj 隣接拡張は分離。version 1.0.0 → **1.1.0**
- **`KcHermiteNonlocalStiffnessProcess`**（新設）— Hermite 非局所 `K_hermite_adj`
  （隣接ノード mat-only 拡張、status-271〜274/295 の mat-only 形式）
- **`KcClosestPointStiffnessProcess`**（新設）— 最近接点追従 `K_closest`
  （K_st 全体の「(s,t) 摂動に伴う $p_n$ 追従」成分 $-(dpn\_ds \cdot g\_shape) \otimes ds\_du$）
- `KcGeoStiffnessProcess` — 変更なし
- `ContactForceStStiffnessProcess` — **K_st 残差項のみ** に縮退。
  `_assemble_term_coo(term="residual"/"closest")` classmethod で共通セットアップを
  公開し、`KcClosestPointStiffnessProcess` も同じ classmethod を呼び出す（C5 uses 宣言遵守）

`tangent_components()` は **5 Process を orchestrate**し、後方互換のため 3-tuple
`(K_mat_nn + K_hermite_adj, K_geo, K_closest + K_st_residual)` を返す。総和不変。

設計計画: `/root/.claude/plans/deep-wiggling-seal.md`（v1.0.0 凍結）
— Phase A〜E / status-346〜356 の **6/11 完了**。

## 成果物

### 新規クラス / 公開 API

| 名前 | 種別 | 役割 |
|------|------|------|
| `KcHermiteNonlocalStiffnessOutput` | frozen dataclass | `K_hermite_adj: sp.csr_matrix` |
| `KcHermiteNonlocalStiffnessProcess` | `SolverProcess` | Hermite 隣接ノード mat-only 拡張（status-295） |
| `KcClosestPointStiffnessOutput` | frozen dataclass | `K_closest: sp.csr_matrix \| sp.coo_matrix` |
| `KcClosestPointStiffnessProcess` | `SolverProcess` | 最近接点 (s,t) 追従 p_n 寄与 |
| `ContactForceStStiffnessProcess._assemble_term_coo` | `classmethod` | term="residual"/"closest" 分岐の共通 COO 組み立てヘルパ |

### `_K_C_TERM_EXPANSION_CONTRACT` 更新（3 項 → 5 項）

```python
_K_C_TERM_EXPANSION_CONTRACT: TermExpansionContract = TermExpansionContract(
    name="K_c_term_expansion",
    equation_ref="03_huber_contact_penalty.md#eq-kc-full-decomposition",
    total_name="K_c",
    term_names=("K_mat_nn", "K_closest", "K_hermite_adj", "K_geo", "K_st"),
    providers=(
        "KcNormalStiffnessProcess",
        "KcClosestPointStiffnessProcess",
        "KcHermiteNonlocalStiffnessProcess",
        "KcGeoStiffnessProcess",
        "ContactForceStStiffnessProcess",
    ),
    combinator="add_sub",
    tol_rel=5e-3,
    severity="nightly",
    description=(
        "status-351 Phase C-2: tangent_components() から 5 項を抽出。"
        "K_mat_nn / K_closest / K_hermite_adj / K_geo / K_st が独立 Process。"
        "K_mat_ndir は status-352 本命修正で追加予定。"
    ),
)
```

- 5 Process 全てが **同一インスタンス** を ClassVar `contracts` に宣言
- `term_names` / `providers` は同順同長。`TermExpansionContract.__post_init__` で整合検査

### K_st の代数的分割（K_closest + K_st_residual）

$\partial\boldsymbol{f}_c/\partial s$ の完全微分を 2 成分に分解:

$$
\frac{\partial f_c}{\partial s} =
\underbrace{\frac{\partial p_n}{\partial s} \cdot g_{\text{shape}}}_{\text{K\_closest 寄与}}
\;+\;
\underbrace{p_n \cdot \left(\frac{\partial c}{\partial s}\cdot \hat{n}
+ c \cdot \frac{\partial \hat{n}}{\partial s}\right)}_{\text{K\_st 残差寄与}}
$$

- **K_closest**: $(s,t)$ 摂動に伴う $p_n$ 追従 = $-(\partial p_n/\partial(s,t))\,(c\otimes\hat{n})\,\partial(s,t)/\partial u$
- **K_st 残差**: 形状関数・法線の $(s,t)$ 追従 = $-p_n\,(\partial c/\partial(s,t)\cdot \hat{n} + c \cdot \partial \hat{n}/\partial(s,t))\,\partial(s,t)/\partial u$

両者の和 = 旧 K_st 全体（数値恒等）。`test_orchestrator_k_st_equals_kst_residual_plus_closest`
で `atol=1e-14` 検証。

### 変更ファイル

| ファイル | 内容 |
|---------|------|
| `xkep_cae/contact/contact_force/strategy.py` | 5 項 TermExpansionContract、`KcHermiteNonlocalStiffnessOutput` / `KcClosestPointStiffnessOutput` dataclass、`KcNormalStiffnessProcess` から K_mat_adj 分離（v1.0.0 → 1.1.0）、`KcHermiteNonlocalStiffnessProcess` 新設、`ContactForceStStiffnessProcess._assemble_term_coo` / `_process_batch_term` classmethod 化で term 分岐、`KcClosestPointStiffnessProcess` 新設（`uses = [ComputeStJacobianProcess, ContactForceStStiffnessProcess]` で C5 遵守）、`tangent_components()` orchestrator を 5 Process 呼び出しへ拡張 |
| `xkep_cae/contact/contact_force/__init__.py` | 新 4 シンボル（`KcClosestPointStiffnessOutput` / `Process`、`KcHermiteNonlocalStiffnessOutput` / `Process`）を公開 |
| `xkep_cae/contact/contact_force/tests/test_kc_term_processes.py` | `TestKcHermiteNonlocalStiffnessProcess`（4 テスト）+ `TestKcClosestPointStiffnessProcess`（4 テスト）+ `TestTangentComponentsOrchestration` に `test_orchestrator_k_st_equals_kst_residual_plus_closest` 追加（既存 14 → 25）、`TestKcTermExpansionContract` を 5 項へ更新（providers_registered_on_all_five_processes / contract_structure） |
| `docs/math/03_huber_contact_penalty.md` | 章 3.1 項一覧表を 5 項完了状態に更新（K_mat_nn ✅ / K_closest ✅ / K_hermite_adj ✅ / K_geo ✅ / K_st ✅ / K_mat_ndir のみ未実装） |

## 設計判断

### 共通 classmethod による K_st ↔ K_closest の共有

`ContactForceStStiffnessProcess._process_batch_term(inp, term)` に集約:

- ペアデータ抽出（status-322 pre-bound states パターン）
- 距離 culling（status-324）
- バッチ StJacobian（Hermite / 線形）
- 形状関数微分 (dc_ds / dc_dt / coeffs)
- ∂n/∂s, ∂n/∂t, ∂p_n/∂s, ∂p_n/∂t 計算
- `term` 分岐で `K_closest_local` or `K_st_residual_local` を選択して COO 組み立て

`_assemble_term_coo` は `_process_batch_term` の薄い公開 wrapper。
`KcClosestPointStiffnessProcess.uses = [ComputeStJacobianProcess,
ContactForceStStiffnessProcess]` で AST 依存解析（C5）を通過。

**脱法 pattern 3（wrapper 被せ）の回避**: classmethod は同一 Process の
別の項を切り替えるパラメトリック分岐であり、`KcClosestPointStiffnessProcess`
は独立の公開 API（`process(inp) -> KcClosestPointStiffnessOutput`）を持つ。
呼び出し側（orchestrator・FD テスト）は共通 classmethod の存在を意識せず
両 Process を独立に呼び出す。実数値は代数的に 2 項に分かれており、
pattern 3 の「wrapper だけで中身が空」には該当しない。

### 3-tuple 後方互換

既存 caller（`_newton_dynamic.py` / `test_kc_component_fd.py` 7 件 /
`verify/kc_component_fd.py` 12 件）は 3-tuple `(K_mat, K_geo, K_st)` を期待する。
orchestrator は内部で 5 Process を呼び、:

```python
K_mat = K_mat_nn + K_hermite_adj           # KcNormal + KcHermiteNonlocal
K_geo = K_geo                              # KcGeo
K_st  = K_closest + K_st_residual          # KcClosest + ContactForceSt
return K_mat, K_geo, K_st
```

数値恒等を維持（総和不変）、既存 FD テスト 19 件 **無変更で pass**。

## 検証

### gate 1: 既存 FD テスト無変更 pass（数値恒等性）

```
$ uv run pytest xkep_cae/contact/contact_force/tests/test_kc_component_fd.py xkep_cae/verify/tests/test_kc_component_fd.py -q
19 passed in 1.67s
```

- status-350 で確立された 19 件の FD 整合性テスト（contact_force 7 + verify 12）を
  無変更のまま維持。総和 K_c = K_mat - K_geo + K_st が数値恒等なのでそのまま通る

### gate 2: 新 Process 単体テスト + orchestrator 一致テスト

```
$ uv run pytest xkep_cae/contact/contact_force/tests/test_kc_term_processes.py -q
25 passed in 0.69s
```

- `@binds_to(KcHermiteNonlocalStiffnessProcess)` / `@binds_to(KcClosestPointStiffnessProcess)`
  で C3 紐付け
- 5 項 TermExpansionContract の 5 Process 全てでの宣言検証
- **orchestrator 一致テスト**:
  - `test_orchestrator_matches_process_outputs`: K_mat = K_mat_nn + K_hermite_adj を `atol=1e-14` で確認
  - `test_orchestrator_k_st_equals_kst_residual_plus_closest`: K_st (orchestrator) = K_closest + K_st_residual を `atol=1e-14` で確認

### gate 3: K_st 関連テスト無変更 pass

```
$ uv run pytest xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py -q
17 passed in 1.41s
```

- `test_kst_adj_disabled_status294` 等、隣接ノード列ゼロ性テストは K_st_residual
  に対して引き続き成立（dm_ext 無効化後は K_st 残差の隣接列も 0）

### gate 4: 契約違反 0 件 + ruff + 全テスト

```
$ uv run ruff check xkep_cae/ tests/
All checks passed!

$ uv run ruff format --check xkep_cae/ tests/
191 files already formatted

$ uv run python contracts/validate_process_contracts.py
契約違反なし、条例違反なし

$ uv run python -m pytest xkep_cae/ -q -k "not slow"
774 passed, 10 skipped, 14 deselected, 1 xfailed

$ uv run python -m pytest tests/ -q -k "not slow"
249 passed, 10 skipped, 64 deselected
```

### gate 5: 7本撚線 frac=1.0 回帰（47s、status-350 82s から高速化は本 status 範囲外）

```
$ uv run pytest xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -q
18 passed, 3 warnings in 47.08s
```

含む: `test_7strand_90deg_dynamic_completes` / `test_center_strand_tip_displacement` /
`test_mpc_90deg_nocontact_completes` — 全 frac=1.0 完走。

## 次セッション引き継ぎ（status-352 向け: Phase C-3 本命修正）

**開始前に必ず読むファイル**:

1. `/root/.claude/plans/deep-wiggling-seal.md` を**全文読む**（要約禁止）
2. 本 `status-351.md` + `status-350.md`（Phase C-1/C-2 継続断面）
3. 計画書「🚫 脱法実装パターン 10 項」を読み返し、status-352 で陥りそうな
   項目（特に **pattern 4: rename で済ませる**、**pattern 7: tol 事後緩和**）を自己チェック

### status-352 の目標（Phase C-3: `KcNormalDirectionStiffnessProcess` 新設 — x/z 成分カップリング解消）

計画書 Phase C（status-350〜353）の **本命修正**:

1. **`KcNormalDirectionStiffnessProcess`** を新規実装（rename 禁止、pattern 4 回避）
   - 数理台帳 `#eq-dn-du`: $\partial \hat{n}/\partial u = (1/d)\,P_\perp \cdot \partial r/\partial u$
   - ペア局所: $-p_n \cdot \partial \hat{n}/\partial u$（status-344 で観測された K_mat x/z カップリング欠落を解消）
2. `TermExpansionContract.term_names` を 5 項 → **6 項**に拡張:
   `("K_mat_nn", "K_mat_ndir", "K_closest", "K_hermite_adj", "K_geo", "K_st")`
3. `test_kc_component_fd.py` 19 件 pass 維持（`rel_err` が下がることを確認、
   tol は 0.05 / 0.1 を変更しない — pattern 7 禁止）
4. 19本撚線 Type D stall 解消（frac=0.48→1.0）を確認

### status-352 の禁止事項

- ❌ `KcNormalDirectionStiffnessProcess` を既存項の rename で済ませる（pattern 4）
- ❌ `test_kc_component_fd.py` の tol を 0.05/0.1 から緩和（pattern 7）
- ❌ 19本撚線が不収束の場合「ベースラインが悪い」と主張（pattern 8）

### 成功基準（status-352）

- `KcNormalDirectionStiffnessProcess` 新規実装、FD self-consistency テスト `rel_err < 1e-2`
- `TermExpansionContract.term_names` を 6 項化、全 6 Process が契約宣言
- `test_kc_component_fd.py` 19 件無変更 pass + 新 Process 単体テスト追加
- 19本撚線 Type D stall 解消（frac=1.0、300s 以内完走）
- 契約違反 0 件維持、Phase A〜E の **7/11 完了**

## コミット（予定）

```
(次) feat(contact): MCDD Phase C-2 step-1 — KcHermiteNonlocal/KcClosestPoint Process 抽出 (status-351)
(次) feat(contact): MCDD Phase C-2 step-2 — _assemble_term_coo classmethod + 5 項 TermExpansionContract + tangent_components orchestrator 拡張 (status-351)
(次) test(contact): MCDD Phase C-2 step-3 — 新 Process 単体テスト + orchestrator 等式テスト (status-351)
(次) docs(status): status-351 + README/roadmap/status-index/math doc 03 章更新 (Phase C-2 完了)
```

Plan: `/root/.claude/plans/deep-wiggling-seal.md` (v1.0.0 frozen)
Phase A〜E / status-346〜356 の **6/11 完了**。
