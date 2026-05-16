# status-350: MCDD Phase C-1 — `KcNormal` / `KcGeo` / `KcSt` 項別 Process 抽出

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-18
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+**14**（Phase C-1 で **+14**）

## 概要

`HuberContactForceProcess.tangent_components()` が内製していた K_c の 3 項分解
（`K_mat` / `K_geo` / `K_st`）を独立 Process に分離し、`TermExpansionContract`
（status-346 で新設）の `providers` と 1:1 対応させた。
これによって **MCDD Phase C**（項別 Process への解体）の骨組みが立ち上がる。

- `KcNormalStiffnessProcess` — 法線材料剛性 $\boldsymbol{K}_\mathrm{mat}$
  （$w_\mathrm{mat}\,\hat{\boldsymbol{n}}\otimes\hat{\boldsymbol{n}}$）と
  隣接ノード材料剛性拡張（status-295 の `K_c_adj` mat-only）を合算
- `KcGeoStiffnessProcess` — 幾何補正項 $\boldsymbol{K}_\mathrm{geo}
  = (p_n/d)(I - \hat{\boldsymbol{n}}\otimes\hat{\boldsymbol{n}})$
- `ContactForceStStiffnessProcess`（status-256 で既存）に
  `contracts: ClassVar = (_K_C_TERM_EXPANSION_CONTRACT,)` を追記、`version 1.1.0`

`tangent_components()` は 3 Process の出力を組み立てる **orchestrator** に縮退。
`test_kc_component_fd.py`（contact_force 7 + verify 12 = **19 件**）を
**無変更で pass**（gate 遵守）。

設計計画: `/root/.claude/plans/deep-wiggling-seal.md`（v1.0.0 凍結）
— Phase A〜E / status-346〜356 の **5/11 完了**。他ロードマップ項目は凍結。

## 成果物

### 新規クラス / 公開 API

| 名前 | 種別 | 役割 |
|------|------|------|
| `KcTermAssemblyInput` | frozen dataclass | 3 Process 共通入力（pairs, k_pen, delta_h, ndof_total, ...） |
| `KcNormalStiffnessOutput` | frozen dataclass | `K_mat: sp.csr_matrix` |
| `KcGeoStiffnessOutput` | frozen dataclass | `K_geo: sp.csr_matrix` |
| `KcNormalStiffnessProcess` | `SolverProcess` | K_mat + K_mat_adj（隣接ノード材料剛性拡張） |
| `KcGeoStiffnessProcess` | `SolverProcess` | K_geo ペア局所組み立て |
| `_K_C_TERM_EXPANSION_CONTRACT` | `TermExpansionContract` | 3 Process が同一インスタンスを宣言（module-level） |

### 新規テスト

- `xkep_cae/contact/contact_force/tests/test_kc_term_processes.py` — **14 テスト**
  - `@binds_to(KcNormalStiffnessProcess)` / `@binds_to(KcGeoStiffnessProcess)`
    で C3 紐付け成立
  - meta / contract ClassVar / 空ペア / 非活性ペア / 単一ペア非ゼロ
  - `TermExpansionContract` 骨格検証（`term_names` / `providers` / `combinator`）
  - **orchestrator 一致テスト**: `tangent_components()` 出力 ≡ `KcNormal + KcGeo` 直接呼び出し

### 変更ファイル

| ファイル | 内容 |
|---------|------|
| `xkep_cae/contact/contact_force/strategy.py` | `KcTermAssemblyInput` / 2 Output / 2 Process 新設、`_extract_kc_active_pair_data` 共通ヘルパ + `_assemble_12x12_pair_block` 共通アセンブラ、`ContactForceStStiffnessProcess` に `contracts` ClassVar 追加（version 1.0.0 → 1.1.0）、`tangent_components()` を orchestrator 化 |
| `xkep_cae/contact/contact_force/__init__.py` | `KcTermAssemblyInput` / `KcNormal*` / `KcGeo*` を公開 |
| `xkep_cae/contact/contact_force/tests/test_kc_term_processes.py` | **新規**、14 テスト |
| `xkep_cae/mathematics/tests/test_c15_extension.py` | **autouse fixture** で `ProcessContractRegistry.default()` をテスト毎に分離（contact_force import による contracts 汚染で C15 テストが fake ledger に対して誤検出するのを防止） |
| `docs/math/03_huber_contact_penalty.md` | 章 3.1 表の「K_mat / K_geo / K_st を status-350 で抽出」ステータスを ✅ 更新 |

## 設計

### `_K_C_TERM_EXPANSION_CONTRACT`（module-level）

```python
_K_C_TERM_EXPANSION_CONTRACT: TermExpansionContract = TermExpansionContract(
    name="K_c_term_expansion",
    equation_ref="03_huber_contact_penalty.md#eq-kc-full-decomposition",
    total_name="K_c",
    term_names=("K_mat", "K_geo", "K_st"),
    providers=(
        "KcNormalStiffnessProcess",
        "KcGeoStiffnessProcess",
        "ContactForceStStiffnessProcess",
    ),
    combinator="add_sub",
    tol_rel=5e-3,
    severity="nightly",
    description="status-350 Phase C-1: tangent_components() から 3 項を抽出...",
)
```

- 各 Process は **同一インスタンス** を ClassVar `contracts` に宣言
  （`ProcessContractRegistry` は `(process_name, contract_name)` を一意キーとする）
- `term_names=("K_mat", "K_geo", "K_st")` は **現時点 3 項** に限定。
  status-351 以降で `K_closest` / `K_hermite_adj` / `K_mat_ndir` を追加予定
- `combinator="add_sub"` は `K_c = K_mat - K_geo + K_st`（符号は呼び出し側処理）

### 共通ヘルパ（脱法実装 pattern 3 防止）

`_extract_kc_active_pair_data(inp)` は 2 Process の重複コード排除のため導入。

- `has_state & ((h_deriv > 1e-30) | (p_n > 1e-30))` でアクティブ集合抽出
- `w_mat = h_deriv_act * k_pen` / `w_geo = p_n / dist`
- Hermite or 線形形状係数、`nn` / `I_nn` / `cc` / `gdofs` を返す
- 各 Process は独立に本ヘルパを呼び出し、他 Process の内部状態を参照しない
  （**pattern 3: wrapper 被せ禁止** の構造的防止）

### `tangent_components()` orchestrator 化（Before / After）

**Before**（status-349 まで）:
```python
def tangent_components(self, u, manager, k_pen, *, node_coords=None):
    # 1000+ 行の monolith: ペア抽出 → アクティブ判定 → 3 項組み立てを
    # 1 メソッド内で逐次計算
    ...
```

**After**（status-350）:
```python
def tangent_components(self, u, manager, k_pen, *, node_coords=None):
    # delta_h / use_hermite / node_counts / adj_node_map 等の前処理のみ担当
    term_input = KcTermAssemblyInput(...)
    K_mat = KcNormalStiffnessProcess().process(term_input).K_mat
    K_geo = KcGeoStiffnessProcess().process(term_input).K_geo
    K_st  = ContactForceStStiffnessProcess().process(
        ContactForceStStiffnessInput(...)
    ).K_st
    return K_mat, K_geo, K_st
```

3-tuple 返却の既存 API は保存（`test_kc_component_fd.py` / `_newton_dynamic.py`
が依存）。呼び出し側の署名・意味は変更なし。

## 検証

### gate 1: 既存 FD テスト無変更 pass

```
xkep_cae/contact/contact_force/tests/test_kc_component_fd.py  ...  7 passed
xkep_cae/verify/tests/test_kc_component_fd.py                 ... 12 passed
合計 19 passed（0 skipped / 0 xfailed / 0 modified）
```

**status-349 の handoff に明記された「12 件無変更 pass」** を遵守
（status-345 の test 追加で 11→12、status-350 時点で 7+12=19）。

### gate 2: 新 Process 単体テスト

```
xkep_cae/contact/contact_force/tests/test_kc_term_processes.py ... 14 passed
```

- `@binds_to` で C3 契約（Process ↔ テスト紐付け）成立
- `TermExpansionContract` 3 Process 全てで宣言確認
- **orchestrator 一致テスト**: `tangent_components()` 出力と Process 直接呼び出しが
  `atol=1e-14` で一致（意味的 wrapper でないことを実数値で担保）

### gate 3: 契約違反 0 件 + ruff + 全テスト

```
$ uv run ruff check xkep_cae/ tests/
All checks passed!

$ uv run ruff format --check xkep_cae/ tests/
191 files already formatted

$ uv run python contracts/validate_process_contracts.py
契約違反なし、条例違反なし

$ uv run python -m pytest xkep_cae/ -q -k "not slow"
1 failed, 762 passed, 10 skipped, 14 deselected, 1 xfailed
（1 failed = status-349 以前からの pre-existing: test_stress_contour.py）
```

### gate 4: 7本撚線 frac=1.0 回帰（82s）

```
xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py ... 18 passed in 82s
（含む test_7strand_90deg_dynamic_completes）
```

## 次セッション引き継ぎ（status-351 向け: Phase C-2）

**開始前に必ず読むファイル**:

1. `/root/.claude/plans/deep-wiggling-seal.md` を**全文読む**（要約禁止）
2. 本 `status-350.md` + `status-349.md`（Phase B-C 継続断面）
3. 計画書「🚫 脱法実装パターン 10 項」を読み返し、status-351 で陥りそうな
   項目（特に **pattern 3: wrapper 被せ**、**pattern 5: skip/xfail**）を自己チェック

### status-351 の目標（Phase C-2: `KcClosestPoint` / `KcHermiteNonlocal` 分離）

計画書 Phase C（status-350〜353）のうち status-351 では:

1. `KcNormalStiffnessProcess` が現在暫定で包含する **K_hermite_adj**（隣接ノード
   mat-only 拡張）を独立 Process `KcHermiteNonlocalStiffnessProcess` に分離
2. `K_closest` 項（最近接点 $(s,t)$ 摂動に伴う $\partial (s,t)/\partial \boldsymbol{u}$
   経由の材料剛性寄与）を `KcClosestPointStiffnessProcess` として新設
3. `TermExpansionContract.term_names` を 3 項 → 5 項に拡張:
   `("K_mat_nn", "K_closest", "K_hermite_adj", "K_geo", "K_st")`
4. `test_kc_component_fd.py` **無変更 pass** を維持（gate）

### status-351 の禁止事項

- ❌ `KcHermiteNonlocalStiffnessProcess` を `KcNormalStiffnessProcess` の
  wrapper だけで済ませる（pattern 3）
- ❌ `K_mat_ndir`（status-352 本命）を本 status に前倒しで混ぜる
- ❌ `test_kc_component_fd.py` を skip/xfail で pass 扱い（pattern 5）

### 成功基準（status-351）

- 2 新 Process 追加、`TermExpansionContract.providers` / `term_names` 5 項化
- `test_kc_component_fd.py` 19 件無変更 pass + 新 Process 単体テスト追加
- 7本撚線 frac=1.0 回帰確認（Gate 4、82s 目安）
- 契約違反 0 件維持、Phase A〜E の **6/11 完了**

## コミット（予定）

```
(次) feat(contact): MCDD Phase C-1 step-1 — KcNormal/KcGeo Process 抽出 + tangent_components orchestrator 化 (status-350)
(次) test(contact): MCDD Phase C-1 step-2 — KcNormal/KcGeo 単体テスト + C15 fixture (status-350)
(次) docs(status): status-350 + README/roadmap/status-index/CLAUDE 更新 (Phase C-1 完了)
```

Plan: `/root/.claude/plans/deep-wiggling-seal.md` (v1.0.0 frozen)
Phase A〜E / status-346〜356 の **5/11 完了**。
