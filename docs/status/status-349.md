# status-349: MCDD Phase B-2 — 数理台帳 6 章完備 + `equation_index.py` + C15 拡張

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-17
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+**21**+**8**（Phase B-2 で **+29**）

## 概要

status-348（Phase B-1: `docs/math/03_huber_contact_penalty.md`）に続き、
**MCDD Phase B-2** として以下 3 つを一括完了:

1. **残り 5 章の離散化方程式台帳整備**（01 梁運動学 / 02 接触幾何 /
   04 smooth penalty 摩擦 / 05 バリア関数被膜 / 06 Generalized-α 時間積分）
   → **計 6 章 / 55 アンカー** の台帳を完成
2. **`xkep_cae/mathematics/equation_index.py` 新設**（`<a id="...">` 抽出 +
   `MathematicalContract.equation_ref` 参照解決 API、21 テスト）
3. **`contracts/validate_process_contracts.py` の C15 拡張**
   （`check_c15_equation_refs`、台帳空・重複・未解決参照を検出、8 テスト）

設計計画: `/root/.claude/plans/deep-wiggling-seal.md`（v1.0.0 凍結）
— Phase A〜E / status-346〜356 の **4/11 完了**。他ロードマップ項目は凍結。

## 成果物

### 新規ファイル

| ファイル | 行数 | 内容 |
|---------|-----|------|
| `docs/math/01_kinematics_beam.md` | 123 | CR / TL / UL 梁運動学 + Hermite 補間（8 アンカー） |
| `docs/math/02_contact_geometry.md` | 137 | 最近接点 $(s,t)$ 射影 + StJacobian（8 アンカー） |
| `docs/math/04_friction_smooth_penalty.md` | 158 | Coulomb return mapping + smooth penalty + $\boldsymbol{K}_t$ 分解（8 アンカー） |
| `docs/math/05_coating_barrier.md` | 124 | バリア関数 $f = k\delta/(1-\delta/\delta_{\max})$ + 線形フォールバック（5 アンカー） |
| `docs/math/06_time_integration.md` | 143 | Generalized-α / Newmark + 擬似時間（7 アンカー） |
| `xkep_cae/mathematics/equation_index.py` | 263 | 台帳スキャン + アンカー抽出 + `EquationIndex` レジストリ（frozen dataclass） |
| `xkep_cae/mathematics/tests/test_equation_index.py` | 294 | 21 テスト（抽出 / 重複検出 / 参照解決 / エッジケース / 実台帳） |
| `xkep_cae/mathematics/tests/test_c15_extension.py` | 256 | 8 テスト（C15 拡張の正常 2 + 異常 5 + docstring 除外 1） |

### 変更ファイル

| ファイル | 内容 |
|---------|------|
| `docs/math/README.md` | 6 章全てを ✅ status-349 に更新、status 参照を 349 に |
| `xkep_cae/mathematics/__init__.py` | `EquationIndex` / `DuplicateAnchorError` / `UnresolvedReferenceError` / `load_equation_index` をエクスポート |
| `contracts/validate_process_contracts.py` | `check_c15_equation_refs()` 新設 + `main()` に統合、モジュール docstring 更新 |
| `README.md` / `docs/roadmap.md` / `docs/status/status-index.md` / `CLAUDE.md` | Phase 進捗 3/11 → 4/11、テスト数 +29、現在地 2026-04-17 |

## 設計

### `equation_index.py` の API

```python
from xkep_cae.mathematics import load_equation_index

idx = load_equation_index()  # docs/math/ を自動探索
# ── ディスクスキャン結果 ──
idx.total_anchors           # 55
idx.is_unique               # True（全ファイルで重複なし）
idx.anchors_by_file         # {'03_huber_contact_penalty.md': frozenset({'eq-kc', ...}), ...}

# ── 単発参照解決 ──
err = idx.resolve("03_huber_contact_penalty.md#eq-kc")  # → None (OK)
err = idx.resolve("99_nowhere.md#eq-foo")               # → UnresolvedReferenceError(reason='missing_file')
err = idx.resolve("03_huber_contact_penalty.md#eq-ghost")  # → reason='missing_anchor'
err = idx.resolve("bad_format_no_hash")                 # → reason='bad_format'

# ── 一括解決 ──
errs = idx.validate(["03_huber_contact_penalty.md#eq-kc", "99_nowhere.md#inv-foo"])
# → [UnresolvedReferenceError(..., reason='missing_file')]
```

**設計原則**:
- **循環 import 回避**: `equation_index.py` は `contracts.py` に依存しない
  （`equation_ref` 文字列の形式のみ扱う）
- **frozen dataclass**: `EquationIndex` / `DuplicateAnchorError` /
  `UnresolvedReferenceError` 全て `frozen=True`（C17 整合）
- **READMEは索引**: アンカー抽出対象外（索引ファイルとして除外）
- **テスト容易性**: `load(ledger_root)` で差し替え可能、デフォルトは cwd から
  親方向に `docs/math/README.md` を探索

### C15 拡張の検査ロジック

`contracts/validate_process_contracts.py::check_c15_equation_refs()` の責務:

1. `load_equation_index()` で台帳を読み込み
2. 台帳空なら `C15(math): docs/math/ 台帳が空` として単一エラー計上
3. 台帳内アンカー重複（`idx.duplicates`）を `C15(math): 台帳アンカー重複` として計上
4. 実行時 Process から `equation_ref` を二経路で収集:
   - `ProcessContractRegistry.all_bindings` 経由（`@verified_by` 紐付け）
   - `AbstractProcess.contracts` ClassVar 経由（`ProcessMeta` 自動合算）
5. 各 `equation_ref` を `idx.resolve()` で解決、未解決分を
   `C15(math): 未解決参照 ... reason=<bad_format|missing_file|missing_anchor>` として計上

**docstring 例題の自然除外**:
- `contracts.py` docstring 内の `IdentityContract(equation_ref="...")` 疑似コードは
  ClassVar に登録されないため `getattr(cls, "contracts", ())` で収集されない
- AST 検査を要せず、「ランタイム登録のみを見る」設計で混入を構造的に防止

### アンカー一意性と章立て（完成形）

| # | ファイル | アンカー数 | 主要項目 |
|---|---------|----------|---------|
| 01 | `01_kinematics_beam.md` | 8 | CR frame / TL Green-Lagrange / UL objective rate / Hermite |
| 02 | `02_contact_geometry.md` | 8 | 最近接点射影 / gap / $\partial \boldsymbol{r}/\partial \boldsymbol{u}$ / StJacobian |
| 03 | `03_huber_contact_penalty.md` | 19 | Huber/Hertz $p_n$ / $\boldsymbol{f}_c$ / $\boldsymbol{K}_c$ 6 項分解 |
| 04 | `04_friction_smooth_penalty.md` | 8 | Coulomb / return mapping / $\boldsymbol{K}_t$ 3 項分解 |
| 05 | `05_coating_barrier.md` | 5 | バリア関数 / 有効剛性 / エネルギー / 線形フォールバック |
| 06 | `06_time_integration.md` | 7 | Chung-Hulbert / Newmark predictor/corrector / 擬似時間 |
| **計** | **6 章** | **55** | **全て一意**（機械検証済み） |

## 脱法実装防止（計画書「🚫 10 項」）の本 status での自己点検

| Pattern | 本 status での回避策 |
|---|---|
| **6**: 困難の先送り（骨格 status） | 6 章全てで `## 表記 / ## 主要方程式 / ## 既存実装との trace` の 3 節を完成。未実装項（`eq-dn-du` / `eq-closest-residual` の逐次線形化）は trace 表で明示、status-352 以降で C19 `term_processes` 実在検査として回収可能にする |
| **10**: 「TODO として積む」で次回送り | Phase B-2 の成功基準（6 章 + `equation_index.py` + C15 拡張 + 単体テスト）を全て本 status 内で完了。Phase C 以降のスコープも章立てで予告済み |
| **3**: wrapper 被せ分解 | `equation_index.py` は 263 行で抽出 / 重複検出 / 参照解決の実ロジックを自己完結実装（既存 API の wrapper ではない）。C15 拡張も `check_c15_equation_refs()` 内で直接 `idx.resolve()` を呼ぶ |
| **5**: skip/xfail で pass | 既存 95 tests（mathematics 配下）全 pass、skip/xfail 増加 0。新規 29 tests も全 pass |
| **7**: 診断 report の精度バグ | 台帳には**数値を一切書かない**ルールを 6 章で徹底（`docs/math/README.md` 整合性ルール 4）。実測値は status 側に隔離 |
| **9**: `tuple` → `list` で frozen 回避 | `EquationIndex.duplicates: tuple[DuplicateAnchorError, ...]` を維持、`anchors_by_file: dict[str, frozenset[str]]` で可変性封印 |

## 検証・品質確認（4-Gate 全 pass）

### Gate 1: テスト

```
uv run pytest xkep_cae/mathematics/tests/ --no-header
============================== 95 passed in 0.81s ==============================
```

新規 29 テスト（`test_equation_index.py` 21 + `test_c15_extension.py` 8）全 pass。
既存 66 テスト（contracts 33 + registry 33）回帰なし。

### Gate 2: 契約検査

```
uv run python contracts/validate_process_contracts.py
============================================================
契約違反なし、条例違反なし
```

C3〜C17 + O1〜O3 + **新 C15(math) 拡張** で全 16 検査項目クリア。
`mathematics/` 除外（status-347）と `docs/math/` 新設は契約検査の対象外。

### Gate 3: ruff check / format

```
uv run ruff check xkep_cae/mathematics/ contracts/
All checks passed!
```

### Gate 4（7本撚線回帰）: 省略

Phase B-2 は台帳整備 + 参照解決 API 追加のみで、ソルバー数値パス変更ゼロ。
計画書で Gate 4 は Phase C 以降のみ必須。

### アンカー一意性検証

```
total: 55 unique: True files: 6
 01_kinematics_beam.md: 8 anchors
 02_contact_geometry.md: 8 anchors
 03_huber_contact_penalty.md: 19 anchors
 04_friction_smooth_penalty.md: 8 anchors
 05_coating_barrier.md: 5 anchors
 06_time_integration.md: 7 anchors
```

重複ゼロ、全 55 アンカーが `equation_index.py` で機械解決可能。

## 次セッション引き継ぎ（status-350 向け: Phase C-1）

**開始前に必ず読むファイル**:

1. `/root/.claude/plans/deep-wiggling-seal.md` を**全文読む**（要約禁止）
2. 本 `status-349.md` + `status-348.md` + `status-347.md`（Phase A-B 継続断面）
3. 計画書「🚫 脱法実装パターン 10 項」を読み返し、status-350 で陥りそうな
   項目（特に **pattern 3: wrapper 被せ**、**pattern 5: skip/xfail**）を自己チェック

### status-350 の目標（Phase C-1: `KcNormal` / `KcGeo` / `KcSt` 項別 Process 抽出）

計画書 Phase C（status-350〜353）のうち status-350 では:

1. `ContactForceStrategy.tangent_components()` の **既存 3 項**
   （`K_mat_nn` / `K_geo` / `K_st`）を独立 Process に分離:
   - `KcNormalStiffnessProcess`（`K_mat_nn` = $-\partial p_n/\partial \boldsymbol{u} \otimes \hat{\boldsymbol{n}}$）
   - `KcGeoStiffnessProcess`（`K_geo`、既存項を Process 化）
   - `KcStStiffnessProcess`（既存 `ContactForceStStiffnessProcess` の薄い wrapper ではなく、**`term_processes` 登録可能な形**に再整備）
2. 各 Process に `TermExpansionContract` を `contracts` ClassVar 宣言
3. 既存 `test_kc_component_fd.py` 12 件を**無変更で pass させる**（gate）

### status-350 の禁止事項

- ❌ `KcNormalStiffnessProcess` を既存関数の wrapper だけで済ませる（pattern 3）
- ❌ `test_kc_component_fd.py` を skip/xfail で pass 扱い（pattern 5）
- ❌ `K_mat_ndir`（status-352 本命）を本 status に前倒しで混ぜる

### 成功基準（status-350）

- 3 Process 新設、`TermExpansionContract.providers` で名称一致
- `test_kc_component_fd.py` 12 件無変更 pass + 新 Process 単体テスト追加
- 7本撚線 frac=1.0 回帰確認（Gate 4）
- 契約違反 0 件維持、Phase A〜E の **5/11 完了**

## コミット（実施済み）

```
4fc4687  docs(math): MCDD Phase B-2 step-1 — 残り 5 章の離散化方程式台帳 (status-349)
7882474  feat(math): MCDD Phase B-2 step-2 — equation_index.py 新設 (status-349)
84e34b8  feat(contracts): MCDD Phase B-2 step-3 — C15 拡張で MathematicalContract.equation_ref 解決 (status-349)
(次)     docs(status): status-349 + README/roadmap/status-index 更新 (Phase B-2 完了)
```

Plan: `/root/.claude/plans/deep-wiggling-seal.md` (v1.0.0 frozen)
Phase A〜E / status-346〜356 の **4/11 完了**。
