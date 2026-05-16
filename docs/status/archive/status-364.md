# status-364: Phase E C24 — hollow VerifyProcess 構造的封じ込め（脱法 pattern 2 裏口対策）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-23
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12 passed（mathematics +12: C24 tests）

## 概要

status-363 引継ぎ 4. の **Phase E C24 候補**（「`@verified_by`
VerifyProcess の `process()` 内で FD 整合検証が実際に呼ばれているか
AST 検査、MCDD 脱法 pattern 2 裏口対策」）を実装。

既存 `_reject_dummy_process` は `pass` / `...` / 裸 `return` /
`raise NotImplementedError` のみの本体を弾くが、以下のような
**non-trivial だが計算していない** hollow 本体は通過していた:

- `return True` / `return False`（定数 return）
- `return MyOutput(rel_err=0.0, passed=True)`（全引数 constant の
  Output コンストラクタ）
- `val = 1.0 + 2.0; return val`（入力を読まないダミー計算）

本 status で `_reject_hollow_process` runtime guard と
`check_c24_verify_has_computation` 静的検査を追加し、両経路で
hollow verifier を拒否する。本体ソルバー（`xkep_cae/contact/`、
`xkep_cae/solve/`）は無変更、19 本 frac=1.0 未達の本命課題
（候補 (e) 接触減衰 escape hatch）は次 status (status-365) で
着手する。

## 1. 実装

### 1.1 Runtime guard: `_reject_hollow_process`（`xkep_cae/mathematics/registry.py`）

- 新規例外 `HollowVerifyProcessError(DummyVerifyProcessError)`。
  既存の `except DummyVerifyProcessError` ハンドラが変更なしで
  hollow verifier も捕捉できる（下位互換）。
- ヘルパ `_collect_verifier_body_signals(body, param_name)` で
  AST 走査し `{"reads_input": bool, "has_computation": bool}` を収集:
  - `reads_input`: 第1引数名（通常 `input_data`）が `ast.Name` として
    本体内で1回以上参照されるか（Attribute/Subscript の base 経由
    でも `Name` ノードとして出現）
  - `has_computation`: `ast.BinOp` / `ast.Compare` / 非定数引数を
    持つ `ast.Call` のいずれかが存在するか
- `_reject_hollow_process(verify_cls)`:
  docstring 除外後の本体に対し上記 2 シグナルを必須化し、
  満たさない場合 `HollowVerifyProcessError` 送出。
- `bind_verifier` で `_reject_dummy_process` 直後に呼び出し
  （dummy → hollow の順で検査）。
- `__init__.py` で `HollowVerifyProcessError` を公開 API に追加。

### 1.2 静的検査: `check_c24_verify_has_computation`（`contracts/validate_process_contracts.py`）

- `ProcessContractRegistry.all_bindings()` を走査し、同一 verify_cls
  重複紐付けは 1 回だけ検査（`seen_verify: set[type]`）。
- `_extract_process_method_source` で取得したソースを AST parse
  → docstring 除外 → `_collect_verifier_body_signals` 再利用。
- エラーメッセージは紐付け元 `(proc_name, contract_name)` を併記し、
  原因（reads_input / has_computation いずれの欠落か）を区別:
  - `C24: Foo.process() が入力 'input_data' を一度も参照していない`
  - `C24: Foo.process() に計算痕跡（BinOp/Compare/非定数 Call）が存在しない`
- `main()` に登録、ヘッダを `C3-C24` に更新、修正ガイドに C24 行追加。

### 1.3 テスト追加（`xkep_cae/mathematics/tests/test_registry.py`）

12 テスト追加（mathematics/tests 97→109 passed）:

- フィクスチャ 3 種（既存 `_Dummy*` に対応）:
  - `_HollowConstantReturnVerifyProcess` （`return True`）
  - `_HollowConstantArgsOutputVerifyProcess` （全引数 constant の
    `dict(...)` コンストラクタ）
  - `_HollowInputUnreadComputationVerifyProcess` （`1.0 + 2.0`
    BinOp はあるが入力参照なし）
- `TestBindVerifier::test_bind_hollow_rejected[...]` 3 パラメータ化 +
  `test_hollow_error_is_dummy_subclass`
- `TestVerifierBodySignals` 4 テスト（ヘルパ単体検証）
- `TestCheckC24StaticValidator` 4 テスト（静的検査経路検証、
  `_fresh._verifiers` 直挿入で runtime guard をバイパス）
- `TestPackageExports::test_exports` に `HollowVerifyProcessError` 追加

## 2. Gate

- `python contracts/validate_process_contracts.py` → **契約違反 0 件 / 条例違反 0 件（全 24 検査 OK）**
- `python -m pytest xkep_cae/mathematics/tests/ -q` → **109 passed in 0.91s**
- `ruff check xkep_cae/ tests/ contracts/` → All checks passed
- `ruff format --check xkep_cae/ tests/ contracts/` → 既存整形と一致

## 3. MCDD 脱法 pattern 2 の裏口封鎖マップ

| 脱法サブパターン | 既存 guard | 本 status C24 追加 |
|---|---|---|
| `pass` / `...` のみ | `_reject_dummy_process`（status-347） | — |
| 裸 `return` / `return None` / `return ...` | `_reject_dummy_process` | — |
| `raise NotImplementedError` のみ | `_reject_dummy_process` | — |
| `return True` / `return False`（定数） | 通過 | `_reject_hollow_process` で拒否 |
| `return Output(const=0.0, ...)` 全引数 constant | 通過 | `_reject_hollow_process` で拒否 |
| 入力を読まない BinOp のみ | 通過 | `_reject_hollow_process` で拒否 |

Phase E 契約検査は **C18〜C24 の 7 項**（C18: `@verified_by` 紐付け /
C19: providers 実在 / C20: 双方向紐付け / C21: term_names・providers
重複 / C22: contracts 同名重複 / C23: verifier カテゴリ / C24:
verifier 本体計算痕跡）で MCDD 脱法実装禁止パターン 10 項のうち
pattern 2・4 を静的・動的の両面で封じ込め。

## 4. 限界と既知の偽陰性

`_collect_verifier_body_signals` は conservative な AST 検査で、
以下の病理ケースは **C24 では検出できない**（known limitation）:

```python
def process(self, input_data):
    x = input_data.u  # reads_input=True
    y = x             # rebinding only
    return Output(rel_err=y - y)  # BinOp あり（has_computation=True）だが恒等 0
```

- 入力を読み、BinOp も存在するが実質的検証を行っていない。
- 検出するには dataflow 解析または FD 数値的な反駁（e.g. 既知
  mismatching パラメータで `rel_err > tol` を強制する challenge test）
  が必要で、C24 のスコープ外。
- 必要になれば Phase E C25 以降で `VerifyProcess` に
  challenge-test fixture の紐付けを義務化する拡張を検討。

## 5. 次 status 候補（status-365）

status-363 引継ぎ 1. の **候補 (e) 接触減衰 escape hatch** を
最優先に復帰:

1. `ContactNormalDampingProcess`（仮称、`xkep_cae/contact/solver/`）
   新規。`-c_n v_n n̂` 減衰力 + `c_n / dt * I_nn` 接線剛性寄与
   （Generalized-α `γ/(β dt)` 同期）。
2. `StrandBendingOscillationConfig` に
   `contact_damping_coefficient` + `contact_damping_budget_ratio`、
   default OFF。
3. `ContactDampingEnergyMonitorProcess`（仮称）で
   `E_damp = Σ c_n v_n² dt` と `E_strain` 比を 10 step 毎に監査。
4. validation: 7 本撚線 Papailiou 解析解 vs 減衰 1/2/5/10/20%
   → energy budget 許容線決定 → 19 本撚線で budget 内最大減衰探索。

副次は status-363 と同様 (d) 接触凍結モード 19 本 / (f) Phase C-3'
s-tracking 19 本再評価。

## ファイル変更

| ファイル | 変更 |
|---------|------|
| `xkep_cae/mathematics/registry.py` | `HollowVerifyProcessError` 新設、`_collect_verifier_body_signals` / `_reject_hollow_process` 追加、`bind_verifier` に hollow hook 配線、`__all__` 更新 |
| `xkep_cae/mathematics/__init__.py` | `HollowVerifyProcessError` を公開 API に追加 |
| `contracts/validate_process_contracts.py` | `check_c24_verify_has_computation` 追加、`main()` に登録、ヘッダ C3-C24 化、修正ガイド C24 行追加 |
| `xkep_cae/mathematics/tests/test_registry.py` | hollow フィクスチャ 3 種 + C24 テスト 12 件 + exports 期待値追加 |
| `docs/status/status-364.md` | **新規**: 本ファイル |
| `docs/status/status-index.md` | status-364 行追加 |
| `README.md` | 現在状況に C24 追記 |
| `docs/roadmap.md` | Phase E C24 完了行追記 |

## 引継ぎ（status-365 へ）

1. **最優先**: 候補 (e) 接触減衰 escape hatch 実装（status-363 §4 計画）
2. **副次**: 候補 (d) 接触凍結モード 19 本適用
3. **最終手段**: 候補 (f) Phase C-3' s-tracking 経路の 19 本再評価
4. **Phase E C25 候補**: VerifyProcess の challenge-test fixture
   紐付け義務化（本 status §4 で提示した偽陰性パターンへの対策）
5. **`Strand3DContourProcess` 統合**（status-362 から継続）
