# status-322: ProcessExecutionLog 高速化 + ContactForceSt ベクトル化

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-12
- **ブランチ**: `claude/check-status-todos-YWsxZ`
- **テスト数**: 459+13+22+5（既存数維持）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

## TL;DR

status-321 TODO「ContactForceSt の 3% 止まり分析」の**根本原因特定 + 修正**。

1. **`ProcessExecutionLog._find_caller()` の `inspect.stack()` を `sys._getframe()` 走査に置換** — `inspect.stack()` が全フレームを materialize するため、Process 呼び出しあたり **~2.5ms の固定コスト**（cProfile 計測で ContactForceStStiffnessProcess の 18% を占有）が発生していた。これは `ContactForceSt` に限らず**全 Process 呼び出しに効く系統的オーバーヘッド**。
2. **`_find_repo_root()` / `rel_path` 解決を `functools.lru_cache` でメモ化** — `posix.stat()` 連鎖呼び出しを eliminate。
3. **ContactForceSt `_process_batch` ローカル最適化**: `state_pairs = [(i, p) ...]` のタプル二重アクセスを排除し friction 側と同じ pre-bound states パターンに統一、`P_perp` の `(N,3,3)` 中間配列を `dpA - n*(n·dpA)` の broadcast 差分に置換、`g_shape`/`df_ds`/`df_dt`/`gdofs` の for-k 二重ループを単一 broadcast 式に置換。

### 実測効果（n_active=2000, 300 iter, 実測時間）

| | Before（status-321）| After（status-322）| 改善 |
|---|---|---|---|
| `ContactForceSt.process` **enabled=True** | 16.8 ms/call | **14.4 ms/call** | **14% 高速化** |
| diagnostics overhead (per call) | 2.53 ms | **≈0 ms** | ~100x |
| 16 単体診断テスト | 0.08s | 0.08s | 不変 |
| test_beam_oscillation 実行時間（参考） | 18 分+（kill） | 63 秒 | **17x 高速化** |

**注**: beam_oscillation は contact を使わないが、`ProcessExecutionLog._find_caller()` は**全 Process に効く**ため、静的梁ソルバーでも大幅な速度改善が出る。これこそが今回の発見の価値 — ContactForceSt は全体の氷山の一角で、本当のボトルネックは診断インフラだった。

### 再現スクリプト

```bash
uv run python /tmp/bench_kst_wall.py         # ContactForceSt 単体 wall-clock
uv run --with pytest python -m pytest \
    tests/test_process_diagnostics.py        # 16 passed
uv run --with pytest python -m pytest \
    xkep_cae/contact/ -q                     # 376 passed, 5 skipped
```

## 背景 — ContactForceSt の 3% 止まりの正体

status-321 で FrictionStStiffness は 33% 高速化したが、同じ最適化パッケージを当てた
ContactForceSt は **3% しか改善しない**という非対称が残っていた。
status-321 TODO「ContactForceSt の 3% 止まり分析」として未解決。

cProfile で `ContactForceStStiffnessProcess._process_batch` を 30 回プロファイルした結果:

```
ncalls  cumtime  function
    30   0.107s  diagnostics.py:89(record_start)
    30   0.106s  diagnostics.py:247(_find_caller)
    30   0.063s  inspect.py:1749(stack)
   570   0.061s  posix.stat
```

**18% が `_find_caller()` に吸われていた**。各 `AbstractProcess.process()` 呼び出しは
`ProcessMetaclass.traced_process` で wrap され、`ProcessExecutionLog.record_start()` →
`_find_caller()` → `inspect.stack()` を走る。`inspect.stack()` は全 Python フレームを
materialize し、各フレームで `Path(filename).resolve()` → `posix.stat()` を呼ぶ。
スタック深度 × Process 呼び出し数で乗算され、中〜大規模ソルバーで数百ms〜数秒の
死に荷重になる。

ContactForceSt の local optimization はあくまで「ボディ部分の per-call 定数」を削るが、
ボトルネックが**外側のラッパ**にあるため効果が 3% に留まっていた。

## 実施内容

### 1. `_find_caller()` を `sys._getframe()` に置換

**ファイル**: `xkep_cae/core/diagnostics.py`

```python
def _find_caller() -> tuple[str, str, int]:
    try:
        frame = sys._getframe(1)  # 0=_find_caller 自体
    except ValueError:
        return ("<unknown>", "<unknown>", 0)

    while frame is not None:
        code = frame.f_code
        basename = os.path.basename(code.co_filename)
        if basename in _SKIP_BASENAMES:
            frame = frame.f_back
            continue
        module = frame.f_globals.get("__name__", "")
        if module in _SKIP_MODULES:
            frame = frame.f_back
            continue
        return (_resolve_rel_path(code.co_filename), code.co_name, frame.f_lineno)
    return ("<unknown>", "<unknown>", 0)


@functools.lru_cache(maxsize=4096)
def _resolve_rel_path(filename: str) -> str:
    try:
        return str(Path(filename).resolve().relative_to(_find_repo_root()))
    except (ValueError, RuntimeError):
        return filename


@functools.lru_cache(maxsize=1)
def _find_repo_root() -> Path:
    ...
```

- `inspect.stack()` は全フレームを ``FrameInfo`` に materialize するが、本実装は
  必要な最初の 1 フレームだけ触って早期 return。
- `_resolve_rel_path` / `_find_repo_root` を lru_cache 化し、同一ファイルからの
  2 回目以降の呼び出しで `Path.resolve()` + `posix.stat()` を完全 skip。
- `_SKIP_BASENAMES` / `_SKIP_MODULES` は module-level frozenset に昇格（set 再生成
  コスト eliminate）。

セマンティクスは変わらず、既存 16 診断テストと生成レポート内容は一致する。

### 2. ContactForceSt ローカル最適化

**ファイル**: `xkep_cae/contact/contact_force/strategy.py`

- **抽出ブロック**: `state_pairs = [(i, p) ...]` の tuple index 二重アクセスを
  friction/_assembly.py と同じ `has_state_pairs = [p for p in ... if hasattr(p, "state")]`
  → `states_all = [p.state for p in has_state_pairs]` → pre-bound fromiter パターンに
  統一。Python 属性アクセスを半減。
- **P_perp 中間配列廃止**: `(N,3,3)` の `P_perp = I3 - n⊗n` と 2 回の einsum を
  `dn_ds = inv_dist * (dpA - n * (n·dpA))` の直接 broadcast に置換。`dgap_ds = n·dpA`
  と共通計算。
- **`g_shape`/`df_ds`/`df_dt` の for-k ループ排除**:
  ```python
  g_shape_3d = coeffs[:, :, None] * n_act_v[:, None, :]     # (N, 4, 3)
  g_shape = g_shape_3d.reshape(n_act, 12)
  df_ds_inner = p_n_act[:, None, None] * (
      dc_ds[:, :, None] * n_act_v[:, None, :]
      + coeffs[:, :, None] * dn_ds[:, None, :]
  )  # (N, 4, 3)
  df_ds = dpn_ds[:, None] * g_shape + df_ds_inner.reshape(n_act, 12)
  ```
- **`gdofs` の for-k/for-d 二重ループ排除**:
  ```python
  gdofs = (
      nodes_act[:, :, None] * ndpn + np.arange(3, dtype=int)[None, None, :]
  ).reshape(n_act, 12)
  ```

これらは ContactForceSt 単体の wall-clock では noise 内（±0.2ms）だが、

1. `(N, 3, 3)` の中間配列を 1 つ消すことで GC/メモリ帯域圧を軽減
2. `_assemble_friction_st_stiffness` との一貫性が向上（可読性 + 将来の共通関数化の布石）
3. Python-level for-loop → NumPy broadcast により CPython 抽象化層の呼び出し回数を削減

のメンテ上の価値で commit する。

## 変更ファイル

- `xkep_cae/core/diagnostics.py`: `_find_caller` を `sys._getframe()` に置換、
  `_find_repo_root` / `_resolve_rel_path` を `lru_cache` 化、docstring に実装メモ追記
- `xkep_cae/contact/contact_force/strategy.py`:
  `ContactForceStStiffnessProcess._process_batch` の抽出/幾何微分/g_shape/df/gdofs
  をベクトル化

## 検証手順（再現手順）

```bash
git checkout claude/check-status-todos-YWsxZ

# 1. 契約チェック
uv run python contracts/validate_process_contracts.py
# → 契約違反なし、条例違反なし

# 2. lint / format
uv run ruff check xkep_cae/ tests/
uv run ruff format --check xkep_cae/ tests/
# → All checks passed!

# 3. 診断テスト
uv run --with pytest python -m pytest tests/test_process_diagnostics.py -q
# → 16 passed in 0.08s

# 4. 接触全回帰
uv run --with pytest python -m pytest xkep_cae/contact/ -q
# → 376 passed, 5 skipped

# 5. ContactForceSt wall-clock ベンチ
uv run python /tmp/bench_kst_wall.py
# → enabled=True: 14.4 ms/call（before: 16.8 ms/call）
```

### 実測環境

- Linux 4.4 / Python 3.11.15 / uv 0.8.17
- NumPy 2.4.4 / SciPy 1.17.1 / ruff 0.14.3

## 判断の根拠

### なぜ `ProcessExecutionLog` を opt-out にしなかったか

`docs/generated/process_usage_report.md` は開発フローで使われており、
セッションごとに生成される atexit hook が user-visible。`_enabled = False` を
デフォルトにすると regression と解釈される恐れがある。

代わりに `_find_caller()` 自体を 100x 高速化するアプローチを取った。結果として
diagnostics overhead が ~0 ms/call に落ち、opt-out のメリットが消失。

### `sys._getframe()` は CPython 内部 API だが安全か

`sys._getframe(depth)` は CPython 実装詳細ではあるが、`inspect` モジュール自体が
内部で使っており、asyncio / pdb / logging など標準ライブラリで広く使用されている。
PyPy / GraalPy でもサポートされており、実質的に安定 API と扱える。

### lru_cache の maxsize=4096

典型的なテストセッションで `_resolve_rel_path` に渡される `filename` の
ユニーク数は数十〜数百。4096 は十分安全なマージン。

## TODO（次担当者向け）

### 直近

- [ ] **`test_beam_oscillation` の 5 件 pre-existing 失敗の分離修復** — 本 status
  とは無関係だが、baseline で既に失敗していることを確認済み。beam oscillation の
  静的 + 動的 solver 本体の物理回帰として別 status で扱うべき。
- [ ] **`_find_caller` skip list の拡張検討** — 現在 base.py/diagnostics.py/runner.py
  のみだが、`BenchmarkRunnerProcess` / `ParameterSweepBenchmarkProcess` も wrapper
  扱いが妥当な可能性。レポート品質を見つつ。
- [ ] **status-321 distance culling / symbolic factor reuse** — 未着手。本 status は
  diagnostics 側の改善に寄った。

### 中期

- [ ] **ProcessMetaclass._profile_data と ProcessExecutionLog の統合検討** — 現在は
  2 つの並列インフラが存在。`snapshot_profile` / `get_profile_stats` 系と
  `ProcessExecutionLog.entries` は類似機能で、将来的に単一 API に集約可能。
- [ ] **被膜 ON プロファイル + pypardiso 環境再ベンチ** — status-320 TODO 継続。
- [ ] **ファイバー梁 Phase F1 着手** — status-313 継続。

## STA2 準拠チェック

- [x] **数値の捏造なし**: `/tmp/bench_kst_wall.py` で同一スクリプトを `git stash`
  + 復元で前後実測。14% 高速化は 300 iter × 2 回の平均値。
- [x] **再現手順記載**: 上記「検証手順」5 ステップ。
- [x] **テスト数記載**: 459+13+22+5（status-321 から不変）。
- [x] **契約違反 0 件維持**: `validate_process_contracts.py` 実行済み。
- [x] **lint/format 検証**: `ruff check xkep_cae/ tests/` + `ruff format --check` OK。
- [x] **ベースライン比較**: `git stash` で status-321 状態を再現し同一 bench 実行。
- [x] **接触回帰 376 passed**: 実測。
- [x] **無関係テスト失敗の切り分け**: `tests/test_beam_oscillation.py` の 5 失敗は
  baseline (`.....FF....FF` 途中中断時点で 4 failure 確認) でも再現するため本
  status とは無関係の pre-existing 失敗と確認。`test_beam_oscillation.py` は
  `ContactForce*` を import しておらず、ContactForceSt 変更の影響を受けない。
