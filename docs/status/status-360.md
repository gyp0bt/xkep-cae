# status-360: 仮説 C 候補 (a') 19本撚線検証で却下 + Phase E C21/C22/C23 追加

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-22
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25（+C21/C23 で mathematics/tests は 97 件に増加）

## 概要

status-359 の最優先 TODO 2 件を実施:

1. **仮説 C 候補 (a') `smoothing_delta=1000` の 19 本撚線検証** — status-359 で
   7 本撚線 90° 曲げにて `frac=1.0` 完走 + `elapsed -42.5%` を達成した設定を
   19 本撚線 (Type D stall 本体) に適用。**結果: `frac=0.3723` で却下**
   （baseline `frac=0.4839` 比 **-23.1% 退化**）。`StrandBendingOscillationConfig.
   smoothing_delta` の default 変更 (2000→1000) は**実施せず**、7 本のみの
   最適値として記録。

2. **MCDD Phase E C21 / C22 / C23 追加** — status-358 引継ぎの残 Phase E 候補
   を実装:
   - **C21**: `TermExpansionContract.term_names` / `providers` 重複静的検出
     （`__post_init__` に `term_names` 重複チェック追加 + 静的検査）。
   - **C22**: `contracts` ClassVar 同名契約重複検出（`ProcessContractRegistry.
     register_contracts` の動的検査を静的検査で先行）。
   - **C23**: `@verified_by` 検証 Process カテゴリ必須化（`SolverProcess` /
     `VerifyProcess` いずれかの継承必須）。`bind_verifier` + 静的検査。

## 1. 仮説 C 候補 (a') 19 本撚線検証

### 実装

`work/beam_hysteresis/16_hypothesis_c_aprime_19strand.py` を新規追加。
構成は `10_kcr_measurement_19strand.py` と同一、`smoothing_delta=1000.0` のみ
差し替え。ContactPairAnalysisProcess + ContactPairLayerClassifierProcess の
層別 κ_cr 統計も同時実行。

### 実測結果

```bash
uv run python work/beam_hysteresis/16_hypothesis_c_aprime_19strand.py 2>&1 \
    | tee /tmp/hypothesis_c_aprime_19strand_<ts>.log
```

| 指標 | baseline (smoothing_delta=2000、status-339) | **候補 (a') (=1000)** |
|------|------|------|
| frac_completed | 0.4839 | **0.3723** ❌ |
| converged | False | False |
| n_increments | 271 | 164 |
| n_cutbacks | 39 | 23 |
| elapsed [s] | 534.68 | 365.29 |
| frac 変化 | — | **-23.1%（退化）** |

NR 内訳（Incr 164 不収束時）: `D+E:72%`（直近 10 回 `D+E:10`）、
`R_c=3.08e-05, active=35, sliding=375`。δ_h を 2x 広げると Huber 遷移帯の
接触力精度が低下し、**7 本では有効だった active flip 抑制効果は 19 本では
K_c 不整合の増幅に負ける**。

層別 κ_cr（参考、早期停滞下）: `(0,1)=5.15e-3`, `(1,1)=4.67e-3`,
`(1,2)=3.39e-3`, `(2,2)=5.22e-3` — ペア数 51/スリップ 46、CV=0.285。
status-339 (46/57) の傾向と整合。

### 判定: **却下**

- frac -23.1% は status-358 候補 (a) と同パターン（解析の早期打切り）。
- ユーザー指示「frac=1.0 完走 + 10% 以上改善」の完走条件を満たさない。
- **`StrandBendingOscillationConfig.smoothing_delta` の default 変更は
  実施しない**。7 本のみの最適値として `15_hypothesis_c_7strand.py` に
  記録済み、本 status は 19 本での失敗実験を `16_hypothesis_c_aprime_19strand.py`
  に記録（status-358 candidate (a) revert と対称方針）。

実装本体（`xkep_cae/contact/`、`StrandBendingOscillationConfig`）は**無変更**。

### 次候補（status-361）

- **(c) line search 強化**: NR 反復途中で接触残差が増加する step を
  backtracking line search で reject。`_newton_dynamic.py` に line search
  hook 追加（`LineSearchUpdateProcess` の接触感知拡張として実装）。
- 19 本 Type D stall は active 集合振動支配領域で、δ_h 調整よりも
  各反復の step size 制御が本質的対策。

## 2. MCDD Phase E C21 / C22 / C23 追加

### C21: TermExpansionContract の項名 / providers 重複静的検出

`TermExpansionContract.__post_init__` に `term_names` 重複チェックを追加
（既存の providers 重複チェックと対称）。項名の重複は合計検証 $\Sigma K_k$ の
参照同一性を崩すため構造的に排除する。

追加の静的検査 `check_c21_term_expansion_no_duplicates()` は `ProcessRegistry`
走査で term_names / providers の重複を再検出する二重防御線
（`object.__setattr__` 等の frozen 回避経路対策）。

### C22: contracts ClassVar 同名契約重複検出

`ProcessContractRegistry.register_contracts` は登録時に同名契約の重複を
動的検査しているが、`check_c22_contracts_no_duplicate_names()` として
静的検査に昇格。`AbstractProcess.__init_subclass__` による自動登録前に
検出可能。

### C23: @verified_by 検証 Process カテゴリ検査

`ProcessContractRegistry.bind_verifier` に `SolverProcess` / `VerifyProcess`
カテゴリ必須チェックを追加（既存の `AbstractProcess` サブクラスチェック直後）。
PreProcess / PostProcess / BatchProcess / CompatibilityProcess を verifier に
指定する脱法を排除。

静的検査 `check_c23_verifier_category()` は `all_bindings()` スナップショットを
走査して二重防御。現行唯一の verifier `ContactKcComponentFDDiagnosticProcess`
は `SolverProcess` 継承で OK。

### テスト追加

- `test_contracts.py`: `test_duplicate_term_names_rejected` 1 件追加
  （C21 runtime ガード）。
- `test_registry.py`: `_InvalidCategoryVerifier(PreProcess[...])` フィクスチャ +
  `test_bind_invalid_category_rejected` 1 件追加（C23 runtime ガード）。

### gate

```
$ uv run python contracts/validate_process_contracts.py
...
--- C21: TermExpansionContract 項名/providers 重複 --- OK
--- C22: contracts ClassVar 同名契約重複 --- OK
--- C23: @verified_by 検証器カテゴリ --- OK
...
契約違反なし、条例違反なし  (C18-C23 を含む全 23 検査 OK)
```

## 3. 回帰確認

| 項目 | 結果 |
|------|------|
| `python -m pytest xkep_cae/mathematics/tests/` | **97 passed** (+2 新規) |
| `python -m pytest xkep_cae/contact/` | **421 passed, 5 skipped** |
| `python -m pytest tests/numerical_tests/test_strand_bending_convergence.py -k oscillation_converges` | **1 passed** |
| `contracts/validate_process_contracts.py` | 契約違反 0 / 条例違反 0（23 検査 OK） |
| `ruff format --check` / `ruff check` | pass |

## Phase A〜E 進捗更新

Phase A〜E / status-346〜 の **12/N 完了**（status-360 で C21/C22/C23 追加
+ 仮説 C (a') 19 本却下記録）。

- [x] Phase A-1〜A-2（status-346〜347）
- [x] Phase B-1〜B-2（status-348〜349）
- [x] Phase C-1〜C-2（status-350〜351）
- [x] 数理台帳訂正（status-353）
- [x] Phase C-3 仮説 A 実験（status-354）
- [x] Phase C-3' 診断〜実装（status-355〜356）
- [x] 19 本 FD 再計測 + C18/C19（status-357）
- [x] C20 + 仮説 C 候補 (a) 反証（status-358）
- [x] 仮説 C 候補 (a') 7 本採択記録（status-359）
- [x] **status-360（本 status）**: 仮説 C (a') 19 本却下 + Phase E C21/C22/C23
- [ ] status-361: 仮説 C 候補 (c) line search 強化 or 次 Phase E 候補

## 引継ぎ（status-361 へ）

### 最優先 TODO

1. **仮説 C 候補 (c) line search 強化** — status-360 で候補 (a') 19 本却下。
   次手は NR 反復途中の active flip を backtracking line search で抑制。
   `_newton_dynamic.py` の NR 反復ループに `LineSearchUpdateProcess` hook を
   追加し、ペナルティ残差または `||R_t||` が増加する step を reject。
   - 実装規模: 中（NR ループ + 接触残差評価の再配線が必要）
   - 合否基準: 19 本撚線 frac=1.0 完走 + 7 本撚線 回帰なし

2. **仮説 C 候補 (d) 接触凍結モードの 19 本適用**（(c) で効果薄の場合）:
   status-284 で 7 本 frac 0.40→0.70 を達成した接触凍結モードを 19 本に
   適用。接触パラメータのチューニングが必要。

3. **Phase E 仕上げ候補の追加**:
   - C24: `@verified_by` の VerifyProcess `process()` 内で実際に
     `compute_residual` / `FDConsistencyContract` 要求項目が呼ばれているか
     AST 検査（脱法 pattern 2 の裏口対策）。
   - C25: `equation_ref` 台帳アンカーの「セクション重複定義」検出強化。

### 凍結中の TODO（MCDD 完了まで再開禁止）

Phase E 完成 + 19 本 frac=1.0 完走 + `mat_only` rel_err < 1e-2 を満たした
時点で以下を再開可能:

- 7 本撚線ピッチ依存性検証（p=50/100/200）
- ファイバー梁キャリブレーション
- リスタート解析方式
- 被膜圧縮モデル改善（バリア関数 / 二層モデル）
- 空間ブロック分離 / ペアクラスタリング

## ファイル変更

| ファイル | 変更内容 |
|---------|---------|
| `work/beam_hysteresis/16_hypothesis_c_aprime_19strand.py` | **新規**: 仮説 C (a') 19 本撚線検証スクリプト（**失敗実験記録**として残置） |
| `xkep_cae/mathematics/contracts.py` | `TermExpansionContract.__post_init__` に `term_names` 重複チェック追加（C21） |
| `xkep_cae/mathematics/registry.py` | `bind_verifier` に `SolverProcess` / `VerifyProcess` カテゴリ継承必須チェック追加（C23） |
| `xkep_cae/mathematics/tests/test_contracts.py` | `test_duplicate_term_names_rejected` 1 件追加（C21 ガード） |
| `xkep_cae/mathematics/tests/test_registry.py` | `_InvalidCategoryVerifier` フィクスチャ + `test_bind_invalid_category_rejected` 1 件追加（C23 ガード） |
| `contracts/validate_process_contracts.py` | `check_c21_` / `check_c22_` / `check_c23_` 3 関数追加、主ループ + 修正ガイド + header docstring に C21/C22/C23 追記、タイトル更新 |
| `docs/status/status-360.md` | **新規**: 本ファイル |
| `docs/status/status-index.md` | status-360 行追加 |
| `docs/roadmap.md` | 仮説 C (a') 19 本却下記録 + Phase E C21/C22/C23 追加 + status-361 次手 |
| `README.md` | 現在状態を status-360 に更新 |
| `CLAUDE.md` | 現在状態・次の課題を status-360 基準に更新 |

## コミット構成

本 status の変更は feature 単位で 3 コミット:

1. `experiment(work): 仮説 C 候補 (a') smoothing_delta=1000 19本撚線検証で却下記録（status-360）`
2. `feat(contracts): Phase E C21/C22/C23 追加 — term_names 重複 + contracts 重複 + verifier カテゴリ（status-360）`
3. `docs(status): status-360 + README/status-index/roadmap/CLAUDE.md 更新`
