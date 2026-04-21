# status-358: 仮説 C 候補 (a) 7本撚線 90° 実測 → 未完走で却下 + C20 双方向紐付け検査追加

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-21
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25（変動なし）

## 概要

status-357 で積んだ最優先 TODO 2 件を実施:

1. **仮説 C 候補 (a) smoothing_delta 拡大の 7本撚線 90° 曲げ実測**
   — ユーザー指示（「19本でなく 7本 90° で効果見積もれる。効果なければ revert、
   10% 以上を目安」）に従い、ベースライン取得 + smoothing_delta=500 候補実測。
   **結果: frac=0.9241 で未完走 → 候補 (a) を却下（revert）**。

2. **MCDD Phase E: C20 双方向紐付け整合性検査追加**
   — `TermExpansionContract.providers` に列挙された Process クラスが、自身の
   `contracts` ClassVar で**同一の**契約を宣言していることを静的検査。
   **OK 判定取得、5 既存 providers で回帰なし**。

## 1. 仮説 C 候補 (a) smoothing_delta 拡大の実測検証

### 方針（status-357 の最優先 TODO 1「仮説 C 立案」への対応）

status-357 は Phase C-3' 実装後の 19 本 FD 再計測で frac=0.3739 退化と判定し、
次は **active 集合振動対策** と位置付けた。候補 4 件:

- (a) `smoothing_delta` の遷移帯広げ（status-260 の δ=1000 延長）
- (b) active 判定の履歴平滑化（low-pass）
- (c) line search 強化（NR 反復途中の過剰 active flip 抑制）
- (d) 接触凍結モード（status-284）の 19 本適用

ユーザー指示により **7本撚線 90°曲げで対策効果を見積もる** 方針を採用
（19本は Type D stall で frac=0.37 止まりのため 10% 判定ができない）。
候補のうち (a) が最小コード変更（`StrandBendingOscillationConfig.smoothing_delta`
パラメータ経由で試験可能）のため先行実測。

### 実装

新規スクリプト `work/beam_hysteresis/15_hypothesis_c_7strand.py` を作成。
ベースライン（`09_kcr_measurement_7strand.py`）と同一設定から
`smoothing_delta=500.0` のみ差し替え（default 2000 に対し 1/4、δ_h は 4 倍拡大）。

### ベースライン実測（`09_kcr_measurement_7strand.py`）

```bash
uv run python work/beam_hysteresis/09_kcr_measurement_7strand.py 2>&1 \
    | tee /tmp/baseline_7strand_1776787500.log
```

| 指標 | ベースライン |
|------|-------------|
| frac_completed | **1.0000** |
| n_increments | 524 |
| n_cutbacks | 57 |
| elapsed | 452.02 s |
| 接触凍結モード + Type D 対策 発火回数 | 166 |
| κ_cr mean | 5.74e-3（status-338 実測 5.80e-3 と整合） |

### 候補 (a) smoothing_delta=500 実測

```bash
uv run python work/beam_hysteresis/15_hypothesis_c_7strand.py 2>&1 \
    | tee /tmp/hypothesis_c_a_7strand_1776788900.log
```

| 指標 | 候補 (a) | ベースライン比 |
|------|---------|-------|
| frac_completed | **0.9241** | **未完走** |
| converged | **False** | — |
| n_increments | 421 | -19.7% |
| n_cutbacks | 49 | -14.0% |
| elapsed | 376.77 s | -16.6% |
| 接触凍結モード + Type D 対策 発火回数 | 150 | -9.6% |

### 判定: **却下（revert）**

- cutback -14%、elapsed -17% の見かけ改善はあるが、**`frac=0.9241` で未完走**。
- 「時間短縮」の原因は **解析の早期打切り**であり、対策効果ではない。
- **frac=1.0 完走を維持した上での 10% 以上改善** という合否基準を満たさない。
- δ_h を 4 倍拡大すると Huber 遷移帯が広がりすぎ、接触力の物理的精度が低下、
  終盤（frac > 0.9）で残差許容値達成不能になり打切り。

**ユーザー指示に従い候補 (a) を却下**。コード変更（default 変更）は行わず、
`15_hypothesis_c_7strand.py` を**失敗実験の記録**として残す
（status-354 の失敗実験 revert と同方針）。

### 次のアクション候補（status-359 以降）

候補 (a) は δ_h 拡大**定常化**が厳し過ぎた。次の 2 方針:

- **(a') 中間値で再試行**: `smoothing_delta=1000`（2x 拡大、default 2000 の半分）で
  再測定。δ_h 2x 拡大なら精度と安定性のバランスで 10% 改善 + frac=1.0 を達成する
  可能性。`15_hypothesis_c_7strand.py` の parameter 変更のみ。
- **(c) line search 強化**: NR 反復途中の過剰 active flip を `backtracking
  line search` で抑制。ペナルティ残差が増加する方向への step を rejection する。
  コード変更量は中程度（`_newton_dynamic.py` に line search hook 追加）。

仮説 (b) の active 判定履歴平滑化は物理的根拠が薄く（active の履歴依存は
bang-bang 的性質を失わせる可能性）、次サイクルの優先度は下げる。

## 2. MCDD Phase E: C20 双方向紐付け整合性検査追加

### 動機

status-357 で C18（`@verified_by` 紐付け）+ C19（`TermExpansionContract.
providers` 実在）を追加した。しかし双方向整合性が未検査で、以下の脱法が
すり抜ける懸念:

- **`providers` に追加のみで provider 側 `contracts` 未宣言**: orchestrator
  `tangent_components()` からは呼ばれないまま契約宣言だけ更新。
- **rename 亜種**: pattern 4 で新クラスを作成したが `contracts` 宣言を
  継承し忘れ、C19 は通るが実体が宣言していない状態。

### 実装: C20 検査

`contracts/validate_process_contracts.py` に `check_c20_term_expansion_
bidirectional()` を追加。各 `TermExpansionContract` の `providers` に
列挙された Process クラスが、自身の `contracts` ClassVar で同名契約
（`name` 一致）を宣言しているかを確認する。

主要アルゴリズム:

```python
for proc_name, cls in sorted(registry.items()):
    contracts = getattr(cls, "contracts", ())
    for contract in contracts:
        if not isinstance(contract, TermExpansionContract):
            continue
        for provider in contract.providers:
            provider_cls = registry.get(provider)
            if provider_cls is None:
                continue  # C19 側で検出
            provider_contract_names = {
                getattr(c, "name", None)
                for c in getattr(provider_cls, "contracts", ())
            }
            if contract.name not in provider_contract_names:
                errors.append(f"C20: ... {provider} が同名契約未宣言 ...")
```

### gate

```
$ uv run python contracts/validate_process_contracts.py
...
--- C20: TermExpansionContract 双方向紐付け ---
  OK
...
契約違反なし、条例違反なし
```

K_c_term_expansion の 5 providers（`KcNormalStiffnessProcess` /
`KcClosestPointStiffnessProcess` / `KcHermiteNonlocalStiffnessProcess` /
`KcGeoStiffnessProcess` / `ContactForceStStiffnessProcess`）は status-357 で
全て `_K_C_TERM_EXPANSION_CONTRACT` を `contracts` ClassVar に宣言済みのため
**回帰なし**。

## 3. 回帰確認

### 契約検査

```
$ uv run python contracts/validate_process_contracts.py
契約違反なし、条例違反なし （C18 / C19 / C20 + 既存 15 検査 全 20 検査 OK）
```

### 接触全体回帰

```
$ uv run --with pytest python -m pytest xkep_cae/contact/
421 passed, 5 skipped in 48.38s
```

### 7本撚線曲げ揺動回帰（軽量）

```
$ uv run --with pytest python -m pytest \
  tests/numerical_tests/test_strand_bending_convergence.py::...oscillation_converges
1 passed in 15.34s
```

status-357 と同数で回帰なし。

### lint / format

```
$ uv run ruff format work/beam_hysteresis/15_hypothesis_c_7strand.py contracts/validate_process_contracts.py
2 files reformatted
$ uv run ruff check (同 2 files)
All checks passed!
```

## Phase A〜E 進捗更新

Phase A〜E / status-346〜 の **10/N 完了**（status-358 で C20 追加 + 仮説 C 候補 (a) 反証）。

- [x] Phase A-1（status-346）: `MathematicalContract` 型 5 種
- [x] Phase A-2（status-347）: `ProcessContractRegistry` + `@verified_by`
- [x] Phase B-1（status-348）: `docs/math/03_huber_contact_penalty.md`
- [x] Phase B-2（status-349）: 6 章 / 55 アンカー + `equation_index.py` + C15 拡張
- [x] Phase C-1（status-350）: `KcNormal` / `KcGeo` 抽出
- [x] Phase C-2（status-351）: `KcHermiteNonlocal` / `KcClosestPoint` 抽出
- [x] 数理台帳訂正（status-353）: `K_mat,ndir ≡ K_geo` 同一性
- [x] Phase C-3 仮説 A 実験（status-354）: 単独フル項化は反証
- [x] Phase C-3' 診断（status-355）: active×adj ブロック局在化
- [x] Phase C-3' 実装（status-356）: 2 経路同時導入で FD 機械精度
- [x] status-357: 19 本 FD 再計測 + 回帰 + C5 解消 + C18/C19
- [x] **status-358（本 status）**: C20 双方向紐付け + 仮説 C 候補 (a) 反証
- [ ] status-359: 仮説 C 候補 (a') smoothing_delta=1000 再試行 or (c) line search 強化

## 引継ぎ（status-359 へ）

### 最優先 TODO

1. **仮説 C 候補 (a') smoothing_delta=1000 の 7本撚線 90° 曲げ再試行** — 候補 (a)
   の δ_h 4x 拡大は厳し過ぎた。2x 拡大（`smoothing_delta=1000`、default 2000 の半分）
   なら精度と安定性のバランスで合否基準（10% 以上改善 + frac=1.0 完走）達成の
   可能性あり。`15_hypothesis_c_7strand.py` の `smoothing_delta=500.0` を
   `1000.0` に書き換え、同 script を再実行。10% 未達なら revert。

2. **仮説 C 候補 (c) line search 強化**（(a') で効果薄の場合の次手）:
   NR 反復途中で接触残差が増加する step を backtracking line search で
   rejection。`_newton_dynamic.py` に line search hook を追加、
   `LineSearchUpdateProcess` の接触感知拡張として実装。

3. **MCDD Phase E 仕上げ** — C21 以降の候補検討:
   - C21: `TermExpansionContract.term_names` / `providers` の重複検出
     （現状 `__post_init__` で providers 重複のみチェック、term_names も同様が必要）
   - C22: `contracts` ClassVar の同名契約重複検出
   - C23: `@verified_by` の VerifyProcess 側が SolverProcess 継承必須

### 凍結中の TODO（MCDD 完了まで再開禁止）

Phase E 完成 + 19本 frac=1.0 完走 + `mat_only` rel_err < 1e-2 を満たした時点で
以下の凍結 TODO を再開可能:

- 7本撚線ピッチ依存性検証（p=50/100/200）
- ファイバー梁キャリブレーション
- リスタート解析方式
- 被膜圧縮モデル改善
- 空間ブロック分離 / ペアクラスタリング

## ファイル変更

| ファイル | 変更内容 |
|---------|---------|
| `contracts/validate_process_contracts.py` | **+71 行**: `check_c20_term_expansion_bidirectional()` 新設、主ループ追加、header docstring と修正ガイドに C20 追記 |
| `work/beam_hysteresis/15_hypothesis_c_7strand.py` | **新規**: 仮説 C 候補 (a) 検証スクリプト（**失敗実験記録**として残置） |
| `docs/status/status-358.md` | 本ファイル |
| `docs/status/status-index.md` | status-358 行追加 |
| `docs/roadmap.md` | 仮説 C 候補 (a) 却下記録 + status-359 次手 |
| `README.md` | 現在状態を status-358 に更新 |
| `CLAUDE.md` | 現在状態・次の課題を status-358 基準に更新 |

## コミット構成

本 status の変更は feature 単位で 3 コミットに分割:

1. `feat(contracts): C20 双方向紐付け整合性検査追加（status-358 Phase E）`
2. `chore(work): 仮説 C 候補 (a) smoothing_delta=500 実測スクリプト追加（status-358 失敗実験記録）`
3. `docs(status): status-358 + README/status-index/roadmap/CLAUDE.md 更新`
