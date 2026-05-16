# status-372: 候補 (g1) active 履歴 EMA 平滑化 α 掃引 — 7 本部分達成 / 19 本却下

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-25
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10 passed（status-371 維持、回帰なし）

## 概要

status-371 で実装した候補 (g1) `HuberContactForceProcess.active_ema_alpha`
に対し α ∈ {0.0, 0.1, 0.3, 0.5} を **7 本 / 19 本撚線 90° 曲げ**で実機掃引。
status-371 の引継ぎ最優先 TODO に対応。

**判定**:

- **7 本撚線**: α=0.30/0.50 で frac=1.0 維持 + **cutback -61〜-75% 削減**
  （cb 57→14/22）、α=0.50 で **elapsed -11%**（298→265s）。α=0.10 のみ
  早期 stall（frac=0.3350、74s）— 弱平滑化が逆効果。
- **19 本撚線**: gate「frac ≥ 0.6」**全ケース未達**で候補 (g1) **却下方向**。
  α=0.50 で frac=0.5133（baseline 0.3739 比 +37.3% 改善、status-339 baseline
  0.4839 比 +6.1%）の部分改善は得たが elapsed +131%（251→582s）でコスト過大、
  α=0.10/0.30 は退化（-41%/-47%）。
- **default 変更なし**: `StrandBendingOscillationConfig.active_ema_alpha` の
  default=0.0 を維持。`active_ema_alpha=0.5` は **7 本系 cutback 削減
  opt-in escape hatch** として運用可能（status-369 「撚線規模別 opt-in
  チューニング」表に反映）。
- **次候補**: **(g3) pair-wise relaxation**（status-284 接触凍結を pair
  granularity 拡張）→ (g2) AL 再導入。

## 1. 実測条件

`work/beam_hysteresis/26_active_ema_alpha_sweep.py`（status-371 で実装、
default α リスト `0.0,0.1,0.3,0.5` で本 status の実測値を取得）:

```
uv run python work/beam_hysteresis/26_active_ema_alpha_sweep.py \
    --n-strands {7,19} --alphas 0.0,0.1,0.3,0.5
```

問題設定（共通）:

- `wire_radius=0.5`, `pitch_length=100.0`, `n_elements_per_pitch=16`
- `bending_curvature=0.015`（90° 曲げ）, `n_increments_per_cycle=20`
- `mu=0.15`, `max_nr_attempts=200`, `tol_force=1e-8`, `max_increments=10000`
- `free_end_mode=True`, `penalty_exponent=1.5` (Hertz)
- `smoothing_delta=自動`（=2000）, BT/damping/freeze は default

## 2. 7 本撚線 結果（status-358 baseline 互換）

| α | frac | conv | n_inc | n_cb | elapsed [s] | 備考 |
|---|------|:---:|------:|----:|-----------:|------|
| 0.00 | **1.0000** | Y | 524 | 57 | 298.55 | baseline（status-358 と一致、byte-identical） |
| 0.10 | **0.3350** | N | 170 | 15 | 74.76 | **早期 stall**（弱平滑化逆効果） |
| 0.30 | **1.0000** | Y | 793 | 14 | 305.00 | cb -75% / incr +51% / elapsed +2% |
| 0.50 | **1.0000** | Y | 647 | 22 | 265.10 | cb -61% / incr +23% / **elapsed -11%** |

**観察**:

- **非単調性**: α=0.10 のみ frac=0.3350 退化、α=0.30/0.50 は frac=1.0 完走。
  これは smoothing_delta 非単調性（status-262: delta_h=0.025 最速で 0.020/
  0.040 周辺は遅い）と類似。中間 α が「active 集合の短周期振動を固定する」
  共鳴的逆効果を持つ仮説。
- **α=0.50 の cutback 削減効果**: cb 57→22（-61%）が効果絶大。チャタリング
  pair の p_n 履歴を 50% 平滑化することで NR 反復の active flip が直接
  抑制された結果。
- **elapsed -11% は限定的**: incr 数が +23% 増加するため、cutback 削減効果が
  net で 1.1x 高速化しか生まない（α=0.30 では incr +51% で elapsed ほぼ
  同等）。

## 3. 19 本撚線 結果（status-339 / status-357 系設定）

| α | frac | conv | n_inc | n_cb | elapsed [s] | 備考 |
|---|------|:---:|------:|----:|-----------:|------|
| 0.00 | **0.3739** | N | 177 | 18 | 251.77 | status-357 baseline と一致 |
| 0.10 | 0.2225 | N | 158 | 17 | 282.27 | **-40% 退化** |
| 0.30 | 0.1988 | N | 113 | 16 | 149.68 | **-47% 退化** |
| 0.50 | **0.5133** | N | 332 | 19 | 582.05 | **+37% 改善**（gate 未達） |

**判定**:

- **gate「frac ≥ 0.6」全ケース未達** → 候補 (g1) **却下**
- α=0.50 の frac=0.5133 は status-339 baseline 0.4839 比でも +6.1%（限定的）、
  かつ elapsed +131%（251→582s）でコスト過大
- α=0.10/0.30 は 7 本 α=0.10 と同じ「弱平滑化逆効果」を 19 本でも再現

**物理解釈**: 19 本 Type D stall は status-370 で **K_c は active 境界でも FD
機械精度（rel_err=2.18e-07）を維持** することを確定済み（結果 B）。本 status の
α 掃引で示された通り、p_n 履歴平滑化単独では active 集合振動を **完全には
抑制できず**、α=0.50 の部分改善は得たが gate 達成には不足。多 pair 相互作用
（status-370 §6 診断限界の単一 pair / 静的を超える領域）に対しては pair
単位の制御が必要 → 候補 (g3) pair-wise relaxation へ。

## 4. MCDD 観点

### 4.1 脱法回避チェック

- **パターン 1**（tol 事後緩和）: gate「7 本 frac=1.0 維持」「19 本 frac ≥ 0.6」
  を事後緩和せず、α=0.10 退化と 19 本 gate 未達を**そのまま記録**
- **パターン 5**（既存テスト skip）: 既存 446 contact tests + 10 EMA tests
  全 pass、skip/xfail 増加なし（default α=0.0 で byte-identical）
- **パターン 6**（骨格だけ status）: 本 status は **実測 8 ケース（7 本 4 ×
  19 本 4）の frac/incr/cb/elapsed 数値**と却下判定の完結成果物
- **パターン 8**（回帰の根拠なき正当化）: α=0.10 退化を「弱平滑化逆効果」と
  解釈する仮説は smoothing_delta 非単調性 status-262 / 候補 (a) 反証
  status-358 / 候補 (a') 19 本却下 status-360 の系譜で、過去実証パターンに
  沿う（数値で反証可能）

### 4.2 候補 (g1) 数理的位置づけの確認

EMA 平滑化は K_c 自体を変更しない（FD 整合性は status-356 機械精度 2.18e-07
を維持）。`TermExpansionContract` 5 項分解は無変更で、C18-C24 全 24 検査は
pass。本 status は時間ステッピング側 escape hatch の有効性検証で、Phase E
契約検査拡張は不要。

## 5. ファイル変更サマリ

| ファイル | 変更 |
|---------|------|
| `work/beam_hysteresis/26_active_ema_alpha_sweep.py` | docstring に status-372 実測 8 ケース結果埋込（+38 行）|
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `active_ema_alpha` docstring 拡充（実測結果 + opt-in 案内）|
| `docs/status/status-372.md` | **新規** 本ファイル |
| `docs/status/status-index.md` | status-372 行追加 |
| `README.md` | 現在状況に status-372 追記 |
| `CLAUDE.md` | 「現在の状態」を status-372 に更新、status-373 へ TODO 引継ぎ |
| `docs/roadmap.md` | 候補 (g1) 19 本却下記録、(g3) を最優先候補に昇格 |

実装本体（`xkep_cae/contact/`、`xkep_cae/solve/`、`tests/`、`contracts/`）は
**無変更**。default α=0.0 維持で 7 本既存挙動完全保持。

## 6. opt-in ガイドライン更新（19 本 = `chattering_freeze_nr_max=30` と
   同列）

`StrandBendingOscillationConfig.active_ema_alpha` を以下の opt-in escape
hatch として `docs/roadmap.md`「撚線規模別 opt-in チューニング」表に追加:

| パラメータ | 7 本既定 | 7 本推奨 | 実測効果 | 根拠 status |
|----------|:------:|:------:|---------|:----------:|
| `active_ema_alpha` | `0.0` | `0.5`（任意） | 7 本: cb -61%（57→22）, elapsed -11% | status-372 |

19 本以上では却下方向で **opt-in 推奨せず**（19 本 gate 未達、α=0.50 で
elapsed +131% コスト過大）。

## 7. 引継ぎ（status-373 へ）

1. **最優先（候補 (g3) pair-wise relaxation 着手）**: status-284 接触凍結
   モードを pair granularity 拡張する設計仕様を `phase_c3prime_19strand_plan.md`
   §3.2 に追記。チャタリング pair のみ freeze + 残りは active 維持で 19 本
   多 pair 相互作用に介入する。実装規模見積もり ~150 行（status-365 候補 (e)
   Phase 1 と同程度）。
2. **副次（候補 (g2) AL 再導入の事前評価）**: status-221 で凍結した Uzawa の
   外側ループ 1〜2 回限定再導入（強収束化）の数理仕様確認。Phase C-3' s-tracking
   と直交する補強手段として記録。
3. **凍結中 TODO 棚卸し**: status-371 と同じ。Phase E 完了 + 19 本 frac=1.0
   完走 + `KcNormalDirectionStiffness` rel_err < 1e-2 を満たすまで全凍結維持。

## 8. Gate

| 項目 | 結果 |
|------|------|
| `ruff check xkep_cae/ tests/` | OK（197 files） |
| `ruff format --check xkep_cae/ tests/` | OK |
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK |
| `pytest xkep_cae/contact/` | **456 passed, 5 skipped**（status-371 維持） |
| `pytest xkep_cae/mathematics/` | 109 passed（status-364 維持） |
| 7 本撚線 90° 曲げ regression（α=0.0） | frac=1.0000, 298.55s（status-358 互換 byte-identical） |
| 19 本撚線 90° 曲げ baseline（α=0.0） | frac=0.3739（status-357 baseline 一致） |

## 9. 運用所見

- **EMA は時間ステッピング側 escape hatch の限定的成功**: 7 本系で
  cb -61〜-75% 削減という明確な効果はあるが、19 本 Type D stall（多 pair
  相互作用支配）には到達できず。候補 (a)/(a')/(c)/(d)/(e)/(g1) と同じく
  時間ステッピング系 escape hatch 6 連敗のパターンに加わる
- **多 pair 相互作用への介入が必要**: status-370 結果 B（K_c 機械精度）
  + 本 status α 掃引（履歴平滑化不足）の合算で、19 本 stall 主因は **pair 間
  力交換と active flip の同時発火**と再確定。pair granularity 制御（候補
  (g3)）か AL 外側ループ強収束化（候補 (g2)）が候補
- **MCDD 凍結解除条件再確認**: Phase E 完了 + 19 本 frac=1.0 + rel_err < 1e-2
  の 3 条件は本 status でも変更なし。候補 (g3)/(g2) で 19 本 frac=1.0 達成
  できなければ Phase D（実機 1000 本撚線到達）への移行は不可
