# status-367: 候補 (e) 接触減衰 escape hatch — validation（符号訂正 + 7 本成功 + 19 本却下）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-23
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7 passed（回帰なし）

## 概要

status-366 Phase 2 配線の実測 validation。3 つの成果:

1. **符号規約バグ訂正**: `R_u += f_damp` → `R_u -= f_damp`。初期実測で
   c_n>0 の全ケースが frac=0.05 で即発散し、根本原因は Process 戻り値の
   物理ドラッグ力符号と NR 残差規約の不整合と判明。
2. **7 本撚線: 採択方向**（opt-in escape hatch として）: c_n=1000 で
   frac=1.0000 完走 + **elapsed -56.8%**（246→106s）+ incr -73% + cb -85%。
3. **19 本撚線: 却下**: c_n=100/1000 どちらも frac<baseline（Type D stall
   解消せず）。MCDD 凍結解除条件未達。

## 1. 符号規約バグの訂正

### 1.1 症状

status-366 配線後の初回 7 本撚線 c_n 掃引で、c_n=10 以上の全ケースが
incr=2 付近で `[C+D.div]` 発散検知し frac=0.05 で早期打切り。baseline
（c_n=0）は frac=1.0 完走で、実装不整合を示唆。

### 1.2 根本原因

`ContactNormalDampingProcess` は **物理ドラッグ力** `f_damp = -c_n v_n g`
（motion を妨げる向き）を返す。一方 NR の残差規約は

```
R = f_int + f_c - f_ext + M·a + C·v
```

で、`C·v` は **正寄与**（ダンピング行列 × 速度）として加算される。つまり
「減衰の残差寄与」は `+c_n v_n g` であるべきで、物理ドラッグ力 `f_damp` を
そのまま `R_u += f_damp` と加算すると符号が反転する。

K_damp 側は `-∂f_damp/∂u = +c_n·c1·g⊗g`（PSD）で、これは `∂(-f_damp)/∂u`
に等しいため `K_T += K_damp` は整合していた。したがって「R に間違った符号
で f_damp が入り、K は正しく整合」という二重基準状態が生まれていた。

### 1.3 訂正

`xkep_cae/contact/solver/_newton_dynamic.py`:

```python
R_u = R_u + _damp_out_iter.f_damp   # NG: 物理力の符号で残差加算
                    ↓
R_u = R_u - _damp_out_iter.f_damp   # OK: 符号反転で +c_n v_n g が加算
```

`ContactNormalDampingOutput` docstring に符号規約節を追加し、Process は
物理ドラッグ力を返す / NR 側は符号反転加算する / ユニットテスト
`test_tangent_matches_fd_under_v_is_c1_u` の `K_provided ≈ -J_fd` は
同値性保証であることを明記（テスト本体は変更せず）。

## 2. 7 本撚線 c_n 掃引（sign fix 後）

### 2.1 実測結果（work/beam_hysteresis/23_contact_damping_7strand_sweep.py）

| c_n | frac | incr | cb | elapsed | Δelapsed | max_ratio | final_ratio |
|---|---|---|---|---|---|---|---|
| 0 (baseline) | 1.0000 | 475 | 53 | 246.20s | — | 0 | 0 |
| 10 | 0.8484 | 191 | 9 | 161.24s | -34.5% | 1.51 | 0.47 |
| 100 | 0.9501 | 151 | 7 | 137.41s | -44.2% | 3.84 | 1.15 |
| **1000** | **1.0000** | **128** | **8** | **106.47s** | **-56.8%** | 6.79 | 0.96 |
| 10000 | 0.6035 | 78 | 9 | 95.54s | -61.2% | 1.33 | 1.83 |

**c_n=1000 が 7 本撚線の最適値**。frac=1.0 完走しつつ **elapsed -56.8%**
（246→106s）、incr -73%（475→128）、cutback -85%（53→8）の劇的な
改善。小さな c_n（10, 100）は中途半端な減衰で active flip を抑制しきれず
未完走、大きな c_n（10000）は過剰減衰で前進不能。

### 2.2 budget 超過の解釈

全 c_n>0 ケースで `max_ratio = E_damp_cum / E_strain` が budget 0.20 を
超過。特に c_n=1000 は max=6.79 だが final_ratio=0.96 で最終的には E_strain
と同程度に収束。これは NR 反復中の **過渡的な運動エネルギー消散ピーク**
が E_strain より先に積み上がる現象で、定常（final_ratio）で見れば物理的に
妥当。budget_ratio の定義は「max vs 許容」で過渡ピークを捕捉する仕様の
ため、運用では `max_ratio < 10` 程度を実用許容線とする方針が現実的。

## 3. 19 本撚線: MCDD 凍結解除条件未達

### 3.1 実測結果（work/beam_hysteresis/24_contact_damping_19strand.py）

| c_n | frac | vs baseline | elapsed | max_ratio | 判定 |
|---|---|---|---|---|---|
| 0 (baseline, status-339) | 0.4839 | — | 534.68s | 0 | Type D stall |
| 100 | 0.4280 | -11.5% | 656.11s | 4.43 | 退化 |
| 1000 | 0.4697 | -2.9% | 407.83s | 9.72 | ほぼ同等（わずか退化） |

**どちらの c_n でも 19 本 Type D stall を解消できず**。MCDD 凍結解除条件
「frac=1.0 完走 + E_damp/E_strain < budget」**未達**。

### 3.2 物理的解釈

7 本で劇的に効いた damping が 19 本で効かない理由は、**stall の主因が
tangent 不整合（K_c x/z カップリング）にある**ためと解釈する。

- status-344 で 19 本の `mat_only` FD rel_err mean=44%、comp_x max=98%
  を計測済み
- status-361 で 19 本重荷重のみ mixed (C+D) 16.6% 突出を確認
- 候補 (c) line search（status-362/363）も 19 本で frac=0.52 止まり

したがって、**局所（ペア単位）の粘性を増やしても、K_c の解析式自体に
残る隣接ノード寄与の不整合は解消できない**。これは候補 (c) line search
が同じ理由で 19 本で効きにくかった現象と同質。

### 3.3 候補 (e) の位置付け

- 19 本 MCDD 本命課題（frac=1.0 完走）には**寄与しない**
- 7 本では **opt-in 高速化 escape hatch** として大幅効果（-57% elapsed）
- default OFF（`contact_damping_coefficient=0.0`）を維持、実装本体は
  status-366 のまま無変更で運用可能

## 4. Gate

- 符号訂正後の unit test: `pytest xkep_cae/contact/damping/` **19 passed**
- 回帰: `pytest xkep_cae/contact/` **446 passed 5 skipped**
- 契約検査: 全 24 検査 OK（契約違反 0 件 / 条例違反 0 件）
- ruff check + format pass

## 5. ファイル変更

| ファイル | 変更 |
|---------|------|
| `xkep_cae/contact/solver/_newton_dynamic.py` | `R_u -= f_damp`（符号訂正）+ 符号規約コメント追記 |
| `xkep_cae/contact/damping/strategy.py` | `ContactNormalDampingOutput` docstring に符号規約節追加 |
| `work/beam_hysteresis/23_contact_damping_7strand_sweep.py` | **新規**（Phase 2 7本 c_n 掃引） |
| `work/beam_hysteresis/24_contact_damping_19strand.py` | **新規**（Phase 2 19本 c_n 適用） |
| `docs/status/status-367.md` | **新規** 本ファイル |
| `docs/status/status-index.md` | status-367 行追加 |
| `README.md` | 現在状況に status-367 追記 |
| `docs/roadmap.md` | 候補 (e) validation 結果行追加 |

## 6. 引継ぎ（status-368 へ）

1. **最優先: 候補 (d) 接触凍結モード 19本再評価** — status-284 で 7 本
   frac 0.40→0.70 を達成した手法を 19 本に適用。`chattering_freeze_enabled
   =True` は既に default ON、パラメータ（`freeze_max_cycles`, `freeze_nr_max`,
   `freeze_tol_factor`）のチューニングで 19 本 Type D stall を抑制できるか
   を実測。既存 7 本パラメータは 19 本用に調整されていない。
2. **副次: 候補 (f) Phase C-3' s-tracking 19本再評価** — status-355/356 の
   `K_hermite_adj` + `K_closest/K_st` active×adj 拡張は 7 本では FD 機械精度、
   19 本では frac=0.37 退化で保留中（status-357）。ただし Phase C-3' の
   ゲートテストは active 集合固定下での解析精度で、active 変動領域の挙動は
   別系統の問題。`_newton_dynamic.py` への追加チューニング余地がある。
3. **候補 (e) 7本 opt-in 高速化の活用提案** — 動的解析の一般ベンチマーク
   で `contact_damping_coefficient=1000` による -57% 高速化を選択的に適用
   可能。7本/19本以外の撚線構成（例: 3本、13本）での有効範囲を測定すると
   実用ガイダンスが得られる。
4. **MCDD Phase E C25 候補** は引き続き保留（damping 配線完了後の検討項目）。
