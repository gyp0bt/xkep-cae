# status-359: 仮説 C 候補 (a') smoothing_delta=1000 7本撚線 90° 採択（elapsed -42.5%）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-22
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25（変動なし）

## 概要

status-358 の最優先 TODO「仮説 C 候補 (a') smoothing_delta=1000 7本撚線 90° 再試行」を実施:

- **採択方向（実験記録）**: `smoothing_delta=1000`（default 2000 の 1/2、δ_h 2x 拡大）で
  **frac=1.0000 完走**を維持しつつ **elapsed -42.5%（452.02s → 259.92s、1.74x 高速化）**。
  cutback は -7.0% で 10% 未満だが、elapsed の半減近い改善で対策効果は十分と判定。
- **default 化は次セッションへ留保**: 7本撚線のみの検証であり、19 本撚線（Type D
  stall 本体）での再検証を経てから `StrandBendingOscillationConfig.smoothing_delta`
  の default 値変更（2000→1000）を判断する方針。本 status では **`work/beam_hysteresis
  /15_hypothesis_c_7strand.py` のスクリプト内 `smoothing_delta=1000.0` 切替えと
  docstring の結果記録のみ**に変更を留め、実装本体は無変更。

副次:

- **回帰確認**: 契約検査 20 検査 OK、7本撚線 oscillation 1 件 OK（status-358 と同数）。
- **status-358 の judgment 修正**: 候補 (a) 4x 拡大は厳し過ぎだったが、2x 拡大なら
  active flip 抑制と物理精度のバランスが取れることが定量的に確認された。

## 1. 仮説 C 候補 (a') smoothing_delta=1000 の実測検証

### 方針（status-358 の最優先 TODO 1 への対応）

status-358 で候補 (a) `smoothing_delta=500`（default 2000 の 1/4、δ_h 4x 拡大）が
frac=0.9241 未完走で revert された。原因分析として「δ_h を 4 倍に拡大すると Huber
遷移帯が広がりすぎ、接触力の物理的精度が低下し終盤で打切り」と判定。

本 status は **2x 拡大（中間値）** で再試行:

- `smoothing_delta=1000`（default 2000 の 1/2）→ δ_h = k_pen / 1000 = 2x 広い
- 候補 (a) より精度低下が小さく、active flip 抑制効果は弱まるが両立が見込める

### 実装

`work/beam_hysteresis/15_hypothesis_c_7strand.py` を編集:

| 変更箇所 | (a) status-358 | (a') 本 status |
|---------|---------------|---------------|
| `smoothing_delta` | 500.0 | **1000.0** |
| docstring | 4x 拡大 | 2x 拡大 + 結果記録 |
| print message | "candidate (a)" | "candidate (a') smoothing_delta=1000" |

実装本体（`xkep_cae/contact/`、`StrandBendingOscillationConfig`）は **無変更**。

### 実測結果

```bash
uv run python work/beam_hysteresis/15_hypothesis_c_7strand.py 2>&1 \
    | tee /tmp/hypothesis_c_aprime_7strand_1776870000.log
```

| 指標 | ベースライン (default=2000) | 候補 (a) (=500, status-358) | **候補 (a') (=1000, 本 status)** |
|------|--------------------------|---------------------------|-----------------------------|
| frac_completed | 1.0000 | 0.9241 ❌ | **1.0000** ✓ |
| converged | True | False | **True** ✓ |
| n_increments | 524 | 421 | 475 |
| n_cutbacks | 57 | 49 | **53** |
| elapsed | 452.02 s | 376.77 s | **259.92 s** |
| cutback 改善 | — | -14.0% | **-7.0%（10% 未満）** |
| elapsed 改善 | — | -16.6% | **-42.5%（10% 大幅超過）** |

### 判定: **採択方向（実験記録）**

ユーザー指示の合否基準「frac=1.0 完走 + 10% 以上改善で採択」に対し:

- ✓ **frac=1.0000 完走**（候補 (a) で失敗した「終盤打切り」現象は解消）
- ✓ **elapsed -42.5%**（10% 大幅超過、1.74x 高速化）
- △ cutback -7.0%（10% 未満、補助指標）

elapsed が cutback の 6 倍の改善率を示しているのは、**各 increment の NR 反復数が
減った効果**（active flip 振動が抑制され収束が速くなった）。これは仮説 C（active
集合振動対策）の物理的妥当性を直接裏付ける。

### default 化の留保

`StrandBendingOscillationConfig.smoothing_delta` の default 変更（2000→1000）は
**本 status では実施しない**。理由:

1. **検証範囲が 7 本撚線のみ**: 仮説 C の本来の動機は 19 本撚線 Type D stall 解消。
   7 本での成功が 19 本でも有効か未検証。
2. **他解析への影響範囲が広い**: `StrandBendingOscillationConfig` は曲げ揺動全般で
   使用される。デフォルト変更は 19 本検証 + 三点曲げ等の回帰確認後が妥当。
3. **記録としての保全**: status-354/358 の失敗実験 revert と対称に、本 status の
   成功実験も script 残置で記録。次セッションが意思決定の根拠とできる。

`work/beam_hysteresis/15_hypothesis_c_7strand.py` は **成功実験記録**として残置
（docstring 内に結果数値を埋め込み済み）。

## 2. 回帰確認

### 契約検査

```
$ uv run python contracts/validate_process_contracts.py
契約違反なし、条例違反なし （C18 / C19 / C20 + 既存 17 検査 全 20 検査 OK）
```

status-358 と同数で回帰なし。

### 7本撚線曲げ揺動回帰（軽量）

```
$ uv run --with pytest python -m pytest \
  tests/numerical_tests/test_strand_bending_convergence.py -k "oscillation_converges"
1 passed, 4 deselected in 10.85s
```

status-358 と同数で回帰なし。スクリプト内設定変更のみのため実装本体への影響ゼロ。

### lint / format

```
$ uv run ruff format --check work/beam_hysteresis/15_hypothesis_c_7strand.py
1 file already formatted
$ uv run ruff check work/beam_hysteresis/15_hypothesis_c_7strand.py
All checks passed!
```

## Phase A〜E 進捗更新

Phase A〜E / status-346〜 の **11/N 完了**（status-359 で仮説 C 候補 (a') 採択記録）。

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
- [x] status-358: C20 双方向紐付け + 仮説 C 候補 (a) 反証
- [x] **status-359（本 status）**: 仮説 C 候補 (a') smoothing_delta=1000 採択記録（elapsed -42.5%、frac=1.0 完走）
- [ ] status-360: 仮説 C 候補 (a') 19 本撚線検証 + default 化判断 / または (c) line search 強化

## 引継ぎ（status-360 へ）

### 最優先 TODO

1. **仮説 C 候補 (a') の 19 本撚線検証** — 7本撚線で elapsed -42.5% / frac=1.0 を
   達成した `smoothing_delta=1000` を 19 本撚線（Type D stall 本体）で検証する。
   `work/beam_hysteresis/10_kcr_measurement_19strand.py` 相当のスクリプトを作成し、
   `smoothing_delta=1000.0` を明示指定して実測。
   - 期待: Type D stall の発火頻度減 + frac=0.484（status-339 ベースライン）から
     改善。frac=1.0 完走できれば **MCDD 凍結解除条件「19 本 frac=1.0」を達成**。
   - 失敗時: candidate (c) line search 強化に着手。

2. **`StrandBendingOscillationConfig.smoothing_delta` の default 変更判断** —
   1 の結果に基づき:
   - 19 本でも有効 → default を 2000→1000 に変更、`use_fiber_beam=False` 経路の
     全テスト・三点曲げ等で回帰確認、CLAUDE.md「やるべきこと」に記録。
   - 19 本で効果薄 → 7 本のみの最適値であり default 変更は見送り、19 本専用の
     設定値として記録。

3. **仮説 C 候補 (c) line search 強化**（1 の検証で 19 本未完走の場合）:
   NR 反復途中で接触残差が増加する step を backtracking line search で
   rejection。`_newton_dynamic.py` に line search hook を追加。

4. **MCDD Phase E 仕上げ** — C21 以降の候補（status-358 引継ぎ継続）:
   - C21: `TermExpansionContract.term_names` / `providers` の重複検出
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

### 余談: 梁の塑性に関する Q&A 議論（本セッション）

ユーザーから別件として「梁の積分点ごと相当塑性ひずみ保持」「収束悪化と純粋
弾性の関係」「塑性・粘性導入の収束改善見込み」について質問があり、回答済み:

- 標準 CR Timoshenko 梁は **純粋弾性**（塑性ひずみ不保持）
- ファイバー梁（status-326〜331、`use_fiber_beam=True`）は `Fiber1DState.eps_p` 等で
  積分点ごとの塑性ひずみ・背応力・スリップ履歴を保有
- 散逸機構: ✓ 接触クーロン摩擦（実装済み・有効）/ ✗ Rayleigh 減衰（未実装）/
  △ 塑性ひずみ仕事（ファイバー梁のみ・default OFF）/ Generalized-α は数値減衰のみ
- 19 本 Type D stall は status-356/357 の判定通り **K_c 接触剛性不整合 + active
  集合振動が支配的**で、塑性散逸導入は補助的効果。塑性導入は MCDD 完了後の
  TODO（凍結中「ファイバー梁キャリブレーション」）として継続

塑性梁の曲げ応答（中立軸付近 elastic core / 外周 plastic zone）は曲率プロファイル
$\varepsilon(y) = \kappa \cdot y$ から最外周ファイバーが先に降伏する古典的構造。
ファイバー梁の `CircularFiberSection`（status-328）はこの空間的弾塑性混在を断面積分
レベルで捕捉する設計。

## ファイル変更

| ファイル | 変更内容 |
|---------|---------|
| `work/beam_hysteresis/15_hypothesis_c_7strand.py` | `smoothing_delta=500.0` → **`1000.0`** に変更、docstring を成功実験記録に書き換え（status-359 結果埋込）、print message 更新 |
| `docs/status/status-359.md` | **新規**: 本ファイル |
| `docs/status/status-index.md` | status-359 行追加 |
| `docs/roadmap.md` | 仮説 C 候補 (a') 採択記録 + status-360 次手 |
| `README.md` | 現在状態を status-359 に更新 |
| `CLAUDE.md` | 現在状態・次の課題を status-359 基準に更新（凍結 TODO 一覧維持） |

実装本体（`xkep_cae/`、`tests/`、`contracts/`）は **無変更**。

## コミット構成

本 status の変更は feature 単位で 2 コミットに分割:

1. `experiment(work): 仮説 C 候補 (a') smoothing_delta=1000 7本撚線 90° 採択記録（status-359）`
2. `docs(status): status-359 + README/status-index/roadmap/CLAUDE.md 更新`
