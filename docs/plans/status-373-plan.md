# status TODO 整理計画（コンパクト版）

## Context

CLAUDE.md「やるべきこと」「凍結中の TODO」に積み上がった項目を、status-372
完了時点の現実に合わせて整理する。MCDD Phase E は (g1) 採択方向（7 本系
opt-in）/ 19 本却下で継続中。承認済み 3 方針:

- **A**: MCDD 基盤は **力評価系（HuberContact / K_c expansion / 契約検査）
  のみ縮退保持**。診断・実験ハーネスのうち成果が status に取り込み済み
  のものは削除可。
- **B**: 既存陰解法ソルバーは **`solver_mode` フラグで併存**（リスタート
  方式と当面併存、デフォルトは現行陰解法）。
- **C**: `work/beam_hysteresis/` 配下の **症状緩和系 experiment 5 本のみ削除**
  （結論が status に記録済みで再実行不要なもの）。

## 変更対象

### 1. CLAUDE.md「次の課題」「凍結中 TODO」整理

- 「次の課題」3 項目（K_mat x/z カップリング修正 / 7本ピッチ依存性 /
  リスタート方式）を **status-373 候補 (g3) に集約**、完了済み打消し線項目を
  status-index への参照に置き換えて圧縮
- 「凍結中 TODO」打消し線 6 項目を削除し、Phase E 凍結解除条件のみ残す
- 「現在の状態」block の status 列挙を status-360 以降のみに圧縮

### 2. `work/beam_hysteresis/` 症状緩和系 experiment 削除（5 本）

status の結論で**完全クローズ**かつ再実行価値なしの 5 本のみ:

- `15_hypothesis_c_7strand.py` (status-358 却下記録)
- `16_hypothesis_c_aprime_19strand.py` (status-360 却下記録)
- `22_bt_parameter_sweep_19strand.py` (status-363 候補 (c) クローズ)
- `25_freeze_param_sweep_19strand.py` (status-368 候補 (d) クローズ)
- `26_active_ema_alpha_sweep.py` (status-372 19 本却下、7 本結果は
  docstring と status に転記済み)

**保持**: `14_kc_active_boundary_diagnostic.py`（status-370 結果 B 確定の根拠）、
`14_kc_closest_adj_diagnostic.py`（Phase C-3' 診断の検証ライン）。

### 3. ソルバー `solver_mode` フラグ追加（方針 B、設計のみ）

実装は別 status。今回は **設計仕様の追記のみ**:

- `xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md` §4 に
  「solver_mode 併存方針」節を追加（陰解法 default / リスタート opt-in /
  切替境界の I/O 契約）
- `docs/roadmap.md`「撚線規模別 opt-in チューニング」表に
  `solver_mode` 行を追加（status-373 以降で実装）

### 4. status-373 新規作成

- 本整理作業を status-373 として記録（TODO 整理 + experiment 削除 5 本 +
  solver_mode 設計追記）
- `docs/status/status-index.md` に追加
- 候補 (g3) 着手は status-374 以降に正式着手として明記

## 変更ファイル

- `/home/user/xkep-cae/CLAUDE.md` (整理)
- `/home/user/xkep-cae/work/beam_hysteresis/{15,16,22,25,26}_*.py` (削除)
- `/home/user/xkep-cae/xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md` (追記)
- `/home/user/xkep-cae/docs/roadmap.md` (追記)
- `/home/user/xkep-cae/docs/status/status-373.md` (新規)
- `/home/user/xkep-cae/docs/status/status-index.md` (追記)

## 検証

1. `python contracts/validate_process_contracts.py` → 全 24 検査 OK
2. `pytest xkep_cae/contact/ xkep_cae/mathematics/` → 既存テスト全 pass
   （456 + 109 passed 5 skipped 維持）
3. `ruff check xkep_cae/ tests/ && ruff format --check xkep_cae/ tests/` → pass
4. `git grep -l "15_hypothesis_c_7strand\|16_hypothesis_c_aprime\|22_bt_parameter_sweep\|25_freeze_param_sweep\|26_active_ema_alpha_sweep"`
   → CLAUDE.md / status / roadmap での参照が打消し線または「削除済み」明記に
   なっていること

## コミット & push

- branch: `claude/execute-status-todos-bKVuY`
- commit 1: `chore(status): 症状緩和系 experiment 5 本削除（status-373）`
- commit 2: `docs(status): TODO 整理 + solver_mode 設計追記（status-373）`
- push: `git push -u origin claude/execute-status-todos-bKVuY`
