# status-370: Phase C-3' Step 3.1 完了 — active 境界 FD 診断で結果 B 確定（K_c 項欠落ではなく NR alg 側問題）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-24
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7（status-369 から変動なし）

## 概要

status-369 §5 最優先 TODO「Phase C-3' Step 3.1 active 境界 FD 診断実施」に対応。`work/beam_hysteresis/14_kc_active_boundary_diagnostic.py` を新設（`14_kc_closest_adj_diagnostic.py` を雛形）、`test_helical_3d_hermite` シナリオで `gap_target` を deep contact から active 境界まで sweep + 強制 flip を加え、K_c 解析値 vs FD の rel_err を 20 測定点で計測。

**結論: 結果 B 確定** — 全 20 点で rel_err が status-356 の機械精度 **2.18e-07〜2.20e-07** に張り付いた。19 本 Type D stall は **K_c 項の欠落ではなく NR アルゴリズム側の動力学**（反復間 active 振動 / pair 間相互作用 / 摩擦活性切替）と確定。Step 3.2 は当初計画の新項 `KcActiveFlipStiffness` 追加から、**候補 (g) NR alg 側 3 サブライン (g1)〜(g3)** へ再配分。

## 1. 実験内容

### 1.1 新設スクリプト `14_kc_active_boundary_diagnostic.py`

3 Block 構成、`penalty_exponent=1.5, use_hermite=True, helical_z=True, n_elems=3`:

- **Block 1**: δ_h=0 (C0 Huber), gap_target ∈ [-1e-2, -5e-3, -1e-3, -5e-4, -1e-4, -1e-5, -1e-6] の 7 点
- **Block 2**: δ_h=5 (smoothing_delta=2000, `StrandBendingOscillationConfig` 既定), gap_target ∈ [-1e-3, -5e-4, -2e-4, -1e-4, 0, +1e-4, +2e-4, +5e-4] の 8 点（Huber 平滑化ゾーン全域跨ぎ、inactive 側も含む）
- **Block 3**: 強制 flip（fd_eps ≥ |gap|）5 点 — gap=-5e-8/eps=1e-7（C0 符号跨ぎ）、gap=-1e-5/eps=1e-4（C0 大 flip）、gap=+1e-8/eps=1e-7（initial inactive→active）、同 2 点の δ=2000 版

### 1.2 実測結果

| Block | 条件 | worst rel_err |
|-------|------|:-------------:|
| 1 | δ_h=0, gap ∈ [-1e-2, -1e-6] | **2.19e-07** |
| 2 | δ_h=5, gap ∈ [-1e-3, +5e-4]（inactive 側含む） | **2.20e-07** |
| 3 | 強制 flip (eps=1e-7): gap=-5e-8, +1e-8 | **2.20e-07** |
| 3 | 強制 flip (eps=1e-4): gap=-1e-5 | 2.19e-04（FD truncation, O(eps)）|

判定サマリ: baseline rel_err=2.180e-07 / worst boundary rel_err=2.192e-07 / degradation=**1.01x (+0.00 桁)**。smoothed ゾーン worst=2.201e-07、flip 強制 (eps=1e-7) worst=2.20e-07。

eps=1e-4 の rel_err=2.19e-04 は `2.18e-7 × 1e3` と整合的で、前進差分の O(eps) truncation 誤差であり K_c 不整合ではない。

### 1.3 シェア分解

全 20 点で diff の 99%+ が active×active ブロック（`aa%≥99.6`）、adj 関連は ≤0.4%。status-356 の 2 経路相殺定理は active 境界跨ぎ・flip 強制下でも完全成立。

## 2. MCDD 観点: Step 3.2 再設計

当初計画（`phase_c3prime_19strand_plan.md` 旧 §3.2）は「**結果 A** なら新項 `KcActiveFlipStiffness` を `TermExpansionContract` 6 項目化で追加」だったが、**結果 B 確定で新項追加は不要**と判明。plan doc §3.2 を候補 (g) 3 サブライン再配分に書き換え:

- **(g1) active 履歴平滑化**（最優先、~130 行）: 反復間で `p_n` を EMA 平滑化、`active_ema_alpha` 追加
- **(g3) pair-wise relaxation**（次点）: status-284 接触凍結モードを pair granularity 拡張
- **(g2) augmented Lagrangian 再導入**（保守的）: status-221 で凍結した Uzawa 外側ループの限定再導入（数理台帳整合性要確認）

gate 基準: 7 本 frac=1.0 維持 / 19 本 frac≥0.6（baseline 0.3739 の 60%）/ `test_helical_3d_hermite` rel_err<1e-5 維持

### 2.1 本 status の診断的限界

本 diagnostic は **単一 pair / 摩擦なし / 静的シナリオ** — 多 pair 交互作用 / 摩擦 K_st 整合性 / NR 反復間 active 振動は捕捉しない。結果 B は「**個々の pair の K_c は境界・flip 下でも正確**」を確定したに留まる。多 pair / 摩擦下の整合性は候補 (g) 実装と並行して追加診断可能（status-371+ で必要に応じて）。

## 3. ファイル変更

| ファイル | 変更 |
|---------|------|
| `work/beam_hysteresis/14_kc_active_boundary_diagnostic.py` | **新規**（+280 行、3 Block 構成の診断スクリプト）|
| `xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md` | §3.1 に実測結果テーブル追記、§3.2 を結果 B 分岐で候補 (g1)/(g2)/(g3) に再設計、§6 に本 status を追加 |
| `docs/status/status-370.md` | **新規** 本ファイル |
| `docs/status/status-index.md` | status-370 行追加 + 末尾メモ追記 |
| `README.md` | 現在状況に status-370 追記 |
| `CLAUDE.md` | 「現在の状態」を status-370 に更新、次課題を (g1) 実装向けに書き換え |
| `docs/roadmap.md` | MCDD 項に status-370 結果 B 追記（該当節のみ短く）|

## 4. Gate

- `ruff check xkep_cae/ tests/` / `ruff format --check xkep_cae/ tests/`: **OK**（実装本体変更なし）
- 契約違反 **0 件**（全 24 検査 OK）/ 条例違反 **0 件**
- `python contracts/validate_process_contracts.py`: OK
- `pytest xkep_cae/contact/`: 回帰なし（実装本体無変更のため skip、周辺 diagnostic のみ追加）

## 5. 引継ぎ（status-371 へ）

1. **最優先**: 候補 (g1) active 履歴平滑化を実装。`HuberContactForceProcess` に `active_ema_alpha` field 追加、`evaluate()` で前反復 `p_n_prev` を保持し `p_n_eff = α·p_n_new + (1-α)·p_n_prev` を計算。NR ソルバー側の反復コールバックで `p_n_prev` を更新。7 本 frac=1.0 回帰確認 → 19 本 Type D stall に適用。α ∈ {0.1, 0.3, 0.5} 掃引。
2. **副次**: 必要に応じて多 pair 診断 `14b_kc_multi_pair_diagnostic.py` を追加（現診断の単一 pair 限界補完）。ただし (g1) が効けば本命ではない。
3. **Phase E C25 候補**: 継続保留。(g1)〜(g3) 進展後に再検討。
4. **凍結中 TODO 棚卸し**: status-363/368/369 と同じ。

## 6. 運用所見

- **診断 status の価値**: 実装本体 0 行でも「仮説反証 + 代替ライン確立」の定量結果は 1 status の価値として十分（status-354 仮説 A 反証と同構造）。status-352 中断スナップショットとは異なり、正規 status 番号で完結。
- **plan doc の sub-plan 再配分**: Step 3.2 を事前に A/B 分岐で scoping しておいた status-369 の deliverable が本 status で分岐判断を即実行可能にした。MCDD 脱法パターン 6（骨先行）回避の実例。
- **候補 (g) 命名**: status-363 では「(g) NR alg 側」と総称だったが、本 status で (g1)/(g2)/(g3) に細分。実装オーダー可視化で次セッション見通し改善。
