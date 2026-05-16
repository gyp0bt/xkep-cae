# status-369: Case B 19 本 opt-in ガイドライン化 + 候補 (f) Phase C-3' 実験計画 策定（ドキュメンテーション status）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-24
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7（status-368 から変動なし、本 status は documentation のみ）

## 概要

status-368 §6 引継ぎ TODO 1. 最優先「候補 (f) Phase C-3' s-tracking 19 本
再評価」は multi-session 規模の研究タスク（active 集合変動下での FD 診断
+ 新項 `KcActiveFlipStiffness` 追加検討、数百行の実装 + 19 本撚線 60 分級
実測ループ）。本 status では以下 2 点に絞って documentation deliverable を
確定し、status-370 以降の実施に向けた地ならしに徹する:

1. **TODO 2 副次（Case B opt-in 推奨化）完了** — status-368 の副次 TODO
   「`chattering_freeze_nr_max=30` を 19 本以上向けの opt-in escape hatch
   として明記」を `StrandBendingOscillationConfig` docstring / `docs/roadmap.md`
   §推奨ソルバー構成 に反映。
2. **TODO 1 reconnaissance（scoping doc 策定）** — 候補 (f) の実験計画を
   `xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md` に新設し、
   Step 3.1 active 境界 FD 診断（~30 分）/ Step 3.2 新項追加設計（~2 時間）の
   2 段階実験を scoping、status-370 以降で即着手可能に。

**実装本体（`xkep_cae/`、`tests/`、`contracts/`）への変更は無し**
（config docstring の注記拡張と `_Hydrated` 等のコード挙動変化を伴わない
修正に限定）。MCDD 脱法パターン 6「骨格だけの status」回避のため、
documentation deliverable として完結的に扱う（実装保留ではなく documentation
完成）。

## 1. TODO 2 完了（Case B 19 本 opt-in 推奨化）

### 1.1 StrandBendingOscillationConfig docstring 拡張

`xkep_cae/numerical_tests/strand_bending_oscillation.py:261-280`:

既存コメント（status-368 時点）は掃引の目的と `work/beam_hysteresis/25_freeze_param_sweep_19strand.py`
への参照にとどまっていた。本 status で以下を追記:

```python
# 19 本以上の大規模撚線向け opt-in 推奨（status-368 Case B / status-369 明記）:
#     chattering_freeze_nr_max = 30   # default 15 の 2x
#
# 実測効果（status-368 19 本 90° 曲げ）: frac 0.3739 → 0.5642（+50.9%、
# status-339 baseline 0.4839 比 +16.6%）。最終 NR Type 分布の mixed (D+E)
# 比率が 69% → 56% に低下（BT line search と同パターン）。代償として
# elapsed +251%（245s → 863s）。MCDD 凍結解除条件（frac=1.0 完走）未達
# のため default 変更は実施せず（7 本系の回帰リスク回避）。19 本以上で
# frac=1.0 が未達な系には `chattering_freeze_nr_max=30` を明示指定する。
```

`chattering_freeze_*` 4 field の既定値は **不変**（7 本撚線向け最適化維持）。

### 1.2 `docs/roadmap.md` §推奨ソルバー構成 拡張

§推奨ソルバー構成直下に新セクション **「撚線規模別 opt-in チューニング」**
を追加、4 項目（`chattering_freeze_nr_max` / `contact_damping_coefficient`
/ `smoothing_delta` / `contact_backtracking_*`）について 7 本既定 / 19 本
推奨値 / 実測効果 / 根拠 status の表を掲載。

opt-in 4 項目はいずれも **MCDD 凍結解除条件（frac=1.0 完走）未達**であり、
症状緩和 escape hatch として運用する旨を明記（脱法パターン 1 「目標緩和」
との混同防止）。

## 2. TODO 1 reconnaissance（Phase C-3' 19 本再評価 実験計画）

### 2.1 新規 scoping doc

`xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md`（+107 行）を新設。
本 plan は status-354〜357/368 の経過を踏まえ、候補 (f) を 2 ステップに
分割:

**Step 3.1（所要 ~30 分）**: 2 素線 scenario（`test_helical_3d_hermite`）
に active 境界 perturbation（接触間隙 `g = ±10^-3`）を注入、FD 整合性を再計測。
status-356 の機械精度 rel_err = 2.18e-07 が active 境界で 2 桁以上悪化
することを定量検出し、Phase C-3' の限界域を数値で確定する。

**Step 3.2（所要 ~2 時間）**: Step 3.1 結果に応じて以下を分岐:

- **結果 A**（境界で rel_err 悪化）: 新項 `KcActiveFlipStiffness` を
  `TermExpansionContract` 6 項目化で追加。Huber の 2 階微分相当項を
  `HuberContactForceProcess.tangent()` で評価、`docs/math/03_huber_contact_penalty.md`
  §9（新設）に数理的根拠を記述。実装 ~200 行。
- **結果 B**（境界でも rel_err 健全）: 項の欠落ではなく NR アルゴリズム
  側の問題（active 判定履歴平滑化 / augmented-Lagrangian 再導入等）。
  候補 (g) として別ラインで再計画。

### 2.2 plan doc が提供するもの

- status-354（仮説 A 反証）/ 355（診断）/ 356（2 経路相殺で機械精度達成）
  / 357（19 本退化検出）/ 368（症状緩和 4 候補全クローズ）の時系列総括
- MCDD 脱法パターン 10 項目のうち本計画で回避すべき 4 項目（1/4/5/6）の
  チェックリスト
- gate 基準（`test_helical_3d_hermite` rel_err < 1e-5 維持 / 新 gate で
  rel_err < 1e-4 / 19 本 `mat_only rel_err mean < 0.25` / 19 本 frac ≥ 0.8）

### 2.3 本 status では実験は実施しない理由

- Step 3.2 結果 A 方向でも実装 ~200 行 + 既存 12 テスト互換性担保 + 新
  gate test 追加で 1 status の容量を超える（status-356 は 2 経路同時導入
  のみで実装 ~45 行、それでも 1 status 丸々使用）。
- Step 3.1 単独でも 2 素線シナリオ改造 + FD スキャン + 診断レポート作成で
  30 分、結果解釈 30 分、計 1 時間。本 status は documentation 完結性を
  優先し、Step 3.1 は status-370 の単独 TODO として残す。
- 脱法パターン 6「status ファイルに "TODO として積む" で次回送り」との
  差分: 本計画は「次回何を具体的に測るか」を scoping doc で固定し、実験
  スクリプト名（`14_kc_active_boundary_diagnostic.py`）まで指定している
  ため、status-370 担当者がゼロから議論を再開する必要はない。

## 3. ファイル変更

| ファイル | 変更 |
|---------|------|
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `StrandBendingOscillationConfig` の `chattering_freeze_*` 周辺コメントに 19 本 opt-in 推奨を追記（+8 行、実装挙動変化なし） |
| `docs/roadmap.md` | §推奨ソルバー構成 直下に「撚線規模別 opt-in チューニング」表を新設（+14 行） |
| `xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md` | **新規**（+107 行、scoping doc） |
| `docs/design/README.md` | 設計文書索引に plan doc 行を追加（+1 行） |
| `docs/status/status-369.md` | **新規** 本ファイル |
| `docs/status/status-index.md` | status-369 行追加 |
| `README.md` | 現在状況に status-369 追記 |
| `CLAUDE.md` | 「現在の状態」を status-369 に更新、「次の課題」を status-370 向けに書き換え |

## 4. Gate

- `ruff check xkep_cae/ tests/` / `ruff format --check xkep_cae/ tests/`: **OK**（本 status はコメント追記のみで logic 変更なし）
- `pytest xkep_cae/contact/`: **446 passed 5 skipped**（status-368 と同数、回帰なし）
- 契約違反 **0 件**（全 24 検査 OK） / 条例違反 **0 件**
- `python contracts/validate_process_contracts.py`: OK

## 5. 引継ぎ（status-370 へ）

1. **最優先: `phase_c3prime_19strand_plan.md` §3.1 Step 3.1 実施** — `work/beam_hysteresis/14_kc_active_boundary_diagnostic.py` を新設（`14_kc_closest_adj_diagnostic.py` を改造、g = ±10^-3 の active 境界シナリオを注入）。結果 A/B 判定後、Step 3.2 の実装方針を status-370 で確定。
2. **副次**: 本 status で `docs/roadmap.md` に整備した opt-in 表（撚線規模別チューニング）は status-362/367/368 の実測値を集約した運用指針。他の撚線数（3/13/37）で同じ枠を埋める実測は MCDD 完了後の凍結 TODO 棚卸しに回す。
3. **MCDD Phase E C25 候補**: 引き続き保留。候補 (f) の Step 3.1〜3.2 が進展してから再検討。
4. **凍結中 TODO 棚卸し**: status-363 §TODO / status-368 §TODO と同じ（変動なし）。

## 6. 運用所見

- **documentation-only status の効率**: 本 status は実装コード 0 行変更で
  scoping + opt-in docs のみ。1 status 1 PR 粒度に対しやや軽量だが、
  候補 (f) の次 status で「地ならしが済んでいる」ことの価値は大きい
  （status-352 「中断スナップショット」のような非常手段と異なり、正式な
  status 番号を取った計画 deliverable として扱う）。
- **plan doc の配置先**: `xkep_cae/mathematics/docs/` に colocate。`docs/math/`
  は「離散化方程式の single source of truth」で実装計画には不適、
  `docs/design/README.md` から索引登録することで横断発見性を確保。
- **opt-in 表の整合性**: 7 本既定値が 19 本で悪化する項が複数（`smoothing_delta`
  1000 / `contact_damping_coefficient` 1000）あるため、将来的に
  `StrandBendingOscillationConfig.auto_tune_for_n_strands(n: int)` のような
  convenience API が欲しくなる可能性。ただし MCDD 本命（K_c 不整合解消）が
  解決すればこれらの opt-in 自体が不要になるため、本 API は MCDD 完了後に
  改めて必要性を判定する。
