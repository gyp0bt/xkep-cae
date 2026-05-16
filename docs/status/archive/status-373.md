# status-373: TODO 整理 + 症状緩和系 experiment 5 本削除 + solver_mode 設計追記（documentation status）

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-25
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10 passed（status-372 維持、回帰なし）

## 概要

status-372 までで Phase E 候補 (a)〜(g1) を全て検証完了し、(c)/(d)/(e)/(g1) は
7 本系 opt-in escape hatch として `docs/roadmap.md` 「撚線規模別 opt-in チューニング」
表に取り込み済み、19 本 Type D stall 本体は未解決のまま候補 (g3) pair-wise
relaxation に引き継ぎが決定。本 status は **実装本体無変更**の documentation
status。3 つの書類整理を実施:

1. **症状緩和系 experiment 5 本削除**: 結論が status に確定記録済の失敗・採択
   実験スクリプトを `git rm`
2. **CLAUDE.md TODO 圧縮**: `次の課題`/`凍結中 TODO` を status-index 参照に
   圧縮、現在のアクティブラインのみ残置
3. **`solver_mode` 併存方針 設計追記**: リスタート解析方式（旧「次の課題」第 3 項）
   を `solver_mode` opt-in フラグとして陰解法と併存させる方針を設計仕様化

## 1. 削除した experiment スクリプト 5 本

`work/beam_hysteresis/` 配下で、結論が status に確定記録済かつ再実行価値が
ない症状緩和系実験のみを削除。

| 削除ファイル | 確定 status | 結論 |
|--------------|-----------|------|
| `15_hypothesis_c_7strand.py` | status-358 | 仮説 C 候補 (a) `smoothing_delta=500` 却下記録 |
| `16_hypothesis_c_aprime_19strand.py` | status-360 | 仮説 C 候補 (a') 19 本却下記録（frac -23.1%） |
| `22_bt_parameter_sweep_19strand.py` | status-363 | 候補 (c) line search パラメータ 4 ケース全却下、BT default 局所最適 |
| `25_freeze_param_sweep_19strand.py` | status-368 | 候補 (d) freeze パラメータ 6 ケース、Case B `nr_max=30` のみ +16.6%、frac=1.0 未達 |
| `26_active_ema_alpha_sweep.py` | status-372 | 候補 (g1) α 掃引、7 本 α=0.5 採択方向 / 19 本却下 |

**保持した診断スクリプト**:

- `14_kc_active_boundary_diagnostic.py`（status-370 結果 B 確定の根拠、Phase
  C-3' Step 3.1）
- `14_kc_closest_adj_diagnostic.py`（status-355 Phase C-3' 仮説 B 診断、
  status-356 機械精度達成の検証ライン）

これら 2 本は **MCDD 数理裏付けの再現性確保**に直結するため残置。

## 2. CLAUDE.md TODO 整理

### 2.1 「次の課題」セクション圧縮

旧版の status-284〜372 完了履歴の打消し線リスト（約 90 行）を `<details>` で
折り畳むのではなく、**status-index への参照に置き換え**、現在のアクティブ
ライン 3 項目（status-374 候補 (g3) Phase 1 / status-374 副次 `solver_mode`
実装 / 多 pair 診断スクリプト）のみ箇条書きに残置。

### 2.2 「凍結中の TODO」打消し線 6 項目削除

旧 6 項目（19本 Type D stall K_mat x/z 単発対応 / 7本撚線ピッチ依存性 / ファイバー梁
キャリブレーション / リスタート解析方式 / 被膜圧縮モデル改善 / 空間ブロック分離）
は全て打消し線または status-345 までで列挙済のため、**1 行参照**に圧縮し
凍結解除条件のみ残置。

### 2.3 status-373 エントリ追加 + status-374 引継ぎ更新

`### ★最優先: MCDD ... Phase A〜E` セクション末尾に status-373/status-374 を
追記。

### 2.4 「現在の状態」block 更新

ヘッダ行を **MCDD status-373** に更新、Phase A〜E / status-346〜373 の
**24/N 完了**。

## 3. `solver_mode` 併存方針 設計追記

### 3.1 背景

旧 CLAUDE.md「次の課題」第 3 項に **リスタート解析方式への移行**（動的摩擦
接触ソルバーが `(u, v, a, 接触ペア)` を初期条件として受け取り同型を返す I/O
リファクタリング、CR 梁 UL の `f_int=0` 問題の根本解決）が記載されていたが、
現行陰解法を一気に置換するのはリスクが高い。

### 3.2 方針

**`solver_mode` フラグで現行陰解法とリスタート方式を併存させる**:

- `solver_mode="implicit"`（default）: 現行陰解法、既存挙動と完全互換
- `solver_mode="restart"`（opt-in）: リスタート I/O、解析ステップ単位で
  `(u, v, a, 接触ペア)` 入出力

### 3.3 候補 (g3) との関係

- 候補 (g3) `PairwiseFreezingProcess` は **陰解法側の改善**（NR 反復内介入）
- `solver_mode="restart"` は **解析ステップ間** の I/O 整備
- 両者直交、同時 opt-in 可。default は両方 OFF
- (g3) で 19 本 frac=1.0 達成できればリスタート方式は subsequent 高速化、
  達成できなければ I/O 整備が次の本命候補

### 3.4 反映先

- `xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md` §4'（新規節、
  約 35 行追加）
- `docs/roadmap.md` 「撚線規模別 opt-in チューニング」表に `solver_mode` 行追加
  （status-368 `chattering_freeze_nr_max=30` / status-372 `active_ema_alpha=0.5`
  と同レイヤ）
- 実装は **status-374 以降**

## 4. 実装計画書のレポジトリ内常設化

セッション環境の `/root/.claude/plans/` は揮発するため、本 status の
実装計画書を `docs/plans/status-373-plan.md` として **レポジトリ内に常設**。

## 5. Gate

| 項目 | 結果 |
|------|------|
| `ruff check xkep_cae/ tests/` | OK |
| `ruff format --check xkep_cae/ tests/` | OK |
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK |
| `pytest xkep_cae/contact/` | **456 passed, 5 skipped**（status-372 維持） |
| `pytest xkep_cae/mathematics/` | 109 passed（status-372 維持） |
| `test_helical_3d_hermite` | rel_err=2.18e-07 維持（status-356/372 維持） |
| 実装本体（`xkep_cae/`、`tests/`、`contracts/`） | **無変更**（書類整理のみ） |

実装本体を一切変更していないため、回帰テストは status-372 と完全一致。

## 6. 引継ぎ（status-374 へ）

1. **最優先**: 候補 (g3) Phase 1 — `xkep_cae/contact/freeze/` 新設、
   `PairwiseFreezingProcess` 単体実装 + 単体テスト。設計仕様は
   `xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md` §3.2。
   実装規模見積もり ~150 行（status-365 候補 (e) Phase 1 と同程度）
2. **副次**: `solver_mode` フラグ実装着手 —
   `StrandBendingOscillationConfig.solver_mode: Literal["implicit","restart"]`
   追加、`solver_mode="implicit"` default、リスタート側は別 status で詳細
3. **却下時**: (g3) で 19 本 frac=1.0 未達なら **(g2) AL 再導入**（status-221
   で凍結した Uzawa 外側ループ 1〜2 回限定再導入）に進む
4. **凍結中 TODO 棚卸し**: status-372 と同じ、Phase E 完了 + 19 本 frac=1.0
   完走 + `KcNormalDirectionStiffness` rel_err < 1e-2 を満たすまで全凍結維持

## 7. 運用所見

- **書類整理 status の意義**: TODO リストの陳腐化は「やるべきこと」を不明瞭に
  し、MCDD 脱法パターン 10「TODO として積む」を誘発する。完了履歴の status-index
  集約は trade-off だが、`次の課題` セクションの可読性を回復させる効果が大きい
- **症状緩和系 experiment の取り扱い**: 失敗実験スクリプトを残置すべきか
  削除すべきかは綱引き。本 status では「結論が status 本体に十分記録済 +
  再実行価値が低い」5 本のみを削除。診断スクリプトは保持
- **計画書のレポジトリ内常設**: セッション間で `/root/.claude/plans/` が
  揮発する問題に対して `docs/plans/` に格納する運用パターンを確立
