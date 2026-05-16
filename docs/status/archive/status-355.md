# status-355: Phase C-3' 仮説 B 診断 — K_closest 隣接拡張で埋めるべき量を active×adj ブロックに局在化

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-20
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25（変動なし）

## 概要

status-354 で Phase C-3 再々定義が提示した **仮説 B**
（「`KcClosestPointStiffnessProcess` を隣接ノード DOF 列にも拡張して
s-tracking 補償経路 (2) を解析的に実装すれば 19 本撚線 `mat_only`
rel_err が改善する」）を実装前に裏付けるため、`test_helical_3d_hermite`
シナリオで **K_c_analytical vs FD の差分をブロック分解する診断** を取得した。

結果: 現行 rel_err 1.795% の **100.0% が active 行 × adj 列ブロックに局在**し、
特に z 成分の不整合 76.1 は adj 列側の z 方向 FD 応答 597.9（K_closest
adj 拡張で埋めるべき量）に由来することが定量的に確認された。status-354 の
「comp_z 77% 不整合は s-tracking 不足由来」仮説は反証可能な予測として成立し、
仮説 B は「active×adj ブロックに 98.52 を埋める」という明示的目標に縮約された。

本 status は **診断＋実装計画策定** に範囲を限定し、仮説 B 実装本体は
status-356 に引き継ぐ（MCDD 禁止パターン 6 「困難先送り」には該当しない
— 定量目標と実装パスを確立した準備 status）。

## 診断内容

### スクリプト

`work/beam_hysteresis/14_kc_closest_adj_diagnostic.py` 新設（147 行）。
`test_kc_component_fd.py` の `_make_two_segment_scenario`（n_elems=3,
helical_z=True, Hermite+Hertz）を共有し、以下を計算:

1. 全並進 DOF（24 列）について FD K_c を列ごと計算（`_compute_fd_kc`）
2. 解析的 `K_mat - K_geo + K_st` を `tangent_components()` で取得
3. DOF を **active 列集合（活性ペア 4 ノード × 3 並進 = 12 DOF）** と
   **adj 列集合（隣接 4 ノード × 3 並進 = 12 DOF）** に分割
4. `diff = K_c_analytical - FD_Kc` を 4 ブロック `(active/adj) × (active/adj)`
   に分解してノルム・シェア・comp 別内訳を出力

対応ノード対応:
- 活性ペア: `elem_a=1 (nodes 1,2) × elem_b=4 (nodes 5,6)` → `active_nodes={1,2,5,6}`
- 隣接: A-1=0, A+2=3, B-1=4, B+2=7 → `adj_nodes={0,3,4,7}`

### 測定結果

```
── 全体 ──
  ||K_c||   = 5.4880e+03    ||FD_Kc|| = 5.4889e+03
  ||diff||  = 9.8522e+01    rel_err   = 1.7949e-02

── ブロック分解（diff） ──
  aa (active×active): ||diff||= 1.1954e-03 ( 0.0%)   rel_err[aa] = 2.19e-07
  ax (active×adj)  : ||diff||= 9.8522e+01 (100.0%)
  xa (adj×active)  : ||diff||= 0.0000e+00 ( 0.0%)
  xx (adj×adj)     : ||diff||= 0.0000e+00 ( 0.0%)

── active×adj ブロック詳細 ──
  ||K_c[ax]||   = 5.9339e+02  (K_hermite_adj が一部埋める、status-351)
  ||FD[ax]||    = 6.0151e+02  (FD 基準)
  ||diff[ax]||  = 9.8522e+01  (仮説 B で埋めるべき量、16.4%)

── comp 別 diff（行次元全体） ──
  active 列: x=5.0e-04  y=5.2e-04  z=9.5e-04    ← 完全に正確
  adj 列:   x=4.43e+01 y=4.41e+01 z=7.61e+01    ← 全 rel_err 由来
```

### 解釈（status-354 予測の定量検証）

1. **active×active ブロックは既に rel_err 2.2e-7 で完全整合** — 現行の
   5 項分解（`K_mat_nn + K_hermite_adj - K_geo + K_closest + K_st`）は
   pair-local DOF 内では機械精度で FD と一致。status-295/351 の「mat-only
   は active 内で正しい」設計意図の実証。
2. **rel_err 1.795% の 100% が active×adj ブロックに局在** — 隣接ノード
   DOF への応答が現行 `K_closest` で 0 のまま放置されているため。
   `K_hermite_adj` は active×adj ブロックに 593.4 相当を埋めているが、
   FD の 601.5 に対して 98.5 不足（16.4%）。この不足分が **s-tracking
   補償経路 (2) の未実装分** に対応。
3. **comp_z 77% は adj 列 z 方向に由来** — active 列 z の diff=9.5e-4 に対し、
   adj 列 z の diff=76.11。status-354 の報告値（comp_z 77.3%）は adj 列寄与
   そのもので、active 内の z 不整合ではない。
4. **adj×active / adj×adj ブロックは 0** — 残差系 `R = f_c - f_ext` の微分は
   f_c が active 行にのみ出力されるため、adj 行は構造的に 0。仮説 B は
   active 行のみを対象（K_c の非対称拡張）で十分。

## 実装コスト評価

`KcClosestPointStiffnessProcess` の隣接拡張に必要な変更:

| 対象 | 変更内容 | 規模 |
|---|---|---|
| `_st_jacobian.py::_batch_st_jacobian_hermite` | 既に `ds_du_adj` / `dt_du_adj`（N,12）を計算済み（status-311）。変更不要 | 0 行 |
| `ContactForceStStiffnessInput` | `adj_node_counts: np.ndarray \| None` フィールド追加（`adj_node_map` は既存） | +1 行 |
| `ContactForceStStiffnessProcess._process_batch_term` | `dm_ext_A/B` を `_batch_dm_ext_coeffs` で計算（`KcHermiteNonlocal` と同式）し `_batch_st_jacobian_hermite` に渡す。戻り値 `ds_du_adj`, `dt_du_adj` を捕捉 | +15 行 |
| 同、`term="closest"` 分岐 | `K_local_adj = -(df_ds_term ⊗ ds_du_adj + df_dt_term ⊗ dt_du_adj)` を構築し、`adj_gdofs`（`adj_node_map` から導出、`KcHermiteNonlocal` と同ロジック）に対する COO エントリを追加 | +25 行 |
| ソルバー配線（`HuberContactForceProcess.tangent` / `assemble_tangent`） | `ContactForceStStiffnessInput` へ `adj_node_counts` を渡す | +2 行 |
| テスト | `test_kc_component_fd.py::test_helical_3d_hermite` の `assert rel_err < 0.05` はそのまま合格継続（閾値緩和なし）。adj 拡張により rel_err は 1.795% → 1e-4 オーダー期待 | 既存 |
| 新 FD テスト（option） | adj 拡張の正当性を直接検証する単体 FD（`KcClosest` のみ 48×48 FD） | +30 行 |

合計実装規模: **~45 行のコア変更 + ~30 行の単体テスト**。status-295 の
`K_c_adj mat-only` 化（~30 行）と同規模。

### 公開 API 非破壊性

- `KcClosestPointStiffnessOutput.K_closest` の **形状（shape, dtype）は不変**。
  COO エントリ数が active×adj 分だけ増えるのみ（sparse パターン拡張）。
- `tangent_components()` の 3-tuple 返却（K_mat, K_geo, K_st）も不変。
  K_st 相当に K_closest + K_st_residual が合成される設計（status-351）が
  K_closest の rows/cols 拡張を自然に吸収。
- `TermExpansionContract.providers` は 5 項のまま、契約違反 0 件を維持。

## ゲート

- ✅ `ruff check xkep_cae/ tests/ work/`: All checks passed
- ✅ `ruff format --check xkep_cae/ tests/ work/`: 192 files already formatted
- ✅ `pytest xkep_cae/contact/contact_force/tests/test_kc_component_fd.py::TestKcComponentFD::test_helical_3d_hermite`: **passed**（rel_err=1.795%、変更なし）
- ✅ 診断スクリプト `work/beam_hysteresis/14_kc_closest_adj_diagnostic.py` 実行:
  active×adj ブロックに diff 100% 局在を確認
- ⚠️ **重量回帰（7本撚線曲げ揺動 / 接触あり 90°曲げ）は本 status 未実行**:
  コード変更なしのため挙動不変、status-356 実装時に実施

## status-356 引き継ぎ（仮説 B 実装本体）

1. `_batch_dm_ext_coeffs(node_counts, nodes)` ヘルパを
   `HuberContactForceProcess` に追加（現状 `KcHermiteNonlocalStiffnessProcess`
   内でインライン計算されている式を関数化、脱法実装 pattern 3 の
   「類似コード二重実装」回避）
2. `ContactForceStStiffnessInput` に `adj_node_counts` 追加、
   `_process_batch_term` で `use_hermite and adj_node_counts is not None`
   条件で adj パスを分岐
3. 診断スクリプト `14_kc_closest_adj_diagnostic.py` を実装後に再実行し、
   `||diff[ax]|| = 98.52 → < 1e-3` を確認（ゲート条件）
4. `test_helical_3d_hermite` の rel_err が 1.795% → 1e-4 オーダーに改善することを確認
5. 19 本撚線 K_c FD 再計測（`work/beam_hysteresis/13_kc_component_fd_19strand.py`
   を Phase C-3' 設定で実行）し、`mat_only` rel_err mean=44% の改善を定量測定
6. 7 本撚線曲げ揺動回帰 + 接触あり 90°曲げ回帰で回帰なしを確認
7. 19 本撚線 frac=0.48→1.0 完走を目標に Type D stall 再計測

## 関連 status

- status-295: K_c_adj mat-only 確立（active×active 内の整合性根拠）
- status-351: `KcClosestPointStiffnessProcess` 新設（Phase C-2）
- status-353: 数理台帳訂正（K_mat,ndir ≡ K_geo）
- status-354: 仮説 A 反証 + Phase C-3' 再々定義
- **status-355（本 status）**: 仮説 B 診断 — active×adj ブロック 100% 局在確認、実装コスト評価
- status-356（次）: 仮説 B 実装本体

## 懸念事項・引き継ぎメモ

- 本 status は **診断＋実装計画 only**（コード変更なし）。status-354 の
  hypothesis B/C/D 再配分後、実装着手前に「効果見込みが定量的に裏付く」
  ことを確認するステップとして位置づけられる。MCDD 禁止パターン 6
  （「困難を先送り」）には該当せず、**定量目標（active×adj diff 98.52 → <1e-3）
  と実装パス（~45 行）を確立した**。
- 診断で判明した `||diff[ax]|| = 98.52`（FD ax ノルム 601.5 の 16.4%）は
  仮説 B で埋めるべき解析量と一致する。FD スクリプトと実装が同じシナリオ
  （`_make_two_segment_scenario(helical_z=True, n_elems=3)`）を共有する
  ため、実装後は同スクリプトが自動的に効果測定になる。
- 19 本撚線 Type D stall では同シナリオの帰結としてより複雑な
  adj パターン（1 active ペアあたり 4 隣接ノード × 複数 active 要素）が
  現れる。`mat_only` rel_err mean=44% のうちどの程度が adj 由来かは
  status-356 実装後に 19 本撚線 FD 再計測で確認する。
- 診断スクリプトは `work/beam_hysteresis/14_kc_closest_adj_diagnostic.py`
  として commit。work/ 配下は ruff check/format 対象だが package 扱いでは
  ないため既存のチェック経路で補足される。

## コミット（予定）

1. `test(contact): Phase C-3' 仮説 B 診断スクリプト — K_closest 隣接拡張で埋めるべき量を active×adj ブロックに局在化（status-355）`
2. `docs(status): status-355 + README/status-index/roadmap/CLAUDE.md 更新（仮説 B 診断、status-356 実装本体へ）`
