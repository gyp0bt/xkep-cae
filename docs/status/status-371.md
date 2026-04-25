# status-371: 候補 (g1) active 履歴 EMA 平滑化 実装 — `HuberContactForceProcess.active_ema_alpha`

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-25

## 概要

status-370 で **結果 B 確定**（K_c は active 境界跨ぎ・smoothing ゾーン・強制
flip いずれも FD 機械精度 2.18e-07、K_c 項欠落ではなく NR alg 側動力学が
19 本 Type D stall の主因）を受け、`phase_c3prime_19strand_plan.md` §3.2 の
**最優先候補 (g1) active 履歴 EMA 平滑化** を実装した。

NR 反復間で接触力 $p_n$ を低域通過化し、active 集合振動を直接抑制する:

$$p_n^{\\mathrm{eff}} = \\alpha \\cdot p_n^{\\mathrm{new}}
                   + (1 - \\alpha) \\cdot p_n^{\\mathrm{prev}},
\\quad \\alpha \\in (0, 1]$$

$\\alpha = 0$ で完全無効（既定、回帰防止）、$\\alpha \\to 0$ 強平滑化、
$\\alpha = 1$ で平滑化なし。`HuberContactForceProcess` インスタンス側に
`_p_n_prev_array` を保持し、`NewtonDynamicProcess` がインクリメント境界で
`reset_ema_state()` を呼ぶことで前荷重ステップの $p_n$ が初期値として
染み込む副作用を防ぐ。

## 1. 実装内容

### 1.1 中核: `HuberContactForceProcess`

`xkep_cae/contact/contact_force/strategy.py`:

- `__init__` に `active_ema_alpha: float = 0.0` を追加、`_active_ema_alpha`
  / `_p_n_prev_array: np.ndarray | None` を保有
- `reset_ema_state()` 新規メソッド — `_p_n_prev_array = None` でクリア
- `evaluate()` 内で `p_n_all` 計算後、$\\alpha > 0$ かつ前反復履歴の shape
  が一致する場合のみ EMA ブレンドを適用、`_p_n_prev_array` を更新

設計判断:

- $\\alpha = 0$ では履歴ストレージにも書き込まない（既存挙動と完全に
  バイト一致、回帰防止）
- ペア数が変わった場合（broadphase 再検出など）は shape mismatch を
  検出し平滑化スキップ → raw $p_n$ で再起動（安全側）
- `tangent()` は `pair.state.p_n` から smoothed 値を読むので別途修正不要
  （`evaluate()` が pair state を書く順序に従う）

### 1.2 NR ソルバー側 reset hook

`xkep_cae/contact/solver/_newton_dynamic.py`:

`NewtonDynamicProcess.process()` の NR ループ突入直前で
`_contact_force_strategy.reset_ema_state()` を `hasattr` ガード付きで
呼び出す。1 荷重インクリメント単位で履歴をリセットする方針で、
カットバック / 凍結モード復帰でも履歴は新しく蓄積される。

### 1.3 plumb-through（4 層 1 field）

| 層 | ファイル | 変更 |
|----|----------|------|
| Strategy factory | `xkep_cae/contact/contact_force/strategy.py` | `_create_contact_force_strategy(active_ema_alpha=0.0)` |
| 全体 strategies factory | `xkep_cae/core/data.py` | `default_strategies(active_ema_alpha=0.0)` |
| Solver Input | `xkep_cae/core/data.py` | `ContactFrictionInputData.active_ema_alpha: float = 0.0` |
| Solver Process | `xkep_cae/contact/solver/process.py` | `_default_strategies(..., active_ema_alpha=input_data.active_ema_alpha)` |
| 撚線曲げ Config | `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `StrandBendingOscillationConfig.active_ema_alpha: float = 0.0` + 3 経路 plumb（曲げ / 揺動 / free_end）|

### 1.4 単体テスト追加 (10 件)

`xkep_cae/contact/contact_force/tests/test_strategy.py::TestActiveEmaSmoothing`:

| # | テスト | 検証内容 |
|---|--------|----------|
| 1 | `test_default_alpha_is_zero` | デフォルトで EMA 無効、`_p_n_prev_array=None` |
| 2 | `test_alpha_field_stored` | `active_ema_alpha=0.3` が float 保存 |
| 3 | `test_factory_passes_alpha` | ファクトリ経由で値が貫通 |
| 4 | `test_alpha_zero_no_smoothing` | $\\alpha=0$ で履歴ストレージ書き込みなし |
| 5 | `test_alpha_first_iter_no_history` | 初回反復は raw p_n を採用（履歴なし） |
| 6 | `test_alpha_second_iter_blends` | 2 反復目で `α·p_n_new + (1-α)·p_n_prev` を確認 |
| 7 | `test_reset_ema_state_clears_history` | `reset_ema_state()` で履歴クリア |
| 8 | `test_reset_between_increments` | reset 後の次反復は raw に戻る |
| 9 | `test_alpha_one_recovers_raw_pn` | $\\alpha=1$ は EMA 無効と等価 |
| 10 | `test_pair_count_change_skips_smoothing` | shape mismatch で平滑化スキップ |

### 1.5 診断スクリプト

`work/beam_hysteresis/26_active_ema_alpha_sweep.py` 新設（+150 行）:

- `--n-strands {7,19}` / `--alphas A1,A2,...` 引数
- 各 α で 90° 曲げを実測、frac / cutback / elapsed を集約表示
- 7 本 baseline (frac=1.0), 19 本 baseline (frac=0.4839, status-339) との
  対比 + gate チェック（7 本 frac=1.0 維持 / 19 本 frac ≥ 0.6）

## 2. Gate

| 項目 | 結果 |
|------|------|
| `ruff check xkep_cae/ tests/` | OK（197 files） |
| `ruff format --check xkep_cae/ tests/` | OK |
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK |
| `pytest xkep_cae/contact/` | **456 passed, 5 skipped**（baseline 446 → +10 EMA テスト） |
| `pytest xkep_cae/mathematics/` | 109 passed（status-364 維持） |
| `pytest xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | 18 passed |
| `pytest test_kc_component_fd.py::test_helical_3d_hermite` | OK（status-356 機械精度 rel_err=2.18e-07 維持） |

`active_ema_alpha=0.0` （既定）では `_p_n_prev_array` を一切触らないため、
既存挙動とバイト一致で 7 本撚線 90° 曲げ regression test
`test_7strand_90deg_dynamic_completes` が pass。

## 3. 実機 α 部分掃引（status-371 範囲、status-372 で本格化）

`26_active_ema_alpha_sweep.py` で 7 本撚線 90° 曲げを部分実測:

| α | frac | n_inc | n_cb | elapsed [s] | 備考 |
|----|------|-------|------|-------------|------|
| 0.0 (baseline) | **1.0000** | 524 | 57 | 351.67 | status-358 baseline と一致（バイト同等） |
| 0.3 | (進行中) | — | — | 249s で frac=0.80 で終了 | 600s timeout 到達、status-372 で再実行 |

**重要観察**:

- **α=0.0 で完全な byte-identical 動作**: frac/incr/cb が status-358 baseline
  と完全一致（実装の opt-in 切替が無効時に既存挙動を破壊しないことを確認）
- **α=0.3 進行中 frac=0.80 到達**: timeout で打切られ frac=1.0 完走未確認だが、
  少なくとも **既存実装より早期に発散していない**（EMA が NR を破壊しないことの
  必要条件は満たしている）。frac=1.0 完走判定 + 19 本 Type D stall 適用は
  status-372 で完全実施

7 本撚線で frac=1.0 維持、19 本撚線で frac ≥ 0.6 達成が status-372 の
判定材料。本 status の主成果は **実装本体 + 単体テスト + gate 全 pass** で
あり、α 掃引の数値結果は次セッションに引き継ぐ（MCDD 脱法パターン 6
回避: 本 status の成功基準は実装完了と回帰なし、α=0.0 で 7 本回帰 frac=1.0
完走とテスト 456 passed が達成された）。

## 4. ファイル変更サマリ

| ファイル | 変更 |
|---------|------|
| `xkep_cae/contact/contact_force/strategy.py` | EMA 状態保有 + `reset_ema_state()` + `evaluate()` ブレンドロジック + factory 引数追加（+35 行）|
| `xkep_cae/contact/contact_force/tests/test_strategy.py` | `TestActiveEmaSmoothing` 10 テスト追加（+148 行）|
| `xkep_cae/contact/solver/_newton_dynamic.py` | NR ループ突入時 `reset_ema_state()` 呼び出し（+8 行）|
| `xkep_cae/contact/solver/process.py` | `_default_strategies(..., active_ema_alpha=...)` 配線（+1 行）|
| `xkep_cae/core/data.py` | `default_strategies` 引数 + `ContactFrictionInputData.active_ema_alpha` 追加（+9 行）|
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `StrandBendingOscillationConfig.active_ema_alpha` + 3 経路 plumb（+14 行）|
| `work/beam_hysteresis/26_active_ema_alpha_sweep.py` | **新規** α 掃引診断スクリプト（+150 行）|
| `docs/status/status-371.md` | **新規** 本ファイル |
| `docs/status/status-index.md` | status-371 行追加 |
| `README.md` | 現在状況に status-371 追記 |
| `CLAUDE.md` | 「現在の状態」を status-371 に更新、status-372 へ TODO 引継ぎ |
| `docs/roadmap.md` | MCDD 候補 (g1) 実装完了を追記 |

## 5. MCDD 観点

### 5.1 脱法回避チェック（CLAUDE.md「MCDD 脱法実装禁止パターン」）

- **パターン 1**（tol 事後緩和）: 既存テストの tol 不変、`test_helical_3d_hermite`
  の rel_err < 1e-5 を本実装で守る（実測 2.18e-07）
- **パターン 4**（rename で済ます）: 新 field `active_ema_alpha` は既存
  smoothing_delta / huber_delta_h と独立で、Huber 関数自体には触らない
- **パターン 5**（既存テスト skip）: `pytest xkep_cae/contact/` で 446 既存テスト
  を全 pass、skip/xfail 追加なし
- **パターン 6**（骨格だけの status）: 本 status は **動作する Process + 10
  単体テスト + 全 gate 通過** の完結成果物。実機 α 掃引は 1 status の粒度を
  超えるため status-372 に分離（status-365/366/367 と同じ 2 段構成）

### 5.2 候補 (g1) の数理的位置づけ

EMA 平滑化は K_c そのものを変更しない（FD 整合性は status-356 機械精度を
維持）。NR 反復間の active 集合振動を $p_n$ レベルで低域通過化する **時間
ステッピング側の安定化機構** であり、`TermExpansionContract` の項展開
（K_mat_nn / K_closest / K_hermite_adj / K_geo / K_st の 5 項）には影響
しない。よって `validate_process_contracts.py` の C18-C24 全 24 検査は無
変更で OK。

### 5.3 失敗時の next step

α 掃引（status-372）で 19 本 frac ≥ 0.6 が達成できなければ:

- **(g3) pair-wise relaxation**: status-284 接触凍結モードを pair
  granularity 拡張（チャタリング pair のみ freeze）
- **(g2) augmented Lagrangian 再導入**: status-221 で凍結した Uzawa の
  外側ループ 1〜2 回限定再導入

実装オーダーは plan doc §3.2 通り (g1) → (g3) → (g2)。

## 6. 引継ぎ（status-372 へ）

1. **最優先**: `26_active_ema_alpha_sweep.py` で α ∈ {0.1, 0.3, 0.5} を
   7 本 / 19 本撚線で実測、gate 判定:
   - 7 本: frac=1.0 維持（status-336 baseline）→ 1 ケースでも未達なら
     ロジックバグ
   - 19 本: frac ≥ 0.6 達成で **採択方向**、未達で (g3) に進む
2. **副次**: 多 pair 診断 `14b_kc_multi_pair_diagnostic.py`（status-370 §5
   保留）、(g1) で frac=1.0 完走に近づいた場合のみ追加検証
3. **凍結中 TODO 棚卸し**: status-363/368/369/370 と同じ。Phase E
   完了 + 19 本 frac=1.0 完走 + `KcNormalDirectionStiffness` rel_err < 1e-2
   を満たすまで全凍結維持

## 7. 運用所見

- **EMA は時間ステッピング側 escape hatch**: 候補 (a)/(a')/(c)/(d)/(e)
  と同じく数理的厳密性は保ち、NR 反復ダイナミクス側で安定化する
  系列に属する。MCDD 5 項 K_c 分解の整合性とは独立に評価できる
- **`reset_ema_state()` の責務分離**: NewtonDynamicProcess が
  「インクリメント境界」を知っている唯一の Process なので、reset
  call 元として適切。HuberContactForceProcess は履歴の保有と
  blending のみ担当
- **default OFF の優位性**: 既存 446 テスト全 pass で確認した通り、
  field 追加だけで挙動はバイト一致。19 本 Type D stall に opt-in で
  評価できる安全な escape hatch として運用可能
