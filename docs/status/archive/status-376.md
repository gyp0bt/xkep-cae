[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-376: 候補 (g2) AL 外側ループ限定再導入 + 19 本実機検証で却下

**日付**: 2026-04-28
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11 passed（status-375 比 +11、新 AL ユニットテスト）

## 概要

候補 (g2) Augmented Lagrangian 限定再導入を実装。`HuberContactForceProcess` に
`set_al_lambda_offset()` / `get_last_p_n_eff()` API を追加し、`p_n_eff = max(0, p_n_huber + λ)` を
f_c に内包（K_geo は pair.state.p_n 経由で自動整合、K_mat は dp_n_huber/du のみで `dλ/du=0` を
反映）。`NewtonDynamicProcess` に AL 外側 for ループ（max `al_n_uzawa_max` cycle）を配線、
内側 NR 収束後に Uzawa 更新 `λ_new = max(0, p_n_eff_converged)` を実施。
法線成分のみ AL 適用（摩擦は status-147 NCP 鞍点系符号問題回避のため対象外）。

**判定: 候補 (g2) AL 再導入 却下**。19 本撚線 90° 曲げで `al_n_uzawa_max ∈ {2, 3}` 掃引、
両ケースで Gate `frac ≥ 0.6` 未達:

| al_n_uzawa_max | frac | incr | cb | elapsed [s] | baseline 比 | Gate |
|----------------|------|------|----|-------------|-------------|------|
| 2 (1 Uzawa 更新) | **0.5746** | 365 | 53 | 1240.22 | **+53.7%** | FAIL |
| 3 (2 Uzawa 更新) | 0.1973 | 84 | 19 | 148.67 | -47.2% | FAIL |

baseline (status-357 19 本): `frac=0.3739`。

**n=2 が候補 (g) サブライン全 7 候補中の最良 19 本実績** (+53.7%) だが gate 0.026 不足。
n=3（2 回目 Uzawa 更新）で過修正発散し early abort。**1〜2 cycle 限定 AL は部分有効だが
MCDD 凍結解除条件未達**で確定。

## 1. 実装

### 1.1 `HuberContactForceProcess` 拡張（`xkep_cae/contact/contact_force/strategy.py`）

| 追加 API | 役割 |
|----------|------|
| `__init__` `_lambda_offset_pairs: np.ndarray \| None = None` | per-pair λ オフセット保持（None で完全無効化、default） |
| `_last_p_n_eff: np.ndarray \| None = None` | 直近 evaluate() の p_n_eff スナップショット |
| `set_al_lambda_offset(lam)` | 外側 AL ループから per-pair λ を設定（None でクリア） |
| `get_last_p_n_eff()` | Uzawa 更新で読む converged p_n_eff スナップショット |

`evaluate()` 内で `p_n_all = np.maximum(0.0, p_n_all + self._lambda_offset_pairs)` を
EMA 平滑化の後に適用。`p_n_eff` を `pair.state.p_n` に書き込むことで K_geo
（`w_geo = p_n/d`）が自動整合（数理台帳 §9.2 参照）。

合計実装: `strategy.py` +43 行（API 4 個 + evaluate 5 行 + docstring）.

### 1.2 NR 配線（`xkep_cae/contact/solver/_newton_dynamic.py`）

既存の `while att + 1 < _effective_max:` インナー NR ループを `for _al_cycle in
range(_al_n_uzawa_max):` 外側ループで包み、各サイクル先頭で `set_al_lambda_offset(λ)`、
収束後に `get_last_p_n_eff()` から `λ_new = max(0, p_n_eff_converged)` で Uzawa 更新。
`NewtonDynamicInput` に 2 field 追加（`al_outer_enabled` / `al_n_uzawa_max`、default OFF）。

NR-local 状態（`_freeze_*`/`_relax_*`/`_consecutive_*`/`_pwise_*` 等）はサイクル境界で
完全リセット。`diag` / `total_attempts` / `n_active` / `f_c` / `_damp_energy_rate_last` は
サイクル跨いで累積保持（最終サイクルの値が DynamicStepOutput に反映）。

合計実装: `_newton_dynamic.py` +37 行（AL setup 18 + Uzawa 更新 19）+ 既存ループ全体を
4 spaces 字下げ.

### 1.3 plumb-through（4 経路 2 field）

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/core/data.py` | `ContactFrictionInputData.al_outer_enabled / al_n_uzawa_max` (default OFF) |
| `xkep_cae/contact/solver/process.py` | `nr_config_dyn` に AL 2 field 貫通 |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `StrandBendingOscillationConfig.al_outer_enabled / al_n_uzawa_max` + 3 経路（曲げ / 揺動 / free_end）で plumb |
| `work/beam_hysteresis/28_al_outer_loop_19strand.py` | 新規 19 本撚線検証スクリプト（97 行） |

### 1.4 単体テスト（`xkep_cae/contact/contact_force/tests/test_strategy.py`）

`TestAugmentedLagrangianOffset` 11 テスト追加:

- `test_default_lambda_is_none`: 既定で AL 無効
- `test_set_al_lambda_offset_stores_copy`: 配列コピー保持（外部破壊耐性）
- `test_set_al_lambda_offset_none_clears`: None でクリア
- `test_lambda_zero_no_op`: λ=0 で raw p_n_huber と同値
- `test_lambda_positive_augments_p_n`: λ>0 で `p_n_eff = p_n_huber + λ`
- `test_lambda_keeps_inactive_pair_active`: gap>0 でも λ>0 で active 化
- `test_lambda_clamped_to_nonneg`: 負 λ で `max(0, ...)` 保護
- `test_get_last_p_n_eff_after_evaluate`: snapshot 取得
- `test_get_last_p_n_eff_before_evaluate_is_none`: 未呼出で None
- `test_lambda_shape_mismatch_skips_offset`: shape 不一致で no-op
- `test_uzawa_update_pattern`: 2 サイクル Uzawa 更新パターン

## 2. 数理台帳更新

`docs/math/03_huber_contact_penalty.md` に **§9 "Augmented Lagrangian 動機と
Uzawa 外側ループ"** を追記（+96 行）:

- §9.1 古典的 AL 定式化と Uzawa 反復（[#eq-al-pn] / [#eq-uzawa]）
- §9.2 K_c 整合性（modified Newton 不要 — `pair.state.p_n` 更新で K_geo 自動整合）
- §9.3 status-221 凍結根拠（n_uzawa_max=1 で純ペナルティと等価 + 摩擦符号問題）
- §9.4 status-376 限定再導入動機（19 本 Type D stall escape 試行）
- §9.5 摩擦接線剛性符号問題（status-147）— 法線のみ AL で構造的回避

## 3. 検証

### 3.1 Default OFF 回帰（gate 必達）

| 項目 | 結果 |
|------|------|
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK |
| `pytest xkep_cae/contact/ xkep_cae/mathematics/` | **588 passed, 5 skipped**（status-375 比 +11 AL テスト） |
| `test_helical_3d_hermite` | rel_err=2.18e-07 維持（status-356 機械精度継続） |
| 7 本撚線 90° 曲げ (`test_7strand_90deg_dynamic_completes`) | **PASS, 8.43s**（frac=1.0 完走、回帰なし） |
| `ruff check` / `ruff format --check` | OK |

### 3.2 19 本撚線実機検証（gate 未達）

`work/beam_hysteresis/28_al_outer_loop_19strand.py` で `al_n_uzawa_max ∈ {2, 3}`:

| ケース | al_n_uzawa_max | frac | incr | cb | elapsed [s] | converged | 判定 |
|--------|---------------|------|------|----|-------------|-----------|------|
| baseline (status-357) | — | 0.3739 | — | — | ~720 | False | — |
| **n=2** | 1 Uzawa 更新 | **0.5746** | 365 | 53 | 1240.22 | False | gate 直前 (-0.026) |
| **n=3** | 2 Uzawa 更新 | 0.1973 | 84 | 19 | 148.67 | False | 過修正発散 |

**観察**:

- **n=2** で +53.7% は **(g) サブライン全候補で最良の 19 本実績**
  （(c) +6.5% / (d) +50.9% / (e) -2.9% / (g1) +37.3% / (g3) -6.9%）
- λ 規模: ||λ|| ~ 6e-4〜9e-4、max(λ) ~ 2.5e-4〜3.4e-4、`n_λ_active=46/3590 pair`
  （Uzawa が誘発する追加接触は ~1.3% pair 程度に局在）
- elapsed 1240s vs baseline ~720s で +72%（NR 内側 ×2 サイクル相当）。コスト見合いで
  改善幅は許容範囲（候補 (g1) α=0.5 の +131% より良い）
- **n=3** は 2 回目 Uzawa 更新で incr=4 直後に発散（早期 abort）。λ 蓄積で
  `R_c=2.96e+04` 規模の残差爆発 → 5 反復連続増加で diverged 検知
- 19 本最終 NR Type 分布 n=2: `D+E:62%, E:12%`（baseline `D+E:67%, E:28%` より
  marginal mixed 減少、AL の active 集合安定化効果は限定的）

### 3.3 物理的解釈

n=2 の +53.7% 改善は、Uzawa 1 回更新で「stall 直前にあった active 集合の振動を
λ で 1 step 平滑化」した効果。これは status-372 候補 (g1) EMA α=0.5（NR 内側
平滑化）と同等の安定化機構だが、AL は **解析ステップ単位**で λ を更新するため
過渡的な振動を逃すことができる。

n=3 の発散は、2 回目 Uzawa 更新が λ を過剰に積み上げて f_c が過大評価され、
NR が探索する解の盆地を変えてしまった結果。これは AL の「λ 漸近収束 →
真の Lagrange 乗数」という古典的性質と整合（n_uzawa_max を増やせば一般に
Uzawa は収束するが、有限の k_pen + 19 本 Type D stall 領域では収束半径が狭い）。

**結論**: AL 外側ループは局所的な NR 安定化機構として有効（n=2 で +53.7%）だが、
**MCDD 凍結解除条件 frac ≥ 0.6 + frac=1.0 完走には不十分**。19 本 Type D stall の
主因（K_c x/z カップリング不整合、status-344 mat_only rel_err mean=44%）を
直接解消しない限り、NR alg 側の escape hatch のみでは frac=1.0 到達は困難と確定。

## 4. MCDD 脱法 pattern 回避

- pattern 1（tol 緩和）: 数値結果は実測値のまま、目標緩和なし
- pattern 5（既存テスト skip）: contact 468 + math 109 全 pass、`test_helical_3d_hermite`
  rel_err=2.18e-07 維持
- pattern 6（骨格 status）: Phase 1+2 を一括実装（API + NR 配線 + 11 単体テスト + 19 本実機）
- pattern 8（baseline ベンチ捏造）: baseline 0.3739 を status-357 から引継ぎ
- pattern 10（次回送り）: n=2 で gate 未達を即却下判定、n=3 過修正も実測で確定

## 5. 引継ぎ（status-377 へ）

### 5.1 候補 (g) サブライン総括

候補 (g) 3 サブライン全て却下が確定:

| 候補 | 19 本 frac | baseline 比 | gate | status |
|------|-----------|-------------|------|--------|
| (g1) active 履歴 EMA α=0.5 | 0.5133 | +37.3% | FAIL | status-371/372 却下 |
| (g3) pair-wise relaxation | 0.3482 | -6.9% | FAIL | status-374/375 却下 |
| **(g2) AL 再導入 n=2** | **0.5746** | **+53.7%** | FAIL | **本 status 却下（最良）** |

### 5.2 次候補: explicit 時間積分（`solver_mode="explicit"`）への移行

NR alg 側 escape hatch（候補 (a)〜(g)）を全て探索完了。**陽解法時間積分への移行が
本命候補**:

- 設計参照: `xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md` §4'
  "solver_mode 併存方針"（status-373 で追記）
- 実装方針: `StrandBendingOscillationConfig.solver_mode: Literal["implicit","explicit"]`
  新設、explicit は dt 制限と引き換えに 19 本以上の K_c 構造的不整合を時間積分自体で
  安定化
- gate: 19 本 frac=1.0 完走（implicit + AL n=2 の 0.5746 を上回ること）

### 5.3 副次（保留）: K_mat の x/z 二次補正項追加

status-370 結果 B（active 境界 FD 機械精度維持）+ 候補 (g) 全却下から、**K_c の
構造的不整合は active 集合変動下では NR alg 側で escape できない**ことが確定。
**MCDD 数理側の追加調査**として K_mat の x/z 成分二次補正項（dp_n/du の二階微分
寄与など）を再検討する選択肢があるが、これは status-353/356 で K_geo ≡ K_mat,ndir 
同一性を確立した数理整理を覆す可能性があり保守的アプローチ。

### 5.4 検証スクリプトの取扱い

`work/beam_hysteresis/28_al_outer_loop_19strand.py`: **gate 直前まで詰めた最良実験記録**
として残置（status-358/360/372 と同方針）。n=2 の log は `/tmp/al_logs/al_n2_19strand.log`、
n=3 は `/tmp/al_logs/al_n3_19strand.log`。

### 5.5 AL field の運用方針

`al_outer_enabled=False` （default）を維持。19 本以上向け **opt-in escape hatch** として
`al_outer_enabled=True, al_n_uzawa_max=2` を `docs/roadmap.md`「撚線規模別 opt-in
チューニング」表に追加（status-368 `chattering_freeze_nr_max=30` / status-372
`active_ema_alpha=0.5` と同レイヤ）。

## 6. 運用所見

### 6.1 AL 実装の数理的綺麗さ

`p_n_eff = max(0, p_n_huber + λ)` の単純な加算でλが f_c / K_c 両方に整合的に伝播し、
modified Newton 補正が不要な点は実装上の大きな利点。`pair.state.p_n` 更新が K_geo
重みに直結するため、tangent 経路を 1 行も触らずに AL を導入できた。これは数理台帳 §4
で確立した「K_geo ≡ -p_n·∂n̂/∂u」の同一性（status-353）が効いた結果。

### 6.2 status-221 削除（凍結ではなく完全削除）からの再導入工数

status-222 で `lam_all` / `UzawaUpdateProcess` / `n_uzawa_max` パラメータが**完全削除**
された後で 5 年（仮想時系列）を経て再導入。Phase 1（API 設計 + 単体テスト）+
Phase 2（NR 配線 + plumb + 19 本検証）を 1 status で完結（status-365/366 ContactDamping
の 2 status 分割と異なる）。理由: AL は HuberContactForceProcess 内部で完結し、別
サブパッケージ（freeze や damping）を新設する必要がなかったため。

### 6.3 候補 (g) クローズの意味

候補 (g) 3 サブライン全却下により、**NR alg 側 escape hatch アプローチは限界に到達**。
次の打ち手は (a) **時間積分方式の根本変更**（陽解法）か、(b) **数理側の K_c 補正**
（修正 K_mat や AL の収束加速変種）に二分される。(a) は実装コストは大きいが直接的、
(b) は MCDD 数理整合性へのリスク。次セッション status-377 で方針決定。
