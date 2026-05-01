# xkep-cae コーディング規約

## 基本

- 全ての回答・設計仕様は**日本語**で記述
- markdown 文書には `README.md` へのバックリンクを貼る
- lint/format: `ruff check xkep_cae/ tests/` && `ruff format xkep_cae/ tests/`
- 機能は可能な限りprocessクラスとして実装すること。

## 2交代制運用（Codex / Claude Code）

常に互いへの引き継ぎを想定。statusファイルに状況を詳細記録。

### ステータス管理

- `docs/status/status-{index}.md` に記録（index最大が現在の状況）
- `docs/status/status-index.md` に一覧管理
- status に書いた内容は **commit メッセージと整合**を取る
- **アーカイブルール**: アクティブ status は最大 **50 件**（status-{最新-49} 以降）を維持。超過時は最古バッチを `docs/status/archive/` へ移動し、`status-index.md` にマイルストーン要約行を残す（STA2 トレーサビリティ維持）

### 作業完了時の必須手順

1. README.md 更新 → 2. status 新規作成/更新 → 3. status-index.md 更新 → 4. roadmap.md 更新
5. 不整合はその場で修正 or TODO追加 → 6. feature ごとにコミット → push

### ログ出力ルール

- 計算実行は**必ず tee でファイル出力**: `python script.py 2>&1 | tee /tmp/log-$(date +%s).log`
- `| tail -N` のみは禁止（途中経過が残らない）
- 収束ログには以下を含める: 時間増分カットバック、接触チャタリング、エネルギー収支、条件数

## ソルバー診断ログ規約（status-307）

**ログ情報は開発の根幹。判断が曖昧にならない出力を厳守。**

### 必須出力項目
- **`[f_ref]`**: NR初回反復でf_ref値と判定モード（dynamic_ref/f_ext）を出力。残差の絶対水準が不明な状態を排除
- **`[CUTBACK:原因]`**: カットバック時に原因タグ（nr_limit/diverged/relax_fail/solve_fail）+ dt値を出力。対策の方向性を即判断可能に
- **`[SPIKE]`**: NR残差が前回比10倍以上増加した際に5反復刻みを待たず即時出力。転換点の見逃しを防止
- **`[coat]`**: 被膜あり時、50ステップごとに圧縮統計（mean%, max%, n_penetrated）を出力。芯線貫入発生時は即時出力
- **`[収束型統計]`**: 解析完了サマリでforce/disp/energy収束の分布を出力。変位収束偏重は力未収束の警告

### 出力設計の原則
- **対策が一意に決まる情報を出力する**: 「不収束」ではなく「不収束:nr_limit（反復数不足）」
- **分母を必ず示す**: `||R||/||f||=3e-4` だけでなく `f_ref=1.23e+03` も出力
- **異常検知は即時出力**: 5反復刻みの定期出力に依存せず、閾値超過時にリアルタイム出力
- **統計はサマリで集約**: 毎ステップの被膜統計は冗長。50ステップ刻み＋異常時の2段構成

## 新機能の収束検証フロー（厳格化）

**原則: 新機能の収束テストは pytest で実行する。必要に応じて `contracts/` に検証スクリプトを配置。**

1. **テストで検証**: `tests/` に正式テストを追加
   - tee でログファイル出力必須
   - 収束後は3D梁形状の2D投影スナップショットで物理的妥当性を目視確認
   - 判断材料: カットバック回数、接触状態変化、エネルギー収支、条件数
2. **視覚検証**: 変形メッシュの2D投影図をdocs/verification/に保存

## テストの分類

### プログラムテスト（API・収束）
- ソルバー収束、例外発生、API仕様準拠
- **16要素/ピッチ以上**厳守
- クラス名: `Test〇〇API`, `Test〇〇Convergence`

### 物理テスト（物理的妥当性）
- 貫入量、応力連続性、荷重オーダー、変形対称性、エネルギー保存
- クラス名: `Test〇〇Physics`

## 互換ヒストリー

移行完了。`__xkep_cae_deprecated/` は status-207 で完全削除。
詳細な移行履歴は status-107〜206 を参照。

## 推奨ソルバー構成

- Fischer-Burmeister NCP（Huber）が主力接触力評価
- UL+NCP統合: `ul_assembler` + `adaptive_timestepping=True`
- 解析的接線剛性: `analytical_tangent=True`（デフォルト）
- Line-to-line Gauss積分 + 同素線除外（`exclude_same_strand=True`）
- **摩擦あり**: `contact_mode="smooth_penalty"`（必須。NCP鞍点系は摩擦接線剛性符号問題で発散: status-147）
- **Uzawa凍結**: `n_uzawa_max=1`（純粋ペナルティ。拡大ラグランジアンは status-221 で凍結）

## 現在の状態

**459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11 テスト** — 2026-05-01 | 契約違反 **0件** | 条例違反 **0件** | **MCDD status-385（候補 (z1c) 2 段階質量スケーリング API（β_stiff + β_outside）実装 — API 完成、validation で β_stiff cap が支配的と確認、(z1d) loading rate 縮小が必須と判明）**。status-384 §6.1 最有力候補 (z1c) として `ExplicitCentralDifferenceProcess` に `mass_scaling_beta_outside` 引数 + `set_mass_scaling_beta_outside()` API（KE 保存 v/a リスケール対応）を追加。`_compute_scaled_mass()` で mask=False の DOF（梁）に β_outside² を、mask=True の DOF（stiff）に β² を適用。`_explicit_dynamic.py` の dt_c_beam 推定で mask 設定時は `β_outside` を乗じる。`ContactFrictionInputData` / `StrandBendingOscillationConfig` 各 1 field + 3 経路 plumb-through。**+11 単体テスト**（`TestTwoStageMassScaling`）全 pass。**`38_z1c_two_stage_validation.py` 8 ケース実機検証**: API は設計通り動作（log で post-cutback target β が β_outside=10 で 8.8e6 → 8.8e5 に **10x 縮小**）も、initial target β=4.7e4（β_stiff cap=1e3〜1e4 を超過）が支配的で全 explicit ケース frac=0 で divergence。aggressive scaling（β_outside=10, β_stiff_max=1e6, α=10）で frac=0.425 進むも max\|u\|=1.6e5mm で精度 gate 完全違反。**結論**: (z1c) infrastructure は完成、しかし MCDD 凍結解除条件 (5) 達成には **(z1d) `t_cycle` 下限緩和** で loading rate を物理 T1 ベースに縮小し target β 自体を下げる必要がある。次候補は (z1d) 最優先 / (z2) Cosserat 梁プロトタイプ並行検討。回帰: 全 24 契約検査 OK / contact + math + time_integration + strand_bending_osc = **737 passed 5 skipped**（status-384 比 +11）/ `test_helical_3d_hermite` rel_err=2.18e-07 維持 / 7 本 implicit frac=1.0 / ruff pass。Phase A〜E / status-346〜385 の **36/N 完了**.

前 status: status-384（候補 (z1a) 要素ごと波速 Δt + (z1b) selective mass scaling 実装 — Abaqus/Explicit 標準アプローチへの移行 Phase 1 完了、validation で 2 段階スケーリング要件を発見）。ユーザーから「応力波の速度と要素サイズから dt 目安を決められる（Abaqus 様式）」+「Cosserat 梁の大回転ネイティブ特性」という根本的指摘を受け、status-383 までの "explicit + UL は原理的に成立しない" 結論を踏まえて方針転換。**(z1a)** `_estimate_critical_dt_per_element(connectivity, node_coords, beam_E, beam_rho)` 新設で `dt_e = L_e / √(E/ρ)` を要素ごとに計算し Gerschgorin 全体上界と min を取る。**(z1b)** `_detect_stiff_dofs()` で Gerschgorin row-sum / M が median × `threshold_ratio` を超える DOF を自動検出、`ExplicitCentralDifferenceProcess.set_mass_scaling_dof_mask()` で β² 倍化を限定。`_compute_scaled_mass(beta)` ヘルパで mask 反映、`set_mass_scaling_beta()` の v/a rescale を mask 対応。**+17 単体テスト**（per-element dt 6 + stiff DOF detect 5 + selective scaling 6）全 pass。**実機検証**: 単梁 90° 曲げで K がほぼ一様 → stiff DOF 検出ゼロ → 実質 β=1 → frac=0 発散（実装 bug ではなく selective が heterogeneous K を要求する性質）。7 本撚線で stiff DOF 112/714 検出 (15.7%) も、残り 84% の beam DOF が β=1 のまま dt 1.6μs 制約を支配し target β=8.8×10⁶（cap 1000 を超過）で frac<<1.0。**真の解**: 2 段階スケーリング（β_stiff=1000, β_beam=10）+ loading rate 縮小の組合せ。次候補は (z1c) per-DOF β 配列 API + (z1d) `t_cycle` 下限緩和 + (z2) Cosserat 梁プロトタイプ並行検討。回帰: 全 24 契約検査 OK / contact + math + time_integration + strand_bending_osc = **726 passed 5 skipped**（status-383 比 +17）/ `test_helical_3d_hermite` rel_err=2.18e-07 維持 / 7 本 implicit frac=1.0 / ruff pass。Phase A〜E / status-346〜384 の **35/N 完了**.

前 status: status-383（候補 (q1) `explicit_ul_update_interval` 実装 — 4 ケース掃引で却下、UL 凍結が真因と再確証、精度 gate 未達のまま）。status-382 §6.1 最有力候補として `solver_mode="explicit"` のとき UL `update_reference()` を **N 増分ごと** に呼出するよう gate を導入。`ContactFrictionInputData.explicit_ul_update_interval: int = 1` field 追加（default 1 で既存挙動完全不変）、`StrandBendingOscillationConfig` に同 field + 3 経路 plumb-through、process.py 主ループ内 update_reference 呼出箇所に `(_next_incr % interval == 0)` ゲート追加。**+5 単体テスト**（`TestExplicitULUpdateInterval`、`_MockULAssembler` で update_reference 呼出回数を直接計測、interval ∈ {1, 2, 100, 0} の挙動 + implicit short-circuit）。**`36_explicit_ul_interval_validation.py` 5 ケース掃引で全 FAIL** — interval=1 baseline 29.57mm（status-382 と一致、default 完全保持）/ interval=5 で relax phase 発散 (NaN) / interval=10 max\|u\|=6.21e6 mm / interval=20 max\|u\|=5.16e21 mm。**根本要因**: CR 梁 UL 定式化は「u_incr 微小」前提で線形化、N 増分蓄積は K_T(u_incr) を線形化レンジ外へ押し出し explicit dynamics が爆発的発散。status-382 §3 解析と整合：(a) update 毎呼出 → f_int(u_incr)≈0、(b) update 間引き → K_T(u_incr) 線形化崩壊、両方破綻。**MCDD 凍結解除条件 (5)「精度 < 10%」未達のまま**。次候補は (q2) 増分内 sub-cycling（最有力、UL 動作は通常通り保持）/ (q3) implicit + AL n>2 復活 / (h5) bending 段階処方。回帰: 全 24 契約検査 OK / contact + math + time_integration + strand_bending_osc = **709 passed 5 skipped**（status-382 比 +5）/ `test_helical_3d_hermite` rel_err=2.18e-07 維持 / 7 本 implicit frac=1.0 / ruff pass。Phase A〜E / status-346〜383 の **34/N 完了**.

前 status: status-382（候補 (p3) damping + (p1) relax API 実装 — UL update_reference 凍結が真の根本原因と判明、精度 gate 未達のまま）。status-381 §7「explicit 解を implicit / 解析解と一致させる」の最優先 TODO に対し仮説 (p3) 質量比例 Rayleigh damping + (p1) BC 完了後 relax phase の 2 API を実装: `ExplicitCentralDifferenceProcess.mass_proportional_damping_alpha` 引数追加（`a -= α·v`、Courant/β 独立）、`ContactFrictionProcess` 末尾に relax phase 追加。`ContactFrictionInputData` / `StrandBendingOscillationConfig` 各 3 field + 3 経路 plumb、+7 単体テスト。**`35_explicit_accuracy_validation.py` 6 ケース全 FAIL** — `exp_no_damp_relax500` が baseline 35.37 と本質的に同値の 35.41mm（解析解 73.30mm の 51% off）、`[RELAX] converged at step 1 ||R||=0` ログで relax 即終了。**真の根本原因**: UL `update_reference` が各増分の dynamic lag を reference に凍結 → `_ul_internal_force_wrapper(state.u)` で `u_incr = state.u − _ul_ref_base ≈ 0` → `f_int(0) = 0` → relax で平衡へ駆動できない。MCDD 凍結解除条件 (5)「精度 < 10%」未達のまま。回帰: 全 24 契約検査 OK / **704 passed 5 skipped**（status-381 比 +7）/ `test_helical_3d_hermite` rel_err=2.18e-07 維持 / ruff pass。Phase A〜E / status-346〜382 の **33/N 完了**.

前 status: status-381（mass scaling 実装 bug 修正 — 発散停止、ただし explicit 解は解析解の 50% で精度 gate 未達、凍結解除判定再撤回）。status-380 §4.0 最優先 TODO に対し 3 仮説切り分けで **h-bug-1（v/a リスケール欠落）+ h-bug-3（β 急成長）** を確定し修正: (1) `set_mass_scaling_beta()` で KE 保存 v/a リスケール（`v *= β_old/β_new`, `a *= (β_old/β_new)²`）、(2) `mass_scaling_max_growth_per_update=4.0` cap、(3) 増分 1 warm-start。**実機**: 7 本 explicit 1.58×10⁸mm → 40.1mm、19 本 1.59×10⁸mm → 41.2mm で発散停止 + 形式 gate（frac=1.0 / E_ratio<5% / max\|u\|<1m）全 PASS。**しかしユーザー指摘「7本 implicit 70.7mm vs explicit 40.1mm は倍違う、解析解と合うか」で精査**: 90° 曲げ単梁の解析解 73.3mm（quarter circle）に対し implicit 70mm（96%）/ explicit 40mm（48%）と **explicit 系統的 50% アンダー**。動的緩和の未収束 + KE 保存リスケールの累積過減衰が原因と推定。**MCDD 凍結解除条件達成判定再撤回**、追加 gate (5)「解の精度 < 10%」未達。次 status は仮説 (p1) BC 完了後 relax-step 追加 / (p2) リスケール方式変更 / (p3) artificial damping / (p4) β 抑制で精度確保が最優先。回帰: 697 passed 5 skipped（+6: KE 保存 4 + warm-start 2）/ 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07 維持 / ruff pass。Phase A〜E / status-346〜381 の **32/N 完了**.

前 status: status-380（物理的妥当性検証 — 7本/19本ともに explicit 解は max\|u\|=1.59×10⁸mm 数値発散、status-379 凍結解除判定を撤回）。status-379 引継ぎ §6.1 最優先 TODO の物理的妥当性検証を実施。`30_implicit_vs_explicit_7strand.py`（+330 行）で 7 本撚線 90° 曲げを両 solver_mode で完走させ比較、`31_render_19strand_explicit.py`（+155 行）で status-379 採択設定を再現 + 3D 可視化。**重大発見**: 7 本 implicit max\|u\|=159mm（妥当、24 接触要素）vs explicit max\|u\|=1.58×10⁸mm（≈158km、active pair=0、撚線が空間に飛散）。19 本 explicit も同様 max\|u\|=1.59×10⁸mm（status-379 数値再現性は完全）。**根本原因**: `frac=1.0` は処方変位 BC 達成のみを意味し、`E_kin/E_strain<5%` は β² 倍化された両エネルギー比なので β に独立、両 gate は数学的構造由来で発散時にも PASS する。**status-379 MCDD 凍結解除条件達成判定は撤回**、CLAUDE.md 凍結解除条件に `max\|u_trans\| < L_strand × C` 追加。次候補: (h1') β cap 強化 / (h2) dt subcycling / (h3) selective explicit / (h4) implicit AL n>2 延伸 / (h5) bending 段階処方。実装本体無変更、回帰 691 passed 5 skipped（status-379 と同一）/ 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07 / ruff pass。Phase A〜E / status-346〜380 の **31/N 完了**。前 status: status-379（陽的中央差分 Phase 3 候補 (h1) mass scaling auto-tune で 19 本 frac=1.0 完走、ただし変位の物理的妥当性 gate 欠落により本 status-380 で撤回）。status-378 で実測した Courant 比 3×10⁵ を **集中質量スケーリング**（Belytschko §6.4.2）で吸収。`ExplicitCentralDifferenceProcess` に `mass_scaling_beta` 引数 + `set_mass_scaling_beta()` API 追加（β² · M_lump で集中質量倍化、Δt_c → β·Δt_c）。`ExplicitDynamicProcess` の Courant 監視に **β auto-tune** を統合: 違反検知時に必要 β を逆算し `set_mass_scaling_beta()` で上方更新、cap 到達時は `failure_reason="courant_cap"` で adaptive dt cutback と組合せ。`ExplicitDynamicInput` 3 field（`mass_scaling_auto` / `mass_scaling_max_beta` / `kinetic_energy_budget_ratio`）+ `ContactFrictionInputData` / `StrandBendingOscillationConfig` 各 4 field 追加 + 3 経路 plumb-through。新規 11 単体テスト: 質量スケーリング 8（β² 集中質量 / 単調増加 / 不正値拒否 / factory plumb）+ auto-tune 3（disabled→courant / scales-within-cap / cap→courant_cap）。**19 本撚線 90° 曲げ実機**（`work/beam_hysteresis/29_mass_scaling_19strand.py auto`、max β=10³）: **frac=1.0000 完走 + 269 incr / 31 cb / 131s + E_kin/E_strain=1.15%（gate 5% の 23%）**。status-376 implicit + AL n=2 の **0.5746 を +74% 上回る**、Gate 両条件 PASS（frac=1.0 / E_kin/E_strain<5%）。**MCDD Phase E 凍結解除条件「19 本 frac=1.0 完走」達成**。Default `solver_mode="implicit"` で既存挙動完全不変、回帰: 全 24 契約検査 OK / contact + math + time_integration + strand_bending = **691 passed 5 skipped**（status-378 比 +11）/ `test_helical_3d_hermite` rel_err=2.18e-07 / 7 本 frac=1.0（implicit）/ ruff pass。設計仕様 `time_integration_explicit.md` §質量スケーリング + `explicit_dynamic.md` §auto-tune 追記。前 status: status-378（陽的中央差分 Phase 2 — solver path 配線 + 7 本 smoke で Courant 比 3×10⁵ 実測）。Phase A〜E / status-346〜379 の **30/N 完了**

### ターゲット

> **1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

### 次の課題

完了履歴の詳細は `docs/status/status-index.md` 参照。**status-380 で status-379 の
MCDD Phase E 凍結解除条件達成判定を撤回**、status-381 で発散停止 + 50% アンダー、
**status-382 で UL `update_reference` 凍結が真の根本原因と判明**、**status-383 で
候補 (q1) 却下、explicit + UL の組合せは原理的に成立しないと確定**。
**status-384 で Abaqus/Explicit 標準アプローチ (z1a)+(z1b) を実装**: 要素ごと
波速ベース Δt 推定 + selective mass scaling。infrastructure 完備、validation で
**2 段階スケーリング**が真の解と判明。**status-385 で (z1c) 2 段階質量スケーリング
API を実装**: `mass_scaling_beta_outside` を独立 field 化、KE 保存リスケールも
mask 依存。validation で API は設計通り動作（target β 10x 縮小確認）も、initial
target β=4.7e4 が β_stiff cap を超過し全 explicit ケース frac=0、**(z1d) loading
rate 縮小が必須**と判明。現在のアクティブライン:

- **次 status（最優先）— 候補 (z1d) `t_cycle` 下限緩和**: 現 `t_cycle = max(10·T1, 1.0)`
  の **下限 1 秒** を削除し `t_cycle = 10·T1` または `max(10·T1, 0.1·T1)` 等の
  物理ベース下限に変更。実装規模小（3 経路 1 行修正）、ただし implicit 7 本
  frac=1.0 維持の regression を要確認。dt_sub を 100x 程度縮小すれば target β も
  100x 縮小し、β_outside=10 + β_stiff_max=1e3 の組合せで gate 達成可能と予測。
- **副次 — 候補 (z2) Cosserat 梁プロトタイプ**: UL を捨てて explicit + 大回転を
  本質解決。geometrically exact (Simo-Reissner) beam、SO(3) 回転 DOF、Lie 群更新。
  実装中規模（~1000 行）、(z1d) で精度 gate 達成可なら不要、未達なら必須。
- **副次 — 候補 (q3) implicit + AL n>2 復活**: status-376 で却下された (g2) AL n>2
  を Uzawa update under-relaxation で再試行。explicit 路線が完全に行き詰まった
  ときの最終 fallback。
- **副次 — 候補 (h5) bending 段階処方**: 19 本 implicit で `bending_curvature` を
  0.005 → 0.010 → 0.015 と段階的に増加させ Newton 良条件再開。
- **凍結解除 TODO 再開**: 被膜圧縮モデル / リスタート方式 / ファイバー梁キャリブレーション /
  7本撚線ピッチ依存性 / 空間ブロック分離（status-345 で凍結、再開可能）。
- **多 pair 診断 `14b_kc_multi_pair_diagnostic.py`** — status-370 §5 保留、優先度低。

詳細は `docs/roadmap.md` および `docs/status/status-index.md` を参照。

## フォーカスガード（AI セッション向け）

**以下を厳守すること。違反は作業のやり直しになる。**

## やるべきこと

### ★最優先: MCDD（数理契約駆動開発）Phase A〜E（status-346〜359、status-354 で 1 status 後ろ倒し）

**計画（LOST）**: `/root/.claude/plans/deep-wiggling-seal.md` は **永久ロスト**
（2026-04-19 時点、ファイルは復旧不可）。以降、計画書参照箇所は本 CLAUDE.md・
`docs/roadmap.md`・`docs/status/status-{N}.md` に同等情報を転記して運用する。
脱法実装禁止パターン 10 項は下記「MCDD 脱法実装禁止パターン」を参照。
**設計仕様**: `xkep_cae/mathematics/docs/mathematics.md`

status-346 で **MCDD Phase A-1 完了**（`MathematicalContract` 型 5 種新設、
33 テスト追加）。status-347 で **MCDD Phase A-2 完了**（`ProcessContractRegistry`
+ `@verified_by` デコレータ + dummy VerifyProcess AST 検査拒否、33 テスト追加）。
status-348 で **Phase B-1 完了**（`docs/math/03_huber_contact_penalty.md` 19
アンカー）。status-349 で **Phase B-2 完了**（残り 5 章 + `equation_index.py`
+ C15 拡張、29 テスト追加）。status-350 で **Phase C-1 完了**
（`KcNormal` / `KcGeo` Process 抽出 + `tangent_components()` orchestrator 化、
`TermExpansionContract` 3 Process 紐付け、14 テスト追加）。status-351 で
**Phase C-2 完了**（`KcHermiteNonlocalStiffnessProcess` + `KcClosestPointStiffnessProcess`
新設、`TermExpansionContract` 5 項化、11 テスト追加で 14→25）。status-352 で
**計画書ロスト記録 + Phase C-3 前提疑義提示**（中断スナップショット）。
status-353 で **数理台帳訂正完了**（`K_mat,ndir` ≡ `K_geo` の同一性確立、
当初 Phase C-3 計画を撤回、5 項で完結化、`docs/math/03_huber_contact_penalty.md`
§3/§3.1/§4/§5/§8 訂正、`strategy.py` モジュールコメント / 関連 docstring 訂正、
7本撚線回帰 frac=1.0000 完走、421 passed 5 skipped）。status-354 で
**Phase C-3 再定義実験**（仮説 A `K_hermite_adj` フル項拡張 = `-w_geo * I_nn`
隣接ノード項追加）を直接実験し、gate テスト `test_helical_3d_hermite` の
rel_err が **1.795% → 38.49%** に 21 倍悪化して **反証**、mat-only（status-295）
継続。数理台帳 §7/§3.1/§4/§8 に仲裁追記、`strategy.py` docstring に実測
結果記録（実装変更なし）。Phase C-3 を **Phase C-3' 再々定義**
（hypothesis B/C/D）へ再配分。status-355 で **Phase C-3' 診断完了**
（active×adj ブロック局在化）、status-356 で **Phase C-3' 実装完了**
（仮説 A + B 同時導入で FD 機械精度）。status-357 で **Phase E 着手 +
19 本撚線実機規模検証**（Phase C-3' は active 集合固定下限定、19 本 Type D
stall は active 振動支配領域で未解決、仮説 C に昇格。C5 違反解消 + C18/C19
契約検査追加）:

- ~~status-347（Phase A-2）~~: 完了
- ~~status-348-349（Phase B）~~: 完了（6 章 / 55 アンカー + `equation_index.py` + C15 拡張）
- ~~status-350（Phase C-1）~~: 完了（`KcNormal` / `KcGeo` + `ContactForceStStiffnessProcess` の 3 Process 抽出）
- ~~status-351（Phase C-2）~~: 完了（`KcHermiteNonlocal` / `KcClosestPoint` 分離、5 項 TermExpansionContract）
- ~~status-352（中断スナップショット）~~: 完了（計画書ロスト記録 + Phase C-3 前提疑義提示）
- ~~status-353（数理台帳訂正）~~: 完了（`K_mat,ndir` ≡ `K_geo` 確立、当初 Phase C-3 撤回、5 項完結化、§3/§4/§5/§8 訂正、7本撚線回帰 frac=1.0 完走）
- ~~status-354（Phase C-3 再定義実験）~~: 完了（仮説 A = `K_hermite_adj` + `-w_geo * I_nn` を単独で実験、rel_err 1.795%→38.49% 21倍悪化、当時は revert・mat-only 継続、数理台帳 §7 仲裁追記、Phase C-3' 再々定義）
- ~~status-355（Phase C-3' 診断）~~: 完了（`work/beam_hysteresis/14_kc_closest_adj_diagnostic.py` 新設、rel_err 1.795% の 100% が active×adj ブロックに局在、仮説 B の定量目標 `||diff[ax]|| 98.52 → <1e-3` と実装パス ~45 行を確立）
- ~~status-356（Phase C-3' 実装）~~: 完了（**仮説 A + 仮説 B 同時導入**で 2 経路 (i)(ii) の $P_\perp$ 成分を相殺、`test_helical_3d_hermite` rel_err **1.795% → 2.18e-07**、`||diff[ax]|| 98.52 → 4.75e-05` 達成。status-354 の「mat-only 最良」解釈は (ii) 未実装時のワークアラウンドと訂正、数理台帳 §7 を 2 経路解析 / 相殺定理 / 診断裏付けに再構成。`_batch_dm_ext_coeffs` ヘルパ抽出で MCDD 脱法 3 回避）
- ~~status-357（Phase E 着手 + 19 本 FD 再計測）~~: 完了（**frac=0.3739 退化 / mat_only rel_err +15% 悪化**。Phase C-3' の FD 機械精度達成は active 集合固定下限定、19 本 Type D stall の active 振動支配領域は未解決と判定。副次: C5 違反を `_batch_dm_ext_coeffs` module 関数化で解消。**Phase E 着手**: C18（`@verified_by` 紐付け検査）+ C19（`TermExpansionContract.providers` 実在検査）を `validate_process_contracts.py` に追加、5 term-provider Process に `@verified_by("K_c_term_expansion", ContactKcComponentFDDiagnosticProcess)` 付与）
- ~~status-358（Phase E C20 + 仮説 C 候補 (a) 7本撚線 90° 実測）~~: 完了（**仮説 C 候補 (a) 却下** — `smoothing_delta=500`（default 2000 の 1/4、δ_h 4x 拡大）を 7本撚線 90° 曲げで実測、frac=0.9241 で未完走、cutback -14%/elapsed -17% の見かけ改善は解析の早期打切りで対策効果ではない。ユーザー指示「10% 以上改善 + frac=1.0 完走」未達で revert、コード変更なし、`15_hypothesis_c_7strand.py` は失敗実験の記録として残置。**Phase E C20 追加**: `TermExpansionContract` 双方向紐付け検査（providers ↔ contracts 同名契約宣言）を `validate_process_contracts.py` に追加、C18/C19 の片側更新による脱法すり抜けを防御、5 既存 providers で回帰なし）
- ~~status-359（仮説 C 候補 (a') 中間値再試行）~~: 完了（**仮説 C 候補 (a') 採択方向（実験記録）** — `smoothing_delta=1000`（default 2000 の 1/2、δ_h 2x 拡大）を 7本撚線 90° 曲げで実測、**frac=1.0000 完走 + n_increments=475（-9.4%）+ n_cutbacks=53（-7.0%、10% 未満）+ elapsed=259.92s（-42.5%、1.74x 高速化）**。ユーザー指示「frac=1.0 完走 + 10% 以上改善」に対し elapsed -42.5% で大幅クリア。判定: 採択方向。ただし `StrandBendingOscillationConfig.smoothing_delta` の default 変更（2000→1000）は本 status では実施せず（7 本撚線のみの検証で 19 本 Type D stall 本体への有効性未検証）、`15_hypothesis_c_7strand.py` を成功実験記録として残置、実装本体無変更。余談: 梁の塑性／粘性導入と収束の関係について Q&A あり、ファイバー梁 `Fiber1DState.eps_p` 等の状態保持と凍結中 TODO 整理を status-359 §引継ぎに記録）
- ~~status-360（仮説 C (a') 19 本撚線検証 + Phase E C21/C22/C23）~~: 完了（**仮説 C 候補 (a') 却下** — `smoothing_delta=1000` を 19 本撚線（Type D stall 本体）で実測、`frac=0.3723`（baseline 0.4839 比 -23.1% 退化）。NR 内訳 D+E:72% で最終 10 反復支配、δ_h 2x 拡大は Type D stall 領域で逆効果。`StrandBendingOscillationConfig.smoothing_delta` の default 変更は**実施せず**、`16_hypothesis_c_aprime_19strand.py` を失敗実験記録として残置。次候補は **(c) line search 強化**（`_newton_dynamic.py` に backtracking hook 追加）。**Phase E C21/C22/C23 追加**: C21 `TermExpansionContract.term_names` 重複静的検出（`__post_init__` に runtime ガード + 静的検査）、C22 `contracts` ClassVar 同名契約重複検出（`register_contracts` の静的版）、C23 `@verified_by` 検証 Process が `SolverProcess` / `VerifyProcess` いずれかの継承必須（`bind_verifier` に runtime ガード + 静的検査）。`test_duplicate_term_names_rejected` + `test_bind_invalid_category_rejected` 2 テスト追加で mathematics/tests 97 passed、全 23 契約検査 OK）
- ~~status-361（仮説 C 候補 (c) line search 強化）~~: status-362 で `ContactBacktrackingLineSearchProcess` 実装完了（7本 frac=1.0 回帰なし、19本 frac 0.4839→0.5153 +6.5% 改善）
- ~~status-362（仮説 C (c) 実装 + 実機検証 + 3D 可視化基盤）~~: 完了（`_newton_steps.py` +112 行、4 層 9 field plumb-through、default OFF、6 テスト。7本 frac=1.0000 / +9.9% elapsed で回帰なし、19本 frac=0.5153（+6.5%）で stall 点前進も MCDD 凍結解除条件 frac=1.0 未達。`Strand3DContourProcess` 新設で 6 フィールド 3D レンダリング、BenchmarkRunner `post_processes` 自動起動統合）
- ~~status-363（仮説 C (c) パラメータ感度掃引）~~: 完了（**4 ケース全却下、BT 既定が局所最適、候補 (c) クローズ** — `22_bt_parameter_sweep_19strand.py` 新設で 3 軸 4 ケース（A: rate_threshold=0.70 / B: active_flip_ratio=0.15 / C: mixed_only=False / D: A+B+C）を 19 本 90° 曲げ実測、全ケース frac<1.0（A=0.5153 BT default 同値 / B=0.4701 -8.8% / C=0.4817 -6.5% / D=0.5156 +0.06%）。BT 既定設定が実測最良点、default 変更なし。最終 NR Type 分布 `D+E:68%, E:26%` で line search は active 振動を根本抑制できないと確定。次候補は (e) 接触減衰 escape hatch 最有力）
- ~~status-364（Phase E C24 + 候補 (e) 方針策定）~~: 完了（`HollowVerifyProcessError` + AST 2 シグナル必須化で hollow VerifyProcess を構造的封じ込め、全 24 契約検査に拡張、mathematics tests 109 passed）
- ~~status-365（候補 (e) Phase 1: Process 単体実装 + 12 テスト）~~: 完了（`xkep_cae/contact/damping/` 新設、`ContactNormalDampingProcess` + 設計仕様 + `StrandBendingOscillationConfig` 2 field 追加、solver 未配線）
- ~~status-366（候補 (e) Phase 2: NR ソルバー配線 + Monitor + 7 テスト）~~: 完了（`ContactFrictionProcess.damping_slot` 追加、NR 反復で `R_u += f_damp` / `K_T += K_damp`、`ContactFrictionInputData`/`StrandBendingOscillationConfig` plumb-through、`SolverResultData.damping_energy_history` 公開、`ContactDampingEnergyMonitorProcess` PostProcess 新設、default OFF で既存動作不変、contact 446 passed 5 skipped）
- ~~status-367（候補 (e) validation — 符号訂正 + 7 本採択方向 + 19 本却下）~~: 完了（(1) 符号規約バグ訂正: `R_u += f_damp` → `R_u -= f_damp`（物理ドラッグ力と NR 残差規約の C·v 正寄与との不整合）、docstring に符号規約節追加、unit test 本体無変更。(2) 7 本 c_n=1000 で **frac=1.0 完走 + elapsed -56.8%（246→106s）** 劇的改善、採択方向。(3) 19 本 c_n=100/1000 で frac=0.43/0.47 と baseline 0.48 より悪化で却下、Type D stall の主因は K_c x/z カップリング不整合で局所減衰では解消できない。`contact_damping_coefficient=0.0` default 維持、7 本系 opt-in 高速化として運用、実装本体無変更）
- ~~status-368（候補 (d) 接触凍結モード 19 本再評価）~~: 完了（`chattering_freeze_*` 3 パラメータ × 6 ケース感度掃引で **Case B `nr_max=30`（default 15 の 2x）のみ有意改善 frac=0.5642（+50.9%、status-339 baseline 0.4839 比 +16.6%）**、他 5 ケース効果軽微〜悪化。disabled は `D+E:98%` 200 反復ハマり（**freeze mode が D+E ロック回避の支柱**と確定）。MCDD 凍結解除条件（frac=1.0）未達で**候補 (d) クローズ**、default 変更なし（7 本向け最適化維持、19 本 opt-in escape hatch として運用）。`25_freeze_param_sweep_19strand.py` 新設）
- ~~status-369（Case B 19 本 opt-in ガイドライン化 + 候補 (f) Phase C-3' 実験計画 策定、documentation status）~~: 完了（実装本体変更なしの documentation status、2 成果: (1) TODO 2 副次: `chattering_freeze_nr_max=30` を 19 本以上向けの opt-in escape hatch として `StrandBendingOscillationConfig` docstring + `docs/roadmap.md` 「撚線規模別 opt-in チューニング」表に明記（7 本既定/19 本推奨/実測効果/根拠 status 4 項目表）。(2) TODO 1 reconnaissance: `xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md` 新設（+107 行）で候補 (f) を Step 3.1 active 境界 FD 診断 / Step 3.2 新項 `KcActiveFlipStiffness` 追加設計に分割 scoping、MCDD 脱法パターン回避チェックリスト + gate 基準明記。`docs/design/README.md` に索引登録）
- ~~status-370（Phase C-3' Step 3.1 完了 — active 境界 FD 診断で結果 B 確定）~~: 完了（`work/beam_hysteresis/14_kc_active_boundary_diagnostic.py` 新設 +280 行、3 Block 構成で 20 測定点、**全点 rel_err=2.18e-07〜2.20e-07 機械精度維持**（degradation 1.01x +0.00 桁）、eps=1e-4 の 2.19e-04 は FD truncation。**結果 B 確定**で新項 `KcActiveFlipStiffness` 追加は不要、19 本 Type D stall は NR alg 側動力学。plan doc §3.2 を候補 (g) 3 サブライン再配分: (g1) active 履歴平滑化 最優先 / (g3) pair-wise relaxation / (g2) AL 再導入）
- ~~status-371（候補 (g1) active 履歴平滑化 実装）~~: 完了（`HuberContactForceProcess` に `active_ema_alpha: float = 0.0` field 追加 + `_p_n_prev_array` 保有 + `reset_ema_state()` メソッド + `evaluate()` ブレンドロジック。`NewtonDynamicProcess` インクリメント境界で reset 呼出。4 層 1 field plumb-through + 3 経路、`TestActiveEmaSmoothing` 10 テスト + `26_active_ema_alpha_sweep.py` 診断スクリプト 150 行。default α=0.0 で既存 446 contact テスト全 pass、`test_helical_3d_hermite` rel_err=2.18e-07 維持。実機 α 掃引は status-372 へ分離（Phase 1+2 構成）。22/N 完了）
- ~~status-372（候補 (g1) α 掃引 実機検証）~~: 完了（α ∈ {0.0, 0.1, 0.3, 0.5} を 7 本 / 19 本撚線 90° 曲げで実測。**7 本**: α=0.30/0.50 で frac=1.0 維持 + cb -61〜-75%（57→14/22）、α=0.50 で elapsed -11%、α=0.10 のみ早期 stall（弱平滑化逆効果、status-262 smoothing_delta 非単調性類似）。**19 本**: gate「frac ≥ 0.6」全ケース未達で **却下方向**、α=0.50 frac=0.5133（baseline 0.3739 比 +37.3%）部分改善も elapsed +131% でコスト過大。default 変更なし、`active_ema_alpha=0.5` を 7 本系 cutback 削減 opt-in 表に追加。実装本体無変更、456 contact + 109 mathematics 全 pass。23/N 完了）
- ~~status-373（TODO 整理 + 症状緩和系 experiment 5 本削除 + solver_mode 設計追記、documentation status）~~: 完了（実装本体無変更、`work/beam_hysteresis/{15,16,22,25,26}_*.py` を `git rm`、`次の課題`/`凍結中 TODO` を status-index 参照に圧縮、`phase_c3prime_19strand_plan.md` §4 に `solver_mode` 併存方針追記、`docs/roadmap.md` opt-in 表に `solver_mode` 行追加。実装計画書を `docs/plans/status-373-plan.md` にレポジトリ内常設化）
- ~~status-374（候補 (g3) pair-wise relaxation Phase 1 — `PairwiseFreezingProcess` 単体実装）~~: 完了（`xkep_cae/contact/freeze/` サブパッケージ新設、`strategy.py` 261 行で `PairwiseFreezingProcess` + Input/Output + private ヘルパ純関数 2 本（C16 滅菌のため `_update_pair_active_flips` / `_is_type_d_dominant` を `__init__.py` 非 export）。判定: `skip_global := skip_when_type_d_dominant ∧ _is_type_d_dominant(chattering_type)`、`freeze[k] := (flip_counts[k] ≥ threshold) ∧ is_active_now[k] ∧ ¬skip_global`。`docs/pairwise_freezing.md` 159 行 + 12 単体テスト（`@binds_to` 付与は API クラスに 1 回のみ、Logic/Helpers は独立クラス）。NR 配線は Phase 2 / status-375 へ分離（status-365 ContactNormalDamping と同 Phase 1/2 分割）。実装本体（`_newton_dynamic.py` / `StrandBendingOscillationConfig` / `ContactFrictionProcess`）は無変更。gate: 全 24 検査 OK / contact 468 passed 5 skipped (+12) / mathematics 109 passed / `test_helical_3d_hermite` rel_err=2.18e-07 維持 / ruff pass。25/N 完了）
- ~~status-375（候補 (g3) Phase 2 NR 配線 + 19 本実機検証で却下）~~: 完了（`PairwiseFreezingProcess` を NR ループに配線、`_newton_dynamic.py` +88 行で is_active_now 構築 → flip_counts 更新 → DOF block 上書き、`process.py` `freeze_slot` 追加、`core/data.py` + `StrandBendingOscillationConfig` 各 3 field（default OFF）、3 経路 plumb-through。Default OFF 回帰: 7 本撚線 frac=1.0 + contact 468 passed 5 skipped。**19 本撚線**: `flip_threshold ∈ {2,3,5}` 全 3 ケース Gate `frac ≥ 0.6` 未達で **候補 (g3) 却下**（t=2 -47.2% / t=3 -6.9% / t=5 -47.5%）。pair-wise freeze 発動で NR Type `A+B+D.div:71%` 集中 + DOF block 上書きの隣接 pair 正フィードバックが原因。19 本 Type D stall は K_c x/z カップリング不整合が主因で active 集合振動の per-pair 凍結では解消できないと確定。`pairwise_freeze_*` 3 field は default OFF のまま 19 本以上向け opt-in escape hatch として保持。26/N 完了）
- ~~status-376（候補 (g2) AL 外側ループ限定再導入 + 19 本実機検証で却下）~~: 完了（`HuberContactForceProcess.set_al_lambda_offset/get_last_p_n_eff` API 追加で `p_n_eff = max(0, p_n_huber + λ)` を evaluate() 内包、K_geo 自動整合（modified Newton 不要、§9.2）、NewtonDynamicProcess の NR while を AL 外側 for ループで包み Uzawa 更新 `λ_new = max(0, p_n_eff_converged)`。法線のみ AL（摩擦は status-147 回避）。実装: strategy.py +43 / _newton_dynamic.py +37（既存ループ字下げ）/ 4 経路 2 field plumb / 11 単体テスト / 数理台帳 §9 追記 +96 行。Default OFF 回帰: 588 passed 5 skipped / `test_helical_3d_hermite` rel_err=2.18e-07 維持 / 7本 frac=1.0 / ruff + 24 契約検査 OK。**19 本検証**: n=2 で **frac=0.5746（+53.7%、(g) 全候補で最良）** だが Gate 0.6 を 0.026 不足で FAIL、n=3 で過修正発散（frac=0.1973、-47.2%）。**判定: 候補 (g2) 却下、候補 (g) 3 サブライン全終了**（(g1)+37.3% / (g3)-6.9% / (g2)+53.7%）。NR alg 側 escape hatch 限界到達、19 本 Type D stall は K_c x/z カップリング不整合（status-344）が主因。次候補は explicit 時間積分。`al_outer_enabled=False` default 維持、`al_outer_enabled=True, al_n_uzawa_max=2` を opt-in escape hatch として運用。27/N 完了）
- ~~status-377（陽的中央差分時間積分 Phase 1 — Process 単体実装 + `solver_mode` config + 設計仕様）~~: 完了（`xkep_cae/time_integration/strategy.py` に `ExplicitCentralDifferenceProcess` 新設 +216 行で集中質量ロンピング `row_sum`/`diagonal`/`none` + 中央差分 `step()` API + Courant 臨界 dt + Verlet 予測子 + チェックポイント / Protocol 適合 4 メソッド。`_create_time_integration_strategy()` に `solver_mode="explicit"` 分岐、`StrandBendingOscillationConfig.solver_mode: Literal["implicit","explicit"]` (default `"implicit"`)、`solver_mode="explicit"` 実行時は `NotImplementedError` で Phase 2 待機を明示。設計仕様 `xkep_cae/time_integration/docs/time_integration_explicit.md` 新設 +126 行。**Phase 1 制約**: Process 単体実装 + 28 単体テスト + 設計仕様のみ、solver path 配線は Phase 2 で実施。Default OFF 回帰: contracts 全 24 OK / contact 468 + math 109 + time_integration 61 + strand_bending_oscillation 21 = **649 passed 5 skipped** / `test_helical_3d_hermite` rel_err=2.18e-07 / 7本 frac=1.0 / ruff pass。新規 28 ユニット内訳: SDoF 自由振動 1 周期戻り < 5%、5 周期エネルギー有界 < 10%、Courant 越え発散 100x、ロンピング数値、固定 DOF、減衰 C·v、対角質量 K_eff、`solver_mode` config 3 件。28/N 完了）
- ~~status-378（陽解法 Phase 2 solver path 配線 + 7 本 smoke test）~~: 完了（`_explicit_dynamic.py` 新設 +251 行で `ExplicitDynamicProcess` + `_estimate_critical_dt`（sparse Gerschgorin 上界）。`ContactFrictionProcess` を `solver_mode` で分岐、explicit モードでは `predict()` / `correct()` / `_u_pred` MPC 射影をスキップ。`ContactFrictionInputData` / `StrandBendingOscillationConfig` 各 4 field 追加（3 経路 plumb）、`NotImplementedError` ガード削除。設計仕様 `contact/solver/docs/explicit_dynamic.md` 新設、新規 10 ユニット。**7 本 90° smoke**: `dt_sub=0.333` vs `dt_c=1.055e-06` で Courant 比 3×10⁵、3 回 cutback でも frac=0.0052。Wiring 正常動作、19 本 frac=1.0 完走には mass scaling / dt subcycling が必須と確定。Default `solver_mode="implicit"` で既存挙動完全不変、680 passed 5 skipped / 7 本 frac=1.0（implicit）/ ruff pass。29/N 完了）
- ~~status-379（陽解法 Phase 3 候補 (h1) mass scaling auto-tune — 19 本 frac=1.0 完走、〜status-380 で判定撤回〜）~~: 完了（`ExplicitCentralDifferenceProcess` に `mass_scaling_beta` 引数 + `set_mass_scaling_beta()` API（β² · M_lump で集中質量倍化、Δt_c → β·Δt_c）。`ExplicitDynamicProcess` の Courant 監視に β auto-tune 統合: 違反検知 → 必要 β 逆算 → `set_mass_scaling_beta()` 上方更新、cap 到達は `failure_reason="courant_cap"` で adaptive dt cutback と組合せ。`ExplicitDynamicInput` 3 field + `ContactFrictionInputData` / `StrandBendingOscillationConfig` 各 4 field 追加 + 3 経路 plumb-through、11 単体テスト。19 本 90° 曲げ（max β=10³）で frac=1.0000 完走 / 269 incr / 31 cb / 131s / E_kin/E_strain=1.15%、Gate 両条件（frac/E_ratio）PASS と判定したが、**status-380 で max\|u\|=1.59×10⁸mm の数値発散が発覚し撤回**。Default `solver_mode="implicit"` で既存挙動完全不変、回帰 691 passed 5 skipped。30/N 完了）
- ~~status-380（物理的妥当性検証 — 7本/19本ともに explicit 解は数値発散、status-379 凍結解除判定を撤回）~~: 完了（max\|u\|=1.59×10⁸mm 発散を検出、CLAUDE.md 凍結解除条件に `max\|u_trans\| < L_strand × C` 追加。31/N 完了）
- ~~status-381（mass scaling 実装 bug 修正 — 発散停止、ただし explicit 解は解析解の 50% で精度 gate 未達、凍結解除判定再撤回）~~: 完了（3 仮説切り分けで **h-bug-1（v/a リスケール欠落）+ h-bug-3（β 急成長）** を確定。修正: (1) `set_mass_scaling_beta()` で KE 保存 v/a リスケール、(2) `mass_scaling_max_growth_per_update=4.0` cap、(3) 増分 1 warm-start。実機: 7 本 explicit 1.58e8→40.1mm、19 本 1.59e8→41.2mm で発散停止。**ユーザー指摘で精査**: 90° 曲げ単梁解析解 73.3mm に対し implicit 70mm（96%）/ explicit 40mm（48%）で **explicit 系統的 50% アンダー**。動的緩和未収束 + KE 保存リスケール累積過減衰が原因と推定。**MCDD 凍結解除条件達成判定再撤回**、追加 gate (5)「解の精度 < 10%」未達。回帰 697 passed 5 skipped（+6）/ 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07 維持。32/N 完了）
- ~~status-382（候補 (p3) damping + (p1) relax API 実装 — UL update_reference 凍結が真の根本原因と判明、精度 gate 未達のまま）~~: 完了（status-381 §7「explicit 解を implicit / 解析解と一致させる」最優先 TODO に対し仮説 (p3) 質量比例 Rayleigh damping `ExplicitCentralDifferenceProcess.mass_proportional_damping_alpha`（`a -= α·v`、Courant/β 独立）+ (p1) BC 完了後 relax phase（`ContactFrictionProcess` 末尾に追加、BC frac=1.0 保持で `explicit_relax_steps` 回 step）の 2 API を実装。`ContactFrictionInputData` / `StrandBendingOscillationConfig` 各 3 field + 3 経路 plumb、+7 単体テスト。**`35_explicit_accuracy_validation.py` 6 ケース全 FAIL** — `exp_no_damp_relax500` が baseline 35.37 と本質的に同値 35.41mm、`[RELAX] converged at step 1 ||R||=0` ログで relax 即終了。**真の根本原因**: UL `update_reference` が各増分の dynamic lag を reference に凍結 → `_ul_internal_force_wrapper(state.u)` で u_incr ≈ 0 → f_int(0) = 0 → relax で平衡へ駆動できない。MCDD 凍結解除条件 (5)「精度 < 10%」未達のまま。次候補は (q1) explicit 中の UL update 周期化（最有力、`explicit_ul_update_interval`）/ (q2) 増分内 sub-cycling / (q3) implicit + AL n>2 復活。回帰 704 passed 5 skipped（+7）/ 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07 維持。33/N 完了）
- ~~status-383（候補 (q1) `explicit_ul_update_interval` 実装 — 4 ケース掃引で却下、UL 凍結が真因と再確証、精度 gate 未達のまま）~~: 完了（status-382 §6.1 最有力候補として `solver_mode="explicit"` のとき UL `update_reference()` を **N 増分ごと** に呼出する gate を導入。`ContactFrictionInputData.explicit_ul_update_interval: int = 1` field 追加（default 1 で既存挙動完全不変）、`StrandBendingOscillationConfig` に同 field + 3 経路 plumb-through、process.py 主ループ内 update_reference 呼出箇所に `(_next_incr % interval == 0)` ゲート追加。**+5 単体テスト**（`TestExplicitULUpdateInterval`、`_MockULAssembler` で update_reference 呼出回数を直接計測）。**`36_explicit_ul_interval_validation.py` 5 ケース掃引で全 FAIL** — interval=1 baseline 29.57mm（status-382 と一致、default 完全保持）/ interval=5 で relax phase 発散 (NaN) / interval=10 max\|u\|=6.21e6 mm / interval=20 max\|u\|=5.16e21 mm。**根本要因**: CR 梁 UL 定式化は「u_incr 微小」前提で線形化、N 増分蓄積は K_T(u_incr) を線形化レンジ外へ押し出し explicit dynamics が爆発的発散。status-382 §3 解析と整合：(a) update 毎呼出 → f_int(u_incr)≈0、(b) update 間引き → K_T(u_incr) 線形化崩壊、両方破綻。**MCDD 凍結解除条件 (5)「精度 < 10%」未達のまま**。次候補は (q2) 増分内 sub-cycling（最有力、UL 動作は通常通り保持）/ (q3) implicit + AL n>2 復活 / (h5) bending 段階処方。回帰 709 passed 5 skipped（+5）/ 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07 維持 / 7 本 implicit frac=1.0 / ruff pass。34/N 完了）
- ~~status-384（候補 (z1a) 要素ごと波速 Δt + (z1b) selective mass scaling — Abaqus/Explicit 標準アプローチへの移行 Phase 1 完了、validation で 2 段階スケーリング要件を発見）~~: 完了（ユーザーから「応力波の速度と要素サイズから dt」+「Cosserat 梁の大回転ネイティブ特性」指摘を受け、status-383 までの "explicit + UL は原理的に不整合" を踏まえて方針転換。**(z1a)** `_estimate_critical_dt_per_element(connectivity, node_coords, beam_E, beam_rho)` 新設で `dt_e = L_e / √(E/ρ)` を要素ごとに計算し Gerschgorin 全体上界と min を取る。**(z1b)** `_detect_stiff_dofs()` で Gerschgorin row-sum / M が median × `threshold_ratio` を超える DOF を自動検出、`ExplicitCentralDifferenceProcess.set_mass_scaling_dof_mask()` で β² 倍化を限定。`_compute_scaled_mass(beta)` ヘルパ + `set_mass_scaling_beta()` の v/a rescale を mask 対応。**+17 単体テスト**（per-element dt 6 + stiff detect 5 + selective scaling 6）全 pass。**実機検証**: 単梁 90° で K がほぼ一様 → stiff DOF 検出ゼロ → 実質 β=1 → frac=0 発散（実装 bug ではなく selective が heterogeneous K を要求する性質）。7 本撚線で stiff DOF 112/714 検出 (15.7%) も、残り 84% の beam DOF が β=1 のまま dt 1.6μs 制約を支配し target β=8.8×10⁶（cap 1000 を超過）で frac<<1.0。**真の解**: 2 段階スケーリング（β_stiff=1000, β_beam=10）+ loading rate 縮小の組合せ。次候補は (z1c) per-DOF β 配列 API + (z1d) `t_cycle` 下限緩和 + (z2) Cosserat 梁プロトタイプ並行検討。回帰 726 passed 5 skipped（+17）/ 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07 維持 / ruff pass。35/N 完了）
- ~~status-385（候補 (z1c) 2 段階質量スケーリング API（β_stiff + β_outside）実装 — API 完成、validation で β_stiff cap が支配的と確認、(z1d) loading rate 縮小が必須と判明）~~: 完了（status-384 §6.1 最有力候補 (z1c) として `ExplicitCentralDifferenceProcess` に `mass_scaling_beta_outside` 引数 + `set_mass_scaling_beta_outside()` API（KE 保存 v/a リスケール対応、mask=False の DOF のみ rescale）を追加。`_compute_scaled_mass()` で mask=False の DOF（梁）に β_outside² を、mask=True の DOF（stiff）に β² を適用。`_explicit_dynamic.py` の dt_c_beam 推定で mask 設定時は `β_outside` を乗じる（mask=None 時は従来通り `β`）。`ContactFrictionInputData` / `StrandBendingOscillationConfig` 各 1 field + 3 経路 plumb-through。**+11 単体テスト**（`TestTwoStageMassScaling`）全 pass。**`38_z1c_two_stage_validation.py` 8 ケース実機検証**: API は設計通り動作（log で post-cutback target β が β_outside=10 で 8.8e6 → 8.8e5 に **10x 縮小**）も、initial target β=4.7e4（β_stiff cap=1e3〜1e4 を超過）が支配的で全 explicit ケース frac=0 で divergence。aggressive scaling（β_outside=10, β_stiff_max=1e6, α=10）で frac=0.425 進むも max\|u\|=1.6e5mm で精度 gate 完全違反。**結論**: (z1c) infrastructure は完成、しかし MCDD 凍結解除条件 (5) 達成には **(z1d) `t_cycle` 下限緩和** で loading rate を物理 T1 ベースに縮小し target β 自体を下げる必要がある。次候補は (z1d) 最優先 / (z2) Cosserat 梁プロトタイプ並行検討。回帰 737 passed 5 skipped（+11）/ 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07 維持 / ruff pass。36/N 完了）

**凍結中の TODO**（MCDD 完了まで再開禁止）: 詳細項目（被膜圧縮モデル / リスタート方式 / ファイバー梁キャリブレーション / 7本撚線ピッチ依存性 / 空間ブロック分離 / 19本 Type D stall K_mat x/z 単発対応）は status-345 までで列挙、status-373 で本ブロックから削除。

**凍結解除条件**（status-381 で **解の精度 gate を追加**）:

1. Phase E 完了
2. 19 本 frac=1.0 完走（`load_history[-1] = 1.0`）
3. **解の物理的妥当性 gate**: `max |u_trans| < L_strand × C`（C=10、撚線長 100mm に
   対し最大変位 1m 以内）。status-380 で発覚した「frac=1.0 + E_kin/E_strain<5% は
   両方とも数学的構造由来で発散時にも PASS する」盲点を塞ぐ。
4. `KcNormalDirectionStiffness` FD rel_err < 1e-2
5. **解の精度 gate（status-381 追加）**: `|u_explicit − u_implicit|/|u_implicit| < 0.1`
   または `|u_explicit − u_analytical|/|u_analytical| < 0.1`。
   status-381 で発覚した「形式 gate (1)〜(4) は under-relaxation 解でも PASS する」
   盲点を塞ぐ。90° 曲げ単梁の解析解 73.3mm に対し implicit 70mm / explicit 40mm
   と systematicalに 50% アンダーであった。

status-379 の達成判定は条件 3 欠落、status-381 の判定は条件 5 欠落で誤判定。

### セッション開始時の必須確認（MCDD 規範）

次セッションを Claude Code / Codex で開始する際は、以下を**順に**実行:

1. ~~`/root/.claude/plans/deep-wiggling-seal.md` を**全文読む**（要約禁止）~~
   → **計画書は永久ロスト**（status-352 で記録）。代替として本 CLAUDE.md の
   「やるべきこと」「MCDD 脱法実装禁止パターン」および最新 status を参照
2. 最新 `docs/status/status-{N}.md` を読み、前セッションの停止点を確認
3. 本ファイル「MCDD 脱法実装禁止パターン 10 項」を読み返し、本セッションで
   陥りそうな項目を自己チェック
4. その上で着手

## やってはいけないこと
- 管理上processクラスとすべきロジックをあえてプライベート関数や迂回ロジックに替えること
- 収束トライ時に目標を緩和して本質的対策を先送りにすること

### MCDD 脱法実装禁止パターン（旧計画書より転記、status-346〜356 で厳守）

1. **契約の tol を事後緩和して pass させる**（数理的正当化なき `tol_rel` 変更は禁止）
2. **dummy VerifyProcess を `@verified_by` に紐付けて C18 を通す**
3. **`tangent_components()` を wrapper だけで済ませる**（中身が旧 monolith 呼び出しだけ）
4. **`KcNormalDirectionStiffnessProcess` を rename で済ませる**（新規実装必須）
5. **既存テスト 12 件を skip/xfail で pass させる**（`test_kc_component_fd.py` 無変更 pass が gate）
6. **「Phase C を Phase C' に分割」等で困難を先送り**（骨格だけの status は禁止、
   コンテキスト不足は status 中断で正規手順）
7. **診断 report を `{:5.2f}` 等で丸める**（status-345 の教訓、share/ratio は `{:.3e}` 必須）
8. **回帰を「ベースライン側が誤っていた」と根拠なく主張**（数値で反証必須）
9. **`tuple[...]` を `list[...]` に変えて frozen 契約を回避**
10. **status ファイルに「TODO として積む」で次回送り**（各 status で成功基準を達成）

コンテキスト不足時は `git stash` で保留 + 「中断スナップショット」section を
status ファイルに書き残し、**妥協実装を push して status を締めない**。

## STA2 防止ルール（STAP細胞の二の舞防止）
- **increment の定義**: increment は成功した dt ステップの数。カットバック（時間増分の縮小リトライ）は increment に含めない。`_incr_count` は成功パスでのみインクリメントし、`max_increments` はカットバック回数に侵食されない。
- **結果の再現性**: 全ての収束結果は tee でログ保存し、YAML 出力と照合可能にすること。ベースライン（変更前）を先に確認してから改善テストを実施。
- **数値の捏造禁止**: 収束しない場合は「収束しなかった」と報告する。目標を事後的に緩和して達成を装わない。

### 担当者間再現性ルール（status-246 追加）
- **ベンチマーク条件の記録**: テスト名、ブランチ名、コミットハッシュ、実行コマンドを tee ログおよび status ファイルに記録すること。
- **変更前ベースラインの先行取得**: 性能改善テスト前に必ず `git stash` で変更前コードのベースラインを計測し、ログに残す。
- **再現手順の status 記載**: status ファイルに「再現手順」セクションを設け、別の担当者が同じ結果を得られるコマンド列を明記する。
- **Process profiling の活用**: `ProcessMetaclass._profile_data` による自動計測結果を活用し、手動計測に頼らない仕組みを推進する。

### セッション開始時の確認手順
1. `docs/status/status-index.md` → 最新 status 番号を確認
2. 最新 `docs/status/status-{N}.md` を読む
3. `python contracts/validate_process_contracts.py` を実行し、エラー一覧を確認
4. 上の「やるべきこと」に合致する作業のみ実施
