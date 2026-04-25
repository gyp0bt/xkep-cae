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

**459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10 テスト** — 2026-04-25 | 契約違反 **0件** | 条例違反 **0件** | **MCDD status-373（TODO 整理 + 症状緩和系 experiment 5 本削除 + solver_mode 設計追記、documentation status）**。status-372 までで Phase E 候補 (a)〜(g1) を全て検証完了（(c)/(d)/(e)/(g1) は 7 本系 opt-in escape hatch、19 本 Type D stall 本体は未解決）。本 status は実装本体無変更で書類整理のみ: (1) `work/beam_hysteresis/{15,16,22,25,26}_*.py` を `git rm`（status-358/360/363/368/372 で却下確定済の症状緩和実験、結論は各 status に記録済）、(2) `次の課題`/`凍結中 TODO` を status-index 参照に圧縮、(3) `phase_c3prime_19strand_plan.md` §4 に `solver_mode` 併存方針追記（陰解法 default / リスタート opt-in）、(4) `docs/roadmap.md` opt-in 表に `solver_mode` 行追加。次候補 **(g3) pair-wise relaxation**（status-284 接触凍結を pair granularity 拡張）→ (g2) AL 再導入は status-374 以降で着手。gate: 契約違反 **0 件**（全 24 検査 OK）/ 条例違反 **0 件** / `pytest xkep_cae/contact/` **456 passed 5 skipped** 維持 / mathematics 109 passed / `test_helical_3d_hermite` rel_err=2.18e-07 維持 / ruff check + format pass。Phase A〜E / status-346〜373 の **24/N 完了**

### ターゲット

> **1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

### 次の課題

完了履歴の詳細は `docs/status/status-index.md` 参照。現在のアクティブライン:

- **status-374 候補 (g3) pair-wise relaxation 実装** — 19 本 Type D stall 解消の最後の系列実験。status-284 接触凍結を pair granularity 拡張（チャタリング pair のみ freeze + 残り active 維持）。設計仕様: `xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md` §3.2。Phase 1: `xkep_cae/contact/freeze/PairwiseFreezingProcess` 単体実装 + 単体テスト。Phase 2: `ContactFrictionProcess.freeze_slot` 配線 + 19 本実機検証（gate: frac ≥ 0.6）。却下時は (g2) AL 再導入へ。
- **status-374 副次** — `solver_mode` フラグ実装（陰解法 default / リスタート opt-in）。設計は `phase_c3prime_19strand_plan.md` §4 / `docs/roadmap.md` opt-in 表。
- **多 pair 診断 `14b_kc_multi_pair_diagnostic.py` 追加** — status-370 §5 保留。(g3) 採択方向時に検証ライン補強。

凍結中の派生 TODO（被膜圧縮モデル / リスタート方式 / ファイバー梁キャリブレーション / 7本撚線ピッチ依存性 / 空間ブロック分離）は MCDD Phase E 凍結解除後に再開（下記「凍結中の TODO」参照）。

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
- **status-374（次セッション・候補 (g3) Phase 1 実装着手）**: `xkep_cae/contact/freeze/PairwiseFreezingProcess` 単体実装 + 単体テスト。設計仕様は `phase_c3prime_19strand_plan.md` §3.2、却下時は (g2) AL 再導入

**凍結中の TODO**（MCDD 完了まで再開禁止）: 詳細項目（被膜圧縮モデル / リスタート方式 / ファイバー梁キャリブレーション / 7本撚線ピッチ依存性 / 空間ブロック分離 / 19本 Type D stall K_mat x/z 単発対応）は status-345 までで列挙、status-373 で本ブロックから削除。

**凍結解除条件**: Phase E 完了 + 19本 frac=1.0 完走 + `KcNormalDirectionStiffness`
FD rel_err < 1e-2。

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
