# xkep-cae ロードマップ

[← README](../README.md)

## プロジェクトビジョン

汎用FEMソフトでは解けないニッチドメイン問題を解くための自作有限要素ソルバー基盤。
構成則・要素・ソルバー・積分スキーマ・接触・非線形をモジュール化し、
問題特化ソルバーを構成するフレームワーク。

> **ターゲット: 1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

---

## 現在地（2026-05-12）

**459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+34+10+11+12+5+17+11+6+12 テスト** | 契約違反**0件** | [最新status](status/status-index.md) | [達成確認マトリクス](status/verification_matrix.md) | [数理台帳](math/README.md)

> **★ status-399 `explicit_n_sub_cycles_per_increment` 実装 — ε-1 で N=1000 で
> rel_err 6.07% 達成（MCDD 凍結解除条件 (5) 単 strand 規模で PASS）**:
> status-398 で確定した hypothesis 1（stepwise prescribed BC × mass scaling
> auto-tune の interaction）に対する architectural fix を実装。
> `ContactFrictionInputData.explicit_n_sub_cycles_per_increment: int = 1` field +
> `StrandBendingOscillationConfig` 3 経路 plumb-through + `process.py` の
> explicit 経路に sub-cycle 内部ループ実装（線形補間 prescribed BC + f_ext +
> MPC 射影、`_dt_inner = dt_sub / N`）。Default OFF (N=1) で既存挙動完全保持。
> **ε-1 再検証**: implicit u_x=4.996mm vs explicit-TL N=1 0.186 (96.29%) → N=10
> 0.759 (84.82%) → N=100 2.323 (53.50%) → **N=1000 5.299 mm (rel_err 6.07%)** で
> **MCDD 凍結解除条件 (5)（精度 < 10%）を単 strand 規模で PASS**。**status-398
> n_inc=20000 (β≈46 / u_x≈5.27 mm) と独立軸数値整合** で hypothesis 1 を確証。
> 単体テスト 8 件追加（monkeypatch で `ExplicitDynamicProcess.process` 呼出
> 回数直接計装）。回帰 **755 passed 5 skipped** / 全 24 契約検査 OK /
> `test_helical_3d_hermite` rel_err=2.18e-07 維持 / ruff pass。
> `verification_matrix.md` §2 ε-1 ✅（単 strand）+ §3 driver 行 🟡→✅
> （fix 実装、3 strand / 接触 / 多 strand は ⬜ 未検証）。Phase A〜E /
> status-346〜399 の **50/N 完了**.
>
> **★ status-398 `_process_free_end` × explicit-TL 3 仮説切り分け診断 — 仮説 1
> 確定（stepwise prescribed BC × mass scaling auto-tune の interaction）、
> n_inc 掃引で asymptotic 5.45% rel_err 到達、fix 実装は status-399 へ持ち越し**:
> status-397 ε-1 確定（`_process_free_end` × explicit-TL の under-deformation を
> 1 strand 規模で再現）を受け、CLAUDE.md 3 仮説切り分けを実施。
> `work/beam_hysteresis/42_status398_hypothesis_diagnostic.py` 新設（~250 行）で
> n_inc / β_max / t_cycle 軸を独立に振る 5 ケース resilient diagnostic + n_inc=20000
> asymptote 確認。**実測**: u_x baseline 0.186 mm → A1 0.748 → A2 2.252 → **n_inc=20000 で
> 5.268 mm (rel_err 5.45%)** で implicit 4.996 mm に asymptotic 収束。
> **根本機構**: stepwise prescribed BC + mass scaling auto-tune の interaction で
> β_auto=O(10⁴) が T_1_scaled=β·T_1_raw を肥大化、t_total<<T_1_scaled で wave 伝播
> 不能。仮説 2/3 は単一機構（仮説 1）で全観測値説明可能なため保留判定。
> **fix 設計（status-399）**: `explicit_n_sub_cycles_per_increment` field +
> `process.py` sub-cycle 内部ループ + 線形補間 prescribed BC（pseudo-code レベル
> 明記）。Default OFF で既存挙動完全保持、N>>1 で mass scaling auto-tune が
> β=O(10-100) に target → quasi-static 化。回帰 **747 passed 5 skipped 維持**
> （実装本体無変更）/ 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07
> 維持 / ruff pass。`verification_matrix.md` §3 driver 行を ❌→🟡。Phase A〜E /
> status-346〜398 の **49/N 完了**.
>
> **★ status-397 ε-1 失敗 — `_process_free_end` × explicit-TL の精度問題を
> 1 strand 規模で再現、改修対象を BC/process driver 層に局在化**:
> status-396 で API 化された `explicit_ul_disable_update=True` を **3 strand
> helical + 接触なし** の実機系で初検証する ε-1 を実施。
> `work/beam_hysteresis/41_epsilon1_3strand_helical_no_contact.py` 新設（~330 行）。
> **ε-1 主実験 FAIL**: implicit baseline は解析 cantilever 解と機械精度級一致
> （`u_x=4.996mm` vs 解析 4.996mm）、explicit-TL は `u_x=0.182mm` で
> **96.36% under-deformation**（u_z 96.54%）、frac=1.0 + E_kin/E_str=4e-10 で
> 動的緩和完了済みの定常解。**sub-experiment n_strands=1（直線、ヘリカルでない
> 単一 strand）でも FAIL 再現**（u_x 96.29% / u_z 96.40%）。CLAUDE.md 3 候補
> のうち (a) ヘリカル初期 κ / (b) 多 strand global assembler を即時除外、**(c)
> `_process_free_end` 駆動経路 + explicit-TL の組合せ自体が主因**と局在化
> （status-394 Mode C 専用ドライバ + status-395 γ-3 inline chain solver は
> 機械精度 PASS していたため、改修対象は process 主ループそのもの）。仮説 3 つ:
> (1) prescribed BC TL 増分処理 / (2) explicit reaction force 累積 /
> (3) `_ExtendedULAssemblerWrapper` 等の TL モード対応 — status-398 で切り分け。
> ロードマップ 5→6 段階拡張（397 FAIL → 398 仮説検証 → 399 修正後 ε-1 再検証
> + ε-2 → 400 ε-3 → 401 ε-4）。回帰 **747 passed 5 skipped 維持**（実装本体無変更）
> / 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07 維持 / ruff
> check + format pass（204 files）。`verification_matrix.md` §2 Phase ε section
> 新設 + §3 上位層改修対象に `_process_free_end` driver 行追加。Phase A〜E /
> status-346〜397 の **48/N 完了**.
>
> **★ status-396 explicit-TL 固定 API 化 — `explicit_ul_disable_update` 独立フィールド
> 追加（候補 (z3) Phase 1、API 化完結 / 実機検証 scope 外）**:
> status-395 §6.2 で確定した最優先項目 (z3) を実施。`solver_mode="explicit"` でも UL
> `update_reference()` を一切呼ばない TL 固定モードを **独立フィールド**
> `explicit_ul_disable_update: bool = False` で API 化。`ContactFrictionInputData` +
> `StrandBendingOscillationConfig` 各 1 field 追加（既存 `explicit_ul_update_interval`
> と独立、AND ゲート評価）+ 3 経路（曲げ / 揺動 / free_end）plumb-through。
> `process.py` ゲート式を `_solver_mode != "explicit" OR (not disable AND interval gate)`
> に更新（implicit 経路完全無変更）。`TestExplicitULDisableUpdate` 4 ケース追加
> （disable=True 0 回 / interval override / default 既存挙動保持 / ゲート式直接検証、
> `_MockULAssembler` で呼出回数直接計測）。**Default OFF 完全保持**: ゲート式は
> `disable=False` で status-383 と数式的等価、既存 743 passed 5 skipped 無変更。
> Phase α/β/γ で foundation 健全性確定 → 本 status で公開 API レベル運用可能化。
> 19 本 / 多 strand 実機検証は status-397 (ε-1: 3 strand helical + 接触なし +
> `disable=True`) で別 scope。回帰 **747 passed 5 skipped**（status-395 の 743 +
> 新規 4）/ 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07 維持 /
> ruff check + format pass（203 files）。`verification_matrix.md` §3 上位層改修対象
> に explicit-TL 固定 API 行追加 + §8 達成済リスト追記。Phase A〜E / status-346〜396
> の **47/N 完了**.
>
> **★ status-395 Phase γ-3 完了 — 多要素 explicit + TL で circular arc 収束を
> O(1/n²) 再現実証（4/5 PASS、log-log slope=-2.000、γ-1 implicit と数値一致）**:
> ユーザー指示「implicit 完全凍結」を受け explicit 一本路線の foundation 確認
> として Phase γ-3 を実施。`work/beam_element_validation/51_gamma3_multi_element_explicit.py`
> 新設（~370 行）で n_elements ∈ {1,2,4,8,16} を α-3 / β-2 / γ-1 と同 BC +
> slow ramp 5T_1 + hold 5T_1 + ζ=2 過減衰で駆動。explicit chain solver を inline
> 実装（lumped mass + leap-frog Verlet）、UL `update_reference` を呼ばない TL モード
> 固定。**実測**: n=1 のみ FAIL（24.95% chord 長保存制約、期待通り）、n=2,4,8,16
> で 3 指標すべて PASS、**log-log slope=-2.000**（O(1/n²)）、**γ-1 implicit と
> 全 n で差 < 0.01% の数値一致**。CR closed form / polyline 長保存も機械精度 0.000%。
> Phase α (1 要素 implicit) → Phase β (1 要素 explicit) → Phase γ-1 (多要素 implicit)
> → Phase γ-3 (多要素 explicit + TL) で **CR foundation の static / dynamic /
> multi-element / explicit 全領域での健全性が定量実証**、status-394「explicit + UL
> per step のみ FAIL」を裏付け。(z2) Cosserat 路線は不要、implicit 凍結方針下で
> plan B も scope 外。**次セッション最優先**: 候補 (z3) explicit モード TL 固定
> API 化 + 19 本撚線適用 / 副次 Phase δ 接触あり 2 本撚線 sanity check。
> `verification_matrix.md` §2.3 γ-3 ✅ 化。実装本体無変更、回帰 743 passed 5 skipped。
> Phase A〜E / status-346〜395 の **46/N 完了**.
>
> **★ status-394 assembler / UL update_reference 1 要素再現実験 — 改修対象を
> explicit + UL のみ に局在化（4 モード比較で A/B/C PASS、D FAIL 99.85%）**:
> status-393 §6.1 で次セッション最優先候補として明示された assembler / UL
> update_reference の 1 要素規模再現実験を実施。Phase β-2 直接駆動（status-391
> 機械精度 0.000%）+ Phase γ closed form 機械精度（status-392）の foundation 健全
> 実証を踏まえ、status-381〜387 の精度問題（解析解の 50%〜99% アンダー）を
> 1 要素規模で **再現**。`work/beam_element_validation/49_beta2_with_assembler_ul.py`
> 新設（~330 行）で 4 モード比較: A=implicit+TL / B=implicit+UL / C=explicit+TL /
> D=explicit+UL（毎 step）。**実測**: A/B/C すべて 3 指標機械精度 PASS（0.000%、
> Hermite 解 u_x=-0.02811 mm / u_z=0.7493 mm / L_chord=10.000 一致）、**Mode D のみ
> FAIL（u_x 99.85% / u_z 96.14% アンダー、L_chord は 10.000 保存）**。改修対象は
> **explicit + UL update_reference per step の組合せのみ** に局在し、(z2) Cosserat
> 路線は **不要** が 1 要素規模で定量実証。物理的解釈: 毎 step UL 更新 →
> `u_incr` がほぼゼロにリセット → `f_int(u_incr) ≈ 0` で elastic restoring force
> 不発（status-382 §3 解析と完全整合）。**次セッション最優先**: 候補 (z3) explicit
> モード TL 固定 API 化（`explicit_ul_update_interval=0` で update_reference を一切
> 呼ばない解釈）+ 19 本撚線適用 / 副次 (z4) sub-cycling / Phase δ 接触あり 2 本撚線。
> verification_matrix §3「上位層改修対象」更新。実装本体（`xkep_cae/`、`tests/`、
> `contracts/`）**無変更**、回帰 743 passed 5 skipped（status-393 と同数）/
> 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07 維持 / ruff pass。
> Phase A〜E / status-346〜394 の **45/N 完了**.
>
> **★ status-393 達成確認マトリクス導入 — STA2 連鎖撤回の構造的予防（documentation
> status）**:
> ユーザー指示を受け `docs/status/verification_matrix.md` を**永続ドキュメント**
> として新設（8 セクション初版）。status-379 / 381 / 387 の連鎖撤回事例を踏まえ、
> 達成 ✅ / 部分 🟡 / 未達 ❌ / 未検証 ⬜ / 凍結 ⏸ / 撤回 🔁 の独立な状態凡例で
> 「実証されていない」と「未検証」を明確分離、status-379 系の偽陽性を記号レベルで
> 構造的予防。§5 STA2 撤回履歴は削除禁止（透明性ルール）で「失敗の再演を防ぐ
> 予防接種」として機能。CLAUDE.md「作業完了時の必須手順」§5「マトリクス該当行を
> 更新」+「セッション開始時の必須確認」§3「マトリクス読込」追加で運用化。
> 現時点サマリ: 凍結解除条件 ✅ 1 / 🟡 1 / ❌ 3 / Phase α 4 件 ✅ / Phase β 2 件 ✅ /
> Phase γ-1 6 件 ✅ + 1 件 ❌（n=1 既知）+ 2 件 ⬜ / Phase δ ⬜ / 上位層改修対象
> 9 項目（⬜ 2 + ⏸ 1 + 🔁 6）/ 既存 validation gate 化 5 項目全 ⬜。実装本体（`xkep_cae/`）
> 無変更、回帰 743 passed 5 skipped。次セッション最優先候補（変更なし）: assembler /
> UL update_reference の 1 要素再現実験。
>
> **★ status-392 Phase γ 完了 — multi-element CR Timoshenko 梁の circular arc
> 収束を O(1/n²) で実証（4/5 PASS、log-log slope=-2.000）**:
> status-391 §6.1 Phase γ 計画に従い、CR Timoshenko 3D 梁要素を **直線チェーン**で
> n_elements ∈ {1, 2, 4, 8, 16} に並べた系を α-3 と同じ BC（左端 fix、右端
> θ_y=0.15 rad 処方）で **implicit static** に解き、circular arc 解への収束を
> 3 指標 AND gate（status-388 透明性ルール）で確認。`work/beam_element_validation/`
> に `_gamma_common.py`（`ChainedBeamSection` + `assemble_internal_force/tangent` +
> `solve_static_nr_chain`、~280 行）+ `47_gamma_multi_element_convergence.py`
> （~270 行）新設。**4/5 ケース PASS**: n=1 のみ FAIL（u_x で 24.95% — α-3 で
> 実証済み chord 長保存制約による既知の離散化誤差）、n=2,4,8,16 で 3 指標すべて
> PASS。**err(\|u_x\|): 24.95%(n=1) → 6.23%(n=2) → 1.56%(n=4) → 0.39%(n=8)
> → 0.10%(n=16)** で **log-log slope=-2.000**（理論値 O(1/n²) と完全一致）。
> **CR closed form 一致**（chord rotation φ_e=θ(e-1/2)/n の sum-to-product 解、
> `x_n = L sin(θ/2)cos(θ/2)/(n sin(θ/(2n)))`）は全 5 ケースで \|u_x\| / \|u_z\| /
> L_chord すべて **機械精度（10⁻¹³%〜10⁻¹²%）** — 実装が CR 多要素解析理論と
> 完全整合。polyline 長 = Σ L_elem も全ケース機械精度で 10.000 mm を保存。
> **結論**: CR foundation の multi-element アセンブル健全性確定、「16 要素/ピッチ
> 厳守」規範は典型 curvature レンジで十分なマージン（θ=0.15 rad ≈ 8.6° 単一曲げで
> n=2 から 10% gate を通過）。Phase α (1 要素 implicit static) → β (1 要素 explicit
> dynamic) → γ (multi-element implicit static) で **CR foundation の static /
> dynamic / multi-element 全領域での健全性が定量実証**。実装本体（`xkep_cae/`）
> **無変更**、回帰 743 passed 5 skipped（status-391 と同数）/ 全 24 契約検査 OK /
> `test_helical_3d_hermite` rel_err=2.18e-07 維持 / ruff pass（10 files already
> formatted）。**次セッション最優先候補**: assembler / UL update_reference の
> 1 要素再現実験（status-381〜387 精度問題の根因特定、`49_beta2_with_assembler_ul.py`）/
> 副次 Phase δ 接触あり 2 本撚線 / 副次 Phase γ-2 大 curvature 拡張（θ=π/2）/
> 副次 既存テスト 3 指標 gate 化。
>
> **★ status-391 Phase β 完了 — 1 要素 cantilever explicit central diff +
> lumped mass で β-1 自由振動 + β-2 explicit quasi-static 両 PASS（CR foundation
> explicit 健全確定）**:
> status-390 Phase α 完了を踏まえ Phase β に移行、`work/beam_element_validation/`
> に共通ヘルパ `_beta_common.py`（`solve_explicit_central_diff` leap-frog Verlet +
> `compute_natural_frequencies_fe` + `compute_strain_energy_cr`、Rayleigh 質量比例
> 減衰対応、~370 行）+ 検証スクリプト 2 本（45_β1 / 46_β2、~480 行）を新設。
> **β-1 自由振動**（v_z(tip)=1 mm/s、5 周期、α=0、lumped、dt/dt_crit=0.150）:
> T_period(FE 第 1 モード) **0.056%** / |u_z_max|(v_0/ω_1) **4.85%** /
> E_drift **0.016%** で 3 指標全 PASS、L_chord drift 7e-13 mm（実質ゼロ）。
> **β-2 explicit quasi-static**（α-3 と同 BC θ_y=0.15 rad、slow ramp 5T_1 +
> hold 5T_1、ζ=2 過減衰、dt/dt_crit=0.500）: |u_x_tip|=0.02811 / |u_z_tip|=0.7493 /
> L_chord=10.000 すべて **機械精度 0.000%** で α-3 implicit Hermite 解と完全一致、
> settle 残差 ||f_int_a||=2.13e-14 N、KE/SE=3.78e-27（quasi-static gate 完璧）。
> **重要含意**: status-381〜387 explicit + UL の精度問題は **CR 要素自体ではなく
> 上位層**（assembler / UL formulation / mass scaling 戦略）に局在することを定量
> 実証 — 1 要素直接駆動（assembler 経由なし、UL update_reference 不要）では explicit
> + 大回転で機械精度一致が成立する。**(z2) Cosserat 路線は absolute necessity ではない**
> — 主目的は explicit + 大回転 robust 化（assembler / UL update_reference 由来の
> 問題解消）に絞れる。実装本体（`xkep_cae/`）**無変更**、回帰 743 passed 5 skipped
> （status-390 と同数）/ 全 24 契約検査 OK / `test_helical_3d_hermite` rel_err=2.18e-07
> 維持 / ruff pass。**次セッション最優先**: Phase γ multi-element 検証
> （n_elements ∈ {2, 4, 8, 16} で α-3 を再実施 → circular arc 解への収束を確認、
> 「16 要素/ピッチ厳守」規範の妥当性再確認）/ 副次 Phase δ 接触あり 2 本撚線 /
> 副次 既存テストの 3 指標 gate 化 / 副次 assembler / UL 1 要素再現実験。
>
> **★ status-390 Phase α 完了 — CR Timoshenko 1 要素 implicit static 全 4 ケース
> PASS（foundation 健全確定）**:
> status-389 §2 計画に従い `work/beam_element_validation/` 新設（共通ヘルパ +
> 4 検証スクリプト ~860 行）、3 指標 AND gate（status-388 透明性ルール）で 1 要素
> implicit static 検証を実施。**全 4 ケース PASS（機械精度 0.000〜0.001%）**:
> α-1 純軸引張 F_x=100 N（u_x / u_z(=0) / L_arc / iters=2）/ α-2 純粋曲げ
> small κ M_y=10 N·mm（\|u_z\| / \|θ_y\| / \|f_int\| / iters=4）/ α-3 純粋曲げ
> large κ θ_y=0.15 rad（\|u_x\|=0.0281 / \|u_z\|=0.7493 / L_chord=10.000 /
> iters=40 with load stepping）/ α-4 cantilever 横荷重 F_z=0.01 N（\|u_z\| /
> \|θ_y\| / \|M_base\| / iters=2）。**重要発見 1（α-3）**: 1 要素 CR は
> **chord 長保存制約**で chord rotation α=θ_R/2 の Hermite 解（u_x=L(cosα-1),
> u_z=L sinα）を出す。circular arc 解との 25% 差は **1 要素の本質的離散化誤差**で
> Phase γ で n_elements ↑ により消失するはず。**重要発見 2（α-2）**: 実装の局所
> 剛性 Ke[u_z, θ_y]=+6 EI/L² 規約と plan の解析式 u_z=+M·L²/(2·EI) は符号が逆 →
> `MetricRow.compare_abs=True` で吸収。status-389 plan 表の数値ミス（3 ヶ所）は
> 実装側で**式から動的計算**することで透明性ルール準拠の正しい解析解 gate を実現。
> **結論**: status-389 §4 シナリオ「Phase α-3 で 1 要素 implicit が 3 指標 PASS
> → CR は static 規模で妥当 → (z2) Cosserat は explicit + 大回転 robust 化に
> 主目的を絞れる」を**支持**。Foundation 健全確定。実装本体（`xkep_cae/`）**無変更**、
> 回帰 743 passed 5 skipped（status-389 と同数）/ 全 24 契約検査 OK / ruff pass。
> **次セッション最優先**: Phase β-1 自由振動（SDoF Timoshenko 第 1 モード周期 +
> KE+SE 保存 + L_chord 保存 の 3 指標）+ β-2 explicit + slow ramp で α-3 と
> 10% 一致検証（**β-2 で FAIL → (z2) Cosserat 移行根拠 absolute 確定**、PASS →
> 大回転 robust 化に絞れる）。
>
> **★ status-389 引き継ぎ — 梁要素 1 つから系統的再検証 Phase 計画策定**:
> status-388 で透明性ルール（独立解析解 3 個以上同時一致）が status-387 の
> 「sweet spot 達成」誤判定を 11 分で反証したことを踏まえ、Phase α (1 要素静的)
> → β (1 要素動的) → γ (multi-element) → δ (接触あり 2 本撚線) の系統的計画を
> 策定。Phase α 計画は status-390 で完了。
>
> **★ status-388 で status-387 訂正・撤回 + 妥当性テスト透明性ルール策定（独立解析解 3
> 個以上同時一致を必須化）+ 単梁 explicit + UL は L_arc 不伸長性 gate で全 n_inc
> で大破綻**:
> ユーザーから **STA2 厳罰** + **透明性策定** + **3 個以上の解析解同時一致** の
> 要求を受け、status-387 の二重ミス（(1) 解析解 90° (73.30mm) を使うが実 BC は
> 86° (70.44mm)、(2) 単一指標 max\|u\| のみで判定）を撤回し訂正。CLAUDE.md
> 「STA2 防止ルール」に「**妥当性テストの透明性ルール（status-388 追加・厳罰）**」
> 追記: 独立 3 指標必須化（kinematics 2 + energetics-or-geometric 1）、`\|u\|`
> ノルムは導出値で独立指標カウント不可、SE 信頼できない場合は L_arc 等で代替可。
> **訂正版実機検証 14 ケース**: implicit baseline は 3 指標すべて PASS（kinematic
> err 0.1% / L_arc err 0.0%）、**全 13 explicit ケース FAIL**。**n_inc=8000（旧
> sweet spot）は kinematic 12.6%（10% gate 越え）+ L_arc 233.75mm（134% 過大、
> 梁が 2.3x に非物理ストレッチ）**、n_inc=16000 は L_arc 200% 過大（3x スケール
> 300mm）。**「sweet spot」の真相**: 梁が 2.3x に伸びる + 曲率が 1/3 に薄まる +
> 座標が偶然 (37.7, 62.4) で \|u\|≈73 → 90° 解析解 73.30 と偶然交差。3 指標 AND
> gate で確実に検出される非物理解。**MCDD 凍結解除条件 (5) 未達続行**、status-387
> 撤回確定。次候補は **(z2) Cosserat 梁プロトタイプ最優先**（explicit + UL は本質
> 的破綻と確定、SO(3) 回転 DOF + reference 更新不要 + 軸方向拘束 exact 維持で唯一
> の本質解決路）。実装本体（`xkep_cae/`、単体テスト、契約検査）は **無変更**、
> 回帰 743 passed 5 skipped（status-386/387 と同数）/ 全 24 契約検査 OK / ruff pass。
>
> **★【⚠️ status-388 で撤回】status-387 で単梁 90° 曲げの `n_increments` 大化掃引で
> sweet spot 発見 — explicit + UL の精度 gate (5) を `n_inc=8000` で達成（err 0.58%）**:
> status-386 §5.4 副次「t_cycle 据え置き + n_increments 大」探索を実施。
> `work/beam_hysteresis/40_explicit_n_inc_sweep.py` 新設（+233 行、13 ケース）で
> `n_inc ∈ {200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000}` を
> uniform β² (selective=False) / `max_beta=10⁴` / `t_cycle_min=1.0` 据え置きで掃引。
> **主要発見: n_inc=8000 で max\|u\|=72.88mm（解析解 73.30mm の 99.4%、err 0.58%）**を観測、
> **MCDD 凍結解除条件 (5)「精度 < 10%」を単梁で達成**（status-381 以降の explicit + UL
> 路線で初の gate 通過）。収束は **単峰非単調**: n_inc=200→8000 で max\|u\| が
> 6.57→72.88mm へ単調増加、n_inc≥10000 で **overshoot**（n_inc=16000 で 106.10mm、
> err=44.76%、β=58 で残存質量不足）。**Damping + relax 併用は逆効果**（α=5.0 で
> n_inc=8000 max\|u\| 72.88→19.22mm に圧縮、UL 凍結のため `[RELAX] converged at step 1
> ||R||=0` で動かす力源なし — status-382 §3 知見と整合）。**sweet spot β=116 の物理
> 解釈**: t_cycle=1.0s 内で波が梁を 329 回横断（過渡応答完全減衰）+ 残存質量で
> 動的振動有効減衰 + UL 凍結問題化なし（Δu/incr=0.011° で CR 梁 UL 線形化レンジ内）。
> **status-386 結論部分修正**: 「(z1*) 全候補で精度 gate 達成不能」は
> 「(z1d) 方向では達成不能、(z1d) 反対方向 + n_inc 大 + damping=0 + sweet spot で
> **単梁では**達成可能、19 本適用は未検証」へ。**MCDD 凍結解除条件達成判定は時期尚早**
> （条件 (2) 19 本 frac=1.0 未検証、19 本領域で sweet spot 機能するかは別途）。
> 次候補は **(z2) Cosserat 梁プロトタイプ最優先**（sweet spot 依存を脱却するため
> UL 凍結を本質解決）/ 副次 (5.3) 7 本 + n_inc=8000 1 ケース実測 / (5.4) 19 本
> n_inc 掃引（条件 (5.3) 確認後）。実装本体（`xkep_cae/`、単体テスト、契約検査）は
> **無変更**、回帰 743 passed 5 skipped（status-386 と同数）/ 全 24 契約検査 OK /
> `test_helical_3d_hermite` rel_err=2.18×10⁻⁷ 維持 / 7 本 implicit frac=1.0 / ruff pass。
>
> **★ status-386 で候補 (z1d) `t_cycle` 下限緩和実装 — z1d は方向自体が逆と
> 単梁実機で実証、explicit + UL 精度 gate 未達続行**:
> status-385 §6.1 最有力候補 (z1d) として
> `StrandBendingOscillationConfig.t_cycle_min_seconds: float = 1.0` field を追加、
> `t_cycle = max(10·T1, cfg.t_cycle_min_seconds)` で下限を外部制御可能化
> （default 1.0 で既存挙動完全保持）。**+6 単体テスト**（`TestTCycleMinSeconds`）全 pass。
> **`39_z1d_t_cycle_validation.py` 11 ケース単梁中心実機検証**: (a) z1d 自体は
> 設計通り動作（initial target β 4.7×10⁴ → 3.1×10³ の **15x 縮小**ログ確認）/
> (b) **implicit 側 regression なし**（t_cycle_min=0.0 で frac=1.0 完走、err 4.86%、
> baseline 3.90% との差 1pt 未満）/ (c) **explicit 側で逆効果**（selective+z1d 全
> DIVERGED、non-selective uniform β² 完走するも max\|u\|=0.77mm vs 解析解 73.30mm で
> **err 99%**、大 β_outside=2000 でも 1.83mm/97.5%）/ (d) **逆方向対照実験 (#11)
> `n_inc=200, t_cycle 据え置き`** で max\|u\|=6.57mm（z1d 方向の **10x 改善**）—
> **z1d は方向自体が逆と定量実証**。
> **真の物理原因**: mass scaling β は波速を `c→c/β` に減速、β=3000 で波の梁長
> 100mm 横断時間 78ms が t_cycle=67ms（z1d 適用後）を超過し変形が伝播しないまま
> frac=1.0 到達。`t_cycle_min_seconds` field は default 1.0 で保持
> （implicit 完全保持、explicit opt-in）。
> **MCDD 凍結解除条件 (5) 未達続行**、(z1*) 全候補で精度 gate 達成不能と確定、
> 次候補は **(z2) Cosserat 梁プロトタイプ最優先**（UL を捨てて explicit + 大回転を
> 本質解決）。
>
> **★ status-385 で候補 (z1c) 2 段階質量スケーリング API（β_stiff + β_outside）実装 —
> API 完成、validation で β_stiff cap が支配的と確認、(z1d) loading rate 縮小が必須と判明**:
> status-384 §6.1 最有力候補 (z1c) として `ExplicitCentralDifferenceProcess` に
> `mass_scaling_beta_outside` 引数 + `set_mass_scaling_beta_outside()` API
> （KE 保存 v/a リスケール、mask=False の DOF のみ）を追加。`_compute_scaled_mass()` で
> mask=False の DOF（梁）に β_outside² を、mask=True の DOF（stiff）に β² を適用。
> `_explicit_dynamic.py` の dt_c_beam 推定で mask 設定時は `β_outside` を乗じる。
> **+11 単体テスト**（`TestTwoStageMassScaling`）全 pass。
> **`38_z1c_two_stage_validation.py` 8 ケース実機検証**: API は設計通り動作（log で
> post-cutback target β が β_outside=10 で 8.8×10⁶ → 8.8×10⁵ に **10x 縮小**）も、
> initial target β=4.7×10⁴（β_stiff cap=10³〜10⁴ を超過）が支配的で全 explicit ケース
> frac=0 で divergence。aggressive scaling（β_outside=10, β_stiff_max=10⁶, α=10）で
> frac=0.425 進むも max\|u\|=1.6×10⁵ mm で精度 gate 完全違反。
> **結論**: (z1c) infrastructure は完成、しかし MCDD 凍結解除条件 (5) 達成には
> **(z1d) `t_cycle` 下限緩和** で loading rate を物理 T1 ベースに縮小し target β
> 自体を下げる必要がある。次候補は **(z1d) 最優先** / (z2) Cosserat 梁プロトタイプ
> 並行検討。
>
> **★ status-384 で Abaqus/Explicit 標準アプローチへの移行 Phase 1 完了**:
> ユーザー指摘「応力波の速度と要素サイズから dt」+「Cosserat 梁の大回転
> ネイティブ特性」を受け、status-383 までの "explicit + UL は原理的に
> 成立しない" 結論を踏まえて方針転換。**(z1a)** `_estimate_critical_dt_per_element`
> で `dt_e = L_e / √(E/ρ)` を要素ごとに計算し Gerschgorin 全体上界と min を取る。
> **(z1b)** `_detect_stiff_dofs()` で row-sum/M が median × `threshold_ratio` を
> 超える DOF を自動検出、`set_mass_scaling_dof_mask()` で β² 倍化を限定。
> **+17 単体テスト**全 pass。実機検証で **2 段階スケーリング要件**が判明:
> 単梁は K 一様で selective 検出ゼロ、7 本撚線でも beam DOF (β=1) が dt 制約を
> 支配し target β=8.8×10⁶ → cap 1000 超過。**真の解**: β_stiff=1000, β_beam=10
> 等の per-DOF β 配列 + loading rate 縮小。次候補は **(z1c) per-DOF β 配列 API**
> + (z1d) `t_cycle` 下限緩和 + (z2) Cosserat 梁プロトタイプ並行検討。
>
> **★ status-383 で候補 (q1) `explicit_ul_update_interval` 4 ケース掃引で却下、
> UL 凍結が真因と再確証**: status-382 §6.1 最有力候補として `solver_mode="explicit"`
> のとき UL `update_reference()` を **N 増分ごと** に呼出する gate を導入。
> `36_explicit_ul_interval_validation.py` 5 ケース掃引で全 FAIL — interval=1
> baseline 29.57mm（status-382 と一致、default 完全保持）/ interval=5 で relax
> phase 発散 (NaN) / interval=10 max\|u\|=6.21×10⁶ mm / interval=20 max\|u\|=5.16×10²¹ mm。
> **根本要因**: CR 梁 UL 定式化は「u_incr 微小」前提で線形化、N 増分蓄積は
> K_T(u_incr) を線形化レンジ外へ押し出し explicit dynamics が爆発的発散。
> status-382 §3 解析と整合：(a) update 毎呼出 → f_int(u_incr)≈0、(b) update 間引き
> → K_T(u_incr) 線形化崩壊、両方破綻。**explicit + UL の組合せは原理的に成立しない**
> ことが (q1) で再確証。MCDD 凍結解除条件 (5)「精度 < 10%」未達のまま、次候補は
> **(q2) 増分内 sub-cycling**（最有力、UL 動作を 1 BC 増分内で保ちつつ内部力評価を
> 意味あるものに）/ (q3) implicit + AL n>2 復活 / (h5) bending 段階処方。
>
> **★ status-382 履歴**: status-381 §7 引継ぎの仮説 (p3) 質量比例 Rayleigh damping
> + (p1) BC 完了後 relax phase の 2 API を実装したが、`35_explicit_accuracy_validation.py`
> 6 ケース全 FAIL。`exp_no_damp_relax500` が baseline 35.37 と本質的に同値の 35.41mm
> （解析解 73.30mm の 51% off）、`[RELAX] converged at step 1 ||R||=0` ログで relax
> 即終了。**真の根本原因**: UL `update_reference` が各増分の dynamic lag を reference
> に凍結 → `_ul_internal_force_wrapper(state.u)` で `u_incr = state.u − _ul_ref_base ≈ 0`
> → `f_int(0) = 0` → relax で平衡へ駆動できない。
>
> **★ status-380〜381 履歴**: status-379 候補 (h1) mass scaling auto-tune が形式
> gate（frac=1.0 + E_kin/E_strain<5%）を満たしたが、status-380 物理的妥当性検証で
> **7 本/19 本ともに `max\|u_trans\|=1.59×10⁸ mm`（≈159 km）の数値発散**が発覚、
> status-379 凍結解除判定撤回。status-381 で 3 仮説切り分けによる **h-bug-1（v/a
> リスケール欠落）+ h-bug-3（β 急成長）** 確定 → KE 保存リスケール + 4× growth cap +
> 増分 1 warm-start 実装で発散停止（41 mm）。しかしユーザー指摘で精査すると単梁
> 解析解 73.3mm に対し explicit 40mm（48%）と **系統的 50% アンダー**、精度 gate (5)
> 「< 10%」未達で再撤回。MCDD 数理契約駆動開発（MCDD）Phase A〜E（status-346〜382、
> status-354 で 1 status 後ろ倒し）
> （**31/N 完了** — status-379 候補 (h1) mass scaling auto-tune が形式 gate（frac=1.0 + E_kin/E_strain<5%）を満たしたが、status-380 物理的妥当性検証で **7 本/19 本ともに `max\|u_trans\|=1.59×10⁸ mm`（≈159 km）の数値発散**が発覚。3D 可視化で撚線が空間に飛散することを実証、status-379 の凍結解除判定は **撤回**。**根本原因**: `frac=1.0` は処方変位 BC 達成のみ、`E_kin/E_strain` は β² 倍化された両エネルギー比なので β に独立で発散時にも PASS、両 gate は数学的構造由来。CLAUDE.md 凍結解除条件に **物理的妥当性 gate**（`max\|u_trans\| < L_strand × C`、C=10）を追加。次候補は (h4) implicit AL n>2 with Uzawa under-relaxation（implicit は max\|u\| が物理範囲）/ (h1') β cap 強化 / (h5) bending 段階処方 / (h2) dt subcycling / (h3) selective explicit。19 本既知最良は status-376 implicit AL n=2 の frac=0.5746。それ以前の経緯: Phase A-1〜A-2 + B-1〜B-2 + C-1〜C-2 + 数理台帳訂正 status-353 + Phase C-3 再定義実験 status-354 + Phase C-3' 診断 status-355 + Phase C-3' 実装 status-356 + Phase E 着手 status-357 + Phase E C20 + 仮説 C 候補 (a) 反証 status-358 + 仮説 C 候補 (a') 採択 status-359 + (a') 19本却下 + Phase E C21/C22/C23 status-360 + 7/19 挙動反転の幾何・Type 分布分析 status-361 + status-362〜363: 候補 (c) line search 実装+掃引（クローズ）+ status-364: C24 hollow VerifyProcess 封じ込め + status-365〜367: 候補 (e) 接触減衰（7本採択 -57% / 19本却下）+ status-368: 候補 (d) 接触凍結モード 19 本（nr_max=30 で +16.6% / クローズ）+ status-369: Case B opt-in 化 + 候補 (f) 計画 + status-370: Phase C-3' Step 3.1 結果 B 確定（NR alg 側動力学）+ status-371〜372: 候補 (g1) EMA 平滑化（7本部分達成 / 19本却下）+ status-373: TODO 整理 + status-374〜375: 候補 (g3) pair-wise relaxation Phase 1+2（19本却下）+ status-376: 候補 (g2) AL 外側ループ（19本 +53.7% gate 0.026 不足で却下）+ status-377〜378: 陽解法 Phase 1（Process 単体実装）+ Phase 2（solver path 配線、Courant 比 3×10⁵ 実測）+ status-379: 陽解法 Phase 3 候補 (h1) **mass scaling auto-tune で 19 本 frac=1.0 完走 ★凍結解除**。それ以前の経緯: Phase A-1〜A-2 + B-1〜B-2 + C-1〜C-2 + 数理台帳訂正 status-353 + Phase C-3 再定義実験 status-354 + Phase C-3' 診断 status-355 + Phase C-3' 実装 status-356 + Phase E 着手 status-357 + Phase E C20 + 仮説 C 候補 (a) 反証 status-358 + 仮説 C 候補 (a') 採択 status-359 + (a') 19本却下 + Phase E C21/C22/C23 status-360 + 7/19 挙動反転の幾何・Type 分布分析 status-361 + status-362〜363: 仮説 C 候補 (c) line search 実装+掃引（候補 (c) クローズ）+ status-364: Phase E C24 — hollow VerifyProcess 構造的封じ込め + status-365〜367: 候補 (e) 接触減衰 escape hatch（Phase 1+2+validation、7本採択方向 -57% / 19本却下）+ status-368: 候補 (d) 接触凍結モード 19 本再評価（nr_max=30 で +16.6%、未達でクローズ）+ status-369: Case B 19 本 opt-in ガイドライン化 + 候補 (f) Phase C-3' 実験計画 策定 + status-370: Phase C-3' Step 3.1 完了（active 境界 FD 診断で結果 B 確定）+ status-371〜372: 候補 (g1) active 履歴 EMA 平滑化（実装+α 掃引、7 本部分達成 / 19 本却下） + status-373: TODO 整理 + 症状緩和系 experiment 5 本削除 + solver_mode 設計追記 + status-374: 候補 (g3) pair-wise relaxation Phase 1 — `PairwiseFreezingProcess` 単体実装 + 12 単体テスト + **status-375: 候補 (g3) Phase 2 NR 配線 + 19 本実機検証で却下（threshold ∈ {2,3,5} 全 3 ケース Gate frac≥0.6 未達 / DOF block 上書きが隣接 pair 正フィードバック誘発 / 次候補 (g2) AL 再導入）**）。
> 旧計画書 `/root/.claude/plans/deep-wiggling-seal.md` は **永久ロスト**
> （status-352 で記録）。以降、計画情報は本 roadmap と CLAUDE.md・
> `docs/status/status-{N}.md` に転記して運用する。
> **status-353 訂正**: 当初 Phase C-3 の `KcNormalDirectionStiffnessProcess` は
> **既存 `KcGeoStiffnessProcess` と数理的に同一**（重み $p_n/d$、$1/d$ は
> $\hat n = r/d$ 内在項）で撤回。**status-354 反証**: 仮説 A（`K_hermite_adj`
> 単独フル項化 = `-w_geo * I_nn` 追加）は `test_helical_3d_hermite` rel_err
> 1.795% → 38.49%（21 倍悪化）で**単独では過剰計上**。**status-355 診断**:
> rel_err の 100% が active×adj ブロックに局在、仮説 B 目標を
> `||diff[ax]|| 98.52 → <1e-3` に定量化。
> **status-356 解決**: 仮説 A（フル項 (i) 直接経路）と仮説 B（`K_closest` /
> `K_st` active×adj 拡張 = (ii) s-tracking 経路）を**同時導入**することで
> 2 経路の $P_\perp$ 成分が解析的に相殺し、`test_helical_3d_hermite`
> **rel_err 1.795% → 2.18e-07**（FD 機械精度）、`||diff[ax]|| 98.52 → 4.75e-05`
> （6 桁改善）達成。
> **status-357 実機規模検証 + Phase E 着手**: 19 本撚線 K_c FD 再計測で
> **frac=0.3739（status-344 比 -22.7% 退化）、mat_only rel_err mean=0.508
> （+15% 悪化）**。gate テスト rel_err 2.18e-07 達成は active 集合固定下の
> 解析的 K_c 限定であり、19 本 Type D stall（NR Type D+E:67%, E:28%）の
> active 集合振動支配領域には波及しないと判定。**仮説 C（active 集合振動対策）**
> を status-358 最優先に昇格。副次: status-356 で混入していた C5 違反
> （`KcHermiteNonlocalStiffnessProcess.process()` の `_batch_dm_ext_coeffs`
> クラスメソッド直接参照）を module-level 関数化で解消。**Phase E 着手**:
> C18（`@verified_by` 紐付け）+ C19（`TermExpansionContract.providers` 実在）
> を `contracts/validate_process_contracts.py` に追加、5 term-provider Process
> に `@verified_by("K_c_term_expansion", ContactKcComponentFDDiagnosticProcess)` 付与。
> **status-358 仮説 C 候補 (a) 反証 + C20 追加**: 仮説 C 候補 (a)
> （`smoothing_delta` 遷移帯 4x 拡大、default 2000→500）を 7本撚線 90° 曲げで
> 実測（ユーザー指示「19本でなく 7本 90°、10% 以上改善 + frac=1.0 完走が採択基準」）。
> ベースライン（frac=1.0000, incr=524, cb=57, 452.02s, チャタリング 166 件）に対し
> **候補 (a) は frac=0.9241 で未完走**（converged=False）、cutback -14% /
> elapsed -17% の見かけ改善は解析の早期打切りで対策効果ではない。**却下（revert）**、
> コード変更なし。**Phase E C20 追加**: `TermExpansionContract.providers` に
> 列挙された Process クラスが自身の `contracts` ClassVar で同名契約を宣言
> しているかを静的検査。C18/C19 の片側更新による脱法すり抜けを防御、
> 5 既存 providers で回帰なし（C18/C19/C20 含む 20 検査 OK）。
> **status-359 仮説 C 候補 (a') 採択（実験記録）**: 候補 (a) 4x 拡大が厳し
> 過ぎたため **2x 拡大中間値**（`smoothing_delta=1000`、default 2000 の 1/2）で
> 再試行。**frac=1.0000 完走 + n_increments=475（-9.4%）+ n_cutbacks=53（-7.0%）+
> elapsed=259.92s（-42.5%、1.74x 高速化）**。ユーザー指示の合否基準
> 「frac=1.0 + 10% 以上改善」に対し elapsed -42.5% で大幅クリア（cutback は
> 補助指標で 10% 未満だが elapsed 半減近い改善は active flip 抑制で各
> increment の NR 反復数が減った効果として十分）。**判定: 採択方向**。
> ただし `StrandBendingOscillationConfig.smoothing_delta` の default 変更
> （2000→1000）は **本 status では実施せず**（7 本撚線のみの検証で 19 本
> Type D stall 本体への有効性未検証）、`15_hypothesis_c_7strand.py` を
> 成功実験記録として残置（status-358 の (a) 失敗実験 revert と対称）、
> 実装本体（`xkep_cae/`、`tests/`、`contracts/`）は **無変更**。
> 次セッション最優先は (i) 仮説 C (a') の 19 本撚線検証 → (ii) default 化
> 判断 / 失敗時 (c) line search 強化（`_newton_dynamic.py` に line search
> hook 追加）。
> **status-360**: 仮説 C (a') を 19 本撚線に適用、**frac=0.3723（baseline 0.4839
> 比 -23.1% 退化）で却下**。δ_h 2x 拡大は Type D stall 領域で逆効果。default
> 変更は実施せず、`16_hypothesis_c_aprime_19strand.py` を**失敗実験記録**として
> 残置。次候補は **(c) line search 強化**（`_newton_dynamic.py` に backtracking
> hook 追加）。副次: **Phase E C21/C22/C23 追加** — C21 `TermExpansionContract.
> term_names` 重複静的検出、C22 `contracts` ClassVar 同名契約重複検出、
> C23 `@verified_by` 検証 Process カテゴリ（SolverProcess / VerifyProcess 必須）。
> 2 テスト追加で mathematics/tests 97 passed、全 23 契約検査 OK。
> **status-361 挙動反転原因切分**: 7/19 本挙動反転の原因を (1) 幾何モデル検証
> （逆巻き S/Z は正常実装、ただし全層同一 pitch で外層ヘリックス角 2x、接触
> ペア密度 8.5x）、(2) 19 本軽荷重 κ=0.005 で frac=1.0 完走 → **幾何モデル
> 自体は正常、挙動反転は接触密度依存の数値問題**、(3) Type 分布実測で
> **19 本重荷重のみ mixed (C+D) 16.6% 突出**（他 3 ケース 1-4%）、K_c x/z
> カップリング不整合が active flip と同時発火する領域が本質。δ_h 拡大は
> mixed (C+D) に悪化。次手は (c) line search 強化で mixed 領域を直接抑制。
> **status-362 仮説 C 候補 (c) 実装 + 実機検証（部分的前進）**:
> `ContactBacktrackingLineSearchProcess` 新設（`_newton_steps.py` +112 行）
> で既存 `NCPLineSearch` の ||R_u|| 全体発散判定では捉えられない**接触残差比
> / active flip 過剰増加**を検知し α を半減する backtracking を
> `_newton_dynamic.py` NR 主ループに組込。トリガー条件 `att≥2 &
> n_active≥1 & active_set_changed & _conv_rate>0.85` で mixed 狭義検知、
> `active_flip_threshold` は `max(abs=3, ratio=0.3 × n_active_pre)` の
> 相対判定。4 層で 9 field plumb-through、**default OFF で既存動作不変**。
> `TestContactBacktrackingLineSearchProcessAPI` 6 テスト + default OFF
> regression 163 passed 6 skipped 1 xfailed で回帰なし。
> **実機検証結果**: 7 本撚線（status-359 設定）**frac=1.0000 完走 /
> elapsed=285.64s（+9.9%、20% 許容内）で回帰なし**。19 本撚線（baseline
> `frac=0.4839` stall）**frac=0.5153（+6.5% 改善）/ cb 39→38（-2.6%）/
> elapsed 534→729s（+36.4%）** で MCDD 凍結解除条件「frac=1.0 完走」
> **未達**。最終停滞時 NR Type 分布 `D+E:51%, E:43%`（baseline `D+E:67%,
> E:28%` より mixed 減、BT 部分効果を示唆）。BT 発動数 52（全 NR 反復の
> ~1%）で trigger が保守的過ぎる可能性、次候補は (c) パラメータ感度
> 探索（`rate_threshold=0.7` / `active_flip_ratio=0.15` / `mixed_only=False`
> の掃引） or (d) 接触凍結モード 19 本適用。
>
> **status-363 仮説 C (c) パラメータ感度掃引 — 4 ケース全却下、BT 既定が
> 局所最適**: `work/beam_hysteresis/22_bt_parameter_sweep_19strand.py`
> 新設で 3 軸 4 ケース（A: rate_threshold=0.70 relaxed / B:
> active_flip_ratio=0.15 strict / C: mixed_only=False always_on / D:
> A+B+C combined）を 19 本撚線 90° 曲げで実測。結果: **全ケース
> frac<1.0 未達**（A=0.5153 BT default 同値 / B=0.4701 **-8.8% 悪化** /
> C=0.4817 -6.5% 悪化 / D=0.5156 +0.06% 実測誤差範囲）、**BT 既定設定が
> 本掃引で局所最適**として確認。default 値変更は実施せず、
> **候補 (c) line search 強化はクローズ**（status-362 で効果ほぼ全量抽出
> 済み、パラメータチューニングで frac は伸びない）。最終 NR Type 分布
> `D+E:68%, E:26%`（mixed 比率むしろ高い）で **line search では active
> 集合振動を根本抑制できない**ことを確定。次候補は **(e) 接触減衰
> escape hatch（最有力）** / 副次 (d) 接触凍結モード 19 本適用 / (f)
> Phase C-3' s-tracking の 19 本再評価。`22_bt_parameter_sweep_19strand.py`
> は失敗実験の記録として残置、実装本体（`xkep_cae/`、`tests/`、
> `contracts/`）は**無変更**。
>
> **status-368 候補 (d) 接触凍結モード 19 本再評価 — nr_max=30 で +16.6%、
> frac=1.0 未達で候補クローズ**: `chattering_freeze_*` 3 パラメータ × 6
> ケース感度掃引で **Case B `chattering_freeze_nr_max=30`（default 15 の
> 2x）のみ有意改善 frac=0.5642（default 0.3739 比 +50.9%、status-339
> baseline 0.4839 比 +16.6%）**、他 5 ケース効果軽微〜悪化。disabled は
> `D+E:98%` 200 反復ハマり（**freeze mode が D+E ロック回避の支柱**と確定）。
> MCDD 凍結解除条件未達で候補 (d) クローズ、default 変更は実施せず（7 本向け
> 最適化維持、19 本 opt-in escape hatch として運用）。次候補は (f) Phase C-3'
> s-tracking 19 本再評価（症状緩和 4 候補 (c)/(d)/(e) 全クローズで MCDD 本命
> K_c x/z カップリング不整合に復帰）。
>
> **status-369 Case B 19 本 opt-in ガイドライン化 + 候補 (f) Phase C-3'
> 実験計画 策定（documentation status）**: status-368 §6 引継ぎに対応した
> documentation status。(1) `chattering_freeze_nr_max=30` を 19 本以上向けの
> opt-in escape hatch として `StrandBendingOscillationConfig` docstring +
> 本 roadmap §推奨ソルバー構成下の「撚線規模別 opt-in チューニング」表
> （4 項目）に明記。(2) `xkep_cae/mathematics/docs/phase_c3prime_19strand_plan.md`
> 新設（+107 行）で候補 (f) を **Step 3.1 active 境界 FD 診断**（~30 分、
> `14_kc_active_boundary_diagnostic.py` 新設、g=±10^-3 active 境界
> perturbation で status-356 rel_err=2.18e-07 機械精度が境界で何桁悪化するか
> 定量）/ **Step 3.2 新項 `KcActiveFlipStiffness` 追加設計**（~2 時間、
> `TermExpansionContract` 6 項目化、Huber 2 階微分相当項を
> `HuberContactForceProcess.tangent()` で評価）に分割 scoping。MCDD 脱法
> パターン 1/4/5/6 回避チェックリスト + gate 基準（`test_helical_3d_hermite`
> rel_err <1e-5 維持 / 新 gate <1e-4 / 19 本 mat_only rel_err mean <0.25 /
> 19 本 frac≥0.8）明記。実装本体（`xkep_cae/`、`tests/`、`contracts/`）は
> **無変更**。
>
> **status-370 Phase C-3' Step 3.1 完了 — active 境界 FD 診断で結果 B 確定**:
> `14_kc_active_boundary_diagnostic.py` 新設（+280 行、3 Block 構成）、
> `test_helical_3d_hermite` で gap_target を deep contact から active 境界まで
> sweep + 強制 flip を加え K_c 解析値 vs FD の rel_err を 20 測定点計測。
> **全 20 点で rel_err が status-356 の機械精度 2.18e-07〜2.20e-07 に張り付き**
> （baseline=2.180e-07 / worst boundary=2.192e-07 / degradation=1.01x +0.00 桁
> / smoothed ゾーン worst=2.201e-07 / 強制 flip (eps=1e-7) worst=2.20e-07）。
> eps=1e-4 の 2.19e-04 は FD truncation (O(eps))、K_c 不整合ではない。diff の
> 99%+ が active×active ブロックに局在（adj ≤0.4%）。**結果 B 確定**で当初計画の
> 新項 `KcActiveFlipStiffness` 追加は不要、19 本 Type D stall は K_c 項欠落では
> なく **NR alg 側動力学**（反復間 active 振動 / pair 間相互作用 / 摩擦活性
> 切替）と確定。`phase_c3prime_19strand_plan.md` §3.2 を候補 (g) 3 サブライン
> 再配分: **(g1) active 履歴平滑化**（最優先、~130 行、`p_n_eff = α·p_n_new +
> (1-α)·p_n_prev`）/ (g3) pair-wise relaxation / (g2) AL 再導入。診断限界:
> 単一 pair / 摩擦なし / 静的（多 pair / 摩擦 / NR 振動未捕捉）。実装本体は
> **無変更**、diagnostic script 追加と plan doc 再配分のみ。
>
> **status-371 候補 (g1) active 履歴 EMA 平滑化 実装**: status-370 §5 最優先
> TODO に対応、`HuberContactForceProcess` に `active_ema_alpha: float = 0.0`
> field を追加 + `_p_n_prev_array` 保有 + `reset_ema_state()` メソッド +
> `evaluate()` 内で `p_n_eff = α·p_n_new + (1-α)·p_n_prev` ブレンド
> （α=0 で履歴ストレージにも書き込まない byte-identical 動作）。
> `NewtonDynamicProcess.process()` の NR ループ突入時に `reset_ema_state()`
> を呼び出してインクリメント境界で履歴をクリア（責務分離: HuberContactForce
> は履歴保有 + blending のみ、NR ソルバーが境界を制御）。4 層 1 field plumb
> -through（`_create_contact_force_strategy` / `default_strategies` /
> `ContactFrictionInputData` / `StrandBendingOscillationConfig`）+ 3 経路（曲げ
> / 揺動 / free_end）。`TestActiveEmaSmoothing` 10 単体テスト + 診断スクリプト
> `work/beam_hysteresis/26_active_ema_alpha_sweep.py`（150 行、`--n-strands
> {7,19}` × `--alphas` で α 掃引と gate 判定）。実機 α 掃引は **status-372
> に分離**（status-365/366/367 と同じ Phase 1+2 構成、各 status は 1 PR 粒度
> を維持）。EMA 平滑化は K_c 自体を変更しないため `TermExpansionContract`
> 5 項分解の整合性に影響なし、C18-C24 全 24 検査は無変更で OK。default
> α=0.0 で `pytest xkep_cae/contact/` 446 → **456 passed**（baseline 全 pass、
> +10 EMA テスト）、`test_helical_3d_hermite` rel_err=2.18e-07 維持。
>
> **status-372 候補 (g1) α 掃引 実機検証 — 7 本部分達成 / 19 本却下**:
> α ∈ {0.0, 0.1, 0.3, 0.5} を **7 本 / 19 本撚線 90° 曲げ**で実測。
> **7 本**: α=0.30/0.50 で frac=1.0 維持 + **cb -61〜-75% 削減**（57→14/22）、
> α=0.50 で elapsed -11%（298→265s）。α=0.10 のみ早期 stall（frac=0.3350、
> 弱平滑化逆効果、status-262 smoothing_delta 非単調性と類似）。
> **19 本**: gate「frac ≥ 0.6」**全ケース未達**で候補 (g1) **却下方向**。
> α=0.50 で frac=0.5133（baseline 0.3739 比 +37.3% / status-339 baseline
> 0.4839 比 +6.1%）の部分改善はあるが elapsed +131%（251→582s）でコスト
> 過大、α=0.10/0.30 は -41%/-47% 退化。**default 変更なし**:
> `StrandBendingOscillationConfig.active_ema_alpha` の default=0.0 を維持、
> `active_ema_alpha=0.5` を 7 本系 cutback 削減 opt-in escape hatch として
> 「撚線規模別 opt-in チューニング」表（5 項目目）に追加。
> `26_active_ema_alpha_sweep.py` docstring に 8 ケース実測結果を埋込。
> 次候補は **(g3) pair-wise relaxation**（status-284 接触凍結を pair
> granularity 拡張、~150 行）→ (g2) AL 再導入。実装本体は**無変更**、
> 456 contact tests + 109 mathematics tests 全 pass、`test_helical_3d_hermite`
> rel_err=2.18e-07 維持。Phase A〜E / status-346〜372 の **23/N 完了**。
>
> **他 TODO（7本ピッチ依存性 / ファイバー梁キャリブレーション / リスタート
> 方式 / 被膜圧縮モデル改善 / 空間ブロック分離）は MCDD 完了まで凍結**。
> 離散化方程式の正規参照は [`docs/math/`](math/README.md) 全 6 章（status-348〜349 整備、
> status-353 で 03 章 §3/§4/§5/§8 訂正、status-354 で §7 仮説 A 反証仲裁追記、
> **status-356 で §7 全面再構成（2 経路解析 + 相殺定理）**、
> `equation_index.py` で C15 機械検証）。

| 到達点 | 概要 |
|--------|------|
| FEM基盤 | 梁（EB/Timo/CR/Cosserat）+ 平面 + HEX8、非線形、動的解析 — 完了 |
| 接触 | NCP + Line contact + Mortar + smooth penalty Coulomb摩擦 — 完了 |
| 撚線 | 7本摩擦曲げ+揺動収束、被膜+シース、ヒステリシス — 完了 |
| 高速化 | NCP 6x + 要素12.6x バッチ化 + 接触アセンブリ12-16x バッチ化（大規模向け） — status-246 |
| NR安定化 | **NR接触チャタリング対策**: 接触力リラクゼーション + 接線スケーリング → n_periods=30 frac=1.0 完走 — status-247 |
| MPC | **DOF消去MPC剛体結合**: T^T K T 変換 + LinearSolveProcess統合 + 端部参照点 — status-253 |
| MPC収束 | **MPC u伝搬修正 + NR内再射影 + 縮退系残差判定**: frac=0.35到達（接線不整合で壁） — status-255 |
| FD診断 | **TangentFDDiagnosticProcess**: MPC+接触の接線方向有効性をFDで検証 — status-256 |
| K_c特定 | **FD診断でK_c不整合を決定的に特定**: 全体系94-100%誤差、MPC変換は原因でない — status-257 |
| K_c再解析 | **K_c不整合は活性集合変化が原因**: consistent_st_tangent=TrueでK_c自体は完全一致(4.4e-10) — status-258 |
| smoothing_delta | **Huber smoothing_deltaパイプライン貫通**: ContactSetupConfig→HuberProcess全経路で設定可能、自動推定(5000/r)有効化 — status-259 |
| δチューニング | **smoothing_delta最適化**: 1000/rに変更でfrac 0.35→0.59（69%改善）、δ=1000手動指定でfrac=1.0完走達成。FD診断に活性DOFフィルタ追加 — status-260 |
| delta_h API | **huber_delta_h直接指定API**: k_penスケール非依存の遷移幅指定。active_contact_dofs NRソルバー結合。δ=1000完走テスト追加 — status-261 |
| delta_h探索 | **delta_h=0.025が最速完走（132s）**: 梁-梁で有効範囲[0.020,0.025]∪{0.040}、非単調性あり。three_point_bend貫通。3Dパイプ貫入なし — status-262 |
| delta_h検討 | **delta_hデフォルト値は0.0維持**: 剛体-梁では直接指定改善なし、問題依存性高くグローバルデフォルト時期尚早 — status-263 |
| E=25回帰修正 | **frozen_hermite_tangent + _cur_ratio統一 + n_elems=8**: E=25 frac=0.0003→0.67回復 — status-264 |
| STA2自動記録 | **BenchmarkRunnerProcess**: 実行マニフェスト自動記録（git+Config+結果）で担当者間再現性を保証 — status-265 |
| frozen_hm安定化 | **frozen_hermite_tangent=False安定化**: tangent()はdm凍結、evaluate()のみdm補正（修正NR法）。E=25 frac=0.0003→0.47 — status-266 |
| chattering分析 | **チャタリング詳細分析 + divergedフラグ修正**: リラクゼーション91/91全失敗の原因特定。abort時diverged=False化でE=25 frac 0.4837→0.4950 — status-267 |
| delta_hブースト | **チャタリング時delta_hブースト + NR反復動的拡張**: ボトルネック確定=frozen_hermite_tangent線形収束率(0.97/iter)。delta_hは深い貫入に無効。frac 0.4950→0.4978 — status-268 |
| NRリストア | **NR残差最小値リストア**: 発散検知時に最小残差状態にロールバック。過修正防止でfrozen=True 0.4978→0.5341、frozen=False 0.4732→0.5408 — status-269 |
| frac1回帰修正 | **n_elems_wire=20復元**: パラメータbisectで主因特定（n_elems 20→8が唯一の原因、use_rigid_surface無影響）。frac進行率9x改善 — status-270 |
| frozen=False検証 | **frozen_hermite_tangent=False + n_elems=20**: frac=1.0, incr=607, cutback=389。frozen=True比**35%高速、43%カットバック減** — status-271 |
| Hermite非局所Step1 | **StJacobian隣接ノード微分**: ds_du_adj/dt_du_adj計算実装。FD検証atol=1e-5合格（2テスト追加） — status-271 |
| Hermite非局所Step2 | **K_st隣接ノードDOF拡張**: adj_node_map計算、ds_du_adj/dt_du_adjをK_stアセンブリに結合。FD検証atol=1e-4合格（2テスト追加） — status-272 |
| Hermite非局所Step3 | **K_c隣接ノードDOF拡張**: K_mat+K_geoにalpha_adjベースの隣接ノード寄与追加。FD検証atol=1e-2合格（2テスト追加） — status-273 |
| 摩擦K_st非局所Step4 | **摩擦K_st隣接ノードDOF拡張**: _assemble_friction_st_stiffnessにHermite非局所寄与追加。ソルバーパイプライン貫通。3テスト追加 — status-274 |
| テスト品質改善 | **非平行座標化 + atol厳格化**: TestKstNonlocalFD/TestKcAdjFDのtrivially passing問題修正。asymmetric atol 1e-2→1e-5 — status-275 |
| NR壁根本原因 | **evaluate/tangent dm不整合を特定**: 複合回帰（NR+接触コード相互作用）、NR min restore OFF + diverged=True + tangent scaling復元 — status-277 |
| チェックポイント再開 | **チェックポイント途中再開パイプライン実装**: load_frac_start + stepping/state初期化。N-サイクル検知/リスタートは逆効果で無効化。ul_frac_base処方変位バグ修正。ベースライン frac=0.5543 維持 — status-279 |
| free_end_mode | **MPC不使用端部直接処方**: 各素線θ_z直接処方。MPC frac=0.55→free_end frac=1.0完走 — status-280 |
| UL参照配置更新 | **接触なし90度曲げ完走**: ContactFrictionProcess UL増分変位+update_reference()。理論値0.02%一致 — status-281 |
| 接触ありベースライン | **接触あり90度曲げfrac=0.40**: active=8-9で2サイクルチャタリング停滞。接触なし比で60%低下 — status-282 |
| MPC T動的再構築 | **MPC接触なし90度曲げ完走**: T行列をUL更新時に変形座標で再構築。frac 0.14→1.0 — status-283 |
| 接触凍結モード | **チャタリング検知→接触凍結（陽解法スイッチ）**: 低残差振動検知+接触力凍結+K_c除外+再評価サイクル。frac 0.40→0.70（75%改善）— status-284 |
| C16修正+Hertz型 | **C16修正（RebuildMPCTransformProcess化）+ 凍結テスト + Hertz型非線形ペナルティ実装**: penalty_exponent=1.5でHertz接触。frac=0.998達成 — status-285 |
| チャタリング内訳分析 | **NR反復レベル詳細診断（Type A/B/C/D/E分類）**: frac>0.10で活性集���振動ゼロ、接線剛性不整合(D=52%)が主因。凍結モードは原理的に無効 — status-287 |
| Type D対策基盤 | **収束診断ログ構造化 + Type D自動検知・FD診断トリガー**: NR進捗にType+rate追加、連続Type D→FD診断自動実行+NR上限拡張、低残差Type D分岐 — status-288 |
| K_c不整合特定 | **Hertz ∂p/∂g正確、K_c幾何項不足を特定**: comp=2(z方向)にFD不整合集中。frozen-m近似（∂m/∂u=0）がz方向DOFカップリング欠落の根本原因 — status-289 |
| smooth遷移帯 | **StJacobian smooth blending + unclamped IFT修正**: 1×1/2×2系のhard threshold→w_t/w_s連続補間。IFT幾何をunclamped座標で評価。frozen-m内部接触点K_st FD誤差3x確認（既知制限） — status-293 |
| frozen-m部分解消 | **dm_A/dm_B有効化 + dm_ext無効化**: K_c FD誤差15.5%→11.0%。K_st_adjとK_c_adjの二重計上を発見・解消。残余11%はz方向高次効果 — status-294 |
| K_c_adj mat-only | **隣接ノード幾何剛性除外**: K_c FD誤差11.0%→1.8%。s追従による法線変化相殺を理論的に解明 — status-295 |
| K_c FD 1.8%分析 | **K_st_adj再有効化→38.5%悪化**: 接平面内で K_c_adj geo と K_st_adj が同一寄与。mat-only(1.8%)が最適解 — status-296 |
| 端部接触除外 | **exclude_end_elements実装**: MPC+contactで端部2要素除外→frac 0.001→0.004（不十分）。T^T K_c Tグローバルカップリングが根本原因 — status-296 |
| frozen-m効果検証 | **Hertz型+frozen-m解消でfrac 0.40→0.9997（事実上完走）**: 541 incr, 41 cutback。正しい接線での完走達成 — status-296 |
| 微小dt耐性 | **dt snap改善 + atol_force**: 端数dt吸収（snap閾値→next_delta基準）、NR絶対許容値（global_f_ref×tol_force）で微小dt収束保証 — status-297 |
| ベースライン検証 | **Hertz型+atol_force frac=1.0完走確認**: incr=535, cutback=45, 752s。status-285比でcutback 25%削減 — status-298 |
| 曲げ+揺動±48mm | **統合モード(prescribed_func)でfrac=1.0完走**: incr=1900, cutback=72, 1504s — status-299 |
| 2D投影可視化 | 変形メッシュ2D投影スクリプト実装（XZ側面+XY端面 4パネル + 時系列スナップショット）— status-300 |
| 性能分析 | **被膜でincr半減**: 被膜なし(incr=1900,cb=72,1527s) → 被膜付き(incr=965,cb=31,555s)。被膜バグ修正(core_radii=None→計算)。高速化フェーズ移行 — status-301 |
| 被膜貫入診断 | **被膜平均54%圧縮、8.6%芯線貫入**: k_coat=1e6線形バネは有限厚を表現できず数値的正則化として機能。バリア関数 or 二層モデル要検討 — status-302 |
| バリア関数被膜 | **バリア関数 f=kδ/(1-δ/δ_max) 実装**: 芯線貫入防止。接線剛性 k/(1-δ/δ_max)²。FD整合・対称性・半正定値性検証済み。11テスト追加 — status-303 |
| FD精度+パラメータ | **FD誤差67%=幾何接線欠落**: バリア材料接線は正確(rtol<1e-5)。k_coat=1e6は物理値の800-4000倍、数値的正則化 — status-304 |
| バリア被膜検証 | **バリア被膜90度曲げ: incr535→308(42%削減), 752s→224s(70%高速化)**: coating_barrierパイプライン貫通。被膜の接触平滑化効果を定量確認 — status-305 |
| 被膜エネルギー診断 | **被膜弾性エネルギー解析積分**: E=k·δ_max²·[-ln(1-r)-r]。エネルギー診断統合。収束テストを推奨構成(free_end+Hertz)に更新。7テスト追加 — status-306 |
| 診断ログ強化 | **`[CUTBACK:原因]`+`[f_ref]`+`[SPIKE]`+`[coat]`+`[収束型統計]`**: カットバック原因即判断、f_ref値表示、残差急増即時出力、被膜圧縮統計、収束型分布。CLAUDE.mdにログ規約追加 — status-307 |
| 収束型統計修正+KD-tree | **収束型統計デッドコード修正**: process.pyのインデント不正で[収束型統計]・エネルギー診断が出力されないバグ修正。**broadphase KD-tree化**: 空間ハッシュ→cKDTree置換。14テスト追加 — status-308 |
| K_stベクトル化 | **K_stアセンブリベクトル化**: バッチStJacobian(線形+Hermite)+einsum COO構築でペアforループ排除。broadphase大規模ベンチマーク(1000本5.6s)。6テスト追加 — status-309 |
| 高速化第2弾完了 | **Hermite dpA/dpBバッチ化 + 摩擦K_stベクトル化 + adj_node_map配列化 + K_st性能測定(69-208x高速化)**: 接触力アセンブリ全体のforループ排除完了。9テスト追加 — status-310 |
| 高速化第3弾+adj | **adj batchバッチ化 + BC適用20,000x高速化 + pypardiso統合**: tolil排除(83s→0.004s)。摩擦K_st隣接ノード完全バッチ化。3テスト追加 — status-311 |
| BC+責務修正 | **BC forループNumPyベクトル化 + MPC forループ排除 + strand_bending_oscillation責務分離違反修正**: _zero_sparse_rows実装。5テスト追加 — status-312 |
| ファイバー梁設計 | **撚線ファイバー梁モデル設計仕様策定**: `xkep_cae/elements/docs/fiber_beam_strand.md` 新規作成。`work/beam_hysteresis/` Stage 01-08（N=150 多層摩擦＋β=0.25 接触劣化＋繊維断面でティアドロップ再現）を正式設計化。Strategy Protocol、状態 dataclass、積分 Process、テスト計画、F1-F6 実装フェーズを明文化。コード実装は後続で段階的に — status-313 |
| プロファイルAPI | **ProcessMetaclass構造化プロファイル統計API強化 + BenchmarkRunnerへのprofile_breakdown自動キャプチャ**: snapshot_profile/get_profile_stats/get_profile_report（sort_by/top_n/since）+ RunManifestへYAML出力。13テスト追加 — status-314 |
| スイープ基盤 | **ParameterSweepBenchmarkProcess 新設 + manifest 連番衝突回避**: 任意 frozen dataclass の 1 フィールドを掃引し、ケースごとの `profile_breakdown` を集約 YAML 化する汎用 BatchProcess。`BenchmarkRunnerProcess._save_manifest` に連番フォールバックを追加し、同一秒内の複数ケースで manifest が上書きされる bug も同時修正。10+1 テスト追加 — status-315 |
| 掃引実測#1 | **n_strands=7/19/37 掃引初回実測完了**: 軽量構成（n_pitches=0.25, contact ON, 被膜 OFF）で 162.74s 完走。LinearSolve 占有率 75%（NR 反復数線形成長が主因）、TangentAssembly/接触剛性が **n² スケール**（n=37/n=7 で 34.6x/94.6x）。1000 本では接触アセンブリが支配的になる示唆 — status-316 |
| 葉プロセス抽出 | **`ParameterSweepBenchmarkProcess.summary_rows` に `dominant_leaf_process` 追加**: wrapper process が占めて真のボトルネックが見えなくなる status-316 の問題に対応。`target_process` の `uses` グラフを再帰走査して `uses=[]` のクラスを葉として先頭から抽出。registry 非依存で `_skip_registry=True` のテストフィクスチャでも機能。`parameter_sweep_benchmark.py` docstring に `case.manifest.results_summary` 参照の stdout サンプルも追記。11 テスト追加 — status-317 |
| 掃引6ケース拡張 | **n_strands=7/19/37/61/91/127 6 ケース掃引完走 + dominant_leaf_process 実測検証**: scipy `spsolve` 環境で 198.32s 完走。**全ケースで dominant_leaf_process=TangentAssemblyProcess** を抽出（status-317 の wrapper 読み飛ばし機能を実証）。avg/call ベース正規化で **TangentAssembly per-call が n=19 以降ほぼ線形〜準線形**を確認。**ただし接触ほぼ未活性化（曲げ角 0.7°）+ gap 自動補正の n_strands 依存バイアスあり** — status-318（status-319 で条件付けて維持） |
| バイアス補正掃引 | **初期 gap 固定 + 大曲率でのバイアス補正掃引**: status-318 の 3 点バイアス（gap 自動補正 n_strands 依存、曲げ角 0.7°（90° の 0.8%）、n_inc=4）を補正（gap=0.07 固定、κ=0.005 → 7.16°、n_inc=10）。n=7, 19, 37 の 3 ケース取得後 n=61 以降は Type D stall で中断。**avg/call scaling 分析（n=19→37）**: ContactForceStStiffness **α≈2.07（n²）**、FrictionStStiffness **α≈2.04（n²）**、TangentAssembly α≈1.65（K_st 混合の super-linear）、ContactForceAssembly α≈0.98（線形）。**status-318 の「TangentAssembly 線形」は接触ほぼ未活性化の条件限定**と判定、1000 本本実測での最大ボトルネックは **ContactForceStStiffness / FrictionStStiffness の n² 成長抑制**（空間ブロック分離 / 距離カット / ML 削減）— status-319 |
| usesグラフ拡張 | **`StrategySlot.default_types` 追加 + `_collect_uses_graph`/`_is_leaf_process` の StrategySlot 対応**: status-319 TODO「`ContactForceStStiffness`/`FrictionStStiffness` 到達可能化」を実装。`StrategySlot` に `default_types: tuple[type, ...]` キーワード引数を追加し、`default_strategies()` 注入の具象 Process 型をクラスレベル宣言可能に。`ParameterSweepBenchmarkProcess._collect_uses_graph()` が MRO 経由で StrategySlot 経由依存を再帰走査。`ContactFrictionProcess` の 4 slot に宣言を入れることで、グラフサイズ **10→30**（HuberContactForce/ContactForceStStiffness/CoulombReturnMapping/FrictionStStiffness/FrictionTangentStiffness/FrictionGeometricStiffness/GeneralizedAlpha/LineToLineGauss/ComputeStJacobian に到達）。**`_is_leaf_process` も静的 `uses=[]` + StrategySlot.default_types 非空なら wrapper 判定**に拡張。5 テスト追加 — status-320 |
| K_st経路最適化 | **K_st アセンブリ CSR/COO 経路最適化（定数項削減フェーズ）**: status-319 TODO「n² 成長抑制」の前段として per-call 定数を削減。(1) K_st / K_mat / K_geo の `tocsr()` を skip し raw COO を返す出力型に緩和、(2) `np.einsum("ni,nj->nij", ...)` を直接ブロードキャスト `a[:,:,None] * b[:,None,:]` に置換、(3) K_st COO 構築の mask filter を skip、(4) `CoulombReturnMappingProcess.tangent()` で K_mat + K_geo + K_st を COO concat → 1 回 tocsr() する fast path、(5) ペア抽出ループを total-pair 比例から active-pair 比例に圧縮。**実測（n_active=2000）: FrictionStStiffness 17.84ms → 11.91ms（33% 高速化）**、ContactForceSt 15.48ms → 14.97ms（3%）— status-321 |
| 診断ログ高速化 | **`ProcessExecutionLog._find_caller()` の `sys._getframe()` 化 + `lru_cache` メモ化**: status-321 TODO「ContactForceSt の 3% 止まり分析」の**根本原因特定 + 修正**。cProfile で `ContactForceStStiffnessProcess._process_batch` を計測し、**`_find_caller()` の `inspect.stack()` が per-call ~3.6ms（全体 18%）を占有**していることを発見。`inspect.stack()` は全フレームを `FrameInfo` に materialize するため極めて遅く、さらに `Path(filename).resolve()` が `posix.stat()` を連鎖する。`sys._getframe()` によるフレーム単体走査に置換し、`_find_repo_root` / `_resolve_rel_path` を `functools.lru_cache` 化。**全 `AbstractProcess.process()` 呼び出しに効く**ため、contact 以外の静的ソルバーでも大幅な速度改善（`test_beam_oscillation` で 18 分+ → 63 秒 ≈ 17x 高速化を実測）。併せて ContactForceSt `_process_batch` の抽出/幾何微分/g_shape/df/gdofs をベクトル化。**実測（n_active=2000, 300 iter）: ContactForceSt 16.8ms → 14.4ms（14% 高速化）**、diagnostics overhead 2.53ms → ≈0ms — status-322 |
| beam振動修復 | **beam oscillation 物理テスト修復**: UL `update_reference()` が自由振動の復元力を消失させる問題を `ul_assembler=None` で回避。集中加振→モード形状分布初速度に変更で amplitude_ratio 安定化。5 FAILED → 0 FAILED + 1 SKIPPED（matplotlib）。_find_caller skip list 評価（拡張不要）、distance culling/symbolic factor reuse 調査 — status-323 |
| distance culling | **K_st distance culling 実装**: Huber 遷移幅ベース gap pre-filter（Contact K_st 自動閾値計算 + Friction K_st パイプライン貫通）。gap > delta_h/k_pen のペアを step 1 で除外し、遠方ペアの overhead 削減。8 テスト追加 — status-324 |
| symbolic fact cache | **symbolic factorization reuse**: `_SolverCache` で pypardiso `PyPardisoSolver` インスタンスを保持。スパースパターン不変時は symbolic analysis (phase 11) スキップ。`(shape, indptr)` 比較でパターン検出。scipy fallback は従来通り。12 テスト追加 — status-325 |
| ファイバー梁F1+計測 | **Phase F1（Elastic1D/BilinearKH）+ culling/cache 効果定量計測**: ContactForceStStiffness 96-99% 高速化、scaling α=2.07→1.24 — status-326 |
| ファイバー梁F2 | **Phase F2（MultiLayerFrictionDegrading1D）**: N 層並列摩擦+弾性バックボーン+接触剛性劣化。frozen dataclass C17 準拠。05_smooth_teardrop.py 完全再現 rtol<1%。KH 等価性検証。12 テスト追加 — status-327 |
| ファイバー梁F3 | **Phase F3（CircularFiberSection + FiberSectionIntegratorProcess）**: 円形断面ファイバー離散化（strip/polar）+ 断面積分 Process。FD 接線検証合格（Elastic/BilinearKH/MultiLayerFriction 3 材料）。弾性 EI 誤差 < 1%。25 テスト追加 — status-328 |
| ファイバー梁F4 | **Phase F4（StrandFiberBeamProcess + ULCRFiberBeamAssembler）**: CR Timoshenko 梁にファイバー断面積分を統合。Battini & Pacoste 解析的接線（K_mat+K_geo）。UL マルチ要素アセンブラ配線（checkpoint/rollback 対応）。弾性内力 < 0.2%、接線対角 < 1%、FD 自己整合検証合格。26 テスト追加 — status-329 |
| ファイバー梁F5 | **Phase F5（StrandBendingOscillationProcess use_fiber_beam 統合）**: use_fiber_beam フラグで素線メッシュ→ファイバー梁切替。直線梁メッシュ+断面積分+TL定式化（非線形材料のCR UL f_int=0問題回避）。弾性先端変位0.02%一致、BilinearKH/MultiLayerFriction NR収束合格。10 テスト追加 — status-330 |
| F5散逸検証 | **Phase F5 散逸エネルギー検証（CableDissipationProcess）**: M-κ ヒステリシス追跡 + ループ面積計算。散逸 ∝ κ^1.9（超線形）、n_strands 超線形（EI比駆動）、β=0.10-0.50 でティアドロップ非対称性。checkpoint bugfix（TL mode section state commit）。15 テスト追加 — status-331 |
| 解析モデル | **断面接触点統計モデル（Papailiou 1997 + 分布閾値拡張）**: 単層 W∝κ^1.0 → 分布閾値 W∝κ^(α+1) でκ冪1.85完全再現。n≥7で±10%精度。ピッチ非依存性を解析的に証明。**ただしキャリブレーション先がファイバー梁（近似モデル同士の比較）のため、CR梁接触動解析での直接検証が必要** — status-332 |
| M-κ直接観測基盤 | **CR梁接触動解析でM-κ追跡 + 接触ペアスナップショット**: ContactFrictionProcess に `track_mk` / `track_contact_pairs` 追加、`mk_moment_dofs` 合算で f_int から曲げモーメント取得、`mk_curvature_func` で load_frac→κ 変換。StrandBendingOscillationProcess free-end モード配線（combined/2-phase 両対応）。軽量 ContactPairSnapshotEntry（elem_a/elem_b/p_n/gap/slip_s/slip_t/stick/dissipation）。2本撚線で M-κ 単調増加＋非ゼロM検証合格。**近似モデル循環論法解消の基盤完成。次: 7本撚線ヒステリシスループ＋ピッチ依存性の直接計測** — status-333 |
| 契約整理 | **C16 契約違反 12 件解消**: `cable_dissipation.py` 4 関数 + `strand_cross_section_model.py` 8 関数を `_` prefix で private 化。内部 Process 呼び出し・テスト・`work/beam_hysteresis` スクリプト import を整合。契約違反 12→0、ruff/contracts 全 OK、既存 15 テスト全 pass — status-334 |
| M-κループ検証 | **2本撚線 M-κ ヒステリシスループ直接観測（infra 検証）**: status-333 基盤 + `n_oscillation_cycles=1` 統合モードで load+unload を 6.88 秒で完走、M-κ entries=41、κ 下降 14 回、loop_area=1.17e-2 を観測。CI 時間内で closed-loop と非零散逸を検証。7本撚線は `@pytest.mark.slow` + work/ スクリプトで後続対応 — status-335 |
| 散逸率厳格化 | **M-κ ループ散逸率を load-only 弾性仕事基準に厳格化**: status-335 の外接矩形比 `loop_area/(M_peak·κ_peak)` は弾性スケールの粗近似で物理解釈不能だったため、`_compute_mk_metrics` の `loading_work = Σ max(0,dκ)·M_avg` を分母に採用。2本撚線ケースで `W_load=6.99e-3, W_unload=4.75e-3, loop_area=2.24e-3, dissipation_ratio=0.32` を検証。散逸率上限（<2.0）および `metrics["dissipation_ratio"]` 一致を assert 化 — status-336 |
| 接触ペア後処理 | **ContactPairAnalysisProcess 新設**: `contact_pair_history` から **κ_cr 分布**（初回スリップ曲率）・**各ペア最終散逸**・**活性ペア数時系列** を抽出する PostProcess。`_compute_mk_metrics` の M-κ 集約と責務直交（素線レベル観測）。9 テスト追加（合成履歴 8 + 2本撚線統合 1）。CLAUDE.md「CR梁接触動解析での直接検証（最優先）」のキャリブレーションデータ抽出経路が確立 — status-337 |
| κ_cr 初回実測 | **7本撚線 κ_cr 実測（90°曲げ・281s・frac=1.0）**: `work/beam_hysteresis/09_kcr_measurement_7strand.py` で status-337 Process を初適用。**κ_cr mean=5.80e-3, std=1.74e-3, CV=0.30, min=3.52e-3, max=1.23e-2**。n_unique_pairs=26, n_slipped_pairs=24（92%）、max active=15、total_dissipation=1.71e-7。右裾型（対数正規様）分布でピークは 4.4-5.3e-3 帯。Papailiou 単一 κ_cr 仮定に対し **30% 広がり**を実測 — status-338 |
| 19本スケール試行 | **19本撚線 κ_cr 実測 — frac=0.484 で Type D stall（未完走）**: `work/beam_hysteresis/10_kcr_measurement_19strand.py` で n=19 へスケールアップ、status-319 既知の Type D stall 再現。534s で incr=271, cb=39。ただし 57/59 ペアのデータ取得成功（mean=4.50e-3, CV=0.33、バイモーダル気配）。7本対比で mean 22% 低下（接触早期化）、CV は scale invariant。Type D 対策ガイド（K_c FD 診断 / n_incr=40 / gap_cull 掃引 / 仮説 A: z 成分不整合）を次セッション向けに策定 — status-339 |
| ペア層分類 PostProcess | **`ContactPairLayerClassifierProcess` 新設**: `(elem_a, elem_b)` を `(layer_min, layer_max)` に正規化し、層ペアごとの κ_cr 分布統計（mean/std/min/max）と累積散逸を集約する PostProcess。`StrandMeshResult.strand_layers` を新規追加して `StrandInfoOutput.layer` を外部へ公開、`work/beam_hysteresis/10_kcr_measurement_19strand.py` に層分類出力を統合。status-339 のバイモーダル気配（内層対 vs 外層対）を 19本撚線完走時に内層(1,1)/層跨ぎ(1,2)/外層(2,2) で定量切り分け可能に。8 テスト追加 — status-340 |
| 仮説 C 反証 | **19本撚線 n_incr=40 リトライ — 曲率プロファイル過粗さ仮説を反証**: status-339 推奨アクション 2（`n_increments_per_cycle=20→40`）を `work/beam_hysteresis/11_kcr_19strand_nincr40.py` で実施。**frac=0.4839 → 0.1991 退化**（-59%）、n_incr=116/cb=11/154s で早期 Type D stall。stall 主 comp が z=65% → **x=72-97%** に変化し、成分選択的ではなく広範な K_c 不整合の可能性を示唆。仮説 A（StJacobian z 成分）優先度を 1 に引き上げ、次セッションは **K_c FD 診断取得を最優先** に更新（status-340 層分類器は 11 ペア全て (1,2) を正常抽出し実運用初検証合格） — status-341 |
| K_c FD 診断実測 | **19本撚線 K_c FD 診断 166 レコード取得 — 仮説 A を再定義**: `work/beam_hysteresis/12_kc_fd_diagnostic_19strand.py` で `tangent_fd_diagnostic=True`+`type_d_auto_fd=True`（既定）により Type D stall 中の FD レポートを stdout 捕捉→正規表現パース→CSV 化。frac=0.3743/incr=175/cb=19/530.78s で n=166 レコード取得。**`f_c` FD 相対誤差 mean=115%, median=110%, max=191%** — K_c 自体が 2 倍近く狂う瞬間あり。不整合方向は **`f_c` comp x=68.3% / y=44.2% / z=41.5%（x 支配）**、一方全体系 K@du は z=89.7%/x=40.8%。**status-341 の「z 支配」は beam coupling（曲率軸 y で x 法線 → z 変位結合）の 2 次効果**で、K_c 原因は **x 成分が primary driver**。仮説 A を「StJacobian z 成分」から「K_c の x 成分寄与（mat/geo/st のどこか）」に再定義。次は K_c 分解 FD 診断で mat/geo/st 由来切り分け — status-342 |
| K_c 成分分解 FD 診断 Process | **`ContactKcComponentFDDiagnosticProcess` 新設（status-342 推奨アクション 1）**: `xkep_cae/verify/kc_component_fd.py` に `SolverProcess` ベースの診断 Process を追加。K_c = K_mat - K_geo + K_st の 4 組み合わせ（`full` / `mat_only` / `mat_geo` / `mat_st`）で FD 相対誤差 + 成分別（x/y/z/θ）不整合シェア + 寄与率 `||K_i @ du||/||K_c @ du||` を報告。既存 `TangentFDDiagnosticProcess`（合成接線の単一 rel_err のみ）を補完し、status-342 で特定された x 成分 68% 不整合の由来を部分行列レベルで切り分ける基盤を整備。単体テスト 11 件追加（線形系セルフチェック / mat-only 検出 / st primary driver 分離 / comp shares 範囲検証等）— status-343 |
| 19本 K_c 成分分解 FD 初回実測 — 仮説 A 決着 | **status-343 Process をソルバー配線 + 19本実測（183 レコード）**: `ContactFrictionInputData.kc_component_fd_diagnostic` 追加 + `_newton_dynamic.py` の `tangent_fd_diagnostic` トリガー契機に `ContactKcComponentFDDiagnosticProcess` フックを埋め込み、`work/beam_hysteresis/13_kc_component_fd_19strand.py` で 19本撚線 frac=0.3743 stall 中の FD 診断 183 件を取得。**仮説 A 決着**: 最良組み合わせ = **`mat_only` 100%（183/183）**、`share_geo = 0.000` 全件（K_geo は 19本接触で寄与なし）、K_st 追加で rel_err 平均 +16pp / 最大 +52pp 悪化、mat_only rel_err mean=44% / **comp_x max=98%**（status-341 の x=97% と一致）。7本（1.8%）→ 19本（44%）で 25 倍不整合が拡大し、**K_mat 主導 + K_st 追従**という部分行列構造が確定。次工事は **K_mat の x/z 成分カップリング再検**（status-295 `K_c_adj mat-only` 規模）— status-344 |
| status-344「K_geo=0」誤認の訂正 — report 精度バグ | `ContactKcComponentFDDiagnosticProcess` report の寄与率フォーマット `{:5.2f}` が微小値を 0.00 に丸めていた表示バグを特定。既存 log 再解析で **K_geo share mean=1.02e-3 / max=3.79e-3**（K_mat の 0.1% で非ゼロ）と高精度復元。report を `{:.3e}` に修正 + Output dataclass に `mat/geo/st/full_du_norm` + `dfc_fd_norm` 5 フィールド公開。status-344 推奨アクション 3（K_geo==0 原因調査）は実装バグでなく表示精度問題として**クローズ**。仮説 A 主旨（K_mat 主導）は不動、次は K_mat 修正に集中可能。テスト 1 件追加（11→12） — status-345 |
| **MCDD Phase A-1** | **MathematicalContract 型システム新設**: `xkep_cae/mathematics/` パッケージ新設（`contracts.py` + `docs/mathematics.md` + `tests/`）。5 種の frozen dataclass 契約型（`IdentityContract` / `InequalityContract` / `FDConsistencyContract` / `SymmetryContract` / **`TermExpansionContract`** ★MCDD の核 — `K = Σ K_term_k` の項網羅性を型で宣言）を実装。`providers` 重複検出・長さ一致検証・frozen/severity 必須性で脱法実装 pattern 2/3/9 を型レベルで封じ込め。既存 Process 改変なし、33 テスト追加、契約違反 0 件。計画 `/root/.claude/plans/deep-wiggling-seal.md` の Phase A〜E / status-346〜356 の 1/11 を完了 — status-346 |
| **MCDD Phase A-2** | **`ProcessContractRegistry` + `@verified_by` デコレータ新設**: `xkep_cae/mathematics/registry.py`（469 行）で契約↔Process↔検証 Process の三者紐付けレジストリを実装。`AbstractProcess.contracts: ClassVar[tuple[MathematicalContract, ...]]` + `ProcessMeta.math_contracts` の二経路宣言 + `__init_subclass__` 自動合算。`@verified_by(contract_name, verify_cls)` デコレータで紐付け宣言、**dummy VerifyProcess の AST 検査拒否**（`process()` 本体が `pass`/`...`/`return`/`raise NotImplementedError` のみなら `DummyVerifyProcessError`）で脱法実装 pattern 2 を型レベルで封じ込め。`unverified_contracts` / `all_bindings` で Phase E の C18 静的検査前段 API を提供。C16 滅菌除外に `mathematics/` を追加（`core/registry.py::ProcessRegistry` と構造同等の基盤）。33 テスト追加、契約違反 0 件、既存 skip/xfail 増加 0 — status-347 |
| **MCDD Phase B-1** | **`docs/math/03_huber_contact_penalty.md` 先行整備**（status-348）— Huber 接触ペナルティ系の離散化方程式を Markdown + TeX で台帳化、各式にアンカーを付与し `TermExpansionContract.equation_ref` から参照可能に。8 節 / 19 アンカー |
| **MCDD Phase B-2** | **数理台帳 6 章完備 + `equation_index.py` + C15 拡張**（status-349）— 残り 5 章（01 梁運動学 / 02 接触幾何 / 04 smooth penalty 摩擦 / 05 バリア関数被膜 / 06 Generalized-α 時間積分）を整備し、計 6 章 / 55 アンカー完成。`equation_index.py` で `<a id="...">` 抽出 + 参照解決 API（21 テスト）、C15 拡張で台帳空・アンカー重複・未解決参照を契約違反計上（8 テスト） |
| **MCDD Phase C-1** | **`KcNormal` / `KcGeo` Process 抽出 + `tangent_components()` orchestrator 化**（status-350）— `HuberContactForceProcess.tangent_components()` の K_c 3 項（K_mat / K_geo / K_st）を独立 Process に分離、`TermExpansionContract` `providers` で 1:1 対応。新 Process 14 テスト追加、既存 `test_kc_component_fd.py` 19 件無変更 pass、7本撚線 frac=1.0 回帰合格（82s） |
| **MCDD Phase C-2** | **`KcHermiteNonlocal` / `KcClosestPoint` Process 抽出 + 5 項 TermExpansionContract**（status-351）— K_mat_adj 隣接拡張を `KcNormalStiffnessProcess` から `KcHermiteNonlocalStiffnessProcess` に分離。K_st の「(s,t) 摂動に伴う p_n 追従」成分を `KcClosestPointStiffnessProcess` に分離（`ContactForceStStiffnessProcess._assemble_term_coo(term)` classmethod で共通セットアップ共有）。`term_names` 5 項 `("K_mat_nn", "K_closest", "K_hermite_adj", "K_geo", "K_st")` / providers 5 Process に拡張、orchestrator は後方互換 3-tuple 返却。新 Process 11 テスト追加（14→25）、`test_kc_component_fd.py` 19 件無変更 pass、7本撚線 frac=1.0 回帰合格（47s） |
| **MCDD 数理台帳訂正** | **K_mat,ndir ≡ K_geo 同一性確立 + Phase C-3 再定義**（status-353）— A-A 同側ペア局所導出から `K_geo = -p_n · ∂n̂/∂u` のペア局所形そのもの（$1/d$ は $\hat n = r/d$ 内在項）であることを証明、`KcGeoStiffnessProcess` が法線方向感度を担うことを確立。「`K_mat_ndir` 独立追加」の当初 Phase C-3 計画を撤回、5 項 `TermExpansionContract` で完結。`docs/math/03_huber_contact_penalty.md` §3/§3.1/§4/§5/§8 訂正、`strategy.py` モジュールコメント / `KcNormalStiffness` / `KcGeoStiffness` docstring 訂正。19本 Type D stall の真の候補を `K_hermite_adj` mat-only 近似（`I_nn` 隣接拡張漏れ、status-295 で意図的に除外）に再設定。7本撚線曲げ揺動 frac=1.0000, 10.20s 完走、Hertz 完走 9.96s、`pytest xkep_cae/contact/` 421 passed 5 skipped |
| **MCDD Phase C-3 再定義実験** | **`K_hermite_adj` フル項拡張の仮説 A 反証**（status-354）— `KcHermiteNonlocalStiffnessProcess` に `-w_geo * I_nn` 隣接ノード項を追加する仮説 A を直接実験、gate テスト `test_kc_component_fd.py::test_helical_3d_hermite` rel_err が **1.795% → 38.49%（21倍悪化）** で反証、mat-only（status-295）継続。隣接ノード摂動の `I_nn` 方向は min-distance 射影の s-tracking 経路で補償されるため Process 側追加は FD 乖離拡大を招く。数理台帳 03 章 §7/§3.1/§4/§8 に仲裁追記、`strategy.py` モジュールコメント + `KcHermiteNonlocalStiffnessProcess` docstring に実測結果記録（実装変更なし）。Phase C-3 を **Phase C-3' 再々定義**（hypothesis B/C/D）へ再配分 |
| **MCDD Phase C-3' 仮説 B 診断** | **K_closest 隣接拡張で埋めるべき量を active×adj ブロックに局在化**（status-355）— `work/beam_hysteresis/14_kc_closest_adj_diagnostic.py` 新設、`test_helical_3d_hermite` シナリオで `diff = K_c_analytical - FD_Kc` を 4 ブロック (active/adj×active/adj) に分解。**rel_err 1.795% の 100% が active×adj ブロックに局在**（aa rel_err=2.2e-7、ax ||diff||=98.52、xa/xx=0）。comp_z 77% は adj 列 z (76.11) そのもの。`||FD[ax]||=601.5` vs `||K_c[ax]||=593.4` で K_hermite_adj が一部埋めるも 16.4% 不足、**98.52 が仮説 B で埋めるべき解析量と一致**。実装コスト評価 ~45 行（`_batch_st_jacobian_hermite` 既存 `ds_du_adj` 活用、`adj_node_counts` 追加、`term="closest"` adj 列分岐）、公開 API 非破壊。コード変更なし診断 status |
| **次（MCDD Phase C-3' 実装本体）** | **仮説 B 実装 — `KcClosestPointStiffnessProcess` の隣接ノード拡張**（status-356）— status-355 診断で局在化した active×adj ブロック 98.52 を埋める ~45 行の実装。`ContactForceStStiffnessInput` に `adj_node_counts` 追加、`_process_batch_term` で `dm_ext_A/B` 計算 + `_batch_st_jacobian_hermite` 既存 `ds_du_adj`/`dt_du_adj` 捕捉、`term="closest"` 分岐で adj 列 COO エントリ追加。ゲート条件: `||diff[ax]||<1e-3` + `test_helical_3d_hermite` rel_err < 1e-4、19 本撚線 K_c FD 再計測で `mat_only` rel_err mean=44% の改善、19 本 frac=0.48→1.0 完走を目標。後続: Phase D 診断ディスパッチャ（status-357）→ Phase E C18/C19 契約検査（status-358）|
| **仮説 C (c) 実装 + 実機検証** | **`ContactBacktrackingLineSearchProcess` 新設**（status-362）— `_newton_steps.py` +112 行で接触残差比 / active flip 過剰増加検知に基づく α 半減 backtracking を NR 主ループに追加。`_newton_dynamic.py` のトリガー `att≥2 & n_active≥1 & active_set_changed & _conv_rate>0.85` で mixed 狭義検知、4 層で 9 field plumb-through、default OFF。`TestContactBacktrackingLineSearchProcessAPI` 6 テスト、default OFF regression 163 passed。**実機**: 7本 frac=1.0000 / elapsed +9.9% 回帰なし、19本 frac 0.4839→0.5153（+6.5% 改善）/ elapsed +36.4% で **MCDD 凍結解除条件未達**。併せて `Strand3DContourProcess` + BenchmarkRunner `post_processes` + 6 フィールド 3D 可視化、19 本撚線 3D レンダリング 6 PNG を出力 |
| **仮説 C (c) パラメータ感度掃引（クローズ）** | **4 ケース全却下、BT 既定が局所最適**（status-363）— `work/beam_hysteresis/22_bt_parameter_sweep_19strand.py` 新設（+228 行）で 3 軸 4 ケース（A: rate_threshold=0.70 relaxed / B: active_flip_ratio=0.15 strict / C: mixed_only=False always_on / D: A+B+C combined）を 19 本撚線 90° 曲げで実測。結果: 全ケース frac<1.0 未達（A=0.5153 BT default 同値 / B=0.4701 **-8.8% 悪化** / C=0.4817 -6.5% 悪化 / D=0.5156 +0.06% 実測誤差範囲）。**BT 既定が実測最良点**、default 変更なし、**候補 (c) クローズ**。最終 NR Type 分布 `D+E:68%, E:26%` で line search は active 集合振動を根本抑制できないと確定。次候補: (e) 接触減衰 escape hatch（最有力）/ (d) 接触凍結モード 19 本適用 / (f) Phase C-3' s-tracking の 19 本再評価。実装本体無変更 |
| **Phase E C24（hollow VerifyProcess 封鎖）** | **脱法 pattern 2 裏口対策**（status-364）— `_reject_dummy_process` を通過する「non-trivial だが計算しない」verifier（`return True` / 全引数 constant の Output コンストラクタ / 入力未参照 BinOp）を静的・動的の両面で拒否。`HollowVerifyProcessError(DummyVerifyProcessError)` 新設、`_collect_verifier_body_signals` ヘルパで reads_input（第1引数 Name 参照）+ has_computation（BinOp / Compare / 非定数引数 Call）の AST 2 シグナル必須化。`bind_verifier` に `_reject_hollow_process` を dummy 検査直後配線、`contracts/validate_process_contracts.py::check_c24_verify_has_computation` 追加で **全 24 契約検査** に拡張。mathematics tests 97→109 passed、hollow フィクスチャ 3 種 + TestVerifierBodySignals 4 + TestCheckC24StaticValidator 4 + bind rejection 3 パラメータ化。実装本体（`xkep_cae/contact/`、`xkep_cae/solve/`、`tests/`）無変更、19 本 frac=1.0 本命課題は status-365 候補 (e) 接触減衰で継続 |
| **候補 (e) 接触減衰 escape hatch — Phase 1** | **Process 単体実装 + 12 ユニットテスト**（status-365）— 候補 (e) の Phase 1 インフラ（solver 未配線）。`xkep_cae/contact/damping/` 新設で `ContactNormalDampingProcess`（219 行、線形形状）+ Input/Output + 設計仕様 `docs/contact_damping.md` + 12 ユニットテスト（`@binds_to` 紐付け済み、C3 OK）。接触ペア単位で `f_damp = -c_n v_n n̂`（残差加算向き）と整合接線剛性 `K_damp = c_n · c1 · (g_shape ⊗ g_shape)`（c1 = γ/(β·dt)、常に対称半正定値）、消散率 `E_damp_rate = Σ c_n v_n² ≥ 0`。Generalized-α 時間積分モジュール無変更（c1 は呼び出し側が計算、責務分離）。`StrandBendingOscillationConfig` に `contact_damping_coefficient` + `contact_damping_energy_budget_ratio` 追加（Phase 1 は保有のみ）。テスト検証: c_n=0/空 pairs/INACTIVE ゼロ出力、単一ペア closing/separating/接線方向解析解一致、多ペア重畳、K_damp 対称・半正定値・FD 整合（rel/abs 1e-5）。MCDD 脱法 pattern 6 回避: 骨格ではなく完結成果物。Phase 2（status-366 予定）で `ContactFrictionProcess.damping_slot` 追加 + `_newton_dynamic.py` NR 加算 + `ContactDampingEnergyMonitorProcess` 新設 + 7本撚線 c_n ∈ {1,2,5,10,20}% 掃引 → 19 本 Type D stall 検証。gate: 契約違反 0 件（C3-C24 全 24 検査 OK）/ contact 439 passed 5 skipped / tests/ 314 passed 11 skipped / damping 12 tests |
| **候補 (d) 接触凍結モード 19 本再評価（クローズ）** | **nr_max=30 で +16.6%、frac=1.0 未達で候補クローズ**（status-368）— status-367 引継ぎ 1. に対応。status-284 で 7 本撚線 frac 0.40→0.70 を達成した `chattering_freeze_*` 既定パラメータ（`max_cycles=5 / nr_max=15 / tol_factor=10`）を 19 本 Type D stall 本体に対し 6 ケース感度掃引（default / max_cycles=10 / nr_max=30 / tol_factor=100 / combined / disabled）。(1) **Case B `chattering_freeze_nr_max=30`（default 15 の 2x）のみ有意改善**: frac=0.5642（default 0.3739 比 +50.9%、status-339 baseline 0.4839 比 +16.6%、elapsed 245→863s）。(2) **他 5 ケース効果軽微〜悪化**: `max_cycles=10` / `tol_factor=100` 単独は default とバイト一致（default 設定では nr_max=15 の反復上限で早期 cutback され max_cycles / tol_factor は発動機会なし）、combined は逆相関 -14.4%、**disabled は `D+E:98%` 200 反復ハマり**（freeze mode が D+E ロック回避の支柱と確定）。(3) **MCDD 凍結解除条件（frac=1.0）未達で候補 (d) クローズ**、default 変更は実施せず（7 本向け最適化維持）。(4) **19 本 opt-in escape hatch として `StrandBendingOscillationConfig.chattering_freeze_*` 4 field 公開 + 3 経路 plumb-through** を配備。最終 NR Type 分布 Case B `D+E:56%, E:40%`（default `D+E:69%, E:25%` より 13 ポイント mixed 減、status-362 BT と同パターン）。次候補: **(f) Phase C-3' s-tracking 19 本再評価**（症状緩和 4 候補 (c)/(d)/(e) 全クローズで MCDD 本命 K_c x/z カップリング不整合に復帰）。`work/beam_hysteresis/25_freeze_param_sweep_19strand.py` 新設、実装本体（solver / tests / contracts）は挙動変更なし。gate: 契約違反 0 件（全 24 検査 OK）/ 条例違反 0 件 / contact 446 passed 5 skipped / ruff pass |
| **候補 (e) 接触減衰 escape hatch — validation** | **符号訂正 + 7本採択方向 + 19本却下**（status-367）— status-366 Phase 2 配線の実測 validation。(1) **符号規約バグ訂正**: 初回実測で c_n>0 全ケース frac=0.05 発散、Process 戻り値の物理ドラッグ力 `-c_n v_n g` と NR 残差規約 `R = f_int + f_c - f_ext + M·a + C·v`（C·v 正寄与）の符号不整合が原因。`R_u += f_damp` → `R_u -= f_damp` 訂正、`ContactNormalDampingOutput` docstring に符号規約節追加（unit test は K_provided≈-J_fd 同値性保証として本体変更なし）。(2) **7本撚線採択方向**: c_n=1000 で **frac=1.0000 完走 + elapsed 246→106s (-56.8%) + incr 475→128 (-73%) + cb 53→8 (-85%)** の劇的改善、c_n=10/100 未完走（減衰不足）、c_n=10000 未完走（過剰減衰 frac=0.60）、budget 超過は NR 過渡ピーク由来で final_ratio=0.96 に収束。(3) **19本撚線却下**: c_n=100 で frac=0.4280（baseline 0.4839 比 -11.5%）、c_n=1000 で 0.4697（-2.9%）、MCDD 凍結解除条件（frac=1.0）未達。物理解釈: Type D stall の主因は K_c x/z カップリング不整合（status-344 mat_only rel_err 44%）で局所減衰では解消できず、候補 (c) line search 却下（status-362/363）と同パターン。`contact_damping_coefficient=0.0` default 維持、7本系 opt-in 高速化 escape hatch として運用、実装本体無変更。次候補: (d) 接触凍結モード 19本再評価 / (f) Phase C-3' s-tracking。gate: 契約違反 0 件 / damping 19 tests / contact 446 passed 5 skipped |
| **候補 (e) 接触減衰 escape hatch — Phase 2** | **NR ソルバー配線 + ContactDampingEnergyMonitorProcess**（status-366）— Phase 1 で完成した `ContactNormalDampingProcess` をソルバーに配線。`ContactFrictionProcess.damping_slot`（`StrategySlot(ContactNormalDampingProcess,)`、default OFF）追加、`_newton_dynamic.py` の NR 反復で `effective_residual`/`effective_stiffness` 適用後に `R_u += f_damp` / `K_T += K_damp` を `c_n>0 & manager.pairs & dt_sub>1e-30` 条件下で加算。`ContactFrictionInputData.contact_damping_coefficient` + `contact_damping_energy_budget_ratio` 追加、`StrandBendingOscillationConfig` から 3 経路で plumb through。`DynamicStepOutput.damping_energy_rate` + `SolverResultData.damping_energy_history = tuple[(load_frac, E_damp_cum)]` 公開（dt 乗算で累積）。新 PostProcess `ContactDampingEnergyMonitorProcess`（`contact/damping/monitor.py` 182 行）が `damping_energy_history` + `EnergyHistory` を読み max/final 比と budget 超過件数を report 出力、7 ユニットテスト（`@binds_to` 紐付け）。既定 `c_n=0` で完全無効化され既存動作不変。次（status-367）は validation（7本撚線 c_n ∈ {1,2,5,10,20}% 掃引 + 19 本 Type D stall 検証、MCDD 凍結解除条件 frac=1.0 + E_damp/E_strain < budget）。gate: 契約違反 0 件（C3-C24 全 24 検査 OK）/ contact 446 passed 5 skipped / tests/ 314 passed 11 skipped |

---

## フェーズ依存関係

```
Phase 1-5, C0-C6, S1-S2 ← 完了（status-001〜096）
  ↓
S3 (大規模収束) ← 現在地（status-097〜）
  ↓ + R1 (プロセスアーキテクチャ) ← S3並行
S4 (剛性比較) ← S3並行可
  ↓
S5 (ML導入) → 候補ペア削減でS6の前提条件
  ↓
S6 (1000本6時間) ← ターゲットマイルストーン
  ↓
S7 (GPU)
```

---

## S3: 大規模収束改善 ← 凍結中（R1 Phase 7 完了まで）

**目標**: NCPソルバーで91本撚りの曲げ揺動が収束する。

**完了済み: 53項目**

### アクティブTODO

- [ ] 全テストのmm-ton-MPa移行（~100ファイルの定数変換）
- [ ] 被膜摩擦μ=0.25の収束達成（接触チャタリング対策が必要）
- [ ] 19本→37本のスケールアップ
- [ ] CR梁の摩擦接触不収束の原因調査
- [ ] 37本Layer1+2圧縮の段階的活性化による収束改善確認
- [x] ~~NCPソルバー版S3ベンチマーク（AL法との計算時間比較）~~ — AL完全削除済み（status-167）
- [ ] Cosserat Rodの解析的接線剛性実装
- [x] ~~**UL+GeneralizedAlpha結合修正**: state.u増分/累積管理の明確化~~ — status-215 で修正完了（動的時UL更新スキップ + モーダル質量補正）
- [x] ~~**動的三点曲げ解析解一致**: FFT振動周期5%以内+振幅10%以内~~ — status-217 で達成
- [x] ~~**UnifiedTimeStepProcess統合**: ContactFrictionProcess内のdt_sub二重管理解消~~ — status-217 で統合完了
- [x] ~~**数値粘性の定量評価**: rho_inf 依存性の検証~~ — status-217 でパラメータ感度81.5%確認
- [x] ~~**動的三点曲げ接触収束**: increment カウント修正で frac=1.0 到達（Hermite OFF 202N, Hermite ON 176N）~~ — status-231 で修正完了（旧 frac=0.86/0.98 はカットバックがmax_increments予算を食い潰すバグが原因）
- [x] ~~**n_periods=30 での数百 N 確認**: n_periods=30 frac=1.0 到達、208.6N（status-234）~~
- [x] ~~**摩擦アセンブリの Hermite 完全対応**: use_hermite=True デフォルト化（status-245）~~
- [ ] **freeze_geometry_in_nr の見直し**: NR内 s,t 凍結は K_st と相互排他（status-239）。freeze=True は修正Newton相当で力2次収束不可。freeze=False + K_st + LM が正しい組合せ
- [ ] **Node tangent 計算の局所化**: 現在 `_compute_node_tangents()` が全体メッシュ依存で、大変形時に接線急変→Hermite曲線形状ジャンプ→gap不連続→active接触点激減。隣接要素のみで局所計算に変更
- [ ] **曲面連続関数化の代替手法調査**: Hermite 補間は接線感度が高すぎて大規模モデルで破綻。代替候補: B-spline/NURBS 曲線表現、Subdivision surface、Moving least squares (MLS) 近似、Isogeometric 接触（IGA-C）など。接線ベクトルへの過敏性を回避しつつ C1 連続性を確保する手法を比較検討
- [x] **解析的剛体円柱表面**: ジグ離散セグメント→C∞連続な解析的円柱（status-237）
- [x] **梁メッシュ粗化**: L_elem > wire_diameter で梁サーフェス面連続化（status-237: n_elems 20→4）
- [x] **SDI 排除**: 全候補ペア Huber 評価 + 力ベース dt 制御 + g_off ワイド化（status-233）
- [x] ~~**n_periods=30 で frac=1.0 到達**: SDI 排除後の dt 改善検証（status-234: 1592 incr, 4403s, fc=208.6N）~~
- [x] ~~**adaptive stepping 高速化**: dt_max緩和+growth damping撤廃+接触力閾値緩和~~ — status-236 で **完全リバート**（n_periods=30 で逆効果: frac=0.24 で壁、98%カットバック）。パラメータ調整だけでは NR 収束性問題を解決できない
- [ ] **NR力収束改善**: Hermite K_st のFD不整合はK_c_adj mat-only化で11.0%→1.8%に改善（status-295）。次: 残余1.8%解消（K_st_adj部分有効化）
- [x] ~~**n_periods=30 剛体表面効果検証**: incr 1592→707（55%削減）、cutback 2477→400（84%削減）、frac=1.0 fc=216.96N（status-238）~~

### 既知の問題

- **NCP摩擦接線剛性符号問題**: `d(f_fric)/du = -k_t*g_t⊗g_t`（負定値）で鞍点系が不安定化。smooth penalty（Uzawa凍結、n_uzawa_max=1）で回避中。（status-147, status-221）
- ~~**slow テスト不収束**: NCP 7本90°曲げ Phase1 が不安定（環境依存）。xfail で安定化済み。~~ → status-212 で接触収束テスト全削除。

### 計測済みスケーリングデータ

| 素線数 | DOF | 計算時間 | 対7本比 | 候補ペア |
|---:|---:|---:|---:|---:|
| 7 | 210 | 92s | 1.0x | — |
| 19 | 570 | 239s | 2.6x | — |
| 37 | 1,110 | 501s | 5.4x | — |
| 61 | 1,830 | 903s | 9.8x | — |
| 91 | 2,730 | 1,476s | 16.0x | 66,066 |
| 1000 | 30,000 | — | — | 7,335,879 |

---

## 推奨ソルバー構成

| 項目 | 設定 | 根拠 |
|------|------|------|
| **ソルバー** | `newton_raphson_contact_ncp`（`solver_ncp.py`） | Outer loop 不要 |
| **摩擦** | `contact_mode="smooth_penalty"` | NCP鞍点系は符号問題あり（status-147） |
| **接触離散化** | Line-to-line Gauss 積分 | セグメント間力の連続性 |
| **同素線除外** | `exclude_same_strand=True` | ~80% ペア削減 |
| **k_pen** | 自動推定（beam EI ベース） | 手動設定不要 |
| **線形ソルバー** | DOF閾値自動切替（直接法 / GMRES+ILU） | スケーラビリティ |

> AL法（`solver_hooks.py`）は status-167 で完全削除済み。NCP一本化。

### 撚線規模別 opt-in チューニング

`StrandBendingOscillationConfig` の既定値は 7 本撚線向けに最適化済み。
19 本以上の Type D stall 支配領域（status-339〜/357/368）では以下を明示指定する。

| 項目 | 7 本既定 | 19 本推奨 | 効果 / 根拠 |
|------|---------|----------|-----------|
| `chattering_freeze_nr_max` | **15** | **30** | frac 0.3739 → 0.5642（+50.9%、status-339 baseline 0.4839 比 +16.6%）。NR Type 分布の mixed (D+E) 比率 69% → 56%（status-368 Case B）。代償 elapsed +251%。**frac=1.0 未達のため default 変更せず、opt-in として公開**。|
| `contact_damping_coefficient` | **0.0** | 適用非推奨 | 7 本で c_n=1000 が frac=1.0 完走 + elapsed -56.8% を達成したが 19 本では frac=0.47（baseline -2.9%）で却下（status-367）。局所減衰は K_c x/z カップリング不整合を解消できない。|
| `smoothing_delta` | 自動（2000） | 既定維持 | 7 本では `smoothing_delta=1000`（2x δ_h 拡大）で elapsed -42.5% を達成（status-359）。19 本では frac -23.1% 退化で却下（status-360）。|
| `contact_backtracking_*` | OFF | 部分効果 | 19 本 `frac=0.5153`（+6.5%、status-362）。パラメータ掃引でも frac=1.0 未達（status-363 候補 (c) クローズ）。|
| `active_ema_alpha` | **0.0** | 適用非推奨 | 7 本では `active_ema_alpha=0.5` で frac=1.0 維持 + cb 57→22（**-61%**）+ elapsed -11%（status-372）。**α=0.10 のみ早期 stall**（弱平滑化逆効果）。19 本では gate「frac ≥ 0.6」全ケース未達、α=0.50 で frac=0.5133（+37% 改善）でも elapsed +131% で却下（status-372）。|
| `solver_mode`（status-378 Phase 2 配線完了、19 本 frac=1.0 は mass scaling 待機） | **"implicit"** | **"explicit"**（wiring 完成、ただし Courant 比 3×10⁵ で frac=1.0 完走には mass scaling/subcycling 必須） | 陽的中央差分時間積分への切替。status-378 で `ExplicitDynamicProcess` + `_estimate_critical_dt` が `ContactFrictionProcess` に配線、Courant 監視 + cutback 連携が動作。7 本撚線 smoke test で `dt_c=1.055e-06` vs `dt_sub=0.333` を実測。19 本 frac=1.0 完走には status-379 で mass scaling（β² 倍質量で dt_c を β 倍化）/ dt subcycling 実装予定。設計: `xkep_cae/time_integration/docs/time_integration_explicit.md` + `xkep_cae/contact/solver/docs/explicit_dynamic.md`。|
| `al_outer_enabled` + `al_n_uzawa_max`（status-376） | **False / 2** | **True / 2** | 19 本で **frac=0.5746（baseline 0.3739 比 +53.7%、(g) サブライン全候補で最良 19 本実績）** の改善を確認、ただし gate 0.6 を 0.026 不足で FAIL（status-376）。AL 外側 Uzawa 1 回更新で active 集合変動を解析ステップ単位で平滑化（候補 (g1) EMA α=0.5 と類似機構だがコスト elapsed +72% で (g1) +131% より良）。`n=3` は過修正発散（frac=0.1973、-47.2%）のため n=2 固定推奨。**frac=1.0 未達のため default 変更せず、19 本以上向け opt-in escape hatch として公開**。法線のみ AL 適用（摩擦は status-147 NCP 鞍点系符号問題回避）。数理: `docs/math/03_huber_contact_penalty.md` §9。|

> **MCDD 凍結解除条件未達**: 上記 opt-in は全て症状緩和であり、19 本 Type D
> stall の根本原因 `mat_only rel_err 44%`（K_c x/z カップリング不整合、
> status-344）の解消ではない。MCDD 本命は候補 (f) Phase C-3' s-tracking の
> 19 本再評価（`docs/mcdd/phase_c3prime_19strand_plan.md` 参照）。

---

## 後続フェーズ

### R1: プロセスアーキテクチャリファクタリング（完了）

AbstractProcess + Strategy分解によるソルバー契約化。status-150〜174 で完了。

### 脱出ポット計画（Phase 1〜16 完了）

新 xkep_cae を Process Architecture でゼロ構築。旧パッケージからの完全移行。

| Phase | status | 概要 |
|-------|--------|------|
| 1 | 175 | xkep_cae → __xkep_cae_deprecated リネーム + PenaltyStrategy |
| 2 | 178-181 | Strategy 全移行（Friction/ContactForce/Geometry/TimeIntegration/Coating）|
| 3-4 | 183-184 | concrete プロセス + ContactFrictionProcess + BatchProcess |
| 5-6 | 185-186 | ソルバー結果連携 + C14 強化 |
| 7-8 | 187-189 | deprecated 依存完全除去 + friction/geometry 実装完成 |
| 9 | 190-192 | solver Process 化 + NUzawa 分離 |
| 10-11 | 193-194 | deprecated テスト無効化 + __deprecated リネーム |
| 12-13 | 195-197 | numerical_tests + ビームアセンブラ移植 |
| 14 | 198 | S3 xfail テスト Process API 版 |
| 15-16 | 200-206 | C16/C17 違反ゼロ + frozen dataclass + 旧テスト一掃 |

| 17 | 208 | BackendRegistry 完全廃止 + 被膜モデル物理検証テスト（O2 条例違反0件） |

- **次**: S3 凍結解除（変位制御7本撚線曲げ揺動 Phase2 xfail 解消）

### S4: 撚線構造剛性比較

被膜/シース付き撚線の等価剛性を計測し、文献値（Costello, Foti）と比較。
- ✅ 素線+被膜/シース等価剛性 20テスト（status-098）
- ❌ フルモデル + 文献値比較

### S5: ML導入

接触候補削減と k_pen 推定の自動化。GNN/PINN サロゲート PoC 完了（R²=0.995）。

### S6: 1000本撚線

1000本撚線の曲げ揺動計算を6時間以内。メッシュ生成・broadphaseは実装済み。

### S7: GPU対応

S6のボトルネックに応じたGPU化（CuPy/JAX）。

---

## 完了済みフェーズ

| Phase | 内容 | テスト数 | status |
|-------|------|---------|--------|
| 1 | アーキテクチャ（Protocol/ABC） | 16 | 001-003 |
| 2 | 空間梁要素（EB/Timo/Cosserat） | ~360 | 004-015 |
| 3 | 幾何学的非線形（NR/弧長/CR/TL/UL） | ~100 | 015-042 |
| 4.1-4.2 | 弾塑性 + ファイバーモデル | ~70 | 021-023 |
| 5 | 動的解析 | ~60 | 026-030 |
| C0-C6 | 梁–梁接触（AL→NCP+Mortar+摩擦） | ~320 | 033-086 |
| 4.7 | 撚線基礎 + シース | ~420 | 052-064 |
| HEX8/I/O | 3D固体 + Abaqusパーサー | ~210 | 031-063, 105-106 |
| 6.0 | GNN/PINNサロゲート | ~100 | 066-069 |
| S1-S2 | 同層除外 + CPU並列化基盤 | ~90 | 083-096 |

> 詳細: [status-index](status/status-index.md)

---

## 凍結・将来計画

| Phase | 内容 | 状態 |
|-------|------|------|
| 4.3 | von Mises 3D 塑性 | 凍結（45件テスト済） |
| 4.4-4.6 | ヒステリシス減衰、粘弾性、異方性 | **Phase F3 完了**（status-328: `CircularFiberSection` + `FiberSectionIntegratorProcess` + 25テスト）。設計仕様: [fiber_beam_strand.md](../xkep_cae/elements/docs/fiber_beam_strand.md)（status-313）。次: Phase F4（`StrandFiberBeamProcess`） |
| 6.1-6.3 | NN構成則、PI制約、ハイブリッド | 未実装 |
| 7-8 | モデルレジストリ、FE² | 未実装 |

---

## 設計原則

1. **モジュール合成可能性**: 要素・構成則・ソルバー・積分スキーマを自由に組み合わせ
2. **Protocol/ABCベース**: インタフェース依存
3. **テスト駆動**: 解析解・リファレンスソルバーとの比較必須
4. **段階的拡張**: 後方互換性保持

---

## 参考文献

- Crisfield, M.A. "Non-linear Finite Element Analysis of Solids and Structures" Vol. 1 & 2
- Bathe, K.J. "Finite Element Procedures"
- de Souza Neto et al. "Computational Methods for Plasticity"
- Simo, J.C. & Hughes, T.J.R. "Computational Inelasticity"
- Costello, G.A. "Theory of Wire Rope"
- Foti, F. & Martinelli, L. (2016) "Hysteretic bending of spiral strands"
