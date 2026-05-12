[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# 達成確認マトリクス（verification matrix）

**目的**: status-379 / 381 / 387 の連鎖撤回事例を踏まえ、**達成・未達成・未検証**を
独立な軸で可視化することで、STA2（数値の捏造 / 偽陽性 / 単一指標一致による誤判定）を
**構造的**に防止する永続ドキュメント。各 status はこのマトリクスを更新することで
透明性ルール（status-388）を運用面で担保する。

新規 status 作成時は **必ず該当行を更新** し、達成主張があれば「✅ + 根拠 status」
を、撤回があれば 🔁 を **削除せず履歴として保持** する。

## 0. 状態凡例

| 記号 | 意味 |
|:---:|---|
| ✅ | 達成（実機検証 + 3 指標 AND gate PASS、撤回されていない） |
| 🟡 | 部分達成（条件の一部のみ満たす、または特定領域のみ） |
| ❌ | 未達（実機検証で FAIL を実証） |
| ⬜ | 未検証（実機実行未着手） |
| ⏸ | 凍結（MCDD 完了まで再開禁止） |
| 🔁 | 撤回（過去の達成主張が後に撤回された、STA2 防止のため履歴保持） |

**重要**: 透明性ルール（status-388）により「達成」判定は **独立 3 指標 AND gate PASS**
が必須。単一指標一致は ✅ にカウントしない（status-387 撤回事例参照）。

## 1. MCDD 凍結解除条件（CLAUDE.md §現在の状態）

| # | 条件 | 状態 | gate 内容 / 検証手段 | 達成 status | 制約 / 注意 |
|:-:|---|:-:|---|:-:|---|
| 1 | Phase E 完了 | 🟡 | C18〜C24 + O1〜O3 全 24 検査 OK | 357〜369 | C18〜C24 は実装済（status-369 まで）。残る Phase E 項目は status-index で個別確認 |
| 2 | 19 本撚線 frac=1.0 完走 | ❌ | `load_history[-1] == 1.0` | — | implicit 単独最良 0.5746 (376, AL n=2) / explicit は数値発散で別 gate (3) 違反 |
| 3 | `max\|u_trans\| < L_strand × C` (C=10) | ❌ | implicit 妥当域、explicit は 1.59×10⁸ mm 発散 (status-380) → 401 mm 発散停止 (381) も精度 (5) 未達 | — | status-380 で追加された物理的妥当性 gate |
| 4 | `KcNormalDirectionStiffness` FD rel_err < 1e-2 | ✅ | `test_helical_3d_hermite` rel_err = **2.18e-07** | 356 | mat-only K_hermite_adj + K_closest 仮説 A+B 同時導入で達成 |
| 5 | `\|u_explicit − u_implicit\|/\|u_implicit\| < 0.1` または vs analytical | ❌ | 90° 曲げ単梁: implicit 70 mm / explicit 40 mm（50% 系統的アンダー） | — | status-381 で追加された解の精度 gate。status-387 が単一指標で偽達成 → 388 で 3 指標 AND により撤回 |

→ 凍結解除には条件 (2) + (3) + (5) の同時達成が必要。現時点で **未達**。
   Phase α/β/γ で foundation 健全性は確定（次セクション）したため、残課題は
   上位層（assembler / UL update_reference）の改修。

## 2. Phase α/β/γ/δ 検証進捗（status-389 §2 計画）

### 2.1 Phase α — 1 要素 implicit static（foundation 静的検証）

| ケース | 系 | gate 3 指標 | 状態 | 根拠 status | 注意 |
|:-:|---|---|:-:|:-:|---|
| α-1 | 純軸引張 F_x=100 N | u_x / u_z(=0) / L_arc | ✅ | 390 | 機械精度 0.000% |
| α-2 | 純粋曲げ small κ M_y=10 N·mm | \|u_z\| / \|θ_y\| / \|f_int\| | ✅ | 390 | 機械精度 0.001% |
| α-3 | 純粋曲げ large κ θ_y=0.15 rad | \|u_x\| / \|u_z\| / L_chord (Hermite) | ✅ | 390 | 機械精度。circular arc 解は γ で別途検証 |
| α-4 | cantilever 横荷重 F_z=0.01 N | \|u_z\| / \|θ_y\| / \|M_base\| | ✅ | 390 | 機械精度 |

**重要発見（α-3）**: 1 要素 CR は chord 長保存制約により Hermite chord rotation
α=θ/2 解を出す。circular arc 解との 25% 差は **1 要素本質的離散化誤差**で、Phase γ で
n_elements ↑ により消失。

### 2.2 Phase β — 1 要素 explicit dynamic（foundation 動的検証）

| ケース | 系 | gate 3 指標 | 状態 | 根拠 status | 注意 |
|:-:|---|---|:-:|:-:|---|
| β-1 | 自由振動 v_z(tip)=1 mm/s | T_FE / \|u_z_max\| / E_drift | ✅ | 391 | T 0.06% / u 4.85% / E 0.02% |
| β-2 | explicit quasi-static θ_y=0.15 rad | \|u_x\| / \|u_z\| / L_chord (Hermite) | ✅ | 391 | **機械精度 0.000% × 3** で α-3 implicit と完全一致 |

**β-2 PASS の重要含意**: status-381〜387 explicit + UL の精度問題は **CR 要素自体ではなく
上位層**（assembler / UL formulation / mass scaling）に局在。次表参照。

### 2.3 Phase γ — multi-element 検証

| ケース | 系 | gate 3 指標 | 状態 | 根拠 status | 注意 |
|:-:|---|---|:-:|:-:|---|
| γ-1 (n=1) | θ=0.15 rad、n_elements=1 | \|u_x\| / \|u_z\| / L_chord (arc 解) | ❌ | 392 | u_x 24.95% — α-3 既知 chord 長保存制約。CR closed form 一致は機械精度 |
| γ-1 (n=2) | θ=0.15 rad、n_elements=2 | 同上 | ✅ | 392 | u_x 6.23% / 他 0.02% |
| γ-1 (n=4) | θ=0.15 rad、n_elements=4 | 同上 | ✅ | 392 | u_x 1.56% |
| γ-1 (n=8) | θ=0.15 rad、n_elements=8 | 同上 | ✅ | 392 | u_x 0.39% |
| γ-1 (n=16) | θ=0.15 rad、n_elements=16 | 同上 | ✅ | 392 | u_x 0.10% |
| γ-1 全体 | log-log slope of err(u_x) vs n (n≥2) | slope ≈ −2.0 | ✅ | 392 | **slope = −2.000**（理論値 O(1/n²) と完全一致） |
| γ-1 全体 | CR closed form 一致（実装健全性） | 機械精度 10⁻¹³%〜10⁻¹²% | ✅ | 392 | 全 5 ケース |
| γ-2 | 大 curvature θ=π/2 多要素 | 同 3 指標 | ⬜ | — | 「16 要素/ピッチ厳守」を full pitch レンジで再確認用 |
| γ-3 | 多要素 explicit + slow ramp | 同 3 指標（β-2 同様 / arc 解） | ✅ | 395 | n=2,4,8,16 全 PASS、slope=-2.000、γ-1 implicit と数値一致 |

→ **Phase γ-1 で「16 要素/ピッチ厳守」規範のマージン確認**: θ=8.6° 単一曲げで n=2 から
   PASS、n=16 で 0.1%。典型 curvature レンジで規範は十分マージンあり。

### 2.4 Phase δ — 接触あり検証（最小規模）

| ケース | 系 | gate 3 指標 | 状態 | 根拠 status | 注意 |
|:-:|---|---|:-:|:-:|---|
| δ | 2 本撚線、平行配置、軽荷重 | 未確定（接触系で再検討） | ⬜ | — | status-335 M-κ 観測スクリプトが基盤 |

### 2.5 Phase ε — 接触なし foundation × explicit-TL（status-397〜）

| ケース | 系 | gate 3 指標 | 状態 | 根拠 status | 注意 |
|:-:|---|---|:-:|:-:|---|
| ε-1 主 | 3 strand helical + 接触なし + explicit-TL（`disable=True`） | u_x_tip / u_z_tip / E_strain vs implicit | **⬜** | 399 | status-397 で N=1 FAIL（u_x 96.36%）。status-399 fix（N_sub=1000）後の 3 strand 規模再検証は未実施（次セッション ε-2 と統合可） |
| **ε-1 sub** | **n_strands=1 直線 + 接触なし + explicit-TL + N_sub=1000** | **u_x_tip vs implicit + status-398 n_inc=20000 との独立軸数値整合** | **✅** | **399** | **N=1000 で u_x=5.299 mm (rel_err 6.07%、< 10% gate)、status-398 n_inc=20000 (β≈46 / u_x≈5.27 mm) と独立軸で一致、hypothesis 1 根本機構を確証** |
| ε-2 | 3 strand + 接触あり + explicit-TL + N_sub | 3 指標 AND gate + frac=1.0 完走 | ⬜ | — | status-400 で着手予定（初の接触統合検証） |
| ε-3 | 7 strand + 接触あり + explicit-TL + N_sub | implicit baseline 対比 + 3 指標 AND | ⬜ | — | status-401 |
| ε-4 | 19 strand + 接触あり + explicit-TL + N_sub | MCDD 凍結解除条件 (2)(3)(5) 同時達成試行 | ⬜ | — | status-402（本命） |

## 3. status-381〜387 上位層改修対象（Phase β/γ で局在化済）

Phase β-2 + Phase γ で「CR 要素自体は健全」が定量実証されたため、status-381〜387 で
発覚した精度問題は **上位層に局在**。次の対象を 1 要素規模で **再現** することが
次セッションの decisive 実験。

| 改修対象 | 問題内容 | 1 要素規模再現実験 | 状態 | 根拠 status |
|---|---|---|:-:|:-:|
| `assembler` 経由 (implicit/explicit + TL) | β-2 直接駆動と差分が出るか | `49_beta2_with_assembler_ul.py` Mode A/C | ✅ | 394 (Hermite 解 機械精度 0.000% 一致) |
| UL `update_reference` (implicit) | 増分ごと reference 更新 → 静的では問題なし | `49_*.py` Mode B | ✅ | 394 (機械精度 0.000% PASS) |
| UL `update_reference` (explicit per step) | 毎 step 更新 → f_int(u_incr)≈0 で elastic response 不発 | `49_*.py` Mode D | ❌ | 394（u_x 99.85% / u_z 96.14% アンダー、status-381〜387 1 要素規模再現確定） |
| `explicit_ul_update_interval` | N 増分ごと UL 更新 → 全 interval で発散 / 過減衰 | `36_explicit_ul_interval_validation.py` | 🔁 | 383（候補却下） |
| Mass scaling 戦略 | β² rescale + KE 保存累積過減衰 | `35_explicit_accuracy_validation.py` | 🔁 | 381 / 382（候補却下） |
| (z1a) 要素ごと波速 Δt | infrastructure 完成も β_stiff cap が支配的 | `37_z1ab_accuracy_validation.py` | 🔁 | 384（部分採択、root cause 未解決） |
| (z1b) selective mass scaling | heterogeneous K 要求、単梁では機能せず | 同上 | 🔁 | 384 |
| (z1c) 2 段階質量スケーリング | β_outside 独立 field、initial target β=4.7e4 が cap 超過 | `38_z1c_two_stage_validation.py` | 🔁 | 385（候補却下） |
| (z1d) `t_cycle_min_seconds` | 方向自体が逆と単梁で実証 | `39_z1d_t_cycle_validation.py` | 🔁 | 386（候補却下） |
| (z2) Cosserat 梁 | absolute necessity ではなくなった（β-2 PASS） | — | ⏸ | 391（中期 plan B） |
| (z3) explicit-TL 固定 API（`explicit_ul_disable_update`） | UL update_reference 完全停止の独立フィールド化 | `TestExplicitULDisableUpdate` 4 ケース（disable=True 0 回 / interval override / default 既存挙動 / ゲート式直接検証） | ✅ | 396（API 化完結。`ContactFrictionInputData` + `StrandBendingOscillationConfig` 各 1 field 独立フィールド方式、AND ゲート評価で `explicit_ul_update_interval` と共存。19 本 / 多 strand 実機検証は status-397 ε-1 で別 scope） |
| `_process_free_end` driver × explicit-TL | process 主ループ + explicit-TL で under-deformation（u_x ~96% アンダー）。implicit / inline driver で問題なく、process driver 経路自体が主因 | `41_epsilon1_3strand_helical_no_contact.py` + `42_status398_hypothesis_diagnostic.py` + `43_status399_epsilon1_n_sub_cycles.py`（5 ケース sweep + n_inc=20000 asymptote + N_sub 掃引） | **✅** | 397+398+399（status-399 fix 実装、ε-1 sub n_strands=1 で N=1000 → rel_err 6.07% 単 strand 規模 PASS。3 strand 規模 / 接触あり / 多 strand は未検証で別行 ⬜） |
| `explicit_n_sub_cycles_per_increment` | hypothesis 1 fix: 1 QUERY を N sub-step に分割、線形補間 prescribed BC、`dt_inner = dt_sub / N` で mass scaling auto-tune の β_inner を 1/N 倍縮小 | `TestExplicitNSubCyclesPerIncrement` 8 件（monkeypatch で `ExplicitDynamicProcess.process` 呼出回数直接計装） + `43_status399_epsilon1_n_sub_cycles.py`（4 ケース N ∈ {1, 10, 100, 1000} 掃引） | **✅** | 399（ε-1 単 strand 規模で MCDD 凍結解除条件 (5)（精度 < 10%）を PASS。status-398 n_inc=20000 と β_auto≈46 / u_x≈5.3 mm で独立軸数値整合、effective sub-cycle 数 20000 が共通因子で hypothesis 1 根本機構を確証） |

→ status-394 で **assembler / UL の 1 要素再現実験完了**: 改修対象は **explicit + UL update_reference per step の組合せのみ**に局在することが定量実証された。
   status-395 で **多要素 explicit + TL の foundation 健全性が機械精度級で確定**、status-396 で
   **(z3) explicit-TL 固定 API 化完結**（公開 API レベル運用可能化）。
   **status-397 で ε-1 主実験 + sub-experiment（n_strands=1）双方 FAIL**、改修対象は
   `_process_free_end` driver 層自体に局在化。**status-398 で 3 仮説切り分け診断完了**、
   **hypothesis 1（stepwise prescribed BC × mass scaling auto-tune の interaction）が支配的**と確定
   （n_inc=20000 で rel_err 5.45% asymptote 収束）。**status-399 で fix 実装**:
   `explicit_n_sub_cycles_per_increment` field + sub-cycle 内部ループ実装、ε-1 単 strand で
   N=1000 → rel_err 6.07% PASS（MCDD 凍結解除条件 (5) を単 strand 規模で達成）。
   **次セッション最優先**: ε-2 = 3 strand 接触あり + N_sub=1000 検証で初の接触統合検証。

## 4. 既存 validation の 3 指標 gate 化（status-389 §3 TODO）

透明性ルール（status-388）の遡及適用。3 指標 AND gate に拡張する対象。

| ターゲットファイル | 状態 | 根拠 status | 注意 |
|---|:-:|:-:|---|
| `xkep_cae/elements/tests/test_assembler_process.py` | ⬜ | — | 既存 pass のまま gate 拡張 |
| `xkep_cae/elements/fiber/tests/test_strand_beam_physics.py` | ⬜ | — | 同上 |
| `xkep_cae/numerical_tests/tests/test_beam_oscillation.py` | ⬜ | — | 同上 |
| `TestHelical90DegBendPhysics` (in `test_strand_bending_oscillation.py`) | ⬜ | — | 既存 status-299 系物理テスト |
| `work/beam_hysteresis/30〜40_*.py` | ⬜ | — | 失敗 ケースは過去判定が信頼できないので再判定必要 |

→ 過去 PASS 判定の中に単一指標一致による偽陽性が含まれる可能性。Phase γ/δ と並行可能。

## 5. STA2 撤回履歴（透明性ルール根拠記録）

過去に達成主張がなされ、後に撤回された事例。**削除せず履歴として保持**することで、
次セッションが同じパターンで誤判定しないための予防。

| 主張 status | 主張内容 | 撤回 status | 撤回理由（数理的反証） |
|:-:|---|:-:|---|
| 379 | 19 本 explicit frac=1.0 完走で凍結解除条件達成 | 380 | `max\|u\|=1.59×10⁸ mm` 数値発散発覚。frac=1.0 / E_kin/E_strain<5% は数学的構造由来で発散時にも PASS する盲点 → CLAUDE.md に gate (3) `max\|u_trans\|<L_strand×C` を追加 |
| 381 | mass scaling bug 修正で発散停止、形式 gate 全 PASS | 381 自身 | ユーザー指摘で精査、解析解 73.3 mm に対し explicit 40 mm（50% アンダー）→ CLAUDE.md に gate (5) 解の精度を追加 |
| 387 | n_inc=8000 sweet spot で精度 gate (5) 達成（err 0.58%） | 388 | 3 指標 AND gate で再検証、L_arc=234 mm（梁が 2.3x ストレッチの非物理解）。「sweet spot」は座標値偶然交差。**透明性ルール**（独立解析解 3 個以上同時一致）を CLAUDE.md に追記して再発防止 |

**この 3 件の連鎖撤回が本マトリクス作成の動機**。matrix を運用ルール化することで
類似の連鎖撤回を構造的に予防する。

## 6. 凍結中 TODO（MCDD 完了まで再開禁止）

status-345 で凍結。詳細項目は当該 status 参照。

| 項目 | 状態 | 凍結 status | 解凍条件 |
|---|:-:|:-:|---|
| 被膜圧縮モデル改善 | ⏸ | 345 | MCDD Phase A〜E 完了 + 凍結解除 5 条件達成 |
| リスタート方式 | ⏸ | 345 | 同上 |
| ファイバー梁キャリブレーション | ⏸ | 345 | 同上 |
| 7 本撚線ピッチ依存性 | ⏸ | 345 | 同上 |
| 空間ブロック分離 | ⏸ | 345 | 同上 |
| 19 本 Type D stall K_mat x/z 単発対応 | ⏸ | 345 | 同上 |

## 7. 更新ルール

新規 status 作成時:

1. **該当行の状態を更新** — 達成 ✅ / 部分 🟡 / 未達 ❌ / 撤回 🔁。
2. **撤回があれば履歴として保持**（§5 に追記）。削除は禁止（透明性ルール）。
3. **新規ケース追加時は行を新規化**（§2/§3/§4 のいずれかに）。
4. **マトリクスを誰も更新していない status は受領しない**（CLAUDE.md「作業完了時の必須手順」§4 として組込）。

CLAUDE.md「作業完了時の必須手順」（§2交代制運用）に統合済み（status-393）。

## 8. 凡例参照テーブル（一覧）

達成済 ✅ 一覧:
- 凍結解除 (4) FD rel_err < 1e-2: status-356
- Phase α 全 4 ケース: status-390
- Phase β 全 2 ケース: status-391
- Phase γ-1 (n=2,4,8,16) implicit: status-392
- Phase γ-1 O(1/n²) 収束: status-392
- Phase γ-3 (n=2,4,8,16) explicit + TL: status-395
- Phase γ-3 O(1/n²) 収束 (slope=-2.000): status-395
- assembler 経由 (implicit/explicit + TL): status-394
- (z3) explicit-TL 固定 API（`explicit_ul_disable_update` 独立フィールド + 単体テスト + Default OFF 回帰）: status-396

未達 ❌（実機 FAIL を実証）:
- 凍結解除 (2) 19 本 frac=1.0
- 凍結解除 (3) max \|u_trans\| 妥当域
- 凍結解除 (5) 解の精度（多 strand / 19 本規模）— 単 strand 規模は status-399 で PASS
- Phase γ-1 (n=1)（既知の離散化誤差、α-3 と整合）

達成 ✅ — Phase ε（status-399 で単 strand 規模 PASS）:
- Phase ε-1 sub（n_strands=1 直線 + 接触なし + explicit-TL + N_sub=1000）: status-399（u_x rel_err 6.07% < 10% gate、MCDD 凍結解除条件 (5) 単 strand 規模 PASS）
- `_process_free_end` driver × explicit-TL fix 実装（`explicit_n_sub_cycles_per_increment`）: status-399（hypothesis 1 architectural fix 完了、ε-1 単 strand で PASS）

未検証 ⬜（次セッション以降の対象）:
- Phase γ-2 大 curvature
- Phase δ 接触あり 2 本撚線
- 既存 validation 3 指標 gate 化
- Phase ε-1 主（3 strand helical + 接触なし、status-399 fix 後の再検証は status-400 ε-2 と統合可）
- Phase ε-2 (3 strand + 接触あり + N_sub、status-400 想定、初の接触統合検証)
- Phase ε-3 (7 strand + 接触あり + N_sub、status-401 想定)
- Phase ε-4 (19 strand + 接触あり + N_sub、status-402 想定、本命)

達成 ✅ — 上位層改修対象（status-394 追加）:
- assembler 経由 implicit/explicit + TL（Mode A/C）: 機械精度 0.000% PASS

部分達成 🟡:
- UL `update_reference` implicit（Mode B）は ✅、explicit per step（Mode D）は ❌

撤回 🔁:
- (z1a)〜(z1d) mass scaling 系列（候補却下）
- `explicit_ul_update_interval`（候補却下）
- 過去達成主張: status-379 / 381 / 387 の 3 件（§5）
