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

**459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25 テスト** — 2026-04-21 | 契約違反 **0件** | 条例違反 **0件** | **MCDD status-358（Phase E C20 追加 + 仮説 C 候補 (a) 7本撚線 90° 反証） — status-357 の最優先 TODO（仮説 C 立案 + Phase E 仕上げ）に対応。(1) 仮説 C 候補 (a)（`smoothing_delta` 遷移帯 4x 拡大、default 2000→500）を 7本撚線 90° 曲げで実測。ベースライン frac=1.0000, incr=524, cb=57, 452.02s に対し候補 (a) は frac=0.9241 で未完走、cutback -14%/elapsed -17% の見かけ改善は解析の早期打切りで対策効果ではない。ユーザー指示「10% 以上改善 + frac=1.0 完走」未達で却下（revert）、コード変更なし、`work/beam_hysteresis/15_hypothesis_c_7strand.py` は失敗実験の記録として残置。(2) Phase E C20 追加: `TermExpansionContract` 双方向紐付け検査（providers ↔ contracts 同名契約宣言）を `validate_process_contracts.py` に追加、C18/C19 の片側更新による脱法すり抜けを防御、5 既存 providers で回帰なし**

### ターゲット

> **1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

### 次の課題

**接触あり90度曲げ frac=1.0完走** — status-298（ベースライン: incr=535, cutback=45, 752s）:
- ~~接触凍結モード（status-284）で frac=0.40→0.70（75%改善）~~ ← status-284で完了
- ~~Hertz型非線形ペナルティ（`p_n ∝ δ^{1.5}`）~~ ← status-285で完了（frac=0.70→0.998、事実上完走）
- ~~チャタリング内訳分析~~ ← status-287で完了（**活性集合振動ではなく接線剛性���整合(Type D=52%)が主因**）
- ~~収束診断ログ構造化 + Type D自動検知基盤~~ ← status-288で完了（NR進捗にType+rate、FD自動トリガー、Type D時NR拡張）
- ~~FD接線診断でHertz型∂p/∂g整合性検証~~ ← status-289で完了（**Hertz導関数は正確、K_c幾何項のcomp=2(z方向)不整合がType Dの根本原因**）
- ~~K_c不整合の根本原因特定~~ ← status-291で完了（**K_st過大の原因はs_unclamped未伝搬。Hermite 20%→0.0001%改善**）
- ~~frozen-m部分解消（dm_A/dm_B有効化 + dm_ext無効化）~~ ← status-294で完了（K_c FD誤差15.5%→11.0%）
- ~~K_c_adj mat-only化（z方向DOFカップリング追加）~~ ← status-295で完了（K_c FD誤差11.0%→1.8%）
- ~~K_c FD残余1.8%分析~~ ← status-296で完了（**K_st_adj有効化→38.5%悪化。mat-only(1.8%)が最適解**）
- ~~端部接触除外(exclude_end_elements)実装~~ ← status-296で完了（MPC+contactでfrac 0.001→0.004、不十分）
- ~~frozen-m効果検証~~ ← status-296で完了（**Hertz型+frozen-mでfrac 0.40→0.9997、事実上完走！**）
- ~~微小dt耐性改善~~ ← status-297で完了（**dt snap改善 + atol_force絶対許容値で微小dt収束保証**）
- ~~Hertz型+atol_force frac=1.0完走確認~~ ← status-298で完了（**frac=1.0000, incr=535, cutback=45, 752s**）
- ~~90度曲げ+先端横変位±48mm揺動~~ ← status-299で完了（**統合モード frac=1.0000, incr=1900, cutback=72, 1504s**）
- ~~cutback数削減（72→30以下）~~ ← status-301で完了（**被膜付きでincr半減: 1900→965, cb 72→31, 1527s→555s。被膜バグ修正(core_radii計算)**）
- ~~被膜貫入量診断~~ ← status-302で完了（**平均54%圧縮、8.6%芯線貫入。k_coat=1e6線形バネは数値的正則化として機能、物理的被膜モデルではない**）
- **次**: 被膜圧縮モデル改善 — バリア関数(`f = k*δ/(1-δ/δ_max)`) or 二層モデル(ソフト被膜+ハードコア)で物理的に正確な被膜力。シース-素線接触統合（旧SheathModel/HEX8のProcess化）
- ~~接触ペア検出KD-tree化~~ ← status-308で完了（空間ハッシュ→cKDTree置換）
- ~~K_stアセンブリベクトル化~~ ← status-309で完了（バッチStJacobian+einsum COO構築でペアforループ排除）
- ~~摩擦K_stベクトル化 + Hermite dpA/dpBバッチ化~~ ← status-310で完了（K_st性能69-208x高速化確認）
- ~~adj batchバッチ化 + BC適用20,000x高速化 + pypardiso統合~~ ← status-311で完了
- ~~BC forループNumPyベクトル化 + MPC forループ排除 + 責務分離違反修正~~ ← status-312で完了
- ~~撚線ファイバー梁モデル 設計仕様策定（work/beam_hysteresis 統合）~~ ← status-313で完了（`xkep_cae/elements/docs/fiber_beam_strand.md` 新規作成、Phase F1-F6 計画）
- ~~プロファイル統計API強化 + BenchmarkRunnerプロファイル自動キャプチャ~~ ← status-314で完了（snapshot_profile/get_profile_stats/profile_breakdown YAML統合）
- ~~ParameterSweepBenchmarkProcess 新設 + manifest 連番衝突回避~~ ← status-315で完了（汎用 1 フィールド掃引 BatchProcess、`BenchmarkRunnerProcess._save_manifest` の同一秒衝突バグも同時修正）
- ~~n_strands=7/19/37 掃引初回実測（dominant Process 推移データ取得）~~ ← status-316で完了（軽量構成 162.74s 完走、**LinearSolve 75% 占有だが avg/call ほぼ定数**、**TangentAssembly/接触剛性が n² 成長（n=37/n=7 で 34.6x/94.6x）→1000本ではアセンブリ支配の示唆**）
- ~~`ParameterSweepBenchmarkProcess.dominant_leaf_process` 追加~~ ← status-317で完了（`uses` グラフ再帰走査で wrapper 占有を読み飛ばして真のボトルネック抽出、registry 非依存、11 テスト）
- ~~n_strands=7/19/37/61/91/127 6 ケース掃引拡張 + dominant_leaf_process 実測検証~~ ← status-318で完了（**全ケースで dominant_leaf=TangentAssemblyProcess** を抽出、avg/call ベースで n=19 以降**線形〜準線形スケール**を確認、198.32s 完走、scipy spsolve 環境）
- ~~status-318 の 3 点バイアス補正掃引（gap 自動補正 / 曲げ角 0.7° / n_inc=4）~~ ← status-319で完了（gap=0.07 固定、κ=0.005 → 7.16°、n_inc=10。**n=7/19/37 取得後 n=61 以降は Type D stall で中断**。scaling 分析 n=19→37: **ContactForceStStiffness α≈2.07（n²）、FrictionStStiffness α≈2.04（n²）、TangentAssembly α≈1.65（K_st 混合）、ContactForceAssembly α≈0.98（線形）**。status-318 の「TangentAssembly 線形」は**小曲率・接触未活性化の狭義結論**と判定）
- ~~`uses` グラフ拡張（`StrategySlot.default_types`）— `ContactFrictionProcess` から `ContactForceStStiffness/FrictionStStiffness` 等 8 Process をクラスレベル到達可能化~~ ← status-320で完了（`_is_leaf_process` も StrategySlot 併合判定に拡張、5 テスト追加）
- ~~K_st アセンブリ CSR/COO 経路最適化 — FrictionStStiffness per-call 17.84ms→11.91ms 33% 高速化~~ ← status-321で完了（tocsr skip + einsum→broadcasting + mask filter skip + 抽出ループ active 比例化）
- ~~`ProcessExecutionLog._find_caller` 高速化（status-321 の ContactForceSt 3% 止まり分析）— 全 Process 呼び出しの ~2.5ms 固定オーバーヘッド eliminate、ContactForceSt 16.8ms→14.4ms 14% 高速化~~ ← status-322で完了（`sys._getframe()` + `lru_cache` 化、ContactForceSt のローカルベクトル化併用）
- ~~K_st distance culling（Huber遷移幅ベース gap pre-filter + Friction パイプライン貫通）~~ ← status-324で完了（ContactForceStStiffnessProcess gap 自動閾値計算 + FrictionStStiffnessInput gap_cull_threshold + TangentAssemblyProcess→friction tangent パイプライン + 8テスト）
- ~~symbolic factorization reuse（pypardiso analyze() キャッシュ）~~ ← status-325で完了（`_SolverCache` クラス新設、`LinearSolveProcess` v1.2.0、パターン検出+factorize reuse、12テスト追加）
- ~~n=37 掃引で culling + cache 効果定量計測~~ ← status-326で完了（**ContactForceStStiffness 96-99% 高速化、scaling α=2.07→1.24**）
- ~~ファイバー梁 Phase F1 着手~~ ← status-326で完了（`Elastic1D` + `BilinearKinematicHardening1D` + `Fiber1DMaterialStrategy` Protocol + 12テスト）
- ~~ファイバー梁 Phase F2（MultiLayerFrictionDegrading1D）~~ ← status-327で完了
- ~~ファイバー梁 Phase F3（CircularFiberSection + FiberSectionIntegratorProcess）~~ ← status-328で完了（FD接線検証合格、弾性EI誤差<1%、25テスト追加）
- ~~ファイバー梁 Phase F4（StrandFiberBeamProcess + _beam_assembler 配線）~~ ← status-329で完了（CR Timoshenko 梁ファイバー積分統合 + ULCRFiberBeamAssembler 配線、弾性内力<0.2%・接線対角<1%・FD自己整合検証合格、26テスト追加）
- ~~ファイバー梁 Phase F5（StrandBendingOscillationProcess use_fiber_beam 統合）~~ ← status-330で完了（弾性先端変位0.02%・BilinearKH/MultiLayerFriction NR収束合格、TL定式化でf_int=0問題回避、10テスト追加）
- ~~Phase F5 散逸エネルギー検証（CableDissipationProcess）~~ ← status-331で完了（M-κヒステリシス追跡、散逸∝κ^1.9、撚線本数超線形、checkpoint bugfix、15テスト追加）
- ~~断面接触点統計モデル（Papailiou解析 + 分布閾値拡張）~~ ← status-332で完了（κ冪1.85完全再現、n≥7で±10%精度、ピッチ非依存性証明、曲げ+捻り複合モード閉形式）
  - **反省**: 近似モデル同士の比較は循環論法。CR梁接触動解析で直接検証すべき
- ~~CR梁接触動解析でM-κヒステリシス直接取得（M-κ追跡 + 接触ペアスナップショット基盤）~~ ← status-333で完了（2本撚線 infra 検証、ContactPairSnapshotEntry 軽量フォーマット）
- ~~2本撚線 M-κ ヒステリシスループ観測 + loop_area / W_load=0.32 厳格化~~ ← status-335/336で完了
- ~~ContactPairAnalysisProcess 新設（κ_cr 分布・各ペア散逸・活性ペア時系列 PostProcess）~~ ← status-337で完了（9 テスト追加）
- ~~7本撚線 κ_cr 初回実測~~ ← status-338で完了（**κ_cr mean=5.80e-3, CV=0.30, n_slipped=24/26, 90°曲げ frac=1.0, 281s**。右裾型分布、Papailiou 単一κ_cr 仮定に対し 30% 広がり）
- ~~19本撚線 κ_cr 実測試行~~ ← status-339で**部分成果**（frac=0.484 で Type D stall、ただし 57/59 ペア取得: mean=4.50e-3, CV=0.33、バイモーダル気配、7本対比で mean 22% 低下・CV scale invariant）
- ~~ペアインデックス→層分類ヘルパー（バイモーダル仮説検証基盤）~~ ← status-340で完了（`ContactPairLayerClassifierProcess` 新設、`StrandMeshResult.strand_layers` 公開、19本実測スクリプトに層別 κ_cr 出力統合、8 テスト追加）
- ~~n_incr=40 リトライで仮説 C 検証~~ ← status-341で**反証**（frac=0.4839→**0.1991 退化**、-59%、154s で早期 stall）。stall 主 comp が z=65% → **x=72-97%** に変化、成分選択的ではなく広範な K_c 不整合示唆。**仮説 A 優先度 4→1 に更新**。層分類器は 11 ペア全て (1,2) を正常抽出し実運用初検証合格
- ~~19本撚線 K_c FD 診断取得~~ ← status-342で完了（166 レコード、`f_c` FD rel_err mean=115%/max=191%、**x 成分支配（68.3%）**、仮説 A を「x 成分 primary driver」に再定義）
- ~~K_c 成分分解 FD 診断 Process 新設~~ ← status-343で完了（`ContactKcComponentFDDiagnosticProcess`、4 組み合わせ FD 相対誤差 + 成分別不整合、11 テスト）
- ~~19本撚線 K_c 成分分解 FD 初回実測~~ ← status-344で完了（183 レコード、**最良=mat_only 100%、share_geo=0.000 全件、mat_only rel_err mean=44% / comp_x max=98%**、K_st 追加で +16pp 悪化、**K_mat 主導 + K_st 追従構造確定**）
- ~~status-344「K_geo=0」誤認の訂正~~ ← status-345で完了（report `{:5.2f}` 精度バグを特定、既存 log 再解析で **share_geo mean=1.02e-3 / max=3.79e-3**（K_mat の 0.1% で非ゼロ）と復元、report を `{:.3e}` 化 + Output に `*_du_norm` 5 フィールド追加、**推奨アクション 3（K_geo==0 調査）はクローズ**）
- **次**: **K_mat x/z 成分カップリング修正 → frac=1.0 完走** — status-344 決着を受け、(1) **`ContactForceStrategy.tangent_components()` の K_mat 構築経路で `∂(p_n·n̂)/∂u` x/z 成分を再展開（status-295 `K_c_adj mat-only` 規模、本命工事）**、(2) ~~K_geo=0 原因調査~~（status-345 でクローズ）、(3) gap_cull_threshold 手動掃引（低コスト）
- **次**: **7本撚線ピッチ依存性検証** — p=50/100/200 で κ_cr 分布・mean・CV の変化を実測（Papailiou 非依存予測の CR梁実測検証、完走確実な 7本で実施）
- **次**: リスタート解析方式への移行 — 動的摩擦接触ソルバーが `(u, v, a, 接触ペア)` を初期条件として受け取り `(u, v, a, 接触ペア)` を返すI/Oに整理。曲げ・揺動は境界条件を渡すだけの薄いラッパーとし、解析ステップ単位でのリスタートを可能にする（CR梁ULのf_int=0問題の根本解決: update_referenceを跨がない設計）

**NR収束改善（活性集合変化対策）** — status-264:
- ~~MPC u伝搬修正 + NR内再射影 + 拡張系ラッパー~~ ← status-254で完了
- ~~MPC縮退系残差判定 + u_pred MPC射影 + ストール検知拡張~~ ← status-255で完了
- ~~B1-B4 摩擦アセンブリProcess化~~ ← status-256で完了
- ~~TangentFDDiagnosticProcess実装~~ ← status-256で完了
- ~~FD診断compute_residual実装 + 不整合箇所特定~~ ← status-257で完了
- ~~K_c不整合再解析~~ ← status-258で完了（K_c自体は正確、94-100%不整合は活性集合変化が原因）
- ~~consistent_st_tangent=TrueデフォルトON~~ ← status-258で完了
- ~~Huber smoothing_deltaパイプライン貫通~~ ← status-259で完了
- ~~smoothing_deltaチューニング + FD診断活性DOFフィルタ~~ ← status-260で完了（δ=1000/rで frac 0.35→0.59改善）
- ~~δ=1000完走テスト + active_contact_dofs NR結合 + delta_h直接指定API~~ ← status-261で完了
- ~~delta_h最適値の問題非依存探索~~ ← status-262で完了（delta_h=0.025最速、非単調性あり）
- ~~delta_hデフォルト値検討（three_point_bend検証）~~ ← status-263で完了（0.0維持、問題依存性高くグローバルデフォルト時期尚早）
- ~~E=25回帰修正（frozen_hermite_tangent + _cur_ratio統一 + n_elems=8）~~ ← status-264で完了（frac=0.0003→0.67）
- ~~frozen_hermite_tangent=False安定化（修正NR法: evaluate()のみdm補正）~~ ← status-266で完了（frac=0.0003→0.47）
- ~~チャタリング分析 + リラクゼーション diverged フラグ修正~~ ← status-267で完了（frac=0.4837→0.4950）
- ~~チャタリング対策 delta_hブースト + NR反復動的拡張~~ ← status-268で完了（frac=0.4950→0.4978、**ボトルネック確定: frozen tangent線形収束率0.97/iter**）
- ~~NR残差最小値リストア（過修正防止）~~ ← status-269で完了（frozen=True 0.4978→0.5341、frozen=False 0.4732→0.5408）
- ~~E=25 frac=1.0回帰修正（n_elems_wire=20復元）~~ ← status-270で完了（n_elems 8→20が唯一の原因、frac進行率9x改善）
- ~~frozen=False + n_elems=20検証~~ ← status-271で完了（frac=1.0, incr=607, cutback=389。frozen=True比35%高速）
- ~~Hermite非局所∂g/∂u Step1（StJacobian隣接ノード微分）~~ ← status-271で完了（FD検証atol=1e-5合格）
- ~~Hermite非局所∂g/∂u Step2（K_st隣接ノードDOF拡張）~~ ← status-272で完了（FD検証atol=1e-4合格）
- ~~Hermite非局所∂g/∂u Step3（K_c拡張）~~ ← status-273で完了（K_mat+K_geo隣接ノードDOF拡張+FD検証）
- ~~摩擦K_st隣接ノード拡張（Step4）~~ ← status-274で完了（_assemble_friction_st_stiffness + ソルバーパイプライン貫通）
- ~~frozen_hermite_tangent=True回帰修正~~ ← status-275で完了（デフォルトFalse化、frac 0.38→0.41）
- ~~NR壁根本原因特定~~ ← status-277で完了（evaluate/tangent dm不整合 + NR制御複合回帰）
- ~~ContactFrictionProcess UL参照配置更新~~ ← status-281で完了（動的ソルバーで7本90度曲げ frac=0.065→1.0）
- ~~チャタリング検知→接触凍結モード~~ ← status-284で完了（frac 0.40→0.70、75%改善）
- **次**: frac=0.70→1.0（Hertz型非線形ペナルティ or 凍結パラメータ最適化）— status-284 参照

詳細は `docs/roadmap.md` および `docs/status/status-index.md` を参照。

## フォーカスガード（AI セッション向け）

**以下を厳守すること。違反は作業のやり直しになる。**

## やるべきこと

### ★最優先: MCDD（数理契約駆動開発）Phase A〜E（status-346〜358、status-354 で 1 status 後ろ倒し）

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
- **status-359（次セッション・仮説 C 続行 + Phase E 仕上げ）**: (1) **仮説 C 候補 (a') `smoothing_delta=1000` 7本撚線 90° 再試行** — 候補 (a) の δ_h 4x 拡大は厳し過ぎた。2x 拡大中間値（default 2000 の半分）なら精度と安定性のバランスで合否基準達成の可能性。`15_hypothesis_c_7strand.py` の `smoothing_delta=500.0` を `1000.0` に書き換え同 script を再実行、10% 未達なら revert。(2) **仮説 C 候補 (c) line search 強化**（(a') 効果薄の場合）: NR 反復途中の過剰 active flip を backtracking line search で rejection、`_newton_dynamic.py` に line search hook 追加。(3) **Phase E 仕上げ** — C21 以降の候補（`TermExpansionContract.term_names` / `providers` 重複検出、`contracts` ClassVar 同名契約重複検出、`@verified_by` VerifyProcess 側 SolverProcess 継承必須）

**凍結中の TODO**（MCDD 完了まで再開禁止）:

- ~~19本 Type D stall の K_mat x/z 単発対応~~ → Phase C で解消
- ~~7本撚線ピッチ依存性検証（p=50/100/200）~~
- ~~ファイバー梁キャリブレーション~~
- ~~リスタート解析方式~~
- ~~被膜圧縮モデル改善（バリア関数 / 二層モデル）~~
- ~~空間ブロック分離 / ペアクラスタリング~~

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
