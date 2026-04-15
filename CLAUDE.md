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

**459+13+22+5+8+12+12+25+26+10+15+10+9+8 テスト** — 2026-04-15 | 契約違反 **0件** | 条例違反 **0件**

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
- **次**: **19本撚線 Type D 対策 → frac=1.0 完走** — status-341 で優先順位更新後の推奨順: (1) **K_c FD 診断取得（最優先、7本 vs 19本の成分別不整合スケーリング比較）**、(2) gap_cull_threshold 手動掃引、(3) 凍結モード OFF 検証、(4) **仮説 A: StJacobian z 成分カップリング再検（本命工事、status-291〜296 規模）**。~~n_incr=40~~ は反証済みにつき非推奨
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
- **19本撚線 Type D stall 対策（最優先）** — status-341 で仮説 C（曲率過粗さ）反証済み。次セッション開始時は **status-341 + status-339 の Type D 対策ガイドを最初に読むこと**
  - **推奨アクション 1（新）**: K_c FD 診断ダンプを 50 増分毎に取得し、7本（frac=1.0 完走）vs 19本（frac=0.199 stall）の comp別不整合スケーリングを比較。n_incr=40 stall では x=97%/z=70% 両成分が同時悪化しており、単一成分の issue ではない可能性
  - **推奨アクション 2**: gap_cull_threshold 手動掃引（リスク極小、パラメータ 1 つ）
  - **仮説 A（StJacobian z 成分カップリング）**: 優先度 4→1 に更新（status-341）。ただし status-291〜296 規模の工事なので、先に FD 診断で根拠固め
  - ~~n_incr=40 倍化~~: status-341 で反証済み（frac 退化）、再試行禁止
- **ファイバー梁キャリブレーション** — status-338 で得た 7本撚線 κ_cr 分布（mean=5.80e-3, CV=0.30）で `MultiLayerFrictionDegrading1D` のパラメータを推定
  - **7本撚線ピッチ依存性検証**: p=50/100/200 で κ_cr 分布を実測（完走確実な 7本で実施、19本は Type D 対策後）
  - **揺動サイクル履歴依存性観測**: `n_cycles=2` + `n_oscillation_cycles=1` で κ_cr の load/unload 非対称性を観測
  - **端部外れ値の物理確認**: pair (2, 18) の κ_cr=1.23e-2 外れ値が `exclude_end_elements` で説明されるか切り分け
- **リスタート解析方式**: ContactFrictionProcess の I/O を `(u, v, a, 接触ペア)` 入出力に整理
  - ソルバーが初期条件 `(u0, v0, a0, contact_pairs)` を受け取り、結果 `(u, v, a, contact_pairs)` を返す
  - 曲げ・揺動プロセスは境界条件（prescribed_dofs, prescribed_func, f_ext等）を渡すだけの薄いラッパー
  - update_reference を解析ステップ間で跨がない設計（CR梁ULのf_int=0問題の根本解決）
- **被膜圧縮モデル改善** — バリア関数 or 二層モデルで物理的に正確な被膜力
- **空間ブロック分離 or ペアクラスタリング**（n² 根本対策、1000本撚線への道）

## やってはいけないこと
- 管理上processクラスとすべきロジックをあえてプライベート関数や迂回ロジックに替えること
- 収束トライ時に目標を緩和して本質的対策を先送りにすること

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
