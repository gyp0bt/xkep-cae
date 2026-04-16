# ステータス一覧（status-index）

[← README](../../README.md) | [← roadmap](../roadmap.md)

> 本ファイルはステータスファイルの一覧メモです。新規status作成時に必ず更新すること。

## アクティブ status（275〜 — 接触完走・高速化フェーズ）

| # | 日付 | タイトル | テスト数 |
|---|------|---------|---------|
| [275](status-275.md) | 2026-03-31 | テスト品質改善 + frozen_hermite_tangent回帰修正(0.38→0.41) | 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+3 |
| [276](status-276.md) | 2026-03-31 | NR収束改善調査 — 接線不整合特定・対策方針策定 | 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+3 |
| [277](status-277.md) | 2026-03-31 | NR収束壁の根本原因特定 — dm不整合 + NR制御改善 | 600 passed |
| [278](status-278.md) | 2026-04-01 | NR収束壁の根本原因特定 — K_c/K_struct比問題 + dm一貫化 | 600 passed |
| [279](status-279.md) | 2026-04-02 | チェックポイント途中再開 + NR収束改善トライ（回帰なし） | 600 passed |
| [280](status-280.md) | 2026-04-02 | free_end_mode実装 — MPC不使用端部直接処方（frac 0.55→1.0完走） | 602 passed |
| [281](status-281.md) | 2026-04-03 | ヘリカル素線接触なし90度曲げ完走 — UL参照配置更新 | 606 passed |
| [282](status-282.md) | 2026-04-03 | 接触あり90度曲げベースライン — frac=0.40停滞（チャタリング） | 606 passed |
| [283](status-283.md) | 2026-04-03 | MPC変換行列T動的再構築 — MPC接触なし90度曲げ完走 | 606 passed |
| [284](status-284.md) | 2026-04-03 | チャタリング検知→接触凍結モード — frac 0.40→0.70改善 | 606 passed |
| [285](status-285.md) | 2026-04-03 | C16修正 + Hertz型非線形ペナルティ — frac 0.70→0.998 | 621 passed |
| [286](status-286.md) | 2026-04-04 | 揺動サイクル基盤 — prescribed_func + checkpoint自工程保証 | 621 passed |
| [287](status-287.md) | 2026-04-04 | チャタリング内訳分析 — 活性集合振動ではなく接線剛性不整合が主因 | 621 passed |
| [288](status-288.md) | 2026-04-04 | 収束診断ログ構造化 + Type D対策（FD自動トリガー・NR拡張） | 621 passed |
| [289](status-289.md) | 2026-04-04 | FD接線診断でHertz型∂p/∂g整合性検証 + K_c不整合箇所特定 | 624 passed |
| [290](status-290.md) | 2026-04-04 | FD接線診断強化 + Type D不整合のcomp別・DOF別分解 | 624 passed |
| [291](status-291.md) | 2026-04-04 | K_c不整合根本原因特定 + s_unclamped修正（Hermite 20%→0.0001%） | 624+ passed |
| [292](status-292.md) | 2026-04-04 | StJacobian 2×2カップリング修正（K_st FD不整合94%→0.0001%） | 631 passed |
| [293](status-293.md) | 2026-04-04 | StJacobian smooth遷移帯 + frozen-m内部接触点検証 | 631+ passed |
| [294](status-294.md) | 2026-04-05 | frozen-m部分解消（dm_A/dm_B有効化 + dm_ext無効化） | 631+ passed |
| [295](status-295.md) | 2026-04-05 | K_c_adj mat-only化（K_c FD誤差11%→1.8%）+ MPC+contact調査 | 631+ passed |
| [296](status-296.md) | 2026-04-05 | K_c FD 1.8%分析 + 端部接���除外 + frozen-m検証(**frac=0.9997完走**) | 442+ passed |
| [297](status-297.md) | 2026-04-05 | 微小dt耐性改善（dt snap + atol_force） | 442+ passed |
| [298](status-298.md) | 2026-04-06 | Hertz型+atol_force frac=1.0完走確認（ベースライン検証） | 442+ passed |
| [299](status-299.md) | 2026-04-06 | 90度曲げ+先端横変位±48mm揺動 完走（統合モード） | 442+ passed |
| [300](status-300.md) | 2026-04-07 | 変形メッシュ2D投影可視化スクリプト実装 | 442+ passed |
| [301](status-301.md) | 2026-04-07 | 7本撚線ソルバー性能分析 — 被膜でincr半減(1900→965) + 高速化フェーズ移行 | 442+ passed |
| [302](status-302.md) | 2026-04-08 | 被膜貫入量診断 — 平均54%圧縮、8.6%で芯線貫入 | 442+ passed |
| [303](status-303.md) | 2026-04-08 | バリア関数被膜モデル — 芯線貫入防止 | 442+11 passed |
| [304](status-304.md) | 2026-04-08 | 被膜接線剛性FD精度検証 + パラメータ物理的根拠分析 | 442+13 passed |
| [305](status-305.md) | 2026-04-08 | バリア被膜90度曲げ収束性検証 — incr42%削減・70%高速化 | 442+13 passed |
| [306](status-306.md) | 2026-04-08 | 被膜エネルギー比診断 + 収束テスト回帰修正 | 442+20 passed |
| [307](status-307.md) | 2026-04-08 | ソルバー診断ログ強化 — カットバック原因タグ・f_ref出力・収束型統計 | 442+20 passed |
| [308](status-308.md) | 2026-04-08 | 収束型統計デッドコード修正 + 接触ペア検出KD-tree化 | 442+20+14 passed |
| [309](status-309.md) | 2026-04-08 | K_c/K_stアセンブリベクトル化 + broadphase大規模ベンチマーク | 442+20+14+6 passed |
| [310](status-310.md) | 2026-04-08 | Hermite dpA/dpBバッチ化 + 摩擦K_stベクトル化 + K_st性能69-208x | 442+20+14+6+3+6 passed |
| [311](status-311.md) | 2026-04-08 | adj batchバッチ化 + BC適用20,000x高速化 + pypardiso統合 | 445+20+14+6+3+6 passed |
| [312](status-312.md) | 2026-04-09 | BC適用ベクトル化 + 責務分離違反修正 + MPC forループ排除 | 459 passed |
| [313](status-313.md) | 2026-04-10 | 撚線ファイバー梁モデル 設計仕様策定（work/beam_hysteresis 統合） | 459 passed |
| [314](status-314.md) | 2026-04-10 | プロファイル統計API強化 + BenchmarkRunnerプロファイル自動キャプチャ | 459+13 passed |
| [315](status-315.md) | 2026-04-10 | ParameterSweepBenchmarkProcess 新設 + manifest 連番衝突回避 | 459+13+11 passed |
| [316](status-316.md) | 2026-04-10 | n_strands 掃引プロファイリング初回実測（7/19/37）+ ボトルネック順位付け | 459+13+11 passed |
| [317](status-317.md) | 2026-04-10 | ParameterSweepBenchmark `dominant_leaf_process` — wrapper 占有を読み飛ばす真のボトルネック抽出 | 459+13+22 passed |
| [318](status-318.md) | 2026-04-11 | n_strands 掃引 6 ケース拡張（7/19/37/61/91/127）+ dominant_leaf_process 実測検証 + TangentAssembly avg/call 線形性確認（小曲率限定） | 459+13+22 passed |
| [319](status-319.md) | 2026-04-11 | 初期ギャップ固定 + 大曲率でのバイアス補正掃引 — ContactForceStStiffness/FrictionStStiffness α≈2.07 の n² scaling 実測、status-318 結論の scaling 視点再解釈 | 459+13+22 passed |
| [320](status-320.md) | 2026-04-11 | `uses` グラフ拡張 — `StrategySlot.default_types` で `ContactFrictionProcess` から接触/摩擦 K_st 系 8 Process をクラスレベルで到達可能化、`_is_leaf_process` も StrategySlot 併合判定、5 テスト追加 | 459+13+22+5 passed |
| [321](status-321.md) | 2026-04-11 | K_st アセンブリ CSR/COO 経路最適化 — tocsr skip + einsum→broadcasting + mask filter skip + friction 戦略単一 COO concat + 抽出ループ active 比例化、**FrictionStStiffness per-call 33% 高速化** | 459+13+22+5 passed |
| [322](status-322.md) | 2026-04-12 | `ProcessExecutionLog._find_caller` を `sys._getframe()` + `lru_cache` 化 — **全 Process 呼び出しの ~2.5ms 固定オーバーヘッド eliminate**、ContactForceSt per-call 16.8ms→14.4ms（14% 高速化）、test_beam_oscillation 実行時間 17x 改善、ContactForceSt ローカルベクトル化併用 | 459+13+22+5 passed |
| [323](status-323.md) | 2026-04-12 | beam oscillation 物理テスト修復（UL参照更新無効化 + モード形状分布初速度）— 5 failed→0 failed + 1 skip + `_find_caller` skip list 評価（拡張不要）+ distance culling/symbolic factor reuse 調査 | 459+13+22+5 passed |
| [324](status-324.md) | 2026-04-12 | K_st distance culling 実装 — Huber 遷移幅ベースの gap pre-filter（Contact K_st 自動閾値計算 + Friction K_st パイプライン貫通 + `compute_gap_cull_threshold()` 公開メソッド + 8 テスト追加） | 459+13+22+5+8 passed |
| [325](status-325.md) | 2026-04-12 | symbolic factorization reuse — `_SolverCache` で pypardiso symbolic 分析キャッシュ（パターン検出 + `factorize()` reuse + `LinearSolveProcess` v1.2.0 + 12 テスト追加） | 459+13+22+5+8+12 passed |
| [326](status-326.md) | 2026-04-12 | ファイバー梁 Phase F1 実装（`Elastic1D` / `BilinearKinematicHardening1D` + 12 テスト）+ culling/cache 効果定量計測（ContactForceStStiffness **96-99% 高速化**、scaling α=2.07→1.24） | 459+13+22+5+8+12+12 passed |
| [327](status-327.md) | 2026-04-13 | ファイバー梁 Phase F2 実装（`MultiLayerFrictionDegrading1D` — N 層並列摩擦+弾性バックボーン+接触剛性劣化、`05_smooth_teardrop.py` 完全再現 rtol<1%、KH 等価性検証、12 テスト追加） | 459+13+22+5+8+12+12 passed |
| [328](status-328.md) | 2026-04-13 | ファイバー梁 Phase F3 実装（`CircularFiberSection` + `FiberSectionIntegratorProcess` — 断面ファイバー離散化 strip/polar + 断面積分 Process、FD 接線検証 Elastic/BilinearKH/MultiLayerFriction 3 材料合格、弾性 EI 誤差 < 1%、25 テスト追加） | 459+13+22+5+8+12+12+25 passed |
| [329](status-329.md) | 2026-04-13 | ファイバー梁 Phase F4 実装（`StrandFiberBeamProcess` + `ULCRFiberBeamAssembler` — CR Timoshenko 梁要素ファイバー積分統合 + UL マルチ要素アセンブラ配線、弾性内力 < 0.2%・接線対角 < 1%・FD 自己整合検証合格、26 テスト追加） | 459+13+22+5+8+12+12+25+26 passed |
| [330](status-330.md) | 2026-04-14 | ファイバー梁 Phase F5 実装（`StrandBendingOscillationProcess` に `use_fiber_beam` フラグ — 直線梁メッシュ+ファイバー断面積分+TL定式化、弾性先端変位0.02%・BilinearKH/MultiLayerFriction NR収束合格、10テスト追加） | 459+13+22+5+8+12+12+25+26+10 passed |
| [331](status-331.md) | 2026-04-14 | Phase F5 散逸エネルギー検証 — `CableDissipationProcess` + M-κ ヒステリシス追跡（散逸∝κ^1.9 超線形、撚線本数/劣化比/BilinearKH 検証、checkpoint bugfix、15テスト追加） | 459+13+22+5+8+12+12+25+26+10+15 passed |
| [332](status-332.md) | 2026-04-14 | 断面接触点統計モデル — Papailiou解析 + 分布閾値拡張（κ冪1.85完全再現、n≥7で±10%精度、ピッチ非依存性証明） | 459+13+22+5+8+12+12+25+26+10+15 passed |
| [333](status-333.md) | 2026-04-14 | M-κ追跡 + 接触ペアスナップショット — CR梁接触動解析でのM-κヒステリシス直接取得基盤（9テスト追加） | 459+13+22+5+8+12+12+25+26+10+15+9 passed |
| [334](status-334.md) | 2026-04-14 | C16 契約違反 12 件解消 — `cable_dissipation.py` / `strand_cross_section_model.py` の純粋関数 12 本を `_` prefix で private 化（契約違反 12→0） | 459+13+22+5+8+12+12+25+26+10+15+9 passed |
| [335](status-335.md) | 2026-04-14 | 2本撚線 M-κ ヒステリシスループ直接観測（infra 検証）— `n_oscillation_cycles=1` 統合モードで load+unload、κ 下降14回・loop_area=1.17e-2 観測（1テスト追加） | 459+13+22+5+8+12+12+25+26+10+15+10 passed |
| [336](status-336.md) | 2026-04-14 | M-κ ループ散逸率を load-only 弾性仕事基準に厳格化 — `_compute_mk_metrics` を活用して `loop_area/W_load=0.32` を観測、外接矩形比 0.86 を廃止（テスト数変更なし） | 459+13+22+5+8+12+12+25+26+10+15+10 passed |
| [337](status-337.md) | 2026-04-14 | ContactPairAnalysisProcess 新設 — `contact_pair_history` から κ_cr 分布・各ペア散逸・活性ペア数推移を抽出する後処理 Process（PostProcess、9テスト追加） | 459+13+22+5+8+12+12+25+26+10+15+10+9 passed |
| [338](status-338.md) | 2026-04-14 | 7本撚線 κ_cr 実測（ContactPairAnalysisProcess 初回運用） — κ_cr mean=5.80e-3, std=1.74e-3, CV=0.30, n_slipped=24/26（右裾型分布、281s で 90°曲げ完走）。ファイバー梁校正データ取得開始 | 459+13+22+5+8+12+12+25+26+10+15+10+9 passed |
| [339](status-339.md) | 2026-04-14 | 19本撚線 κ_cr 実測 — **frac=0.484 で Type D stall（未完走）**。ただし 57/59 ペアのデータ取得成功（mean=4.50e-3, CV=0.33、バイモーダル気配）。次セッション向け Type D 対策ガイド策定（K_c FD 診断 / n_incr=40 / gap_cull 掃引 / 仮説 A: z 成分不整合） | 459+13+22+5+8+12+12+25+26+10+15+10+9 passed |
| [340](status-340.md) | 2026-04-14 | `ContactPairLayerClassifierProcess` 新設 — `(elem_a, elem_b)` を `(layer_min, layer_max)` に集約する PostProcess、status-339 のバイモーダル気配（内層対 vs 外層対）を定量検証可能に。`StrandMeshResult.strand_layers` 追加、19本実測スクリプトに層分類出力統合（8 テスト追加） | 459+13+22+5+8+12+12+25+26+10+15+10+9+8 passed |
| [341](status-341.md) | 2026-04-15 | 19本撚線 n_incr=40 リトライ — **仮説 C（曲率プロファイル過粗さ）反証**。frac=0.4839→**0.1991 退化**（n_incr=20 比 -59%）。stall 主 comp が z=65% → x=72-97% に変化し、成分選択的ではなく **広範な K_c 不整合** の可能性を示唆。仮説 A（StJacobian z 成分）優先度を 1 に引き上げ、次セッションは K_c FD 診断取得を最優先アクションに更新 | 459+13+22+5+8+12+12+25+26+10+15+10+9+8 passed |
| [342](status-342.md) | 2026-04-15 | 19本撚線 K_c FD 診断取得（166 レコード） — `work/beam_hysteresis/12_kc_fd_diagnostic_19strand.py` 新設、`tangent_fd_diagnostic`+`type_d_auto_fd` で Type D stall 中の FD レポートを stdout 捕捉→CSV 化。**`f_c` FD 相対誤差 mean=115%/max=191%** で K_c 自体が大きく狂うことを実測、不整合方向は **x 成分支配（f_c comp x=68.3%, y=44.2%, z=41.5%）**。status-341 の「z 支配」は beam coupling の 2 次効果と判明し、**仮説 A を「x 成分 primary driver」に再定義**。次は K_c mat/geo/st 分解 FD 診断で由来切り分け | 459+13+22+5+8+12+12+25+26+10+15+10+9+8 passed |
| [343](status-343.md) | 2026-04-15 | K_c 成分分解 FD 診断 Process 新設 — `ContactKcComponentFDDiagnosticProcess` を `xkep_cae/verify/kc_component_fd.py` に追加。K_c = K_mat - K_geo + K_st の 4 組み合わせ（full/mat_only/mat_geo/mat_st）で FD 相対誤差 + 成分別不整合シェア + 寄与率を報告し、status-342 の x 成分 68% 不整合の由来を部分行列レベルで切り分ける基盤を整備。単体テスト 11 件追加（線形系セルフチェック + st_primary_driver 分離検証等） | 459+13+22+5+8+12+12+25+26+10+15+10+9+8+11 passed |
| [344](status-344.md) | 2026-04-15 | 19本撚線 K_c 成分分解 FD 初回実測（183 レコード） — status-343 Process をソルバーに配線（`ContactFrictionInputData.kc_component_fd_diagnostic` + `_newton_dynamic.py` フック）、`work/beam_hysteresis/13_kc_component_fd_19strand.py` で Type D stall 断面 183 件の FD 診断取得。**仮説 A 決着**: 最良組み合わせ = `mat_only` 100%（183/183）、**share_geo=0.000 全件**、K_st 追加で rel_err 平均 +16%/最大 +52% 悪化、mat_only で rel_err mean=44% / comp_x max=98%。K_mat 主導 + K_st 追従の構造が確定し、次工事は **K_mat の x/z 成分カップリング再検**（status-295 規模） | 459+13+22+5+8+12+12+25+26+10+15+10+9+8+11 passed |
| [345](status-345.md) | 2026-04-15 | status-344「K_geo=0」誤認の訂正 — `ContactKcComponentFDDiagnosticProcess` の report 寄与率フォーマットが `{:5.2f}` で微小値を 0.00 に丸めていたことに起因。既存 log 再解析で **K_geo share mean=1.02e-3 / max=3.79e-3**（K_mat の 0.1% 程度で非ゼロ）と復元。report を `{:.3e}` に修正 + Output に `mat/geo/st/full_du_norm` + `dfc_fd_norm` 公開。status-344 推奨アクション 3（K_geo==0 原因調査）は実装バグでなく表示精度問題として **クローズ**。仮説 A 主旨（K_mat 主導）は維持。テスト 1 件追加（11→12） | 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12 passed |
| [346](status-346.md) | 2026-04-16 | **MCDD Phase A-1 — `MathematicalContract` 型システム新設**: `xkep_cae/mathematics/` パッケージ新設（`contracts.py` + `docs/mathematics.md` + `tests/`）。5 種の frozen dataclass 契約型（`IdentityContract` / `InequalityContract` / `FDConsistencyContract` / `SymmetryContract` / **`TermExpansionContract`** ★MCDD の核）を実装。`providers` 重複検出・長さ一致検証・frozen/severity 必須性で脱法実装 pattern 2/3/9 を型レベルで封じ込め。既存 Process 改変なし、33 テスト追加、契約違反 0 件。計画 `/root/.claude/plans/deep-wiggling-seal.md`（v1.0.0 凍結）の Phase A〜E / status-346〜356 の 1/11 を完了。他ロードマップ項目は MCDD 完了まで凍結 | 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33 passed |

## アーカイブ（175〜274 — 新 xkep_cae R1完遂・NR収束改善・Hermite非局所対応）

status-175〜274 は [archive/](archive/) に移動済み（status-322 で実施）。

| # | 日付 | マイルストーン | テスト数 |
|---|------|--------------|---------|
| [175](archive/status-175.md) | 2026-03-15 | 脱出ポット計画 Phase 1 — xkep_cae リネーム + PenaltyStrategy 書き直し | ~2260+34p |
| [179](archive/status-179.md) | 2026-03-15 | Phase 2 後半 — Strategy 全移行 + 契約違反ゼロ | ~2260+186p |
| [188](archive/status-188.md) | 2026-03-16 | R1 Phase 7 完了 — C14/C16 違反ゼロ | ~2260+284p |
| [207](archive/status-207.md) | 2026-03-18 | deprecated コード完全削除 + コンテキスト大掃除 | 248p |
| [210](archive/status-210.md) | 2026-03-18 | smooth_penalty ソルバー復元 + HEX8 基盤 | 412+14p |
| [222](archive/status-222.md) | 2026-03-21 | Huber ペナルティ統一（ソルバー一本化） | 499 |
| [226](archive/status-226.md) | 2026-03-22 | K_st 実装 — ∂(s,t)/∂u 整合接線 + FD 検証 11 件 | 175+11 |
| [253](archive/status-253.md) | 2026-03-26 | DOF 消去 MPC 実装 + StrandBendingOscillation | 200+10s+16+3+23+1+6+18 |
| [258](archive/status-258.md) | 2026-03-28 | K_c 不整合再解析 + consistent_st_tangent ON | 200+10s+16+3+23+1+6+18+2+4+3+9 |
| [264](archive/status-264.md) | 2026-03-29 | E=25 回帰修正（frozen_hermite_tangent + n_elems=8） | 200+10s+16+3+23+1+6+18+2+4+3+9+4 |
| [271](archive/status-271.md) | 2026-03-30 | frozen=False 検証 + Hermite 非局所 ∂g/∂u Step1 | 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2 |
| [274](archive/status-274.md) | 2026-03-31 | 摩擦 K_st 隣接ノード拡張（Hermite 非局所 Step4 完了） | 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+3 |

## アーカイブ（097〜174 — 旧 xkep_cae S3/R1 フェーズ）

status-097〜174 は [archive/](archive/) に移動済み。

| # | 日付 | マイルストーン | テスト数 |
|---|------|--------------|---------|
| [097](archive/status-097.md) | 2026-03-01 | S3 開始 — xfail テスト根本対策 | 1906 |
| [112](archive/status-112.md) | 2026-03-05 | 19 本 NCP 収束達成 | 2122 |
| [130](archive/status-130.md) | 2026-03-07 | UL+CR 梁 — 7 本 90° 曲げ収束達成 | 2271 |
| [147](archive/status-147.md) | 2026-03-09 | smooth penalty 摩擦曲げ揺動収束達成 | 2271 |
| [162](archive/status-162.md) | 2026-03-13 | R1 Phase 7 完遂 — 契約違反 0 件 | 2477 |
| [174](archive/status-174.md) | 2026-03-15 | solver_smooth_penalty.py 分解 → Process 実体化 | ~2260+343p |

## アーカイブ（001〜096 — Phase 1〜S2）

status-001〜096 は [archive/](archive/) に移動済み。

| # | 日付 | マイルストーン | テスト数 |
|---|------|--------------|---------|
| [001](archive/status-001.md) | 2026-02-12 | プロジェクト棚卸し・ロードマップ策定 | — |
| [015](archive/status-015.md) | 2026-02-14 | Phase 2 完了 — 空間梁要素 | 374 |
| [030](archive/status-030.md) | 2026-02-18 | Phase 5 完了 — 動的解析+接触骨格 | 615 |
| [046](archive/status-046.md) | 2026-02-21 | Phase C0-C5 完了 — 梁–梁接触基盤 | 993 |
| [081](archive/status-081.md) | 2026-02-28 | Phase C6 完了 — Line contact+NCP+摩擦 | 1850 |
| [096](archive/status-096.md) | 2026-03-01 | S2++/S3 基盤完了 — COO/CSR 高速化 | 1886 |

## テスト数推移（主要マイルストーン）

```
Phase 1 完了:     16  (2026-02-12)
Phase 2 完了:    374  (2026-02-14)
Phase 3 完了:    407  (2026-02-14)
Phase 4.1-4.2:   471  (2026-02-16)
Phase 5 完了:    615  (2026-02-18)
過渡応答出力:    789  (2026-02-18)
Phase C0-C5:     993  (2026-02-21)
撚線基礎:      1311  (2026-02-25)
HEX8:           1478  (2026-02-26)
GNN/PINN PoC:  1629  (2026-02-26)
Phase C6:       1850  (2026-02-28)
S1-S2:          1822  (2026-02-28)
S2++/S3基盤:    1886  (2026-03-01)
R1 Phase 7:    2477+314p (2026-03-13)
新xkep_cae開始:  ~2260+34p(新) (2026-03-15) ← status-175（脱出ポット計画Phase1）
R1完遂+契約0:    ~2260+284p(新) (2026-03-16) ← status-188（C14/C16違反ゼロ）
deprecated全削除: 248p(新) (2026-03-18) ← status-207（コンテキスト大掃除）
Huber統一:        499(新) (2026-03-21) ← status-222（ソルバー一本化）
K_st実装:         175+11(新) (2026-03-22) ← status-226（∂(s,t)/∂u整合接線+FD検証）
MPC+NR改善開始:  200+10s+16+3+23+1+6+18 (2026-03-26) ← status-253（DOF消去MPC）
Hermite非局所完了: 200+10s+16+3+23+1+6+18+2+2+2+3 (2026-03-31) ← status-274（Step4完了）
                   ↑ status-229〜274 の詳細は archive/ 参照
Hertz+frac=1.0:    200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4 (2026-04-06) ← status-298（frac=1.0完走）
smooth遷移帯:          200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4(新) (2026-04-04) ← status-293（StJacobian smooth blending+unclamped IFT+frozen-m検証）
K_c_adj mat-only:      200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4 (2026-04-05) ← status-295（K_c_adj mat-only化+MPC+contact調査）
ベースライン検証:      200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4 (2026-04-06) ← status-298（Hertz+atol_force frac=1.0完走確認）
2D投影可視化:          200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4 (2026-04-07) ← status-300（変形メッシュ2D投影可視化スクリプト実装）
被膜貫入診断:          200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4 (2026-04-08) ← status-302（被膜貫入量診断 — 平均54%圧縮、8.6%芯線貫入）
バリア関数被膜:        200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+11(新) (2026-04-08) ← status-303（バリア関数 f=kδ/(1-δ/δ_max) + 11テスト）
FD精度+パラメータ:     200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13(新) (2026-04-08) ← status-304（FD誤差67%=幾何接線欠落、k_coat=非物理的正則化）
バリア被膜検証:        200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13(新) (2026-04-08) ← status-305（バリア被膜90度曲げ: incr535→308(42%削減), 752s→224s(70%高速化)）
デッドコード修正+KDtree: 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14(新) (2026-04-08) ← status-308（収束型統計デッドコード修正+broadphase KD-tree化）
K_stベクトル化:         200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6(新) (2026-04-08) ← status-309（K_stアセンブリベクトル化+broadphase大規模ベンチマーク）
高速化第2弾完了:        200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9(新) (2026-04-08) ← status-310（Hermite dpA/dpBバッチ化+摩擦K_stベクトル化+adj_node_map配列化+K_st性能69-208x高速化）
高速化第3弾+adj:        200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3(新) (2026-04-08) ← status-311（adj batchバッチ化+BC適用20000x高速化+pypardiso統合）
BC+責務修正:            200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5(新) (2026-04-09) ← status-312（BC forループNumPyベクトル化+MPC forループ排除+strand_bending_oscillation責務分離修正）
ファイバー梁設計:        200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5(新) (2026-04-10) ← status-313（撚線ファイバー梁モデル 設計仕様策定 / work/beam_hysteresis 統合）
プロファイルAPI:         200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13(新) (2026-04-10) ← status-314（ProcessMetaclassプロファイル統計API強化+BenchmarkRunnerプロファイル自動キャプチャ）
スイープ基盤:           200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13+11(新) (2026-04-10) ← status-315（ParameterSweepBenchmarkProcess新設+manifest連番衝突回避bugfix）
掃引実測#1:             200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13+11(新) (2026-04-10) ← status-316（n_strands=7/19/37 掃引実測+ボトルネック順位付け、テスト数変更なし）
葉プロセス抽出:         200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13+11+11(新) (2026-04-10) ← status-317（ParameterSweepBenchmark dominant_leaf_process 追加+_collect_uses_graph/_is_leaf_process/_first_leaf_breakdown_entry+11テスト）
掃引6ケース拡張:        200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13+11+11(新) (2026-04-11) ← status-318（n_strands=7/19/37/61/91/127 掃引拡張+dominant_leaf_process 実測検証+TangentAssembly avg/call 線形性確認、テスト数変更なし）
バイアス補正掃引:       200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13+11+11(新) (2026-04-11) ← status-319（初期gap固定+大曲率での補正掃引 n=7/19/37、ContactForceStStiffness/FrictionStStiffness α≈2.07 の n² scaling 実測、status-318 の線形性結論を小曲率限定と判定、テスト数変更なし）
usesグラフ拡張:         200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13+11+11+5(新) (2026-04-11) ← status-320（`StrategySlot.default_types` 追加+`_collect_uses_graph`/`_is_leaf_process` StrategySlot 対応+`ContactFrictionProcess` から K_st 系 8 Process 到達可能化+5 テスト追加）
K_st経路最適化:         200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13+11+11+5(新) (2026-04-11) ← status-321（K_st/K_mat/K_geo の tocsr() skip + einsum→broadcasting + mask filter skip + friction 戦略単一 COO concat + 抽出ループ active 比例化、FrictionStStiffness per-call 17.84ms→11.91ms 33% 高速化、テスト数変更なし）
診断ログ高速化:         200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13+11+11+5(新) (2026-04-12) ← status-322（ProcessExecutionLog._find_caller を sys._getframe()+lru_cache 化、全 Process の ~2.5ms 固定オーバーヘッド eliminate、ContactForceSt 16.8ms→14.4ms 14% 高速化、ContactForceSt のベクトル化ローカル最適化併用、テスト数変更なし）
beam振動修復:           200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13+11+11+5(新) (2026-04-12) ← status-323（beam oscillation 5件失敗修復: UL参照更新無効化+モード形状分布初速度+time_arr修正+matplotlib skipif、_find_caller skip list評価（拡張不要）、distance culling/symbolic factor reuse調査、テスト数変更なし）
distance culling:       200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13+11+11+5+8(新) (2026-04-12) ← status-324（Contact K_st gap pre-filter + Friction K_st gap_cull_threshold パイプライン貫通 + compute_gap_cull_threshold() 公開メソッド + 8テスト追加）
symbolic fact cache:    200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4+13+14+6+9+3+5+13+11+11+5+8+12(新) (2026-04-12) ← status-325（_SolverCache クラス新設、LinearSolveProcess v1.2.0、パターン検出+factorize reuse、12テスト追加）
ファイバー梁F5統合:     459+13+22+5+8+12+12+25+26+10(新) (2026-04-14) ← status-330（StrandBendingOscillationProcess use_fiber_beam フラグ + TL定式化 + 弾性0.02%/BilinearKH/MultiLayerFriction収束 + 10テスト追加）
```

## 備考

- テスト数「—」はドキュメント更新・計画策定のみのステータス
- status-001〜096 は `docs/status/archive/` に移動（status-100 で実施）
- status-097〜174 は `docs/status/archive/` に移動（status-177 で実施）
- status-175〜274 は `docs/status/archive/` に移動（status-322 で実施）
- status-275〜 がアクティブ status（接触完走・高速化フェーズ）
- **アーカイブ方針**: アクティブ 50 件超過時に最古バッチを archive/ へ移動

---
