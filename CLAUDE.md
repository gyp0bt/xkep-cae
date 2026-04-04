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

### 作業完了時の必須手順

1. README.md 更新 → 2. status 新規作成/更新 → 3. status-index.md 更新 → 4. roadmap.md 更新
5. 不整合はその場で修正 or TODO追加 → 6. feature ごとにコミット → push

### ログ出力ルール

- 計算実行は**必ず tee でファイル出力**: `python script.py 2>&1 | tee /tmp/log-$(date +%s).log`
- `| tail -N` のみは禁止（途中経過が残らない）
- 収束ログには以下を含める: 時間増分カットバック、接触チャタリング、エネルギー収支、条件数

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

**621 テスト** — 2026-04-04 | 契約違反 **0件** | 条例違反 **0件**

### ターゲット

> **1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

### 次の課題

**接触あり90度曲げ frac=0.998達成** — status-285:
- ~~接触凍結モード（status-284）で frac=0.40→0.70（75%改善）~~ ← status-284で完了
- ~~Hertz型非線形ペナルティ（`p_n ∝ δ^{1.5}`）~~ ← status-285で完了（frac=0.70→0.998、事実上完走）
- ~~チャタリング内訳分析~~ ← status-287で完了（**活性集合振動ではなく接線剛性���整合(Type D=52%)が主因**）
- ~~収束診断ログ構造化 + Type D自動検知基盤~~ ← status-288で完了（NR進捗にType+rate、FD自動トリガー、Type D時NR拡張）
- ~~FD接線診断でHertz型∂p/∂g整合性検証~~ ← status-289で完了（**Hertz導関数は正確、K_c幾何項のcomp=2(z方向)不整合がType Dの根本原因**）
- **次**: frozen-m解消（∂m/∂uの正確計算）→ z方向DOFカップリングをK_stに追加 → NR 2次収束回復

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
- **MPC+接触のNR収束改善**（frac=0.35→1.0）
  - ~~DOF消去MPC実装（端部剛体結合）~~ ← status-253で完了
  - ~~StrandBendingOscillationProcess 実装~~ ← status-253で完了
  - ~~MPC u伝搬修正 + 収束実行テスト~~ ← status-254で完了（frac=0.35到達）
  - ~~MPC + 動的ソルバーの力残差整合性確認~~ ← status-254で確認（slave残差問題特定）
  - ~~MPC縮退系残差判定 + u_pred射影 + ストール検知拡張~~ ← status-255で完了
  - ~~MPC+接触の接線剛性FD診断~~ ← status-256で完了（TangentFDDiagnosticProcess実装）
  - ~~FD診断compute_residual実装 + 不整合箇所特定~~ ← status-257で完了
  - ~~K_c不整合再解析 + consistent_st_tangent=TrueデフォルトON~~ ← status-258で完了（K_c正確、不整合は活性集合変化）
  - ~~Huber smoothing_deltaパイプライン貫通 + 自動推定有効化~~ ← status-259で完了
  - ~~smoothing_deltaチューニング（1000/rで frac 0.35→0.59）~~ ← status-260で完了
  - ~~smoothing_delta=1000（手動）でfrac=1.0完走テスト実装 + active_contact_dofs NRソルバー結合~~ ← status-261で完了
  - ~~delta_h直接指定API実装（huber_delta_h パイプライン貫通）~~ ← status-261で完了
  - ~~delta_h最適値探索 + three_point_bend huber_delta_h貫通~~ ← status-262で完了（delta_h=0.025最速、非単調性あり）
  - ~~delta_hデフォルト値検討（three_point_bend検証 + 剛体-梁での検証）~~ ← status-263で完了（0.0維持、問題依存性高い）
- プロセス脱法修正（Phase D〜E、status-249 参照）
  - ~~A1-A3: アセンブラProcess化~~ ← status-250で完了
  - ~~C2-C3: 幾何計算Process化~~ ← status-255で完了
  - ~~B1-B4: 摩擦アセンブリProcess化~~ ← status-256で完了
- NR 残差収束速度の改善（中盤後〜終盤で 25 反復が力収束に不足、disp 収束で抜ける状態）
- Hermite 非局所 ∂g/∂u 対応（4ノードペア外の DOF 結合）
  - ~~Step1: StJacobian隣接ノード微分（ds_du_adj/dt_du_adj）~~ ← status-271で完了
  - ~~Step2: K_st拡張（隣接ノードDOFへの接線剛性エントリ追加）~~ ← status-272で完了
  - ~~Step3: K_c拡張（Hermite形状関数の隣接ノード依存性）~~ ← status-273で完了
  - ~~Step4: 摩擦K_st隣接ノード拡張~~ ← status-274で完了

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
