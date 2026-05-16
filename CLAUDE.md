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
5. **`docs/status/verification_matrix.md` 該当行を更新**（達成 ✅ / 部分 🟡 / 未達 ❌ /
   撤回 🔁。撤回があれば §5 STA2 撤回履歴に履歴として追記、削除は禁止。status-393 で
   運用化、STA2 連鎖撤回防止のための必須手順）
6. 不整合はその場で修正 or TODO追加 → 7. feature ごとにコミット → push

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

**766 passed 5 skipped** — 2026-05-16 | 契約違反 **0件** | 条例違反 **0件**

**最新**: [status-400](docs/status/status-index.md) — `VtkExportProcess` 実装。ParaView 用 VTK XML 出力 PostProcess（依存追加なし、生 XML 直接書き）、汎用 1D 梁モデル対応、+11 単体テスト。`.pvd` 時系列 + `.vtu` で電線曲げ揺動の視覚確認が可能に。

**前 status**: [status-399](docs/status/status-index.md) — `explicit_n_sub_cycles_per_increment` 実装、ε-1 で N=2000 で rel_err 0.01% asymptote、MCDD 凍結解除条件 (5) を単 strand 規模で PASS。

**前後関係の詳細は `docs/status/status-index.md` および各 status ファイル参照**（status-275〜376 は `docs/status/archive/` へ移動、本セッションで実施）。

### ターゲット

> **1000本撚線（10万節点）の曲げ揺動計算を6時間以内に完了する。**

### 次の課題

**実行方針**: implicit は完全凍結（解除想定なし）、explicit 一本路線。
詳細な完了履歴は `docs/status/status-index.md` および
`docs/status/verification_matrix.md` 参照。

#### Phase ε ロードマップ（段階的検証）

| status | scope | gate |
|---|---|---|
| ~~399~~ | ε-1 = 1 strand + 接触なし + explicit-TL + sub-cycling | ✅ N=2000 で rel_err 0.01% PASS |
| ~~400~~ | `VtkExportProcess` 実装（副次・基盤、ParaView 視覚確認） | ✅ +11 テスト、依存追加なし |
| **401 (次)** | ε-2 = 3 strand + 接触あり + N_sub=2000 | 3 指標 AND + frac=1.0 + 初の接触統合検証 |
| 402 | ε-3 = 7 strand + 接触あり | 3 指標 AND + frac=1.0 + max\|u\| < L×10 + implicit baseline 対比 |
| 403 | ε-4 = 19 strand + 接触あり（本命） | MCDD 凍結解除条件 (2)(3)(5) 同時達成試行 |

#### MCDD 凍結解除条件（status-381 で精度 gate (5) 追加）

1. Phase E 完了（C18〜C24 + O1〜O3、status-369 までに大半完了）
2. 19 本 frac=1.0 完走 — ε-4 で試行
3. **max\|u_trans\| < L_strand × 10**（撚線 100mm に対し最大変位 1m 以内）—
   status-380「frac=1.0 + E_kin/E_strain<5% は両方とも数学的構造由来で発散時にも PASS する」盲点対策
4. `KcNormalDirectionStiffness` FD rel_err < 1e-2 — ✅ status-356 達成
5. **解の精度 < 10%**（`|u_explicit − u_implicit| / |u_implicit|` または vs analytical）—
   status-381「形式 gate (1)〜(4) は under-relaxation 解でも PASS する」盲点対策。
   ε-1 単 strand は status-399 で N=2000 PASS、本命は ε-4

**status-379 撤回**: 条件 3 欠落で誤判定。**status-381 撤回**: 条件 5 欠落で誤判定。
**status-387 撤回**: 単一指標一致による偽 PASS（透明性ルール参照）。

#### 副次タスク（並行可能 / 後回し）

- **Phase γ-2 大 curvature 拡張**（θ=π/2）— γ-1/γ-3 は θ=0.15 rad small-medium。
  full pitch (2π rad) で「16 要素/ピッチ厳守」を再確認
- **既存 validation の 3 指標 gate 化**（status-388/389 TODO）—
  `test_assembler_process.py` / `test_strand_beam_physics.py` /
  `test_beam_oscillation.py` / `TestHelical90DegBendPhysics` を順次拡張
- **Phase δ 接触あり 2 本撚線** — ε-2 の前段、優先度は ε-1/ε-2 結果次第

#### scope 外（再開しないこと）

- ~~候補 (z2) Cosserat 梁プロトタイプ~~ — status-391 で absolute necessity 消失
- ~~候補 (q3) implicit + AL n>2 復活 / (h5) bending 段階処方~~ — implicit 凍結で scope 外
- ~~n_inc=8000 sweet spot 探索~~ — status-388 で偽（非物理解）と確定
- **凍結中 TODO 再開**: 被膜圧縮モデル / リスタート方式 / ファイバー梁キャリブレーション /
  7 本撚線ピッチ依存性 / 空間ブロック分離 / 19 本 Type D stall K_mat x/z 単発対応 —
  status-345 で凍結、MCDD 凍結解除条件達成後に再開

## フォーカスガード（AI セッション向け）

**以下を厳守すること。違反は作業のやり直しになる。**

## やるべきこと

### MCDD（数理契約駆動開発）の現状

**計画書**: `/root/.claude/plans/deep-wiggling-seal.md` は**永久ロスト**（status-352 で
記録、復旧不可）。代替として本 CLAUDE.md の規範セクション + 最新 status + 数理台帳
`xkep_cae/mathematics/docs/mathematics.md` で運用。

**完了済み**（詳細は status-index）:

- **Phase A** (status-346/347): `MathematicalContract` 5 種 + `@verified_by` +
  AST 検査による dummy/hollow VerifyProcess の構造的封じ込め
- **Phase B** (status-348/349): 数理台帳 6 章 / 55 アンカー + `equation_index.py` + C15 拡張
- **Phase C** (status-350-356): `KcNormal` / `KcGeo` / `KcHermiteNonlocal` /
  `KcClosestPoint` 抽出、5 項 `TermExpansionContract`、Phase C-3' 仮説 A+B 同時導入で
  `test_helical_3d_hermite` rel_err **1.795% → 2.18e-07** 達成
- **Phase E** (status-357-364): C18〜C24 契約検査 = 全 24 項目稼働
- **NR escape hatch 全候補 (a)〜(g)** (status-358-376): 19 本 Type D stall を NR 側で
  解消する試行は全て却下、K_c x/z カップリング不整合が主因と確定 → explicit 路線へ移行
- **explicit 時間積分** (status-377-399): `ExplicitCentralDifferenceProcess` 実装、
  mass scaling auto-tune、selective scaling、2 段階 scaling、Phase α/β/γ で CR foundation
  健全性確定、status-399 で sub-cycling 実装 → ε-1 で N=2000 で MCDD 凍結解除条件 (5) PASS

**次セッション最優先**: ε-2 = 3 strand + 接触あり + N_sub=2000（status-400）。
3 指標 AND gate + frac=1.0 完走、初の接触統合検証。

### セッション開始時の必須確認

1. **`docs/status/status-index.md` で最新 status 番号 + 直近数件の見出しを確認**
2. **最新 `docs/status/status-{N}.md` を読む** — 前セッションの停止点・引継ぎ
3. **`docs/status/verification_matrix.md` を読む**（status-393 で運用化） —
   達成 ✅ / 部分 🟡 / 未達 ❌ / 未検証 ⬜ / 凍結 ⏸ / 撤回 🔁 の状態凡例で
   STA2 連鎖撤回を構造的予防。「自分の作業がどの行を更新するか」を先に把握
4. **本ファイル「MCDD 脱法実装禁止パターン 10 項」を読み返し**、本セッションで
   陥りそうな項目を自己チェック
5. **`python contracts/validate_process_contracts.py` で全 24 検査 OK 確認**

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

### 妥当性テストの透明性ルール（status-388 追加・厳罰）

**「max\|u\| 単一指標一致」は偶然の交差を許容するため STA2 該当**。物理的妥当性を
主張するには **独立な解析解 3 個以上の同時一致** が必須。違反は status-387 の
ような誤判定を生む（解析解 73.30mm（90°）と実 BC 解析解 70.44mm（86°）の
取り違えで「err 0.58% 達成」と誤報告した事例）。

**最低 3 指標**（互いに独立、kinematics と energetics-or-geometric の両方を含む）:

1. **位置成分 1**（例: 先端 x 変位 `u_x`）
2. **位置成分 2**（例: 先端 z 変位 `u_z`）
3. **kinematics と独立な指標** — 以下のいずれか:
    - **エネルギー量**: 歪エネルギー `SE_final = (1/2) EI κ² L` / 外力仕事 `W_ext` / 反力モーメント `M_reaction`
    - **不伸長性**: 変形後 chord 和 `L_arc ≈ L`（pure bending 想定）
    - **曲率分布**: `κ(s)` 一様性または midspan `κ_mid`
    - **断面回転**: 内部節点での tangent 方向角

`|u|` ノルムは `u_x` / `u_z` から導出されるため独立指標としてカウント不可。
SE は実装の `0.5 u^T f_int` が MPC 拘束 DOF 消去で信頼できない場合があり
（status-388 で実証）、その場合は `L_arc` 等の geometric 指標で代替する。
**(1)(2) のみで判定するのは不可** — 必ず kinematics と独立な検証量を 1 個以上加える。

**判定基準**: **全 3 指標**が gate（10%）を通過したときのみ「精度達成」と判定。
1 指標 PASS / 2 指標 FAIL は「**達成と装わない**」（数値の捏造禁止に該当）。

**実 BC の解析解を使う**: BC が θ=κ·L で 86° なら 86° の解析解を使う。90° の
解析解で代用すると、勾配付近の座標一致は座標値そのものの違いを誤差として捕捉
できない（status-387 の根本ミス）。

**status 記載形式**:

```
| n_inc | u_x [mm] | u_z [mm] | SE [N·mm] | gate (3 指標 AND) |
| ----: | -------: | -------: | --------: | :----------------: |
| anal  |  −33.50  |  +61.96  |    71.79  |        —           |
| ...   |     ...  |     ...  |       ... |     PASS / FAIL    |
```

### 担当者間再現性ルール（status-246 追加）
- **ベンチマーク条件の記録**: テスト名、ブランチ名、コミットハッシュ、実行コマンドを tee ログおよび status ファイルに記録すること。
- **変更前ベースラインの先行取得**: 性能改善テスト前に必ず `git stash` で変更前コードのベースラインを計測し、ログに残す。
- **再現手順の status 記載**: status ファイルに「再現手順」セクションを設け、別の担当者が同じ結果を得られるコマンド列を明記する。
- **Process profiling の活用**: `ProcessMetaclass._profile_data` による自動計測結果を活用し、手動計測に頼らない仕組みを推進する。

