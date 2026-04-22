# xkep-cae

[![CI](https://github.com/gyp0bt/xkep-cae/actions/workflows/ci.yml/badge.svg)](https://github.com/gyp0bt/xkep-cae/actions/workflows/ci.yml)

ニッチドメイン問題を解くための自作有限要素ソルバー基盤。
構成則・要素・ソルバー・接触をモジュール化し、問題特化ソルバーを構成する。

## ターゲットマイルストーン

> **1000本撚線（10万節点）の曲げ揺動シミュレーションを6時間以内に完了する。**

| 項目 | 現状 | 目標 |
|------|------|------|
| 素線数 | **37本**（径方向圧縮Layer1で収束達成）| 1000本（~30,000 DOF, 長手分割で~100,000節点） |
| 計算時間 | 91本で~25分/曲げ揺動 | 1000本で6時間以内 |
| ソルバー | NCP: **37本収束達成**、S3改良12項目実装済み | NCP: 91本収束 |
| 接触ペア | 91本で~66,000候補 | 1000本で~730万候補→ML削減 |

## 現在の状態

**459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25 テスト** — 2026-04-22時点 | **仮説 C 候補 (a') 19本撚線検証で却下 + Phase E C21/C22/C23 追加**（[status-360](docs/status/status-360.md)）— status-359 で 7 本撚線にて `smoothing_delta=1000` が `frac=1.0 + elapsed -42.5%` を達成した設定を 19 本撚線 (Type D stall 本体) に適用、**`frac=0.3723`（baseline `0.4839` 比 **-23.1% 退化**）で却下**。δ_h 2x 拡大は Type D stall 領域で逆効果（NR 内訳 D+E:72%）、`StrandBendingOscillationConfig.smoothing_delta` の default 変更は**実施せず**、`work/beam_hysteresis/16_hypothesis_c_aprime_19strand.py` を**失敗実験記録**として残置（`15_hypothesis_c_7strand.py` 7 本成功実験と対称）。次候補は **(c) line search 強化**（`_newton_dynamic.py` に backtracking hook）。副次: **Phase E C21/C22/C23 追加** — C21 `TermExpansionContract.term_names` 重複静的検出（`__post_init__` に runtime ガード + 静的検査）、C22 `contracts` ClassVar 同名契約重複検出、C23 `@verified_by` 検証 Process が `SolverProcess` / `VerifyProcess` いずれかの継承必須（`bind_verifier` に runtime ガード + 静的検査）。`test_duplicate_term_names_rejected` + `test_bind_invalid_category_rejected` 2 テスト追加で mathematics/tests 97 passed。gate: 契約違反 **0 件**（C18〜C23 含む全 23 検査 OK）/ 条例違反 **0 件** / `pytest xkep_cae/contact/` 421 passed 5 skipped / 7本撚線曲げ揺動 1 passed / ruff check + format pass。Phase A〜E / status-346〜360 の **12/N 完了** | **仮説 C 候補 (a') smoothing_delta=1000 7本撚線 90° 採択（elapsed -42.5%、frac=1.0 完走）**（[status-359](docs/status/status-359.md)）— status-358 の最優先 TODO「候補 (a') 中間値再試行」に対応。`work/beam_hysteresis/15_hypothesis_c_7strand.py` の `smoothing_delta=500.0` を **`1000.0`**（default 2000 の 1/2、δ_h 2x 拡大）に書き換え 7本撚線 90° 曲げで実測。**frac=1.0000 完走 + n_increments=475（-9.4%）+ n_cutbacks=53（-7.0%、10% 未満）+ elapsed=259.92s（-42.5%、1.74x 高速化）**。ユーザー指示「frac=1.0 完走 + 10% 以上改善」に対し elapsed -42.5% で大幅クリア（cutback は補助指標で 10% 未満だが elapsed 半減近い改善は active flip 抑制で各 increment の NR 反復数が減った効果として十分）。**判定: 採択方向（実験記録）**。`StrandBendingOscillationConfig.smoothing_delta` の default 変更（2000→1000）は **本 status では実施せず**（7本撚線のみの検証で 19 本 Type D stall 本体への有効性未検証）、`15_hypothesis_c_7strand.py` を成功実験記録として残置（status-358 の (a) 失敗実験 revert と対称）、実装本体（`xkep_cae/`、`tests/`、`contracts/`）は **無変更**。次セッション最優先 TODO は (i) 仮説 C (a') の 19 本撚線検証 → (ii) default 化判断 / 失敗時 (c) line search 強化。余談: ユーザーから「梁の積分点ごと相当塑性ひずみ保持 / 純粋弾性と収束悪化の関係 / 塑性・粘性導入の収束改善見込み」について Q&A 議論あり、status-359 §引継ぎに記録（標準 CR 梁は弾性のみ / ファイバー梁 `Fiber1DState.eps_p` で保持 / 塑性散逸導入は MCDD 完了後の凍結 TODO）。gate: 契約違反 **0 件**（C18/C19/C20 含む 20 検査 OK）/ 条例違反 **0 件** / 7本撚線曲げ揺動 1 passed in 10.85s / ruff check + format pass。Phase A〜E / status-346〜359 の **11/N 完了** | **仮説 C 候補 (a) 7本撚線 90° 実測 → 未完走で却下 + C20 双方向紐付け検査追加**（[status-358](docs/status/status-358.md)）— status-357 の最優先 TODO（仮説 C 立案 + Phase E 仕上げ）に対応。(1) **仮説 C 候補 (a)**（`smoothing_delta` 遷移帯 4x 拡大、default 2000→500）を 7本撚線 90° 曲げで実測。ベースライン（frac=1.0000, incr=524, cb=57, 452.02s, チャタリング 166 件）に対し候補 (a) は **frac=0.9241 で未完走**、cutback -14% / elapsed -17% の見かけ改善は解析の早期打切り。ユーザー指示「10% 以上改善 + frac=1.0 完走」基準未達で**却下（revert）**、コード変更なし。`work/beam_hysteresis/15_hypothesis_c_7strand.py` は失敗実験の記録として残置。次候補は (a') `smoothing_delta=1000`（2x 拡大中間値）再試行 or (c) line search 強化。(2) **Phase E C20 追加**: `TermExpansionContract` の `providers` に列挙された Process クラスが自身の `contracts` ClassVar で同名契約を宣言していることを静的検査。C18/C19 の片側更新による脱法すり抜けを防御、5 既存 providers で回帰なし。gate: 契約違反 **0 件**（C18/C19/C20 含む 20 検査 OK）/ 条例違反 **0 件** / `pytest xkep_cae/contact/` 421 passed 5 skipped（48.38s）/ 7本撚線曲げ揺動 1 passed 15.34s / ruff check + format pass。Phase A〜E / status-346〜358 の **10/N 完了** | **19 本撚線 K_c FD 再計測 + C5 解消 + C18/C19 契約検査追加（Phase E 着手）**（[status-357](docs/status/status-357.md)）— status-356 Phase C-3' 実装の実機規模検証: 19 本撚線 FD 再計測で **frac=0.3739（status-344 比 -22.7% 退化）、mat_only rel_err mean=0.508（+15% 悪化）**。gate テスト `test_helical_3d_hermite` での FD 機械精度達成は **active 集合固定下の解析的 K_c**（status-356 §7 2 経路相殺定理）に限定され、19 本 Type D stall（NR Type D+E:67%, E:28%）のような active 集合振動支配領域には波及しないと判定。仮説 C（active 集合振動対策）を status-358 最優先に昇格。副次: status-356 で混入していた **C5 違反**（`KcHermiteNonlocalStiffnessProcess.process()` の `HuberContactForceProcess._batch_dm_ext_coeffs` クラスメソッド直接参照）を `_batch_dm_ext_coeffs` のモジュール関数昇格で解消。Phase E 着手: **C18**（`@verified_by` 紐付け検査、MCDD 脱法 pattern 2 前段）+ **C19**（`TermExpansionContract.providers` 実在検査、pattern 4 対策）を `contracts/validate_process_contracts.py` に追加、5 term-provider Process に `@verified_by("K_c_term_expansion", ContactKcComponentFDDiagnosticProcess)` 付与。gate: 契約違反 **0 件**（C18/C19 含む 19 検査 OK）/ 条例違反 **0 件** / `pytest xkep_cae/contact/` 421 passed 5 skipped（39.65s）/ 7本撚線曲げ揺動 1 passed 10.69s / ruff check + format pass。Phase A〜E / status-346〜358 の **9/13 完了** | **Phase C-3' 仮説 B 実装 — K_closest/K_st 隣接拡張 + K_hermite_adj フル項化で FD 機械精度一致**（[status-356](docs/status/status-356.md)）— status-354 仮説 A（`K_hermite_adj` フル項 = 直接経路 (i)）と status-355 仮説 B（`K_closest`/`K_st` の active×adj 拡張 = s-tracking 経路 (ii)）を**同時導入**して 2 経路の $P_\perp$ 成分を解析的に相殺。結果: `test_helical_3d_hermite` **rel_err 1.795% → 2.18e-07**（5 桁改善）、`||diff[ax]|| 98.52 → 4.75e-05`（6 桁改善、status-355 目標を約 5 桁オーバーシュート）、comp_z 77.3% → 1.16e-05。status-354 の「mat-only 最良」解釈は (ii) 未実装時のワークアラウンドと訂正、数理台帳 §7 を §7.1 2 経路解析 / §7.2 相殺定理 / §7.3 status-354 反証 ⇒ status-356 解決 / §7.4 診断裏付け に再構成。実装: `strategy.py` `_batch_dm_ext_coeffs` 抽出 + `ContactForceStStiffnessInput.adj_node_counts` 追加 + `_process_batch_term` の `term in {"closest","residual"}` 両経路で active×adj COO 追加 + `KcHermiteNonlocalStiffnessProcess` の `K_3x3_mat` フル項化（`w_mat n⊗n - w_geo I_nn`）。gate: 契約違反 **0 件** / 条例違反 **0 件** / `pytest xkep_cae/contact/` 421 passed 5 skipped / `test_kc_component_fd.py` 7 passed（`test_helical_3d_hermite` rel_err=2.18e-07）/ 7本撚線曲げ揺動回帰 frac=1.0000, 10.18s / ruff check + format pass。Phase A〜E / status-346〜358 の **8/13 完了**（次は status-357 で 19 本撚線 FD 再計測 + Type D stall 再試行） | **Phase C-3' 仮説 B 診断**（[status-355](docs/status/status-355.md)）— `work/beam_hysteresis/14_kc_closest_adj_diagnostic.py` 新設、`test_helical_3d_hermite` シナリオで `diff = K_c_analytical - FD_Kc` を 4 ブロック (active/adj×active/adj) に分解。**rel_err 1.795% の 100% が active×adj ブロックに局在**（aa rel_err=2.2e-7、ax ||diff||=98.52、xa/xx=0）。comp_z 77% は adj 列 z (76.11) そのもの。`||FD[ax]||=601.5` vs `||K_c[ax]||=593.4` で K_hermite_adj が一部埋めるも 16.4% 不足、**98.52 が仮説 B で埋めるべき解析量と一致**。実装コスト評価 ~45 行（`_batch_st_jacobian_hermite` 既存 `ds_du_adj` 活用、`adj_node_counts` 追加、`term="closest"` adj 列分岐）、公開 API 非破壊。status-356 で本実装 → `||diff[ax]||<1e-3` + 19 本 frac=0.48→1.0 目標。コード変更なし診断 status（MCDD 禁止パターン 6 非該当 — 定量目標と実装パス確立） | **Phase C-3 再定義実験 — K_hermite_adj フル項拡張の仮説 A 反証**（[status-354](docs/status/status-354.md)）— status-353 提示の仮説 A（「`KcHermiteNonlocalStiffnessProcess` に `-w_geo * I_nn` の隣接ノード項を追加すると 19 本撚線 `mat_only` rel_err 44% が改善する」）を直接実験して反証。`test_kc_component_fd.py::test_helical_3d_hermite` の rel_err が **1.795% → 38.49%** に 21 倍悪化し（MCDD 脱法パターン 5 該当）、変更を全て revert し mat-only 継続（status-295）。反証の数理的解釈: 隣接ノード摂動は (1) Hermite 接線経由の直接 `p_A` 変化 + (2) min-distance 射影での s-tracking 補償の 2 経路を持ち、`I_nn`（法線直交）方向は (2) でほぼ相殺されるため Process 側に追加すると FD との乖離が拡大。Phase C-3 を **Phase C-3' 再々定義**（hypothesis B/C/D）へ再配分: **仮説 B**（`KcClosestPointStiffnessProcess` の隣接ノード拡張で s-tracking 経路を解析的実装）を最有力に昇格。数理台帳 03 章 §7/§3.1/§4/§8 に仲裁追記、`strategy.py` モジュールコメント + `KcHermiteNonlocalStiffnessProcess` docstring に実測結果を記録（実装変更なし、mat-only 維持）。gate: 契約違反 **0 件** / 条例違反 **0 件** / `pytest xkep_cae/contact/` 421 passed 5 skipped / `test_kc_component_fd.py` 7 passed（`test_helical_3d_hermite` rel_err=1.795% 合格）/ 7本撚線曲げ揺動回帰（弱曲げスモーク）frac=1.0000, incr=51, cutback=4, 10.54s 完走。Phase A〜E / status-346〜358（さらに 1 status 後ろ倒し）の **7/13 完了**（仮説 A 反証という定量結果を伴う Phase C-3 再定義として記録） | 数理台帳訂正詳細は [status-353](docs/status/status-353.md)— status-352 の中断スナップショットで提示した「Phase C-3 前提疑義」に対し**選択肢 A（数理台帳 §4 訂正）を実施**。`docs/math/03_huber_contact_penalty.md` の §3 / §3.1 / §4 / §5 / §8 を訂正、A-A 同側ペア局所導出から $\boldsymbol{K}_{\mathrm{geo}} = -p_n\,\partial\hat n/\partial u$ のペア局所形そのもの（$1/d$ は $\hat n = r/d$ 内在項）であることを証明し、`KcGeoStiffnessProcess` が法線方向感度を担うことを確立。「`K_mat_ndir` 独立追加」の当初 Phase C-3 計画を**撤回**、5 項 `TermExpansionContract` で完結。`strategy.py` モジュールコメント / `KcNormalStiffnessProcess` / `KcGeoStiffnessProcess` docstring 訂正（実装変更なし）。19本撚線 Type D stall (`mat_only` rel_err mean=44%, comp_x max=98%) の真の原因候補を **`K_hermite_adj` の mat-only 近似（`w_mat * nn` のみ、status-295 で意図的に `I_nn` 隣接拡張除外）** に再設定、Phase C-3 を「`K_hermite_adj` フル項拡張」に再定義（status-354 着手予定）。gate: 契約違反 **0 件** / 条例違反 **0 件** / `pytest xkep_cae/contact/` 421 passed 5 skipped / MCDD 関連 114 passed / 7本撚線曲げ揺動回帰（**κ=0.001, bending_angle≈5.73°, 接触未活性スモーク**, max contact F=0）frac=1.0000, 10.20s 完走 / Hertz 同条件 9.96s / **接触あり 90° 曲げ（status-298/299 系）は本 status 未実行（挙動無変更の訂正のため、status-354 で実施予定）**。Phase A〜E / status-346〜357（1 status 後ろ倒し）の **7/12 完了** | 中断スナップショット詳細は [status-352](docs/status/status-352.md) —  (1) `/root/.claude/plans/deep-wiggling-seal.md` **永久ロスト**を確認、`CLAUDE.md` / `docs/roadmap.md` / `docs/math/README.md` / `xkep_cae/mathematics/registry.py` / `xkep_cae/mathematics/docs/mathematics.md` の計画書参照を「永久ロスト」表記に更新（status-346〜351 はヒストリカルとして改変せず）。(2) Phase C-3「`KcNormalDirectionStiffnessProcess` 新設」の前提を数理的に再検証。`HuberContactForceProcess.tangent()` (`strategy.py:1595`) のペア局所形 `w_mat · nn − w_geo · I_nn` を直接導出し、**現行の `K_geo` が既に $-p_n \cdot \partial\hat{n}/\partial u$ と同一のテンソル形 `-cc · (p_n/d) · (I - \hat{n}\hat{n}^T)`** を返すことを確認。新 Process 追加は K_mat,ndir の **二重計上**で `test_kc_component_fd.py` 19 件を fail させるリスクが高く、数理台帳 §4 の訂正（選択肢 A）または UL/回転 DOF 寄与の再導出（選択肢 B）を先行すべきと判定、実装は次セッションに引き継ぎ（脱法パターン 8 回避）。19本撚線 Type D stall の再原因候補は `K_hermite_adj` mat-only の I_nn 隣接拡張漏れ等に再設定。テスト数変動なし、契約違反 **0件** 維持、既存 skip/xfail 増加 0。Phase A〜E / status-346〜356 の **6/11 完了 維持**（中断スナップショット） | 契約違反 **0件** | 条例違反 **0件** | [ロードマップ](docs/roadmap.md) | [ステータス一覧](docs/status/status-index.md)

| 分野 | 概要 |
|------|------|
| FEM基盤 | 梁（EB/Timo/CR/Cosserat）+ 非線形 + 動的解析 |
| 接触 | NCP + Line contact + smooth penalty Coulomb摩擦 |
| 撚線 | 7本摩擦曲げ+揺動収束、被膜+シース |
| アーキテクチャ | Process Architecture + Strategy Protocol + BenchmarkRunner |

**推奨ソルバー構成**: `contact_mode="smooth_penalty"` + NCP + 同層除外（[詳細](docs/roadmap.md#推奨ソルバー構成)）

## パッケージ構成

```
xkep_cae/
├── core/              # プロセスアーキテクチャ基盤（base, registry, runner 等）
│   ├── strategies/    # Strategy Protocol 定義
│   └── batch/         # BatchProcess（ワークフローオーケストレーション）
├── mathematics/       # MCDD 基盤（status-346〜）: MathematicalContract 型 + ProcessContractRegistry + @verified_by
├── contact/           # 接触モジュール
│   ├── penalty/       # PenaltyStrategy + 法線力 Process
│   ├── friction/      # FrictionStrategy + return mapping
│   ├── coating/       # CoatingStrategy + Kelvin-Voigt
│   ├── contact_force/ # ContactForceStrategy
│   ├── geometry/      # ContactGeometryStrategy
│   ├── setup/         # ContactSetupProcess
│   └── solver/        # ContactFrictionProcess + NUzawa
├── time_integration/  # TimeIntegrationStrategy（準静的/動的）
├── elements/          # 要素（CR梁/UL梁アセンブラ）
├── mesh/              # メッシュ生成（撚線メッシュ）
├── numerical_tests/   # 数値試験フレームワーク
├── output/            # 出力（CSV/JSON/VTK/GIF）
├── verify/            # 検証 Process（収束/エネルギー/接触）
└── tuning/            # チューニング
```

## ドキュメント

| ドキュメント | 内容 |
|------------|------|
| [ロードマップ](docs/roadmap.md) | 全体計画・マイルストーン・TODO |
| [ステータス一覧](docs/status/status-index.md) | 全statusファイル + テスト数推移 |
| [設計文書一覧](docs/design/README.md) | 設計仕様書リンク集 |
| [MCDD 設計仕様](xkep_cae/mathematics/docs/mathematics.md) | 数理契約駆動開発の型システム設計（status-346〜）|
| [数理台帳（docs/math/）](docs/math/README.md) | 離散化方程式の単一のソース・オブ・トゥルース（6 章 / 55 アンカー、status-348〜349）|
| [撚線ファイバー梁 設計仕様](xkep_cae/elements/docs/fiber_beam_strand.md) | 内部摩擦ヒステリシスを1本の梁で等価化する設計 |
| [beam_hysteresis 概念検証](work/beam_hysteresis/README.md) | 上記設計の裏付け数値実験 |

## インストール

```bash
pip install -e ".[dev]"
```

## テスト実行

```bash
# 高速テストのみ（~3分）
pytest tests/ -v -m "not slow and not external"

# 全テスト（~30分, slow含む）
pytest tests/ -v -m "not external"

# 新パッケージテストのみ
pytest xkep_cae/ -v
```

## Lint / Format

```bash
ruff check xkep_cae/ tests/
ruff format xkep_cae/ tests/
```

## ライセンス

[MIT License](LICENSE)

## 運用

本プロジェクトはCodexとClaude Codeの2交代制で運用。
引き継ぎ情報は [docs/status/](docs/status/status-index.md) を参照。
