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

**459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25 テスト** — 2026-04-20時点 | **Phase C-3' 仮説 B 診断 — K_closest 隣接拡張で埋めるべき量を active×adj ブロックに局在化**（[status-355](docs/status/status-355.md)）— `work/beam_hysteresis/14_kc_closest_adj_diagnostic.py` 新設、`test_helical_3d_hermite` シナリオで `diff = K_c_analytical - FD_Kc` を 4 ブロック (active/adj×active/adj) に分解。**rel_err 1.795% の 100% が active×adj ブロックに局在**（aa rel_err=2.2e-7、ax ||diff||=98.52、xa/xx=0）。comp_z 77% は adj 列 z (76.11) そのもの。`||FD[ax]||=601.5` vs `||K_c[ax]||=593.4` で K_hermite_adj が一部埋めるも 16.4% 不足、**98.52 が仮説 B で埋めるべき解析量と一致**。実装コスト評価 ~45 行（`_batch_st_jacobian_hermite` 既存 `ds_du_adj` 活用、`adj_node_counts` 追加、`term="closest"` adj 列分岐）、公開 API 非破壊。status-356 で本実装 → `||diff[ax]||<1e-3` + 19 本 frac=0.48→1.0 目標。コード変更なし診断 status（MCDD 禁止パターン 6 非該当 — 定量目標と実装パス確立） | **Phase C-3 再定義実験 — K_hermite_adj フル項拡張の仮説 A 反証**（[status-354](docs/status/status-354.md)）— status-353 提示の仮説 A（「`KcHermiteNonlocalStiffnessProcess` に `-w_geo * I_nn` の隣接ノード項を追加すると 19 本撚線 `mat_only` rel_err 44% が改善する」）を直接実験して反証。`test_kc_component_fd.py::test_helical_3d_hermite` の rel_err が **1.795% → 38.49%** に 21 倍悪化し（MCDD 脱法パターン 5 該当）、変更を全て revert し mat-only 継続（status-295）。反証の数理的解釈: 隣接ノード摂動は (1) Hermite 接線経由の直接 `p_A` 変化 + (2) min-distance 射影での s-tracking 補償の 2 経路を持ち、`I_nn`（法線直交）方向は (2) でほぼ相殺されるため Process 側に追加すると FD との乖離が拡大。Phase C-3 を **Phase C-3' 再々定義**（hypothesis B/C/D）へ再配分: **仮説 B**（`KcClosestPointStiffnessProcess` の隣接ノード拡張で s-tracking 経路を解析的実装）を最有力に昇格。数理台帳 03 章 §7/§3.1/§4/§8 に仲裁追記、`strategy.py` モジュールコメント + `KcHermiteNonlocalStiffnessProcess` docstring に実測結果を記録（実装変更なし、mat-only 維持）。gate: 契約違反 **0 件** / 条例違反 **0 件** / `pytest xkep_cae/contact/` 421 passed 5 skipped / `test_kc_component_fd.py` 7 passed（`test_helical_3d_hermite` rel_err=1.795% 合格）/ 7本撚線曲げ揺動回帰（弱曲げスモーク）frac=1.0000, incr=51, cutback=4, 10.54s 完走。Phase A〜E / status-346〜358（さらに 1 status 後ろ倒し）の **7/13 完了**（仮説 A 反証という定量結果を伴う Phase C-3 再定義として記録） | 数理台帳訂正詳細は [status-353](docs/status/status-353.md)— status-352 の中断スナップショットで提示した「Phase C-3 前提疑義」に対し**選択肢 A（数理台帳 §4 訂正）を実施**。`docs/math/03_huber_contact_penalty.md` の §3 / §3.1 / §4 / §5 / §8 を訂正、A-A 同側ペア局所導出から $\boldsymbol{K}_{\mathrm{geo}} = -p_n\,\partial\hat n/\partial u$ のペア局所形そのもの（$1/d$ は $\hat n = r/d$ 内在項）であることを証明し、`KcGeoStiffnessProcess` が法線方向感度を担うことを確立。「`K_mat_ndir` 独立追加」の当初 Phase C-3 計画を**撤回**、5 項 `TermExpansionContract` で完結。`strategy.py` モジュールコメント / `KcNormalStiffnessProcess` / `KcGeoStiffnessProcess` docstring 訂正（実装変更なし）。19本撚線 Type D stall (`mat_only` rel_err mean=44%, comp_x max=98%) の真の原因候補を **`K_hermite_adj` の mat-only 近似（`w_mat * nn` のみ、status-295 で意図的に `I_nn` 隣接拡張除外）** に再設定、Phase C-3 を「`K_hermite_adj` フル項拡張」に再定義（status-354 着手予定）。gate: 契約違反 **0 件** / 条例違反 **0 件** / `pytest xkep_cae/contact/` 421 passed 5 skipped / MCDD 関連 114 passed / 7本撚線曲げ揺動回帰（**κ=0.001, bending_angle≈5.73°, 接触未活性スモーク**, max contact F=0）frac=1.0000, 10.20s 完走 / Hertz 同条件 9.96s / **接触あり 90° 曲げ（status-298/299 系）は本 status 未実行（挙動無変更の訂正のため、status-354 で実施予定）**。Phase A〜E / status-346〜357（1 status 後ろ倒し）の **7/12 完了** | 中断スナップショット詳細は [status-352](docs/status/status-352.md) —  (1) `/root/.claude/plans/deep-wiggling-seal.md` **永久ロスト**を確認、`CLAUDE.md` / `docs/roadmap.md` / `docs/math/README.md` / `xkep_cae/mathematics/registry.py` / `xkep_cae/mathematics/docs/mathematics.md` の計画書参照を「永久ロスト」表記に更新（status-346〜351 はヒストリカルとして改変せず）。(2) Phase C-3「`KcNormalDirectionStiffnessProcess` 新設」の前提を数理的に再検証。`HuberContactForceProcess.tangent()` (`strategy.py:1595`) のペア局所形 `w_mat · nn − w_geo · I_nn` を直接導出し、**現行の `K_geo` が既に $-p_n \cdot \partial\hat{n}/\partial u$ と同一のテンソル形 `-cc · (p_n/d) · (I - \hat{n}\hat{n}^T)`** を返すことを確認。新 Process 追加は K_mat,ndir の **二重計上**で `test_kc_component_fd.py` 19 件を fail させるリスクが高く、数理台帳 §4 の訂正（選択肢 A）または UL/回転 DOF 寄与の再導出（選択肢 B）を先行すべきと判定、実装は次セッションに引き継ぎ（脱法パターン 8 回避）。19本撚線 Type D stall の再原因候補は `K_hermite_adj` mat-only の I_nn 隣接拡張漏れ等に再設定。テスト数変動なし、契約違反 **0件** 維持、既存 skip/xfail 増加 0。Phase A〜E / status-346〜356 の **6/11 完了 維持**（中断スナップショット） | 契約違反 **0件** | 条例違反 **0件** | [ロードマップ](docs/roadmap.md) | [ステータス一覧](docs/status/status-index.md)

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
