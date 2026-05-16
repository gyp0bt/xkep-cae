# ステータス一覧（status-index）

[← README](../../README.md) | [← roadmap](../roadmap.md)

> 本ファイルはステータスファイルの一覧メモです。新規 status 作成時に必ず更新すること。

## アクティブ status（377〜 — explicit 時間積分・foundation 再検証・ε フェーズ）

| # | 日付 | タイトル |
|---|------|---------|
| [377](status-377.md) | 2026-04-28 | 陽的中央差分 Phase 1 — Process 単体実装 + `solver_mode` config + 設計仕様（28 unit）|
| [378](status-378.md) | 2026-04-29 | 陽的中央差分 Phase 2 — solver path 配線 + Courant 監視（7 本 smoke で β=3×10⁵ 障壁実測）|
| [379](status-379.md) ⚠️ status-380 で撤回 | 2026-04-30 | mass scaling auto-tune — 19 本 frac=1.0 完走、ただし数値発散判定漏れで撤回 |
| [380](status-380.md) | 2026-04-30 | 物理的妥当性検証 — 7本/19本とも max\|u\|=1.59×10⁸mm 発散、status-379 撤回 + 凍結解除条件 (3) 追加 |
| [381](status-381.md) | 2026-05-01 | mass scaling bug fix — 発散停止、explicit 解は解析解の 50% で精度 gate 未達、凍結解除条件 (5) 追加 |
| [382](status-382.md) | 2026-05-01 | 候補 (p3) damping + (p1) relax API — UL update_reference 凍結が真因と判明 |
| [383](status-383.md) | 2026-05-01 | 候補 (q1) `explicit_ul_update_interval` — 全 4 ケース FAIL、UL 凍結再確証 |
| [384](status-384.md) | 2026-05-01 | 候補 (z1a) 要素ごと波速 Δt + (z1b) selective mass scaling 実装 |
| [385](status-385.md) | 2026-05-01 | 候補 (z1c) 2 段階質量スケーリング API — β_stiff cap 支配で精度 gate 未達 |
| [386](status-386.md) | 2026-05-01 | 候補 (z1d) `t_cycle` 下限緩和 — 方向自体が逆と単梁で実証 |
| [387](status-387.md) ⚠️ status-388 で撤回 | 2026-05-02 | 【撤回】単梁 90° 曲げ n_inc=8000 sweet spot 偽陽性、解析解取り違え |
| [388](status-388.md) | 2026-05-02 | 透明性ルール策定 — 独立解析解 3 個 AND gate 必須化、status-387 撤回 + 単梁 explicit + UL 大破綻 |
| [389](status-389.md) | 2026-05-02 | 引き継ぎ — 梁要素 1 つから foundation 系統的再検証 Phase 計画策定 |
| [390](status-390.md) | 2026-05-02 | Phase α 完了 — 1 要素 implicit static 全 4 ケース PASS（機械精度）|
| [391](status-391.md) | 2026-05-05 | Phase β 完了 — 1 要素 explicit dynamic 自由振動 + quasi-static 両 PASS |
| [392](status-392.md) | 2026-05-06 | Phase γ-1 完了 — multi-element implicit で O(1/n²) 収束実証（slope=-2.000）|
| [393](status-393.md) | 2026-05-06 | 達成確認マトリクス `verification_matrix.md` 導入 — STA2 連鎖撤回の構造的予防 |
| [394](status-394.md) | 2026-05-08 | assembler / UL 1 要素再現実験 — 改修対象を explicit + UL のみ に局在化（4 モード A/B/C PASS, D FAIL）|
| [395](status-395.md) | 2026-05-08 | Phase γ-3 完了 — 多要素 explicit + TL で arc 収束を O(1/n²) 再現（slope=-2.000、γ-1 数値一致）|
| [396](status-396.md) | 2026-05-09 | explicit-TL 固定 API 化 — `explicit_ul_disable_update` 独立フィールド追加（候補 (z3) Phase 1）|
| [397](status-397.md) | 2026-05-10 | ε-1 失敗 — `_process_free_end` × explicit-TL を 1 strand で再現、改修対象を driver 層に局在化 |
| [398](status-398.md) | 2026-05-11 | 3 仮説切り分け診断 — hypothesis 1（stepwise BC × mass scaling auto-tune interaction）確定 |
| [399](status-399.md) | 2026-05-12 | `explicit_n_sub_cycles_per_increment` 実装 — ε-1 N=2000 で rel_err 0.01% asymptote PASS |
| [400](status-400.md) | 2026-05-16 | `VtkExportProcess` 実装 — ParaView 用 VTK XML 出力 PostProcess（依存追加なし、汎用 1D 梁モデル対応、+11 テスト）|

## アーカイブ（275〜376 — 接触完走・MCDD Phase A〜E・NR escape hatch 全候補検証）

status-275〜376 は [archive/](archive/) に移動済み（status-400 で実施）。

| # | 日付 | マイルストーン | テスト数 |
|---|------|--------------|---------|
| [280](archive/status-280.md) | 2026-04-02 | free_end_mode 実装 — 7本撚線 frac 0.55→1.0 完走 | 602 |
| [285](archive/status-285.md) | 2026-04-03 | C16 修正 + Hertz 型非線形ペナルティ — frac 0.998 | 621 |
| [291](archive/status-291.md) | 2026-04-04 | K_c 不整合根本特定 + s_unclamped 修正（Hermite 20%→0.0001%） | 624+ |
| [292](archive/status-292.md) | 2026-04-04 | StJacobian 2×2 カップリング修正（K_st FD 94%→0.0001%） | 631 |
| [298](archive/status-298.md) | 2026-04-06 | Hertz + atol_force frac=1.0 完走確認 | 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4 |
| [301](archive/status-301.md) | 2026-04-07 | 19 本撚線 frac=1.0 達成（implicit baseline） | 200+10s+16+3+23+1+6+18+2+4+3+9+4+18+2+2+2+15+4 |
| [305](archive/status-305.md) | 2026-04-08 | バリア被膜 90 度曲げ — incr 42% 削減 / 70% 高速化 | 200+...+15+4+13 |
| [311](archive/status-311.md) | 2026-04-08 | adj batch + BC 20000x + pypardiso 統合 | 200+...+13+14+6+9+3 |
| [325](archive/status-325.md) | 2026-04-12 | `_SolverCache` + symbolic factorization | 200+...+5+8+12 |
| [330](archive/status-330.md) | 2026-04-14 | ファイバー梁 F5 統合 — TL + BilinearKH + MultiLayer | 459+13+22+5+8+12+12+25+26+10 |
| [344](archive/status-344.md) | 2026-04-16 | 19 本 Type D stall — K_c x/z カップリング不整合特定 | 459+...+15+10+9+8+12 |
| [346](archive/status-346.md) | 2026-04-16 | MCDD Phase A-1 — `MathematicalContract` 5 種 + `TermExpansionContract` | 459+...+33 |
| [349](archive/status-349.md) | 2026-04-17 | MCDD Phase B-2 — 6 章 / 55 アンカー + `equation_index.py` + C15 拡張 | 459+...+33+33+21+8 |
| [353](archive/status-353.md) | 2026-04-20 | 数理台帳訂正 — `K_mat,ndir ≡ K_geo` 確立、5 項完結化 | 459+...+33+33+21+8+14 |
| [356](archive/status-356.md) | 2026-04-21 | Phase C-3' 仮説 A+B 同時導入 — FD 機械精度（rel_err 1.8%→2e-7） | 459+...+33+33+21+8+25 |
| [357](archive/status-357.md) | 2026-04-21 | Phase E 着手 + C18/C19 契約検査追加、19 本 Type D stall は active 振動支配と判定 | 459+...+25 |
| [364](archive/status-364.md) | 2026-04-23 | Phase E C24 — hollow VerifyProcess 構造的封じ込め | 459+...+25+6+12 |
| [367](archive/status-367.md) | 2026-04-23 | 候補 (e) 接触減衰 — 7本 elapsed -57%、19 本却下 | 459+...+25+6+12 |
| [372](archive/status-372.md) | 2026-04-25 | 候補 (g1) active 履歴 EMA — 7本 cb -75%、19 本却下 | 459+...+25+6+12+12+7+10 |
| [373](archive/status-373.md) | 2026-04-25 | TODO 整理 + 症状緩和系 experiment 5 本削除 + solver_mode 設計追記 | 459+...+25+6+12+12+7+10 |
| [375](archive/status-375.md) | 2026-04-26 | 候補 (g3) pair-wise relaxation Phase 2 + 19 本却下 | 459+...+25+6+12+12+7+10+12+11 |
| [376](archive/status-376.md) | 2026-04-28 | 候補 (g2) AL 外側ループ — n=2 で +53.7% 改善も gate 未達、NR escape hatch 路線終了 | 459+...+11+34 |

## アーカイブ（175〜274 — 新 xkep_cae R1 完遂・NR 収束改善・Hermite 非局所対応）

status-175〜274 は [archive/](archive/) に移動済み（status-322 で実施）。

| # | 日付 | マイルストーン | テスト数 |
|---|------|--------------|---------|
| [175](archive/status-175.md) | 2026-03-15 | 脱出ポット計画 Phase 1 — xkep_cae リネーム + PenaltyStrategy 書き直し | ~2260+34p |
| [188](archive/status-188.md) | 2026-03-16 | R1 Phase 7 完了 — C14/C16 違反ゼロ | ~2260+284p |
| [207](archive/status-207.md) | 2026-03-18 | deprecated コード完全削除 + コンテキスト大掃除 | 248p |
| [222](archive/status-222.md) | 2026-03-21 | Huber ペナルティ統一（ソルバー一本化） | 499 |
| [226](archive/status-226.md) | 2026-03-22 | K_st 実装 — ∂(s,t)/∂u 整合接線 + FD 検証 11 件 | 175+11 |
| [253](archive/status-253.md) | 2026-03-26 | DOF 消去 MPC 実装 + StrandBendingOscillation | 200+10s+16+3+23+1+6+18 |
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
Phase 1 完了:           16             (2026-02-12)
Phase 5 完了:           615            (2026-02-18)
Phase C0-C5:            993            (2026-02-21)
Phase C6:               1850           (2026-02-28)
R1 Phase 7:             2477+314p      (2026-03-13)
新 xkep_cae 開始:        ~2260+34p      (2026-03-15)  ← status-175
deprecated 全削除:      248p           (2026-03-18)  ← status-207
Huber 統一:              499            (2026-03-21)  ← status-222
Hermite 非局所完了:      200+10s+...+3 (2026-03-31)  ← status-274
Hertz + frac=1.0:        ...+15+4       (2026-04-06)  ← status-298
バリア被膜 90 度曲げ:    ...+15+4+13    (2026-04-08)  ← status-305
ファイバー梁 F5 統合:    459+13+22+5+8+12+12+25+26+10  (2026-04-14)  ← status-330
MCDD Phase A-1:          ...+33         (2026-04-16)  ← status-346
MCDD Phase E 着手:       ...+25         (2026-04-21)  ← status-357
NR escape hatch 終了:    ...+11+34      (2026-04-28)  ← status-376
陽解法 Phase 2:          ...+10+11      (2026-04-29)  ← status-378
Phase α 完了 (1 要素):   ...+12         (2026-05-02)  ← status-390
Phase γ 完了 (多要素):   ...+12         (2026-05-06)  ← status-392
sub-cycle 実装 (ε-1):    459+...+12 計  (2026-05-12)  ← status-399
```

## 備考

- テスト数「—」はドキュメント更新・計画策定のみのステータス
- status-001〜096 は archive/ に移動（status-100 で実施）
- status-097〜174 は archive/ に移動（status-177 で実施）
- status-175〜274 は archive/ に移動（status-322 で実施）
- status-275〜376 は archive/ に移動（**status-400 で実施**、本セッション）
- status-377〜 がアクティブ status（explicit 時間積分・foundation 再検証・ε フェーズ）
- **アーカイブ方針**: アクティブ 50 件超過時に最古バッチを archive/ へ移動

---

[← README](../../README.md) | [← roadmap](../roadmap.md)
