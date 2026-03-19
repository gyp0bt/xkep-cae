# ステータス一覧（status-index）

[← README](../../README.md) | [← roadmap](../roadmap.md)

> 本ファイルはステータスファイルの一覧メモです。新規status作成時に必ず更新すること。

## アクティブ status（175〜 — 新 xkep_cae）

| # | 日付 | タイトル | テスト数 |
|---|------|---------|---------|
| [175](status-175.md) | 2026-03-15 | 脱出ポット計画 Phase 1 — xkep_cae リネーム + PenaltyStrategy 完全書き直し | ~2260+34p(新) |
| [176](status-176.md) | 2026-03-15 | C16 純粋関数違反の追加 | ~2260+34p(新) |
| [177](status-177.md) | 2026-03-15 | ドキュメント再編 — 新xkep_cae用にドキュメント全体を再構成 | ~2260+34p(新) |
| [178](status-178.md) | 2026-03-15 | モジュール再編 + FrictionStrategy 移行 | ~2260+86p(新) |
| [179](status-179.md) | 2026-03-15 | Phase 2 後半 — Strategy 全移行 + 契約違反ゼロ | ~2260+186p(新) |
| [180](status-180.md) | 2026-03-15 | C16 契約ギャップ修正 — __init__.py re-export チェック強化 | ~2260+186p(新) |
| [181](status-181.md) | 2026-03-15 | Penalty/Coating ファクトリ完備 — default_strategies() 7軸全生成 | ~2260+204p(新) |
| [182](status-182.md) | 2026-03-16 | C16 スコープ拡大 + time_integration 移動 + process/ 削除 | ~2260+204p(新) |
| [183](status-183.md) | 2026-03-16 | Phase 3 — concrete プロセス移行 + BatchProcess フル実装 | ~2260+266p(新) |
| [184](status-184.md) | 2026-03-16 | Phase 4 — ContactFrictionProcess 移行 + 完全ワークフロー実現 | ~2260+279p(新) |
| [185](status-185.md) | 2026-03-16 | Phase 5 — ソルバー結果連携 + output re-export クリーンアップ | ~2260+275p(新) |
| [186](status-186.md) | 2026-03-16 | Phase 6 — C14 強化 + ソルバー deprecated 依存除去 | ~2260+275p(新) |
| [187](status-187.md) | 2026-03-16 | Phase 7 開始 — mesh/output C14 除去 + C16 クリア | ~2260+284p(新) |
| [188](status-188.md) | 2026-03-16 | Phase 7 完了 — ContactManager 移植 + C14/C16 違反ゼロ | ~2260+284p(新) |
| [189](status-189.md) | 2026-03-16 | Phase 8 完了 — C14 抜け道修正 + friction/geometry 実装完成 | ~2260+284p(新) |
| [190](status-190.md) | 2026-03-16 | solver Process 化完了 + NewtonUzawa Static/Dynamic 分離 | ~2260+301p(新) |
| [191](status-191.md) | 2026-03-16 | process.py Process API 移行 + ContactManager Process 分割 | ~2260+315p(新) |
| [192](status-192.md) | 2026-03-16 | Process 内部プライベート関数移行 + Strategy 公開 API + O1 条例検知 | ~2260+315p(新) |
| [193](status-193.md) | 2026-03-16 | deprecated 参照テスト無効化 + 状態操作ユーティリティ維持判断 | ~2260+315p(新) |
| [194](status-194.md) | 2026-03-17 | xkep_cae_deprecated → __xkep_cae_deprecated リネーム（C14 実効性強化） | ~2260+315p(新) |
| [195](status-195.md) | 2026-03-17 | numerical_tests モジュール新 xkep_cae 移植（BackendRegistry パターン） | ~2260+315p(新) |
| [196](status-196.md) | 2026-03-17 | S3 xfail テスト Process API 対応の前提条件調査 | ~2260+315p(新) |
| [197](status-197.md) | 2026-03-17 | Phase 13 完了 — ビームアセンブラ新 xkep_cae 移植 | ~2260+347p(新) |
| [198](status-198.md) | 2026-03-17 | Phase 14 — S3 xfail テスト Process API 対応版 | ~2260+358p(新) |
| [199](status-199.md) | 2026-03-17 | Process 実行診断インフラ — 警告・エラー・使用レポート | ~2260+374p(新) |
| [200](status-200.md) | 2026-03-17 | Process Architecture 監査 — elements/numerical_tests 不遵守検出強化 | ~2260+374p(新) |
| [201](status-201.md) | 2026-03-17 | Phase 15 完了 — C16 違反ゼロ達成 | ~2260+374p(新) |
| [202](status-202.md) | 2026-03-17 | C17 dataclass 衛生チェック + 命名規約修正（82ファイル） | ~2260+374p(新) |
| [203](status-203.md) | 2026-03-18 | C17 例外リスト廃止 + replace() 検知追加 | ~2260+374p(新) |
| [204](status-204.md) | 2026-03-18 | C17 違反ゼロ達成 — frozen dataclass 完全移行 | ~2260+374p(新) |
| [205](status-205.md) | 2026-03-18 | ContactManager Process 分割 — dataclass メソッド完全除去 | ~2260+374p(新) |
| [206](status-206.md) | 2026-03-18 | 旧API呼び出し元整合 + 旧テスト一掃 — 契約違反0件 | 248p(新) |
| [207](status-207.md) | 2026-03-18 | deprecated コード完全削除 + コンテキスト大掃除 | 248p(新) |
| [208](status-208.md) | 2026-03-18 | BackendRegistry 完全廃止 + 被膜モデル物理検証テスト | 412p(新) |
| [209](status-209.md) | 2026-03-18 | 単線の剛体支え＋押しジグ三点曲げ解析解一致 | 412+14p(新) |
| [210](status-210.md) | 2026-03-18 | smooth_penalty ソルバー復元 + HEX8 連続体要素ジグ基盤 | 412+14p(新) |
| [211](status-211.md) | 2026-03-18 | smooth_penalty 正定値接線 + 動的三点曲げ Process 基盤 | 431+3x(新) |
| [212](status-212.md) | 2026-03-19 | slow テスト見直し — 接触収束テスト全削除 + slow 分離 | 506(新) |

## アーカイブ（097〜174 — 旧 xkep_cae S3/R1 フェーズ）

status-097〜174 は [archive/](archive/) に移動済み。

| # | 日付 | マイルストーン | テスト数 |
|---|------|--------------|---------|
| [097](archive/status-097.md) | 2026-03-01 | S3開始 — xfailテスト根本対策 | 1906 |
| [101](archive/status-101.md) | 2026-03-02 | ドキュメント大整理（status-100記念） | 1916 |
| [112](archive/status-112.md) | 2026-03-05 | 19本NCP収束達成 | 2122 |
| [121](archive/status-121.md) | 2026-03-06 | 37本NCP収束 + 摩擦ヒステリシス移行 | 2261 |
| [130](archive/status-130.md) | 2026-03-07 | UL+CR梁 — 7本90°曲げ収束達成 | 2271 |
| [132](archive/status-132.md) | 2026-03-07 | NCP 6x高速化 + 揺動Phase2収束 | 2271 |
| [134](archive/status-134.md) | 2026-03-07 | ソルバー一本化（12.6x要素バッチ化） | 2271 |
| [147](archive/status-147.md) | 2026-03-09 | smooth penalty摩擦曲げ揺動収束達成 | 2271 |
| [150](archive/status-150.md) | 2026-03-10 | R1開始 — プロセスアーキテクチャ設計仕様策定 | 2271 |
| [162](archive/status-162.md) | 2026-03-13 | R1 Phase 7完遂 — 契約違反0件 | 2477 |
| [164](archive/status-164.md) | 2026-03-13 | R1 Phase 8完了 — ProcessRunner/StrategySlot/Preset | 2477+314p |
| [167](archive/status-167.md) | 2026-03-14 | AL完全削除 — NCP一本化 | ~2260+327p |
| [173](archive/status-173.md) | 2026-03-15 | deprecated プロセス完全削除 | ~2260+343p |
| [174](archive/status-174.md) | 2026-03-15 | solver_smooth_penalty.py 分解 → Process 実体化 | ~2260+343p |

## アーカイブ（001〜096 — Phase 1〜S2）

status-001〜096 は [archive/](archive/) に移動済み。

| # | 日付 | マイルストーン | テスト数 |
|---|------|--------------|---------|
| [001](archive/status-001.md) | 2026-02-12 | プロジェクト棚卸し・ロードマップ策定 | — |
| [003](archive/status-003.md) | 2026-02-12 | **Phase 1 完了** — Protocol/ABC アーキテクチャ | 16 |
| [015](archive/status-015.md) | 2026-02-14 | **Phase 2 完了** — 空間梁要素（EB/Timo/Cosserat/数値試験） | 374 |
| [020](archive/status-020.md) | 2026-02-14 | **Phase 3 完了** — 幾何学的非線形（NR/弧長/CR/TL/UL） | 407 |
| [023](archive/status-023.md) | 2026-02-16 | **Phase 4.1-4.2 完了** — 弾塑性+ファイバーモデル | 471 |
| [030](archive/status-030.md) | 2026-02-18 | **Phase 5 完了** — 動的解析+接触骨格 | 615 |
| [046](archive/status-046.md) | 2026-02-21 | **Phase C0-C5 完了** — 梁–梁接触基盤 | 993 |
| [056](archive/status-056.md) | 2026-02-25 | **Phase 4.7 L0 完了** — 撚線基礎（7本撚り+被膜+シース） | 1311 |
| [064](archive/status-064.md) | 2026-02-26 | **Phase 4.7 L0.5 完了** — シースFourier近似 | 1516 |
| [069](archive/status-069.md) | 2026-02-26 | **Phase 6.0 完了** — GNN/PINNサロゲートPoC | 1629 |
| [081](archive/status-081.md) | 2026-02-28 | **Phase C6 完了** — Line contact+NCP+Mortar+摩擦 | 1850 |
| [087](archive/status-087.md) | 2026-02-28 | **Phase S1-S2 完了** — 同層除外+NCP+CPU並列化基盤 | 1822 |
| [096](archive/status-096.md) | 2026-03-01 | **S2++/S3基盤 完了** — COO/CSR高速化+ベンチマーク基盤 | 1886 |

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
脱出ポット Phase 1: ~2260+34p(新) (2026-03-15) ← 新xkep_cae開始
Phase 2 前半:   ~2260+86p(新) (2026-03-15) ← core移行+Friction
Phase 2 後半:   ~2260+186p(新) (2026-03-15) ← Strategy全移行+契約違反0
Phase 2 完了:   ~2260+186p(新) (2026-03-15) ← status-179（契約違反ゼロ達成）
Phase 2 完備:   ~2260+204p(新) (2026-03-15) ← status-181（7軸ファクトリ完備）
C16拡大:        ~2260+204p(新) (2026-03-16) ← status-182（C16拡大+process/削除）
Phase 3完了:     ~2260+266p(新) (2026-03-16) ← status-183（Phase 3 concrete移行）
Phase 4完了:     ~2260+279p(新) (2026-03-16) ← status-184（Phase 4 Solver移行）
Phase 5完了:     ~2260+275p(新) (2026-03-16) ← status-185（Phase 5 ソルバー結果連携）
Phase 6完了:     ~2260+275p(新) (2026-03-16) ← status-186（Phase 6 C14強化+deprecated除去）
Phase 7進行:     ~2260+284p(新) (2026-03-16) ← status-187（Phase 7 mesh/output C14除去）
Phase 7完了:     ~2260+284p(新) (2026-03-16) ← status-188（Phase 7 完了 C14/C16=0）
Phase 8完了:     ~2260+284p(新) (2026-03-16) ← status-189（Phase 8 完了 friction/geometry実装）
Phase 9進行:     ~2260+301p(新) (2026-03-16) ← status-190（solver Process化+NUzawa分離）
Phase 9続:      ~2260+315p(新) (2026-03-16) ← status-191（Process API移行+CM Process分割）
Phase 9完了:     ~2260+315p(新) (2026-03-16) ← status-192（プライベート関数移行+Strategy公開API+O1条例検知）
Phase 10完了:    ~2260+315p(新) (2026-03-16) ← status-193（deprecated参照テスト無効化+状態ユーティリティ維持判断）
Phase 11完了:    ~2260+315p(新) (2026-03-17) ← status-194（deprecated→__deprecatedリネーム）
Phase 12進行:    ~2260+315p(新) (2026-03-17) ← status-195（numerical_tests移植）
Phase 13調査:    ~2260+315p(新) (2026-03-17) ← status-196（S3 xfail前提条件調査）
Phase 13完了:    ~2260+347p(新) (2026-03-17) ← status-197（ビームアセンブラ移植）
Phase 14完了:    ~2260+358p(新) (2026-03-17) ← status-198（S3 xfail Process API版）
診断インフラ:    ~2260+374p(新) (2026-03-17) ← status-199（Process実行診断）
Phase 15完了:    ~2260+374p(新) (2026-03-17) ← status-201（C16違反ゼロ）
C17例外廃止:     ~2260+374p(新) (2026-03-18) ← status-203（C17違反3件正規報告）
C17違反ゼロ:     ~2260+374p(新) (2026-03-18) ← status-204（frozen dataclass完全移行）
CM Process分割:  ~2260+374p(新) (2026-03-18) ← status-205（ContactManager メソッド完全除去→Process化）
旧テスト一掃:    248p(新) (2026-03-18) ← status-206（呼び出し元整合+旧テスト89件削除+契約違反0件）
大掃除:          248p(新) (2026-03-18) ← status-207（deprecated完全削除+scripts/examples/docs/旧テスト一掃）
Phase 17:        412p(新) (2026-03-18) ← status-208（BackendRegistry廃止+被膜物理テスト）
三点曲げ解析:    412+14p(新) (2026-03-18) ← status-209（三点曲げジグ解析解一致）
ソルバー復元:    412+14p(新) (2026-03-18) ← status-210（smooth_penalty復元+HEX8基盤）
正定値接線:      431+3x(新) (2026-03-18) ← status-211（正定値接線+動的三点曲げ基盤）
slowテスト整理:  506(新) (2026-03-19) ← status-212（接触収束テスト全削除+slow分離）
```

## 備考

- テスト数「—」はドキュメント更新・計画策定のみのステータス
- status-001〜096 は `docs/status/archive/` に移動（status-100 で実施）
- status-097〜174 は `docs/status/archive/` に移動（status-177 で実施）
- status-175〜 は新 xkep_cae（脱出ポット計画）のステータス

---
