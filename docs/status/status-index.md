# ステータス一覧（status-index）

[← README](../../README.md) | [← roadmap](../roadmap.md)

> 本ファイルはステータスファイルの一覧メモです。新規status作成時に必ず更新すること。

## 一覧

| # | 日付 | タイトル | テスト数 |
|---|------|---------|---------|
| [001](status-001.md) | 2026-02-12 | プロジェクト棚卸しとロードマップ策定 | — |
| [002](status-002.md) | 2026-02-12 | Phase 1 アーキテクチャ再構成 | 16 |
| [003](status-003.md) | 2026-02-12 | pycae → xkep-cae リネーム & Phase 1 残作業完了 | 16 |
| [004](status-004.md) | 2026-02-12 | Phase 2.1/2.2 梁要素実装 & Abaqus .inp パーサー | 72 |
| [005](status-005.md) | 2026-02-12 | レガシー削除・Protocol API 一本化・Q4 D行列バグ修正 | 74 |
| [006](status-006.md) | 2026-02-12 | EAS-4 Q4要素実装 / B-barバグ修正 / Abaqus梁要素調査 | 88 |
| [007](status-007.md) | 2026-02-12 | Cowper κ(ν) 実装 / Q4 Abaqus比較テスト / Abaqus差異ドキュメント | 115 |
| [008](status-008.md) | 2026-02-13 | ロードマップ拡張 — Cosserat rod & 撚線モデル | — |
| [009](status-009.md) | 2026-02-13 | Phase 2.3/2.4 — 3D Timoshenko梁 & 断面モデル拡張 & SCF & パーサー拡張 | 161 |
| [010](status-010.md) | 2026-02-13 | 3Dアセンブリテスト & 内力ポスト処理 & ワーピング検討 | 174 |
| [011](status-011.md) | 2026-02-13 | 2D断面力ポスト処理 & せん断応力 & 数値試験ロードマップ | 193 |
| [012](status-012.md) | 2026-02-13 | 数値試験フレームワーク（Phase 2.6）実装 | 241 |
| [013](status-013.md) | 2026-02-13 | Cosserat rod 四元数回転実装（Phase 2.5 前半） | 314 |
| [014](status-014.md) | 2026-02-14 | Cosserat rod Phase 2.5 完成 & 数値試験フレームワーク拡張 | 345 |
| [015](status-015.md) | 2026-02-14 | Cosserat rod SRI & Phase 3 幾何学的非線形開始 | 374 |
| [016](status-016.md) | 2026-02-14 | 梁–梁接触モジュール仕様追加とロードマップ統合 | — |
| [017](status-017.md) | 2026-02-14 | 撚線ロードマップの優先順を「接触先行」に再定義 | — |
| [018](status-018.md) | 2026-02-14 | 接触着手時期を「Phase 5後」に再整理 | — |
| [019](status-019.md) | 2026-02-14 | ロードマップ大幅修正（Codex による構造破壊の修復） | — |
| [020](status-020.md) | 2026-02-14 | Phase 3 幾何学的非線形 完了 | 407 |
| [021](status-021.md) | 2026-02-14 | Phase 4.1 1次元弾塑性 完了 | 435 |
| [022](status-022.md) | 2026-02-15 | バリデーションテスト文書化 | 435 |
| [023](status-023.md) | 2026-02-16 | Phase 4.2 ファイバーモデル（曲げの塑性化）完了 | 471 |
| [024](status-024.md) | 2026-02-16 | lintエラー解消 + von Mises 3D降伏ロードマップ追加 | 471 |
| [025](status-025.md) | 2026-02-17 | von Mises 3D塑性テスト計画策定 | — |
| [026](status-026.md) | 2026-02-18 | Phase 5.1〜5.2 動的解析実装 | 498 |
| [027](status-027.md) | 2026-02-18 | 戻り値型リファクタリング + ロードマップ優先順位更新 | 498 |
| [028](status-028.md) | 2026-02-18 | Phase 3.4 TL定式化 + Phase 5.4 非線形動解析 + 動的三点曲げ | 556 |
| [029](status-029.md) | 2026-02-18 | Phase 4.3 von Mises 3D 凍結処理 + TODO整理 | 556 |
| [030](status-030.md) | 2026-02-18 | Phase 3.4 UL + Phase 5 陽解法・モーダル減衰 + Phase C0 接触骨格 | 615 |
| [031](status-031.md) | 2026-02-18 | 過渡応答出力インターフェース（Step/Frame/Increment + CSV/JSON/VTK） | 653 |
| [032](status-032.md) | 2026-02-18 | 過渡応答出力 TODO 消化（run_transient_steps / 非線形RF / VTKバイナリ / 要素データ / .inp統合） | 670 |
| [033](status-033.md) | 2026-02-18 | Phase C1 — Broadphase (AABB格子) + 幾何更新 + Active-setヒステリシス | 701 |
| [034](status-034.md) | 2026-02-18 | FIELD ANIMATION出力 + .inpパーサー拡張（*ELSET / *BOUNDARY / *OUTPUT, FIELD ANIMATION） | 741 |
| [035](status-035.md) | 2026-02-18 | .inpパーサー拡張（*MATERIAL / *ELASTIC / *DENSITY / *PLASTIC）+ pyproject.toml更新 | 753 |
| [036](status-036.md) | 2026-02-18 | テーブル補間型硬化則 + matplotlibテストスキップ対応 | 782 |
| [037](status-037.md) | 2026-02-18 | GIFアニメーション出力 + examplesディレクトリ追加 | 789 |
| [038](status-038.md) | 2026-02-19 | KINEMATIC→AF変換 + TODO整理 | 802 |
| [039](status-039.md) | 2026-02-19 | Phase C2 — 法線AL接触力 + 接触接線剛性 + 接触付きNRソルバー | 845 |
| [040](status-040.md) | 2026-02-19 | .inp実行スクリプト + Abaqus三点曲げバリデーション + inp_runner | 865 |
| [041](status-041.md) | 2026-02-19 | 三点曲げ非線形動解析スクリプト + GIFアニメーション出力 | 865 |
| [042](status-042.md) | 2026-02-19 | Corotational (CR) 定式化による Timoshenko 3D 梁の幾何学的非線形 | 887 |
| [043](status-043.md) | 2026-02-21 | Abaqus弾塑性三点曲げ（idx2）ファイバーモデルバリデーション | 905 |
| [044](status-044.md) | 2026-02-21 | Phase C3 — 摩擦 return mapping + μランプ | 932 |
| [045](status-045.md) | 2026-02-21 | Phase C4 — merit line search + 探索/求解分離の運用強化 | 958 |
| [046](status-046.md) | 2026-02-21 | Phase C5 — 幾何微分込み一貫接線 + slip consistent tangent + PDAS + 平行輸送フレーム | 993 |
| [047](status-047.md) | 2026-02-22 | 摩擦接触バリデーション + 接触付き弧長法設計検討 | 1009 |
| [048](status-048.md) | 2026-02-23 | 梁梁接触 貫入テスト（11テスト） | 1034 |

## テスト数推移

```
001-003: 16       (Phase 1 アーキテクチャ)
004:     72       (Phase 2.1/2.2 梁要素)
005:     74
006:     88       (EAS-4)
007:     115      (Cowper κ)
009:     161      (3D Timoshenko)
010:     174
011:     193      (断面力ポスト処理)
012:     241      (数値試験フレームワーク)
013:     314      (Cosserat rod)
014:     345
015:     374      (SRI + Phase 3 開始)
020:     407      (Phase 3 完了)
021:     435      (Phase 4.1 弾塑性)
022-024: 471      (Phase 4.2 ファイバーモデル)
026-027: 498      (Phase 5 動的解析)
028-029: 556      (Phase 3.4 TL + Phase 5.4 非線形動解析)
030:     615      (UL + 陽解法 + モーダル減衰 + Phase C0 接触骨格)
031:     653      (過渡応答出力インターフェース: Step/Frame/Increment + CSV/JSON/VTK)
032:     670      (過渡応答出力 TODO 消化: run_transient_steps / 非線形RF / VTKバイナリ / 要素データ / .inp統合)
033:     701      (Phase C1: Broadphase AABB格子 + 幾何更新 + Active-setヒステリシス)
034:     741      (FIELD ANIMATION出力 + .inpパーサー拡張: *ELSET / *BOUNDARY / *OUTPUT, FIELD ANIMATION)
035:     753      (.inpパーサー拡張: *MATERIAL / *ELASTIC / *DENSITY / *PLASTIC + pyproject.toml更新)
036:     782      (テーブル補間型硬化則 + matplotlibテストスキップ対応)
037:     789      (GIFアニメーション出力 + examplesディレクトリ追加)
038:     802      (KINEMATIC→AF変換 + TODO整理)
039:     845      (Phase C2: 法線AL + 接触接線 + 接触付きNR)
040:     865      (.inp実行スクリプト + Abaqus三点曲げバリデーション + inp_runner)
041:     865      (三点曲げ非線形動解析スクリプト + GIFアニメーション出力)
042:     887      (CR定式化 Timoshenko 3D梁 幾何学的非線形)
043:     905      (Abaqus弾塑性三点曲げ ファイバーモデルバリデーション)
044:     932      (Phase C3: 摩擦 return mapping + μランプ)
045:     958      (Phase C4: merit line search + 探索/求解分離の運用強化)
046:     993      (Phase C5: 幾何微分込み一貫接線 + slip consistent tangent + PDAS + 平行輸送フレーム)
047:     1009     (摩擦接触バリデーション + 接触付き弧長法設計検討)
048:     1034     (梁梁接触 貫入テスト)
```

## 備考

- テスト数「—」はドキュメント更新・計画策定のみのステータス（コード変更なし）
- status-022 は日付記載なし（推定 2026-02-15）
- 今後、新規statusファイル作成時に本ファイルへの行追加を必須とする

---
