# status-286: 揺動サイクル基盤実装 — prescribed_func + checkpoint自工程保証

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-04
- **ブランチ**: `claude/check-status-todos-eunyp`
- **テスト数**: 621 passed（回帰なし）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

7本撚線の曲げ+揺動サイクルに向けた基盤実装。
checkpoint保存/復元の「自工程保証」原則に基づき、
各インクリメント完了時の保存状態で次のインクリメントをクリーンに開始できるよう
全状態の保存/復元を実装。

---

## 実装内容

### 1. prescribed_func（処方変位時間関数）

| 項目 | 詳細 |
|------|------|
| `BoundaryData.prescribed_func` | `Callable[[float], ndarray]` — frac→処方変位値 |
| ソルバー適用 | `state.u[prescribed_dofs] = prescribed_func(frac)` |
| 揺動対応 | cos波: `θ_amp * (cos(2π*n*frac) - 1.0)` でcoords_ref基準増分 |

### 2. checkpoint API

| 項目 | 詳細 |
|------|------|
| `ContactFrictionInputData.checkpoint_path` | pickle保存先パス |
| `ContactFrictionInputData.checkpoint_frac` | 保存トリガーfrac閾値 |
| `skip_initial_detection` | 復元時に初期接触検出をスキップ |

### 3. 自工程保証: checkpoint完全状��保存

| 保存項目 | 用途 |
|---------|------|
| `state.u` | 変位ベクトル |
| `time_vel/acc` | 速度/加速度 |
| `manager_pairs/config/connectivity` | 接触マネージャ全状態 |
| `ul_u_total_accum` | UL累積変位 |
| **`ul_coords_ref`** | ULアセンブラ参照座標（**今回追加**） |
| **`ul_R_ref`** | ULアセンブラ参照回転行列（**今回追加**） |
| **`ul_ref_base`** | UL参照更新基準変位（**今回追加**） |

### 4. 復元時の状態整合

| 修正 | 根本原因 |
|------|---------|
| `u0 = state.u - ul_ref_base` | 全変位ではなくcoords_ref基準増分を渡す |
| `_ul_ref_base = u0` | 初回UL更新の増分=0を保証 |
| `prescribed_func`: `cos - 1.0` | frac=0でΔθ=0（checkpoint状態維持） |

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/core/data.py` | `prescribed_func`, `checkpoint_path/frac`, `skip_initial_detection` |
| `xkep_cae/contact/solver/process.py` | checkpoint完全保存 + 復元モード + prescribed_func適用 |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | 揺動サ���クル対応 + checkpoint復元パイプライン |

---

## ベンチマーク結果

### 曲げフェーズ（Hertz型 α=1.5）
- frac=0.9981, incr=551, cutback=60（status-285と一致、再現性確認）

### 揺動フェーズ（checkpoint復元）
- 復元成功: ULアセンブラ + 接触マネージャ + u0 の整合性確保
- NR残差: `||R|| ≈ 1e-5`（frac=0.006）→ チャタリング停滞でtol=1e-8未到達
- **曲げ同様のNRチャタリング問題**（接触ペア活性集合の振動）

### 発散→収束の推移（自工程保証の効果）

| バージョン | 初回残差 ||R_r|| | 原因 |
|-----------|--------|------|
| v1（マネージャ未復元） | 8,060 | 初期検出で全ペア再検出 |
| v2（マネージャ復元、検出ス��ップ） | 7,770 | NR内接触更新で爆発 |
| v3（UL coords_ref復元） | 58,400 | u0が全変位のまま |
| **v4（u0=増分、prescribed_func=増分）** | **290→6.2** | **正常（チャタリングのみ）** |

---

## 技術的要点（次の担当者向け）

### checkpoint復元の注意点

1. ULアセンブラの3点セット（coords_ref + R_ref + _u_total_accum）は**必ず同時に**復元する
2. `u0` は coords_ref 基準の増分変位（≈0）であり、初期配置からの全変位ではない
3. `_ul_ref_base` は u0 で初期化する（初回UL更新増分=0保証）
4. `prescribed_func` の返り値も coords_ref 基準の増分
5. 接触マネージャの `skip_initial_detection=True` で初期検出をスキップ

### 揺動収束の次の課題

揺動フェーズの残差が `1e-5` で停滞するのは曲げフ��ーズと同じチャタリング問題。
Hertz型ペナルティ + 凍結モードは適用済みだが、揺動の荷重反転で活性集合が
大きく変動するため追加対策が必要:
- tol_force の緩和（1e-8 → 1e-6）で揺動フェーズのみ許容
- chattering_freeze_max_cycles 増加
- 初回の dt をさらに小さくする初期化戦略

---

## TODO

- [ ] 揺動フェーズのNR収束改善（tol緩和 or dt初期化戦略）
- [ ] 揺動frac > 0.1 達成
- [ ] 曲げ→揺動の1回実行方式も実装（checkpoint不要のシンプル版）
- [ ] checkpoint復元の単体テスト追加

---
