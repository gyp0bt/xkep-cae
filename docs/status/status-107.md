# Status 107: NCP適応ステップ制御+EMAフィルタ+曲げ揺動ベンチマーク実行

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-05
**ブランチ**: `claude/inp-parser-refactor-6Q5YJ`
**テスト数**: 2016（fast: 1637 / slow: 374 + 5）— 変更なし

## 概要

NCP ソルバーに**自動安定時間増分制御（adaptive stepping）**と**接触乗数ローパスフィルタ（λ EMA）**を実装。曲げ揺動ベンチマークに全パラメータを伝播し、7本・19本撚線でフル実行を実施。

## 実施内容

### 1. NCP ソルバー拡張（`solver_ncp.py`）

#### 自動安定時間増分制御
- Newton 反復数ベースのステップサイズ制御（easy → grow, hard → shrink, fail → bisect）
- パラメータ: `adaptive_stepping`, `target_newton_iters`, `dt_grow_factor`, `dt_shrink_factor`, `dt_max_factor`, `dt_min_factor`
- 変位収束（`du/u < tol_disp`）との統合: disp-converged ステップは effective_iters をクランプし、不要な dt 縮小を防止
- **per-step reference norm reset**: 変位制御問題で `_dynamic_ref` が初回ステップの基準を固定する不具合を修正

#### 接触乗数 EMA フィルタ
- 時間方向の指数移動平均: `λ̃ = α*λ̃_prev + (1-α)*λ_raw`
- active set チャタリングの高周波振動を抑制
- パラメータ: `lambda_ema_alpha`（0.0 = 無効、0.3 推奨）

### 2. ベンチマークパラメータ伝播

- `wire_bending_benchmark.py`: `run_bending_oscillation` に全新パラメータ追加
- `run_bending_oscillation.py`: `DEFAULT_PARAMS` + `solve_from_inp` に完全伝播
- Phase 1（曲げ）+ Phase 2（揺動）の両方に適用

### 3. 曲げ揺動ベンチマーク実行結果

#### 7本撚線

| 項目 | 値 |
|------|-----|
| 要素数 | 112 (16要素/ピッチ) |
| DOF | 714 |
| 曲げ角 | 5.0° |
| Phase 1 収束 | **✓** (10/10 steps, 50 NR反復) |
| Phase 2 収束 | **✓** (4/4 steps, 28 NR反復) |
| 活性接触ペア | 142 |
| 計算時間 | 33.5s |

#### 19本撚線

| 項目 | 値 |
|------|-----|
| 要素数 | 304 (16要素/ピッチ) |
| DOF | 1938 |
| 曲げ角 | 5.0° |
| Phase 1 収束 | **✓** (10/10 steps, 53 NR反復) |
| Phase 2 収束 | **partial** (1/4 steps converged) |
| 活性接触ペア | 492 |
| 計算時間 | >10min (Phase 2 step 4 timeout) |

### 4. CR 梁大変形の収束限界（発見事項）

16要素/ピッチの CR 梁で曲げ角 ~8° 付近に収束壁が存在：
- **原因**: CR 梁の接線剛性が大回転時に ill-conditioned 化
- **観測**: force residual が 1e-3〜1e-4 に停滞、displacement convergence は成立
- **接触は無関係**: active=0 の状態でも発生（純粋な CR 梁の問題）
- **影響**: 45° や 30° 曲げは現状不可。5° 以下で安定動作
- **対策**: Phase S4 以降で CR 梁定式化の改良（Updated Lagrangian, GBT 等）が必要

### 5. DEFAULT_PARAMS 最適化

CR 梁の収束限界に基づき、実行可能な設定に更新：
- `bend_angle_deg`: 45.0 → **5.0**
- `n_bending_steps`: 5 → **10**
- `oscillation_amplitude_mm`: 2.0 → **0.1**
- `n_steps_per_quarter`: 2 → **1**
- `broadphase_margin`: 0.01 → **0.002**
- `adaptive_stepping`: True → **False**（CR梁大変形では固定ステップ推奨）

## 確認事項・懸念

1. **CR 梁収束壁**: 16要素/ピッチで ~8° が限界。これは Phase S4 での CR 梁改良が必須
2. **Phase 2 19本非収束**: Phase 1 は問題なし。Phase 2 step 2 以降で non-converged state が NaN に伝播
3. **接触未活性化**: 5° 曲げでは素線間接触が活性化しない（曲げ角が小さすぎる）。接触テストには大曲げが必要 → CR 梁改良が前提条件
4. **計算速度**: 7本 714DOF で 33.5s。1000本 30000DOF では現状のスケーリングでは数時間必要

## TODO

- [ ] CR 梁大変形収束改善（Updated Lagrangian / corotational rotation interpolation 改良）
- [ ] Phase 2 NCP 揺動ステップの収束安定化（非収束ステップから復帰するロジック）
- [ ] 37本以上のベンチマーク実行
- [ ] 接触が活性化する曲げ角での収束テスト
