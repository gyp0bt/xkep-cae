# status-110: ステップ二分法deprecated化 + S3改良10-11（残差スケーリング・接触力ランプ）

[← README](../../README.md) | [← status-index](status-index.md) | [← status-109](status-109.md)

日付: 2026-03-05

## 概要

1. **ステップ二分法（`max_step_cuts`/`bisection_max_depth`）を非推奨化**し、適応時間増分制御（`adaptive_timestepping`）に統合
2. **S3改良10: 残差スケーリング** — 鞍点系の対角スケーリング前処理を追加
3. **S3改良11: 接触力ランプ** — Newton反復初期の接触力を段階的に増大

## 変更詳細

### 1. ステップ二分法のdeprecated化

**背景**: ステップ二分法（`max_step_cuts`/`bisection_max_depth`）はステップ不収束時の事後リカバリ機構であったが、適応時間増分制御（S3改良6）が事前制御と事後リカバリの両方を統合した。

**変更内容**:
- `max_step_cuts > 0` または `bisection_max_depth > 0` 指定時に `DeprecationWarning` を発生
- 自動的に `adaptive_timestepping=True` に変換
- ステップ不収束時のリトライロジックを適応時間増分制御に統合（`dt_shrink_factor` で縮小）
- 旧: 固定の二分割（delta/2）→ 新: 設定可能な縮小係数（`dt_shrink_factor`, デフォルト0.5）

**互換ヒストリー**:

| 旧機能 | 新機能 | 移行status | 状態 |
|--------|--------|-----------|------|
| `max_step_cuts` / `bisection_max_depth` | `adaptive_timestepping=True` | status-110 | deprecated化完了 |

**テスト移行**:
- `TestStepBisection` → `TestStepBisectionDeprecated`（DeprecationWarning発生テスト）
- `test_adaptive_dt_with_bisection` → `test_adaptive_dt_failure_retry`
- 全テストから `max_step_cuts`/`bisection_max_depth` を削除し `adaptive_timestepping=True` に置換

### 2. S3改良10: 残差スケーリング（対角スケーリング前処理）

鞍点系の条件数を改善するため、K行列の対角成分に基づくスケーリングを導入。

```
D = diag(|K|)^{-1/2}
K̃ = D K D,  R̃ = D R,  G̃ = G D
解: ũ = D^{-1} u → du = D ũ
```

- `ContactConfig.residual_scaling: bool = False`
- `_solve_saddle_point_contact()` に `residual_scaling` パラメータ追加
- 2テスト追加

### 3. S3改良11: 接触力ランプ

Newton反復の初期で接触力を段階的に増大させ、接触活性セットの急変を抑制。

```
ramp_factor = (it + 1) / ramp_iters  (it < ramp_iters の場合)
f_c = f_c * ramp_factor
```

- `ContactConfig.contact_force_ramp: bool = False`
- `ContactConfig.contact_force_ramp_iters: int = 5`
- 2テスト追加

## 影響を受けるファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver_ncp.py` | deprecated化ロジック + S3改良10-11実装 |
| `xkep_cae/contact/pair.py` | `residual_scaling`, `contact_force_ramp*` パラメータ追加 |
| `tests/contact/test_solver_ncp_s3.py` | 31テスト（+4テスト: deprecated+scaling+ramp） |
| `tests/contact/test_beam_contact_penetration_ncp.py` | `max_step_cuts` → `adaptive_timestepping` |
| `tests/contact/test_ncp_convergence_19strand.py` | `bisection_max_depth` → `adaptive_timestepping` |
| `xkep_cae/numerical_tests/wire_bending_benchmark.py` | `max_step_cuts` → `adaptive_timestepping` |
| `scripts/run_bending_oscillation.py` | `max_step_cuts` → `adaptive_timestepping` |

## テスト状況

- **S3テスト**: 31件全パス
- **接触テスト（NCP）**: 18件全パス
- **全fastテスト**: 確認中

## S3改良の全体像

| # | 改良 | status | 状態 |
|---|------|--------|------|
| 1 | ILU drop_tol 適応制御 | 107 | ✅ |
| 2 | Schur正則化改善 | 107 | ✅ |
| 3 | GMRES restart適応 | 107 | ✅ |
| 4 | λウォームスタート | 107 | ✅ |
| 5 | Active setチャタリング抑制 | 107 | ✅ |
| 6 | 適応時間増分制御 | 109 | ✅ |
| 7 | AMG前処理 | 109 | ✅ |
| 8 | k_pen continuation | 109 | ✅ |
| 9 | k_pen自動推定（NCP） | 109 | ✅ |
| 10 | 残差スケーリング | **110** | ✅ |
| 11 | 接触力ランプ | **110** | ✅ |

## TODO

- [ ] 19本NCP収束達成（S3改良1-11の組合せチューニング）
- [ ] 37/61/91本の段階的収束テスト
- [ ] NCPソルバー版S3ベンチマーク（AL法との比較）
