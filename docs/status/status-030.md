# status-030: Phase 3.4 UL + Phase 5 陽解法・モーダル減衰 + Phase C0 接触骨格

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-18

## 概要

status-029 の TODO 4項目をすべて実行。陽解法（Central Difference）、モーダル減衰、Updated Lagrangian（参照配置更新）、梁–梁接触モジュール C0 骨格を実装。テスト数 556 → 615（+59テスト）。

## 実施内容

### 1. 陽解法（Central Difference）— 9テスト

`xkep_cae/dynamics.py` に追加:
- `CentralDifferenceConfig` / `CentralDifferenceResult` データクラス
- `critical_time_step(M, K, *, C=None, fixed_dofs=None)` — Δt_cr = 2/ω_max（減衰補正対応）
- `solve_central_difference(M, C, K, f_ext, u0, v0, config, *, fixed_dofs=None, check_stability=True)` — 陽的時間積分
- 対角質量行列の高速パス（`_is_diagonal()` 判定）
- 固定DOF対応

テスト（`tests/test_dynamics.py` → `TestCentralDifference`）:
1. 自由振動（SDOF解析解一致）
2. ステップ応答
3. Newmark-β との一致比較
4. 不安定性検出（Δt > Δt_cr）
5. 臨界時間ステップ計算
6. 対角質量行列高速パス
7. 固定DOF
8. 梁自由振動
9. config バリデーション

### 2. モーダル減衰 — 10テスト

`xkep_cae/dynamics.py` に追加:
- `build_modal_damping_matrix(M, K, damping_ratios, *, fixed_dofs=None)` — C = M·Φ·diag(2·ξᵢ·ωᵢ)·Φᵀ·M

テスト（`tests/test_dynamics.py` → `TestModalDamping`）:
1. SDOF粘性等価
2. 対称性
3. 正半定値性
4. 一様減衰比
5. 異なる減衰比
6. モーダル空間で対角
7. 固定DOF
8. 過渡減衰効果
9. スカラー減衰比
10. 梁収束性

### 3. Updated Lagrangian（参照配置更新）— 10テスト

`xkep_cae/elements/continuum_nl.py` に追加:
- `ULAssemblerQ4` クラス — ガウス点Cauchy応力追跡 `_stress_stored[n_elems, 4, 3]`
  - `_element_internal_force()`: f_int = ∫ B_L^T (σ_stored + D:ΔE) dV
  - `_element_tangent()`: K_T = K_mat + K_geo（累積応力含む）
  - `update_reference(u_inc)`: S→σ プッシュフォワード (σ = (1/J)F·S·Fᵀ)、参照節点更新
- `ULResult` データクラス
- `newton_raphson_ul()` ソルバー

テスト（`tests/test_continuum_nl.py` → `TestUpdatedLagrangian`）:
1. ゼロ変位
2. TL一致（単一ステップ）
3. 応力付き参照配置更新
4. リセット
5. NR小引張
6. TL一致（小荷重）
7. マルチステップ収束
8. TL-UL比較（中荷重、rtol=0.05）
9. 荷重履歴単調
10. 累積変位

### 4. Phase C0: 梁–梁接触モジュール骨格 — 30テスト

新規モジュール `xkep_cae/contact/`:
- `pair.py`: ContactStatus, ContactState, ContactPair, ContactConfig, ContactManager
- `geometry.py`: closest_point_segments, compute_gap, build_contact_frame
- `__init__.py`: ContactPair, ContactState をエクスポート

テスト（`tests/contact/`）:
- `test_geometry.py` (16テスト): 直交2線分、平行、離間、端点クランプ、法線方向、縮退、3Dねじれ、ギャップ計算、接触フレーム正規直交性・連続性・右手系
- `test_pair.py` (14テスト): デフォルト値、深いコピー、探索半径、活性判定、設定値、マネージャCRUD操作

## テスト

**615 passed, 2 skipped**（+59テスト）

## コミット履歴

1. `陽解法（Central Difference）実装` — 9テスト
2. `モーダル減衰（build_modal_damping_matrix）実装` — 10テスト
3. `Updated Lagrangian（参照配置更新）実装` — 10テスト
4. `Phase C0: 梁–梁接触モジュール骨格実装` — 30テスト

## TODO（残タスク）

- [ ] Phase C1: segment-to-segment 最近接 + broadphase（AABB格子）
- [ ] Phase C2: 法線AL + Active-setヒステリシス + 主項接線
- [ ] Phase C3: 摩擦return mapping + μランプ
- [ ] Phase C4: merit line search + 探索/求解分離の運用強化
- [ ] Phase 3.4 roadmap チェックボックス更新（UL 実装完了）

## 確認事項・懸念

- UL実装ではガウス点ごとのCauchy応力追跡が必要。参照配置更新時のS→σプッシュフォワードが正しく動作することをTL比較テストで確認済み。中荷重でrtol=5%程度の差は定式化の差異として許容。
- Phase C0 は骨格のみ。law_normal.py（法線接触力則）、solver_hooks.py（ソルバー統合）は C1〜C2 で実装予定。
- Central Difference は条件付き安定（Δt < Δt_cr）。check_stability=True でステップ中に安定性を監視。
- モーダル減衰は一般化固有値問題を解くため、大規模問題では計算コストに注意。

---
