# 接触テスト カタログ（Phase C0〜C5）

[← README](../../README.md) | [← validation](validation.md) | [← roadmap](../roadmap.md)

本文書は梁–梁接触モジュール（Phase C0〜C5）の全テストを体系的にまとめたカタログである。
テスト名・検証内容・主要パラメータ・許容条件を一覧化し、回帰テストの品質管理に用いる。

## 目次

- [Phase C0: データ構造・ペア管理](#phase-c0-データ構造ペア管理)
- [Phase C1: 幾何アルゴリズム・Broadphase・Active-set](#phase-c1-幾何アルゴリズムbroadphaseactive-set)
- [Phase C2: 法線接触力・アセンブリ・ソルバー統合](#phase-c2-法線接触力アセンブリソルバー統合)
- [Phase C3: Coulomb 摩擦](#phase-c3-coulomb-摩擦)
- [Phase C4: Merit line search](#phase-c4-merit-line-search)
- [Phase C5: 一貫接線・PDAS・バリデーション](#phase-c5-一貫接線pdasバリデーション)
- [実梁要素接触テスト](#実梁要素接触テスト)
- [テスト統計](#テスト統計)

---

## Phase C0: データ構造・ペア管理

**テストファイル**: `tests/contact/test_pair.py`

### ContactState（状態管理）

| テスト名 | 検証内容 | 許容条件 |
|----------|---------|---------|
| `test_default_values` | デフォルト初期化（全ゼロ・INACTIVE） | — |
| `test_copy_independence` | ディープコピーの独立性 | 変更が影響しない |
| `test_z_t_default` | 接線変位 z_t のデフォルトがゼロ (2,) | — |

### ContactPair（ペア構成）

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_search_radius` | search_radius = radius_a + radius_b | r_a=0.5, r_b=0.3 → 0.8 |
| `test_is_active_default` | デフォルト INACTIVE | — |
| `test_is_active_when_active` | ACTIVE 判定 | status=ACTIVE |

### ContactConfig（設定）

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_defaults` | デフォルト値: k_pen_scale=1.0, k_t_ratio=0.5, μ=0.3, use_friction=False | — |
| `test_custom_values` | カスタム設定値の反映 | k_pen_scale=2.0, μ=0.5 |

### ContactManager（管理器）

| テスト名 | 検証内容 | 許容条件 |
|----------|---------|---------|
| `test_empty_manager` | 空管理器: n_pairs=0, n_active=0 | — |
| `test_add_pair` | ペア追加と属性確認 | 4節点, 2要素 |
| `test_active_count` | アクティブペア数のカウント | — |
| `test_get_active_pairs` | ACTIVE/SLIDING のみ返却 | — |
| `test_reset_all` | 全ペアリセット（deactivate + p_n=0） | — |
| `test_custom_config` | カスタム設定の受け渡し | μ=0.5, use_friction=True |

---

## Phase C1: 幾何アルゴリズム・Broadphase・Active-set

### 最近接点計算（Narrowphase 幾何）

**テストファイル**: `tests/contact/test_geometry.py`

#### ClosestPointSegments

| テスト名 | 検証内容 | 許容誤差 |
|----------|---------|---------|
| `test_perpendicular_midpoints` | 直交セグメントの中点一致 (s=0.5, t=0.5) | 1e-14 |
| `test_parallel_segments` | 平行セグメントの距離・平行フラグ | 1e-12 |
| `test_separated_segments` | 離れたセグメントの距離 | 1e-12 |
| `test_endpoint_clamping` | 最適点がセグメント外の場合の端点クランプ | 1e-12 |
| `test_point_a_on_segment` | パラメータ s による補間精度 | 1e-14 |
| `test_normal_direction` | 法線方向 = 差分ベクトル/ノルム | 1e-12 |
| `test_zero_length_segment` | 縮退セグメントの処理 | — |
| `test_skew_segments_3d` | 3D ねじれセグメント | 1e-12 |

#### ComputeGap

| テスト名 | 検証内容 | 解析解 |
|----------|---------|--------|
| `test_positive_gap` | g = distance - r_a - r_b > 0（分離） | dist=5, Σr=3 → g=2 |
| `test_zero_gap` | g = 0（接触点） | dist=3, Σr=3 → g=0 |
| `test_negative_gap` | g < 0（貫入） | dist=1, Σr=2 → g=-1 |

#### BuildContactFrame

| テスト名 | 検証内容 | 許容誤差 |
|----------|---------|---------|
| `test_orthonormal` | (n, t1, t2) が正規直交基底 | 1e-14 |
| `test_various_normals` | 5方向の法線全てで正規直交 | 1e-14 |
| `test_continuity_with_prev_tangent` | 前回 t1 を参照した連続フレーム | — |
| `test_prev_tangent_slightly_off` | Gram-Schmidt 補正の動作 | — |
| `test_right_hand_system` | t2 = n × t1（右手系） | 1e-14 |

### Broadphase（AABB格子）

**テストファイル**: `tests/contact/test_broadphase.py`

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_basic_aabb` | 基本 AABB 計算 | x0=[1,2,3], x1=[4,1,5] |
| `test_aabb_with_radius` | 半径による AABB 拡大 | radius=0.5 |
| `test_aabb_with_margin` | マージン追加 | margin=0.2, radius=0.1 |
| `test_aabb_reversed_endpoints` | 端点順序の不変性 | — |
| `test_per_segment_radii` | セグメント別半径配列 | radii=[1.0, 4.5] |
| `test_two_crossing_segments` | 交差セグメントの検出 | — |
| `test_two_distant_segments` | 離れたセグメントの非検出 | — |
| `test_radius_brings_segments_closer` | 半径による検出範囲拡大 | r=0→不検出, r=2→検出 |
| `test_single_segment` | 単一セグメント: 候補なし | — |
| `test_empty_segments` | 空リスト: 候補なし | — |
| `test_multiple_segments_grid` | 10本平行セグメント: 隣接のみ検出 | margin=0.5 |
| `test_parallel_overlapping_segments` | 平行重複セグメントの検出 | — |
| `test_3d_skew_segments` | 3D ねじれセグメントの検出 | — |
| `test_no_self_pairs` | 自己ペア (i,i) の除外 | — |
| `test_no_duplicates` | 重複候補の排除 | — |

### 候補検出・幾何更新（ContactManager統合）

**テストファイル**: `tests/contact/test_pair.py`

#### DetectCandidates

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_crossing_beams_detected` | 交差梁の候補検出 | radii=0.5 |
| `test_distant_beams_not_detected` | 離れた梁の非検出 | distance >> margin |
| `test_existing_pairs_deactivated` | 範囲外ペアの非活性化 | 座標を100+移動 |
| `test_new_candidates_added` | 新規候補の追加 | 3本梁 |
| `test_per_element_radii` | 要素別半径配列 | radii=[1.0, 4.5] |

#### UpdateGeometry

| テスト名 | 検証内容 | 許容誤差 |
|----------|---------|---------|
| `test_updates_gap_and_closest_point` | ギャップ・最近接点パラメータの更新 | — |
| `test_contact_frame_orthonormal` | 接触フレームの正規直交性 | 1e-14 |
| `test_gap_with_radius` | 半径込みギャップ計算 | r_a=0.5, r_b=0.5 → g=1.0 |
| `test_penetration_gap` | 貫入時の負ギャップ | Σr=1.0, dist=0.5 → g=-0.5 |
| `test_frame_continuity` | フレーム接線ベクトルの連続性 | — |

### Active-set ヒステリシス

**テストファイル**: `tests/contact/test_pair.py`

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_activate_on_contact` | g ≤ g_on で ACTIVE 化 | g_on=0, gap=-0.5 |
| `test_stays_active_in_hysteresis_band` | g_on < g < g_off で ACTIVE 維持 | g_off=0.1, gap=0.05 |
| `test_deactivate_beyond_g_off` | g ≥ g_off で INACTIVE 化 | gap=2.0 >> g_off |
| `test_inactive_stays_inactive_in_band` | INACTIVE はバンド内で維持 | gap=1.05 |
| `test_sliding_deactivates_beyond_g_off` | SLIDING も g_off 超で INACTIVE 化 | — |

---

## Phase C2: 法線接触力・アセンブリ・ソルバー統合

### 法線接触法則（Augmented Lagrangian）

**テストファイル**: `tests/contact/test_law_normal.py`

#### 法線力

| テスト名 | 検証内容 | 解析式 |
|----------|---------|--------|
| `test_penetration_gives_positive_force` | p_n = max(0, λ_n + k_pen·(−g)) | gap=-0.01, k=1e4 → p_n=100 |
| `test_separation_gives_zero_force` | 分離時 p_n = 0（粘着なし） | gap=0.1 → p_n=0 |
| `test_inactive_gives_zero` | INACTIVE ペアは p_n=0 | — |
| `test_al_multiplier_effect` | AL乗数 λ_n の寄与 | λ_n=50 → p_n=150 |
| `test_al_with_positive_gap_and_multiplier` | λ_n が正ギャップを補償 | gap=0.001, λ_n=100 → p_n=90 |
| `test_zero_gap_with_zero_multiplier` | g=0, λ_n=0 → p_n=0 | — |

#### AL乗数更新

| テスト名 | 検証内容 | 許容条件 |
|----------|---------|---------|
| `test_active_pair_update` | ACTIVE: λ_n ← p_n | p_n=100 → λ_n=100 |
| `test_inactive_pair_reset` | INACTIVE: λ_n ← 0 | λ_n=50 → 0 |
| `test_repeated_update_converges` | 複数AL反復で λ_n 単調増加 | 5反復 |

#### 法線力線形化

| テスト名 | 検証内容 | 解析式 |
|----------|---------|--------|
| `test_active_with_positive_force` | dp_n/dg = k_pen（活性接触） | dp/dg = 1e4 |
| `test_inactive_returns_zero` | INACTIVE → dp/dg = 0 | — |
| `test_zero_force_returns_zero` | p_n=0 → dp/dg = 0 | — |

#### ペナルティ初期化

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_basic_initialization` | k_pen, k_t の設定 | k_pen=1e5, k_t_ratio=0.3 |
| `test_default_ratio` | デフォルト k_t_ratio=0.5 | k_pen=1e4 → k_t=5e3 |
| `test_ea_over_l` | EA/L ベース推定 | E=2.1e11, A=1e-4, L=0.1 |
| `test_scale_factor` | スケール係数の倍率 | scale=10 |

### 接触アセンブリ

**テストファイル**: `tests/contact/test_contact_assembly.py`

#### 接触形状ベクトル

| テスト名 | 検証内容 | 解析式 |
|----------|---------|--------|
| `test_midpoint_z_normal` | 中点: 対称分布 g=[−0.5,−0.5,+0.5,+0.5] | s=0.5, t=0.5, n=[0,0,1] |
| `test_endpoint_s0_t1` | 端点: 集中分布 | s=0, t=1 |
| `test_action_reaction_balance` | 作用反作用: Σg_i = 0 | Newton 第3法則 |

#### 接触力計算

| テスト名 | 検証内容 | 許容条件 |
|----------|---------|---------|
| `test_single_pair_midpoint` | 1ペアの力分布 | p_n=100 対称分布 |
| `test_inactive_pair_no_force` | INACTIVE → f_c=0 | — |
| `test_separation_no_force` | 分離 → f_c=0 | — |
| `test_total_force_balance` | グローバル力均衡 (Newton 第3法則) | Σf_c=0 |
| `test_rotation_dofs_zero` | 回転DOF は接触力ゼロ | f_c[rot]=0 |

#### 接触剛性行列

| テスト名 | 検証内容 | 許容誤差 |
|----------|---------|---------|
| `test_symmetry` | K_c が対称 | 1e-10 |
| `test_positive_semidefinite` | 主項が正半定値 | eigvals ≥ 0 |
| `test_rank_one` | 1ペア: rank(K_c) = 1 | K = k·g⊗g |
| `test_inactive_pair_empty_matrix` | INACTIVE → K_c.nnz=0 | — |
| `test_consistent_with_force_fd` | 有限差分による接線検証 | 相対誤差 < 1e-5 |

#### 複数ペア

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_two_pairs_superposition` | 複数ペアの力の重ね合わせ | p_n=100, 200 |

### ソルバー統合

**テストファイル**: `tests/contact/test_solver_hooks.py`（C2 部分）

#### 変形座標

| テスト名 | 検証内容 |
|----------|---------|
| `test_zero_displacement` | u=0 → 変形座標 = 参照座標 |
| `test_translation` | 並進変位の加算 |

#### 接触なし基線

| テスト名 | 検証内容 |
|----------|---------|
| `test_no_contact_reduces_to_nr` | 小荷重（上向き）→ 接触なし, 標準NR結果 |

#### 接触付きNR

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_two_beams_pressed_together` | 押し付け → 接触検出＆安定 | Δu=±50 |
| `test_contact_prevents_penetration` | 大荷重でも過大貫入を阻止 | f=100N |
| `test_contact_force_history` | 接触力履歴の記録 | 10荷重ステップ |

#### ContactManager Phase C2

| テスト名 | 検証内容 |
|----------|---------|
| `test_evaluate_contact_forces` | 全ペアの p_n 計算 |
| `test_update_al_multipliers` | 全ペアの λ_n 更新 |
| `test_initialize_penalty` | k_pen, k_t の初期化 |
| `test_initialize_penalty_skips_inactive` | INACTIVE ペアはスキップ |
| `test_initialize_penalty_skips_already_set` | 設定済み k_pen の上書き防止 |

---

## Phase C3: Coulomb 摩擦

**テストファイル**: `tests/contact/test_law_friction.py`

### Return Mapping

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_stick_small_displacement` | Stick: q = z_t + k_t·Δu_t（弾性） | Δu=0.001, k_t=1000, μ=0.3 |
| `test_slip_large_displacement` | Slip: \|q\| = μ·p_n, q_trial方向 | Δu=0.01, k_t=1000, p_n=10 |
| `test_slip_2d_direction` | 2D接線空間でのスリップ方向保存 | Δu_t=[0.01,0.01] (45°) |
| `test_stick_to_slip_transition` | 累積 z_t によるスリップ遷移 | 2ステップ |
| `test_dissipation_nonnegative_stick` | Stick: D_inc ≥ 0 | D ≥ -1e-15 |
| `test_dissipation_nonnegative_slip` | Slip: D_inc ≥ 0（熱力学整合性） | D ≥ -1e-15 |
| `test_zero_normal_force` | p_n=0 → q=[0,0] | — |
| `test_inactive_pair` | INACTIVE → q=[0,0] | — |
| `test_zero_friction_coefficient` | μ=0 → q=[0,0] | — |
| `test_z_t_updated_correctly` | z_t ← q_returned（履歴更新） | — |

### 摩擦接線剛性

| テスト名 | 検証内容 | 許容条件 |
|----------|---------|---------|
| `test_stick_tangent` | Stick: D_t = k_t·I₂ | k_t=1000 |
| `test_inactive_tangent_zero` | INACTIVE: D_t = 0₂×₂ | — |
| `test_slip_tangent_symmetric` | Slip: D_t が対称 | D_t = D_t^T |
| `test_slip_tangent_positive_semidefinite` | Slip: 固有値 ≥ 0 | eigvals ≥ -1e-12 |

### μランプ

| テスト名 | 検証内容 | 解析式 |
|----------|---------|--------|
| `test_no_ramp` | steps=0: μ_eff = μ_target | μ_eff = 0.3 |
| `test_ramp_start` | counter=0: μ_eff = 0 | μ_eff = 0 |
| `test_ramp_halfway` | counter=2, steps=4: μ_eff = 0.5·μ | μ_eff = 0.2 |
| `test_ramp_complete` | counter=steps: μ_eff = μ | μ_eff = 0.3 |
| `test_ramp_beyond` | counter > steps: μ にクランプ | μ_eff = 0.3 |

### 接線相対変位

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_relative_sliding` | t1方向のスライド抽出 | Δx=0.01 → Δu_t[0]=0.01 |
| `test_normal_displacement_no_tangential` | 法線変位は接線に寄与しない | Δz=0.01 → Δu_t=[0,0] |
| `test_shape_function_weighting` | (s,t) による重み付き補間 | t=0.75 |

### ソルバー統合（摩擦付き）

**テストファイル**: `tests/contact/test_solver_hooks.py`（C3 部分）

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_friction_contact_converges` | 摩擦接触の収束 | μ=0.3, ramp=3 |
| `test_friction_with_tangential_load` | 接線荷重による摩擦応答 | f_t=5 |
| `test_mu_ramp_converges` | μランプの安定収束 | ramp=5 |
| `test_friction_dissipation_nonnegative` | 全ペアで D ≥ 0 | -1e-12 |
| `test_use_friction_false_is_backward_compatible` | 摩擦OFF で C2 互換 | — |

---

## Phase C4: Merit line search

**テストファイル**: `tests/contact/test_line_search.py`

### Merit 関数

| テスト名 | 検証内容 | 解析式 |
|----------|---------|--------|
| `test_zero_residual_no_contact` | R=0, 接触なし → merit=0 | — |
| `test_residual_only` | merit = \|R\| | R=[3,4] → 5 |
| `test_penetration_penalty` | Φ += α·gap²（貫入ペナルティ） | gap=-0.1 → +0.01 |
| `test_no_penalty_for_positive_gap` | gap > 0 → ペナルティなし | — |
| `test_dissipation_penalty` | Φ += β·D（散逸ペナルティ） | D=0.5 → +0.5 |
| `test_negative_dissipation_ignored` | D < 0 → 無視 | — |
| `test_combined_merit` | 残差＋貫入＋散逸の統合 | — |
| `test_alpha_weight` | α 重み係数 | α=10 → 10倍 |
| `test_beta_weight` | β 重み係数 | β=5 → 5倍 |
| `test_inactive_pairs_ignored` | INACTIVE は merit に寄与しない | — |
| `test_multiple_penetrations` | 複数ペアの貫入和 | Σgap² |
| `test_merit_nonnegative` | merit ≥ 0 | — |

### Backtracking line search

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_full_step_accepted` | merit 十分低下 → η=1 受理 | n_steps=1 |
| `test_backtracking_needed` | Armijo 不成立 → バックトラック | — |
| `test_zero_merit_accepts_full_step` | Φ_current=0 → 即 η=1 | n_steps=0 |
| `test_best_eta_returned_on_failure` | 全ステップ失敗 → 最良 η | — |
| `test_shrink_factor` | η が因子で縮小 | shrink=0.5 |
| `test_armijo_condition` | Armijo 条件: Φ_trial ≤ Φ·(1 − c·η) | c=0.1 |
| `test_armijo_condition_marginal_fail` | 限界的失敗 → 次ステップ成功 | — |
| `test_max_steps_respected` | max_steps の制限遵守 | — |
| `test_best_of_multiple_trials` | 複数トライアルから最良選択 | — |

### ソルバー統合（line search 付き）

**テストファイル**: `tests/contact/test_solver_hooks.py`（C4 部分）

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_line_search_converges_normal_contact` | line search ON で法線接触収束 | use_line_search=True |
| `test_line_search_prevents_penetration` | line search + 大荷重で貫入阻止 | f_z=100 |
| `test_line_search_with_friction` | line search + 摩擦の収束 | 両方ON |
| `test_line_search_result_has_ls_steps` | LS ステップ数フィールドの存在 | ≥ 0 |
| `test_line_search_disabled_backward_compatible` | LS OFF → ステップ数=0 | — |

---

## Phase C5: 一貫接線・PDAS・バリデーション

### 幾何剛性 K_geo

**テストファイル**: `tests/contact/test_consistent_tangent.py`

| テスト名 | 検証内容 | 許容誤差 |
|----------|---------|---------|
| `test_zero_for_inactive` | INACTIVE → K_geo = 0 | — |
| `test_zero_for_zero_pn` | p_n = 0 → K_geo = 0 | — |
| `test_symmetry` | K_geo = K_geo^T | 1e-14 |
| `test_normal_direction_zero` | K_geo·g_n ≈ 0（法線方向はゼロ） | — |
| `test_tangential_nonzero` | 接線方向は非ゼロ | — |
| `test_negative_semidefinite` | 固有値 ≤ 0（p_n > 0 で負半定値） | — |
| `test_scales_with_pn` | K_geo ∝ p_n | K(2p_n) ≈ 2·K(p_n) |
| `test_scales_with_inverse_dist` | K_geo ∝ 1/distance | — |
| `test_finite_difference_verification` | 有限差分検証 | — |
| `test_included_in_stiffness` | 全体 K_c への組み込み | — |

### Slip consistent tangent

| テスト名 | 検証内容 | 許容条件 |
|----------|---------|---------|
| `test_stick_tangent_unchanged` | Stick: D_t = k_t·I₂（C3 と同一） | — |
| `test_slip_tangent_different_from_stick` | Slip: D_t ≠ k_t·I₂ | — |
| `test_slip_tangent_formula` | D_t = (μ·p_n/\|q_trial\|)·k_t·(I₂ − q̂⊗q̂) | 公式検証 |
| `test_slip_tangent_symmetric` | D_t = D_t^T | — |
| `test_slip_tangent_positive_semidefinite` | 固有値 ≥ 0 | — |
| `test_slip_tangent_rank_one_deficient` | (I₂ − q̂q̂^T) によるランク落ち | 1固有値 ≈ 0 |
| `test_q_trial_norm_stored` | q_trial_norm の記録 | — |
| `test_q_trial_norm_stick_case` | Stick でも q_trial_norm 保持 | — |

### 平行輸送フレーム

| テスト名 | 検証内容 | 許容誤差 |
|----------|---------|---------|
| `test_identity_rotation` | 回転なし: t1 不変 | — |
| `test_small_rotation` | 微小回転: 滑らかな追従 | θ=0.01 |
| `test_90_degree_rotation` | 90°回転: 直交性保存 | — |
| `test_preserves_orthogonality` | t1_new ⊥ n_new | dot ≈ 0 |
| `test_build_contact_frame_with_parallel_transport` | 平行輸送の統合 | — |
| `test_build_contact_frame_without_prev_normal` | Gram-Schmidt フォールバック | — |
| `test_frame_continuity_over_multiple_steps` | 多ステップでの連続性 | 10ステップ |

### PDAS

| テスト名 | 検証内容 |
|----------|---------|
| `test_pdas_config_default_off` | use_pdas デフォルト False |
| `test_geometric_stiffness_config_default_on` | use_geometric_stiffness デフォルト True |
| `test_q_trial_norm_in_state` | ContactState に q_trial_norm フィールド |
| `test_q_trial_norm_copied` | copy() で q_trial_norm もコピー |

### ソルバー統合（C5 機能）

**テストファイル**: `tests/contact/test_solver_hooks.py`（C5 部分）

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_geometric_stiffness_converges` | 幾何剛性 ON で収束 | use_geometric_stiffness=True |
| `test_geometric_stiffness_disabled` | 幾何剛性 OFF で後方互換 | — |
| `test_pdas_converges` | PDAS: 収束 | use_pdas=True |
| `test_slip_consistent_tangent_with_friction` | Slip consistent tangent + 摩擦 | f_t=10 |
| `test_pdas_with_friction` | PDAS + 摩擦 | 全機能 |
| `test_all_c5_features_combined` | C5 全機能統合テスト | 全ON |

### 摩擦物理バリデーション

**テストファイル**: `tests/contact/test_friction_validation.py`

#### Coulomb 条件

| テスト名 | 検証内容 | 許容条件 |
|----------|---------|---------|
| `test_coulomb_limit_satisfied` | \|q_t\| ≤ μ·p_n（全ペア） | Coulomb 錐内 |
| `test_slip_friction_equals_mu_pn` | Slip: \|q\| ≈ μ·p_n | 相対誤差 <1% |
| `test_stick_condition_small_tangential_load` | 小 f_t → 全 stick | f_t=1 |
| `test_friction_cone_two_axes` | 2D接線: \|q\| ≤ μ·p_n | f_t_x=15, f_t_y=15 |

#### 力均衡

| テスト名 | 検証内容 | 許容条件 |
|----------|---------|---------|
| `test_residual_zero_at_equilibrium` | f_ext = f_int + f_contact | 相対残差 <5% |
| `test_normal_force_positive_at_contact` | p_n ≥ 0（粘着なし） | 全活性ペア |

#### Stick-Slip 遷移

| テスト名 | 検証内容 |
|----------|---------|
| `test_increasing_tangential_load_causes_slip` | f_t 増加で stick→slip 遷移 |
| `test_slip_displacement_larger_than_stick` | Slip 変位 > stick 変位 |

#### エネルギー散逸

| テスト名 | 検証内容 | 許容条件 |
|----------|---------|---------|
| `test_slip_has_positive_dissipation` | Slip: D > 0 | — |
| `test_dissipation_nonnegative` | 全ペア: D ≥ 0 | 熱力学第二法則 |

#### 対称性

| テスト名 | 検証内容 |
|----------|---------|
| `test_opposite_tangential_load_gives_opposite_displacement` | ±f_t → 変位反転 |
| `test_no_tangential_load_gives_zero_tangential_displacement` | f_t=0 → u_t=0 |

#### μ依存性

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_zero_friction_no_tangential_resistance` | μ=0 → q=0 | \|q\| < 1e-10 |
| `test_higher_mu_gives_less_tangential_displacement` | μ増 → u_t 減 | μ=0.1 vs 0.5 |
| `test_friction_force_scales_with_mu` | q/p_n ≈ μ（Slip時） | ±5% |

#### 貫入影響

| テスト名 | 検証内容 |
|----------|---------|
| `test_friction_does_not_increase_penetration` | 摩擦による貫入増加 < 0.1% |

---

## 実梁要素接触テスト

### 梁梁貫入テスト（ばねモデル）

**テストファイル**: `tests/contact/test_beam_contact_penetration.py`

#### 接触検出

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_contact_detected_with_push_down` | z方向押し下げ → 接触検出 | f_z=50 |
| `test_no_contact_without_push_down` | f_z=0 → 接触なし | gap=2mm |

#### 適応的ペナルティ増大

| テスト名 | 検証内容 | 許容条件 |
|----------|---------|---------|
| `test_penetration_within_1_percent` | 適応ペナルティ: 貫入 < 1% search_radius | tol=1% |
| `test_penetration_with_large_force` | 大荷重でも適応で <1% | f_z=200 |
| `test_higher_penalty_reduces_penetration` | k_pen↑ → 貫入↓ | k=[1e4, 1e5, 1e6] |
| `test_adaptive_penalty_improves_penetration` | 適応 vs 非適応の比較 | — |

#### 法線力

| テスト名 | 検証内容 |
|----------|---------|
| `test_normal_force_positive` | p_n ≥ 0 |
| `test_normal_force_increases_with_push` | f_z↑ → p_n↑ |

#### 摩擦＋貫入

| テスト名 | 検証内容 |
|----------|---------|
| `test_penetration_bounded_with_friction` | 摩擦ON: 貫入 <1% |
| `test_friction_does_not_worsen_penetration` | 摩擦による貫入悪化 < 1e-3 |

#### 変位履歴

| テスト名 | 検証内容 |
|----------|---------|
| `test_z_displacement_progresses_downward` | z変位の単調減少（前半） |
| `test_x_tension_positive` | x引張 → u_x > 0 |

#### マルチセグメント

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_multi_segment_contact_detected` | 4セグメント: 接触検出 | n_seg=4 |
| `test_multi_segment_penetration_within_1_percent` | 4セグメント: 貫入 <1% | — |
| `test_multi_segment_large_force_penetration` | 4セグメント大荷重: <1% | f_z=100 |

#### スライド接触

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_sliding_contact_detected` | 横スライド + 押し下げ | f_z=50, f_x=20 |
| `test_sliding_penetration_within_1_percent` | スライド: 貫入 <1% | — |
| `test_sliding_with_friction_penetration` | 摩擦 + スライド: <1% | μ=0.3 |
| `test_sliding_displacement_has_x_component` | スライド荷重 → u_x > 0 | — |
| `test_sliding_friction_both_converge` | 摩擦 OFF/ON 両方収束 | — |

### 実梁要素接触テスト

**テストファイル**: `tests/contact/test_real_beam_contact.py`

**共通パラメータ**: Al E=70GPa, ν=0.33, 円形 d=20mm, L=0.5m, 4分割

#### Timoshenko 3D

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_contact_detected` | Timo3D 接触検出 | f_z=500 |
| `test_no_contact_without_force` | 無荷重で接触なし | — |
| `test_penetration_bounded` | 貫入 <2% search_radius | f_z=500 |
| `test_higher_penalty_reduces_penetration` | k_pen 依存性 | k=[1e4, 1e5] |
| `test_8_segment_contact` | 8セグメント接触 | n=8, f_z=200 |

#### CR梁

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_contact_detected` | CR梁接触検出 | f_z=500 |
| `test_no_contact_without_force` | 無荷重で接触なし | — |
| `test_penetration_bounded` | 貫入 <2% | f_z=500 |
| `test_8_segment_contact` | 8セグメント接触 | n=8, f_z=200 |

#### 比較

| テスト名 | 検証内容 | 許容条件 |
|----------|---------|---------|
| `test_small_load_response_similar` | Timo3D ≈ CR（小変位） | 相対差 <20% |

#### 自動 k_pen

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_estimate_k_pen_reasonable` | EI/L³ ベースの妥当な範囲 | 1 < k/k_bend < 10000 |
| `test_auto_k_pen_converges` | 自動 k_pen で収束 | — |

#### 摩擦付き実梁

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_timo3d_friction_converges` | Timo3D + 摩擦: 収束 | μ=0.3 |
| `test_cr_beam_friction_converges` | CR + 摩擦: 収束 | μ=0.3 |
| `test_friction_does_not_worsen_penetration` | 摩擦貫入悪化 ≤ 3倍 | — |

#### 長距離スライド

| テスト名 | 検証内容 | 主要パラメータ |
|----------|---------|--------------|
| `test_slide_contact_detected` | 8seg スライド接触検出 | f_z=200, f_x=100 |
| `test_slide_penetration_bounded` | スライド貫入 <2% | — |
| `test_slide_x_displacement_positive` | スライド → u_x > 0 | f_x=200 |
| `test_segment_boundary_crossing` | セグメント境界近傍の接触 | x=L/8 |
| `test_slide_with_friction` | 摩擦 + スライド: 貫入 <2% | μ=0.3 |
| `test_cr_beam_slide` | CR梁スライド接触 | f_x=100 |

---

## テスト統計

| フェーズ | テストファイル数 | テスト関数数 | 主要検証対象 |
|---------|----------------|------------|------------|
| C0 | 1 (test_pair.py 一部) | ~15 | 状態管理・ペア構成 |
| C1 | 3 (geometry, broadphase, pair 一部) | ~45 | 幾何, AABB, Active-set |
| C2 | 3 (law_normal, assembly, solver_hooks 一部) | ~35 | 法線AL, アセンブリ, NR |
| C3 | 2 (law_friction, solver_hooks 一部) | ~25 | Coulomb 摩擦, μランプ |
| C4 | 2 (line_search, solver_hooks 一部) | ~25 | Merit 関数, backtracking |
| C5 | 4 (consistent_tangent, friction_validation, solver_hooks 一部, etc.) | ~55 | 幾何剛性, PDAS, 物理バリデーション |
| 梁接触 | 2 (beam_contact_penetration, real_beam_contact) | ~40 | 適応ペナルティ, 実梁, スライド |
| **合計** | **12** | **~240** | — |

---

## 参考文献

1. Wriggers, P. (2006). *Computational Contact Mechanics.* Springer, Ch.5–10.
2. Laursen, T.A. (2002). *Computational Contact and Impact Mechanics.* Springer.
3. Simo, J.C. & Laursen, T.A. (1992). "An augmented Lagrangian treatment of contact problems involving friction." *Comp. Struct.*, 42, 97–116.

---
