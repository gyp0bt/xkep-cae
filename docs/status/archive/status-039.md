# status-039: Phase C2 — 法線AL接触力 + 接触接線剛性 + 接触付きNRソルバー

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-19

## 概要

Phase C2 を実装完了。法線接触力（Augmented Lagrangian）、接触接線剛性（主項 K_c = k_pen * g * g^T）、
接触付き Newton-Raphson ソルバー（Outer/Inner 分離）を実装し、摩擦なし梁–梁接触が安定収束することを検証。

テスト数 802 → 845（+43テスト）。

## 実施内容

### 1. `law_normal.py` 新設 — 法線接触力則（Augmented Lagrangian）

| 関数 | 概要 |
|------|------|
| `evaluate_normal_force(pair)` | AL 反力: p_n = max(0, λ_n + k_pen · (-g)) |
| `update_al_multiplier(pair)` | AL 乗数更新: λ_n ← p_n（Outer loop 終了時） |
| `normal_force_linearization(pair)` | 接線剛性主項係数: ACTIVE かつ p_n > 0 → k_pen, それ以外 → 0 |
| `initialize_penalty_stiffness(pair, k_pen, k_t_ratio)` | ペナルティ剛性初期化 |
| `auto_penalty_stiffness(E, A, L, scale)` | EA/L ベースのペナルティ自動推定 |

### 2. `assembly.py` 新設 — 接触内力・接触接線剛性のアセンブリ

| 関数 | 概要 |
|------|------|
| `_contact_shape_vector(pair)` | 法線方向形状ベクトル g (12成分: 4節点 × 3DOF) |
| `compute_contact_force(manager, ndof_total)` | 接触内力ベクトル f_c（全ACTIVE ペアの p_n × g を全体DOFに組み込み） |
| `compute_contact_stiffness(manager, ndof_total)` | 接触接線剛性 K_c = Σ k_eff · g · g^T（COO→CSR, 対称・半正定値・ランク1/ペア） |

#### 技術詳細

- 形状ベクトル: g = [-(1-s)n, -sn, (1-t)n, tn]（A側に-n方向、B側に+n方向の力配分）
- 接触力は並進DOF（3成分）のみに寄与（回転DOFへの寄与ゼロ）
- 作用・反作用の法則: Σg = 0（全節点の合力がゼロ）

### 3. `solver_hooks.py` 新設 — 接触付き Newton-Raphson ソルバー

#### アルゴリズム（Outer/Inner 分離）

```
for 荷重ステップ:
    for outer in range(n_outer_max):
        1. 変形座標計算
        2. detect_candidates（broadphase + 新規ペア追加）
        3. update_geometry（narrowphase: s, t, gap, normal 更新）
        4. initialize_penalty（未設定ペアの k_pen 初期化）
        for inner NR:
            5. _update_gaps_fixed_st（s,t 固定で gap のみ変位に基づき再計算）
            6. f_int + f_c → 残差
            7. K_struct + K_c → 接線剛性
            8. BC 適用 → 連立方程式求解
            9. 収束判定（力/変位/エネルギーノルム）
        10. Outer 収束判定（|Δs|, |Δt| < tol_geometry）
        11. AL 乗数更新（λ_n ← p_n）
```

#### 主要な設計判断

- **Inner loop 内の gap 更新**: `_update_gaps_fixed_st()` により、NR 反復ごとに変位ベースで gap を再計算。
  s, t, normal は Outer loop で確定した値を保持し、gap のみ現在の変位で再評価する。
  これにより接触力 f_c と接触剛性 K_c の整合性が保たれ、NR が安定収束する。
  gap を固定したままだと K_c との不整合で収束速度が極端に低下（理論的にスペクトル半径 ≈ k_pen/(k_pen+k_struct)）。
- **Outer loop 上限到達時**: Inner が収束していれば結果を受容（step_converged = True）

#### `ContactSolveResult` データクラス

| フィールド | 型 | 概要 |
|-----------|-----|------|
| u | ndarray | 最終変位ベクトル |
| converged | bool | 全ステップ収束 |
| n_load_steps | int | 荷重増分数 |
| total_newton_iterations | int | 全 NR 反復合計 |
| total_outer_iterations | int | 全 Outer 反復合計 |
| n_active_final | int | 最終 ACTIVE ペア数 |
| load_history | list[float] | 各ステップの荷重係数 |
| displacement_history | list[ndarray] | 各ステップの変位 |
| contact_force_history | list[float] | 各ステップの接触力ノルム |

### 4. `pair.py` 拡張 — ContactManager Phase C2 メソッド

| メソッド | 概要 |
|---------|------|
| `evaluate_contact_forces()` | 全ペアの p_n 評価（evaluate_normal_force 呼び出し） |
| `update_al_multipliers()` | 全ペアの λ_n 更新（update_al_multiplier 呼び出し） |
| `initialize_penalty(k_pen, k_t_ratio)` | ACTIVE ペアの k_pen/k_t 初期化（未設定のみ） |

### 5. `contact/__init__.py` 更新

Phase C2 の全シンボルをエクスポート:
- `compute_contact_force`, `compute_contact_stiffness`
- `evaluate_normal_force`, `update_al_multiplier`, `normal_force_linearization`, `initialize_penalty_stiffness`, `auto_penalty_stiffness`
- `ContactSolveResult`, `newton_raphson_with_contact`

### 6. テスト — +43テスト

#### `tests/contact/test_law_normal.py` — 17テスト（新規ファイル）

| クラス | テスト数 | 内容 |
|--------|---------|------|
| TestEvaluateNormalForce | 6 | 貫通→p_n>0, 離間→p_n=0, INACTIVE→0, λ付きAL, 非負性, 大貫通 |
| TestUpdateALMultiplier | 3 | λ←p_n, INACTIVE→λ=0, 連続更新 |
| TestNormalForceLinearization | 3 | 接触中→k_pen, 離間→0, INACTIVE→0 |
| TestInitializePenalty | 2 | k_pen/k_t設定, k_t_ratio反映 |
| TestAutoPenaltyStiffness | 2 | EA/L計算, スケール係数 |

#### `tests/contact/test_contact_assembly.py` — 14テスト（新規ファイル）

| クラス | テスト数 | 内容 |
|--------|---------|------|
| TestContactShapeVector | 3 | 中点z法線, 端点s=0/t=1, 作用反作用バランス |
| TestComputeContactForce | 5 | 1ペア中点, INACTIVE無寄与, 離間無力, 合力ゼロ, 回転DOFゼロ |
| TestComputeContactStiffness | 5 | 対称性, 半正定値, ランク1, INACTIVE空行列, 有限差分整合性 |
| TestMultiplePairs | 1 | 2ペア独立加算 |

#### `tests/contact/test_solver_hooks.py` — 11テスト（新規ファイル）

| クラス | テスト数 | 内容 |
|--------|---------|------|
| TestDeformedCoords | 2 | ゼロ変位, 並進加算 |
| TestContactSolverNoContact | 1 | 接触なし→通常NR同等 |
| TestContactSolverWithContact | 3 | 2梁押し付け→安定収束, 貫通防止, 接触力履歴記録 |
| TestContactManagerPhaseC2 | 5 | evaluate_contact_forces, update_al_multipliers, initialize_penalty, INACTIVE skip, already-set skip |

##### 統合テスト（TestContactSolverWithContact）の設計

- **交差ビーム配置**: 梁A（x方向）と梁B（y方向）が中点で直交交差
  - 梁A: node0 [0,0,+z_sep] → node1 [1,0,+z_sep]
  - 梁B: node2 [0.5,-0.5,-z_sep] → node3 [0.5,0.5,-z_sep]
- **z方向ばねモデル**: 各要素2節点間にz方向のみのばね剛性
- **BC**: node0/node2のz固定、全ノードのx/y/回転を固定
- **荷重**: 自由端(node1/node3)をz方向に互いに押し付ける
- 交差配置により closest_point_segments が s≈0.5, t≈0.5 を返し、接触力が固定端・自由端の両方に分配される

### 7. 既存テストへの影響

- **既存テスト 802 件**: 全パス（破壊なし）
- **新規テスト 43 件**: 全パス
- **合計 845 件**: 全パス、24 skipped

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/law_normal.py` | **新規** — 法線AL接触力則（evaluate_normal_force, update_al_multiplier 等） |
| `xkep_cae/contact/assembly.py` | **新規** — 接触内力・接触接線剛性アセンブリ |
| `xkep_cae/contact/solver_hooks.py` | **新規** — 接触付きNRソルバー（Outer/Inner分離, _update_gaps_fixed_st） |
| `xkep_cae/contact/pair.py` | Phase C2 メソッド追加（evaluate_contact_forces, update_al_multipliers, initialize_penalty） |
| `xkep_cae/contact/__init__.py` | Phase C2 全シンボルのエクスポート追加 |
| `tests/contact/test_law_normal.py` | **新規** — 17テスト |
| `tests/contact/test_contact_assembly.py` | **新規** — 14テスト |
| `tests/contact/test_solver_hooks.py` | **新規** — 11テスト（交差ビーム統合テスト含む） |
| `README.md` | テスト数・Phase C2 反映 |
| `docs/roadmap.md` | Phase C2 完了チェック・現在地更新 |
| `docs/status/status-index.md` | status-039 行追加 |

## テスト数

802 → 845（+43テスト）

## 確認事項・懸念

1. **幾何微分（dn/du）の未実装**: v0.1 では接触接線剛性の主項のみ（k_pen · g · g^T）。法線方向の変化に伴う幾何微分項は v0.2 で追加予定。強い接触角度変化がある場合は Outer loop 収束が遅くなる可能性
2. **接線方向（摩擦）の未実装**: Phase C3 で Coulomb 摩擦（return mapping + μランプ）を実装予定
3. **ペナルティ vs 構造剛性の比率**: k_pen が k_struct に対して大きすぎると NR が不安定化する可能性。auto_penalty_stiffness による EA/L ベースの推定で対処する設計だが、極端なケースでは k_pen_scale の調整が必要
4. **Inner loop gap 更新の必要性**: Inner NR ループ内で gap を変位ベースで再計算する `_update_gaps_fixed_st()` が必須。gap を固定したまま K_contact を追加すると linearization の不整合でスペクトル半径 ≈ k_pen/(k_pen+k_struct) の遅い収束になる

## TODO

- [ ] Phase C3: 摩擦 return mapping + μランプ
- [ ] Phase C4: merit line search + 探索/求解分離の運用強化
- [ ] examples の .inp ファイルを使った実際の解析実行スクリプトの追加（Phase C 完了後）

---
