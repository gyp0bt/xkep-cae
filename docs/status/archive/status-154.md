# status-154: プロセスアーキテクチャ Phase 3 — Strategy 実ロジック移植 + ファクトリ関数

[← status-index](status-index.md) | [← README](../../README.md)

**日付**: 2026-03-10
**テスト数**: 2445（+35）

## 概要

status-153（Phase 2: Strategy 具象実装 13クラス）を受け、Phase 3 として
Strategy スタブに solver_ncp.py の実ロジックを移植し、各 Strategy に対応する
ファクトリ関数 `create_*_strategy()` を追加した。

## U1判断: Process vs Protocol

ProcessMetaclass のオーバーヘッドを計測し、**Strategy を Process として維持する**判断を行った。

| 呼び出し方法 | 所要時間 |
|-------------|---------|
| Protocol (bare) | 0.064 μs/call |
| SolverProcess (direct) | 0.063 μs/call |
| SolverProcess (process()) | 0.855 μs/call |
| np.dot(100) | 0.581 μs/call |

**結論**: process() ラッピングオーバーヘッド 0.8μs/call。NR反復全体で 0.079ms/step。
典型的NR反復の1ステップ計算時間（数百ms〜数秒）に対して無視できる。
→ **Strategy は Process として維持。NR反復内では直接メソッド呼び出しを推奨。**

## 成果物

### 実ロジック移植（4 Strategy 軸）

| Strategy | 移植元 | 移植内容 |
|----------|--------|---------|
| ContactForce (NCP) | `_compute_contact_force_from_lambdas()` | evaluate() に接触力アセンブリ完全移植 |
| ContactForce (Smooth) | `assemble_smooth_contact()` | evaluate() に softplus 力 + 接触力アセンブリ。tangent() に解析的接触剛性 |
| Friction (Coulomb) | `_compute_friction_forces_ncp()` | evaluate() に return mapping + 摩擦力アセンブリ |
| Friction (tangent) | `_build_friction_stiffness()` | tangent() に COO 形式摩擦剛性行列構築 |
| TimeIntegration | solver_ncp lines 1659-1676 | ファクトリで QuasiStatic/GeneralizedAlpha 自動選択 |
| Penalty | solver_ncp lines 1725-1810 | ファクトリで AutoBeamEI/AutoEAL/Manual/Continuation 自動選択 |

### ファクトリ関数（4関数）

| ファクトリ | 入力パラメータ | 返却型 |
|-----------|--------------|--------|
| `create_penalty_strategy()` | k_pen, manager, node_coords_ref, connectivity | AutoBeamEI/AutoEAL/Manual/Continuation |
| `create_time_integration_strategy()` | mass_matrix, damping_matrix, dt_physical, rho_inf | QuasiStatic/GeneralizedAlpha |
| `create_friction_strategy()` | use_friction, contact_mode, ndof, k_pen | NoFriction/Coulomb/SmoothPenalty |
| `create_contact_force_strategy()` | contact_mode, ndof, contact_compliance, smoothing_delta | NCP/SmoothPenalty |

### テスト追加（+35テスト）

| テストファイル | 追加テスト数 | 内容 |
|--------------|------------|------|
| `test_penalty.py` | +8 | create_penalty_strategy ファクトリ |
| `test_time_integration.py` | +6 | create_time_integration_strategy ファクトリ |
| `test_friction.py` | +8 | create_friction_strategy ファクトリ |
| `test_contact_force.py` | +8 | create_contact_force_strategy + softplus導関数 |
| `benchmark_process_overhead.py` | — | U1判断用ベンチマーク（scripts/） |

### 更新ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/process/strategies/penalty.py` | create_penalty_strategy() ファクトリ追加 |
| `xkep_cae/process/strategies/time_integration.py` | create_time_integration_strategy() ファクトリ追加 |
| `xkep_cae/process/strategies/friction.py` | 実ロジック移植 + create_friction_strategy() |
| `xkep_cae/process/strategies/contact_force.py` | 実ロジック移植 + create_contact_force_strategy() + softplus導関数 |
| `xkep_cae/process/strategies/__init__.py` | 4ファクトリ関数のエクスポート追加 |
| `xkep_cae/process/strategies/test_*.py` | 各テストにファクトリテスト追加 |

## ファイル構造（Phase 3 完了後）

```
xkep_cae/process/strategies/
├── __init__.py              ← 13具象 + 5 Protocol + 4 ファクトリ エクスポート
├── protocols.py             ← 5 Strategy Protocol
├── compatibility.py         ← 互換性マトリクス
├── test_protocols.py        ← Protocol テスト
├── penalty.py               ← Penalty 4具象 + create_penalty_strategy()
├── test_penalty.py          ← 30件（+8）
├── time_integration.py      ← TimeInt 2具象 + create_time_integration_strategy()
├── test_time_integration.py ← 26件（+6）
├── friction.py              ← Friction 3具象【実ロジック移植済】+ create_friction_strategy()
├── test_friction.py         ← 26件（+8）
├── contact_force.py         ← ContactForce 2具象【実ロジック移植済】+ create_contact_force_strategy()
├── test_contact_force.py    ← 28件（+8）
├── contact_geometry.py      ← ContactGeom 3具象（Phase 4で移植予定）
└── test_contact_geometry.py ← 20件
```

## 設計判断

| 判断 | 選択 | 理由 |
|------|------|------|
| U1: Process vs Protocol | Process 維持 | オーバーヘッド 0.8μs/call、NR全体で 0.079ms/step |
| NR内呼び出し方式 | 直接メソッド (compute_k_pen 等) | process() 経由は不要 |
| ファクトリ分離 | Strategy モジュール内に配置 | solver_ncp.py の依存逆転準備 |
| ContactGeometry の移植 | Phase 4 に延期 | manager API との結合度が高く、別セッションで段階的に移植 |
| Friction のロジック重複 | CoulombReturnMapping と SmoothPenaltyFriction で共通ロジック | 共通化は Phase 4 で検討 |

## CI 状況

### fast-test: 全通過
### slow-test: 3件の収束失敗（リファクタリングとは無関係）

| テスト | エラー | 原因 |
|--------|--------|------|
| `test_ncp_7strand_bending_90deg` | phase1_converged=False | NCP収束安定性 |
| `test_timo3d_friction_converges` | converged=False | 摩擦付きNCP収束 |
| `test_ncp_61strand_radial_layer1` | pytest INTERNALERROR | OOM/タイムアウト |

## TODO（次セッション: Phase 4）

- [ ] Phase 4: ContactGeometry Strategy 実ロジック移植（manager API との結合度に注意）
- [ ] Phase 5: solver_ncp.py の newton_raphson_contact_ncp() を Strategy 注入に書き換え
  - ファクトリ関数を使って Strategy を構築し、NR ループ内で Strategy.evaluate()/tangent() を呼ぶ
  - 段階的移行: まず Penalty → TimeIntegration → Friction → ContactForce の順
- [ ] Friction Strategy のCoulomb/SmoothPenalty 共通ロジック抽出（DRY化）
- [ ] slow-test 3件の収束問題対応（xfail 追加 or 収束改善）
- [ ] 全テストのmm-ton-MPa移行（status-149 TODO継承）
- [ ] 被膜摩擦μ=0.25の収束達成（status-149 TODO継承）
- [ ] 19本→37本のスケールアップ（status-149 TODO継承）

## 懸念事項・確認事項

- **solver_ncp.py との二重実装**: 現在は Strategy に実ロジックを移植したが、solver_ncp.py 内の元のヘルパー関数は残存。Phase 5 で solver_ncp.py を Strategy 経由に書き換えた後、旧ヘルパー関数を deprecated 化 → 削除する予定。
- **Friction evaluate() のシグネチャ拡張**: Protocol の基本シグネチャ (u, contact_pairs, mu) に加えて、keyword-only で lambdas, u_ref, node_coords_ref を受け取る設計。Protocol 準拠性は維持しつつ実用性を確保。
- **ContactGeometry の移植は複雑**: manager.update_geometry(), _build_constraint_jacobian() など manager API と密結合。Phase 4 で段階的に実施。

---
