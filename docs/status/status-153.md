# status-153: プロセスアーキテクチャ Phase 2 — Strategy 具象実装

[← status-index](status-index.md) | [← README](../../README.md)

**日付**: 2026-03-10
**テスト数**: 2410（+100）

## 概要

status-151 で策定した Phase 1 基盤（Protocol + AbstractProcess）の上に、
Phase 2 として 13 個の Strategy 具象実装を作成した。
5つの Strategy 軸（ContactForce, Friction, TimeIntegration, ContactGeometry, Penalty）
それぞれに対して具象クラスを実装し、全て Protocol 準拠テストを含む
100テストを追加。全テスト通過・lint clean。

## 成果物

### 新規ファイル（10ファイル）

| ファイル | 内容 | テスト数 |
|---------|------|---------|
| `xkep_cae/process/strategies/penalty.py` | AutoBeamEI, AutoEAL, Manual, Continuation | — |
| `xkep_cae/process/strategies/test_penalty.py` | Penalty 4クラスの1:1テスト | 22 |
| `xkep_cae/process/strategies/time_integration.py` | QuasiStatic, GeneralizedAlpha | — |
| `xkep_cae/process/strategies/test_time_integration.py` | TimeIntegration 2クラスの1:1テスト | 20 |
| `xkep_cae/process/strategies/friction.py` | NoFriction, CoulombReturnMapping, SmoothPenalty | — |
| `xkep_cae/process/strategies/test_friction.py` | Friction 3クラスの1:1テスト | 18 |
| `xkep_cae/process/strategies/contact_force.py` | NCP, SmoothPenalty | — |
| `xkep_cae/process/strategies/test_contact_force.py` | ContactForce 2クラスの1:1テスト | 20 |
| `xkep_cae/process/strategies/contact_geometry.py` | PointToPoint, LineToLineGauss, Mortar | — |
| `xkep_cae/process/strategies/test_contact_geometry.py` | ContactGeometry 3クラスの1:1テスト | 20 |

### 更新ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/process/strategies/__init__.py` | 13具象クラスのエクスポート追加 |

### 実装された Strategy 具象クラス（13クラス）

#### ContactForce Strategy（§2.1）
| クラス名 | 概要 | Phase 3 移植元 |
|----------|------|---------------|
| `NCPContactForceProcess` | Alart-Curnier NCP + 鞍点系 | `_compute_contact_force_from_lambdas` |
| `SmoothPenaltyContactForceProcess` | softplus + Uzawa外部ループ | smooth penalty Uzawa loop |

#### Friction Strategy（§2.2）
| クラス名 | 概要 | Phase 3 移植元 |
|----------|------|---------------|
| `NoFrictionProcess` | 摩擦なし（ゼロ返却） | — |
| `CoulombReturnMappingProcess` | Coulomb return mapping | `_compute_friction_forces_ncp` |
| `SmoothPenaltyFrictionProcess` | smooth penalty + Uzawa | smooth penalty 摩擦ループ |

#### TimeIntegration Strategy（§2.3）
| クラス名 | 概要 | Phase 3 移植元 |
|----------|------|---------------|
| `QuasiStaticProcess` | 準静的（identity操作） | — |
| `GeneralizedAlphaProcess` | Generalized-α動的解析 | Newmark-β/Gen-α初期化・predict・correct |

#### ContactGeometry Strategy（§2.4）
| クラス名 | 概要 | Phase 3 移植元 |
|----------|------|---------------|
| `PointToPointProcess` | 最近接点ペア | ContactManager.update_geometry |
| `LineToLineGaussProcess` | L2L Gauss積分 | line_contact.py |
| `MortarSegmentProcess` | Mortar法セグメント | mortar.py |

#### Penalty Strategy（§2.5）
| クラス名 | 概要 | Phase 3 移植元 |
|----------|------|---------------|
| `AutoBeamEIProcess` | EI/L³ベース自動推定 | `auto_beam_penalty_stiffness` |
| `AutoEALProcess` | EA/Lベース自動推定 | `auto_penalty_stiffness` |
| `ManualPenaltyProcess` | 手動指定（deprecated） | — |
| `ContinuationPenaltyProcess` | 段階的増加 | k_pen continuation |

## 設計判断

| 判断 | 選択 | 理由 |
|------|------|------|
| Strategy の粒度 | Process 単位で実装 | Phase 1 の AbstractProcess/SolverProcess を再利用。binds_toによる1:1テスト紐付け |
| Phase 3 移植の範囲 | スタブ + Protocol準拠シグネチャ | 既存solver_ncp.pyへの影響ゼロ。Phase 3で実ロジック移植 |
| ManualPenaltyProcess | deprecated マーカー付与 | status-140 で材料ベース自動推定が標準化済み |
| GeneralizedAlphaProcess | 状態保持型 | predict/correct で内部状態（vel, acc）を更新。QuasiStaticは無状態 |
| Softplus 実装 | SmoothPenaltyContactForceProcess に直接実装 | 数値安定性のため np.log1p + overflow guard |

## ファイル構造（Phase 2 完了後）

```
xkep_cae/process/strategies/
├── __init__.py              ← 13具象 + 5 Protocol エクスポート
├── protocols.py             ← 5 Strategy Protocol（Phase 1）
├── compatibility.py         ← 互換性マトリクス（Phase 1）
├── test_protocols.py        ← Protocol テスト（Phase 1）
├── penalty.py               ← Penalty 4具象        ★NEW
├── test_penalty.py          ← Penalty テスト 22件   ★NEW
├── time_integration.py      ← TimeInt 2具象        ★NEW
├── test_time_integration.py ← TimeInt テスト 20件   ★NEW
├── friction.py              ← Friction 3具象       ★NEW
├── test_friction.py         ← Friction テスト 18件  ★NEW
├── contact_force.py         ← ContactForce 2具象   ★NEW
├── test_contact_force.py    ← ContactForce テスト 20件 ★NEW
├── contact_geometry.py      ← ContactGeom 3具象    ★NEW
└── test_contact_geometry.py ← ContactGeom テスト 20件 ★NEW
```

## TODO（次セッション: Phase 3）

- [ ] Phase 3 開始: solver_ncp.py から Strategy への実ロジック移植
  - evaluate()/tangent() の実装を solver_ncp.py の関数から移植
  - solver_ncp.py の newton_raphson_contact_ncp() を Strategy 注入に書き換え
- [ ] U1 判断: Strategy を Process として維持するか Protocol に降格するか（オーバーヘッド計測）
- [ ] 全テストのmm-ton-MPa移行（status-149 TODO継承）
- [ ] 被膜摩擦μ=0.25の収束達成（status-149 TODO継承）
- [ ] 19本→37本のスケールアップ（status-149 TODO継承）

## 懸念事項・確認事項

- **Phase 3 の影響範囲**: solver_ncp.py の newton_raphson_contact_ncp() は72個のキーワード引数を持つ巨大関数。Strategy 注入への書き換えは段階的に行う必要がある。
- **オーバーヘッド**: AbstractProcess のメタクラスラップにより、process() 呼び出しごとにプロファイリングコストが発生。高頻度呼び出し（Newton反復内）では Protocol に降格する判断が必要な可能性あり（U1判断）。
- **CoulombReturnMapping + NCP の非互換**: status-147 で確認済み。compatibility.py のブラックリストで管理中。

---
