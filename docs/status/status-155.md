# status-155: プロセスアーキテクチャ Phase 4 — ContactGeometry 実ロジック移植 + ファクトリ関数

[← status-index](status-index.md) | [← README](../../README.md)

**日付**: 2026-03-10
**テスト数**: 2471（+26）

## 概要

status-154（Phase 3: Strategy 実ロジック移植 + ファクトリ）を受け、Phase 4 として
ContactGeometry Strategy に実ロジックを移植し、`create_contact_geometry_strategy()` ファクトリを追加した。

## 成果物

### Protocol 拡張

ContactGeometryStrategy Protocol に2メソッドを追加:

| メソッド | 用途 |
|---------|------|
| `update_geometry(pairs, node_coords, *, config)` | 全ペアの幾何情報（s, t, gap, frame）更新 |
| `build_constraint_jacobian(pairs, ndof_total, ndof_per_node)` | 制約ヤコビアン G = ∂g_n/∂u 構築 |

### 実ロジック移植（3 Strategy）

| Strategy | 移植元 | 移植内容 |
|----------|--------|---------
| PointToPoint | `ContactManager.update_geometry()` | バッチ最近接点 + ギャップ + フレーム + Active-set ヒステリシス |
| LineToLineGauss | `ContactManager.update_geometry()` + `line_contact.auto_select_n_gauss()` | PtPベース + Gauss点数自動選択 |
| MortarSegment | `ContactManager.update_geometry()` | PtPベース（完全Mortar制約はPhase 5で統合） |
| 共通 | `solver_ncp._build_constraint_jacobian()` | 制約ヤコビアン G 行列構築 |

### ファクトリ関数

| ファクトリ | 入力パラメータ | 返却型 |
|-----------|--------------|--------|
| `create_contact_geometry_strategy()` | mode, line_contact, use_mortar, n_gauss, auto_gauss, exclude_same_layer | PtP/L2L/Mortar |

### テスト追加（+26テスト）

| テストファイル | 追加テスト数 | 内容 |
|--------------|------------|------|
| `test_contact_geometry.py` | +26 | update_geometry, build_constraint_jacobian, ファクトリ |
| `test_protocols.py` | 修正 | StubContactGeometry に新メソッド追加 |

### 更新ファイル

| ファイル | 変更内容 |
|---------|---------
| `xkep_cae/process/strategies/contact_geometry.py` | 3具象に update_geometry + build_constraint_jacobian 実装 + ファクトリ |
| `xkep_cae/process/strategies/protocols.py` | ContactGeometryStrategy に2メソッド追加 |
| `xkep_cae/process/strategies/__init__.py` | create_contact_geometry_strategy エクスポート追加 |
| `xkep_cae/process/strategies/test_contact_geometry.py` | 26件に拡張 |
| `xkep_cae/process/strategies/test_protocols.py` | StubContactGeometry 更新 |

## ファイル構造（Phase 4 完了後）

```
xkep_cae/process/strategies/
├── __init__.py              ← 13具象 + 5 Protocol + 5 ファクトリ エクスポート
├── protocols.py             ← 5 Strategy Protocol（ContactGeometry 拡張済）
├── compatibility.py         ← 互換性マトリクス
├── test_protocols.py        ← Protocol テスト
├── penalty.py               ← Penalty 4具象 + create_penalty_strategy()
├── test_penalty.py          ← 30件
├── time_integration.py      ← TimeInt 2具象 + create_time_integration_strategy()
├── test_time_integration.py ← 26件
├── friction.py              ← Friction 3具象【実ロジック移植済】+ create_friction_strategy()
├── test_friction.py         ← 26件
├── contact_force.py         ← ContactForce 2具象【実ロジック移植済】+ create_contact_force_strategy()
├── test_contact_force.py    ← 28件
├── contact_geometry.py      ← ContactGeom 3具象【実ロジック移植済】+ create_contact_geometry_strategy()
└── test_contact_geometry.py ← 46件（+26）
```

## 設計判断

| 判断 | 選択 | 理由 |
|------|------|------|
| update_geometry のスコープ | Narrowphase のみ | Broadphase は ContactManager.detect_candidates() に残す |
| 被膜モデル対応 | config パラメータで制御 | coating_stiffness > 0 で自動切替 |
| Active-set 更新 | ヒステリシス付き（g_on/g_off） | ContactManager と同一ロジック |
| Mortar の制約ヤコビアン | PtP フォールバック | 完全 Mortar G はPhase 5で統合 |
| build_constraint_jacobian の配置 | Strategy メソッド | Mortar では加重版が必要なため Strategy に配置 |

## CI 状況

### fast-test: 全通過（170 strategy テスト + 195 process テスト）

## TODO（次セッション: Phase 5）

- [ ] Phase 5: solver_ncp.py の newton_raphson_contact_ncp() を Strategy 注入に書き換え
  - ファクトリ関数を使って Strategy を構築し、NR ループ内で Strategy.evaluate()/tangent() を呼ぶ
  - 段階的移行: まず Penalty → TimeIntegration → Friction → ContactForce → ContactGeometry の順
- [ ] Mortar の完全制約ヤコビアン（mortar.build_mortar_system() 統合）
- [ ] Friction Strategy のCoulomb/SmoothPenalty 共通ロジック抽出（DRY化）
- [ ] slow-test 3件の収束問題対応（xfail 追加 or 収束改善）
- [ ] 全テストのmm-ton-MPa移行（status-149 TODO継承）
- [ ] 被膜摩擦μ=0.25の収束達成（status-149 TODO継承）
- [ ] 19本→37本のスケールアップ（status-149 TODO継承）

## 懸念事項

- **Broadphase は Strategy 外**: detect() はスタブのまま。Broadphase は ContactManager に密結合しており、Phase 5 の solver_ncp.py 書き換え時に統合方針を決める。
- **L2L/Mortar の update_geometry**: 現状は PtP と同一ロジック。力・剛性計算の差異は ContactForce/Friction Strategy 側で吸収済み。

---
