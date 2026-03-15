# status-179: 脱出ポット Phase 2 後半 — Strategy 全移行 + 契約違反ゼロ達成

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-15

## 作業者

Claude Code

## ブランチ

claude/check-status-todos-FnGAk

## 概要

脱出ポット計画 Phase 2 後半。status-178 の TODO を全て実行し、
契約違反を3件→0件に削減。

1. TimeIntegrationStrategy 移行（core/time_integration/）
2. ContactGeometryStrategy 移行（contact/geometry/）
3. ContactForceStrategy 移行（contact/contact_force/）
4. StrandBendingBatchProcess 移行（core/batch/）— C12 解消
5. 接触共通型の新パッケージ化（_types.py, _assembly_utils.py）— C14 解消

## 変更内容

### コミット 1: TimeIntegrationStrategy 移行

新規ファイル:
```
xkep_cae/core/time_integration/
├── __init__.py
├── strategy.py           ← QuasiStaticProcess + GeneralizedAlphaProcess
├── docs/time_integration.md
└── tests/
    ├── __init__.py
    └── test_strategy.py  ← 34テスト
```

| クラス | 概要 |
|--------|------|
| `QuasiStaticProcess` | 準静的（荷重制御）— identity 操作 |
| `GeneralizedAlphaProcess` | Generalized-α 動的解析（Chung & Hulbert 1993） |

### コミット 2: ContactGeometryStrategy + ContactForceStrategy 移行

新規ファイル:
```
xkep_cae/contact/geometry/
├── __init__.py
├── strategy.py              ← PointToPoint + LineToLineGauss + MortarSegment
├── _compute.py              ← バッチ計算純粋関数
├── docs/contact_geometry.md
└── tests/
    ├── __init__.py
    └── test_strategy.py     ← 34テスト

xkep_cae/contact/contact_force/
├── __init__.py
├── strategy.py              ← NCPContactForce + SmoothPenaltyContactForce
├── docs/contact_force.md
└── tests/
    ├── __init__.py
    └── test_strategy.py     ← 24テスト

xkep_cae/contact/_types.py          ← ContactStatus enum
xkep_cae/contact/_assembly_utils.py ← _contact_dofs
```

| クラス | 概要 |
|--------|------|
| `PointToPointProcess` | 最近接点ペア（PtP） |
| `LineToLineGaussProcess` | Line-to-Line Gauss 積分 |
| `MortarSegmentProcess` | Mortar 法セグメント |
| `NCPContactForceProcess` | Alart-Curnier NCP + 鞍点系 |
| `SmoothPenaltyContactForceProcess` | softplus + Uzawa 外部ループ |

### コミット 3: BatchProcess 移行（C12 解消）

新規ファイル:
```
xkep_cae/core/batch/
├── __init__.py
├── strand_bending.py        ← StrandBendingBatchProcess
├── docs/strand_bending.md
└── tests/
    ├── __init__.py
    └── test_strand_bending.py ← 8テスト
```

Phase 2 時点では Strategy uses 宣言のみ。concrete プロセス（Mesh/Export/Verify）は Phase 3 で追加予定。

## テスト結果

- ruff check: 0 error
- ruff format: 0 issue
- time_integration テスト: 34 passed
- geometry テスト: 34 passed
- contact_force テスト: 24 passed
- batch テスト: 8 passed
- 既存テスト: 86 passed
- **合計: 186 passed**

## 契約違反

**0件** — 3件（C6×2 + C12×1）から全解消

| 契約 | 状況 |
|------|------|
| C6: TimeIntegrationStrategy | **解消**（本status） |
| C6: ContactGeometryStrategy | **解消**（本status） |
| C12: BatchProcess | **解消**（本status） |
| C14: deprecated imports | **解消**（_types.py + _assembly_utils.py + _compute.py） |

## 新パッケージ構成

```
xkep_cae/
├── core/
│   ├── time_integration/    ← 本 status で新規
│   │   ├── strategy.py
│   │   ├── docs/
│   │   └── tests/
│   ├── batch/               ← 本 status で新規
│   │   ├── strand_bending.py
│   │   ├── docs/
│   │   └── tests/
│   └── strategies/protocols.py
├── contact/
│   ├── _types.py            ← 本 status で新規（ContactStatus）
│   ├── _assembly_utils.py   ← 本 status で新規（_contact_dofs）
│   ├── penalty/             ← status-175
│   ├── friction/            ← status-178
│   ├── geometry/            ← 本 status で新規
│   │   ├── strategy.py
│   │   ├── _compute.py
│   │   ├── docs/
│   │   └── tests/
│   └── contact_force/       ← 本 status で新規
│       ├── strategy.py
│       ├── docs/
│       └── tests/
└── process/                 ← 後方互換 re-export
```

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `deprecated/.../time_integration.py` | `core/time_integration/strategy.py` | status-179 |
| `deprecated/.../contact_geometry.py` | `contact/geometry/strategy.py` | status-179 |
| `deprecated/.../contact_force.py` | `contact/contact_force/strategy.py` | status-179 |
| `deprecated/.../batch/strand_bending.py` | `core/batch/strand_bending.py` | status-179 |
| `deprecated/contact/geometry.py` (batch関数) | `contact/geometry/_compute.py` | status-179 |
| `deprecated/contact/pair.ContactStatus` | `contact/_types.ContactStatus` | status-179 |
| `deprecated/contact/assembly._contact_dofs` | `contact/_assembly_utils._contact_dofs` | status-179 |

## TODO

- [ ] concrete プロセス移行（StrandMeshProcess, ContactSetupProcess, ExportProcess 等）
- [ ] StrandBendingBatchProcess のフル実装（concrete 依存解消後）
- [ ] assembly モジュール完全移行
- [ ] CoulombReturnMapping/SmoothPenaltyFriction の接触ペアあり evaluate 実装

## 確認事項

- StrandBendingBatchProcess は Phase 2 時点でスタブ実装。concrete プロセス移行後にフル実装予定。
- `contact/geometry/_compute.py` の関数は `_` prefix で private 化（C16 準拠）。
- `contact/_types.py` の ContactStatus は `xkep_cae_deprecated/contact/pair.py` からのコピー。
  deprecated 側はそのまま残す（既存コードの互換性維持）。

---
