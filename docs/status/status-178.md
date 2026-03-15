# status-178: モジュール再編 + FrictionStrategy 移行

[← README](../../README.md) | [← status-index](status-index.md)

## 日付

2026-03-15

## 作業者

Claude Code

## ブランチ

claude/refactor-module-migration-Fzj2W

## 概要

脱出ポット計画 Phase 2 の前半。モジュール構造を再編し、FrictionStrategy を完全書き直し。

1. `xkep_cae/process/` 基盤を `xkep_cae/core/` に移動
2. `xkep_cae/process/strategies/penalty/` を `xkep_cae/contact/penalty/` に移動
3. FrictionStrategy を `xkep_cae/contact/friction/` に完全書き直し

## 変更内容

### コミット 1: process→core, penalty→contact モジュール移行

#### process/ → core/ 移動（10ファイル）

| 旧パス | 新パス |
|--------|--------|
| `process/base.py` | `core/base.py` |
| `process/categories.py` | `core/categories.py` |
| `process/data.py` | `core/data.py` |
| `process/registry.py` | `core/registry.py` |
| `process/runner.py` | `core/runner.py` |
| `process/slots.py` | `core/slots.py` |
| `process/testing.py` | `core/testing.py` |
| `process/tree.py` | `core/tree.py` |
| `process/presets.py` | `core/presets.py` |
| `process/strategies/protocols.py` | `core/strategies/protocols.py` |

#### penalty → contact/penalty 移動

- `process/strategies/penalty/` → `contact/penalty/`（全ファイル）

#### 後方互換

- `xkep_cae/process/__init__.py` を `core` からの re-export 互換レイヤーに変更
- 既存の `from xkep_cae.process import ...` は引き続き動作

#### validate_process_contracts.py 更新

- モジュール走査を `core/` + `contact/` 配下に変更
- C15 ドキュメント検証を `inspect.getfile` ベースの汎用パス解決に変更
- C16 滅菌チェックを複数ルート走査に対応

### コミット 2: FrictionStrategy 完全書き直し

参照元（旧コード）:
- `xkep_cae_deprecated/process/strategies/friction.py`
- `xkep_cae_deprecated/contact/law_friction.py`

新規ファイル:
```
xkep_cae/contact/friction/
├── __init__.py           ← 公開 API
├── law_friction.py       ← 純粋関数 + Process ラッパー
├── strategy.py           ← 3 Process 具象実装 + ファクトリ
├── docs/
│   └── friction.md       ← 設計ドキュメント
└── tests/
    ├── __init__.py
    ├── test_law_friction.py  ← 物理検証テスト（23テスト）
    └── test_strategy.py      ← Protocol適合 + ファクトリ（29テスト）
```

| クラス | 概要 |
|--------|------|
| `ReturnMappingProcess` | Coulomb return mapping（純粋関数ラップ） |
| `FrictionTangentProcess` | 摩擦接線剛性 2×2（純粋関数ラップ） |
| `NoFrictionProcess` | 摩擦なし（デフォルト） |
| `CoulombReturnMappingProcess` | NCP + Coulomb return mapping |
| `SmoothPenaltyFrictionProcess` | Smooth penalty + Uzawa 摩擦 |

純粋関数（private）:
- `_return_mapping_core()`: 弾性予測→Coulomb判定→stick/slip
- `_tangent_2x2_core()`: stick/slip consistent tangent
- `_rotate_friction_history()`: フレーム変換
- `_compute_mu_effective()`: μランプ

## テスト結果

- ruff check: 0 error
- ruff format: 0 issue
- penalty テスト: 34 passed
- friction テスト: 52 passed
- **合計: 86 passed**

## 契約違反

3件（C6×2 + C12×1）— 移行前の5件から改善

| 契約 | 状況 |
|------|------|
| C6: FrictionStrategy | **解消** |
| C6: PenaltyStrategy | **解消**（status-175 で解消済み） |
| C6: ContactForceStrategy | 未解消（Phase 2 後半） |
| C6: TimeIntegrationStrategy | 未解消 |
| C6: ContactGeometryStrategy | 未解消 |
| C12: BatchProcess | 未解消 |

## 新パッケージ構成

```
xkep_cae/
├── __init__.py
├── core/                    ← process 基盤（移動元: process/）
│   ├── __init__.py
│   ├── base.py
│   ├── categories.py
│   ├── data.py
│   ├── registry.py
│   ├── runner.py
│   ├── slots.py
│   ├── testing.py
│   ├── tree.py
│   ├── presets.py
│   └── strategies/
│       ├── __init__.py
│       └── protocols.py
├── contact/                 ← 接触 Strategy 実装
│   ├── __init__.py
│   ├── penalty/             ← status-175 で移行済み
│   │   ├── __init__.py
│   │   ├── strategy.py
│   │   ├── law_normal.py
│   │   ├── docs/penalty.md
│   │   └── tests/
│   └── friction/            ← 本 status で新規
│       ├── __init__.py
│       ├── strategy.py
│       ├── law_friction.py
│       ├── docs/friction.md
│       └── tests/
├── process/                 ← 後方互換 re-export レイヤー
│   └── __init__.py
└── ... (未移行モジュール)
```

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `xkep_cae/process/` 基盤ファイル | `xkep_cae/core/` | status-178 |
| `process/strategies/penalty/` | `contact/penalty/` | status-178 |
| `xkep_cae_deprecated/contact/law_friction.py` | `contact/friction/law_friction.py` | status-178 |
| `xkep_cae_deprecated/process/strategies/friction.py` | `contact/friction/strategy.py` | status-178 |

## TODO

- [ ] ContactForceStrategy 移行（contact/contact_force/）
- [ ] TimeIntegrationStrategy 移行
- [ ] ContactGeometryStrategy 移行
- [ ] assembly モジュール移行（Coulomb/SmoothPenalty の evaluate 完成に必要）
- [ ] CoulombReturnMapping/SmoothPenaltyFriction の接触ペアあり evaluate 実装

## 確認事項

- `xkep_cae/process/__init__.py` の re-export 互換レイヤーは、旧パッケージからの
  移行期間中のみ維持。全移行完了後に削除予定。
- Coulomb/SmoothPenalty の evaluate() は空ペアリスト以外は未実装。
  assembly モジュール移行後に完成する。

---
