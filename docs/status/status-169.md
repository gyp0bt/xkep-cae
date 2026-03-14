# status-169: CoatingStrategy 抽出 — ContactManager 被膜メソッド Strategy 移行

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-14
**作業者**: Claude Code
**ブランチ**: claude/refactor-contact-strategy-23bzk

## 概要

ContactManager（pair.py）の被膜接触モデル関連4メソッドを新しい CoatingStrategy Protocol に移行した。
これにより pair.py の責務が軽量化され、被膜モデルがプラガブルな Strategy として交換可能になった。

## 変更内容

### 新規ファイル

| ファイル | 内容 |
|---------|------|
| `xkep_cae/process/strategies/protocols.py` | CoatingStrategy Protocol 追加（6軸目） |
| `xkep_cae/process/strategies/coating.py` | KelvinVoigtCoatingProcess + NoCoatingProcess + ファクトリ |
| `xkep_cae/process/strategies/docs/coating.md` | CoatingStrategy ドキュメント |
| `xkep_cae/process/strategies/tests/test_coating.py` | 14テスト（Protocol準拠 + 単体テスト） |

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/pair.py` | 4メソッドを deprecated 化、内部で Strategy に委譲 |
| `xkep_cae/contact/solver_ncp.py` | manager.compute_coating_* → _coating_strategy 経由 + フォールバック |
| `xkep_cae/contact/solver_smooth_penalty.py` | 同上（strategies.coating 経由） |
| `xkep_cae/process/data.py` | SolverStrategies に `coating` フィールド追加、default_strategies() 更新 |

### 移行された4メソッド

| ContactManager メソッド | Strategy メソッド | 状態 |
|------------------------|------------------|------|
| `compute_coating_forces()` | `CoatingStrategy.forces()` | deprecated → 委譲 |
| `compute_coating_stiffness()` | `CoatingStrategy.stiffness()` | deprecated → 委譲 |
| `compute_coating_friction_forces()` | `CoatingStrategy.friction_forces()` | deprecated → 委譲 |
| `compute_coating_friction_stiffness()` | `CoatingStrategy.friction_stiffness()` | deprecated → 委譲 |

## テスト結果

- Process テスト: 350 passed（+14 新規 coating テスト）
- Contact テスト: 39 passed（solver_ncp + coated_wire_integration）
- Contract 違反: 0 件
- ruff: 0 error, 0 format issue

## 今後の TODO（PR-2〜PR-6）

- [ ] PR-2: StagedActivation + InitialPenetration 純関数化（pair.py -200行）
- [ ] PR-3: LinearSolverStrategy 抽出（solver_ncp.py if分岐 Strategy 化）
- [ ] PR-4: Process 層 thin wrapper 解消（StrategySlot 実活用）
- [ ] PR-5: テスト名正規化（_ncp 除去）+ NCP 直接呼出 → Process API 移行
- [ ] PR-6: tuning/executor.py NCP 版実装

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `ContactManager.compute_coating_*()` | `CoatingStrategy.*()` | status-169 |

---
