# status-174: solver_smooth_penalty.py 分解 → Process 実体化

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-15
**作業者**: Claude Code
**ブランチ**: claude/refactor-solver-penalty-KTM8G

## 概要

`solver_smooth_penalty.py`（691行モノリシック関数）を3つの独立モジュールに分解し、
`ContactFrictionProcess` を薄いラッパーから実体実装に昇格させた。

**設計動機**: 巨大バッチ分岐処理（if文ベタ書き平文実装）は改良改変で管理不能になる。
Strategy 5軸は分離済みだったが、アルゴリズムフレームワーク（NR反復、Uzawa、適応荷重増分）は
依然としてモノリシック関数に閉じ込められていた。

## 変更内容

### 新規ファイル 3つ

| ファイル | クラス | 責務 | 行数 |
|---------|--------|------|------|
| `process/strategies/solver_state.py` | `SolverState` | 全可変状態の集約 + チェックポイント管理 | ~100 |
| `process/strategies/newton_uzawa.py` | `NewtonUzawaLoop` + `NewtonUzawaConfig` + `StepResult` | 1荷重増分のNR+Uzawaイテレーション | ~230 |
| `process/strategies/adaptive_stepping.py` | `AdaptiveLoadController` + `AdaptiveSteppingConfig` | 適応荷重増分のステップキュー管理 | ~120 |

### 既存ファイル修正

| ファイル | 変更 |
|---------|------|
| `process/concrete/solve_contact_friction.py` | `solve_smooth_penalty_friction()` への委譲を廃止。SolverState + NewtonUzawaLoop + AdaptiveLoadController を直接組み立てるオーケストレーターに書き換え。version 0.2.0 → 0.3.0 |
| `contact/__init__.py` | `solve_smooth_penalty_friction` の import/export 除去 |

### 削除ファイル

| ファイル | 理由 |
|---------|------|
| `contact/solver_smooth_penalty.py` | 分解してprocess配下に再配置済み |

## アーキテクチャ変更

### Before (status-173)
```
ContactFrictionProcess.process()
  └→ solve_smooth_penalty_friction()  ← 691行モノリシック関数
      ├─ 初期化 (87-257行)
      ├─ 荷重ステップループ (261-676行)
      │   ├─ NR+Uzawa (315-523行)
      │   ├─ カットバック (528-575行)
      │   └─ dt制御 (599-636行)
      └─ 結果構築
```

### After (status-174)
```
ContactFrictionProcess.process()  ← 実体オーケストレーター
  ├─ SolverState              ← 可変状態集約
  ├─ AdaptiveLoadController   ← ステップキュー管理
  ├─ NewtonUzawaLoop.run()    ← 1増分のNR+Uzawa
  └─ Strategy 5軸（既存）     ← penalty, friction, time, contact_force, coating
```

## テスト結果

- ruff check: 0 error
- ruff format: 0 issue
- 契約違反: 0 件
- process テスト: 343 passed
- 接触テスト (test_solver_contact.py): 16 passed

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status |
|--------|--------|-----------|
| `contact/solver_smooth_penalty.py` | `process/strategies/{solver_state,newton_uzawa,adaptive_stepping}.py` + `ContactFrictionProcess` 実体化 | status-174 |

## 今後の TODO

- [ ] solver_ncp.py の呼び出しテスト群を Process API 経由に移行（現在は直接 import）
- [ ] S3 凍結解除判断 + BatchProcess パイプライン改善
- [ ] 変位制御7本撚線曲げ揺動のPhase2 xfail解消

## 設計上の懸念・メモ

- `solver_ncp.py`（NCP鞍点系ソルバー、2395行）は今回のスコープ外。22テストファイルが直接importしている。Process API への移行は別statusで実施。
- `NewtonUzawaLoop.run()` は u, lam_all を in-place 更新する設計。SolverState が所有権を持つため、呼び出し側との責任分界は明確。
- `AdaptiveLoadController` は ContactManager.config から設定を引き出す前提。将来的には ContactFrictionInputData に直接含めるべき。

---
