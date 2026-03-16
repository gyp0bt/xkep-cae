# AdaptiveSteppingProcess — 適応荷重増分制御

[← contact_friction](contact_friction.md)

## 概要

荷重分率 (load_frac: 0→1) の刻み幅を Newton 反復数と接触状態変化に基づいて
適応的に制御する SolverProcess。1判定 = 1 `process()` 呼び出し。

## 入出力

- **入力**: `AdaptiveStepInput`（frozen dataclass）
  - `action`: QUERY / SUCCESS / FAILURE
  - ステップ結果（load_frac, n_iters, n_active 等）
- **出力**: `AdaptiveStepOutput`（frozen dataclass）
  - `next_load_frac`: 次の荷重分率
  - `has_more_steps`: 残りステップの有無
  - `can_retry`: 失敗時のリトライ可否

## アルゴリズム

- **QUERY**: skip 対象を内部消化し、次の有効な load_frac を返す
- **SUCCESS**: 反復数に基づきステップ幅を grow/shrink/maintain
  - n_iters ≤ threshold → grow（最大 1.5x、減衰あり）
  - n_iters ≥ threshold → shrink（0.5x）
  - 接触状態急変 → 強制 shrink
- **FAILURE**: カットバック（中間点を挿入してリトライ）
