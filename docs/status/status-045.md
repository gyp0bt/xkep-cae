# status-045: Phase C4 — merit line search + 探索/求解分離の運用強化

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-21

## 概要

status-044 の TODO を消化:
- **Phase C4: merit line search + 探索/求解分離の運用強化** — 本status

テスト数 **958**（+26: line search 単体テスト 21件 + 統合テスト 5件）。

## 実施内容

### 1. line_search.py（新規）

Merit function + backtracking line search モジュールを新規作成。

| 関数 | 説明 |
|------|------|
| `merit_function()` | Phi = \|\|R\|\| + alpha * sum(max(0,-g)^2) + beta * sum(max(0, D_inc)) の評価 |
| `backtracking_line_search()` | Armijo 条件による backtracking: eta を段階的に縮小し Phi が減少する step length を採用 |

#### 設計判断

- **Merit function の構成**: 残差ノルム + 貫通ペナルティ + 散逸ペナルティの3項。設計仕様 §8 に準拠
- **Armijo 条件**: Phi(u + eta*du) <= Phi(u) * (1 - c * eta) で判定。c = 1e-4（デフォルト）
- **最良 eta 返却**: 全ステップで Armijo 未達の場合、最良の merit を示した eta を返す（Newton 発散よりまし）
- **backtracking_line_search の interface**: `eval_merit(u_trial) -> float` コールバックを受け取る設計。solver 側で closure を構成して渡す

### 2. ContactConfig 拡張

| フィールド | デフォルト | 説明 |
|-----------|-----------|------|
| `use_line_search` | `False` | merit line search の有効化（後方互換: デフォルト OFF） |
| `line_search_max_steps` | `5` | backtracking の最大縮小回数 |
| `merit_alpha` | `1.0` | merit function の貫通ペナルティ重み |
| `merit_beta` | `1.0` | merit function の散逸ペナルティ重み |

### 3. solver_hooks.py 拡張

NR ソルバーに line search と merit-based Outer 終了判定を統合。

| 変更 | 説明 |
|------|------|
| Line search 統合 | Inner NR loop の `u += du` を `u = u + eta * du` に拡張。`use_line_search=True` かつ active contact がある場合に発動 |
| Merit eval closure | 摩擦力を固定して gap + 残差を再評価する closure を構成し `backtracking_line_search` に渡す |
| Merit-based Outer 終了 | Outer loop で merit が改善停滞（ratio > 0.99）した場合に早期終了。(s,t) 更新が効果なしと判断 |
| ContactSolveResult 拡張 | `total_line_search_steps` フィールド追加（全ステップの累積 line search 縮小回数） |
| 進捗表示 | line search 発動時の eta と step 数、merit-based 早期終了時のメッセージを表示 |

#### Inner loop の line search フロー

```
for it in range(max_iter):
    # ... gap 更新、摩擦 return mapping、残差・剛性計算 ...
    du = spsolve(K, R)

    if use_line_search and n_active > 0:
        phi_cur = merit_function(residual, manager)
        eta, n_ls = backtracking_line_search(u, du, phi_cur, _eval_merit)
        u = u + eta * du
    else:
        u += du
```

#### Merit-based Outer 終了判定

```
for outer in range(n_outer_max):
    # ... Inner NR loop ...
    # ... (s,t) 更新 ...

    merit_cur = merit_function(residual, manager)
    if outer > 0 and merit_cur / merit_prev > 0.99:
        # merit 改善停滞 → 早期終了
        break
    merit_prev = merit_cur
```

### 4. テスト結果

#### line search 単体テスト（21件）

| テストクラス | テスト数 | 内容 |
|-------------|---------|------|
| TestMeritFunction | 12 | 零/残差のみ/貫通/正gap/散逸/負散逸/合計/alpha重み/beta重み/INACTIVE除外/複数貫通/非負 |
| TestBacktrackingLineSearch | 9 | full step/backtracking/零merit/失敗時最良/shrink/Armijo/Armijo marginal/max steps/最良選択 |

#### 統合テスト（5件）

| テスト | 結果 |
|--------|------|
| line search 付き法線接触収束 | PASS |
| line search 付き貫通防止 | PASS |
| 摩擦 + line search 組み合わせ | PASS |
| total_line_search_steps フィールド存在 | PASS |
| use_line_search=False 後方互換 | PASS |

### 5. 既存テストへの影響

- **既存テスト 932 件**: 全パス — 回帰なし
- **新規テスト 26 件**: line search 単体 21 + 統合 5
- **合計 958 テスト**（948 passed + 10 skipped ≈ テスト構成変動による差異）

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/line_search.py` | **新規** — merit function + backtracking line search |
| `xkep_cae/contact/pair.py` | ContactConfig に line search 設定追加 |
| `xkep_cae/contact/solver_hooks.py` | line search 統合 + merit-based Outer 終了判定 |
| `xkep_cae/contact/__init__.py` | line search 関連 export 追加 |
| `tests/contact/test_line_search.py` | **新規** — line search 単体テスト 21件 |
| `tests/contact/test_solver_hooks.py` | 統合テスト 5件追加 |

## テスト数

958（+26）

## 確認事項・懸念

1. **Line search 発動頻度**: 現在のテストケース（簡易ばねモデル）では Newton step が十分良好なため line search の縮小は稀。実問題（多数接触ペア、stick/slip 遷移）での効果は Phase 4.7（撚線モデル）で検証予定
2. **Merit 重み (alpha, beta)**: デフォルト値（1.0, 1.0）は残差と貫通のスケールが同程度であることを前提。実問題では調整が必要な場合あり
3. **摩擦力の line search 中の扱い**: v0.1 では line search trial 中の摩擦力は固定（NR iteration での評価値を使用）。摩擦 re-evaluation はコスト高のため v0.2 で検討
4. **Merit-based Outer 終了**: ratio 閾値 0.99 はヒューリスティック。厳密な収束保証ではなく実用的な計算コスト削減が目的

## TODO

- [ ] Phase C5: 幾何微分込み一貫接線 + semi-smooth Newton / PDAS 検討
- [ ] slip consistent tangent の実装（v0.2）
- [ ] 摩擦ありの Abaqus バリデーション
- [ ] 撚線モデル（Phase 4.7）での line search 効果検証

---
