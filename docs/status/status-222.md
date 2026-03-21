# status-222: Uzawa有効化試行 + softplus非互換性の確認 + 収束診断基盤

[← README](../../README.md) | [← status-index](status-index.md) | [← status-221](status-221.md)

## 日付

2026-03-21

## 概要

softplus smooth penalty における Uzawa 拡大ラグランジアン有効化を試行。
接線剛性修正（λ→full k_pen）、energy_ref 保持、max_attempts 増加を順次実施したが、
**softplus と Uzawa の根本的非互換性**を確認。純粋ペナルティ（n_uzawa_max=1）に戻し、
max_nr_attempts=100 で force convergence 達成を確認。
腐敗した収束テストを削除し、軽量診断スクリプトに置換。

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/contact_force/strategy.py` | tangent() docstring 整理（λ修正は試行後撤回） |
| `xkep_cae/numerical_tests/three_point_bend_jig.py` | max_nr_attempts=50→100, n_uzawa_max コメント更新 |
| `xkep_cae/contact/solver/_newton_uzawa_dynamic.py` | energy_ref リセットのコメント（変更なし） |
| `tests/contact/test_three_point_bend_jig.py` | 収束テスト4クラス削除（21→17テスト） |
| `contracts/diagnose_three_point_bend.py` | 新規: 軽量収束診断スクリプト |

## 削除したテストクラス

- `TestThreePointBendJigConvergence` — 静的収束テスト
- `TestDynamicThreePointBendJigConvergence` — 動的収束テスト
- `TestDynamicContactJigConvergence` — 動的接触収束テスト
- `TestDynamicContactJigPhysics` — n_periods=5,10 の接触物理テスト

## 検証結果

### 診断スクリプト（n_uzawa_max=1, max_attempts=100）

```
converged: True
increments: 54, cutbacks: 3
elapsed: 57.25 s
wire midpoint deflection: 0.084793 mm
contact force norm: 0.096242 N
```

### 既存テスト

```
17 passed in 219.60s
```

## softplus + Uzawa 非互換性の分析

### NR sigmoid tail 問題

softplus 接線 weight = `k_pen * sigmoid(-δg)` において:
- **g=0 (浅接触)**: sigmoid(0) = 0.5 → 有効接触剛性が 50% に低下
- **結果**: NR 収束率 0.9687/iter（全 dt で一定）
- **原因**: sigmoid(0) = 0.5 は δ に依存しない性質

NR は最初の 5 反復で残差を 1e+00 → 5e-04 に低減（構造+動的接線が正確）。
以降の sigmoid tail で rate 0.97 の線形収束。
energy convergence (|du·R_u| / energy_ref < 1e-10) が att~11 でこの tail をバイパス。

### Uzawa 試行結果

| 修正 | 結果 |
|------|------|
| tangent weight = k_pen (λ>0) | rate 0.85→0.97 に悪化（接線が硬すぎ） |
| energy_ref 保持 | 変化なし（1e-10 閾値が厳しすぎ） |
| max_attempts=100 | Uzawa 内 NR は att~90 で force 収束 |
| Uzawa 外ループ | **||Δλ||/||λ|| ≈ 1.3e-03 で停滞**（softplus(g)>0 により gap≠0） |
| 計算コスト | 159秒/2incr (vs 純粋ペナルティ 57秒/54incr) |

### 根本原因

1. softplus(g, δ) > 0（常に正）→ gap 制約が厳密に満たされない
2. Uzawa update Δλ = k_pen * |g| が毎回ほぼ同一値 → 外ループ非収束
3. 内 NR に 90 iter/uzawa × 5 uzawa = 450 iter/incr → 実用不可能

### 結論

softplus smooth penalty は Uzawa 拡大ラグランジアンと根本的に非互換。
Uzawa を有効にするには max(0, -g) 型ペナルティ（非平滑）への切り替えが必要。
現時点では純粋ペナルティ + energy convergence が最善。

## TODO（次セッションへの引き継ぎ）

- [ ] **非平滑ペナルティ + Uzawa の実装検討**: max(0, -g) 型ペナルティでの拡大ラグランジアン
- [ ] **動的接触物理テスト**: n_periods=5,10 の準静的テスト（max_attempts=100 で再挑戦）
- [ ] **S3 凍結解除**: 変位制御7本撚線曲げ揺動
