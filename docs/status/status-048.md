# status-048: 梁梁接触 貫入テスト

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-23
**作業者**: Claude Code
**テスト数**: 1034（+11）

## 概要

2本の梁（針）を交差させ、接触時の貫入量が適切に制限されることを検証する
バリデーションテストを追加。Phase C（梁–梁接触モジュール）の品質保証テスト。

## テスト設計

### セットアップ

```
梁A（針1）: x軸方向 (0,0,0)→(1,0,0)
  - Node 0: 全固定
  - Node 1: x方向のみ自由（張力）

梁B（針2）: y軸方向 (0.5,-0.5,h)→(0.5,0.5,h)
  - Node 2: 全固定
  - Node 3: y,z方向自由（y張力 + z押し下げ）

初期ギャップ: h - 2r = 0.082 - 0.080 = 0.002
断面半径: r = 0.04
```

### テスト項目（11テスト、5クラス）

| クラス | テスト | 検証内容 |
|-------|-------|---------|
| `TestBeamContactDetection` | `test_contact_detected_with_push_down` | z方向力で接触検出 |
| | `test_no_contact_without_push_down` | 力なしでは非接触 |
| `TestPenetrationBound` | `test_penetration_within_tolerance` | 貫入量 < 断面半径の10% |
| | `test_penetration_with_large_force` | 大荷重でも貫入制限 |
| | `test_higher_penalty_reduces_penetration` | k_pen↑ → 貫入↓ |
| `TestNormalForce` | `test_normal_force_positive` | p_n ≥ 0（引張なし） |
| | `test_normal_force_increases_with_push` | F_z↑ → p_n↑ |
| `TestFrictionPenetrationEffect` | `test_penetration_bounded_with_friction` | 摩擦ありでも貫入制限 |
| | `test_friction_does_not_worsen_penetration` | 摩擦有無でΔgap < 1e-3 |
| `TestDisplacementHistory` | `test_z_displacement_progresses_downward` | 荷重ステップ進行確認 |
| | `test_x_tension_positive` | 張力方向の変位が正 |

### 物理的知見

- AL乗数更新後の荷重ステップで、接触力が増大し梁Bが微小に押し戻される
  現象を確認。これはAugmented Lagrangian法の正常な挙動。
- ペナルティ剛性 k_pen = 1e5 で貫入量は断面半径の10%以下に制限される。
- k_pen = 1e4 → 1e5 → 1e6 と増加させると貫入量は単調に減少。

## ファイル変更

- `tests/contact/test_beam_contact_penetration.py` — 新規作成（11テスト）

## 確認事項

- 既存の接触テスト217件（contact/ ディレクトリ）全パス確認済み
- lint/format パス（`ruff check` + `ruff format`）

## TODO

- より多数の要素（マルチセグメント梁）での貫入テスト
- 接触点移動（スライディング）時の貫入量追跡テスト
- 接触付き弧長法との統合テスト（Phase 4.7前提）

---
