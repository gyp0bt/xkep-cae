# Status 095: テストコード機能棚卸し + 集約 + CI失敗修正

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-01
**ブランチ**: `claude/add-wire-benchmark-PgGdm`
**テスト数**: 1903（fast: 1560 / slow: 343）

## 概要

`tests/contact/test_twisted_wire_contact.py` の機能棚卸しと集約を実施。
テスト品質の改善とメンテナンス性向上。

## 実施内容

### 1. テスト失敗修正

**TestBlockSolverLargeMesh::test_seven_strand_16_elems**:
- 原因: `gap=0.0005` + `g_on=0.0` のヒステリシス死帯域
- 16要素/素線ではヘリックス近似が精密で実ギャップ≈0.45mm → `g_on=0.0` 未満にならず接触が活性化しない
- 4要素版は粗い近似で幾何的に接触が発生（偶発的に通っていた）
- 修正: `gap=0.0` に統一（同クラスの曲げテストと同じ）

### 2. テストコード棚卸し・集約

**棚卸し結果**: 19クラス, ~77テスト → 16クラス, 71テスト

| 変更 | 詳細 |
|---|---|
| xfail 2クラス削除 | TestSevenStrandMultiContact (3), TestSevenStrandConvergenceImprovement (3) |
| クラス統合 | TestALRelaxation + TestIterativeSolver → TestALRelaxationAndIterativeSolver |
| パラメータ集約 | `_STABLE_7STRAND_PARAMS` モジュール定数を新設 |
| 参照先統一 | ImprovedSolver, BlockSolver, Cyclic, BlockDecompositionBasic が共有 |
| 未使用コード削除 | `_fix_both_ends_all()` ヘルパー関数 |

### 3. `_STABLE_7STRAND_PARAMS` の定義

7本撚り接触問題の安定収束に必要な9パラメータを1箇所で管理:

```python
_STABLE_7STRAND_PARAMS = {
    "auto_kpen": True,
    "staged_activation": True,
    "n_outer_max": 1,
    "k_pen_scaling": "sqrt",
    "al_relaxation": 0.01,
    "penalty_growth_factor": 1.0,
    "preserve_inactive_lambda": True,
    "g_off": 0.001,
    "no_deactivation_within_step": True,
}
```

**経緯ドキュメント**: 削除した xfail テストの知見（何が失敗し何が成功したか）を
この定数のコメントに残し、パラメータ選択の根拠を保存。

## テスト結果

| テストスイート | 結果 |
|---|---|
| 変更クラス全24テスト | 24 passed（347秒） |
| テスト総数 | 1903（fast: 1560 / slow: 343） |

### 削除した xfail テスト一覧

| クラス | テスト | xfail 理由 |
|---|---|---|
| TestSevenStrandMultiContact | test_timo3d_{tension,torsion,bending}_converges | 36+ペア同時収束困難 |
| TestSevenStrandConvergenceImprovement | test_{tension,torsion,bending}_modified_newton | Modified Newton + damping でも不十分 |

これらは TestSevenStrandImprovedSolver が同じ問題を `_STABLE_7STRAND_PARAMS` で解決済み。

## TODO

- [ ] TestBlockSolverLargeMesh の gap=0 → 物理的に正しい gap での接触閾値チューニング
- [ ] CR梁アセンブリのCOO/CSRベクトル化（status-094 引き継ぎ）
- [ ] 1000本撚線での速度ベンチマーク実行（status-094 引き継ぎ）
