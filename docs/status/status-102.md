# Status 102: NCP ソルバー収束安定化 + 曲げ揺動分析スクリプト

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-03
**ブランチ**: `claude/fix-solver-convergence-aBSBb`
**テスト数**: 1921（fast: 1542 / slow: 374 + 5）

## 概要

NCP ソルバーに4つの収束安定化パラメータを実装し、
曲げ揺動の分析用実行スクリプト（VTK/GIF/接触グラフ出力付き）を作成。

## 実施内容

### 1. solver_ncp.py: 収束安定化パラメータ

| パラメータ | 説明 |
|-----------|------|
| `adaptive_omega` | メリット関数ベースの適応的緩和係数 |
| `bisection_max_depth` | 不収束時の荷重ステップ二分法 |
| `active_set_update_interval` | NCP active set 更新頻度制限（chattering 抑制） |
| `du_norm_cap` | Newton ステップノルムの相対上限 |

### 2. 曲げ揺動分析スクリプト

`scripts/run_bending_oscillation.py`:
- 7〜91本の曲げ揺動を一括実行
- エクスポートフック: VTK / 変位GIF / 接触グラフGIF / サマリー
- `--strands 7,19` `--outdir results/bending` で制御

### 3. テスト

`tests/contact/test_ncp_convergence_19strand.py`:
- 7本 NCP 基本/omega/bisection: 全パス
- 19本 NCP: 実行可能（収束は S3 改善待ち）

## 次の課題

- k_pen 自動スケーリング（EA/L ベース）
- 19本 NCP 収束達成
