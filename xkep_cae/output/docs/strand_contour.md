# Strand3DContourProcess — 撚線 3D パイプコンターレンダリング

[← README](../../../README.md) | [← output/README](../__init__.py)

## 概要

撚線モデル（7 本・19 本・1000 本等）の `SolverResultData` を受け取り、
3D パイプ状で接触状態・接触力・軸応力・曲率・チャタリングを可視化する
`PostProcess` 実装。status-362（仮説 C (c) backtracking line search 実機
検証）の視覚化需要に応えて新設。

## 入出力

**入力**: `Strand3DContourConfig`

| フィールド | 意味 |
|-----------|------|
| `solver_result` | `SolverResultData`（`ContactFrictionProcess` 等の解析結果）|
| `mesh` | `MeshData`（撚線メッシュ）|
| `output_dir` | PNG 出力先ディレクトリ |
| `prefix` | ファイル名接頭辞（例: `"19strand_bt"`）|
| `young` | Young 率 [MPa]（軸応力 σ=Eε 用、デフォルト 130e3）|
| `requested_fields` | レンダリング対象フィールドのタプル（デフォルト全 6 種）|
| `elev` / `azim` | 3D 視点（仰角 / 方位角）|

**出力**: `Strand3DContourResult`

| フィールド | 意味 |
|-----------|------|
| `image_paths` | 生成された PNG ファイルのパス |
| `field_stats` | 各フィールドの min/max/mean 統計 |
| `n_contact_elements` | 接触状態の要素数 |
| `n_chattering_elements` | チャタリング検出された要素数 |

## 対応フィールド

| フィールド名 | 色付け | 計算元 |
|--------------|-------|--------|
| `contact` | binary（赤=接触, 青=非接触）| `manager.pairs` の `p_n > 0` |
| `contact_force` | hot colormap | 各要素に作用する p_n 合計 |
| `stress` | coolwarm | 軸ひずみ × Young 率 |
| `curvature` | viridis | 隣接要素の接線ベクトル差 / 要素長 |
| `chatter_binary` | binary | チャタリング検出有無 |
| `chatter_score` | magma | increment 間状態遷移 / n_increments |

チャタリングは `solver_result.contact_pair_history` から計算するため、
解析時に `track_contact_pairs=True` を設定する必要がある。

## 出力形式

各フィールドごとに 1 枚の PNG を生成。各 PNG は:
- **左サブプロット**: XZ 平面投影（side view）
- **右サブプロット**: 3D oblique 視点

ファイル名: `{prefix}_{field_name}_{suffix}.png`
（`suffix` は未指定時 `frac{値:.3f}` で自動生成）

## 使用例

```python
from xkep_cae.output import Strand3DContourConfig, Strand3DContourProcess

# 解析実行後
result = StrandBendingOscillationProcess().process(cfg)

# レンダリング
Strand3DContourProcess().process(
    Strand3DContourConfig(
        solver_result=result.solver_result,
        mesh=result.mesh,
        output_dir="docs/verification/19strand_3d",
        prefix="19strand_bt",
    )
)
```

## 設計意図

- **PostProcess 分離**: solver 実行と可視化を分離。solver は無変更、
  可視化は opt-in で後から呼び出す。
- **runner/dialog 統合**: `BenchmarkRunnerProcess.post_processes` フィールド
  経由で自動実行可能（status-363 以降で整備予定）。
- **軽量依存**: matplotlib のみ、外部 3D ライブラリ不要。
