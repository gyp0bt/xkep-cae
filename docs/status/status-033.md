# status-033: Phase C1 — Broadphase (AABB格子) + 幾何更新 + Active-setヒステリシス

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-18

## 概要

Phase C1 の実装。梁–梁接触モジュールに broadphase（AABB格子による候補ペア探索）、
ContactManager の幾何更新（narrowphase: 最近接点・ギャップ・接触フレーム計算）、
Active-set ヒステリシス（g_on/g_off 閾値による活性/非活性化制御）を追加。
テスト数 670 → 701（+31テスト）。

## 実施内容

### 1. Broadphase — AABB格子 (`xkep_cae/contact/broadphase.py`)

新規モジュール。多数セグメントから接触候補ペアを高速に抽出する。

- `compute_segment_aabb(x0, x1, radius, margin)`: セグメントのAABB計算（半径+マージンで膨張）
- `broadphase_aabb(segments, radii, margin, cell_size)`: 空間ハッシュ格子ベースの候補ペア探索
  - 各セグメントのAABBを均一格子にビニング
  - 同一/隣接セル内のペアでAABB重複を精密判定
  - セルサイズ自動推定（AABBサイズ平均の1.5倍）
  - 要素ごとの異なる半径に対応
  - 候補ペアは (i, j) 形式 (i < j) で正規化、重複なし

**テスト（16件）**:
- `TestComputeSegmentAABB` (4件): 基本AABB、半径膨張、マージン膨張、端点順序不依存
- `TestBroadphaseAABB` (12件): 交差検出、離間除外、半径による近接化、1/0セグメント、グリッド配置（隣接のみ検出）、平行重複、3Dねじれ、要素別半径、カスタムセルサイズ、自己ペア除外、重複除外

### 2. ContactManager 幾何更新 (`xkep_cae/contact/pair.py`)

ContactManager に3つの新メソッドを追加。

#### `detect_candidates(node_coords, connectivity, radii, *, margin, cell_size)`
- broadphase_aabb を呼び出して候補ペアを検出
- 新規候補は ContactPair として自動追加
- 候補から外れた既存ペアは INACTIVE に設定
- 節点座標 → セグメント変換を内部で処理

#### `update_geometry(node_coords)`
- 全ペアに対して narrowphase を実行
  - `closest_point_segments()` で最近接点パラメータ (s, t) 更新
  - `compute_gap()` でギャップ計算
  - `build_contact_frame()` で接触フレーム更新（法線履歴連続性あり）
  - `_update_active_set()` でActive-set更新

#### `_update_active_set(pair)`
- ヒステリシスバンドによるチャタリング防止
  - 非活性 → 活性: gap <= g_on
  - 活性 → 非活性: gap >= g_off  (g_off > g_on)
  - ヒステリシスバンド内: 状態維持

**テスト（15件）**:
- `TestDetectCandidates` (5件): 交差検出、離間除外、既存ペア非活性化、新規追加、要素別半径
- `TestUpdateGeometry` (5件): ギャップ・パラメータ更新、接触フレーム正規直交性、半径考慮、貫通ギャップ、フレーム連続性
- `TestActiveSetHysteresis` (5件): 接触活性化、ヒステリシスバンド内活性維持、g_off超非活性化、バンド内非活性維持、SLIDING非活性化

### 3. __init__.py 更新

- `broadphase_aabb`, `compute_segment_aabb` を公開API に追加
- `ContactConfig`, `ContactManager`, `ContactStatus` を公開APIに追加

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/broadphase.py` | **新規**: AABB格子broadphase |
| `xkep_cae/contact/pair.py` | `detect_candidates()`, `update_geometry()`, `_update_active_set()` 追加 |
| `xkep_cae/contact/__init__.py` | エクスポート拡張 |
| `tests/contact/test_broadphase.py` | **新規**: 16テスト |
| `tests/contact/test_pair.py` | 15テスト追加（Phase C1） |

## テスト数

670 → 701（+31テスト）

## 確認事項・懸念

1. **broadphaseの計算量**: 均一格子ハッシュは O(n) だが、セグメントが非一様分布の場合にセル数が増大する可能性がある。大規模問題では階層的AABB（BVHツリー）への移行を検討
2. **Active-setヒステリシスの閾値**: デフォルト g_on=0.0, g_off=1e-6 は非常に狭い。実問題ではスケール依存のため、ユーザーが適切に設定する必要がある
3. **法線フレーム連続性**: `build_contact_frame()` の `prev_tangent1` ベースの連続性保持は、法線が大きく変化する場合に限界がある。フレーム輸送（parallel transport）は Phase C5 で強化予定

## TODO

- [ ] Phase C2: 法線AL + Active-setヒステリシス + 主項接線
- [ ] Phase C3: 摩擦return mapping + μランプ
- [ ] Phase C4: merit line search + 探索/求解分離の運用強化

---
