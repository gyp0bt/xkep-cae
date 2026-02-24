# status-052: 撚線メッシュファクトリ + 多点接触テスト + 接触グラフ表現

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-24
**作業者**: Claude Code
**テスト数**: 1147（+72）

## 概要

Phase 4.7（撚線モデル）Level 0 の基盤構築として3機能を実装。

1. **撚線メッシュファクトリ** (`xkep_cae/mesh/twisted_wire.py`): 理想的なヘリカル配置に基づく撚線梁メッシュ生成。3/7/19/37/61/91本対応。
2. **多点接触撚線テスト** (`tests/contact/test_twisted_wire_contact.py`): 撚線メッシュを用いた接触ソルバー統合テスト。3本撚り5荷重タイプ成功、7本・摩擦は収束課題をxfailで記録。
3. **接触グラフ表現** (`xkep_cae/contact/graph.py`): 多点接触の状態を無向グラフとして表現するポスト処理モジュール。トポロジー変遷追跡、連結成分分析、隣接行列出力。

## 変更内容

### 1. 撚線メッシュファクトリ: `xkep_cae/mesh/twisted_wire.py`

撚線の理想的なヘリカル幾何に基づく梁メッシュ生成モジュール。

| データ構造/関数 | 説明 |
|----------------|------|
| `StrandInfo` | 素線情報（layer, angle_offset, lay_radius, lay_direction） |
| `TwistedWireMesh` | メッシュ（node_coords, connectivity, strand_node/elem_ranges） |
| `make_strand_layout()` | 素線配置パターン生成（3本三つ撚り、7+本同心円層） |
| `make_twisted_wire_mesh()` | メッシュ生成ファクトリ（ヘリカル/直線点列 + 接続行列） |
| `compute_helix_angle()` | ヘリックス角 α = arctan(2πR/pitch) |
| `compute_strand_length_per_pitch()` | 弧長 L = √((2πR)² + pitch²) |

**撚線パターン**:
- 3本: 中心なし、120°配置、r_lay = d/√3
- 7本: 中心1本 + 第1層6本
- 19本: 1 + 6 + 12
- 37本: 1 + 6 + 12 + 18
- 一般: 1 + Σ6k (k=1,2,3,...)、交互撚り方向

### 2. 撚線メッシュテスト: `tests/mesh/test_twisted_wire.py`（32テスト）

| クラス | テスト数 | 検証内容 |
|-------|---------|---------|
| `TestMakeStrandLayout` | 11 | 配置パターン（3/7/19/37本）、層半径単調増加、交互撚り方向、ギャップ効果、ID一意性 |
| `TestMakeTwistedWireMesh` | 14 | 基本生成（3/7/19本）、中心素線直線性、外層ヘリカル性、接続妥当性、半径均一性、n_pitches指定、節点/要素パーティション |
| `TestHelixGeometryUtils` | 4 | ヘリックス角（0/大/既知値）、弧長公式、無効ピッチエラー |
| `TestLargeScaleMesh` | 3 | 61/91本生成、要素品質（最大/最小比<3） |

### 3. 多点接触撚線テスト: `tests/contact/test_twisted_wire_contact.py`（16テスト）

撚線メッシュを接触ソルバー（newton_raphson_with_contact）に入力して解析。

| クラス | テスト数 | 状態 | 検証内容 |
|-------|---------|------|---------|
| `TestThreeStrandBasicContact` | 5 | ✅ pass | 3本撚り: tension/bending/torsion/lateral/lateral_large |
| `TestSevenStrandMultiContact` | 3 | ⚠️ xfail | 7本: 24+接触ペア同時収束の課題 |
| `TestTwistedWireFriction` | 3 | ⚠️ xfail | 3本摩擦: ヘリカル接触幾何での摩擦安定性 |
| `TestTimo3DVsCR` | 1 | ✅ pass | Timo3D vs CR梁の接触応答比較 |
| `TestContactDataCollection` | 4 | ✅ pass | 接触データ収集（ペア数/法線力/ギャップ/状態分布） |

**7本撚りの収束課題（xfail記録）**:
- 24+接触ペアが同時にアクティブ化 → k_pen/構造剛性比の調整困難
- 改善方針: k_pen自動推定、段階的接触アクティベーション（層別）、反復戦略改善

### 4. 接触グラフ表現: `xkep_cae/contact/graph.py`

多点接触の状態を無向グラフとして表現するポスト処理モジュール。

| データ構造/関数 | 説明 |
|----------------|------|
| `ContactEdge` | エッジ属性（elem_a/b, gap, p_n, status, stick, dissipation, s, t） |
| `ContactGraph` | スナップショット（nodes, edges, n_active, n_sliding, degree_map, adjacency_list, connected_components, to_adjacency_matrix, to_dict） |
| `ContactGraphHistory` | 時系列（edge_count_series, node_count_series, total_force_series, dissipation_series, topology_change_steps, to_dict_list） |
| `snapshot_contact_graph()` | ContactManager → ContactGraph 変換 |

### 5. 接触グラフテスト: `tests/contact/test_contact_graph.py`（24テスト）

| クラス | テスト数 | 検証内容 |
|-------|---------|---------|
| `TestSnapshotContactGraph` | 6 | スナップショット生成、ノード抽出、非アクティブ除外、空マネージャ、エッジ属性、SLIDING状態 |
| `TestContactGraphMethods` | 9 | ACTIVE/SLIDING数、法線力/散逸合計、次数マップ、隣接リスト、連結成分（独立/共有）、隣接行列、辞書出力 |
| `TestContactGraphHistory` | 9 | ステップ数、エッジ/ノード/力/散逸/荷重係数時系列、トポロジー変化検出、辞書リスト、空時系列 |

## ファイル変更

### 新規作成
- `xkep_cae/mesh/__init__.py` — メッシュパッケージ初期化
- `xkep_cae/mesh/twisted_wire.py` — 撚線メッシュファクトリ
- `tests/mesh/__init__.py` — テストパッケージ初期化
- `tests/mesh/test_twisted_wire.py` — 撚線メッシュテスト（32テスト）
- `tests/contact/test_twisted_wire_contact.py` — 多点接触撚線テスト（16テスト: 10 pass + 6 xfail）
- `xkep_cae/contact/graph.py` — 接触グラフ表現モジュール
- `tests/contact/test_contact_graph.py` — 接触グラフテスト（24テスト）

## テスト結果

```
tests/mesh/test_twisted_wire.py             32 passed
tests/contact/test_twisted_wire_contact.py  10 passed, 6 xfail
tests/contact/test_contact_graph.py         24 passed
全テスト:                                    1147 collected
lint/format:                                ruff check + ruff format パス
```

## 確認事項

- 既存テスト影響なし（新規ファイルのみ追加）
- lint/format 全クリア
- 7本撚り・摩擦テストの収束課題は xfail で明示的に記録

## TODO

### Phase 4.7 Level 0 に向けた課題

- [ ] k_pen 自動推定（EI/L³ ベース、接触ペア数考慮）
- [ ] 段階的接触アクティベーション（層別に接触を導入）
- [ ] 7本撚りの収束改善（Outer/Inner 反復戦略の調整）
- [ ] ヘリカル幾何での摩擦 return mapping 安定化
- [ ] 接触グラフの可視化出力（matplotlib + GIF）
- [ ] 撚線接触テストでの接触グラフ時系列データ収集
- [ ] 弧長法＋接触統合テスト → Phase 4.7 で座屈必要時に着手

---
