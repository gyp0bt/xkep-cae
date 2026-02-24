# status-053: k_pen自動推定 + 段階的アクティベーション + 摩擦安定化 + 接触グラフ可視化・時系列収集

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-24
**作業者**: Claude Code
**テスト数**: 1206（+59）

## 概要

status-052 の TODO を実行。Phase 4.7 Level 0 の6項目を完了。

1. **k_pen 自動推定**（EI/L³ ベース）: `auto_beam_penalty_stiffness()` で梁曲げ剛性ベースのペナルティ推定。接触ペア数による線形スケーリング。
2. **段階的接触アクティベーション**: 層別（layer）に接触を段階的に導入。`build_elem_layer_map()` + `filter_pairs_by_layer()` + `compute_active_layer_for_step()`。
3. **ヘリカル摩擦安定化**: 摩擦履歴 `z_t` の平行輸送（`rotate_friction_history()`）+ 低 k_t_ratio + auto k_pen 統合。xfail 3件 → PASS に昇格。
4. **接触グラフ可視化**: `plot_contact_graph()`, `plot_contact_graph_history()`, `save_contact_graph_gif()` + 15テスト。
5. **接触グラフ時系列収集**: `ContactSolveResult.graph_history` フィールド追加。各ステップ終了時に `snapshot_contact_graph()` で自動記録。9テスト。
6. **7本撚り統合テスト**: auto k_pen + staged activation で3本撚りテスト4件追加。7本撚りはNR収束限界で引き続きxfail。

## 変更内容

### 1. k_pen 自動推定: `auto_beam_penalty_stiffness()`

`xkep_cae/contact/law_normal.py` に追加。

**推定式**: `k_pen = scale * 12 * E * I / L³ / max(1, n_contact_pairs)`

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `E` | — | ヤング率 [Pa] |
| `I` | — | 代表断面二次モーメント [m⁴] |
| `L_elem` | — | 代表要素長さ [m] |
| `n_contact_pairs` | 1 | 予想同時アクティブペア数 |
| `scale` | 0.1 | 基本スケーリング係数 |

ContactConfig に `k_pen_mode="beam_ei"`, `beam_E`, `beam_I` を追加。solver_hooks で自動適用。

### 2. 段階的接触アクティベーション

ContactConfig に `staged_activation_steps`, `elem_layer_map` を追加。

| 関数/メソッド | 説明 |
|--------------|------|
| `TwistedWireMesh.build_elem_layer_map()` | 素線 layer → 要素 layer マッピング |
| `ContactManager.max_layer()` | 最大層番号 |
| `ContactManager.compute_active_layer_for_step()` | ステップに応じた有効層計算 |
| `ContactManager.filter_pairs_by_layer()` | 層超過ペアを INACTIVE に |

### 3. ヘリカル摩擦安定化

**根本原因**: 摩擦履歴 `z_t` は旧接触フレーム (t1_old, t2_old) で蓄積されるが、`delta_ut` は新フレーム (t1_new, t2_new) で計算される。ヘリカル幾何では接触フレームが大きく回転するため不整合が発生。

**解決策**:
- `rotate_friction_history()`: 旧フレーム→新フレームの2×2回転行列で `z_t` を平行輸送
- solver_hooks: `update_geometry()` 前後で旧フレームを保存し、z_t_conv を回転
- 低 `k_t_ratio=0.01`（摩擦時デフォルト）+ 長い `mu_ramp_steps=10`

**結果**: 3本撚り摩擦テスト3件（tension/lateral/bending）が全て PASS に昇格。

### 4. 接触グラフ可視化

`xkep_cae/contact/graph.py` に追加。

| 関数 | 説明 |
|------|------|
| `plot_contact_graph()` | 1スナップショットの matplotlib 描画（ノード・エッジ・反力ラベル） |
| `plot_contact_graph_history()` | 4パネル時系列プロット（エッジ数/ノード数/反力合計/散逸） |
| `save_contact_graph_gif()` | 時系列 GIF アニメーション（Pillow） |
| `_circular_layout()` | 自動ノード配置（円形レイアウト） |

### 5. 接触グラフ時系列収集

- `ContactSolveResult` に `graph_history: ContactGraphHistory` フィールド追加
- solver_hooks: 各ステップ終了時に `snapshot_contact_graph()` で自動記録
- `__init__.py`: ContactEdge, ContactGraph, ContactGraphHistory, snapshot_contact_graph をエクスポート

## ファイル変更

### 変更
- `xkep_cae/contact/law_normal.py` — `auto_beam_penalty_stiffness()` 追加
- `xkep_cae/contact/law_friction.py` — `rotate_friction_history()` 追加
- `xkep_cae/contact/solver_hooks.py` — graph_history, 摩擦平行輸送, k_pen自動推定, 段階的アクティベーション統合
- `xkep_cae/contact/pair.py` — ContactConfig 拡張（k_pen_mode, beam_E, beam_I, staged_*）, ContactManager 拡張
- `xkep_cae/contact/graph.py` — 可視化関数追加
- `xkep_cae/contact/__init__.py` — エクスポート追加
- `xkep_cae/mesh/twisted_wire.py` — `build_elem_layer_map()` 追加
- `tests/contact/test_twisted_wire_contact.py` — 摩擦テスト xfail→pass, auto k_pen テスト追加, グラフ収集テスト追加
- `tests/contact/test_law_friction.py` — `TestRotateFrictionHistory` 5テスト追加

### 新規作成
- `tests/contact/test_auto_beam_kpen.py` — auto_beam_penalty_stiffness テスト（13テスト）
- `tests/contact/test_staged_activation.py` — 段階的アクティベーションテスト（13テスト）
- `tests/contact/test_contact_graph_viz.py` — 可視化テスト（15テスト）

## テスト結果

```
tests/contact/test_auto_beam_kpen.py         13 passed
tests/contact/test_staged_activation.py      13 passed
tests/contact/test_contact_graph_viz.py      15 passed  (新規)
tests/contact/test_law_friction.py           32 passed  (+5: rotate_friction_history)
tests/contact/test_twisted_wire_contact.py   26 passed, 3 xfail (+13: auto_kpen+staged+graph+friction_pass)
全テスト:                                    1206 collected
lint/format:                                ruff check + ruff format パス
```

## 確認事項

- 既存テスト影響なし（摩擦テストのパラメータ変更のみ後方互換）
- lint/format 全クリア
- 7本撚りテストは引き続き xfail（NR内部ループの収束限界、接触ペア36+）
- 接触グラフ可視化のCJKフォント警告はコスメティック（DejaVu Sansに日本語グリフなし）

## TODO

### 次ステップ

- [ ] 7本撚り収束改善（接触特化プリコンディショナーまたは準ニュートン法）→ Phase 4.7 Level 1
- [ ] 弧長法＋接触統合テスト → Phase 4.7 で座屈必要時に着手
- [ ] 撚線ヒステリシス観測（引張・ねじり・曲げの繰り返し荷重テスト）
- [ ] 接触グラフ時系列の統計分析・可視化ワークフロー構築

---
