# status-133: 適応時間増分制御の改善 — n_load_steps=1対応 + 安定化成長戦略

[← README](../../README.md) | [← status-index](status-index.md) | [← status-132](status-132.md)

**日付**: 2026-03-07

## 概要

status-132のTODOから、自動時間増分制御をn_load_steps=1で動作可能にする改善を実施。物理ベースの初期時間増分パラメータ、安定化成長戦略、フェーズごとの成長閾値チューニングを導入。

## 実施内容

### 1. dt_initial_fraction パラメータ追加

`newton_raphson_contact_ncp` に `dt_initial_fraction` パラメータを追加。n_load_steps=1でも物理ベースの適切な初期時間増分からスタート可能。

- **問題**: n_load_steps=1 → 初期Δt=1.0 → 全荷重をワンステップ → 発散 → カットバック浪費
- **解決**: dt_initial_fraction=1/30（90°/3°=30ステップ相当）で初期Δtを明示指定
- dt_min/dt_max の自動計算もdt_initial_fractionベースに更新

### 2. 安定化成長戦略（TCP輻輳制御類似）

連続成功ステップ数 `_consecutive_good` を追跡し、安定Δtに達したら成長率を漸減:

- **成長フェーズ**（consecutive_good ≤ 2）: 通常の成長率（1.5x）
- **安定フェーズ**（consecutive_good ≥ 3）: 漸減成長率（1 + (grow-1) / consecutive_good）
- **カットバック後**: consecutive_good=0にリセット → 成長モードに復帰

### 3. フェーズごとの成長閾値チューニング

`dt_grow_iter_threshold` パラメータをソルバーに追加（0=config値を使用）:

- **Phase1（曲げ）**: threshold=8（接触なし → 積極的成長で10ステップに削減）
- **Phase2（揺動）**: threshold=5（接触変化あり → 保守的、カットバック回避）

### 4. ベンチマーク結果

7本撚線 90°曲げ（Phase1のみ）:

| 構成 | 時間 | NR反復 | 実効ステップ |
|------|------|--------|------------|
| hardcoded (n_steps=30, adaptive=False) | 8.86s | 120 | 30 |
| adaptive (n_steps=1, dt_init=0.033) | **6.45s** | 81 | ~12 |
| **高速化比** | **1.37x** | | |

7本撚線 90°曲げ + 揺動1周期:

| 構成 | 時間 |
|------|------|
| baseline (n_steps=30, adaptive=True) | 40.6s |
| adaptive (n_steps=1, dt_init=0.033) | **38.97s** |
| **高速化比** | **1.04x** |

**注**: status-132の26.6秒はCI環境（より高速なCPU）での計測値。本環境ではbaseline自体が40.6秒。

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver_ncp.py` | dt_initial_fraction, dt_grow_iter_threshold パラメータ追加、安定化成長戦略 |
| `xkep_cae/contact/pair.py` | dt_grow_iter_threshold デフォルト値（5を維持） |
| `xkep_cae/numerical_tests/wire_bending_benchmark.py` | n_load_steps=1対応、Phase1/2でdt_initial_fraction指定 |
| `scripts/bench_adaptive_dt.py` | 新規: n_load_steps=1 vs hardcoded比較ベンチマーク |
| `scripts/bench_baseline.py` | 新規: status-132設定でのベースライン計測 |

## TODO

### 次の優先

- [ ] 非線形接触動解析ソルバーモジュール完全一本化（ユーザー要求）
  - n_load_steps完全廃止
  - 入出力をデータクラス化
  - step/incrementの用語整理（incrementは時間増分、stepは解析ステップ）
- [ ] graphベースMLによる時間増分スキーマの改善（ロードマップS6）
- [ ] 19本撚線の曲げ揺動収束確認（scripts/で検証）
- [ ] 要素ループのベクトル化（残りの46%ボトルネック）

### 確認事項

- Phase2の揺動ステップ6（方向転換点）でカットバック発生は不可避（dt=0.2で発散）
  - 初期Δtをさらに小さく（0.1）しても全体は遅くなる（ステップ数増加のオーバーヘッド）
  - ソルバー統合時に、前ステップの収束Δtを次ステップに引き継ぐ仕組みが有効
- CI環境と開発環境のCPU速度差が大きい（CI: ~26s, 開発: ~40s）
