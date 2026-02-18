# status-026: Phase 5.1〜5.2 動的解析実装

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-18

## 概要

Phase 5（動的解析）の時間積分スキーマ（5.1）と集中質量行列（5.2）を実装した。

## 実装内容

### Phase 5.1: Newmark-β / HHT-α 時間積分

**新規ファイル**: `xkep_cae/dynamics.py`

- `TransientConfig` — 過渡応答解析の設定（dt, n_steps, β, γ, α_HHT）
- `TransientResult` — 結果データ（時刻歴、変位・速度・加速度）
- `solve_transient()` — Newmark-β / HHT-α 法による線形過渡応答ソルバー
  - 平均加速度法（β=1/4, γ=1/2）がデフォルト（無条件安定）
  - HHT-α法: α ∈ [-1/3, 0] の数値減衰パラメータ
  - LU分解の事前計算（定数係数系で効率的）
  - 固定DOFの自動処理（自由DOFへの縮約）
  - 外力: 定数ベクトルまたは時間関数 f(t)
  - 疎行列入力対応

### Phase 5.2: 集中質量行列（HRZ法）

**更新ファイル**: `xkep_cae/numerical_tests/frequency.py`

- `_beam2d_lumped_mass_local()` — 2D梁の集中質量行列（6×6対角）
- `_beam3d_lumped_mass_local()` — 3D梁の集中質量行列（12×12対角）
- `_assemble_lumped_mass_2d()` — 2Dグローバル集中質量行列アセンブリ
- `_assemble_lumped_mass_3d()` — 3Dグローバル集中質量行列アセンブリ
- HRZ法: 並進 m/2, 回転 mL²/78（非特異性確保）
- 座標変換不要（対角行列は回転不変）

## テスト

**新規ファイル**: `tests/test_dynamics.py`（27テスト）

| クラス | テスト数 | 内容 |
|--------|---------|------|
| TestNewmarkSDOF | 8 | 非減衰/減衰自由振動、ステップ荷重、調和荷重、初速度、固定DOF |
| TestTransientConfig | 5 | dt/n_steps/α/β/γ のバリデーション |
| TestHHTAlpha | 3 | Newmark等価性、数値減衰、ステップ応答収束 |
| TestNewmarkBeam | 3 | 固有振動数（FFT検出）、エネルギー保存、静的収束 |
| TestLumpedMass | 8 | 対角行列性、質量保存（要素/全体、2D/3D）、固有振動数収束、過渡応答 |

**全体テスト結果**: 498 passed, 2 skipped

## バグ修正

- `solve_transient()` の引数順 (M, C, K) とテストの呼び出し順 (K, M, C) の不一致を修正
  - 梁テストのヘルパー `_build_cantilever_beam_matrices()` の戻り値順序を (K, M, C) → (M, C, K) に変更
- デバッグプリントの除去
- 固有振動数テスト: 先端変位（多モード励起）→ 第1固有モード初期変位 + FFT周波数検出に改良

## コミット履歴

1. `feat: Newmark-β時間積分ソルバー実装（Phase 5.1）` — dynamics.py + SDOF テスト 13件
2. `feat: HHT-α法テスト + 梁過渡応答テスト追加（Phase 5.1 Batch 2）` — HHT-α 3件 + 梁 3件
3. `feat: 集中質量行列（HRZ法）実装（Phase 5.2）` — 集中質量 + テスト 8件

## TODO（残タスク）

- [ ] Phase 5.1: 陽解法（Central Difference）— オプション
- [ ] Phase 5.2: ElementProtocol への `mass_matrix()` 統合
- [ ] Phase 5.3: モーダル減衰
- [ ] Phase 4.3: von Mises 3D 弾塑性テスト実行（45テスト計画済み、[status-025](status-025.md)参照）
  - 注: コーディングエージェントのフリーズリスクがあるため見送り中

## 確認事項・懸念

- `solve_transient()` は密行列に変換して計算しており、大規模問題には疎行列ソルバー（直接法 or 反復法）が必要
- 現在は線形系のみ対応。非線形過渡応答（Newton-Raphson + Newmark）は今後の拡張
- 集中質量行列の回転慣性 mL²/78 は HRZ 近似値であり、Timoshenko 梁のせん断補正は考慮していない
