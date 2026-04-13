# status-327: ファイバー梁 Phase F2 — MultiLayerFrictionDegrading1D 実装

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-13
- **ブランチ**: `claude/check-status-todos-D6sZC`
- **テスト数**: 459+13+22+5+8+12+12（F2 API 6 + Physics 6 テスト追加）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

## TL;DR

status-326 TODO「ファイバー梁 Phase F2 着手」を実行。`MultiLayerFrictionDegrading1D` を frozen dataclass として実装し、`05_smooth_teardrop.py` プロトタイプとの完全一致（rtol=1%）を確認。12 テスト全合格。

---

## 1. MultiLayerFrictionDegrading1D 実装

### 概要

status-313 設計仕様 Phase F2 の完了判定:
> `MultiLayerFrictionDegrading1D` 実装、`05_smooth_teardrop.py` 再現 rtol ≤ 1%

### 物理モデル

N 層の並列摩擦要素 + 弾性バックボーン + 接触剛性劣化:

```
σ = E_base · ε + Σ_i σ_i(ε, slip_i, slipped_i)
```

各層 i の return mapping:
1. k = k_degraded[i] if slipped[i] else k_virgin[i]
2. trial = k · (ε - slip[i])
3. |trial| ≤ f_y[i] → 弾性（σ_i = trial, E_t_i = k）
4. |trial| > f_y[i] → slipped=True, 劣化剛性で再計算, slip 更新

### KH との正確なパラメータ対応

σ = E_base·ε + k·(ε - slip) の構造から:
- 弾性剛性: E_base + k = E → k = E²/(E+H)
- スリップ時接線: E_base = E·H/(E+H)
- 降伏条件: f_y = σ_y·E/(E+H)

### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/elements/fiber/materials.py` | `MultiLayerFrictionDegrading1D` クラス追加 |
| `xkep_cae/elements/fiber/__init__.py` | エクスポート追加 |
| `xkep_cae/elements/fiber/tests/test_materials_api.py` | `TestMultiLayerFrictionDegradingAPI` 6 件追加 |
| `xkep_cae/elements/fiber/tests/test_materials_physics.py` | `TestMultiLayerFrictionDegradingPhysics` 6 件追加 |

### 設計の要点

- **frozen dataclass**: `evaluate()` は新しい `Fiber1DState` を返す。入力 state は mutation しない（C17 準拠）
- **状態不変最適化**: 弾性域（全層スリップなし）では入力 state オブジェクトをそのまま返却（`changed` フラグ管理）
- **`initial_state()` メソッド**: N 層分の初期 slip/slipped タプルを生成するファクトリ
- **`n_layers` プロパティ**: 層数アクセサ
- **Fiber1DState の slip/slipped フィールド**: Phase F1 で予約済みの tuple フィールドをそのまま活用

### テスト（12 件追加）

| テスト | 分類 | 内容 |
|--------|------|------|
| `test_protocol_compliance` | API | Fiber1DMaterialStrategy isinstance 検査 |
| `test_evaluate_return_types` | API | 戻り値型 (float, float, Fiber1DState) |
| `test_initial_state_factory` | API | slip/slipped 長さ・初期値 |
| `test_n_layers_property` | API | 層数プロパティ |
| `test_frozen_state_immutable` | API | frozen mutation 不可 |
| `test_elastic_range_state_unchanged` | API | 弾性域で state is 同一 |
| `test_smooth_teardrop_reproduction` | Physics | 05_smooth_teardrop.py 完全再現（N=150, rtol=1%） |
| `test_degradation_asymmetric_slopes` | Physics | β=0.25 で U/L < 1（非対称勾配） |
| `test_n_layers_convergence` | Physics | N=30/60/150 でピーク応力収束 |
| `test_consistent_tangent_fd` | Physics | FD 接線検証（弾性域 + スリップ中） |
| `test_energy_dissipation_positive` | Physics | ヒステリシスループ面積 > 0 |
| `test_single_layer_matches_kh` | Physics | N=1, β=1.0 で BilinearKH 完全一致 |

---

## 2. 被膜 ON プロファイル + pypardiso 環境再ベンチ

pypardiso 未インストール環境のため**保留**。

---

## TODO（次担当者向け）

### 直近

- [ ] **被膜 ON プロファイル + pypardiso 環境再ベンチ** — status-326 TODO 継続
- [ ] **ファイバー梁 Phase F3 着手** — `CircularFiberSection` + `FiberSectionIntegratorProcess` 実装（断面積分ループ + N,M,C_section）
- [ ] **n=61+ Type D stall 対策** — 大 n_strands での接触活性化時に Type D stall が発生

### 中期

- [ ] **リスタート解析方式への移行**: ContactFrictionProcess の I/O を `(u, v, a, 接触ペア)` 入出力に整理
- [ ] **空間ブロック分離 or ペアクラスタリング**: 物理的接触ペア数の n² 成長を抑制する構造的対策

---

## 再現手順

```bash
# Phase F2 テスト
pytest xkep_cae/elements/fiber/tests/ -v

# 契約検証
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: pytest -v 出力 24 件全合格
- [x] **再現手順記載**: 上記
- [x] **テスト数記載**: 459+13+22+5+8+12+12
- [x] **契約違反 0 件維持**: validate_process_contracts.py 実行済み
- [x] **lint/format 検証**: ruff check + ruff format --check OK
