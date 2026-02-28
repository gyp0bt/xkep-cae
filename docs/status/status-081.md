# status-081: Phase C6-L1b — 摩擦力の line contact 拡張 + 撚線収束テスト

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-28
- **テスト数**: 1850（fast: +22, slow: +7）
- **ブランチ**: claude/friction-contact-wire-test-BSSZ8

## 概要

摩擦力評価を従来の PtP（代表点1点）から Gauss 積分による分布評価（line contact）に拡張した。加えて、3ストランド撚線モデルで line contact + Coulomb friction の収束テスト7件を追加した。

## 背景・動機

Phase C6-L1 で法線力の line-to-line Gauss 積分は実装済みだったが、摩擦力は依然として PtP 代表点のみで評価していた。これを GP（Gauss Point）レベルで独立に摩擦状態を管理し、各 GP で Coulomb return mapping を行う方式に拡張した。

## 実施内容

### 1. GP 摩擦状態フィールド（pair.py）

`ContactState` に3つの GP レベルフィールドを追加:
- `gp_z_t: list[np.ndarray]` — 各 GP の摩擦バックストレス
- `gp_stick: list[bool]` — 各 GP の stick/slip 状態
- `gp_q_trial_norm: list[float]` — 各 GP の試行弾性予測子ノルム

### 2. GP 摩擦力の Gauss 積分（line_contact.py）

2つの関数を追加:
- `compute_line_friction_forces()` — GP レベルで Coulomb return mapping を実行し、摩擦力を Gauss 積分
- `compute_line_friction_stiffness()` — GP 摩擦剛性の Gauss 積分

各 GP で独立に:
1. 弾性予測子: `q_trial = k_t * (δ_t - z_t_prev)`
2. Coulomb yield check: `‖q_trial‖ ≤ μ * p_n_gp`
3. Stick → そのまま, Slip → return mapping で投影

### 3. Assembly 統合（assembly.py）

`_assemble_contact_forces()` / `_assemble_contact_stiffness()` に `line_friction_forces` / `line_friction_stiffnesses` パラメータを追加。pre-computed な line friction 力・剛性を直接受け取る。

### 4. Solver hooks 統合（solver_hooks.py）

`newton_raphson_with_contact()` / `newton_raphson_block_contact()` の両方で:
- `line_contact=True` かつ `use_friction=True` のとき GP 摩擦力・剛性を計算
- GP 摩擦状態 (`gp_z_t`, `gp_stick`, `gp_q_trial_norm`) の初期化・更新を管理

### 5. 単体テスト（+22 fast）

| テストファイル | テストクラス | テスト数 | 内容 |
|--------------|-------------|---------|------|
| test_line_friction.py | TestGPFrictionState | 3 | GP 状態フィールド初期化・更新 |
| test_line_friction.py | TestLineFrictionForces | 9 | 摩擦力 Gauss 積分（stick/slip/ゼロ法線力/対称性） |
| test_line_friction.py | TestLineFrictionStiffness | 6 | 摩擦剛性 Gauss 積分（形状・対称性） |
| test_line_friction.py | TestLineFrictionAssembly | 4 | Assembly パイプライン統合 |

### 6. 撚線収束テスト（+7 slow）

3ストランド撚線モデル（Steel, E=200GPa, d=2mm, pitch=40mm, 16要素/ストランド）で line contact + friction を有効にした収束テスト:

| テスト | 内容 | 結果 |
|--------|------|------|
| test_3_strand_line_friction_tension | 引張荷重収束 | PASSED |
| test_3_strand_line_friction_lateral | 横荷重収束 | PASSED |
| test_3_strand_line_friction_bending | 曲げ荷重収束 | PASSED |
| test_3_strand_line_contact_no_friction | 摩擦なし line contact 収束 | PASSED |
| test_line_vs_ptp_friction_similar | PtP vs Line 摩擦の定性的一致 | PASSED |
| test_3_strand_line_friction_gp_states | GP 摩擦状態の初期化確認 | PASSED |
| test_3_strand_penetration_controlled | 侵入量比 < 10% 品質検証 | PASSED |

全7テスト 1659.87s (27:39) で完走。既存テスト（TestTwistedWireFriction 3件）もリグレッションなし。

## ファイル変更

### 新規
- `tests/contact/test_line_friction.py` — 22テスト
- `docs/status/status-081.md` — 本ステータス

### 変更
- `xkep_cae/contact/pair.py` — ContactState に GP 摩擦状態フィールド追加
- `xkep_cae/contact/line_contact.py` — `compute_line_friction_forces()`, `compute_line_friction_stiffness()` 追加
- `xkep_cae/contact/assembly.py` — line friction 力・剛性パラメータ追加
- `xkep_cae/contact/solver_hooks.py` — GP 摩擦状態管理の統合
- `xkep_cae/contact/__init__.py` — 新関数エクスポート
- `tests/contact/test_twisted_wire_contact.py` — `_make_contact_manager()` / `_solve_twisted_wire()` に `line_contact` パラメータ追加 + `TestTwistedWireLineContactFriction` クラス（7テスト）
- `README.md` — テスト数更新
- `docs/roadmap.md` — C6-L1b + テスト数更新
- `docs/status/status-index.md` — status-081 追加
- `CLAUDE.md` — 現在の状態更新

## 設計上の懸念・TODO

### 消化済み（status-080 → 081）
- [x] 摩擦力の line contact 拡張（PtP → Gauss 積分）

### 未解決（引き継ぎ）
- [ ] **C6-L5: Mortar 離散化**（必要に応じて）
- [ ] 接触プリスクリーニング GNN Step 2-5
- [ ] k_pen推定ML v2 Step 2-7
- [ ] NCP ソルバーの摩擦拡張（Coulomb 摩擦の NCP 定式化）
- [ ] NCP ソルバーの line contact 統合
- [ ] 7本撚り line contact + friction テスト

### 設計メモ
- **PtP vs Line friction**: 両手法とも同符号の変位応答を生成。Line friction のほうが分布的に摩擦を評価するため、セグメント間の精度が向上する。
- **GP 状態管理**: 各ペアの GP 数は `len(cs.gp_weights)` で決まる（Gauss 積分点数に連動）。line_contact が無効の場合は従来の PtP 摩擦ループが使われる。
- **計算コスト**: 7テスト合計 ~28分（1テストあたり ~4分）。PtP 摩擦テスト（3件, 4:32）と比較して GP 摩擦のオーバーヘッドは限定的。

---
