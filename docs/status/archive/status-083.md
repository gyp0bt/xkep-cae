# status-083: Phase S1 同層除外フィルタ + NCP摩擦拡張 + line contact統合

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-28
- **テスト数**: 1875（+25: 同層除外14 + NCP摩擦line contact 11）
- **ブランチ**: claude/execute-status-todos-uhlKc

## 概要

Phase S1 の中核2機能を実装:
1. **同層除外フィルタ** (`exclude_same_layer`): broadphase で同層接触ペアを除外（~80%削減）
2. **NCP摩擦拡張 + line contact統合**: Semi-smooth Newtonソルバーに Coulomb摩擦と Gauss積分を統合

## 実装詳細

### 1. 同層除外フィルタ

- `ContactConfig.exclude_same_layer: bool = False` フラグ追加
- `ContactManager.detect_candidates()` に同層ペア除外ロジック追加
  - `elem_layer_map` 参照、layer_map にない要素は除外しない（安全側）
- `ContactManager.count_same_layer_pairs()` 診断メソッド追加
- 3本撚り（全同層）で全ペア除外、7本撚りで異層ペアのみ保持、19本撚りで30%以上削減を確認

### 2. NCP Coulomb摩擦拡張

- `_compute_friction_forces_ncp()`: NCP法線力 p_n = max(0, λ+k_pen*(-g)) ベースの摩擦return mapping
- `_build_friction_stiffness()`: **法線剛性を含まない**摩擦接線剛性行列の構築
  - NCP鞍点系が k_pen * G_A^T G_A で法線剛性を処理するため、`compute_contact_stiffness` を使うと二重カウントになる問題を解決
- `newton_raphson_contact_ncp()` に `use_friction/mu/mu_ramp_steps` パラメータ追加
- `ContactConfig` 経由の摩擦設定伝播

### 3. NCP line contact統合

- `newton_raphson_contact_ncp()` に `line_contact/n_gauss` パラメータ追加
- assembly既存インフラ（`compute_contact_force/compute_contact_stiffness`）を活用
- NCP活性セット判定はPtPギャップ、力・剛性評価はGauss積分のハイブリッド方式

## テスト

### 同層除外 (tests/contact/test_exclude_same_layer.py) — 14テスト

- `TestExcludeSameLayerConfig` (3): デフォルト無効、有効化、layer_map不在時
- `TestExcludeSameLayerDetection` (4): フィルタ有無比較、同層未除外、不明要素
- `TestExcludeSameLayerTwistedWire` (3): 7本撚り削減効果、3本撚り全除外、19本撚り大幅削減
- `TestCountSameLayerPairs` (4): 基本カウント、INACTIVE除外、map不在、不明要素

### NCP摩擦+line contact (tests/contact/test_ncp_friction_line.py) — 11テスト

- `TestNCPFrictionBasic` (4): 収束、摩擦効果検証、μランプ、config伝播
- `TestNCPFrictionForceUnit` (2): 接触なし→ゼロ、μ=0→ゼロ
- `TestNCPLineContactBasic` (3): 収束、PtP比較、config伝播
- `TestNCPLineContactFriction` (2): 全機能統合収束、λ非負性

### 回帰確認

- 既存NCP/接触テスト: 62/62 PASSED（test_ncp, test_solver_ncp, test_staged_activation, test_exclude_same_layer）
- lint/format: `ruff check` / `ruff format --check` 問題なし

## ファイル変更

### 新規
- `tests/contact/test_exclude_same_layer.py` — 同層除外フィルタテスト（14件）
- `tests/contact/test_ncp_friction_line.py` — NCP摩擦+line contactテスト（11件）
- `docs/status/status-083.md` — 本ステータス

### 変更
- `xkep_cae/contact/pair.py` — exclude_same_layer フラグ + detect_candidates 同層除外 + count_same_layer_pairs
- `xkep_cae/contact/solver_ncp.py` — _compute_friction_forces_ncp, _build_friction_stiffness, 摩擦/line contact パラメータ
- `docs/roadmap.md` — S1チェックボックス更新、テスト数更新
- `docs/status/status-index.md` — status-083 追加
- `README.md` — テスト数・現在の状態更新
- `CLAUDE.md` — 現在の状態更新

## 技術的知見

### 法線剛性二重カウント問題

NCP鞍点系（K_eff = K_T + k_pen * G_A^T * G_A）で法線ペナルティ剛性が処理されるため、
摩擦剛性の計算に `compute_contact_stiffness()` を使うと法線剛性が二重にカウントされる。
`_build_friction_stiffness()` を新設し、摩擦接線剛性のみを分離して構築することで解決。

### NCP+摩擦の収束制限

横方向力（接線方向の外力）と摩擦の組み合わせでは、NCP+摩擦の∂f_fric/∂λ
交差結合項が不足するため、Newton反復が収束しにくい。
現状はパラメータ調整（小さいμ、多ステップ、μランプ）で対処。
将来的には完全な摩擦NCP定式化（Alart-Curnier拡張）での改善を検討。

## 未解決（引き継ぎ）

### Phase S1 残り
1. **C6-L5: Mortar離散化** — セグメント境界の接触力連続化（最優先）
2. 7本撚り + Mortar 検証（貫入率 < 1%、Outer loop不要を確認）

### Phase S2以降
3. CPU並列化（要素並列、broadphase並列）
4. 91本ベンチマーク

### 改善候補
5. NCP+摩擦の完全結合定式化（∂f_fric/∂λ の鞍点系への組込み）
6. Newton反復内 z_t 復元による一貫した摩擦linearization

---
