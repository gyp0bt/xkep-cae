# status-074: Stage S3/S4 + ブロック前処理ソルバー実装

[← README](../../README.md) | [← status-index](status-index.md)

- **日付**: 2026-02-27
- **テスト数**: 1684（+55: シース接触46 + ブロックソルバー9）
- **ブランチ**: claude/execute-status-todos-sRJm6

## 概要

status-072/073 の高優先 TODO（#4-6）を消化。CI修正 + Stage S3/S4 シース接触 + 7本撚りブロック前処理ソルバーを実装。

## 実施内容

### 1. CI修正（status-073 TODO #1 消化）

- `test-fast` から torch 依存を除去（`[dev,ml]` → `[dev]`）
- `test-slow` で CPU 版 torch を明示的にインストール
- torch 依存テストは `importorskip` で自動スキップ

### 2. Stage S3: シース-素線/被膜 有限滑り（+37テスト）

`xkep_cae/contact/sheath_contact.py` を新規作成:

| 関数/クラス | 概要 |
|------------|------|
| `SheathContactPoint` | シース接触点データ（θ, gap, p_n, z_t, dissipation） |
| `SheathContactConfig` | 設定（k_pen, k_t_ratio, mu, theta_rebuild_tol） |
| `SheathContactManager` | 管理（points, compliance_matrix, r_eff） |
| `build_sheath_contact_manager()` | TwistedWireMesh からマネージャ構築 |
| `compute_strand_theta()` | 素線の現在角度θ計算（z=0基準） |
| `update_contact_angles()` | 変形追従によるθ再配置 |
| `compute_sheath_gaps()` | ギャップ計算（径方向変位含む） |
| `evaluate_normal_forces()` | 法線力評価（ペナルティ / コンプライアンス行列） |
| `sheath_friction_return_mapping()` | 接線摩擦のreturn mapping |
| `evaluate_sheath_contact()` | 統合接触評価（法線+摩擦） |
| `assemble_sheath_forces()` | グローバル力ベクトルへの組込み |
| `rebuild_compliance_matrix()` | θ大変動時のC行列再構築 |

テスト: 37テスト（角度差, 接触フレーム, θ更新, ギャップ, 法線力, 摩擦, 統合評価, 力アセンブリ, コンプライアンス行列, θ再構築, 内径推定）

### 3. Stage S4: シース-シース接触（+9テスト）

`sheath_contact.py` に追加:

| 関数 | 概要 |
|------|------|
| `sheath_outer_radius()` | シース外径（被膜込み）算出 |
| `build_sheath_sheath_contact_manager()` | 複数ケーブル間接触管理 |
| `sheath_sheath_merged_coords()` | 複数メッシュ座標結合 |

- broadphase_aabb + インターケーブルフィルター（同一ケーブル内ペアを除外）
- テスト: 9テスト（外径, 座標マージ, 2/3ケーブル構成, 遠距離非接触, 単一ケーブルエラー）

### 4. 7本撚りブロック前処理ソルバー（+9テスト）

`xkep_cae/contact/solver_hooks.py` に追加:

| 関数 | 概要 |
|------|------|
| `_extract_strand_blocks()` | K_T から素線対角ブロック抽出 |
| `_build_block_preconditioner()` | 素線ブロック + K_c 対角 → 前処理逆行列 |
| `_solve_block_preconditioned()` | ブロック前処理付き GMRES |
| `newton_raphson_block_contact()` | Outer/Inner 二重ループ付きブロック前処理 NR |

**設計思想**:
- 構造剛性 K_T は素線ごとにブロック対角（素線間は接触 K_c のみ結合）
- 各素線ブロック（例: 30×30）の逆行列を GMRES の前処理に使用
- ILU 前処理と異なり、高 k_pen でも素線ブロックは常に良条件
- GMRES がオフダイアゴナル K_c 結合を正確に反映 → NR 二次収束を維持

**性能比較（7本撚り引張）**:

| ソルバー | 荷重ステップ | AL緩和 | Newton反復 |
|---------|-----------|--------|-----------|
| 改善モノリシック（status-065） | 50 | 0.01 | ~400 |
| ブロック前処理 | 15 | 0.01 | ~119 |
| ブロック前処理 | 20 | 0.10 | ~207 |

**制約事項**:
- 外部ループ（幾何更新 + AL乗数更新）の発散問題は未解決（n_outer_max=1 で回避）
- これはアルゴリズムレベルの問題（線形ソルバーでは解決不可）

テスト: 9テスト（3本撚り引張/曲げ/摩擦/接触力記録 + 7本撚り引張/ねじり/曲げ/15ステップ/AL0.1）

## ファイル変更

### 新規
- `xkep_cae/contact/sheath_contact.py` — S3/S4 シース接触モジュール
- `tests/contact/test_sheath_contact.py` — シース接触テスト（46テスト）
- `docs/status/status-074.md` — 本ステータス

### 変更
- `.github/workflows/ci.yml` — test-fast から torch 除去、test-slow で CPU torch
- `xkep_cae/contact/solver_hooks.py` — ブロック前処理ソルバー追加（~300行）
- `xkep_cae/contact/__init__.py` — S3/S4 + ブロックソルバーのエクスポート
- `tests/contact/test_twisted_wire_contact.py` — ブロックソルバーテスト追加（~330行）
- `README.md` — フェーズ状態更新
- `docs/roadmap.md` — TODO チェックボックス更新
- `docs/status/status-index.md` — status-074 追加

## 設計上の懸念・TODO

### 未解決
- [ ] Outer loop 発散（n_outer_max > 1 + AL omega > 0.3 で不安定）
  - 根本原因: pen_ratio > tol → k_pen 増大 → 解の変動増大 → (s,t) 大変動 → 不安定
  - 解決策候補: Mortar 離散化、反復ソルバー前処理改善、接触ペア間引き
- [ ] 7本撚りサイクリック荷重（ブロックソルバー + 摩擦でのヒステリシス観測）
- [ ] ブロックソルバーの大規模メッシュでの性能検証（16+要素/素線）

### 次ステップ候補（中優先）
- [ ] pen_ratio 改善（adaptive omega で AL 乗数段階的蓄積）
- [ ] 7本撚りサイクリック荷重テスト
- [ ] 接触プリスクリーニング GNN Step 1

---
