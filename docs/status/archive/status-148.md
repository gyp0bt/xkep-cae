# status-148: 摩擦曲げスクリプトのsmooth penalty統一 + Mortar不使用時du未定義バグ修正

[← README](../../README.md) | [← status-index](status-index.md) | [← status-147](status-147.md)

**日付**: 2026-03-09
**テスト数**: 2271（変更なし）

## 概要

摩擦曲げ検証スクリプト群を smooth penalty + Uzawa に統一。
NCP鞍点系の摩擦接線剛性符号問題（status-147）を踏まえ、
次の担当者が誤って NCP モードで実行しないよう、スクリプト・コメント・docstring を整備。

加えて、solver_ncp.py で Mortar 有効かつ接触ペアなしの場合に `du` が未定義になるバグを修正。

## 変更内容

### 1. verify_friction_bending_convergence.py — smooth penalty 統一

- `contact_mode="smooth_penalty"` を追加
- `use_mortar=True` を削除（smooth penalty モードでは不使用）
- docstring・ヘッダー・コメントに「smooth penalty 必須」を明記
- 次の担当者がデフォルト（NCP鞍点系）で実行→発散する事故を防止

### 2. render_friction_bend_3d.py — 不要パラメータ削除

- `use_mortar=True` を削除
- コメントに status-147 への参照を追加

### 3. solver_ncp.py — Mortar+接触なし時の du 未定義バグ修正

**問題**: `_use_mortar=True` かつ `n_ncp_active == 0`（接触なし）かつ `len(mortar_nodes) == 0` の場合、
if/elif の条件分岐すべてをスキップし `du` が未定義のまま `du_norm_val = float(np.linalg.norm(du))` に到達。

**修正**: `else` ブランチを追加し、接触なしの線形求解（`_solve_saddle_point_contact` に空制約行列を渡す）にフォールバック。

### 4. 3Dパイプレンダリング画像（smooth penalty で再生成）

`docs/verification/` に以下の画像を出力:

| 画像 | 内容 |
|------|------|
| `friction_bend_initial_iso.png` | 初期状態 Isometric |
| `friction_bend_initial_xz.png` | 初期状態 Side (XZ) |
| `friction_bend_case1_45deg_iso.png` | Case 1: 45° 曲げ Isometric |
| `friction_bend_case1_45deg_xz.png` | Case 1: 45° 曲げ Side (XZ) |
| `friction_bend_case2_90deg_iso.png` | Case 2: 90° 曲げ Isometric |
| `friction_bend_case2_90deg_xz.png` | Case 2: 90° 曲げ Side (XZ) |
| `friction_bend_case3_mid_iso.png` | Case 3: 揺動途中 Isometric |
| `friction_bend_case3_final_iso.png` | Case 3: 最終状態 Isometric |
| `friction_bend_case3_final_xz.png` | Case 3: 最終状態 Side (XZ) |

## 後方互換に関する注意

solver_ncp.py の `else` ブランチ追加は後方互換を崩す可能性がある。
具体的には、Mortar有効+接触なしの初期反復で「何もしない」→「線形求解する」に変更されるため、
既存のテストで挙動が変わる可能性がある。

**次回セッションでの確認事項**:
1. `ruff check` / `ruff format` の実行
2. `pytest tests/ -x --timeout=60` で既存テストの回帰確認
3. solver_ncp.py の `else` ブランチが既存NCP（摩擦なし）テストに影響しないか確認
4. 影響がある場合は `else` ブランチの条件を `_use_mortar and not _smooth` に限定する等の対応

## 互換ヒストリー

| 旧構成 | 新構成 | 移行status |
|--------|--------|-----------|
| `verify_friction_bending_convergence.py` NCP鞍点系 | smooth penalty + Uzawa | status-148 |

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `scripts/verify_friction_bending_convergence.py` | contact_mode="smooth_penalty" 追加、use_mortar削除、docstring更新 |
| `scripts/render_friction_bend_3d.py` | use_mortar削除、コメント整備 |
| `xkep_cae/contact/solver_ncp.py` | Mortar+接触なし時のdu未定義バグ修正（else分岐追加） |
| `docs/verification/friction_bend_*.png` | 3Dレンダリング画像9枚 |

---
