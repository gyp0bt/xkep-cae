# status-134: ソルバー高速化完遂 + ソルバー一本化

[← README](../../README.md) | [← status-index](status-index.md) | [← status-133](status-133.md)

**日付**: 2026-03-07

## 概要

status-133のTODOに基づき、2つの主要タスクを実施:
1. **要素ループのベクトル化** — assemble_cr_beam3d の全要素バッチ化で **12.6x高速化**
2. **ソルバー一本化** — n_load_steps 廃止、入出力データクラス化、step/increment 用語統一

## 実施内容

### 1. 要素ループのベクトル化（12.6x高速化）

CR梁要素アセンブリの要素ループを NumPy ベクトル化で完全置換。
Pythonのforループを排除し、全要素を一括バッチ処理に変更。

**バッチ化された関数群**:
- `_batch_rotvec_to_rotmat` — 四元数経由のバッチ回転ベクトル→回転行列
- `_batch_rotmat_to_rotvec` — Shepperd法ベクトル化によるバッチ逆変換
- `_batch_rodrigues_rotation` — バッチ最小回転行列（Rodrigues公式）
- `_batch_build_local_axes` — バッチ局所座標系構築
- `_batch_tangent_operator` / `_batch_tangent_operator_inv` — バッチ接線演算子
- `_batch_timo_ke_local` — バッチ局所剛性行列
- `_batch_skew` — バッチ歪対称行列
- `_assemble_cr_beam3d_batch` — 全要素一括の接線剛性+内力計算

**計測結果**:

| 構成 | スカラーループ | バッチ版 | 高速化比 |
|------|-------------|---------|---------|
| 7本(112要素) | 62.88ms | 4.99ms | **12.6x** |
| 19本(304要素) | — | 8.14ms | 線形スケーリング |
| 37本(592要素) | — | 14.17ms | 線形スケーリング |
| 91本(1456要素) | — | 33.92ms | 線形スケーリング |

**正確性検証**: 内力誤差 1.16e-10, 剛性行列相対誤差 3.66e-16（機械精度）

### 2. ソルバー一本化

#### 2a. n_load_steps 廃止
- `adaptive_timestepping` のデフォルトを `True` に変更
- `n_load_steps` パラメータに DeprecationWarning を追加
- `n_load_steps` は `dt_initial_fraction = 1.0/n_load_steps` に自動変換
- wire_bending_benchmark の n_load_steps 参照を dt_initial_fraction に移行

#### 2b. 入出力データクラス化
- `NCPSolverInput` データクラスを追加
  - 全入力パラメータを構造化
  - `solve()` メソッドでソルバー直接実行
  - `to_kwargs()` で既存API互換の辞書生成
- `NCPSolveResult.n_load_steps` → `n_increments` に改名（property で後方互換維持）

#### 2c. 用語統一
- increment: 荷重増分（adaptive_timestepping で自動制御）
- iteration: Newton 反復（各 increment 内での非線形反復）
- 進捗表示: "Step X/N" → "Incr X (frac=Y)" に変更

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/elements/beam_timo3d.py` | バッチ化関数群(+621行)、assemble_cr_beam3d のバッチディスパッチ |
| `xkep_cae/contact/solver_ncp.py` | NCPSolverInput追加、n_load_steps deprecated、n_increments改名、用語統一 |
| `xkep_cae/contact/__init__.py` | NCPSolverInput エクスポート追加 |
| `xkep_cae/numerical_tests/wire_bending_benchmark.py` | n_load_steps → dt_initial_fraction 移行 |

## TODO

### 次の優先
- [ ] 19本撚線の曲げ揺動収束確認（scripts/で検証）
- [ ] 19本→37本のスケールアップ
- [ ] CR梁の摩擦接触不収束の原因調査
- [ ] 37本Layer1+2圧縮の段階的活性化による収束改善確認
- [ ] NCPソルバー版S3ベンチマーク（AL法との計算時間比較）
- [ ] Cosserat Rodの解析的接線剛性実装
- [ ] graphベースMLによる時間増分スキーマの改善（ロードマップS6）
- [ ] テストファイルの n_load_steps → dt_initial_fraction 段階的移行（DeprecationWarning は出るが機能的に問題なし）

### 確認事項
- バッチ化は analytical_tangent=True (デフォルト) でのみ有効。数値微分パスは逐次ループを維持。
- n_load_steps は deprecated だが後方互換性を維持。既存テストは DeprecationWarning を出すが正しく動作する。
- NCPSolverInput.solve() は newton_raphson_contact_ncp のラッパー。新規コードでの使用を推奨。
