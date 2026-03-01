# Status 097: xfailテスト根本対策 + S3パラメータチューニング + S4/S6ベンチマーク

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-01
**ブランチ**: `claude/execute-status-todos-bCMwM`
**テスト数**: 1906（fast: 1542 / slow: 364）

## 概要

status-096 の TODO 4件を実行:
1. xfailテストの根本対策（λ_nキャッピング + best-state fallback）
2. S3パラメータチューニング基盤
3. S4: 剛性比較ベンチマーク（HEX8連続体 vs 梁モデル）
4. S6: 1000本撚線メッシュ生成・速度ベンチマーク

## 実施内容

### 1. xfailテスト根本対策: λ_nキャッピング + best-state fallback

**根本原因**: AL乗数 λ_n が outer loop で蓄積し続け、n_outer_max=5 で暴走。
adaptive omega（ω=0.01→0.16）の段階的増大と合わさり、接触力がオーバーシュート。

**修正**:
- `ContactConfig.lambda_n_max_factor` 追加（λ_n 上限 = factor × k_pen × search_radius）
- `update_al_multiplier()` に `lambda_n_max` パラメータ追加（上限クリッピング）
- `newton_raphson_with_contact` / `newton_raphson_block_contact` に best-state fallback:
  最終 outer iteration の inner NR 失敗時、前回の収束状態に復帰

**変更ファイル**:
- `xkep_cae/contact/pair.py`: `lambda_n_max_factor` フィールド追加
- `xkep_cae/contact/law_normal.py`: `update_al_multiplier()` にクリッピングロジック
- `xkep_cae/contact/solver_hooks.py`: 両ソルバーに best-state fallback

**xfail変更**:
| テスト | 変更前 | 変更後 |
|---|---|---|
| `test_three_strand_outer3_vs_outer5` | xfail | **PASS**（xfail除去） |
| `test_seven_strand_outer5_converges` | xfail | **PASS**（xfail除去） |
| `test_three_strand_tension_hysteresis_area` | — | **新規PASS**（3本撚り摩擦ヒステリシス） |
| `test_seven_strand_tension_hysteresis_area` | xfail | xfail維持（60+ペア同時摩擦は現行限界） |
| `test_diagonal_mode_3strand_converges` | xfail | xfail維持（diagonal接線の本質的制約） |
| `test_structural_only_mode_3strand_converges` | xfail | xfail維持（structural_only接線の制約） |

### 2. S3パラメータチューニング基盤

**変更ファイル**: `tests/test_s3_benchmark_timing.py`

- `_run_benchmark()` に新パラメータ追加:
  `lambda_n_max_factor`, `al_relaxation`, `k_pen_scaling`, `staged_activation`,
  `adaptive_omega`, `use_block_solver`, `gap` 等
- `TestS3ParameterTuning` クラス新設（6テスト）:
  - 7本: ブロックソルバー + adaptive omega で**収束確認**
  - 19/37本: 収束レポート（現状は未収束 → ソルバー改良が必要）
  - 61/91本: タイミングレポート
  - スケーリングレポート

**チューニング結果**:
- `lambda_n_max_factor=0.1` は n_outer_max=5 安定化に有効
- ただし beam EI ベースの k_pen と組み合わせると収束を阻害する場合あり
- 7本: ブロックソルバー + adaptive omega（ω_min=0.01, growth=2.0）で安定収束
- 19本以上: ブロックソルバーでも第1ステップで不収束 → 構造的改善が必要

### 3. S4: 剛性比較ベンチマーク

**新規ファイル**: `tests/test_s4_stiffness_benchmark.py`（4テスト）

矩形断面 HEX8（SRI+B-bar）と Timoshenko 3D 梁の等価剛性比較:

| 剛性 | HEX8 誤差 | 梁 誤差 | 結論 |
|---|---|---|---|
| 軸 (EA/L) | 0.46% | 0.00% | 両方正確 |
| 曲げ (3EI/L³) | 1.92% | 0.09% | 両方正確 |
| ねじり (GJ/L) | 25.50% | 0.00% | 梁が圧倒的に正確 |

**結論**: 梁モデルは円形断面特性を正確に反映でき、撚線解析には梁モデルが適切。
HEX8 は矩形断面近似の制約があり、特にねじり剛性で大きな誤差が出る。

### 4. S6: 1000本撚線メッシュ生成・broadphaseスケーリング

**新規ファイル**: `tests/test_s6_1000strand_benchmark.py`（9テスト）

| 素線数 | 節点数 | 要素数 | DOF | メッシュ生成 | broadphase | 候補ペア |
|---|---|---|---|---|---|---|
| 91 | 455 | 364 | 2,730 | 0.001s | 0.12s | 66,066 |
| 271 | 1,355 | 1,084 | 8,130 | 0.002s | 1.04s | 586,314 |
| 547 | 2,735 | 2,188 | 16,410 | 0.003s | 5.02s | 2,330,304 |
| 1000 | 5,000 | 4,000 | 30,000 | 0.006s | 17.89s | 7,335,879 |

**結論**: メッシュ生成はボトルネックにならない。broadphase は O(n²) スケーリングで
1000本で ~18s、候補ペア ~733万。midpoint prescreening による候補削減が次のステップ。

## テスト結果

| テストスイート | 結果 |
|---|---|
| fast テスト | 1542 |
| slow テスト | 364 |
| テスト総数 | 1906 |
| 接触テスト（fast） | 518 passed, 26 skipped |
| xfail除去 | 2件（n_outer_max=5安定化） |
| 新規テスト | 20件追加 |

## TODO

- [ ] S3: 19本以上の接触NR収束改善（ブロックソルバーの根本改良）
- [ ] S3: 前処理パラメータチューニング（ILU drop_tol, Schur対角近似精度）
- [ ] S5: ML導入（接触プリスクリーニングGNN + k_pen推定ML）
- [ ] S6: 1000本の接触NR収束テスト（S3収束改善後）
- [ ] broadphase候補削減（midpoint prescreeningの効果測定）
- [ ] 残xfailテスト3件の根本対策（7本摩擦60+ペア、diagonal/structural_only接線）

## 確認事項

- lambda_n_max_factor はチューニング対象ペアの k_pen レンジに依存する。
  beam EI推定では k_pen が小さいため factor=0.1 でキャップが効きすぎる場合あり。
  手動k_pen指定時と自動推定時で最適値が異なる可能性がある。
- 19本以上の収束は接触ソルバーの本質的改良が必要。
  候補: マルチレベル前処理、接触面ごとのローカルNR、mortar法の改良。
