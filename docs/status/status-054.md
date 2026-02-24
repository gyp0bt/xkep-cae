# status-054: 7本撚り収束改善 + 撚線ヒステリシス観測 + 接触グラフ統計分析

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-02-24
**作業者**: Claude Code
**テスト数**: 1234（+28）

## 概要

status-053 の TODO を実行。Phase 4.7 Level 0 の残タスク3件を完了。

1. **7本撚り収束改善**: Modified Newton法（K_T再利用）+ 接触力under-relaxation（contact damping）+ sqrt k_penスケーリング。線形アセンブラでは効果限定的、7本撚りは引き続きxfail（3テスト追加）。
2. **撚線ヒステリシス観測**: `run_contact_cyclic()` サイクリック荷重ランナー + `CyclicContactResult` データクラス。3本撚り引張/曲げ/ねじり往復荷重テスト（8テスト追加）。
3. **接触グラフ統計分析**: `ContactGraphHistory` に統計メソッド7件追加（stick/slip比率、法線力統計、連結成分数、接触持続マップ、累積散逸、サマリー）。17テスト追加。

## 変更内容

### 1. 7本撚り収束改善

`xkep_cae/contact/pair.py` に ContactConfig 拡張:

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `use_modified_newton` | `False` | Modified Newton法（構造剛性再利用） |
| `modified_newton_refresh` | `5` | K_T再計算間隔（反復数） |
| `contact_damping` | `1.0` | 接触力under-relaxation係数（1.0=無緩和） |
| `k_pen_scaling` | `"linear"` | k_penのn_pairsスケーリング: `"linear"` / `"sqrt"` |

`xkep_cae/contact/law_normal.py`:
- `auto_beam_penalty_stiffness()` に `scaling` パラメータ追加（`"sqrt"` で √n_pairs スケーリング）

`xkep_cae/contact/solver_hooks.py`:
- Inner loop: Modified Newton（K_T_frozen キャッシュ、refresh間隔で再計算）
- 接触力under-relaxation: `f_c = ω * f_c_raw + (1-ω) * f_c_prev`

**結果**: 線形Timo3Dアセンブラでは K_T が定数のため Modified Newton の効果なし。36+ペア同時収束は Schur 補完法/Uzawa 法等のブロック分解が必要。7本撚りテストは xfail 維持。

### 2. 撚線ヒステリシス観測（サイクリック荷重ランナー）

`xkep_cae/contact/solver_hooks.py`:

**新規データクラス**: `CyclicContactResult`
- `n_phases`: フェーズ数
- `amplitudes`: 荷重振幅リスト
- `load_factors`: 全ステップの荷重係数
- `displacements`: 全ステップの変位ベクトル
- `converged`: 全フェーズ収束フラグ
- `graph_history`: 統合された接触グラフ時系列
- `n_total_steps`: 総ステップ数

**新規関数**: `run_contact_cyclic()`
- 複数フェーズの連続実行（接触状態引き継ぎ）
- `amplitudes=[1.0, 0.0]` で正→逆の往復荷重
- `amplitudes=[1.0, -1.0, 0.0]` でフルサイクル
- `f_ext_base` パラメータで荷重オフセット対応

`xkep_cae/contact/solver_hooks.py`:
- `newton_raphson_with_contact()` に `f_ext_base` パラメータ追加（サイクリック荷重のベース荷重）

### 3. 接触グラフ統計分析

`xkep_cae/contact/graph.py` に `ContactGraphHistory` メソッド追加:

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `stick_slip_ratio_series()` | `np.ndarray` | 各ステップの slip 比率（0〜1） |
| `mean_normal_force_series()` | `np.ndarray` | 各ステップの平均法線力 |
| `max_normal_force_series()` | `np.ndarray` | 各ステップの最大法線力 |
| `connected_component_count_series()` | `np.ndarray` | 各ステップの連結成分数 |
| `contact_duration_map()` | `dict[tuple,int]` | ペア→接触ステップ数マップ |
| `cumulative_dissipation_series()` | `np.ndarray` | 累積散逸エネルギー時系列 |
| `summary()` | `dict` | 8項目の統計サマリー |

## ファイル変更

### 変更
- `xkep_cae/contact/pair.py` — ContactConfig 拡張（use_modified_newton, modified_newton_refresh, contact_damping, k_pen_scaling）
- `xkep_cae/contact/law_normal.py` — auto_beam_penalty_stiffness に sqrt scaling 追加
- `xkep_cae/contact/solver_hooks.py` — Modified Newton, contact damping, f_ext_base, CyclicContactResult, run_contact_cyclic
- `xkep_cae/contact/graph.py` — 統計分析メソッド7件追加
- `xkep_cae/contact/__init__.py` — CyclicContactResult, run_contact_cyclic エクスポート追加
- `tests/contact/test_twisted_wire_contact.py` — 7本撚り収束改善テスト3件追加（xfail）

### 新規作成
- `tests/contact/test_hysteresis.py` — サイクリック荷重テスト（8テスト: 基本動作3 + ヒステリシス5）
- `tests/contact/test_graph_statistics.py` — グラフ統計テスト（17テスト: stick/slip 4, 力統計 3, 連結成分 3, 持続マップ 3, 散逸 1, サマリー 3）

## テスト結果

```
tests/contact/test_hysteresis.py            8 passed  (新規)
tests/contact/test_graph_statistics.py     17 passed  (新規)
tests/contact/test_twisted_wire_contact.py 26 passed, 6 xfail (+3: 7本撚り収束改善)
全テスト:                                  1234 collected
lint/format:                              ruff check + ruff format パス
```

## 確認事項

- 既存テスト影響なし（新パラメータはデフォルトで既存動作を維持）
- 7本撚り収束は本質的に36+同時接触ペアのNR限界。線形アセンブラではModified Newtonに効果なし。ブロック分解（Schur補完/Uzawa法）が次の課題。
- サイクリック荷重テスト（test_hysteresis.py）は各テスト60〜340秒程度（合計336秒）
- 摩擦付きヒステリシスで散逸エネルギー非ゼロを確認
- 接触グラフ統計分析はグラフ時系列のポスト処理ワークフローの基盤

## TODO

### 次ステップ

- [ ] 弧長法＋接触統合テスト → Phase 4.7 で座屈必要時に着手
- [ ] 7本撚りブロック分解ソルバー（Schur補完法 or Uzawa法）
- [ ] 撚線ヒステリシスの荷重-変位曲線可視化 + ヒステリシスループ面積計算
- [ ] 接触グラフ統計の可視化関数（plot_statistics_dashboard）
- [ ] 被膜モデル（剛性寄与 + 摩擦制御）→ Phase 4.7 Level 1 準備

---
