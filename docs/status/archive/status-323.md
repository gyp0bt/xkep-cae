# status-323: beam oscillation 物理テスト修復 + status-322 TODO 消化

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-12
- **ブランチ**: `claude/check-status-todos-gxsY2`
- **テスト数**: 459+13+22+5（既存数維持、beam oscillation 5 失敗→0 失敗 + 1 skip）
- **契約違反**: **0 件**
- **条例違反**: **0 件**

## TL;DR

status-322 TODO 3 件を消化:

1. **`test_beam_oscillation` の 5 件 pre-existing 失敗を修復** — 根本原因は UL `update_reference()` が自由振動の復元力を消失させること + 集中加振による高次モード混入。3 点修正で 13 passed, 1 skipped（matplotlib 未インストール）。
2. **`_find_caller` skip list 拡張検討** — BenchmarkRunnerProcess / ParameterSweepBenchmarkProcess はフレームワーク内部ではなくユーザー可視のラッパーであり、skip list 追加は不要と判断。
3. **distance culling / symbolic factor reuse 調査** — 実装は次 status 向け。距離カットは `_get_active_pairs` レベルの gap 閾値フィルタで実装可能だが、Huber smoothing との不連続性リスクがあり慎重な設計が必要。

## 背景 — beam oscillation 5 失敗の根本原因

status-322 で `test_beam_oscillation.py` の 5 失敗が「baseline で既に存在する pre-existing 失敗」と確認されていた。本 status で根本原因を特定し修正。

### 失敗パターン

| テスト | 失敗値 | 期待値 | 根本原因 |
|--------|--------|--------|----------|
| `test_small_amplitude_ratio` | ratio=17.905 | 0.8-1.3 | UL復元力消失 + 集中加振 |
| `test_small_oscillation_detected` | 方向反転0回 | ≥1回 | UL復元力消失 |
| `test_large_deflection_bounded` | 80.63mm | <50mm | UL復元力消失 |
| `test_numerical_dissipation_rate` | 3950.689 | <2.0 | UL f_int(u_total) 不正 |
| `test_render_produces_images` | 0画像 | ≥3画像 | matplotlib未インストール |

### 根本原因 1: UL `update_reference()` による復元力消失

ContactFrictionProcess は UL モードで毎ステップ参照配置を更新する（status-281）:

```python
if _ul and hasattr(ul_assembler, "update_reference"):
    ul_assembler.update_reference(u_incr)
```

これにより:
- 参照配置が毎ステップ現在の変形形状に更新される
- `f_int(0) = 0`（参照配置での内力ゼロ）が各ステップで成立
- **自由振動の復元力（初期配置への引き戻し力）が失われる**
- 梁は一方向に偏向し続け、振動しない

これは CLAUDE.md に記録されている既知問題:
> CR梁ULのf_int=0問題の根本解決: update_referenceを跨がない設計

### 根本原因 2: 集中加振による高次モード混入

旧実装は中央1節点に集中初速度を与えていた:
```python
velocity[6 * wire_mid_node + 1] = -v0  # 中央のみ
```

- 集中加振は全奇数モードを励起（1次、3次、5次...）
- `modal_ratio = M_modal/m_mid ≈ n_elems/2` で初速度を補正するが、90%以上のエネルギーが高次モードに分配
- 振幅比が要素数依存となり不安定

## 実施内容

### 1. UL 参照更新の無効化

**ファイル**: `xkep_cae/numerical_tests/beam_oscillation.py`

```python
callbacks=AssembleCallbacks(
    assemble_tangent=assembler.assemble_tangent,
    assemble_internal_force=assembler.assemble_internal_force,
    ul_assembler=None,  # 自由振動では参照配置更新不要（復元力保持）
),
```

`ul_assembler=None` により:
- ソルバーは `_ul = False` で動作し、`update_reference()` を呼ばない
- `coords_ref` は初期配置のまま維持
- `f_int(u_total)` が初期配置からの総変位に対する正しい復元力を返す
- CR 梁定式化は小振幅（0.1mm, 5mm for L=100mm）の変形を UL なしで正確に扱える

### 2. 初速度をモード形状分布に変更

```python
# 旧: 集中加振（中央1節点のみ、modal_ratio補正）
velocity[6 * wire_mid_node + 1] = -v0

# 新: 1次モード形状分布（全節点に sin(πx/L) 重み）
v0 = omega1 * cfg.amplitude
for i in range(n_nodes):
    x_i = mesh_data.node_coords[i, 0]
    velocity[6 * i + 1] = -v0 * math.sin(math.pi * x_i / cfg.wire_length)
```

- 1次モードのみを励起（高次モード混入なし）
- `q̇₁(0) = ω₁ * amplitude` → `δ₁_max = amplitude`（解析解と一致）
- `modal_ratio` 計算が不要（要素数非依存）
- amplitude_ratio ≈ 1.0 が保証される

### 3. time_arr をアダプティブ時間増分に対応

```python
# 旧: 等間隔仮定（アダプティブ dt と不整合）
time_arr = np.linspace(0, t_total, n_hist)

# 新: load_history から実際の時刻を復元
load_hist = solver_result.load_history
if len(load_hist) == n_hist:
    time_arr = np.array(load_hist) * t_total
```

- エネルギー計算の中心差分速度近似の精度が向上
- `energy_decay_ratio` の信頼性が改善

### 4. matplotlib importorskip

```python
def test_render_produces_images(self, tmp_path):
    pytest.importorskip("matplotlib")
    ...
```

## 検証結果

```
tests/test_beam_oscillation.py  13 passed, 1 skipped in 45.88s
```

| テスト | Before | After |
|--------|--------|-------|
| `test_small_amplitude_ratio` | FAILED (17.905) | PASSED |
| `test_small_oscillation_detected` | FAILED (0回) | PASSED |
| `test_small_energy_conservation` | PASSED | PASSED |
| `test_large_amplitude_nonlinear` | PASSED | PASSED |
| `test_contour_fields_exist` | PASSED | PASSED |
| `test_large_strain_distribution` | PASSED | PASSED |
| `test_large_deflection_bounded` | FAILED (80.63) | PASSED |
| `test_numerical_dissipation_rate` | FAILED (3950.689) | PASSED |
| `test_render_produces_images` | FAILED (0画像) | SKIPPED (matplotlib) |

- 接触回帰テスト: 376 passed, 5 skipped
- 契約違反: 0件
- lint/format: OK

## 変更ファイル

- `xkep_cae/numerical_tests/beam_oscillation.py`: UL無効化、モード形状初速度、time_arr修正
- `tests/test_beam_oscillation.py`: matplotlib importorskip

## _find_caller skip list 評価結果

現在の skip list:
- `_SKIP_BASENAMES`: `{"base.py", "diagnostics.py", "runner.py"}`
- `_SKIP_MODULES`: `{"xkep_cae.core.base", "xkep_cae.core.diagnostics", "xkep_cae.core.runner"}`

BenchmarkRunnerProcess / ParameterSweepBenchmarkProcess はフレームワークの透過層ではなく、ベンチマーク実行を明示的にオーケストレーションするユーザー可視のプロセス。レポートで呼び出し元として表示するのが正しい（例: ParameterSweepBenchmarkProcess → ContactFrictionProcess の経路はベンチマーク文脈を示す有用情報）。

**結論: skip list 拡張不要。**

## distance culling / symbolic factor reuse 調査メモ

### distance culling

- **実装箇所**: `_get_active_pairs()` に gap 閾値フィルタ追加、または `ContactForceStStiffnessProcess._process_batch` で `state.gap > threshold` をフィルタ
- **閾値案**: `dist > 2.5 * (r_a + r_b)` (status-321 提案)
- **リスク**: Huber smoothing の遷移帯（`delta_h`）で力が非ゼロの領域をカットすると NR 収束に影響
- **推奨**: broadphase の search_radius よりもタイトな narrowphase カットとして実装し、K_st アセンブリのみに適用（力評価は全ペアで維持）
- **規模**: ベンチマーク付きで 1 status 相当

### symbolic factorization reuse

- **実装箇所**: `LinearSolveProcess` で pypardiso の `analyze()` を NR 反復間でキャッシュ
- **前提条件**: sparsity pattern が NR 反復内で不変（活性集合が変化しなければ成立）
- **リスク**: 活性集合変化時にパターン更新が必要（detection logic が複雑）
- **規模**: pypardiso API 調査 + キャッシュ実装で 1 status 相当

## TODO（次担当者向け）

### 直近

- [ ] **distance culling 実装**（1 status 相当）: 上記調査メモに基づき、K_st アセンブリ向け narrowphase gap カットを実装。n=37 以上の掃引でスケーリング改善を確認。
- [ ] **symbolic factorization reuse 実装**（1 status 相当）: pypardiso `analyze()` キャッシュを LinearSolveProcess に統合。
- [ ] **被膜 ON プロファイル + pypardiso 環境再ベンチ** — status-320 TODO 継続
- [ ] **ファイバー梁 Phase F1 着手** — status-313 継続

### 中期

- [ ] **リスタート解析方式への移行**: 今回の UL 修正は `ul_assembler=None` による回避策。根本解決は ContactFrictionProcess の I/O を `(u, v, a, 接触ペア)` 入出力に整理し、`update_reference` を解析ステップ間で跨がない設計にすること。
- [ ] **ProcessMetaclass._profile_data と ProcessExecutionLog の統合** — status-322 TODO 継続

## STA2 準拠チェック

- [x] **数値の捏造なし**: `pytest -v` 出力で 13 passed, 1 skipped を確認
- [x] **再現手順記載**: 上記「検証結果」セクション
- [x] **テスト数記載**: 459+13+22+5（status-322 から不変）
- [x] **契約違反 0 件維持**: `validate_process_contracts.py` 実行済み
- [x] **lint/format 検証**: `ruff check` + `ruff format --check` OK
- [x] **ベースライン比較**: 修正前に同一テストが 5 failed であることを確認済み
- [x] **接触回帰 376 passed**: 実測
