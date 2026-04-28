[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-377: 陽的中央差分時間積分 Phase 1 — Process 単体実装 + `solver_mode` config + 設計仕様

**日付**: 2026-04-28
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+28 passed（status-376 比 +28、新 explicit テスト + solver_mode config テスト）

## 概要

候補 (g) 3 サブライン全却下（status-376）が確定し、陰解法 NR 側 escape hatch
アプローチが限界に到達したことを受けて、**陽的中央差分時間積分** への移行 Phase 1
を実装。19 本撚線以上の K_c x/z カップリング不整合（status-344）を時間積分自体で
安定化する目的で、新 Process `ExplicitCentralDifferenceProcess` を `xkep_cae/time_integration/`
に新設。

**Phase 1 制約**: Process 単体実装 + 単体テスト + 設計仕様のみ。専用 solver path
への配線は **Phase 2（次 status）** で実施し、`solver_mode="explicit"` 実行時は
`NotImplementedError` で Phase 2 待機を明示する。`solver_mode="implicit"` (default)
で既存挙動は完全不変。

## 1. 実装

### 1.1 `ExplicitCentralDifferenceProcess` 新設（`xkep_cae/time_integration/strategy.py`）

**+216 行**。集中質量 $M_\mathrm{lump}$ を用いた陽解法中央差分:

$$
M_\mathrm{lump} \cdot a_n = F_\mathrm{ext} - F_\mathrm{int}(u_n) - C \cdot v_{n-1/2}
$$

| API | 役割 |
|-----|------|
| `__init__(M, damping_matrix, mass_lumping)` | 集中質量 / 減衰 / ロンピング方式（"row_sum" / "diagonal" / "none"）|
| `step(u, f_ext, f_int, dt, fixed_dofs)` | 陽解法 1 ステップ前進（NR 不要、`u_{n+1}` を返す） |
| `critical_dt(k_max_eigenvalue)` | Courant 臨界刻み $\Delta t_c = 2/\omega_\mathrm{max}$ |
| `set_initial_state(velocity, acceleration)` | 初期 v / a 設定 |
| `predict / correct / effective_stiffness / effective_residual` | `TimeIntegrationStrategy` Protocol 適合（Verlet 予測子 + 対角質量 K_eff） |
| `checkpoint / restore_checkpoint` | カットバック対応 |

**集中質量化** (`_lump_mass`):

- "row_sum"（default）: $M_\mathrm{lump}[i] = \sum_j M[i,j]$
- "diagonal": $M_\mathrm{lump}[i] = M[i,i]$
- "none": 既に対角化済みの場合のスキップ

**0 質量 DOF 安全化** (`_invert_diagonal`): 固定 DOF（M[i,i]=0）の逆数を 0 に保ち、ゼロ除算を回避。

### 1.2 ファクトリ拡張（`_create_time_integration_strategy`）

`solver_mode: str = "implicit"` 引数追加。`"explicit"` で `ExplicitCentralDifferenceProcess`
を返す分岐を実装。`mass_lumping: str = "row_sum"` も追加。default `"implicit"` で既存
動作完全不変。

### 1.3 `StrandBendingOscillationConfig.solver_mode` field（`numerical_tests/strand_bending_oscillation.py`）

```python
solver_mode: Literal["implicit", "explicit"] = "implicit"
```

`StrandBendingOscillationProcess.process()` 冒頭で `cfg.solver_mode == "explicit"`
を検知すると `NotImplementedError` を発生させ、Phase 1 段階での誤実行を防止。
エラーメッセージは設計仕様 `time_integration_explicit.md` を参照案内する。

### 1.4 `__init__.py` re-export

```python
from xkep_cae.time_integration.strategy import (
    ExplicitCentralDifferenceProcess,
    GeneralizedAlphaProcess,
    QuasiStaticProcess,
    TimeIntegrationInput,
    TimeIntegrationOutput,
)
```

`__all__` に追加し、外部モジュールからインポート可能化。

## 2. 設計仕様

`xkep_cae/time_integration/docs/time_integration_explicit.md` 新設（+126 行）:

- §概要: status-376 の (g) 全終了 → 陽解法移行の動機
- §数理定式化: 中央差分式 / 集中質量化 / Courant 安定条件
- §API: Process I/O + Protocol 互換メソッド
- §状態保持: $v_{n-1/2}$ 半時刻オフセット + checkpoint
- §19 本撚線への意図: NR 不整合接線剛性に依存しない陽解法の利点と $\Delta t$ 制限の代償
- §Phase 1 / Phase 2 分割: 配線スコープ明確化
- §MCDD 脱法回避: pattern 1 (tol 緩和) / 5 (既存 skip) / 6 (骨格 status) チェック

参考文献: Belytschko 2014 §6.2, Hughes 2000 §9.1.2.

## 3. 単体テスト

### 3.1 `time_integration/tests/test_strategy.py` （+25 テスト）

**`TestExplicitCentralDifferenceProcess`** (`@binds_to`):

| テスト | 検証内容 |
|--------|----------|
| `test_is_dynamic` | 動的解析判定 |
| `test_lumped_mass_row_sum` | 行和ロンピング数値検証 |
| `test_lumped_mass_diagonal` | 対角抽出ロンピング数値検証 |
| `test_lumped_mass_invalid_raises` | 不正方式で ValueError |
| `test_invert_diagonal_zero_safe` | 0 質量 DOF でゼロ除算回避 |
| `test_critical_dt_basic` | $\Delta t_c = 2/\omega$ 数値一致 |
| `test_critical_dt_zero_returns_inf` | 剛性 0 で inf 返却 |
| `test_step_zero_dt_returns_copy` | dt=0 でコピー返却 |
| `test_step_unit_force_advances_velocity` | M=2 / F=1 → a=0.5 数値検証 |
| `test_step_fixed_dofs_zeroed` | 拘束 DOF で a=0 / v=0 |
| `test_sdof_free_vibration_period` | SDoF 1 周期戻り誤差 < 5%（dt=T/100） |
| `test_sdof_energy_bounded` | 5 周期エネルギー有界 < 10%（symplectic 性） |
| `test_sdof_critical_dt_courant` | Courant 越え 1.5x で明確発散（>100x） |
| `test_damping_term_in_step` | $C \cdot v$ 残差減算検証 |
| `test_set_initial_state` | v/a 設定 |
| `test_checkpoint_restore` | チェックポイント往復 |
| `test_effective_stiffness_diagonal_mass` | $K_\mathrm{eff} = M_\mathrm{lump}/dt^2$ 数値検証 |
| `test_effective_stiffness_small_dt_passthrough` | dt=0 で K passthrough |
| `test_effective_residual_no_damping` | 減衰なしで R=R |
| `test_effective_residual_with_damping` | $R - Cv$ 数値検証 |
| `test_dense_mass_matrix_converted` | numpy → CSR 変換 |
| `test_dense_damping_matrix_converted` | numpy → CSR 変換（減衰） |
| `test_process_returns_output` | Process プロトコル一致 |

**`TestCreateTimeIntegrationStrategy`** に +3 テスト:
- `test_explicit_mode_returns_explicit_process`
- `test_explicit_mode_with_initial_state`
- `test_implicit_mode_default`

**`TestTimeIntegrationProtocolConformance`** parametrize に Explicit を追加（+1 ケース）。

### 3.2 `numerical_tests/tests/test_strand_bending_oscillation.py` （+3 テスト）

| テスト | 検証内容 |
|--------|----------|
| `test_solver_mode_default_implicit` | default = "implicit" |
| `test_solver_mode_explicit_constructible` | "explicit" で config 構築可能 |
| `test_solver_mode_explicit_raises_not_implemented` | Phase 1 制約: NotImplementedError |

## 4. 検証

### 4.1 Default OFF 回帰（gate 必達）

| 項目 | 結果 |
|------|------|
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK |
| `pytest xkep_cae/contact/ xkep_cae/mathematics/ xkep_cae/time_integration/` | **649 passed, 5 skipped**（status-376 比 +25 explicit + +36 既存 time_integration） |
| `test_helical_3d_hermite` | rel_err=2.18e-07 維持（status-356 機械精度継続） |
| `test_strand_bending_oscillation.py` | 21 passed（+3 solver_mode テスト） |
| `ruff check` / `ruff format --check` | OK |

### 4.2 数値検証（陽解法精度）

`test_sdof_free_vibration_period`: $m=1$, $k=100$（$\omega=10$, $T \approx 0.628$）の
SDoF 自由振動を $\Delta t = T/100$ で 1 周期積分。初期変位 $u_0=1$ から戻り値を
比較し誤差 < 5% を確認（symplectic Euler の O(dt·ω) 位相誤差水準）。

`test_sdof_energy_bounded`: 同設定で 5 周期積分し、各時刻のエネルギー
$E = \frac{1}{2}ku^2 + \frac{1}{2}mv^2$ の最大値が初期 $E_0$ の 1.10 倍以下に
収まることを確認（symplectic 性）。

`test_sdof_critical_dt_courant`: Courant 臨界 $\Delta t_c = 0.2$ に対し
$\Delta t = 0.3$（1.5x 超過）で 20 ステップ積分。$|u| > 100$ に発散することを
確認し、安定条件の必要性を実証。

## 5. MCDD 脱法 pattern 回避

- pattern 1（tol 緩和）: 単体テスト全 28 本は機械精度ベース。SDoF 戻り誤差 < 5%
  / Courant 越え 100x 発散 / ロンピング数値完全一致。tol 事後緩和は実施せず
- pattern 5（既存テスト skip）: 既存 GeneralizedAlpha / QuasiStatic 35+ tests +
  contact 468 + math 109 全 pass、`test_helical_3d_hermite` rel_err=2.18e-07 維持
- pattern 6（骨格 status）: Phase 1 を Process 単体実装 + 28 unit tests + 設計
  仕様で完結（status-365 ContactDamping / status-374 PairwiseFreezing と同パターン）

## 6. 引継ぎ（Phase 2 / 次 status へ）

### 6.1 Phase 2 で実施すること

1. **陽解法専用 solver path 新設**: `xkep_cae/contact/solver/_explicit_dynamic.py`
   または既存 `_newton_dynamic.py` への分岐追加。`NewtonDynamicProcess` と排他。
2. **インクリメント単位 step() 駆動**: `ContactFrictionProcess` のメインループから
   `ExplicitCentralDifferenceProcess.step(u, f_ext, f_int, dt, fixed_dofs)` を呼び、
   NR 反復を経由しないインクリメント前進を実装
3. **Courant 監視 + adaptive Δt**: 接線剛性の最大固有値推定（power iteration 等）+
   $\Delta t = 0.9 \cdot \Delta t_c$ で安全側に運用、active 集合変動時の縮小ステップ
4. **接触ペア再構築**: ステップ間で gap 再計算 / 摩擦状態継承 / break-up 検知
5. **19 本撚線 90° 曲げで `frac=1.0` 完走**: implicit + AL n=2 の 0.5746 を上回る
   ことが Phase 2 gate

### 6.2 設計上の注意点

- 陽解法は **集中質量近似** に依存。撚線梁要素の質量行列で行和ロンピングが物理的に
  妥当か（Cosserat 回転 DOF への影響）を Phase 2 着手時に再確認
- $\Delta t_c = O(10^{-6})$ オーダーが想定され、implicit (~10^-3) より $10^3$ 倍
  細かい刻み。CPU 時間は increment 数 1000x になり得るため、`step()` 1 回あたりの
  コスト（K_T 不要、F_int 評価 + 対角質量除算のみ）の高速化が鍵
- 接触の active 集合変動は `F_int` の不連続として現れるが、Courant 内なら
  数値発散しない仮説。Phase 2 で 19 本実機検証

### 6.3 副次（保留）: K_mat の x/z 二次補正項追加

status-376 §5.3 から継承。陽解法移行で 19 本 frac=1.0 達成できれば優先度低下、
失敗時は数理側の K_c 補正再開。

## 7. 運用所見

### 7.1 Phase 1/2 分割の意義

陽解法時間積分は数理的にシンプル（NR ループ不要）だが、既存 `NewtonDynamicProcess`
が 1781 行 + 各種 escape hatch（chattering_freeze / pairwise_freeze / AL / EMA /
contact_damping）を持つため、これらと併存する陽解法 path の設計は影響範囲が大きい。
Phase 1 で時間積分 Process 単体を完成させ、Phase 2 で solver wiring に集中する
分割は、status-365/366 (ContactDamping) / status-374/375 (PairwiseFreezing) で
確立された安全な進め方。

### 7.2 既存 `predict / correct` Protocol との互換性

陽的中央差分は本来 NR 反復を経由しないが、`TimeIntegrationStrategy` Protocol への
適合のため Verlet 予測子 + 対角質量 K_eff を提供。これにより将来の陰陽混合解法
（陽解法で初期値、陰解法で精緻化）への拡張余地を残せる。

### 7.3 Phase 2 への接続

現状 `solver_mode="explicit"` 実行時は `NotImplementedError` で停止する。Phase 2
着手時はこのガードを削除し、`_process_explicit()` 等の専用パスへ分岐させる。
config field の interface は固定したため、Phase 2 での破壊的変更は不要。
