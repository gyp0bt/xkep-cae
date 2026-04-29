# 陽的中央差分時間積分（status-377 Phase 1）

[← time_integration.md](time_integration.md) | [← README](../../../../README.md)

## 概要

`ExplicitCentralDifferenceProcess` は集中質量 $M_\mathrm{lump}$ を用いた陽解法
時間積分。Newton-Raphson 反復を要さず、各時間ステップで $F_\mathrm{int}$ /
$F_\mathrm{ext}$ の 1 回評価で進む。

`docs/status/status-376.md` で候補 (g) 3 サブライン全却下が確定し、NR alg 側
escape hatch アプローチが限界に到達したことを受けて status-377 で着手する陽解法
時間積分の Phase 1 実装。

## 数理定式化

### 時間離散化（中央差分）

$$
\begin{aligned}
M_\mathrm{lump} \cdot a_n &= F_\mathrm{ext}(t_n) - F_\mathrm{int}(u_n) - C \cdot v_{n-1/2} \\
v_{n+1/2} &= v_{n-1/2} + \Delta t \cdot a_n \\
u_{n+1} &= u_n + \Delta t \cdot v_{n+1/2}
\end{aligned}
$$

集中質量 $M_\mathrm{lump}$ は対角行列なので $a_n = M_\mathrm{lump}^{-1} \cdot R_n$
が要素ごとの除算で計算でき、線形ソルバーが不要。

### 集中質量化（mass lumping）

行和ロンピング（default）:

$$
M_\mathrm{lump}[i, i] = \sum_j M_\mathrm{consistent}[i, j]
$$

対角抽出（"diagonal"）も提供。

### Courant 安定条件

中央差分は条件付き安定。臨界時間刻み:

$$
\Delta t_c = \frac{2}{\omega_\mathrm{max}}, \quad
\omega_\mathrm{max} = \sqrt{\lambda_\mathrm{max}(K, M_\mathrm{lump})}
$$

実運用では安全係数を見て $\Delta t = 0.9 \cdot \Delta t_c$ 程度を用いる。

## API

### Process I/O

`TimeIntegrationStrategy` Protocol（`predict / correct / effective_stiffness /
effective_residual`）を実装するが、陽解法は本来 NR を経由しないため、`step()`
メソッドが直接的な前進を提供する。

```python
proc = ExplicitCentralDifferenceProcess(
    mass_matrix=M,                 # 一貫質量
    damping_matrix=C,              # オプション
    mass_lumping="row_sum",        # "row_sum" / "diagonal" / "none"
)
proc.set_initial_state(velocity=v0, acceleration=a0)

# 1 ステップ前進
u_new = proc.step(u_n, f_ext, f_int, dt, fixed_dofs=fixed)

# Courant 臨界 dt
dt_c = proc.critical_dt(k_max_eigenvalue=lam_max)
```

### Protocol 互換メソッド

| メソッド | 陽解法での意味 |
|----------|----------------|
| `predict(u, dt)` | Verlet 予測子 $u + \Delta t v + 0.5 \Delta t^2 a$ |
| `correct(u, du, dt)` | $v_{n+1/2} = (u_{n+1} - u_n) / \Delta t$ で速度を逆算 |
| `effective_stiffness(K, dt)` | $K_\mathrm{eff} = M_\mathrm{lump} / \Delta t^2$（K は捨てる） |
| `effective_residual(R, dt)` | $R_\mathrm{eff} = R - C v$ |

ただし default 運用では `step()` で 1 行完結するため、`predict / correct` 経由は
将来の対称化（陰陽混合解法）と Protocol 適合性のための保険である。

## 状態保持

陽解法は半時刻ステップの速度 $v_{n-1/2}$ を保持する。`vel` 属性は最新の
$v_{n+1/2}$ を反映し、ステップ中の状態遷移を追跡する。

`checkpoint()` / `restore_checkpoint()` でカットバック時に $v$ / $a$ を巻き戻し可能。

## 19 本撚線 Type D stall への意図

`status-344 mat_only rel_err mean=44%` の K_c x/z カップリング不整合に対し、
陰解法 NR は active 集合振動下で収束半径が狭まる（候補 (g) 全候補が gate 0.6
未達）。陽解法は線形化を行わないため、不整合な接線剛性に依存しない:

- $a_n$ は $F_\mathrm{int}(u_n)$ の値のみから決まる（K の整合性を要さない）
- active 集合変化は $F_\mathrm{int}$ の不連続として反映されるが、$\Delta t$ が
  Courant 内なら数値発散しない
- 代償: $\Delta t$ が Courant 制限される（陰解法より $O(10^{-3})$ 倍）

## Phase 1 / Phase 2 分割（status-377）

### Phase 1（本 status）

- `ExplicitCentralDifferenceProcess` 単体実装 + 21 単体テスト
- `_create_time_integration_strategy` に `solver_mode="explicit"` 分岐
- `StrandBendingOscillationConfig.solver_mode` field（default `"implicit"`）
- 設計仕様（本ドキュメント）

NR ソルバー path への配線は **未実施**。`solver_mode="explicit"` 指定時は
`NotImplementedError` を発生させ、Phase 2 待機を明示する。

### Phase 2（次 status）

- 陽解法専用 solver path を新設（`ExplicitDynamicProcess`、`NewtonDynamicProcess`
  と排他）
- インクリメント単位での `step()` 駆動 + Courant 監視 + adaptive $\Delta t$
- 接触ペア再構築 / break-up 検知のステップ間処理
- 19 本撚線 90° 曲げで `frac=1.0` 完走（implicit + AL n=2 の 0.5746 を上回ること）

## MCDD 脱法回避

- pattern 1（tol 緩和）: 単体テスト 21 本は機械精度ベース（SDoF 自由振動 5%
  以内、Courant 越えで明確発散 100x 以上）
- pattern 5（既存テスト skip）: GeneralizedAlpha / QuasiStatic 既存 35+ tests
  全 pass、Protocol 適合 parametrize に Explicit を追加（3 件）
- pattern 6（骨格 status）: Phase 1 を Process 単体実装 + 21 unit tests + 設計
  仕様で完結。配線は Phase 2 でユーザー確認の上着手

## 関連 status

- [status-376](../../../../docs/status/status-376.md): 候補 (g2) AL 再導入却下、(g) サブライン全終了
- [status-373](../../../../docs/status/status-373.md): solver_mode 併存方針 § 4'
- [status-377](../../../../docs/status/status-377.md): 陽解法 Phase 1（本実装）

## 参考文献

- Belytschko, Liu, Moran (2014) "Nonlinear Finite Elements for Continua and
  Structures" 2nd ed., §6.2 Explicit Methods, §6.4 Energy and Momentum Conservation
- Hughes (2000) "The Finite Element Method", §9.1.2 Central Difference Method
