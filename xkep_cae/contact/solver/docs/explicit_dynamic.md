# 陽的中央差分接触動的解析 driver（status-378 Phase 2）

[← contact_friction.md](contact_friction.md) | [← README](../../../../README.md)

`ExplicitDynamicProcess` は `ContactFrictionProcess` の `solver_mode="explicit"`
時に呼ばれる 1 増分 driver。`NewtonDynamicProcess` と排他で、NR 反復を経由
せずに `ExplicitCentralDifferenceProcess.step()` を 1 ステップ呼び出して
変位を前進させる。

設計の本体は時間積分モジュール側に集約している:

- 数理定式化 / Courant 安定条件 / API:
  [time_integration/docs/time_integration_explicit.md](../../../time_integration/docs/time_integration_explicit.md)
- status-377 Phase 1（Process 単体実装）:
  [docs/status/status-377.md](../../../../docs/status/status-377.md)
- status-378 Phase 2（solver path 配線）:
  [docs/status/status-378.md](../../../../docs/status/status-378.md)

## driver の責務

1. `ContactForceAssemblyProcess` で `f_int + f_c`（接触力含む内力）を組み立て
2. Courant 監視（`courant_check_interval` 増分ごと）:
   - `K_T = assemble_tangent(u)` を sparse Gerschgorin 上界で評価
   - $\Delta t_\mathrm{sub} > 0.9 \cdot \Delta t_c$ ならカットバック要求として
     `failure_reason="courant"` で `diverged=True` を返す
3. `ExplicitCentralDifferenceProcess.step(u, f_ext, f_int_eff, dt, fixed_dofs)`
   で u を 1 ステップ前進（速度・加速度更新は内部で行う）
4. `DynamicStepOutput` を返す（`converged=True`, `convergence_type="explicit"`）

## 依存の最小化

- NR 反復、接線剛性のフル組立、Uzawa 外側ループ、line search、AL、減衰、
  pair-wise relaxation 等の escape hatch はいずれも陽解法では発動しない
  （いずれも陰解法 NR 専用）
- Courant 監視のために K_T は組み立てるが、線形ソルバーには渡さない

## 既知のスケーリング障壁（status-378 実測 + status-379 解決）

- 19 本撚線（実機規模）では Courant 臨界 $\Delta t_c$ が陰解法 dt より
  $O(10^3)$ 倍小さい想定（status-377 §7.2）。**status-378 7 本実測で**
  $\Delta t_\mathrm{sub} / \Delta t_c = 3 \times 10^5$ を観測。本 driver
  単独では非現実的だが、**status-379 で集中質量スケーリング**
  （Belytschko §6.4.2）の auto-tune を `ExplicitDynamicProcess.process()` の
  Courant 監視に統合し、19 本撚線 90° 曲げで **frac=1.0 完走**
  （E_kin/E_strain=1.15%）を達成した。

## auto-tune（status-379 候補 (h1)）

`ExplicitDynamicInput.mass_scaling_auto=True` のとき、Courant 監視で
$\Delta t_\mathrm{sub} > 0.9 \cdot \Delta t_c$ を検知すると:

1. 必要 $\beta$ を逆算: `target = current_beta * dt_sub / (0.9·dt_c)`
2. cap 適用: `capped = min(target, mass_scaling_max_beta)`
3. 5% 以上の成長要求のみ実適用（数値ノイズ抑制）:
   `current_beta * 1.05 < capped` で `set_mass_scaling_beta(capped)`
4. cap 到達時は `failure_reason="courant_cap"` を返し上位 stepping に
   dt 縮小カットバックを要求

詳細は `time_integration/docs/time_integration_explicit.md §質量スケーリング` 参照。
