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

## 既知のスケーリング障壁

- 19 本撚線（実機規模）では Courant 臨界 $\Delta t_c$ が陰解法 dt より
  $O(10^3)$ 倍小さい想定（status-377 §7.2）。実機 frac=1.0 完走は
  本 driver 単独では困難で、**集中質量スケーリング**（Belytschko §6.4.2）
  または陰陽混合ソルバーが必要となる可能性が高い
- 19 本実機検証は status-378 では smoke test レベルにとどめ、別 status で
  mass scaling / dt subcycling の追加実装と組み合わせて行う
