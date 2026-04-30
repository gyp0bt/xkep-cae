"""陽的中央差分時間積分による接触動的解析 1 増分 driver（status-378 Phase 2）.

`ExplicitCentralDifferenceProcess` を `ContactFrictionProcess` のメインループから
増分単位で駆動する。NR 反復は経由せず、各増分で 1 回だけ:

1. `ContactForceAssemblyProcess` で f_int + f_c を組み立てる
2. `ExplicitCentralDifferenceProcess.step()` で u_{n+1} を 1 ステップ前進
3. Courant 安定条件を確認（任意で adaptive Δt 縮小を要求）

を実行する。`NewtonDynamicProcess` と排他で、`ContactFrictionInputData.solver_mode
== "explicit"` のとき呼ばれる。

設計仕様:
- xkep_cae/time_integration/docs/time_integration_explicit.md §Phase 2
- docs/status/status-378.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.solver._diagnostics import (
    ConvergenceDiagnosticsOutput,
)
from xkep_cae.contact.solver._newton_dynamic import DynamicStepOutput
from xkep_cae.contact.solver._newton_steps import (
    ContactForceAssemblyInput,
    ContactForceAssemblyProcess,
)
from xkep_cae.core import ProcessMeta, SolverProcess
from xkep_cae.time_integration.strategy import ExplicitCentralDifferenceProcess


@dataclass(frozen=True)
class ExplicitDynamicInput:
    """陽解法 1 増分 driver の設定."""

    show_progress: bool = True
    ndof_per_node: int = 6
    courant_safety: float = 0.9
    courant_check_interval: int = 50
    max_dt_shrink_factor: float = 0.25
    # 質量スケーリング auto-tune（status-379 Phase 3 候補 (h1)）.
    # True で Courant 違反検知時に β を逆算し
    # `time_strategy.set_mass_scaling_beta()` で上方更新する。
    # `mass_scaling_max_beta` で上限を cap、超過時は failure_reason="courant".
    mass_scaling_auto: bool = False
    mass_scaling_max_beta: float = 100.0
    # 準静的近似ガード: E_kin / E_strain がこれを超えると警告出力（0.0 で無効）
    kinetic_energy_budget_ratio: float = 0.0


@dataclass(frozen=True)
class ExplicitDynamicStepInput:
    """陽解法 1 増分の入力."""

    config: ExplicitDynamicInput
    u: np.ndarray
    f_ext: np.ndarray
    fixed_dofs: np.ndarray
    assemble_tangent: object
    assemble_internal_force: object
    manager: object
    node_coords_ref: np.ndarray
    strategies: object
    k_pen: float
    mu: float
    u_ref: np.ndarray
    load_frac: float
    load_frac_prev: float
    increment_display: int
    dt_sub: float
    use_coating: bool
    connectivity: np.ndarray | None = None


def _estimate_critical_dt(
    K: sp.csr_matrix | sp.spmatrix,
    M_lump_inv: np.ndarray,
    fixed_dofs: np.ndarray,
) -> float:
    """剛性行列の最大固有値から Courant 臨界時間刻み Δt_c = 2/ω_max を推定する.

    sparse Gerschgorin 上界（行絶対値和）に対角質量を掛けた値を ω_max² の上界として用いる。
    `eigsh` を使うより遥かに高速で、過大評価側に偏るため安全側になる。固定 DOF は推定から除外。

    Args:
        K: 接線剛性行列（CSR）。
        M_lump_inv: 集中質量逆数 M_lump^{-1}（対角）。
        fixed_dofs: 固定 DOF（ω 推定から除外）。

    Returns:
        Δt_c。K が空、M_lump_inv が全ゼロ、または推定不能の場合は inf。
    """
    if K is None or K.shape[0] == 0:
        return float("inf")
    K_csr = K.tocsr() if not sp.isspmatrix_csr(K) else K
    abs_K = abs(K_csr)
    row_sum = np.asarray(abs_K.sum(axis=1)).flatten()
    if fixed_dofs is not None and len(fixed_dofs) > 0:
        mask = np.ones_like(row_sum, dtype=bool)
        mask[fixed_dofs] = False
    else:
        mask = np.ones_like(row_sum, dtype=bool)
    omega_sq = row_sum * np.abs(M_lump_inv)
    omega_sq_eff = omega_sq[mask & (M_lump_inv > 0.0)]
    if omega_sq_eff.size == 0:
        return float("inf")
    omega_sq_max = float(omega_sq_eff.max())
    if omega_sq_max <= 0.0:
        return float("inf")
    omega_max = np.sqrt(omega_sq_max)
    return 2.0 / omega_max


class ExplicitDynamicProcess(
    SolverProcess[ExplicitDynamicStepInput, DynamicStepOutput],
):
    """陽的中央差分による 1 増分 driver（status-378 Phase 2、候補 (h)）.

    `NewtonDynamicProcess` と排他。陰解法 NR escape hatch アプローチ
    （候補 (g) 全候補が gate frac=0.6 未達、status-376）の代替として、
    陽解法時間積分により 19 本以上の K_c x/z カップリング不整合
    （status-344）を時間積分自体で安定化する目的で導入。

    1 増分の挙動:
    1. ContactForceAssemblyProcess で f_int + f_c を組み立て
    2. ExplicitCentralDifferenceProcess.step() で u_{n+1} を 1 ステップ前進
    3. Courant 監視: dt_sub > 0.9·Δt_c なら failure_reason="courant" で
       diverged=True を返し、上位 stepping にカットバックを要求

    NR 反復・接線剛性アセンブリは行わない（Courant 監視時のみアセンブル）。

    設計仕様: xkep_cae/time_integration/docs/time_integration_explicit.md §Phase 2
    """

    meta = ProcessMeta(
        name="ExplicitDynamic",
        module="solve",
        version="1.0.0",
        document_path="docs/explicit_dynamic.md",
    )
    uses = [
        ContactForceAssemblyProcess,
        ExplicitCentralDifferenceProcess,
    ]

    def process(self, input_data: ExplicitDynamicStepInput) -> DynamicStepOutput:
        cfg = input_data.config
        u = input_data.u
        f_ext = input_data.f_ext
        manager = input_data.manager
        strategies = input_data.strategies
        k_pen = input_data.k_pen
        mu = input_data.mu
        load_frac = input_data.load_frac
        increment_display = input_data.increment_display
        dt_sub = input_data.dt_sub

        time_strategy = strategies.time_integration
        if not isinstance(time_strategy, ExplicitCentralDifferenceProcess):
            raise TypeError(
                "ExplicitDynamicProcess requires "
                "ExplicitCentralDifferenceProcess as time_integration strategy. "
                f"got {type(time_strategy).__name__}"
            )
        contact_force_strategy = strategies.contact_force
        friction_strategy = strategies.friction
        coating_strategy = strategies.coating

        diag = ConvergenceDiagnosticsOutput(step=increment_display, load_frac=load_frac)

        # ── 1. 接触力 + 内力 + 残差を組み立て ──
        force_proc = ContactForceAssemblyProcess()
        force_out = force_proc.process(
            ContactForceAssemblyInput(
                u=u,
                f_ext=f_ext,
                fixed_dofs=input_data.fixed_dofs,
                manager=manager,
                node_coords_ref=input_data.node_coords_ref,
                contact_force_strategy=contact_force_strategy,
                friction_strategy=friction_strategy,
                coating_strategy=coating_strategy,
                k_pen=k_pen,
                mu=mu,
                u_ref=input_data.u_ref,
                load_frac=load_frac,
                load_frac_prev=input_data.load_frac_prev,
                increment_display=increment_display,
                ndof_per_node=cfg.ndof_per_node,
                use_coating=input_data.use_coating,
                assemble_internal_force=input_data.assemble_internal_force,
                connectivity=input_data.connectivity,
            )
        )
        f_c = force_out.f_c
        # ContactForceAssemblyProcess の R_u は f_int + f_c - f_ext を返す。
        # 陽解法では a = M^{-1} (f_ext - f_int - f_c) なので R_u 反転で慣性外力を得る。
        R_u_explicit = -force_out.R_u
        R_u_explicit[input_data.fixed_dofs] = 0.0

        # 接触力ノルムを active 数の代理として診断に追加
        n_active = sum(1 for p in manager.pairs if hasattr(p, "state") and p.state.p_n > 0.0)
        diag.n_active_history.append(n_active)
        diag.res_history.append(float(np.linalg.norm(R_u_explicit)))

        # ── 2. Courant 監視（cfg.courant_check_interval 増分ごと） ──
        # status-379 Phase 3: mass_scaling_auto=True のとき、Courant 違反検知時に
        # β = √((dt_sub / (cfg.courant_safety · dt_c_raw))²) を逆算して
        # set_mass_scaling_beta() で時間積分子の集中質量を倍化（dt_c → β·dt_c）。
        # max_beta を超える要求は cap し、cap でも追いつかない場合のみ cutback。
        courant_violated = False
        courant_failure_reason = "courant"
        if cfg.courant_check_interval > 0 and increment_display % cfg.courant_check_interval == 0:
            try:
                K = input_data.assemble_tangent(u)
            except Exception:
                K = None
            if K is not None:
                dt_c = _estimate_critical_dt(K, time_strategy.M_lump_inv, input_data.fixed_dofs)
                dt_safe = cfg.courant_safety * dt_c
                if dt_sub > dt_safe and dt_safe > 0.0 and np.isfinite(dt_safe):
                    if cfg.mass_scaling_auto:
                        # β · dt_c_raw >= dt_sub / safety を満たす β を決定。
                        # 現 β に対し dt_safe = β · dt_safe_raw_per_beta なので、
                        # 必要な追加倍率 r = dt_sub / dt_safe を現 β に乗じる。
                        current_beta = time_strategy.mass_scaling_beta
                        required_extra = dt_sub / dt_safe
                        target_beta = current_beta * required_extra
                        capped_beta = min(target_beta, cfg.mass_scaling_max_beta)
                        # 5% 以上の成長要求のみ実適用（数値ノイズ抑制）
                        if capped_beta > current_beta * 1.05:
                            time_strategy.set_mass_scaling_beta(capped_beta)
                            if cfg.show_progress:
                                print(
                                    f"  [MASS_SCALE] Incr {increment_display} "
                                    f"β: {current_beta:.3e} → {capped_beta:.3e} "
                                    f"(target {target_beta:.3e}, cap "
                                    f"{cfg.mass_scaling_max_beta:.3e})"
                                )
                        if target_beta > cfg.mass_scaling_max_beta:
                            courant_violated = True
                            courant_failure_reason = "courant_cap"
                            if cfg.show_progress:
                                print(
                                    f"  [COURANT] Incr {increment_display} "
                                    f"β cap reached (target={target_beta:.3e} > "
                                    f"{cfg.mass_scaling_max_beta:.3e}) → cutback"
                                )
                    else:
                        courant_violated = True
                        if cfg.show_progress:
                            print(
                                f"  [COURANT] Incr {increment_display} "
                                f"dt_sub={dt_sub:.3e} > "
                                f"{cfg.courant_safety:.2f}·dt_c={dt_safe:.3e} "
                                f"→ cutback request"
                            )

        if courant_violated:
            return DynamicStepOutput(
                converged=False,
                n_attempts=1,
                n_active=n_active,
                f_c=f_c,
                diagnostics=diag,
                diverged=True,
                f_ref_used=float(np.linalg.norm(f_ext)),
                convergence_type="",
                failure_reason=courant_failure_reason,
                damping_energy_rate=0.0,
            )

        # ── 3. 陽的中央差分 1 ステップ ──
        # f_int_effective = f_int + f_c（接触含む内力）。residual = f_ext - f_int_effective。
        # ContactForceAssemblyProcess の f_int は assemble_internal_force(u) なので、
        # f_int_effective を再構築するため R_u_explicit から逆算する:
        #   f_int_effective = f_ext - R_u_explicit
        f_int_effective = f_ext - R_u_explicit

        u_new = time_strategy.step(
            u=u,
            f_ext=f_ext,
            f_int=f_int_effective,
            dt=dt_sub,
            fixed_dofs=input_data.fixed_dofs,
        )
        # in-place 更新（NR と同じ呼び出し規約）
        u[:] = u_new

        return DynamicStepOutput(
            converged=True,
            n_attempts=1,
            n_active=n_active,
            f_c=f_c,
            diagnostics=diag,
            diverged=False,
            f_ref_used=float(np.linalg.norm(f_ext)),
            convergence_type="explicit",
            failure_reason="",
            damping_energy_rate=0.0,
        )
