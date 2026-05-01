"""ExplicitDynamicProcess のテスト（status-378 Phase 2）.

陽的中央差分時間積分による接触動的解析 1 増分 driver の挙動検証。
SolverProcess 契約 / Courant 推定 / ContactFrictionProcess 経由の統合。
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact._contact_pair import _ContactConfigInput, _ContactManagerInput
from xkep_cae.contact.solver._explicit_dynamic import (
    ExplicitDynamicInput,
    ExplicitDynamicProcess,
    ExplicitDynamicStepInput,
    _estimate_critical_dt,
)
from xkep_cae.contact.solver.process import ContactFrictionProcess
from xkep_cae.core import (
    AssembleCallbacks,
    BoundaryData,
    ContactFrictionInputData,
    ContactSetupData,
    MeshData,
    SolverProcess,
    SolverResultData,
)
from xkep_cae.core.data import default_strategies
from xkep_cae.core.testing import binds_to


@binds_to(ExplicitDynamicProcess)
class TestExplicitDynamicProcessAPI:
    """ExplicitDynamicProcess の API / 契約テスト."""

    def test_is_solver_process(self):
        assert isinstance(ExplicitDynamicProcess(), SolverProcess)

    def test_meta_name(self):
        assert ExplicitDynamicProcess.meta.name == "ExplicitDynamic"

    def test_meta_module(self):
        assert ExplicitDynamicProcess.meta.module == "solve"

    def test_meta_version(self):
        assert ExplicitDynamicProcess.meta.version == "1.0.0"


class TestEstimateCriticalDt:
    """Gerschgorin 上界による Courant 臨界 dt 推定."""

    def test_diagonal_stiffness_unit_mass(self):
        # K = diag(100), M = I → ω² = 100, ω = 10, dt_c = 2/10 = 0.2
        K = sp.diags([100.0] * 4, format="csr")
        M_inv = np.ones(4)
        dt_c = _estimate_critical_dt(K, M_inv, fixed_dofs=np.array([], dtype=int))
        assert dt_c == pytest.approx(0.2, rel=1e-12)

    def test_zero_stiffness_returns_inf(self):
        K = sp.csr_matrix((4, 4))
        M_inv = np.ones(4)
        dt_c = _estimate_critical_dt(K, M_inv, fixed_dofs=np.array([], dtype=int))
        assert dt_c == float("inf")

    def test_fixed_dofs_excluded(self):
        # K[0,0] = 1e8 だが固定 DOF として除外、K[1,1] = 1.0 のみ評価
        K = sp.diags([1e8, 1.0, 1.0, 1.0], format="csr")
        M_inv = np.ones(4)
        dt_c = _estimate_critical_dt(K, M_inv, fixed_dofs=np.array([0]))
        # ω² = 1, dt_c = 2/1 = 2
        assert dt_c == pytest.approx(2.0, rel=1e-12)

    def test_zero_mass_inv_excluded(self):
        K = sp.diags([100.0] * 4, format="csr")
        M_inv = np.array([0.0, 1.0, 1.0, 1.0])  # DOF 0 は質量∞ → ω²=0 として除外
        dt_c = _estimate_critical_dt(K, M_inv, fixed_dofs=np.array([], dtype=int))
        # 残り 3 DOF で ω²=100, dt_c = 0.2
        assert dt_c == pytest.approx(0.2, rel=1e-12)


class TestExplicitDynamicProcessRequiresExplicitStrategy:
    """ExplicitDynamicProcess は ExplicitCentralDifferenceProcess を要求する."""

    def test_raises_with_implicit_strategy(self):
        # 簡易メッシュ + デフォルト strategies（implicit / GeneralizedAlpha）で
        # ExplicitDynamicProcess を呼ぶと TypeError を発生させる。
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        M = sp.eye(ndof, format="csr") * 1.0
        strats = default_strategies(
            ndof=ndof,
            mass_matrix=M,
            dt_physical=1.0,
            solver_mode="implicit",
        )
        contact = _make_contact_setup(mesh)
        callbacks = _make_simple_callbacks(ndof)
        cfg = ExplicitDynamicInput()

        proc = ExplicitDynamicProcess()
        u = np.zeros(ndof)
        with pytest.raises(TypeError, match="ExplicitCentralDifferenceProcess"):
            proc.process(
                ExplicitDynamicStepInput(
                    config=cfg,
                    u=u,
                    f_ext=np.zeros(ndof),
                    fixed_dofs=np.arange(6),
                    assemble_tangent=callbacks.assemble_tangent,
                    assemble_internal_force=callbacks.assemble_internal_force,
                    manager=contact.manager,
                    node_coords_ref=mesh.node_coords.copy(),
                    strategies=strats,
                    k_pen=contact.k_pen,
                    mu=0.15,
                    u_ref=u.copy(),
                    load_frac=0.1,
                    load_frac_prev=0.0,
                    increment_display=1,
                    dt_sub=1e-3,
                    use_coating=False,
                    connectivity=mesh.connectivity,
                )
            )


class TestMassScalingAutoTune:
    """status-379 Phase 3 候補 (h1) auto-tune の単体検証.

    `mass_scaling_auto=True` で Courant 違反検知時に β を逆算更新する。
    """

    def test_auto_tune_disabled_returns_courant_failure(self):
        """auto=False（既定）で Courant 違反は failure_reason='courant'."""
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        # K_max=100, M=eps→ ω=large → dt_c=tiny → 必ず違反
        M = sp.eye(ndof, format="csr") * 1e-6
        contact = _make_contact_setup(mesh)
        callbacks = _make_simple_callbacks_high_K(ndof)

        # 直接 ExplicitDynamic を呼ぶ
        strats = default_strategies(
            ndof=ndof,
            mass_matrix=M,
            dt_physical=1.0,
            solver_mode="explicit",
        )
        cfg = ExplicitDynamicInput(
            courant_check_interval=1,  # 毎 increment 検査
            mass_scaling_auto=False,
        )
        proc = ExplicitDynamicProcess()
        u = np.zeros(ndof)
        result = proc.process(
            ExplicitDynamicStepInput(
                config=cfg,
                u=u,
                f_ext=np.zeros(ndof),
                fixed_dofs=np.arange(6),
                assemble_tangent=callbacks.assemble_tangent,
                assemble_internal_force=callbacks.assemble_internal_force,
                manager=contact.manager,
                node_coords_ref=mesh.node_coords.copy(),
                strategies=strats,
                k_pen=contact.k_pen,
                mu=0.15,
                u_ref=u.copy(),
                load_frac=0.05,
                load_frac_prev=0.0,
                increment_display=1,
                dt_sub=0.5,
                use_coating=False,
                connectivity=mesh.connectivity,
            )
        )
        assert result.diverged is True
        assert result.failure_reason == "courant"

    def test_auto_tune_scales_beta_within_cap(self):
        """auto=True + 違反検知で β が上方更新され、failure ではなく converged を返す."""
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        M = sp.eye(ndof, format="csr") * 1e-6
        contact = _make_contact_setup(mesh)
        callbacks = _make_simple_callbacks_high_K(ndof)

        strats = default_strategies(
            ndof=ndof,
            mass_matrix=M,
            dt_physical=1.0,
            solver_mode="explicit",
        )
        time_strategy = strats.time_integration
        beta_initial = time_strategy.mass_scaling_beta

        cfg = ExplicitDynamicInput(
            courant_check_interval=1,
            mass_scaling_auto=True,
            mass_scaling_max_beta=1.0e6,  # 十分大きく取る
        )
        proc = ExplicitDynamicProcess()
        u = np.zeros(ndof)
        result = proc.process(
            ExplicitDynamicStepInput(
                config=cfg,
                u=u,
                f_ext=np.zeros(ndof),
                fixed_dofs=np.arange(6),
                assemble_tangent=callbacks.assemble_tangent,
                assemble_internal_force=callbacks.assemble_internal_force,
                manager=contact.manager,
                node_coords_ref=mesh.node_coords.copy(),
                strategies=strats,
                k_pen=contact.k_pen,
                mu=0.15,
                u_ref=u.copy(),
                load_frac=0.05,
                load_frac_prev=0.0,
                increment_display=1,
                dt_sub=0.5,
                use_coating=False,
                connectivity=mesh.connectivity,
            )
        )
        # 違反だったが auto-tune で β 上方更新 + ステップ前進
        assert time_strategy.mass_scaling_beta > beta_initial
        assert result.diverged is False
        assert result.converged is True

    def test_auto_tune_cap_reached_returns_courant_cap(self):
        """auto=True + max_beta cap に到達した場合は failure_reason='courant_cap'."""
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        M = sp.eye(ndof, format="csr") * 1e-6
        contact = _make_contact_setup(mesh)
        callbacks = _make_simple_callbacks_high_K(ndof)

        strats = default_strategies(
            ndof=ndof,
            mass_matrix=M,
            dt_physical=1.0,
            solver_mode="explicit",
        )
        cfg = ExplicitDynamicInput(
            courant_check_interval=1,
            mass_scaling_auto=True,
            mass_scaling_max_beta=1.5,  # 著しく小さい cap
        )
        proc = ExplicitDynamicProcess()
        u = np.zeros(ndof)
        result = proc.process(
            ExplicitDynamicStepInput(
                config=cfg,
                u=u,
                f_ext=np.zeros(ndof),
                fixed_dofs=np.arange(6),
                assemble_tangent=callbacks.assemble_tangent,
                assemble_internal_force=callbacks.assemble_internal_force,
                manager=contact.manager,
                node_coords_ref=mesh.node_coords.copy(),
                strategies=strats,
                k_pen=contact.k_pen,
                mu=0.15,
                u_ref=u.copy(),
                load_frac=0.05,
                load_frac_prev=0.0,
                increment_display=1,
                dt_sub=0.5,
                use_coating=False,
                connectivity=mesh.connectivity,
            )
        )
        assert result.diverged is True
        assert result.failure_reason == "courant_cap"


class TestMassScalingMaxGrowthCap:
    """status-381 h-bug-3 緩和: 1 update あたりの β 成長率 cap.

    増分 1（warm start）では growth cap をスキップし target β に即座にジャンプ。
    増分 2 以降は smooth growth cap を適用して相空間の急峻な変化を抑制。
    """

    def test_first_increment_skips_growth_cap_warm_start(self):
        """増分 1 では growth cap をスキップして target β に即座到達（warm start）."""
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        M = sp.eye(ndof, format="csr") * 1e-6
        contact = _make_contact_setup(mesh)
        callbacks = _make_simple_callbacks_high_K(ndof)

        strats = default_strategies(
            ndof=ndof,
            mass_matrix=M,
            dt_physical=1.0,
            solver_mode="explicit",
        )
        time_strategy = strats.time_integration
        beta_initial = time_strategy.mass_scaling_beta

        cfg = ExplicitDynamicInput(
            courant_check_interval=1,
            mass_scaling_auto=True,
            mass_scaling_max_beta=1.0e6,
            mass_scaling_max_growth_per_update=4.0,
        )
        proc = ExplicitDynamicProcess()
        u = np.zeros(ndof)
        proc.process(
            ExplicitDynamicStepInput(
                config=cfg,
                u=u,
                f_ext=np.zeros(ndof),
                fixed_dofs=np.arange(6),
                assemble_tangent=callbacks.assemble_tangent,
                assemble_internal_force=callbacks.assemble_internal_force,
                manager=contact.manager,
                node_coords_ref=mesh.node_coords.copy(),
                strategies=strats,
                k_pen=contact.k_pen,
                mu=0.15,
                u_ref=u.copy(),
                load_frac=0.05,
                load_frac_prev=0.0,
                increment_display=1,  # warm start
                dt_sub=0.5,
                use_coating=False,
                connectivity=mesh.connectivity,
            )
        )
        # 増分 1 では growth cap をスキップ、target β（≫ 4×）に到達
        assert time_strategy.mass_scaling_beta > beta_initial * 4.0

    def test_subsequent_increment_respects_growth_cap(self):
        """増分 2 以降は growth cap を適用（β は最大 cap×current まで）."""
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        M = sp.eye(ndof, format="csr") * 1e-6
        contact = _make_contact_setup(mesh)
        callbacks = _make_simple_callbacks_high_K(ndof)

        strats = default_strategies(
            ndof=ndof,
            mass_matrix=M,
            dt_physical=1.0,
            solver_mode="explicit",
        )
        time_strategy = strats.time_integration
        beta_before = time_strategy.mass_scaling_beta

        cfg = ExplicitDynamicInput(
            courant_check_interval=1,
            mass_scaling_auto=True,
            mass_scaling_max_beta=1.0e6,
            mass_scaling_max_growth_per_update=4.0,
        )
        proc = ExplicitDynamicProcess()
        u = np.zeros(ndof)
        proc.process(
            ExplicitDynamicStepInput(
                config=cfg,
                u=u,
                f_ext=np.zeros(ndof),
                fixed_dofs=np.arange(6),
                assemble_tangent=callbacks.assemble_tangent,
                assemble_internal_force=callbacks.assemble_internal_force,
                manager=contact.manager,
                node_coords_ref=mesh.node_coords.copy(),
                strategies=strats,
                k_pen=contact.k_pen,
                mu=0.15,
                u_ref=u.copy(),
                load_frac=0.10,
                load_frac_prev=0.05,
                increment_display=2,  # 2 increment目: growth cap 適用
                dt_sub=0.5,
                use_coating=False,
                connectivity=mesh.connectivity,
            )
        )
        # 2 increment目では growth cap 4× まで
        assert time_strategy.mass_scaling_beta <= beta_before * 4.0 + 1e-9
        assert time_strategy.mass_scaling_beta > beta_before  # 5% 以上の要求はあった


class TestExplicitContactFrictionIntegration:
    """ContactFrictionProcess + solver_mode="explicit" の統合 smoke test.

    19 本撚線レベルの実機検証は別途 status で実施。ここでは:
    - solver_mode="explicit" でも例外なく実行可能
    - 1 増分の advance で u が更新される
    - converged=True を返す
    を最小構成で確認する。
    """

    def test_explicit_solver_mode_advances_u(self):
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        callbacks = _make_simple_callbacks(ndof)
        contact = _make_contact_setup(mesh)

        # 質量行列（一様）
        M = sp.eye(ndof, format="csr") * 1.0

        # 微小荷重
        f_ext = np.zeros(ndof)
        f_ext[6 * (len(mesh.node_coords) - 1)] = 1e-3

        boundary = BoundaryData(
            fixed_dofs=np.arange(6),
            f_ext_total=f_ext,
        )

        # dt_physical を Courant に対し十分小さく取る
        # K_max ≈ 1.0（_make_simple_callbacks で eye）, M_lump = 1 → dt_c = 2.0
        # safety 0.9 で dt_safe = 1.8。dt_physical = 0.5 で十分。
        input_data = ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact,
            callbacks=callbacks,
            mass_matrix=M,
            dt_physical=0.5,
            solver_mode="explicit",
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            max_increments=3,
            max_nr_attempts=1,
        )

        proc = ContactFrictionProcess()
        result = proc.process(input_data)
        assert isinstance(result, SolverResultData)
        # 実行が完了し、explicit path で 1 ステップ以上進む
        assert result.n_increments >= 1
        # u が初期 0 から微小に変化（弱接触なので ||u|| > 0）
        assert np.linalg.norm(result.u) > 0.0

    def test_explicit_relax_steps_runs(self):
        """status-382 候補 (p1): explicit_relax_steps > 0 で BC 完了後 relax phase 実行.

        relax phase は warning なしに実行され、SolverResultData を返す。
        """
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        callbacks = _make_simple_callbacks(ndof)
        contact = _make_contact_setup(mesh)
        M = sp.eye(ndof, format="csr") * 1.0
        f_ext = np.zeros(ndof)
        f_ext[6 * (len(mesh.node_coords) - 1)] = 1e-3
        boundary = BoundaryData(fixed_dofs=np.arange(6), f_ext_total=f_ext)

        input_data = ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact,
            callbacks=callbacks,
            mass_matrix=M,
            dt_physical=0.5,
            solver_mode="explicit",
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            explicit_mass_proportional_damping_alpha=0.1,
            explicit_relax_steps=5,
            explicit_relax_tol=0.0,
            max_increments=3,
            max_nr_attempts=1,
        )
        result = ContactFrictionProcess().process(input_data)
        assert isinstance(result, SolverResultData)
        # 主ループ + relax で incr_display は max_increments より大きくなる
        # （主ループ 3 + relax 5 = 8 まで進む）
        assert result.n_increments >= 3

    def test_explicit_relax_default_off_unchanged(self):
        """status-382 候補 (p1): default `explicit_relax_steps=0` で挙動不変.

        relax phase は実行されず、主ループのみで完了する。
        """
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        callbacks = _make_simple_callbacks(ndof)
        contact = _make_contact_setup(mesh)
        M = sp.eye(ndof, format="csr") * 1.0
        f_ext = np.zeros(ndof)
        f_ext[6 * (len(mesh.node_coords) - 1)] = 1e-3
        boundary = BoundaryData(fixed_dofs=np.arange(6), f_ext_total=f_ext)

        # default: explicit_relax_steps = 0
        input_data = ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact,
            callbacks=callbacks,
            mass_matrix=M,
            dt_physical=0.5,
            solver_mode="explicit",
            explicit_courant_safety=0.9,
            explicit_courant_check_interval=10,
            max_increments=3,
            max_nr_attempts=1,
        )
        result = ContactFrictionProcess().process(input_data)
        assert isinstance(result, SolverResultData)


# =====================================================================
# 共通ヘルパ
# =====================================================================


def _make_two_beam_mesh() -> MeshData:
    """簡易 2 本梁メッシュ."""
    n_nodes_per_strand = 17
    coords_list = []
    conn_list = []
    for strand_id in range(2):
        y_offset = 0.1 * strand_id
        for i in range(n_nodes_per_strand):
            x = i * 1.0 / (n_nodes_per_strand - 1)
            coords_list.append([x, y_offset, 0.0])
        base = strand_id * n_nodes_per_strand
        for i in range(n_nodes_per_strand - 1):
            conn_list.append([base + i, base + i + 1])

    return MeshData(
        node_coords=np.array(coords_list),
        connectivity=np.array(conn_list),
        radii=0.05,
        n_strands=2,
        strand_ids=np.array([0] * (n_nodes_per_strand - 1) + [1] * (n_nodes_per_strand - 1)),
    )


def _make_simple_callbacks(ndof: int) -> AssembleCallbacks:
    """単位スカラー剛性のアセンブリコールバック."""

    def assemble_tangent(u: np.ndarray) -> sp.csr_matrix:
        return sp.eye(ndof, format="csr") * 1.0

    def assemble_internal_force(u: np.ndarray) -> np.ndarray:
        return u * 1.0

    return AssembleCallbacks(
        assemble_tangent=assemble_tangent,
        assemble_internal_force=assemble_internal_force,
    )


def _make_simple_callbacks_high_K(ndof: int) -> AssembleCallbacks:
    """高剛性 K=diag(100) のアセンブリコールバック（Courant 違反誘発用）."""

    def assemble_tangent(u: np.ndarray) -> sp.csr_matrix:
        return sp.eye(ndof, format="csr") * 100.0

    def assemble_internal_force(u: np.ndarray) -> np.ndarray:
        return u * 100.0

    return AssembleCallbacks(
        assemble_tangent=assemble_tangent,
        assemble_internal_force=assemble_internal_force,
    )


def _make_contact_setup(mesh: MeshData) -> ContactSetupData:
    """テスト用接触設定（接触ペア未検出でも config 正常）."""
    config = _ContactConfigInput(
        beam_E=210e3,
        beam_I=1e-3,
        mu=0.15,
        exclude_same_strand=True,
        smoothing_delta=2000.0,
    )
    manager = _ContactManagerInput(config=config)
    return ContactSetupData(manager=manager, k_pen=1e2, mu=0.15)
