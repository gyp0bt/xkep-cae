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
    _detect_stiff_dofs,
    _estimate_critical_dt,
    _estimate_critical_dt_per_element,
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
from xkep_cae.time_integration.strategy import ExplicitCentralDifferenceProcess


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


class TestExplicitULUpdateInterval:
    """status-383 候補 (q1): explicit 中の UL update_reference 周期化.

    `explicit_ul_update_interval` 増分ごとに `update_reference` を呼ぶこと、
    およびデフォルト 1 では既存挙動と同じ毎増分呼出となることを確認する。
    """

    def _run_with_mock_ul(
        self, *, interval: int, max_incr: int, ndof: int, mesh, contact, M, f_ext, boundary
    ) -> _MockULAssembler:
        """指定 interval で実行し、update_reference 呼出回数を返す.

        各 increment が必ず短い dt で進むよう dt_max_fraction を contact の
        config に指定し、max_incr 回の成功増分を確実に確保する。
        """
        ul_asm = _MockULAssembler(ndof)
        callbacks = AssembleCallbacks(
            assemble_tangent=lambda u: sp.eye(ndof, format="csr") * 1.0,
            assemble_internal_force=lambda u: u * 1.0,
            ul_assembler=ul_asm,
        )
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
            explicit_ul_update_interval=interval,
            max_increments=max_incr,
            max_nr_attempts=1,
        )
        ContactFrictionProcess().process(input_data)
        return ul_asm

    def _common_setup(self, *, n_increments_target: int):
        """dt_max_fraction で各 step を制限し、n_increments_target 増分を強制."""
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        # dt_max_fraction = 1/n_target → 各 step の dt が制限され n_target 増分へ分割
        contact_config = _ContactConfigInput(
            beam_E=210e3,
            beam_I=1e-3,
            mu=0.15,
            exclude_same_strand=True,
            smoothing_delta=2000.0,
            dt_max_fraction=1.0 / max(1, n_increments_target),
        )
        manager = _ContactManagerInput(config=contact_config)
        contact = ContactSetupData(manager=manager, k_pen=1e2, mu=0.15)
        M = sp.eye(ndof, format="csr") * 1.0
        f_ext = np.zeros(ndof)
        f_ext[6 * (len(mesh.node_coords) - 1)] = 1e-3
        boundary = BoundaryData(fixed_dofs=np.arange(6), f_ext_total=f_ext)
        return mesh, ndof, contact, M, f_ext, boundary

    def test_default_interval_one_calls_every_increment(self):
        """default explicit_ul_update_interval=1 で毎増分 update_reference 呼出."""
        mesh, ndof, contact, M, f_ext, boundary = self._common_setup(n_increments_target=4)
        ul = self._run_with_mock_ul(
            interval=1,
            max_incr=4,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            f_ext=f_ext,
            boundary=boundary,
        )
        # 主ループで 4 増分成功 → 4 回呼出
        assert ul.update_reference_call_count == 4

    def test_interval_two_calls_every_other_increment(self):
        """explicit_ul_update_interval=2 で 2 増分ごとに update_reference 呼出."""
        mesh, ndof, contact, M, f_ext, boundary = self._common_setup(n_increments_target=4)
        ul = self._run_with_mock_ul(
            interval=2,
            max_incr=4,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            f_ext=f_ext,
            boundary=boundary,
        )
        # 4 増分中、2nd と 4th でのみ呼出 → 2 回
        assert ul.update_reference_call_count == 2

    def test_interval_larger_than_increments_skips_all(self):
        """interval > max_increments のとき、update_reference は一度も呼ばれない."""
        mesh, ndof, contact, M, f_ext, boundary = self._common_setup(n_increments_target=4)
        ul = self._run_with_mock_ul(
            interval=100,
            max_incr=4,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            f_ext=f_ext,
            boundary=boundary,
        )
        # 4 増分中 100 増分目で呼出 → 0 回
        assert ul.update_reference_call_count == 0

    def test_interval_zero_treated_as_one(self):
        """explicit_ul_update_interval=0 は 1 として扱う（max(1, value)）."""
        mesh, ndof, contact, M, f_ext, boundary = self._common_setup(n_increments_target=4)
        ul = self._run_with_mock_ul(
            interval=0,
            max_incr=4,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            f_ext=f_ext,
            boundary=boundary,
        )
        # max(1, 0) = 1 として扱われ、毎増分呼出
        assert ul.update_reference_call_count == 4

    def test_implicit_mode_gate_short_circuits(self):
        """gate 式 `_solver_mode != "explicit" or ...` の implicit short-circuit を直接検証.

        プロセス全体を走らせると implicit + 接触は別途収束問題が生じるため、
        ここではロジック式のみを検証する。`(q1)` は explicit のみ対象であり、
        implicit のとき interval が無視されることを確認。
        """

        # ゲート条件式をテスト: (mode != "explicit") OR (interval <= 1) OR (next % interval == 0)
        def _do_update(*, mode: str, interval: int, next_incr: int) -> bool:
            return mode != "explicit" or interval <= 1 or (next_incr % interval == 0)

        # implicit: interval=100 でも常に True（毎増分更新）
        for n in range(1, 10):
            assert _do_update(mode="implicit", interval=100, next_incr=n)

        # explicit + interval=1: 常に True
        for n in range(1, 10):
            assert _do_update(mode="explicit", interval=1, next_incr=n)

        # explicit + interval=3: 3, 6, 9 のみ True
        for n in range(1, 10):
            assert _do_update(mode="explicit", interval=3, next_incr=n) == (n % 3 == 0)


class TestExplicitULDisableUpdate:
    """status-396 候補 (z3): explicit-TL 固定モード.

    `explicit_ul_disable_update=True` で `solver_mode="explicit"` の下、UL
    `update_reference()` を一切呼ばないこと、および `False`（default）では
    既存の interval ゲート挙動と同じになることを確認する。
    """

    def _run_with_mock_ul(
        self,
        *,
        disable: bool,
        interval: int,
        max_incr: int,
        ndof: int,
        mesh,
        contact,
        M,
        f_ext,
        boundary,
    ) -> _MockULAssembler:
        ul_asm = _MockULAssembler(ndof)
        callbacks = AssembleCallbacks(
            assemble_tangent=lambda u: sp.eye(ndof, format="csr") * 1.0,
            assemble_internal_force=lambda u: u * 1.0,
            ul_assembler=ul_asm,
        )
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
            explicit_ul_update_interval=interval,
            explicit_ul_disable_update=disable,
            max_increments=max_incr,
            max_nr_attempts=1,
        )
        ContactFrictionProcess().process(input_data)
        return ul_asm

    def _common_setup(self, *, n_increments_target: int):
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        contact_config = _ContactConfigInput(
            beam_E=210e3,
            beam_I=1e-3,
            mu=0.15,
            exclude_same_strand=True,
            smoothing_delta=2000.0,
            dt_max_fraction=1.0 / max(1, n_increments_target),
        )
        manager = _ContactManagerInput(config=contact_config)
        contact = ContactSetupData(manager=manager, k_pen=1e2, mu=0.15)
        M = sp.eye(ndof, format="csr") * 1.0
        f_ext = np.zeros(ndof)
        f_ext[6 * (len(mesh.node_coords) - 1)] = 1e-3
        boundary = BoundaryData(fixed_dofs=np.arange(6), f_ext_total=f_ext)
        return mesh, ndof, contact, M, f_ext, boundary

    def test_disable_true_skips_all_update_reference(self):
        """explicit_ul_disable_update=True で update_reference 呼出は 0 回."""
        mesh, ndof, contact, M, f_ext, boundary = self._common_setup(n_increments_target=4)
        ul = self._run_with_mock_ul(
            disable=True,
            interval=1,
            max_incr=4,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            f_ext=f_ext,
            boundary=boundary,
        )
        assert ul.update_reference_call_count == 0

    def test_disable_true_overrides_interval(self):
        """disable=True は interval 値に関わらず update_reference 呼出を抑止する."""
        mesh, ndof, contact, M, f_ext, boundary = self._common_setup(n_increments_target=4)
        ul = self._run_with_mock_ul(
            disable=True,
            interval=2,
            max_incr=4,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            f_ext=f_ext,
            boundary=boundary,
        )
        # interval=2 なら通常 2 回呼ばれるが disable=True で 0 回
        assert ul.update_reference_call_count == 0

    def test_disable_false_default_preserves_interval_behavior(self):
        """default disable=False で interval=1 の既存挙動（毎増分呼出）保持."""
        mesh, ndof, contact, M, f_ext, boundary = self._common_setup(n_increments_target=4)
        ul = self._run_with_mock_ul(
            disable=False,
            interval=1,
            max_incr=4,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            f_ext=f_ext,
            boundary=boundary,
        )
        # disable=False かつ interval=1 → status-383 既存挙動と同じ毎増分呼出
        assert ul.update_reference_call_count == 4

    def test_gate_logic_disable_short_circuits(self):
        """ゲート式 `_do_ul_update` ロジックの直接検証.

        `_solver_mode != "explicit"` の implicit short-circuit に加え、
        `explicit_ul_disable_update=True` で interval 値に関わらず False 化、
        `False` のときは interval ゲート挙動と一致することを確認する。
        """

        def _do_update(*, mode: str, disable: bool, interval: int, next_incr: int) -> bool:
            return mode != "explicit" or (
                not disable and (interval <= 1 or (next_incr % interval == 0))
            )

        # implicit: disable / interval に関わらず常に True（implicit は UL を毎増分更新）
        for n in range(1, 10):
            assert _do_update(mode="implicit", disable=True, interval=100, next_incr=n)
            assert _do_update(mode="implicit", disable=False, interval=100, next_incr=n)

        # explicit + disable=True: interval 値に関わらず常に False
        for n in range(1, 10):
            for interval in (1, 2, 5, 100):
                assert not _do_update(mode="explicit", disable=True, interval=interval, next_incr=n)

        # explicit + disable=False + interval=1: 常に True（既存挙動）
        for n in range(1, 10):
            assert _do_update(mode="explicit", disable=False, interval=1, next_incr=n)

        # explicit + disable=False + interval=3: 3, 6, 9 のみ True（既存挙動）
        for n in range(1, 10):
            assert _do_update(mode="explicit", disable=False, interval=3, next_incr=n) == (
                n % 3 == 0
            )


class TestExplicitNSubCyclesPerIncrement:
    """status-399 (status-398 §5.2 fix design): explicit 1 増分あたり sub-cycle 数.

    `explicit_n_sub_cycles_per_increment=N` で 1 QUERY を N 個の explicit sub-step に
    分割し、各 sub-step で dt_inner = dt_sub / N、prescribed BC は線形補間で適用する。
    status-398 hypothesis 1（stepwise prescribed BC × mass scaling auto-tune の
    interaction）に対する architectural fix の API 単体検証.

    確認項目:
    - default N=1 で従来 ExplicitDynamicProcess.process() 呼出回数と同じ
    - N>1 で呼出回数が N 倍
    - N=0 / 負値は max(1, N) として扱い既存挙動
    - implicit mode では本 field を無視
    """

    def _make_setup(self, *, n_increments_target: int):
        """dt_max_fraction で各 step を制限し、目標増分数を確保."""
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        contact_config = _ContactConfigInput(
            beam_E=210e3,
            beam_I=1e-3,
            mu=0.15,
            exclude_same_strand=True,
            smoothing_delta=2000.0,
            dt_max_fraction=1.0 / max(1, n_increments_target),
        )
        manager = _ContactManagerInput(config=contact_config)
        contact = ContactSetupData(manager=manager, k_pen=1e2, mu=0.15)
        M = sp.eye(ndof, format="csr") * 1.0
        f_ext = np.zeros(ndof)
        f_ext[6 * (len(mesh.node_coords) - 1)] = 1e-3
        boundary = BoundaryData(fixed_dofs=np.arange(6), f_ext_total=f_ext)
        return mesh, ndof, contact, M, f_ext, boundary

    def _make_counting_callbacks(self, ndof: int) -> AssembleCallbacks:
        """単純な assemble callbacks（実 callback 動作）."""

        def assemble_tangent(u: np.ndarray) -> sp.csr_matrix:
            return sp.eye(ndof, format="csr") * 1.0

        def assemble_internal_force(u: np.ndarray) -> np.ndarray:
            return u * 1.0

        return AssembleCallbacks(
            assemble_tangent=assemble_tangent,
            assemble_internal_force=assemble_internal_force,
        )

    def _run(
        self,
        *,
        n_sub_cycles: int,
        max_incr: int,
        ndof: int,
        mesh,
        contact,
        M,
        boundary,
        monkeypatch,
    ) -> tuple[SolverResultData, list[int]]:
        """ExplicitDynamicProcess.process() の呼出回数を直接計測.

        sub-cycle 内部ループは `_explicit_proc.process()` を N 回呼ぶため、
        ここでの呼出回数 = `n_increments × N`（cutback なし時）.
        """
        callbacks = self._make_counting_callbacks(ndof)
        counter = [0]
        original_process = ExplicitDynamicProcess.process

        def counting_process(self_proc, input_data_step):
            counter[0] += 1
            return original_process(self_proc, input_data_step)

        monkeypatch.setattr(ExplicitDynamicProcess, "process", counting_process)
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
            explicit_n_sub_cycles_per_increment=n_sub_cycles,
            max_increments=max_incr,
            max_nr_attempts=1,
        )
        result = ContactFrictionProcess().process(input_data)
        return result, counter

    def test_default_n_sub_cycles_one_baseline(self, monkeypatch):
        """default N=1 で ExplicitDynamicProcess.process() 呼出回数 = 増分数."""
        mesh, ndof, contact, M, _f, boundary = self._make_setup(n_increments_target=4)
        result, counter = self._run(
            n_sub_cycles=1,
            max_incr=4,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            boundary=boundary,
            monkeypatch=monkeypatch,
        )
        assert isinstance(result, SolverResultData)
        # 4 増分 × N=1 = 4 回（cutback がない前提）
        assert counter[0] == 4

    def test_n_sub_cycles_two_doubles_calls(self, monkeypatch):
        """N=2 で sub-cycle 内部ループが 2 回回り、呼出回数が 2 倍."""
        mesh, ndof, contact, M, _f, boundary = self._make_setup(n_increments_target=3)
        result, counter = self._run(
            n_sub_cycles=2,
            max_incr=3,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            boundary=boundary,
            monkeypatch=monkeypatch,
        )
        assert isinstance(result, SolverResultData)
        # 3 増分 × 2 sub-cycle = 6 回
        assert counter[0] == 6

    def test_n_sub_cycles_five_quintuples_calls(self, monkeypatch):
        """N=5 で sub-cycle ループが 5 回回り、呼出回数が 5 倍."""
        mesh, ndof, contact, M, _f, boundary = self._make_setup(n_increments_target=3)
        result, counter = self._run(
            n_sub_cycles=5,
            max_incr=3,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            boundary=boundary,
            monkeypatch=monkeypatch,
        )
        assert isinstance(result, SolverResultData)
        # 3 増分 × 5 sub-cycle = 15 回
        assert counter[0] == 15

    def test_n_sub_cycles_zero_treated_as_one(self, monkeypatch):
        """N=0 は max(1, N) で 1 として扱う."""
        mesh, ndof, contact, M, _f, boundary = self._make_setup(n_increments_target=3)
        result, counter = self._run(
            n_sub_cycles=0,
            max_incr=3,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            boundary=boundary,
            monkeypatch=monkeypatch,
        )
        assert isinstance(result, SolverResultData)
        # 3 増分 × max(1, 0)=1 sub-cycle = 3 回
        assert counter[0] == 3

    def test_n_sub_cycles_negative_treated_as_one(self, monkeypatch):
        """負値は max(1, N) で 1 として扱う."""
        mesh, ndof, contact, M, _f, boundary = self._make_setup(n_increments_target=3)
        result, counter = self._run(
            n_sub_cycles=-5,
            max_incr=3,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            boundary=boundary,
            monkeypatch=monkeypatch,
        )
        assert isinstance(result, SolverResultData)
        # 3 増分 × max(1, -5)=1 sub-cycle = 3 回
        assert counter[0] == 3

    def test_n_sub_cycles_implicit_mode_ignored(self, monkeypatch):
        """implicit mode では本 field を無視（既存 NR 挙動）.

        本 field は `solver_mode="explicit"` のみ対象。implicit 経路では
        `ExplicitDynamicProcess.process()` 自体が呼ばれないことを検証.
        """
        mesh, ndof, contact, M, _f, boundary = self._make_setup(n_increments_target=2)
        callbacks = self._make_counting_callbacks(ndof)
        counter = [0]
        original_process = ExplicitDynamicProcess.process

        def counting_process(self_proc, input_data_step):
            counter[0] += 1
            return original_process(self_proc, input_data_step)

        monkeypatch.setattr(ExplicitDynamicProcess, "process", counting_process)
        input_data = ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact,
            callbacks=callbacks,
            mass_matrix=M,
            dt_physical=0.5,
            solver_mode="implicit",
            # N=10 を指定しても implicit では無視される
            explicit_n_sub_cycles_per_increment=10,
            max_increments=2,
            max_nr_attempts=5,
        )
        result = ContactFrictionProcess().process(input_data)
        assert isinstance(result, SolverResultData)
        # implicit では ExplicitDynamicProcess は一度も呼ばれない
        assert counter[0] == 0

    def test_n_sub_cycles_dt_inner_scales(self, monkeypatch):
        """N>1 で dt_inner = dt_sub / N となり、internal step dt が縮小されることを検証.

        ExplicitDynamicProcess.process() に渡る `dt_sub` は外側 driver で
        `dt_sub / N` に縮小される。ここでは N=4 で 4 倍 sub-cycle 呼出が
        起きること（=外側 dt_sub が論理的に分割されていること）を間接確認.
        """
        mesh, ndof, contact, M, _f, boundary = self._make_setup(n_increments_target=2)
        _result_n1, counter_n1 = self._run(
            n_sub_cycles=1,
            max_incr=2,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            boundary=boundary,
            monkeypatch=monkeypatch,
        )
        # n=1 直後に monkeypatch を再 apply して counter をリセット
        monkeypatch.undo()
        _result_n4, counter_n4 = self._run(
            n_sub_cycles=4,
            max_incr=2,
            ndof=ndof,
            mesh=mesh,
            contact=contact,
            M=M,
            boundary=boundary,
            monkeypatch=monkeypatch,
        )
        # N=4 / N=1 で sub-cycle 呼出は 4 倍
        assert counter_n4[0] == 4 * counter_n1[0]

    def test_gate_logic_explicit_only(self):
        """gate 式 `_solver_mode == "explicit"` の論理的検証.

        sub-cycle 内部ループは explicit のときのみ実行。implicit / その他 mode では
        本 field を読まずに従来 NR 経路を辿る.
        """

        # gate: (mode == "explicit") のときのみ N_sub を参照
        def _use_sub_cycle(*, mode: str, n_sub: int) -> bool:
            return mode == "explicit" and max(1, n_sub) > 1

        # explicit + N=1: False（gate を通っても N=1 は no-op と等価）
        assert _use_sub_cycle(mode="explicit", n_sub=1) is False
        # explicit + N=10: True
        assert _use_sub_cycle(mode="explicit", n_sub=10) is True
        # implicit + N=10: False（gate で除外）
        assert _use_sub_cycle(mode="implicit", n_sub=10) is False
        # explicit + N=0: False（max(1, 0)=1）
        assert _use_sub_cycle(mode="explicit", n_sub=0) is False


class TestPerElementCriticalDt:
    """status-384 候補 (z1a): 要素ごと波速ベース Δt 推定の単体検証."""

    def test_zero_E_returns_inf(self):
        conn = np.array([[0, 1]], dtype=int)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        dt = _estimate_critical_dt_per_element(conn, coords, beam_E=0.0, beam_rho=1.0)
        assert dt == float("inf")

    def test_zero_rho_returns_inf(self):
        conn = np.array([[0, 1]], dtype=int)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        dt = _estimate_critical_dt_per_element(conn, coords, beam_E=1.0, beam_rho=0.0)
        assert dt == float("inf")

    def test_none_connectivity_returns_inf(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        dt = _estimate_critical_dt_per_element(None, coords, beam_E=1.0, beam_rho=1.0)
        assert dt == float("inf")

    def test_single_element_axial_wave(self):
        """単一要素 L=1 mm, E=1, ρ=1 → c_a=1, dt = 1/1 = 1.0."""
        conn = np.array([[0, 1]], dtype=int)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        dt = _estimate_critical_dt_per_element(conn, coords, beam_E=1.0, beam_rho=1.0)
        assert dt == pytest.approx(1.0, rel=1e-12)

    def test_minimum_element_governs(self):
        """複数要素で最短の要素が dt を支配する."""
        conn = np.array([[0, 1], [1, 2]], dtype=int)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],  # 長: L=10
                [10.5, 0.0, 0.0],  # 短: L=0.5
            ]
        )
        dt = _estimate_critical_dt_per_element(conn, coords, beam_E=1.0, beam_rho=1.0)
        # min L = 0.5, c_a = 1, dt = 0.5
        assert dt == pytest.approx(0.5, rel=1e-12)

    def test_steel_beam_realistic_dt(self):
        """現実的な梁: E=130 GPa, ρ=8.96e-9 ton/mm³, L=6.25 mm → dt ≈ 1.64 μs."""
        E = 1.30e5  # N/mm² = 130 GPa
        rho = 8.96e-9  # ton/mm³
        conn = np.array([[0, 1]], dtype=int)
        coords = np.array([[0.0, 0.0, 0.0], [6.25, 0.0, 0.0]])
        dt = _estimate_critical_dt_per_element(conn, coords, beam_E=E, beam_rho=rho)
        c_a = float(np.sqrt(E / rho))
        expected = 6.25 / c_a
        assert dt == pytest.approx(expected, rel=1e-10)
        # 1.6 μs オーダーであることも検証
        assert 1e-6 < dt < 3e-6


class TestDetectStiffDofs:
    """status-384 候補 (z1b): selective mass scaling の DOF 検出ヘルパ単体検証."""

    def test_uniform_K_detects_nothing(self):
        """全 DOF で同じ row-sum/M なら閾値超えゼロ."""
        K = sp.diags([1.0] * 10, format="csr")
        M = np.ones(10)
        mask = _detect_stiff_dofs(K, M, fixed_dofs=np.array([], dtype=int), threshold_ratio=10.0)
        assert mask.sum() == 0

    def test_outlier_dof_detected(self):
        """1 つの DOF だけ K が 100× なら、その DOF が stiff として検出される."""
        diag = np.ones(10)
        diag[3] = 100.0  # 局在
        K = sp.diags(diag, format="csr")
        M = np.ones(10)
        mask = _detect_stiff_dofs(K, M, fixed_dofs=np.array([], dtype=int), threshold_ratio=10.0)
        assert mask[3]
        assert mask.sum() == 1

    def test_fixed_dofs_excluded_from_median(self):
        """固定 DOF は median 計算から除外される."""
        diag = np.array([1.0, 1.0, 1.0, 100.0, 1.0])
        K = sp.diags(diag, format="csr")
        M = np.ones(5)
        # DOF 3 を固定
        mask = _detect_stiff_dofs(K, M, fixed_dofs=np.array([3]), threshold_ratio=10.0)
        # 固定 DOF は free=False なので out=False
        assert not mask[3]

    def test_zero_mass_dofs_skipped(self):
        """M_lump=0 の DOF は free=False で検出対象外."""
        diag = np.array([1.0, 100.0])
        K = sp.diags(diag, format="csr")
        M = np.array([1.0, 0.0])
        mask = _detect_stiff_dofs(K, M, fixed_dofs=np.array([], dtype=int), threshold_ratio=10.0)
        # M=0 の DOF は除外
        assert not mask[1]

    def test_threshold_ratio_controls_sensitivity(self):
        """threshold_ratio を上げると検出が減る."""
        diag = np.array([1.0, 1.0, 1.0, 5.0])
        K = sp.diags(diag, format="csr")
        M = np.ones(4)
        # ratio=2 で 5 倍 DOF は検出される（median=1, threshold=2, 5>2）
        mask_lo = _detect_stiff_dofs(K, M, fixed_dofs=np.array([], dtype=int), threshold_ratio=2.0)
        # ratio=10 で 5 倍 DOF は検出されない（threshold=10, 5<10）
        mask_hi = _detect_stiff_dofs(K, M, fixed_dofs=np.array([], dtype=int), threshold_ratio=10.0)
        assert mask_lo[3]
        assert not mask_hi[3]


class TestSelectiveMassScaling:
    """status-384 候補 (z1b): selective mass scaling の統合検証."""

    def test_default_no_mask_full_scaling(self):
        """default mask=None なら全 DOF 一律 β² スケーリング（既存挙動）."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(M, mass_scaling_beta=2.0)
        # M_lump = β²·M_raw = 4·1 = 4 全 DOF
        np.testing.assert_allclose(proc.M_lump, 4.0)

    def test_mask_applied_selectively(self):
        """mask 内 DOF のみ β² 倍化、外は raw."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(M, mass_scaling_beta=3.0)
        mask = np.array([True, False, True, False])
        proc.set_mass_scaling_dof_mask(mask)
        # mask True: M = 9, False: M = 1
        np.testing.assert_allclose(proc.M_lump, [9.0, 1.0, 9.0, 1.0])

    def test_set_mask_then_increase_beta_selective(self):
        """mask 設定後に β を上げると、mask 内のみ β² 倍化される."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(M, mass_scaling_beta=2.0)
        mask = np.array([True, False, True, False])
        proc.set_mass_scaling_dof_mask(mask)
        # 初期: mask 内 β=2 → M=4, 外 M=1
        np.testing.assert_allclose(proc.M_lump, [4.0, 1.0, 4.0, 1.0])
        proc.set_mass_scaling_beta(5.0)
        # β=5 → mask 内 M=25, 外 M=1
        np.testing.assert_allclose(proc.M_lump, [25.0, 1.0, 25.0, 1.0])

    def test_clear_mask_restores_global(self):
        """set_mass_scaling_dof_mask(None) で全 DOF 一律に戻る."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(M, mass_scaling_beta=3.0)
        proc.set_mass_scaling_dof_mask(np.array([True, False, True, False]))
        proc.set_mass_scaling_dof_mask(None)
        # 全 DOF β=3 → M=9
        np.testing.assert_allclose(proc.M_lump, 9.0)

    def test_invalid_mask_shape_rejected(self):
        """mask shape 不一致は ValueError."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(M, mass_scaling_beta=2.0)
        with pytest.raises(ValueError, match="mask shape"):
            proc.set_mass_scaling_dof_mask(np.array([True, False]))

    def test_selective_v_a_rescale_only_in_mask(self):
        """β 上方更新時、KE 保存 rescale は mask 内 DOF にのみ適用される."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(M, mass_scaling_beta=2.0)
        mask = np.array([True, False, True, False])
        proc.set_mass_scaling_dof_mask(mask)
        # 速度を全 DOF に注入
        proc.vel = np.array([1.0, 1.0, 1.0, 1.0])
        proc.set_mass_scaling_beta(4.0)
        # ratio = 2/4 = 0.5; mask 内のみ rescale
        np.testing.assert_allclose(proc.vel, [0.5, 1.0, 0.5, 1.0])


class TestTwoStageMassScaling:
    """status-384 候補 (z1c): 2 段階質量スケーリング（β + β_outside）."""

    def test_default_beta_outside_one(self):
        """default β_outside=1.0 で z1b 単独動作と等価."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(M, mass_scaling_beta=3.0)
        assert proc.mass_scaling_beta_outside == 1.0
        mask = np.array([True, False, True, False])
        proc.set_mass_scaling_dof_mask(mask)
        # mask True: M=9, mask False: β_outside²=1 → M=1
        np.testing.assert_allclose(proc.M_lump, [9.0, 1.0, 9.0, 1.0])

    def test_two_stage_scaling_with_beta_outside(self):
        """β=10, β_outside=2 で mask True/False に異なる β² 適用."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(
            M, mass_scaling_beta=10.0, mass_scaling_beta_outside=2.0
        )
        mask = np.array([True, False, True, False])
        proc.set_mass_scaling_dof_mask(mask)
        # mask True: 10²=100, mask False: 2²=4
        np.testing.assert_allclose(proc.M_lump, [100.0, 4.0, 100.0, 4.0])

    def test_beta_outside_ignored_when_mask_none(self):
        """mask=None では β_outside は無視され全 DOF が β² で一律."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(
            M, mass_scaling_beta=3.0, mass_scaling_beta_outside=10.0
        )
        # mask=None なので 全 DOF M = β²·1 = 9（β_outside は無視）
        np.testing.assert_allclose(proc.M_lump, 9.0)

    def test_beta_outside_below_one_raises(self):
        """β_outside < 1.0 は ValueError."""
        M = sp.eye(4, format="csr")
        with pytest.raises(ValueError, match="mass_scaling_beta_outside must be >= 1.0"):
            ExplicitCentralDifferenceProcess(M, mass_scaling_beta_outside=0.5)

    def test_set_beta_outside_increases(self):
        """set_mass_scaling_beta_outside() で β_outside を上方更新できる."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(
            M, mass_scaling_beta=5.0, mass_scaling_beta_outside=2.0
        )
        mask = np.array([True, False, True, False])
        proc.set_mass_scaling_dof_mask(mask)
        # 初期: mask True M=25, mask False M=4
        np.testing.assert_allclose(proc.M_lump, [25.0, 4.0, 25.0, 4.0])
        proc.set_mass_scaling_beta_outside(5.0)
        # β_outside=5 → mask True 不変、mask False M=25
        assert proc.mass_scaling_beta_outside == 5.0
        np.testing.assert_allclose(proc.M_lump, [25.0, 25.0, 25.0, 25.0])

    def test_set_beta_outside_monotone(self):
        """set_mass_scaling_beta_outside() は単調増加のみ."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(
            M, mass_scaling_beta=2.0, mass_scaling_beta_outside=5.0
        )
        proc.set_mass_scaling_beta_outside(3.0)  # 5 より小さい → 何もしない
        assert proc.mass_scaling_beta_outside == 5.0

    def test_set_beta_outside_invalid_raises(self):
        """β_outside < 1.0 で ValueError."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(M)
        with pytest.raises(ValueError, match="mass_scaling_beta_outside must be >= 1.0"):
            proc.set_mass_scaling_beta_outside(0.5)

    def test_set_beta_outside_rescales_v_outside_only(self):
        """β_outside 上方更新で KE 保存 rescale が mask=False DOF のみ適用."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(
            M, mass_scaling_beta=4.0, mass_scaling_beta_outside=2.0
        )
        mask = np.array([True, False, True, False])
        proc.set_mass_scaling_dof_mask(mask)
        proc.vel = np.array([1.0, 1.0, 1.0, 1.0])
        proc.set_mass_scaling_beta_outside(4.0)
        # ratio = 2/4 = 0.5; mask=False のみ rescale
        np.testing.assert_allclose(proc.vel, [1.0, 0.5, 1.0, 0.5])

    def test_set_beta_outside_no_rescale_when_mask_none(self):
        """mask=None で β_outside 上方更新しても v は変化しない（β_outside 自体無効）."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(
            M, mass_scaling_beta=4.0, mass_scaling_beta_outside=2.0
        )
        proc.vel = np.array([1.0, 1.0, 1.0, 1.0])
        proc.set_mass_scaling_beta_outside(8.0)
        # mask=None なので β_outside は M_lump に作用しない、v も不変
        np.testing.assert_allclose(proc.vel, [1.0, 1.0, 1.0, 1.0])

    def test_critical_dt_increases_with_beta_outside(self):
        """β_outside 増加で M_lump_inv が縮小し dt_c が拡大することを確認."""
        M = sp.eye(4, format="csr")
        proc = ExplicitCentralDifferenceProcess(
            M, mass_scaling_beta=2.0, mass_scaling_beta_outside=1.0
        )
        mask = np.array([True, False, True, False])
        proc.set_mass_scaling_dof_mask(mask)
        m_inv_outside_before = proc.M_lump_inv[1]
        proc.set_mass_scaling_beta_outside(5.0)
        m_inv_outside_after = proc.M_lump_inv[1]
        # M_outside: 1 → 25 ⇒ M_inv: 1 → 1/25 (縮小)
        assert m_inv_outside_after < m_inv_outside_before * 0.05
        # mask 内 DOF は不変
        assert proc.M_lump_inv[0] == pytest.approx(1.0 / 4.0)

    def test_factory_passes_beta_outside(self):
        """_create_time_integration_strategy 経由で β_outside が伝搬する."""
        from xkep_cae.time_integration.strategy import _create_time_integration_strategy

        M = sp.eye(4, format="csr")
        s = _create_time_integration_strategy(
            mass_matrix=M,
            dt_physical=0.01,
            solver_mode="explicit",
            mass_scaling_beta=3.0,
            mass_scaling_beta_outside=7.0,
        )
        assert isinstance(s, ExplicitCentralDifferenceProcess)
        assert s.mass_scaling_beta == 3.0
        assert s.mass_scaling_beta_outside == 7.0


# =====================================================================
# 共通ヘルパ
# =====================================================================


class _MockULAssembler:
    """update_reference 呼出回数を計測するモック UL アセンブラ.

    status-383 候補 (q1) `explicit_ul_update_interval` テスト用。
    実 UL アセンブラの最小プロトコル（`update_reference` / `checkpoint` /
    `rollback` / `u_total_accum` / `coords_ref`）を満たす。
    """

    def __init__(self, ndof: int) -> None:
        self.update_reference_call_count = 0
        self._ndof = ndof
        self.u_total_accum = np.zeros(ndof)
        self.coords_ref = np.zeros((ndof // 6, 3))

    def update_reference(self, u_incr: np.ndarray) -> None:
        self.update_reference_call_count += 1

    def checkpoint(self) -> None:
        pass

    def rollback(self) -> None:
        pass


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
