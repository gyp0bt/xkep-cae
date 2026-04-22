"""ContactFrictionProcess のテスト."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact._manager_process import (
    DetectCandidatesInput,
    DetectCandidatesOutput,
    DetectCandidatesProcess,
    InitializePenaltyProcess,
    UpdateGeometryInput,
    UpdateGeometryOutput,
    UpdateGeometryProcess,
)
from xkep_cae.contact.solver._adaptive_stepping import (
    AdaptiveStepInput,
    AdaptiveStepOutput,
    AdaptiveSteppingInput,
    AdaptiveSteppingProcess,
    StepAction,
)
from xkep_cae.contact.solver._contact_graph import (
    ContactGraphInput,
    ContactGraphOutput,
    ContactGraphProcess,
)
from xkep_cae.contact.solver._diagnostics import (
    ConvergenceDiagnosticsOutput,
    DiagnosticsInput,
    DiagnosticsOutput,
    DiagnosticsReportProcess,
)
from xkep_cae.contact.solver._initial_penetration import (
    InitialPenetrationInput,
    InitialPenetrationOutput,
    InitialPenetrationProcess,
)
from xkep_cae.contact.solver._newton_dynamic import NewtonDynamicInput, NewtonDynamicProcess
from xkep_cae.contact.solver._newton_steps import (
    ContactBacktrackingLineSearchInput,
    ContactBacktrackingLineSearchOutput,
    ContactBacktrackingLineSearchProcess,
    ContactForceAssemblyProcess,
    ConvergenceCheckInput,
    ConvergenceCheckOutput,
    ConvergenceCheckProcess,
    ConvergenceType,
    LinearSolveInput,
    LinearSolveOutput,
    LinearSolveProcess,
    LineSearchUpdateInput,
    LineSearchUpdateOutput,
    LineSearchUpdateProcess,
    TangentAssemblyProcess,
    _SolverCache,
    _zero_sparse_rows,
)
from xkep_cae.contact.solver._solver_state import (
    SolverStateInitInput,
    SolverStateInitOutput,
    SolverStateInitProcess,
)
from xkep_cae.contact.solver._utils import (
    DeformedCoordsInput,
    DeformedCoordsOutput,
    DeformedCoordsProcess,
    NCPLineSearchInput,
    NCPLineSearchOutput,
    NCPLineSearchProcess,
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
from xkep_cae.core.testing import binds_to


def _make_two_beam_mesh() -> MeshData:
    """テスト用の簡易2本梁メッシュ（16要素/ピッチ以上）."""
    n_nodes_per_strand = 17  # 16要素
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
    """簡易アセンブリコールバック（単位行列のスカラー剛性）."""

    def assemble_tangent(u: np.ndarray) -> sp.csr_matrix:
        return sp.eye(ndof, format="csr") * 1.0

    def assemble_internal_force(u: np.ndarray) -> np.ndarray:
        return u * 1.0  # 線形バネ

    return AssembleCallbacks(
        assemble_tangent=assemble_tangent,
        assemble_internal_force=assemble_internal_force,
    )


def _make_contact_setup(mesh: MeshData) -> ContactSetupData:
    """ContactSetupProcess 経由の接触設定."""
    from xkep_cae.contact.setup.process import ContactSetupConfig, ContactSetupProcess

    setup = ContactSetupProcess()
    setup_config = ContactSetupConfig(
        mesh=mesh,
        k_pen=1e4,
        mu=0.15,
        exclude_same_strand=True,
    )
    return setup.process(setup_config)


@binds_to(ContactFrictionProcess)
class TestContactFrictionProcessAPI:
    """ContactFrictionProcess の API テスト."""

    def test_is_solver_process(self):
        proc = ContactFrictionProcess()
        assert isinstance(proc, SolverProcess)

    def test_meta_name(self):
        assert ContactFrictionProcess.meta.name == "ContactFriction"

    def test_meta_module(self):
        assert ContactFrictionProcess.meta.module == "solve"

    def test_meta_version(self):
        assert ContactFrictionProcess.meta.version == "2.0.0"

    def test_default_strategies(self):
        proc = ContactFrictionProcess()
        assert proc.strategies is not None
        assert proc.strategies.penalty is not None
        assert proc.strategies.friction is not None
        assert proc.strategies.time_integration is not None

    def test_custom_strategies(self):
        from xkep_cae.core.data import default_strategies

        strats = default_strategies(k_pen=999.0)
        proc = ContactFrictionProcess(strategies=strats)
        assert proc.strategies is strats

    @pytest.mark.skip(reason="status-222: 動的ソルバーのみ。mass_matrix 追加が必要。")
    def test_process_returns_solver_result_data(self):
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        callbacks = _make_simple_callbacks(ndof)
        contact = _make_contact_setup(mesh)

        f_ext = np.zeros(ndof)
        # 右端に微小荷重
        f_ext[6 * 16] = 0.001

        boundary = BoundaryData(
            fixed_dofs=np.arange(6),  # 左端固定
            f_ext_total=f_ext,
        )

        input_data = ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact,
            callbacks=callbacks,
        )

        proc = ContactFrictionProcess()
        result = proc.process(input_data)
        assert isinstance(result, SolverResultData)

    @pytest.mark.skip(reason="status-222: 動的ソルバーのみ。mass_matrix 追加が必要。")
    def test_process_converges_simple(self):
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        callbacks = _make_simple_callbacks(ndof)
        contact = _make_contact_setup(mesh)

        f_ext = np.zeros(ndof)
        f_ext[6 * 16] = 0.001

        boundary = BoundaryData(
            fixed_dofs=np.arange(6),
            f_ext_total=f_ext,
        )

        input_data = ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact,
            callbacks=callbacks,
        )

        proc = ContactFrictionProcess()
        result = proc.process(input_data)
        assert result.converged is True

    @pytest.mark.skip(reason="status-222: 動的ソルバーのみ。mass_matrix 追加が必要。")
    def test_process_has_displacement(self):
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        callbacks = _make_simple_callbacks(ndof)
        contact = _make_contact_setup(mesh)

        f_ext = np.zeros(ndof)
        f_ext[6 * 16] = 0.001

        boundary = BoundaryData(
            fixed_dofs=np.arange(6),
            f_ext_total=f_ext,
        )

        input_data = ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact,
            callbacks=callbacks,
        )

        proc = ContactFrictionProcess()
        result = proc.process(input_data)
        assert result.u is not None
        assert len(result.u) == ndof

    @pytest.mark.skip(reason="status-222: 動的ソルバーのみ。mass_matrix 追加が必要。")
    def test_process_records_elapsed(self):
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        callbacks = _make_simple_callbacks(ndof)
        contact = _make_contact_setup(mesh)

        f_ext = np.zeros(ndof)
        f_ext[6 * 16] = 0.001

        boundary = BoundaryData(
            fixed_dofs=np.arange(6),
            f_ext_total=f_ext,
        )

        input_data = ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact,
            callbacks=callbacks,
        )

        proc = ContactFrictionProcess()
        result = proc.process(input_data)
        assert result.elapsed_seconds > 0.0

    @pytest.mark.skip(reason="status-222: 動的ソルバーのみ。mass_matrix 追加が必要。")
    def test_prescribed_displacement(self):
        """処方変位のテスト."""
        mesh = _make_two_beam_mesh()
        ndof = len(mesh.node_coords) * 6
        callbacks = _make_simple_callbacks(ndof)
        contact = _make_contact_setup(mesh)

        f_ext = np.zeros(ndof)
        f_ext[6 * 16] = 0.001

        prescribed_dofs = np.array([6 * 16 + 1])  # y方向
        prescribed_values = np.array([0.001])

        boundary = BoundaryData(
            fixed_dofs=np.arange(6),
            f_ext_total=f_ext,
            prescribed_dofs=prescribed_dofs,
            prescribed_values=prescribed_values,
        )

        input_data = ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact,
            callbacks=callbacks,
        )

        proc = ContactFrictionProcess()
        result = proc.process(input_data)
        assert isinstance(result, SolverResultData)


@binds_to(NewtonDynamicProcess)
class TestNewtonDynamicProcessAPI:
    """NewtonDynamicProcess の API テスト（status-222 で一本化）."""

    def test_is_solver_process(self):
        proc = NewtonDynamicProcess()
        assert isinstance(proc, SolverProcess)

    def test_meta_name(self):
        assert NewtonDynamicProcess.meta.name == "NewtonDynamic"

    def test_meta_module(self):
        assert NewtonDynamicProcess.meta.module == "solve"


@binds_to(AdaptiveSteppingProcess)
class TestAdaptiveSteppingProcessAPI:
    """AdaptiveSteppingProcess の API テスト."""

    def test_is_solver_process(self):
        config = AdaptiveSteppingInput(dt_initial_fraction=0.5)
        proc = AdaptiveSteppingProcess(config)
        assert isinstance(proc, SolverProcess)

    def test_meta_name(self):
        assert AdaptiveSteppingProcess.meta.name == "AdaptiveStepping"

    def test_meta_module(self):
        assert AdaptiveSteppingProcess.meta.module == "solve"

    def test_query_returns_output(self):
        config = AdaptiveSteppingInput(dt_initial_fraction=0.5)
        proc = AdaptiveSteppingProcess(config)
        out = proc.process(AdaptiveStepInput(action=StepAction.QUERY, load_frac_prev=0.0))
        assert isinstance(out, AdaptiveStepOutput)
        assert out.has_more_steps is True
        assert out.next_load_frac > 0.0

    def test_full_cycle(self):
        """QUERY → SUCCESS → QUERY で完了まで回る."""
        config = AdaptiveSteppingInput(dt_initial_fraction=1.0)
        proc = AdaptiveSteppingProcess(config)

        out = proc.process(AdaptiveStepInput(action=StepAction.QUERY, load_frac_prev=0.0))
        assert out.next_load_frac == 1.0

        out = proc.process(
            AdaptiveStepInput(
                action=StepAction.SUCCESS,
                load_frac=1.0,
                load_frac_prev=0.0,
                n_attempts=3,
            )
        )
        assert out.has_more_steps is False

    def test_failure_triggers_retry(self):
        """FAILURE でカットバック → can_retry=True."""
        config = AdaptiveSteppingInput(dt_initial_fraction=0.5)
        proc = AdaptiveSteppingProcess(config)

        out = proc.process(AdaptiveStepInput(action=StepAction.QUERY, load_frac_prev=0.0))
        load_frac = out.next_load_frac

        fail_out = proc.process(
            AdaptiveStepInput(
                action=StepAction.FAILURE,
                load_frac=load_frac,
                load_frac_prev=0.0,
            )
        )
        assert fail_out.can_retry is True
        assert fail_out.next_load_frac < load_frac

    def test_snap_to_one_avoids_micro_dt(self):
        """残りが通常dtの半分未満なら1.0に吸収する（status-296: 微小dt防止）."""
        config = AdaptiveSteppingInput(dt_initial_fraction=0.1)
        proc = AdaptiveSteppingProcess(config)

        # QUERY → frac=0.1
        out = proc.process(AdaptiveStepInput(action=StepAction.QUERY, load_frac_prev=0.0))
        assert out.next_load_frac == pytest.approx(0.1, abs=1e-6)

        # SUCCESS で次ステップを計算。load_frac_prev=0.94, load_frac=0.97 とする。
        # dt=0.03, 残り=0.03。次のdtも ~0.03 なら残り~0 → snap不要。
        # load_frac_prev=0.96, load_frac=0.98 とする。
        # dt=0.02, 次dt~0.03 → next_frac=1.01 → min(1.0) → remaining=0。snap不要。
        # テスト: load_frac=0.985, prev=0.97 → dt=0.015, next~0.015*1.5=0.0225
        # next_frac=0.985+0.0225=1.0075 → clamp to 1.0 → remaining=0 → snap不要
        # テスト: load_frac=0.99, prev=0.985 → dt=0.005, next~0.005*1.5=0.0075
        # next_frac=0.99+0.0075=0.9975 → remaining=0.0025 < 0.0075*0.5=0.00375 → snap!
        out2 = proc.process(
            AdaptiveStepInput(
                action=StepAction.SUCCESS,
                load_frac=0.99,
                load_frac_prev=0.985,
                n_attempts=3,  # good convergence → grow
            )
        )
        # next_frac should snap to 1.0 (remaining < next_delta * 0.5)
        assert out2.next_load_frac == pytest.approx(1.0, abs=1e-10)


# ── InitialPenetrationProcess テスト ─────────────────────


@binds_to(InitialPenetrationProcess)
class TestInitialPenetrationProcessAPI:
    """InitialPenetrationProcess の API テスト."""

    def test_protocol_conformance(self):
        """SolverProcess を継承している."""
        assert issubclass(InitialPenetrationProcess, SolverProcess)

    def test_no_penetration(self):
        """貫入なしの場合 n_penetrations=0."""
        proc = InitialPenetrationProcess()
        out = proc.process(InitialPenetrationInput(pairs=[], node_coords=np.zeros((4, 3))))
        assert isinstance(out, InitialPenetrationOutput)
        assert out.n_penetrations == 0

    def test_with_adjust(self):
        """adjust=True で座標調整モード."""
        proc = InitialPenetrationProcess()
        out = proc.process(
            InitialPenetrationInput(
                pairs=[],
                node_coords=np.zeros((4, 3)),
                adjust=True,
            )
        )
        assert out.n_penetrations == 0
        assert out.adjusted_coords is None


# ── ContactGraphProcess テスト ───────────────────────────


class _MockManager:
    """テスト用 mock manager."""

    def __init__(self) -> None:
        self.pairs: list = []
        self.n_pairs = 0


@binds_to(ContactGraphProcess)
class TestContactGraphProcessAPI:
    """ContactGraphProcess の API テスト."""

    def test_protocol_conformance(self):
        """SolverProcess を継承している."""
        assert issubclass(ContactGraphProcess, SolverProcess)

    def test_empty_manager(self):
        """空の manager から空グラフを生成."""
        proc = ContactGraphProcess()
        out = proc.process(ContactGraphInput(manager=_MockManager(), step=1, load_factor=0.5))
        assert isinstance(out, ContactGraphOutput)
        assert len(out.graph.edges) == 0
        assert out.graph.step == 1
        assert out.graph.load_factor == 0.5


# ── DiagnosticsReportProcess テスト ──────────────────────


@binds_to(DiagnosticsReportProcess)
class TestDiagnosticsReportProcessAPI:
    """DiagnosticsReportProcess の API テスト."""

    def test_protocol_conformance(self):
        """SolverProcess を継承している."""
        assert issubclass(DiagnosticsReportProcess, SolverProcess)

    def test_basic_report(self):
        """基本的な診断レポートが生成される."""
        proc = DiagnosticsReportProcess()
        diag = ConvergenceDiagnosticsOutput(step=5, load_frac=0.8, res_history=[1e-3, 1e-4, 1e-5])
        out = proc.process(DiagnosticsInput(diagnostics=diag, max_attempts=50))
        assert isinstance(out, DiagnosticsOutput)
        assert "Step: 5" in out.report
        assert "0.800000" in out.report


# ── SolverStateInitProcess テスト ────────────────────────


@binds_to(SolverStateInitProcess)
class TestSolverStateInitProcessAPI:
    """SolverStateInitProcess の API テスト."""

    def test_protocol_conformance(self):
        """SolverProcess を継承している."""
        assert issubclass(SolverStateInitProcess, SolverProcess)

    def test_creates_zero_state(self):
        """ゼロ初期状態を生成."""
        proc = SolverStateInitProcess()
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        out = proc.process(SolverStateInitInput(ndof=12, node_coords=coords))
        assert isinstance(out, SolverStateInitOutput)
        assert len(out.state.u) == 12
        assert np.allclose(out.state.u, 0.0)


# ── DeformedCoordsProcess テスト ─────────────────────────


@binds_to(DeformedCoordsProcess)
class TestDeformedCoordsProcessAPI:
    """DeformedCoordsProcess の API テスト."""

    def test_protocol_conformance(self):
        """SolverProcess を継承している."""
        assert issubclass(DeformedCoordsProcess, SolverProcess)

    def test_deformed_coords(self):
        """変形座標を計算."""
        proc = DeformedCoordsProcess()
        coords_ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        u = np.zeros(12)
        u[0] = 0.1  # node0 x方向
        u[7] = 0.2  # node1 y方向
        out = proc.process(DeformedCoordsInput(node_coords_ref=coords_ref, u=u))
        assert isinstance(out, DeformedCoordsOutput)
        assert out.coords[0, 0] == 0.1
        assert out.coords[1, 1] == 0.2


# ── NCPLineSearchProcess テスト ──────────────────────────


@binds_to(NCPLineSearchProcess)
class TestNCPLineSearchProcessAPI:
    """NCPLineSearchProcess の API テスト."""

    def test_protocol_conformance(self):
        """SolverProcess を継承している."""
        assert issubclass(NCPLineSearchProcess, SolverProcess)

    def test_returns_alpha(self):
        """alpha を返す."""
        proc = NCPLineSearchProcess()
        ndof = 12
        u = np.zeros(ndof)
        du = np.ones(ndof) * 0.001
        f_ext = np.ones(ndof) * 0.001

        out = proc.process(
            NCPLineSearchInput(
                u=u,
                du=du,
                f_ext=f_ext,
                fixed_dofs=np.array([0]),
                assemble_internal_force=lambda x: x * 1.0,
                res_u_norm=1.0,
            )
        )
        assert isinstance(out, NCPLineSearchOutput)
        assert 0.0 < out.alpha <= 1.0


# ── NewtonUzawa サブプロセス テスト ──────────────────────


@binds_to(ContactForceAssemblyProcess)
class TestContactForceAssemblyProcessAPI:
    """ContactForceAssemblyProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(ContactForceAssemblyProcess, SolverProcess)


@binds_to(ConvergenceCheckProcess)
class TestConvergenceCheckProcessAPI:
    """ConvergenceCheckProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(ConvergenceCheckProcess, SolverProcess)

    def test_force_convergence(self):
        """力収束を判定."""

        class _MockMgr:
            pairs = []

        proc = ConvergenceCheckProcess()
        R_u = np.array([1e-12, 0.0, 0.0])
        out = proc.process(
            ConvergenceCheckInput(
                R_u=R_u,
                du=None,
                u=np.ones(3),
                f_ext_ref_norm=1.0,
                tol_force=1e-8,
                tol_disp=1e-8,
                dynamic_ref=False,
                is_first_attempt=True,
                energy_ref=None,
                manager=_MockMgr(),
            )
        )
        assert isinstance(out, ConvergenceCheckOutput)
        assert out.converged is True
        assert out.convergence_type == ConvergenceType.FORCE

    def test_atol_force_convergence(self):
        """atol_forceで微小dtの絶対残差収束を判定（status-297）.

        相対判定 res/f_ref = 4.5e-3 >> tol=1e-8 で不収束でも、
        atol_force = 5e-6 に対して res = 1.7e-6 < 5e-6 なら収束。
        """

        class _MockMgr:
            pairs = []

        proc = ConvergenceCheckProcess()

        # 微小dt: f_ref=3.8e-4, res=1.7e-6 → 相対比 4.5e-3 → 不収束
        R_u = np.array([1.7e-6, 0.0, 0.0])
        out_no_atol = proc.process(
            ConvergenceCheckInput(
                R_u=R_u,
                du=None,
                u=np.ones(3),
                f_ext_ref_norm=3.8e-4,
                tol_force=1e-8,
                tol_disp=1e-8,
                dynamic_ref=False,
                is_first_attempt=False,
                energy_ref=None,
                manager=_MockMgr(),
            )
        )
        assert out_no_atol.converged is False

        # atol_force = 5e-6 → res=1.7e-6 < 5e-6 → 収束
        out_with_atol = proc.process(
            ConvergenceCheckInput(
                R_u=R_u,
                du=None,
                u=np.ones(3),
                f_ext_ref_norm=3.8e-4,
                tol_force=1e-8,
                tol_disp=1e-8,
                dynamic_ref=False,
                is_first_attempt=False,
                energy_ref=None,
                manager=_MockMgr(),
                atol_force=5e-6,
            )
        )
        assert out_with_atol.converged is True
        assert out_with_atol.convergence_type == ConvergenceType.FORCE

    def test_atol_force_field_exists(self):
        """NewtonDynamicStepInput.atol_force フィールドの存在確認（status-297）."""
        import dataclasses

        from xkep_cae.contact.solver._newton_dynamic import (
            NewtonDynamicStepInput,
        )

        fields = {f.name: f for f in dataclasses.fields(NewtonDynamicStepInput)}
        assert "atol_force" in fields
        assert fields["atol_force"].default == 0.0


@binds_to(TangentAssemblyProcess)
class TestTangentAssemblyProcessAPI:
    """TangentAssemblyProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(TangentAssemblyProcess, SolverProcess)


class TestZeroSparseRowsAPI:
    """_zero_sparse_rows ベクトル化ヘルパーのテスト (status-312)."""

    def test_single_row(self):
        """1行のゼロ化."""
        K = sp.random(5, 5, density=0.6, format="csr", random_state=42)
        K_ref = K.copy()
        _zero_sparse_rows(K, np.array([2]))
        K_ref_dense = K_ref.toarray()
        K_ref_dense[2, :] = 0.0
        np.testing.assert_array_equal(K.toarray(), K_ref_dense)

    def test_multiple_rows(self):
        """複数行のゼロ化."""
        K = sp.random(10, 10, density=0.5, format="csr", random_state=42)
        dofs = np.array([1, 3, 7, 9])
        K_ref = K.toarray().copy()
        _zero_sparse_rows(K, dofs)
        K_ref[dofs, :] = 0.0
        np.testing.assert_array_equal(K.toarray(), K_ref)

    def test_empty_dofs(self):
        """空DOF配列で無操作."""
        K = sp.random(5, 5, density=0.6, format="csr", random_state=42)
        K_before = K.toarray().copy()
        _zero_sparse_rows(K, np.array([], dtype=int))
        np.testing.assert_array_equal(K.toarray(), K_before)

    def test_all_rows(self):
        """全行のゼロ化."""
        K = sp.random(4, 4, density=0.8, format="csr", random_state=42)
        _zero_sparse_rows(K, np.arange(4))
        np.testing.assert_array_equal(K.toarray(), np.zeros((4, 4)))


@binds_to(LinearSolveProcess)
class TestLinearSolveProcessAPI:
    """LinearSolveProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(LinearSolveProcess, SolverProcess)

    def test_simple_solve(self):
        """単純な線形ソルブ."""
        proc = LinearSolveProcess()
        K = sp.eye(3, format="csr") * 2.0
        R = np.array([1.0, 2.0, 3.0])
        out = proc.process(LinearSolveInput(K_T=K, R_u=R, fixed_dofs=np.array([], dtype=int)))
        assert isinstance(out, LinearSolveOutput)
        assert out.success is True
        assert np.allclose(out.du, [-0.5, -1.0, -1.5])

    def test_solve_with_fixed_dofs(self):
        """固定DOF適用の線形ソルブ（status-312: ベクトル化BC検証）."""
        proc = LinearSolveProcess()
        # 3x3 SPD行列
        K = sp.csr_matrix(np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]))
        R = np.array([1.0, 2.0, 3.0])
        fixed = np.array([0, 2])
        out = proc.process(LinearSolveInput(K_T=K, R_u=R, fixed_dofs=fixed))
        assert out.success is True
        # 固定DOFの変位はゼロ
        assert abs(out.du[0]) < 1e-12
        assert abs(out.du[2]) < 1e-12
        # 自由DOF: K[1,1]*du[1] = -R[1] → 3*du[1] = -2 → du[1] = -2/3
        assert abs(out.du[1] - (-2.0 / 3.0)) < 1e-12


class TestSolverCache:
    """_SolverCache のユニットテスト（status-325）."""

    def test_scipy_fallback_solve(self):
        """pypardiso 不在時は scipy spsolve で正しく求解."""
        cache = _SolverCache()
        cache._use_pardiso = False  # 強制 scipy
        K = sp.eye(3, format="csc") * 2.0
        rhs = np.array([1.0, 2.0, 3.0])
        x = cache.solve(K, rhs)
        np.testing.assert_allclose(x, [0.5, 1.0, 1.5])

    def test_pattern_changed_initial(self):
        """初回は常に pattern_changed = True."""
        cache = _SolverCache()
        K = sp.eye(3, format="csc")
        assert cache._pattern_changed(K) is True

    def test_pattern_changed_same_pattern(self):
        """同一パターンの行列で False."""
        cache = _SolverCache()
        K = sp.eye(3, format="csc") * 2.0
        cache._solver = object()  # non-None sentinel
        cache._cache_pattern(K)
        # 同じパターン、異なる値
        K2 = sp.eye(3, format="csc") * 5.0
        assert cache._pattern_changed(K2) is False

    def test_pattern_changed_different_shape(self):
        """異なるサイズの行列で True."""
        cache = _SolverCache()
        K3 = sp.eye(3, format="csc")
        cache._solver = object()
        cache._cache_pattern(K3)
        K4 = sp.eye(4, format="csc")
        assert cache._pattern_changed(K4) is True

    def test_pattern_changed_different_structure(self):
        """同じサイズだがパターン異なる行列で True."""
        cache = _SolverCache()
        K1 = sp.eye(3, format="csc")
        cache._solver = object()
        cache._cache_pattern(K1)
        # オフダイアゴナル要素を追加
        K2 = sp.eye(3, format="csc") + sp.csc_matrix(([1.0], ([0], [1])), shape=(3, 3))
        assert cache._pattern_changed(K2) is True

    def test_invalidate(self):
        """invalidate() でキャッシュクリア."""
        cache = _SolverCache()
        K = sp.eye(3, format="csc")
        cache._solver = object()
        cache._cache_pattern(K)
        cache.invalidate()
        assert cache._solver is None
        assert cache._indptr is None
        assert cache._shape is None

    def test_stats_counters_scipy(self):
        """scipy path ではカウンタは変化しない."""
        cache = _SolverCache()
        cache._use_pardiso = False
        K = sp.eye(3, format="csc") * 2.0
        rhs = np.array([1.0, 2.0, 3.0])
        cache.solve(K, rhs)
        cache.solve(K, rhs)
        assert cache.n_symbolic == 0
        assert cache.n_numeric_only == 0

    def test_repeated_solve_same_proc_correctness(self):
        """同一 LinearSolveProcess で連続ソルブが正確."""
        proc = LinearSolveProcess()
        empty_fixed = np.array([], dtype=int)
        # Solve 1
        K1 = sp.eye(3, format="csr") * 2.0
        R1 = np.array([2.0, 4.0, 6.0])
        out1 = proc.process(LinearSolveInput(K_T=K1, R_u=R1, fixed_dofs=empty_fixed))
        assert out1.success
        np.testing.assert_allclose(out1.du, [-1.0, -2.0, -3.0])
        # Solve 2: 同じパターン、異なる値
        K2 = sp.eye(3, format="csr") * 4.0
        R2 = np.array([4.0, 8.0, 12.0])
        out2 = proc.process(LinearSolveInput(K_T=K2, R_u=R2, fixed_dofs=empty_fixed))
        assert out2.success
        np.testing.assert_allclose(out2.du, [-1.0, -2.0, -3.0])

    def test_repeated_solve_pattern_change(self):
        """パターン変化しても正しく求解."""
        proc = LinearSolveProcess()
        empty_fixed = np.array([], dtype=int)
        # 3x3 対角行列
        K1 = sp.eye(3, format="csr") * 2.0
        R1 = np.array([2.0, 4.0, 6.0])
        out1 = proc.process(LinearSolveInput(K_T=K1, R_u=R1, fixed_dofs=empty_fixed))
        assert out1.success
        # 4x4 対角行列（パターン変化）
        K2 = sp.eye(4, format="csr") * 3.0
        R2 = np.array([3.0, 6.0, 9.0, 12.0])
        out2 = proc.process(LinearSolveInput(K_T=K2, R_u=R2, fixed_dofs=empty_fixed))
        assert out2.success
        np.testing.assert_allclose(out2.du, [-1.0, -2.0, -3.0, -4.0])


class TestSolverCacheMock:
    """_SolverCache の pypardiso mock テスト（status-325）."""

    def _make_mock_pardiso_module(self):
        """pypardiso モック: factorize/solve を追跡."""
        from types import ModuleType
        from unittest.mock import MagicMock

        mock_mod = ModuleType("pypardiso")

        class MockPyPardisoSolver:
            def __init__(self):
                self.factorize_count = 0
                self.solve_count = 0
                self._A = None

            def factorize(self, A):
                self.factorize_count += 1
                self._A = A.copy()

            def solve(self, A, b):
                self.solve_count += 1
                # scipy fallback for actual computation
                from scipy.sparse.linalg import spsolve

                return spsolve(A, b)

        mock_mod.PyPardisoSolver = MockPyPardisoSolver
        mock_mod.spsolve = MagicMock(
            side_effect=lambda A, b: __import__("scipy").sparse.linalg.spsolve(A, b)
        )
        return mock_mod, MockPyPardisoSolver

    def test_pardiso_symbolic_reuse(self, monkeypatch):
        """同一パターンで factorize() が1回だけ呼ばれる."""
        import sys

        mock_mod, MockCls = self._make_mock_pardiso_module()
        monkeypatch.setitem(sys.modules, "pypardiso", mock_mod)

        cache = _SolverCache()
        cache._use_pardiso = True

        K = sp.eye(3, format="csc") * 2.0
        rhs = np.array([1.0, 2.0, 3.0])

        # 1st solve: full factorization
        x1 = cache.solve(K, rhs)
        np.testing.assert_allclose(x1, [0.5, 1.0, 1.5])
        assert cache.n_symbolic == 1
        assert cache.n_numeric_only == 0

        # 2nd solve: same pattern → reuse
        K2 = sp.eye(3, format="csc") * 4.0
        rhs2 = np.array([4.0, 8.0, 12.0])
        x2 = cache.solve(K2, rhs2)
        np.testing.assert_allclose(x2, [1.0, 2.0, 3.0])
        assert cache.n_symbolic == 1
        assert cache.n_numeric_only == 1

        # factorize called only once
        solver = cache._solver
        assert solver.factorize_count == 1

    def test_pardiso_pattern_change_refactorize(self, monkeypatch):
        """パターン変化で factorize() が再呼出し."""
        import sys

        mock_mod, MockCls = self._make_mock_pardiso_module()
        monkeypatch.setitem(sys.modules, "pypardiso", mock_mod)

        cache = _SolverCache()
        cache._use_pardiso = True

        # 3x3
        K1 = sp.eye(3, format="csc") * 2.0
        cache.solve(K1, np.array([1.0, 2.0, 3.0]))
        assert cache.n_symbolic == 1

        # 4x4 → pattern change → re-factorize
        K2 = sp.eye(4, format="csc") * 3.0
        x2 = cache.solve(K2, np.array([3.0, 6.0, 9.0, 12.0]))
        np.testing.assert_allclose(x2, [1.0, 2.0, 3.0, 4.0])
        assert cache.n_symbolic == 2
        assert cache.n_numeric_only == 0

    def test_pardiso_structure_change_refactorize(self, monkeypatch):
        """同サイズでも構造変化で factorize() 再呼出し."""
        import sys

        mock_mod, MockCls = self._make_mock_pardiso_module()
        monkeypatch.setitem(sys.modules, "pypardiso", mock_mod)

        cache = _SolverCache()
        cache._use_pardiso = True

        K1 = sp.eye(3, format="csc") * 2.0
        cache.solve(K1, np.array([1.0, 2.0, 3.0]))
        assert cache.n_symbolic == 1

        # オフダイアゴナル追加 → 構造変化
        K2 = sp.eye(3, format="csc") * 2.0 + sp.csc_matrix(
            ([0.5, 0.5], ([0, 1], [1, 0])), shape=(3, 3)
        )
        cache.solve(K2, np.array([1.0, 2.0, 3.0]))
        assert cache.n_symbolic == 2


@binds_to(LineSearchUpdateProcess)
class TestLineSearchUpdateProcessAPI:
    """LineSearchUpdateProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(LineSearchUpdateProcess, SolverProcess)

    def test_no_line_search(self):
        """Line search 無効時は scale=1.0."""
        proc = LineSearchUpdateProcess()
        du = np.array([0.1, 0.2])
        out = proc.process(
            LineSearchUpdateInput(
                u=np.zeros(2),
                du=du,
                f_ext=np.ones(2),
                fixed_dofs=np.array([], dtype=int),
                assemble_internal_force=lambda x: x,
                res_u_norm=1.0,
                f_c=np.zeros(2),
                use_line_search=False,
                line_search_max_steps=5,
                du_norm_cap=0.0,
            )
        )
        assert isinstance(out, LineSearchUpdateOutput)
        assert out.scale_factor == 1.0
        assert np.allclose(out.du_scaled, du)


# UzawaUpdateProcess は status-222 で削除。


# ── ContactBacktrackingLineSearchProcess テスト（status-362: 仮説 C (c)） ──


@binds_to(ContactBacktrackingLineSearchProcess)
class TestContactBacktrackingLineSearchProcessAPI:
    """ContactBacktrackingLineSearchProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(ContactBacktrackingLineSearchProcess, SolverProcess)

    def test_accept_alpha_one_when_no_flip(self):
        """active flip + 接触残差とも閾値内なら α=1.0 即採択."""
        proc = ContactBacktrackingLineSearchProcess()
        u = np.zeros(4)
        du = np.array([0.1, 0.1, 0.1, 0.1])

        def _trial(u_eval: np.ndarray) -> tuple[float, int]:
            return 1.0, 5  # contact_res=1.0, n_active=5

        out = proc.process(
            ContactBacktrackingLineSearchInput(
                u=u,
                du=du,
                n_active_pre=5,
                contact_res_pre=1.0,
                compute_trial=_trial,
                max_steps=4,
                active_flip_threshold=3,
                residual_ratio_threshold=2.0,
                alpha_decay=0.5,
                min_alpha=0.0625,
            )
        )
        assert isinstance(out, ContactBacktrackingLineSearchOutput)
        assert out.alpha == 1.0
        assert out.reason == "accepted"
        assert out.n_bt_iter == 0
        np.testing.assert_allclose(out.du_accepted, du)

    def test_backtrack_on_excessive_active_flip(self):
        """active flip が閾値超過なら α を段階的に減らす."""
        proc = ContactBacktrackingLineSearchProcess()
        u = np.zeros(4)
        du = np.ones(4)

        calls = {"n": 0}

        def _trial(u_eval: np.ndarray) -> tuple[float, int]:
            # α=1.0 で 20 flip → reject、α=0.5 で 10 flip → reject、
            # α=0.25 で 3 flip → accept
            calls["n"] += 1
            mag = float(np.max(np.abs(u_eval)))
            if mag > 0.75:
                return 1.0, 25
            if mag > 0.375:
                return 1.0, 15
            return 1.0, 8  # flip=3 → accept at α=0.25

        out = proc.process(
            ContactBacktrackingLineSearchInput(
                u=u,
                du=du,
                n_active_pre=5,
                contact_res_pre=1.0,
                compute_trial=_trial,
                max_steps=4,
                active_flip_threshold=3,
                residual_ratio_threshold=5.0,
                alpha_decay=0.5,
                min_alpha=0.0625,
            )
        )
        assert out.reason == "accepted"
        assert out.alpha == pytest.approx(0.25)
        assert out.n_bt_iter == 2
        np.testing.assert_allclose(out.du_accepted, 0.25 * du)

    def test_backtrack_on_contact_residual_growth(self):
        """接触残差比が閾値超過なら α を減らす."""
        proc = ContactBacktrackingLineSearchProcess()
        u = np.zeros(4)
        du = np.ones(4)

        def _trial(u_eval: np.ndarray) -> tuple[float, int]:
            mag = float(np.max(np.abs(u_eval)))
            # α=1.0: res=5.0 (比 5.0, reject), α=0.5: res=1.5 (比 1.5, accept)
            if mag > 0.75:
                return 5.0, 5
            return 1.5, 5

        out = proc.process(
            ContactBacktrackingLineSearchInput(
                u=u,
                du=du,
                n_active_pre=5,
                contact_res_pre=1.0,
                compute_trial=_trial,
                max_steps=4,
                active_flip_threshold=10,
                residual_ratio_threshold=2.0,
                alpha_decay=0.5,
                min_alpha=0.0625,
            )
        )
        assert out.reason == "accepted"
        assert out.alpha == pytest.approx(0.5)

    def test_min_alpha_guard(self):
        """min_alpha 未満は却下で打ち切る."""
        proc = ContactBacktrackingLineSearchProcess()
        u = np.zeros(2)
        du = np.ones(2)

        def _trial(u_eval: np.ndarray) -> tuple[float, int]:
            return 100.0, 50  # 常に reject

        out = proc.process(
            ContactBacktrackingLineSearchInput(
                u=u,
                du=du,
                n_active_pre=0,
                contact_res_pre=1.0,
                compute_trial=_trial,
                max_steps=10,
                active_flip_threshold=0,
                residual_ratio_threshold=2.0,
                alpha_decay=0.5,
                min_alpha=0.125,
            )
        )
        # alpha=1→0.5→0.25→0.125 の時点で次の半減値が min_alpha 未満 → 打切
        assert out.reason == "min_alpha"
        assert out.alpha >= 0.125

    def test_trial_failure_halves_alpha(self):
        """compute_trial の例外発生時は α を半減して継続."""
        proc = ContactBacktrackingLineSearchProcess()
        u = np.zeros(2)
        du = np.ones(2)
        state = {"calls": 0}

        def _trial(u_eval: np.ndarray) -> tuple[float, int]:
            state["calls"] += 1
            if state["calls"] <= 1:
                raise RuntimeError("simulated failure")
            return 1.0, 0

        out = proc.process(
            ContactBacktrackingLineSearchInput(
                u=u,
                du=du,
                n_active_pre=0,
                contact_res_pre=1.0,
                compute_trial=_trial,
                max_steps=4,
                active_flip_threshold=3,
                residual_ratio_threshold=2.0,
                alpha_decay=0.5,
                min_alpha=0.0625,
            )
        )
        assert out.reason == "accepted"
        assert out.alpha == pytest.approx(0.5)


# =====================================================================
# ContactManager Process ラッパーのテスト
# =====================================================================


@binds_to(DetectCandidatesProcess)
class TestDetectCandidatesProcessAPI:
    """DetectCandidatesProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(DetectCandidatesProcess, SolverProcess)

    def test_detect_basic(self):
        """基本的な候補検出."""
        from xkep_cae.contact._contact_pair import _ContactConfigInput, _ContactManagerInput

        manager = _ContactManagerInput(config=_ContactConfigInput(exclude_same_strand=False))
        mesh = _make_two_beam_mesh()
        proc = DetectCandidatesProcess()
        out = proc.process(
            DetectCandidatesInput(
                manager=manager,
                node_coords=mesh.node_coords,
                connectivity=mesh.connectivity,
                radii=mesh.radii,
            )
        )
        assert isinstance(out, DetectCandidatesOutput)
        assert out.n_pairs >= 0


@binds_to(UpdateGeometryProcess)
class TestUpdateGeometryProcessAPI:
    """UpdateGeometryProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(UpdateGeometryProcess, SolverProcess)

    def test_update_basic(self):
        """基本的な幾何更新."""
        from xkep_cae.contact._contact_pair import _ContactConfigInput, _ContactManagerInput

        manager = _ContactManagerInput(config=_ContactConfigInput(exclude_same_strand=False))
        mesh = _make_two_beam_mesh()
        # まず候補検出（Process API 経由）
        detect_out = DetectCandidatesProcess().process(
            DetectCandidatesInput(
                manager=manager,
                node_coords=mesh.node_coords,
                connectivity=mesh.connectivity,
                radii=mesh.radii,
            )
        )
        manager = detect_out.manager
        proc = UpdateGeometryProcess()
        out = proc.process(UpdateGeometryInput(manager=manager, node_coords=mesh.node_coords))
        assert isinstance(out, UpdateGeometryOutput)
        assert out.n_active >= 0


@binds_to(InitializePenaltyProcess)
class TestInitializePenaltyProcessAPI:
    """InitializePenaltyProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(InitializePenaltyProcess, SolverProcess)


class TestChatteringFreezeConfig:
    """接触凍結モード（status-284）のパラメータ設定テスト."""

    def test_default_parameters(self):
        """凍結モードのデフォルト値が正しい."""
        cfg = NewtonDynamicInput()
        assert cfg.chattering_freeze_enabled is True
        assert cfg.chattering_freeze_max_cycles == 5
        assert cfg.chattering_freeze_nr_max == 15
        assert cfg.chattering_freeze_tol_factor == 10.0

    def test_custom_parameters(self):
        """凍結モードパラメータをカスタマイズできる."""
        cfg = NewtonDynamicInput(
            chattering_freeze_enabled=False,
            chattering_freeze_max_cycles=10,
            chattering_freeze_nr_max=20,
            chattering_freeze_tol_factor=50.0,
        )
        assert cfg.chattering_freeze_enabled is False
        assert cfg.chattering_freeze_max_cycles == 10
        assert cfg.chattering_freeze_nr_max == 20
        assert cfg.chattering_freeze_tol_factor == 50.0

    def test_freeze_disabled_does_not_affect_other_params(self):
        """凍結モード無効でも他のNRパラメータは影響しない."""
        cfg = NewtonDynamicInput(chattering_freeze_enabled=False)
        assert cfg.max_attempts == 50
        assert cfg.tol_force == 1e-8
        assert cfg.stall_window == 4

    def test_contact_friction_input_has_freeze_params(self):
        """ContactFrictionInputData に凍結パラメータが定義されている."""
        import dataclasses

        from xkep_cae.core import ContactFrictionInputData

        field_names = {f.name for f in dataclasses.fields(ContactFrictionInputData)}
        assert "chattering_freeze_enabled" in field_names
        assert "chattering_freeze_max_cycles" in field_names
        assert "chattering_freeze_nr_max" in field_names
        assert "chattering_freeze_tol_factor" in field_names
