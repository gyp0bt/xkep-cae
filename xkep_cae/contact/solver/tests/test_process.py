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
from xkep_cae.contact.solver._newton_uzawa_dynamic import NewtonDynamicProcess
from xkep_cae.contact.solver._nuzawa_steps import (
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
        layer_ids=np.array([0] * (n_nodes_per_strand - 1) + [1] * (n_nodes_per_strand - 1)),
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
        exclude_same_layer=True,
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


@binds_to(TangentAssemblyProcess)
class TestTangentAssemblyProcessAPI:
    """TangentAssemblyProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(TangentAssemblyProcess, SolverProcess)


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

        manager = _ContactManagerInput(config=_ContactConfigInput(exclude_same_layer=False))
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

        manager = _ContactManagerInput(config=_ContactConfigInput(exclude_same_layer=False))
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
