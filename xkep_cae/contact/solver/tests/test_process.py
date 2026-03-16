"""ContactFrictionProcess のテスト."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.solver._adaptive_stepping import (
    AdaptiveStepInput,
    AdaptiveStepOutput,
    AdaptiveSteppingConfig,
    AdaptiveSteppingProcess,
    StepAction,
)
from xkep_cae.contact.solver._newton_uzawa import NewtonUzawaProcess
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
        use_friction=True,
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
        assert ContactFrictionProcess.meta.version == "1.0.0"

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


@binds_to(NewtonUzawaProcess)
class TestNewtonUzawaProcessAPI:
    """NewtonUzawaProcess の API テスト."""

    def test_is_solver_process(self):
        proc = NewtonUzawaProcess()
        assert isinstance(proc, SolverProcess)

    def test_meta_name(self):
        assert NewtonUzawaProcess.meta.name == "NewtonUzawa"

    def test_meta_module(self):
        assert NewtonUzawaProcess.meta.module == "solve"


@binds_to(AdaptiveSteppingProcess)
class TestAdaptiveSteppingProcessAPI:
    """AdaptiveSteppingProcess の API テスト."""

    def test_is_solver_process(self):
        config = AdaptiveSteppingConfig(dt_initial_fraction=0.5)
        proc = AdaptiveSteppingProcess(config)
        assert isinstance(proc, SolverProcess)

    def test_meta_name(self):
        assert AdaptiveSteppingProcess.meta.name == "AdaptiveStepping"

    def test_meta_module(self):
        assert AdaptiveSteppingProcess.meta.module == "solve"

    def test_query_returns_output(self):
        config = AdaptiveSteppingConfig(dt_initial_fraction=0.5)
        proc = AdaptiveSteppingProcess(config)
        out = proc.process(AdaptiveStepInput(action=StepAction.QUERY, load_frac_prev=0.0))
        assert isinstance(out, AdaptiveStepOutput)
        assert out.has_more_steps is True
        assert out.next_load_frac > 0.0

    def test_full_cycle(self):
        """QUERY → SUCCESS → QUERY で完了まで回る."""
        config = AdaptiveSteppingConfig(dt_initial_fraction=1.0)
        proc = AdaptiveSteppingProcess(config)

        out = proc.process(AdaptiveStepInput(action=StepAction.QUERY, load_frac_prev=0.0))
        assert out.next_load_frac == 1.0

        out = proc.process(
            AdaptiveStepInput(
                action=StepAction.SUCCESS,
                load_frac=1.0,
                load_frac_prev=0.0,
                n_iters=3,
            )
        )
        assert out.has_more_steps is False

    def test_failure_triggers_retry(self):
        """FAILURE でカットバック → can_retry=True."""
        config = AdaptiveSteppingConfig(dt_initial_fraction=0.5)
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
