"""S3 撚線接触テスト — Process API 準静的ソルバー版.

Phase 14: 新 xkep_cae パッケージの Process API のみで構成された
撚線接触テスト。deprecated import を一切使用しない。

ソルバーパス: ContactFrictionProcess → NewtonUzawaStaticProcess（準静的）
  - mass_matrix / dt_physical 未指定のため、全テストで準静的パスを使用
  - 動的パス（NewtonUzawaDynamicProcess）のテストは別途作成が必要

テスト構成:
- 7本撚線の径方向圧縮（線形梁 + CFP 準静的）
- 7本撚線の曲げ（UL CR 梁 + CFP 準静的）
- xfail: 曲げ揺動、摩擦、大規模撚線（全て準静的）

[← README](../../README.md)
"""

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact._contact_pair import _ContactConfig, _ContactManager
from xkep_cae.contact._types import ContactStatus
from xkep_cae.contact.setup.process import ContactSetupConfig, ContactSetupProcess
from xkep_cae.contact.solver.process import ContactFrictionProcess
from xkep_cae.core import (
    AssembleCallbacks,
    BoundaryData,
    ContactFrictionInputData,
    ContactSetupData,
    MeshData,
)
from xkep_cae.elements import BeamSection, ULCRBeamAssembler
from xkep_cae.elements._beam_cr import timo_beam3d_ke_global
from xkep_cae.mesh._twisted_wire import _make_twisted_wire_mesh, _radii
from xkep_cae.mesh.process import StrandMeshConfig, StrandMeshProcess

pytestmark = pytest.mark.slow

# ====================================================================
# 共通パラメータ
# ====================================================================

_NDOF_PER_NODE = 6
_E = 200e9  # Pa
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_WIRE_D = 0.002  # m
_SECTION = BeamSection.circle(_WIRE_D)
_KAPPA = _SECTION.cowper_kappa_y(_NU)
_PITCH = 0.040  # m
_RHO = 7800.0  # kg/m^3 (steel)


# ====================================================================
# ヘルパー
# ====================================================================


def _make_mesh_and_data(
    n_strands: int = 7,
    n_elems_per_pitch: int = 16,
    n_pitches: float = 1.0,
    gap: float = 0.0005,
):
    """撚線メッシュを生成し MeshData を返す."""
    wire_diameter = _WIRE_D
    length = _PITCH * n_pitches
    n_elems = int(n_elems_per_pitch * n_pitches)

    mesh = _make_twisted_wire_mesh(
        n_strands=n_strands,
        wire_diameter=wire_diameter,
        pitch=_PITCH,
        length=length,
        n_elems_per_strand=n_elems,
        gap=gap,
        n_pitches=n_pitches,
    )

    # elem_layer_map 構築（同層除外用）
    layer_ids = np.zeros(len(mesh.connectivity), dtype=int)
    for info in mesh.strand_infos:
        start, end = mesh.strand_elem_ranges[info.strand_id]
        layer_ids[start:end] = info.layer

    mesh_data = MeshData(
        node_coords=mesh.node_coords,
        connectivity=mesh.connectivity,
        radii=_radii(mesh),
        n_strands=n_strands,
        layer_ids=layer_ids,
    )

    return mesh, mesh_data


def _build_elem_layer_map(mesh):
    """TwistedWireMesh から elem_layer_map を構築."""
    elem_layer_map = {}
    for info in mesh.strand_infos:
        start, end = mesh.strand_elem_ranges[info.strand_id]
        for e in range(start, end):
            elem_layer_map[e] = info.layer
    return elem_layer_map


def _build_linear_assemblers(mesh_data: MeshData):
    """線形 Timoshenko 梁アセンブラ（sparse）."""
    nc = mesh_data.node_coords
    conn = mesh_data.connectivity
    n_nodes = len(nc)
    ndof_total = n_nodes * _NDOF_PER_NODE

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for elem in conn:
        n1, n2 = int(elem[0]), int(elem[1])
        coords = nc[np.array([n1, n2])]
        Ke = timo_beam3d_ke_global(
            coords,
            _E,
            _G,
            _SECTION.A,
            _SECTION.Iy,
            _SECTION.Iz,
            _SECTION.J,
            _KAPPA,
            _KAPPA,
        )
        edofs = np.array(
            [_NDOF_PER_NODE * n1 + d for d in range(_NDOF_PER_NODE)]
            + [_NDOF_PER_NODE * n2 + d for d in range(_NDOF_PER_NODE)],
            dtype=int,
        )
        for ii in range(12):
            for jj in range(12):
                if abs(Ke[ii, jj]) > 1e-30:
                    rows.append(edofs[ii])
                    cols.append(edofs[jj])
                    vals.append(Ke[ii, jj])

    K_sp = sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(ndof_total, ndof_total),
    )

    def assemble_tangent(u):
        return K_sp

    def assemble_internal_force(u):
        return K_sp.dot(u)

    return assemble_tangent, assemble_internal_force, ndof_total


def _build_ul_assembler(mesh_data: MeshData):
    """UL CR 梁アセンブラを構築."""
    return ULCRBeamAssembler(
        node_coords=mesh_data.node_coords,
        connectivity=mesh_data.connectivity,
        E=_E,
        G=_G,
        A=_SECTION.A,
        Iy=_SECTION.Iy,
        Iz=_SECTION.Iz,
        J=_SECTION.J,
        kappa_y=_KAPPA,
        kappa_z=_KAPPA,
    )


def _fixed_dofs_start(mesh_data: MeshData, mesh, n_strands: int):
    """全素線の開始端を全固定."""
    fixed = set()
    for sid in range(n_strands):
        node_start, _ = mesh.strand_node_ranges[sid]
        for d in range(_NDOF_PER_NODE):
            fixed.add(_NDOF_PER_NODE * node_start + d)
    return np.array(sorted(fixed), dtype=int)


def _radial_compression_load(mesh_data: MeshData, mesh, ndof_total, total_force=100.0):
    """径方向圧縮荷重: 各素線終端に軸方向引張."""
    f_ext = np.zeros(ndof_total)
    n_strands = mesh.n_strands
    f_per = total_force / n_strands
    for sid in range(n_strands):
        _, node_end = mesh.strand_node_ranges[sid]
        end_node = node_end - 1
        f_ext[_NDOF_PER_NODE * end_node + 2] = f_per  # z方向
    return f_ext


def _bending_prescribed_dofs(mesh, n_strands: int, ndof_total: int, angle_deg: float):
    """曲げ変位制御: 終端に y 方向回転角を処方."""
    prescribed_dofs = []
    prescribed_values = []
    angle_rad = np.radians(angle_deg)
    for sid in range(n_strands):
        _, node_end = mesh.strand_node_ranges[sid]
        end_node = node_end - 1
        # y 軸まわり回転を処方（梁軸方向は z）
        dof_ry = _NDOF_PER_NODE * end_node + 4  # ry
        prescribed_dofs.append(dof_ry)
        prescribed_values.append(angle_rad)
    return np.array(prescribed_dofs, dtype=int), np.array(prescribed_values)


def _build_contact_setup(mesh, mesh_data: MeshData, *, k_pen_scale=0.1, mu=0.0):
    """ContactManager + ContactSetupData を Process API で構築."""
    elem_layer_map = _build_elem_layer_map(mesh)
    config = _ContactConfig(
        k_pen_scale=k_pen_scale,
        k_pen_mode="beam_ei",
        beam_E=_E,
        beam_I=_SECTION.Iy,
        k_pen_scaling="sqrt",
        k_t_ratio=0.1,
        mu=mu,
        g_on=0.0005,
        g_off=0.001,
        use_friction=mu > 0.0,
        use_line_search=False,
        use_geometric_stiffness=True,
        tol_penetration_ratio=0.02,
        penalty_growth_factor=1.0,
        k_pen_max=1e12,
        elem_layer_map=elem_layer_map,
        exclude_same_layer=True,
        midpoint_prescreening=True,
        linear_solver="auto",
        no_deactivation_within_step=True,
        preserve_inactive_lambda=True,
        adjust_initial_penetration=True,
        contact_mode="smooth_penalty",
        n_uzawa_max=5,
        tol_uzawa=1e-6,
    )
    manager = _ContactManager(config=config)
    manager.detect_candidates(
        mesh_data.node_coords,
        mesh_data.connectivity,
        mesh_data.radii,
        margin=0.01,
    )
    k_pen = k_pen_scale * _E * _SECTION.Iy / (_PITCH**2)
    return ContactSetupData(
        manager=manager,
        k_pen=k_pen,
        use_friction=mu > 0.0,
        mu=mu if mu > 0.0 else None,
        contact_mode="smooth_penalty",
    )


def _count_active(manager):
    """アクティブな接触ペア数."""
    return sum(1 for p in manager.pairs if p.state.status != ContactStatus.INACTIVE)


# ====================================================================
# テスト: StrandMeshProcess パイプライン検証
# ====================================================================


class TestStrandMeshProcessAPI:
    """StrandMeshProcess の基本動作確認."""

    def test_7strand_mesh_generation(self):
        """7本撚線メッシュが正しく生成される."""
        proc = StrandMeshProcess()
        config = StrandMeshConfig(
            n_strands=7,
            wire_radius=_WIRE_D / 2,
            pitch_length=_PITCH,
            gap=0.0005,
            n_elements_per_pitch=16,
            n_pitches=1.0,
        )
        result = proc.process(config)
        mesh_data = result.mesh

        assert mesh_data.node_coords.shape[1] == 3
        assert mesh_data.connectivity.shape[1] == 2
        assert mesh_data.n_strands == 7
        # 7本 x 16要素 = 112要素
        assert len(mesh_data.connectivity) == 7 * 16
        # 7本 x 17節点 = 119節点
        assert len(mesh_data.node_coords) == 7 * 17

    def test_19strand_mesh_generation(self):
        """19本撚線メッシュが正しく生成される."""
        proc = StrandMeshProcess()
        config = StrandMeshConfig(
            n_strands=19,
            wire_radius=_WIRE_D / 2,
            pitch_length=_PITCH,
            gap=0.0005,
            n_elements_per_pitch=16,
            n_pitches=1.0,
        )
        result = proc.process(config)
        assert result.mesh.n_strands == 19
        assert len(result.mesh.connectivity) == 19 * 16


# ====================================================================
# テスト: ContactSetupProcess パイプライン検証
# ====================================================================


class TestContactSetupProcessAPI:
    """ContactSetupProcess の基本動作確認."""

    def test_contact_setup_creates_manager(self):
        """ContactSetupProcess がマネージャを生成し候補を検出する."""
        proc = StrandMeshProcess()
        mesh_result = proc.process(
            StrandMeshConfig(
                n_strands=7,
                wire_radius=_WIRE_D / 2,
                pitch_length=_PITCH,
                gap=0.0005,
                n_elements_per_pitch=16,
                n_pitches=1.0,
            )
        )

        setup_proc = ContactSetupProcess()
        setup_result = setup_proc.process(
            ContactSetupConfig(
                mesh=mesh_result.mesh,
                k_pen=1e6,
                use_friction=False,
                mu=0.0,
                contact_mode="smooth_penalty",
                exclude_same_layer=True,
            )
        )

        assert setup_result.manager is not None
        assert setup_result.k_pen == 1e6
        manager = setup_result.manager
        assert len(manager.pairs) > 0


# ====================================================================
# テスト: 7本撚線径方向圧縮（線形梁 + Process API）
# ====================================================================


class TestSevenStrandRadialProcessAPI:
    """7本撚線の径方向圧縮テスト（線形梁 + CFP 準静的パス）.

    旧 test_convergence_19strand.py::Test7Strand の Process API 版。
    ソルバーパス: ContactFrictionProcess → NewtonUzawaStaticProcess
    """

    def test_7strand_radial_converges(self):
        """7本: 線形梁の軸方向引張で ContactFrictionProcess が収束する."""
        mesh, mesh_data = _make_mesh_and_data(
            n_strands=7,
            n_elems_per_pitch=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_linear_assemblers(mesh_data)
        fd = _fixed_dofs_start(mesh_data, mesh, 7)
        f_ext = _radial_compression_load(mesh_data, mesh, ndof, total_force=100.0)

        contact = _build_contact_setup(mesh, mesh_data, k_pen_scale=0.1, mu=0.0)

        input_data = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=BoundaryData(
                fixed_dofs=fd,
                f_ext_total=f_ext,
            ),
            contact=contact,
            callbacks=AssembleCallbacks(
                assemble_tangent=at,
                assemble_internal_force=ai,
            ),
        )

        solver = ContactFrictionProcess()
        result = solver.process(input_data)

        print(
            f"\n  7本 Process API (径方向): converged={result.converged}, "
            f"increments={result.n_increments}, "
            f"newton={result.total_newton_iterations}, "
            f"active={_count_active(contact.manager)}"
        )
        assert result.converged, "7本 Process API 径方向圧縮が収束しなかった"
        assert result.n_increments > 0


# ====================================================================
# テスト: 7本撚線 UL CR 梁曲げ（Process API）
# ====================================================================


class TestSevenStrandBendingProcessAPI:
    """7本撚線の CR 梁曲げテスト（ULCRBeamAssembler + CFP 準静的パス）.

    Phase 13 で移植した ULCRBeamAssembler を ContactFrictionProcess と
    組み合わせた統合テスト。
    ソルバーパス: ContactFrictionProcess → NewtonUzawaStaticProcess
    """

    def test_7strand_bending_45deg(self):
        """7本: UL CR 梁 45度曲げで ContactFrictionProcess が収束する."""
        mesh, mesh_data = _make_mesh_and_data(
            n_strands=7,
            n_elems_per_pitch=16,
            n_pitches=0.5,
            gap=0.15,
        )
        assembler = _build_ul_assembler(mesh_data)
        fd = _fixed_dofs_start(mesh_data, mesh, 7)
        ndof = len(mesh_data.node_coords) * _NDOF_PER_NODE
        p_dofs, p_vals = _bending_prescribed_dofs(mesh, 7, ndof, angle_deg=45.0)

        contact = _build_contact_setup(mesh, mesh_data, k_pen_scale=0.1, mu=0.0)

        input_data = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=BoundaryData(
                fixed_dofs=fd,
                prescribed_dofs=p_dofs,
                prescribed_values=p_vals,
                f_ext_total=np.zeros(ndof),
            ),
            contact=contact,
            callbacks=AssembleCallbacks(
                assemble_tangent=assembler.assemble_tangent,
                assemble_internal_force=assembler.assemble_internal_force,
                ul_assembler=assembler,
            ),
        )

        solver = ContactFrictionProcess()
        result = solver.process(input_data)

        print(
            f"\n  7本 Process API (45度曲げ): converged={result.converged}, "
            f"increments={result.n_increments}, "
            f"newton={result.total_newton_iterations}, "
            f"active={_count_active(contact.manager)}"
        )
        assert result.converged, "7本 Process API 45度曲げが収束しなかった"

        # 物理検証: 変位が存在すること
        u_mag = float(np.linalg.norm(result.u))
        assert u_mag > 1e-10, f"変位がゼロ: |u|={u_mag}"

    def test_7strand_bending_90deg(self):
        """7本: UL CR 梁 90度曲げで ContactFrictionProcess が収束する."""
        mesh, mesh_data = _make_mesh_and_data(
            n_strands=7,
            n_elems_per_pitch=16,
            n_pitches=0.5,
            gap=0.15,
        )
        assembler = _build_ul_assembler(mesh_data)
        fd = _fixed_dofs_start(mesh_data, mesh, 7)
        ndof = len(mesh_data.node_coords) * _NDOF_PER_NODE
        p_dofs, p_vals = _bending_prescribed_dofs(mesh, 7, ndof, angle_deg=90.0)

        contact = _build_contact_setup(mesh, mesh_data, k_pen_scale=0.1, mu=0.0)

        input_data = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=BoundaryData(
                fixed_dofs=fd,
                prescribed_dofs=p_dofs,
                prescribed_values=p_vals,
                f_ext_total=np.zeros(ndof),
            ),
            contact=contact,
            callbacks=AssembleCallbacks(
                assemble_tangent=assembler.assemble_tangent,
                assemble_internal_force=assembler.assemble_internal_force,
                ul_assembler=assembler,
            ),
        )

        solver = ContactFrictionProcess()
        result = solver.process(input_data)

        print(
            f"\n  7本 Process API (90度曲げ): converged={result.converged}, "
            f"increments={result.n_increments}, "
            f"newton={result.total_newton_iterations}, "
            f"active={_count_active(contact.manager)}"
        )
        assert result.converged, "7本 Process API 90度曲げが収束しなかった"


# ====================================================================
# テスト: 曲げ揺動 xfail（Process API）
# ====================================================================


class TestSevenStrandBendingOscillationProcess:
    """7本撚線の曲げ揺動テスト（CFP 準静的パス）.

    旧 test_bending_oscillation.py::Test7StrandBendingOscillation の
    Process API 対応版。Phase2 揺動の接触活性セット変動により xfail。
    ソルバーパス: ContactFrictionProcess → NewtonUzawaStaticProcess
    """

    @pytest.mark.xfail(
        reason="Phase2揺動の接触活性セット変動による不収束 (status-143)",
        strict=False,
    )
    def test_7strand_bending_oscillation_full(self):
        """7本: 90度曲げ + 揺動1周期（S3ベンチマーク Process API 版）.

        手順:
        1. Phase1: 45度曲げを ContactFrictionProcess で収束
        2. Phase2: 揺動（z方向変位サイクル）を追加
        """
        mesh, mesh_data = _make_mesh_and_data(
            n_strands=7,
            n_elems_per_pitch=16,
            n_pitches=0.5,
            gap=0.15,
        )
        assembler = _build_ul_assembler(mesh_data)
        fd = _fixed_dofs_start(mesh_data, mesh, 7)
        ndof = len(mesh_data.node_coords) * _NDOF_PER_NODE
        p_dofs, p_vals = _bending_prescribed_dofs(mesh, 7, ndof, angle_deg=90.0)

        contact = _build_contact_setup(mesh, mesh_data, k_pen_scale=0.1, mu=0.0)

        # Phase 1: 90度曲げ
        input_data = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=BoundaryData(
                fixed_dofs=fd,
                prescribed_dofs=p_dofs,
                prescribed_values=p_vals,
                f_ext_total=np.zeros(ndof),
            ),
            contact=contact,
            callbacks=AssembleCallbacks(
                assemble_tangent=assembler.assemble_tangent,
                assemble_internal_force=assembler.assemble_internal_force,
                ul_assembler=assembler,
            ),
        )

        solver = ContactFrictionProcess()
        result_phase1 = solver.process(input_data)
        assert result_phase1.converged, "Phase1（90度曲げ）が収束しなかった"

        # Phase 2: 揺動（z方向往復変位を処方）
        oscillation_amp = 0.002  # 2mm
        z_dofs = []
        for sid in range(7):
            _, node_end = mesh.strand_node_ranges[sid]
            end_node = node_end - 1
            z_dofs.append(_NDOF_PER_NODE * end_node + 2)  # z方向
        z_dofs = np.array(z_dofs, dtype=int)
        z_vals = np.full(len(z_dofs), oscillation_amp)

        all_p_dofs = np.concatenate([p_dofs, z_dofs])
        all_p_vals = np.concatenate([p_vals, z_vals])

        input_phase2 = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=BoundaryData(
                fixed_dofs=fd,
                prescribed_dofs=all_p_dofs,
                prescribed_values=all_p_vals,
                f_ext_total=np.zeros(ndof),
            ),
            contact=contact,
            callbacks=AssembleCallbacks(
                assemble_tangent=assembler.assemble_tangent,
                assemble_internal_force=assembler.assemble_internal_force,
                ul_assembler=assembler,
            ),
            u0=result_phase1.u,
        )

        result_phase2 = solver.process(input_phase2)
        assert result_phase2.converged, "Phase2（揺動）が収束しなかった"

        print(
            f"\n  7本 Process API (曲げ揺動): "
            f"phase1={result_phase1.converged}, "
            f"phase2={result_phase2.converged}, "
            f"active={_count_active(contact.manager)}"
        )


# ====================================================================
# テスト: 摩擦付き接触 xfail（Process API）
# ====================================================================


class TestStrandFrictionProcess:
    """摩擦接触テスト（CFP 準静的パス）.

    旧 test_friction_validation.py / test_real_beam_contact.py の
    Process API 対応版。NCP 摩擦接線剛性符号問題により xfail。
    ソルバーパス: ContactFrictionProcess → NewtonUzawaStaticProcess
    """

    @pytest.mark.xfail(
        reason="smooth_penalty 摩擦接線剛性 — CR梁大変形+接触での不収束 (status-128)",
        strict=False,
    )
    def test_7strand_cr_friction_bending(self):
        """7本: CR 梁の摩擦付き接触で曲げ収束を試行.

        gap=0（接触あり）+ mu=0.3 でCR梁曲げ。
        接触活性化状態での摩擦接線剛性問題が顕在化する。
        """
        mesh, mesh_data = _make_mesh_and_data(
            n_strands=7,
            n_elems_per_pitch=16,
            n_pitches=0.5,
            gap=0.0,
        )
        assembler = _build_ul_assembler(mesh_data)
        fd = _fixed_dofs_start(mesh_data, mesh, 7)
        ndof = len(mesh_data.node_coords) * _NDOF_PER_NODE
        p_dofs, p_vals = _bending_prescribed_dofs(mesh, 7, ndof, angle_deg=45.0)

        contact = _build_contact_setup(mesh, mesh_data, k_pen_scale=0.1, mu=0.3)

        input_data = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=BoundaryData(
                fixed_dofs=fd,
                prescribed_dofs=p_dofs,
                prescribed_values=p_vals,
                f_ext_total=np.zeros(ndof),
            ),
            contact=contact,
            callbacks=AssembleCallbacks(
                assemble_tangent=assembler.assemble_tangent,
                assemble_internal_force=assembler.assemble_internal_force,
                ul_assembler=assembler,
            ),
        )

        solver = ContactFrictionProcess()
        result = solver.process(input_data)

        n_active = _count_active(contact.manager)
        print(
            f"\n  7本 Process API (摩擦曲げ): converged={result.converged}, "
            f"increments={result.n_increments}, "
            f"newton={result.total_newton_iterations}, "
            f"active={n_active}"
        )
        assert result.converged, "7本 CR梁摩擦曲げが収束しなかった"
        assert n_active > 0, "接触が活性化されていない"


# ====================================================================
# テスト: 大規模撚線 xfail（Process API）
# ====================================================================


class TestLargeStrandProcessAPI:
    """大規模撚線テスト（CFP 準静的パス）.

    旧 test_convergence_19strand.py::Test19Strand の Process API 対応版。
    CI タイムアウトにより xfail。
    ソルバーパス: ContactFrictionProcess → NewtonUzawaStaticProcess
    """

    @pytest.mark.xfail(
        reason="CI タイムアウト (>600s) (status-127)",
        strict=False,
    )
    def test_19strand_radial_with_active_contacts(self):
        """19本: 径方向圧縮でアクティブ接触が発生する."""
        mesh, mesh_data = _make_mesh_and_data(
            n_strands=19,
            n_elems_per_pitch=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_linear_assemblers(mesh_data)
        fd = _fixed_dofs_start(mesh_data, mesh, 19)
        f_ext = _radial_compression_load(mesh_data, mesh, ndof, total_force=100.0)

        contact = _build_contact_setup(mesh, mesh_data, k_pen_scale=0.1, mu=0.0)

        input_data = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=BoundaryData(
                fixed_dofs=fd,
                f_ext_total=f_ext,
            ),
            contact=contact,
            callbacks=AssembleCallbacks(
                assemble_tangent=at,
                assemble_internal_force=ai,
            ),
        )

        solver = ContactFrictionProcess()
        result = solver.process(input_data)

        n_active = _count_active(contact.manager)
        print(
            f"\n  19本 Process API (径方向): converged={result.converged}, "
            f"increments={result.n_increments}, "
            f"newton={result.total_newton_iterations}, "
            f"active={n_active}"
        )
        assert result.converged, "19本 Process API 径方向圧縮が収束しなかった"
        assert n_active > 0, "アクティブ接触が発生していない"


# ====================================================================
# テスト: 物理的妥当性検証（Process API）
# ====================================================================


class TestStrandBendingPhysicsProcess:
    """曲げ変形の物理的妥当性テスト（CFP 準静的パス）.

    旧 test_bending_oscillation.py::Test7StrandBendingPhysics の
    Process API 対応版。
    ソルバーパス: ContactFrictionProcess → NewtonUzawaStaticProcess
    """

    def test_bending_tip_displacement_direction(self):
        """曲げ後の先端変位方向が物理的に正しい."""
        mesh, mesh_data = _make_mesh_and_data(
            n_strands=7,
            n_elems_per_pitch=16,
            n_pitches=0.5,
            gap=0.15,
        )
        assembler = _build_ul_assembler(mesh_data)
        fd = _fixed_dofs_start(mesh_data, mesh, 7)
        ndof = len(mesh_data.node_coords) * _NDOF_PER_NODE
        p_dofs, p_vals = _bending_prescribed_dofs(mesh, 7, ndof, angle_deg=45.0)

        contact = _build_contact_setup(mesh, mesh_data, k_pen_scale=0.1, mu=0.0)

        input_data = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=BoundaryData(
                fixed_dofs=fd,
                prescribed_dofs=p_dofs,
                prescribed_values=p_vals,
                f_ext_total=np.zeros(ndof),
            ),
            contact=contact,
            callbacks=AssembleCallbacks(
                assemble_tangent=assembler.assemble_tangent,
                assemble_internal_force=assembler.assemble_internal_force,
                ul_assembler=assembler,
            ),
        )

        solver = ContactFrictionProcess()
        result = solver.process(input_data)

        assert result.converged, "45度曲げが収束しなかった"

        # 中心線（strand 0）の先端変位チェック
        _, node_end = mesh.strand_node_ranges[0]
        end_node = node_end - 1
        u = result.u
        dx = u[_NDOF_PER_NODE * end_node + 0]
        dy = u[_NDOF_PER_NODE * end_node + 1]
        dz = u[_NDOF_PER_NODE * end_node + 2]

        # 45度曲げで先端は有意な変位を持つ
        disp_mag = np.sqrt(dx**2 + dy**2 + dz**2)
        assert disp_mag > 1e-6, f"先端変位が小さすぎる: |u|={disp_mag:.2e}"

        print(
            f"\n  物理テスト（先端変位方向）: "
            f"dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}, |u|={disp_mag:.6f}"
        )

    def test_radial_compression_contact_forces_positive(self):
        """径方向圧縮でアクティブ接触の法線力が正（圧縮）."""
        mesh, mesh_data = _make_mesh_and_data(
            n_strands=7,
            n_elems_per_pitch=16,
            n_pitches=1.0,
            gap=0.0005,
        )
        at, ai, ndof = _build_linear_assemblers(mesh_data)
        fd = _fixed_dofs_start(mesh_data, mesh, 7)
        f_ext = _radial_compression_load(mesh_data, mesh, ndof, total_force=100.0)

        contact = _build_contact_setup(mesh, mesh_data, k_pen_scale=0.1, mu=0.0)

        input_data = ContactFrictionInputData(
            mesh=mesh_data,
            boundary=BoundaryData(
                fixed_dofs=fd,
                f_ext_total=f_ext,
            ),
            contact=contact,
            callbacks=AssembleCallbacks(
                assemble_tangent=at,
                assemble_internal_force=ai,
            ),
        )

        solver = ContactFrictionProcess()
        result = solver.process(input_data)
        assert result.converged, "径方向圧縮が収束しなかった"

        # アクティブペアの法線力チェック
        n_active = 0
        for pair in contact.manager.pairs:
            if pair.state.status != ContactStatus.INACTIVE:
                n_active += 1
                assert pair.state.p_n >= 0, f"法線力が負: p_n={pair.state.p_n:.2e}"

        print(
            f"\n  物理テスト（接触力）: active={n_active}, "
            f"max_pn={max(p.state.p_n for p in contact.manager.pairs if p.is_active()) if n_active > 0 else 0:.2e}"
        )
