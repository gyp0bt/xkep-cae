"""ContactNormalDampingProcess ユニットテスト (status-365 Phase 1).

[← damping README](../__init__.py) | [← project README](../../../../README.md)

解析解との比較で f_damp / K_damp / energy_rate の物理整合性を検証する。
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from xkep_cae.contact._contact_pair import _ContactPairOutput, _ContactStateOutput
from xkep_cae.contact._types import ContactStatus
from xkep_cae.contact.damping import (
    ContactNormalDampingInput,
    ContactNormalDampingOutput,
    ContactNormalDampingProcess,
)
from xkep_cae.core.testing import binds_to


def _make_pair(
    nodes_a: tuple[int, int],
    nodes_b: tuple[int, int],
    *,
    s: float = 0.5,
    t: float = 0.5,
    normal: np.ndarray | None = None,
    status: ContactStatus = ContactStatus.ACTIVE,
) -> _ContactPairOutput:
    if normal is None:
        normal = np.array([1.0, 0.0, 0.0])
    state = _ContactStateOutput(s=s, t=t, normal=normal, status=status)
    return _ContactPairOutput(
        elem_a=0,
        elem_b=1,
        nodes_a=np.asarray(nodes_a, dtype=int),
        nodes_b=np.asarray(nodes_b, dtype=int),
        state=state,
    )


def _make_velocity(
    ndof_total: int,
    node_vel: dict[int, np.ndarray],
    *,
    ndof_per_node: int = 6,
) -> np.ndarray:
    vel = np.zeros(ndof_total)
    for node, v in node_vel.items():
        base = node * ndof_per_node
        vel[base : base + 3] = v
    return vel


@binds_to(ContactNormalDampingProcess)
class TestContactNormalDampingProcessAPI:
    """Process API 契約（no-op / 型 / 返却形状）の検証."""

    def test_zero_c_n_returns_empty(self) -> None:
        """c_n=0 で f_damp/K_damp ゼロ、energy_rate=0、n_active_pairs=0."""
        pair = _make_pair((0, 1), (2, 3))
        vel = _make_velocity(24, {0: np.array([1.0, 0.0, 0.0])})
        inp = ContactNormalDampingInput(
            pairs=[pair],
            velocity=vel,
            c_n=0.0,
            stiffness_factor=100.0,
            ndof_total=24,
        )
        out = ContactNormalDampingProcess().process(inp)
        assert isinstance(out, ContactNormalDampingOutput)
        assert np.allclose(out.f_damp, 0.0)
        assert out.K_damp.nnz == 0
        assert out.energy_rate == 0.0
        assert out.n_active_pairs == 0

    def test_empty_pairs_returns_empty(self) -> None:
        """active ペアなし → 全ゼロ."""
        inp = ContactNormalDampingInput(
            pairs=[],
            velocity=np.zeros(24),
            c_n=1.0,
            stiffness_factor=100.0,
            ndof_total=24,
        )
        out = ContactNormalDampingProcess().process(inp)
        assert np.allclose(out.f_damp, 0.0)
        assert out.K_damp.nnz == 0
        assert out.energy_rate == 0.0
        assert out.n_active_pairs == 0

    def test_inactive_pair_skipped(self) -> None:
        """INACTIVE ペアは無視される."""
        pair = _make_pair((0, 1), (2, 3), status=ContactStatus.INACTIVE)
        vel = _make_velocity(24, {0: np.array([5.0, 0.0, 0.0])})
        inp = ContactNormalDampingInput(
            pairs=[pair],
            velocity=vel,
            c_n=1.0,
            stiffness_factor=100.0,
            ndof_total=24,
        )
        out = ContactNormalDampingProcess().process(inp)
        assert np.allclose(out.f_damp, 0.0)
        assert out.K_damp.nnz == 0
        assert out.n_active_pairs == 0


class TestContactNormalDampingProcessPhysics:
    """単一ペア解析解との比較による物理整合性検証."""

    def test_single_pair_zero_velocity(self) -> None:
        """速度ゼロ → f_damp=0、K_damp は対称半正定値で非ゼロ."""
        pair = _make_pair((0, 1), (2, 3))
        vel = np.zeros(24)
        inp = ContactNormalDampingInput(
            pairs=[pair],
            velocity=vel,
            c_n=2.5,
            stiffness_factor=100.0,
            ndof_total=24,
        )
        out = ContactNormalDampingProcess().process(inp)
        # 速度ゼロ → 減衰力ゼロ
        assert np.allclose(out.f_damp, 0.0)
        assert out.energy_rate == 0.0
        # K_damp は速度に依らず outer product で構築され非ゼロ
        assert out.K_damp.nnz > 0
        # 対称性
        K_dense = out.K_damp.toarray()
        assert np.allclose(K_dense, K_dense.T)

    def test_single_pair_closing_velocity_analytic(self) -> None:
        """s=t=0.5、n̂=(1,0,0)、B 側のみ負 x 速度で解析解に一致.

        v_n = g_shape·v_local
        coeffs = [0.5, 0.5, -0.5, -0.5]
        v_local (変位成分) = [A0:0, A1:0, B0:(-v,0,0), B1:(-v,0,0)]
        v_n = -0.5·(-v) + -0.5·(-v) = v（closing; 両者が近づく向き）
        f_damp = -c_n·v_n·g_shape
        A0 x: -c_n·v·0.5·1 = -0.5 c_n v
        A1 x: -c_n·v·0.5·1 = -0.5 c_n v
        B0 x: -c_n·v·(-0.5)·1 = +0.5 c_n v
        B1 x: -c_n·v·(-0.5)·1 = +0.5 c_n v
        """
        c_n = 3.0
        v_mag = 2.0
        ndof = 24
        pair = _make_pair((0, 1), (2, 3), s=0.5, t=0.5, normal=np.array([1.0, 0.0, 0.0]))
        vel = _make_velocity(
            ndof,
            {
                2: np.array([-v_mag, 0.0, 0.0]),
                3: np.array([-v_mag, 0.0, 0.0]),
            },
        )
        inp = ContactNormalDampingInput(
            pairs=[pair],
            velocity=vel,
            c_n=c_n,
            stiffness_factor=0.0,  # K_damp 無効化して f_damp だけ検証
            ndof_total=ndof,
        )
        out = ContactNormalDampingProcess().process(inp)

        expected_v_n = v_mag
        # A 側 x DOF（負）
        assert out.f_damp[0] == pytest.approx(-0.5 * c_n * expected_v_n)
        assert out.f_damp[6] == pytest.approx(-0.5 * c_n * expected_v_n)
        # B 側 x DOF（正）
        assert out.f_damp[12] == pytest.approx(+0.5 * c_n * expected_v_n)
        assert out.f_damp[18] == pytest.approx(+0.5 * c_n * expected_v_n)
        # y / z 成分は 0
        assert out.f_damp[1] == 0.0
        assert out.f_damp[2] == 0.0
        # energy_rate = c_n * v_n²
        assert out.energy_rate == pytest.approx(c_n * expected_v_n * expected_v_n)
        assert out.n_active_pairs == 1

    def test_single_pair_separating_velocity_force_reverses(self) -> None:
        """separating (v_n < 0) では f_damp が逆向き（減衰は常に速度反対）."""
        c_n = 1.0
        v_mag = 1.0
        ndof = 24
        pair = _make_pair((0, 1), (2, 3), s=0.5, t=0.5, normal=np.array([1.0, 0.0, 0.0]))
        vel = _make_velocity(
            ndof,
            {
                2: np.array([+v_mag, 0.0, 0.0]),
                3: np.array([+v_mag, 0.0, 0.0]),
            },
        )
        inp = ContactNormalDampingInput(
            pairs=[pair],
            velocity=vel,
            c_n=c_n,
            stiffness_factor=0.0,
            ndof_total=ndof,
        )
        out = ContactNormalDampingProcess().process(inp)
        # separating (v_n < 0) で A 側 +x、B 側 -x に向く
        assert out.f_damp[0] > 0
        assert out.f_damp[12] < 0
        # energy_rate は符号に依らず正（= c_n * v_n²）
        assert out.energy_rate > 0

    def test_K_damp_outer_product_structure(self) -> None:
        """K_damp_local = c_n * c1 * (g_shape ⊗ g_shape) の構造検証.

        速度ゼロ + c_n·c1≠0 で K_damp を直接チェック。12x12 block が rank-1。
        """
        c_n = 2.0
        c1 = 50.0
        ndof = 24
        pair = _make_pair((0, 1), (2, 3), s=0.3, t=0.7, normal=np.array([0.0, 1.0, 0.0]))
        inp = ContactNormalDampingInput(
            pairs=[pair],
            velocity=np.zeros(ndof),
            c_n=c_n,
            stiffness_factor=c1,
            ndof_total=ndof,
        )
        out = ContactNormalDampingProcess().process(inp)

        # 期待 g_shape 構築（s=0.3, t=0.7, n̂=(0,1,0)）
        coeffs = np.array([1.0 - 0.3, 0.3, -(1.0 - 0.7), -0.7])
        n_hat = np.array([0.0, 1.0, 0.0])
        g_shape = np.empty(12)
        for k in range(4):
            g_shape[k * 3 : k * 3 + 3] = coeffs[k] * n_hat
        K_local_expected = c_n * c1 * np.outer(g_shape, g_shape)

        # ノード 0, 1, 2, 3 の先頭 3 DOF に対応する 12×12 ブロックを抽出
        ndpn = 6
        dofs = np.array([node * ndpn + comp for node in (0, 1, 2, 3) for comp in range(3)])
        K_block = np.asarray(out.K_damp[dofs, :][:, dofs].todense())
        assert np.allclose(K_block, K_local_expected)

        # 対称性
        assert np.allclose(K_block, K_block.T)
        # 半正定値性（rank-1 なので固有値は {0, ..., 0, g_shape·g_shape·c_n·c1}）
        eigvals = np.linalg.eigvalsh(K_block)
        assert eigvals.min() >= -1e-10  # 数値誤差許容

    def test_multiple_pairs_superposition(self) -> None:
        """複数ペアは加算的（superposition）."""
        c_n = 1.5
        ndof = 48
        pair1 = _make_pair((0, 1), (2, 3), s=0.5, t=0.5)
        pair2 = _make_pair((4, 5), (6, 7), s=0.5, t=0.5)
        vel = _make_velocity(
            ndof,
            {
                2: np.array([-1.0, 0.0, 0.0]),
                3: np.array([-1.0, 0.0, 0.0]),
                6: np.array([-2.0, 0.0, 0.0]),
                7: np.array([-2.0, 0.0, 0.0]),
            },
        )
        inp_both = ContactNormalDampingInput(
            pairs=[pair1, pair2],
            velocity=vel,
            c_n=c_n,
            stiffness_factor=10.0,
            ndof_total=ndof,
        )
        inp1 = ContactNormalDampingInput(
            pairs=[pair1],
            velocity=vel,
            c_n=c_n,
            stiffness_factor=10.0,
            ndof_total=ndof,
        )
        inp2 = ContactNormalDampingInput(
            pairs=[pair2],
            velocity=vel,
            c_n=c_n,
            stiffness_factor=10.0,
            ndof_total=ndof,
        )
        out_both = ContactNormalDampingProcess().process(inp_both)
        out1 = ContactNormalDampingProcess().process(inp1)
        out2 = ContactNormalDampingProcess().process(inp2)

        assert np.allclose(out_both.f_damp, out1.f_damp + out2.f_damp)
        assert np.allclose(
            out_both.K_damp.toarray(),
            out1.K_damp.toarray() + out2.K_damp.toarray(),
        )
        assert out_both.energy_rate == pytest.approx(out1.energy_rate + out2.energy_rate)
        assert out_both.n_active_pairs == 2

    def test_energy_rate_always_nonnegative(self) -> None:
        """energy_rate = Σ c_n v_n² は任意速度で ≥ 0（散逸性）."""
        rng = np.random.default_rng(42)
        ndof = 24
        pair = _make_pair((0, 1), (2, 3))
        for _ in range(20):
            vel = rng.normal(size=ndof) * 5.0
            inp = ContactNormalDampingInput(
                pairs=[pair],
                velocity=vel,
                c_n=abs(float(rng.normal())) + 0.01,
                stiffness_factor=100.0,
                ndof_total=ndof,
            )
            out = ContactNormalDampingProcess().process(inp)
            assert out.energy_rate >= 0.0

    def test_tangential_velocity_no_damping(self) -> None:
        """法線と直交する速度では v_n=0、f_damp=0、energy_rate=0."""
        c_n = 3.0
        ndof = 24
        # n̂ = (1,0,0)、速度は y 方向のみ（法線方向成分なし）
        pair = _make_pair((0, 1), (2, 3), normal=np.array([1.0, 0.0, 0.0]))
        vel = _make_velocity(
            ndof,
            {
                2: np.array([0.0, 5.0, 0.0]),
                3: np.array([0.0, 5.0, 0.0]),
            },
        )
        inp = ContactNormalDampingInput(
            pairs=[pair],
            velocity=vel,
            c_n=c_n,
            stiffness_factor=50.0,
            ndof_total=ndof,
        )
        out = ContactNormalDampingProcess().process(inp)
        assert np.allclose(out.f_damp, 0.0)
        assert out.energy_rate == pytest.approx(0.0, abs=1e-12)


class TestContactNormalDampingProcessTangent:
    """K_damp と ∂f_damp/∂u との FD 整合性（stiffness_factor=c1 が速度-変位感度）.

    v = c1 * u + const の仮定下で K_damp = -∂f_damp/∂u = c_n·c1·g_shape⊗g_shape
    となることを有限差分で検証する（u を微小変化させて v = c1*u と解釈）。
    """

    def test_tangent_matches_fd_under_v_is_c1_u(self) -> None:
        c_n = 1.2
        c1 = 25.0
        ndof = 24
        eps = 1e-6
        pair = _make_pair((0, 1), (2, 3), s=0.4, t=0.6, normal=np.array([1.0, 0.0, 0.0]))

        # u_ref（任意の基準変位、ここでは簡便にゼロ）に対し、v = c1 * u
        def f_damp_of_u(u: np.ndarray) -> np.ndarray:
            v = c1 * u
            inp = ContactNormalDampingInput(
                pairs=[pair],
                velocity=v,
                c_n=c_n,
                stiffness_factor=c1,
                ndof_total=ndof,
            )
            return ContactNormalDampingProcess().process(inp).f_damp

        u_ref = np.zeros(ndof)
        out_ref = ContactNormalDampingProcess().process(
            ContactNormalDampingInput(
                pairs=[pair],
                velocity=c1 * u_ref,
                c_n=c_n,
                stiffness_factor=c1,
                ndof_total=ndof,
            )
        )
        K_provided = out_ref.K_damp.toarray()

        # ペアに関係する DOF だけ FD
        ndpn = 6
        pair_dofs = [node * ndpn + comp for node in (0, 1, 2, 3) for comp in range(3)]
        # FD Jacobian J[i, j] = (f_i(u + eps e_j) - f_i(u)) / eps
        # v = c1 * u なので ∂f/∂u = ∂f/∂v · c1
        J_fd = np.zeros((ndof, ndof))
        f_ref = f_damp_of_u(u_ref)
        for j in pair_dofs:
            u_pert = u_ref.copy()
            u_pert[j] = eps
            f_pert = f_damp_of_u(u_pert)
            J_fd[:, j] = (f_pert - f_ref) / eps
        # 減衰接線は K_damp = -∂f_damp/∂u（残差加算側の符号規約）
        # なので K_provided ≈ -J_fd（ペア DOF サブブロック）
        for i in pair_dofs:
            for j in pair_dofs:
                assert K_provided[i, j] == pytest.approx(-J_fd[i, j], abs=1e-5, rel=1e-5)


class TestContactNormalDampingOutputTypes:
    """Output 型の厳密チェック."""

    def test_output_types(self) -> None:
        pair = _make_pair((0, 1), (2, 3))
        vel = np.zeros(24)
        inp = ContactNormalDampingInput(
            pairs=[pair],
            velocity=vel,
            c_n=1.0,
            stiffness_factor=10.0,
            ndof_total=24,
        )
        out = ContactNormalDampingProcess().process(inp)
        assert isinstance(out.f_damp, np.ndarray)
        assert out.f_damp.shape == (24,)
        assert sp.issparse(out.K_damp)
        assert out.K_damp.shape == (24, 24)
        assert isinstance(out.energy_rate, float)
        assert isinstance(out.n_active_pairs, int)
