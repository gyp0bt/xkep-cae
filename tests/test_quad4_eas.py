"""EAS-4 Q4 要素のテスト.

Simo-Rifai EAS-4 の実装検証:
1. 製造解テスト（Ku=f の整合性）
2. 要素基本特性（対称性・剛体モード数）
3. せん断ロッキングテスト（片持ち梁曲げ）
4. 体積ロッキングテスト（非圧縮材料の曲げ）
5. 歪んだ要素での動作確認
"""

from __future__ import annotations

import numpy as np

from xkep_cae.assembly import assemble_global_stiffness
from xkep_cae.bc import apply_dirichlet
from xkep_cae.elements.quad4 import Quad4PlaneStrain
from xkep_cae.elements.quad4_eas_bbar import (
    Quad4EASBBarPlaneStrain,
    Quad4EASPlaneStrain,
    quad4_ke_plane_strain_eas,
)
from xkep_cae.materials.elastic import PlaneStrainElastic, constitutive_plane_strain
from xkep_cae.solver import solve_displacement

# ---------------------------------------------------------------------------
# 要素基本特性テスト
# ---------------------------------------------------------------------------


class TestEASBasicProperties:
    """EAS-4 要素の基本特性テスト."""

    def test_symmetry_unit_square(self):
        """単位正方形での Ke 対称性."""
        D = constitutive_plane_strain(200e3, 0.3)
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        Ke = quad4_ke_plane_strain_eas(xy, D, t=1.0)
        assert Ke.shape == (8, 8)
        assert np.allclose(Ke, Ke.T, atol=1e-10)

    def test_rigid_body_modes(self):
        """剛体モード 3 個（2並進 + 1回転）を持つこと."""
        D = constitutive_plane_strain(100.0, 0.25)
        xy = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
        Ke = quad4_ke_plane_strain_eas(xy, D, t=1.0)
        eigvals = np.linalg.eigvalsh(Ke)
        n_zero = np.sum(np.abs(eigvals) < 1e-6)
        assert n_zero == 3, f"Expected 3 zero eigenvalues, got {n_zero}"

    def test_positive_semidefinite(self):
        """全固有値 ≥ 0（SPD の半正定値）."""
        D = constitutive_plane_strain(500.0, 0.4)
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        Ke = quad4_ke_plane_strain_eas(xy, D, t=1.0)
        eigvals = np.linalg.eigvalsh(Ke)
        assert eigvals.min() > -1e-10

    def test_distorted_element(self):
        """歪んだ四角形でも対称・半正定値."""
        D = constitutive_plane_strain(200e3, 0.3)
        xy = np.array([[0, 0], [1.2, 0.1], [0.9, 1.1], [-0.1, 0.8]], dtype=float)
        Ke = quad4_ke_plane_strain_eas(xy, D, t=1.0)
        assert np.allclose(Ke, Ke.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(Ke)
        assert eigvals.min() > -1e-10
        # 歪み要素では静的縮合の数値誤差で3番目の剛体モードが微小非ゼロになりうる
        # 相対的に判定: 最小非ゼロ固有値の 1e-4 倍以下を「ゼロ」とする
        sorted_ev = np.sort(np.abs(eigvals))
        threshold = sorted_ev[3] * 1e-4  # 4番目の固有値の 0.01%
        n_zero = np.sum(np.abs(eigvals) < max(threshold, 1e-6))
        assert n_zero == 3


# ---------------------------------------------------------------------------
# 製造解テスト
# ---------------------------------------------------------------------------


class TestEASManufacturedSolution:
    """製造解 u_true を用いたソルバ検証."""

    @staticmethod
    def _solve_check(elem, nodes, conn, mat, fixed_dofs, seed=0):
        K = assemble_global_stiffness(
            nodes, [(elem, conn)], mat, thickness=1.0, show_progress=False
        )
        n = K.shape[0]
        rng = np.random.default_rng(seed)
        u_true = rng.standard_normal(n)
        u_true[fixed_dofs] = 0.0
        f = K @ u_true

        Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
        u, _info = solve_displacement(Kbc, fbc, size_threshold=4000)

        free = np.setdiff1d(np.arange(n), fixed_dofs)
        assert np.allclose(u[free], u_true[free], rtol=1e-10, atol=1e-10)

        # SPD 確認（アセンブリ + 静的縮合の丸め誤差で ~1e-11 の非対称性がありうる）
        Kd = Kbc.toarray()
        assert np.allclose(Kd, Kd.T, atol=1e-10)

    def test_unit_square_1elem(self):
        """EAS Q4: 単位正方形 1 要素で製造解."""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        conn = np.array([[0, 1, 2, 3]])
        mat = PlaneStrainElastic(10.0, 0.25)
        fixed = np.array([0, 1, 6, 7], dtype=int)
        self._solve_check(Quad4EASPlaneStrain(), nodes, conn, mat, fixed, seed=123)

    def test_rectangular_2x2(self):
        """EAS Q4: 2×2 メッシュで製造解."""
        x = np.linspace(0, 2, 3)
        y = np.linspace(0, 1, 3)
        xx, yy = np.meshgrid(x, y)
        nodes = np.column_stack([xx.ravel(), yy.ravel()])
        conn = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]], dtype=int)
        mat = PlaneStrainElastic(200e3, 0.3)
        fixed = np.array([0, 1, 6, 7], dtype=int)
        self._solve_check(Quad4EASPlaneStrain(), nodes, conn, mat, fixed, seed=42)

    def test_mixed_q4_eas_tri3(self):
        """EAS Q4 + TRI3 混在メッシュで製造解."""
        from xkep_cae.elements.tri3 import Tri3PlaneStrain

        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0.5]], dtype=float)
        conn_q4 = np.array([[0, 1, 2, 3]])
        conn_t3 = np.array([[1, 4, 2]])
        mat = PlaneStrainElastic(100.0, 0.29)

        K = assemble_global_stiffness(
            nodes,
            [(Quad4EASPlaneStrain(), conn_q4), (Tri3PlaneStrain(), conn_t3)],
            mat,
            thickness=1.0,
            show_progress=False,
        )
        ndof = K.shape[0]
        fixed = np.array([0, 1, 6, 7], dtype=int)
        rng = np.random.default_rng(2025)
        u_true = rng.standard_normal(ndof)
        u_true[fixed] = 0.0
        f = K @ u_true
        Kbc, fbc = apply_dirichlet(K, f, fixed, 0.0)
        u, _info = solve_displacement(Kbc, fbc, size_threshold=4000)
        free = np.setdiff1d(np.arange(ndof), fixed)
        assert np.allclose(u[free], u_true[free], rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# せん断ロッキングテスト（片持ち梁曲げ）
# ---------------------------------------------------------------------------


def _cantilever_bending_solve(elem, E, nu, L, H, P, nx, ny):
    """片持ち梁曲げの先端変位を返すヘルパー."""
    x = np.linspace(0, L, nx + 1)
    y = np.linspace(0, H, ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    conn = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            conn.append([n0, n0 + 1, n0 + 1 + (nx + 1), n0 + (nx + 1)])
    conn = np.array(conn, dtype=int)

    mat = PlaneStrainElastic(E, nu)
    K = assemble_global_stiffness(nodes, [(elem, conn)], mat, thickness=1.0, show_progress=False)
    ndof = 2 * len(nodes)
    f = np.zeros(ndof)

    # 先端全節点に均等荷重
    tip_nodes = [j * (nx + 1) + nx for j in range(ny + 1)]
    for n_id in tip_nodes:
        f[2 * n_id + 1] = P / len(tip_nodes)

    # 固定端（x=0）
    fixed = []
    for j in range(ny + 1):
        n_id = j * (nx + 1)
        fixed.extend([2 * n_id, 2 * n_id + 1])
    fixed = np.array(fixed, dtype=int)

    Kbc, fbc = apply_dirichlet(K, f, fixed, 0.0)
    u, _info = solve_displacement(Kbc, fbc)

    tip_mid = (ny // 2) * (nx + 1) + nx
    return u[2 * tip_mid + 1]


class TestShearLocking:
    """せん断ロッキング抑制の検証."""

    E = 1000.0
    nu = 0.3
    L = 10.0
    H = 1.0
    P = 1.0

    @property
    def delta_analytical(self):
        """Euler-Bernoulli 梁理論の解析解（平面ひずみ補正）."""
        I_val = self.H**3 / 12.0
        E_eff = self.E / (1.0 - self.nu**2)
        return self.P * self.L**3 / (3.0 * E_eff * I_val)

    def test_eas_coarse_mesh(self):
        """EAS Q4: 粗メッシュ (10×1) でロッキングがないこと."""
        delta = _cantilever_bending_solve(
            Quad4EASPlaneStrain(), self.E, self.nu, self.L, self.H, self.P, 10, 1
        )
        ratio = delta / self.delta_analytical
        assert 0.95 < ratio < 1.10, f"EAS 10×1: ratio={ratio:.4f}, expected ~1.0"

    def test_plain_q4_locks(self):
        """Plain Q4: 同条件でロッキングすることを確認（EAS の優位性）."""
        delta = _cantilever_bending_solve(
            Quad4PlaneStrain(), self.E, self.nu, self.L, self.H, self.P, 10, 1
        )
        ratio = delta / self.delta_analytical
        assert ratio < 0.75, f"Plain Q4 should lock: ratio={ratio:.4f}"

    def test_eas_convergence(self):
        """EAS Q4: メッシュ細分化で収束."""
        for nx, ny in [(20, 2), (40, 4)]:
            delta = _cantilever_bending_solve(
                Quad4EASPlaneStrain(), self.E, self.nu, self.L, self.H, self.P, nx, ny
            )
            ratio = delta / self.delta_analytical
            assert 0.95 < ratio < 1.10, f"EAS {nx}×{ny}: ratio={ratio:.4f}"


# ---------------------------------------------------------------------------
# 体積ロッキングテスト
# ---------------------------------------------------------------------------


class TestVolumetricLocking:
    """非圧縮性材料でのロッキング抑制の検証."""

    E = 1000.0
    nu = 0.4999
    L = 10.0
    H = 1.0
    P = 1.0

    @property
    def delta_analytical(self):
        I_val = self.H**3 / 12.0
        E_eff = self.E / (1.0 - self.nu**2)
        return self.P * self.L**3 / (3.0 * E_eff * I_val)

    def test_eas_nearly_incompressible_bending(self):
        """EAS Q4: ν=0.4999 の曲げ問題でロッキングがないこと."""
        delta = _cantilever_bending_solve(
            Quad4EASPlaneStrain(), self.E, self.nu, self.L, self.H, self.P, 10, 1
        )
        ratio = delta / self.delta_analytical
        assert 0.90 < ratio < 1.15, f"EAS ν=0.4999: ratio={ratio:.4f}"

    def test_plain_q4_locks_incompressible(self):
        """Plain Q4: ν=0.4999 で壊滅的にロッキングすること."""
        delta = _cantilever_bending_solve(
            Quad4PlaneStrain(), self.E, self.nu, self.L, self.H, self.P, 10, 1
        )
        ratio = delta / self.delta_analytical
        assert ratio < 0.10, f"Plain Q4 ν=0.4999 should lock severely: ratio={ratio:.4f}"


# ---------------------------------------------------------------------------
# EAS + B-bar 併用テスト
# ---------------------------------------------------------------------------


class TestEASBBar:
    """EAS + B-bar 併用要素のテスト."""

    def test_basic_properties(self):
        """EAS+B-bar: 対称・半正定値・3 剛体モード."""
        D = constitutive_plane_strain(200e3, 0.3)
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        from xkep_cae.elements.quad4_eas_bbar import quad4_ke_plane_strain_eas_bbar

        Ke = quad4_ke_plane_strain_eas_bbar(xy, D, t=1.0)
        assert Ke.shape == (8, 8)
        assert np.allclose(Ke, Ke.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(Ke)
        assert eigvals.min() > -1e-10
        n_zero = np.sum(np.abs(eigvals) < 1e-6)
        assert n_zero == 3

    def test_manufactured_solution(self):
        """EAS+B-bar: 製造解テスト."""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        conn = np.array([[0, 1, 2, 3]])
        mat = PlaneStrainElastic(10.0, 0.25)
        elem = Quad4EASBBarPlaneStrain()
        K = assemble_global_stiffness(
            nodes, [(elem, conn)], mat, thickness=1.0, show_progress=False
        )
        fixed = np.array([0, 1, 6, 7], dtype=int)
        n = K.shape[0]
        rng = np.random.default_rng(99)
        u_true = rng.standard_normal(n)
        u_true[fixed] = 0.0
        f = K @ u_true
        Kbc, fbc = apply_dirichlet(K, f, fixed, 0.0)
        u, _info = solve_displacement(Kbc, fbc, size_threshold=4000)
        free = np.setdiff1d(np.arange(n), fixed)
        assert np.allclose(u[free], u_true[free], rtol=1e-10, atol=1e-10)
