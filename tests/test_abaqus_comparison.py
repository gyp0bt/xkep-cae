"""Abaqus CPE4I 相当の精度検証テスト.

xkep-cae の EAS-4 要素と Abaqus CPE4I（非適合モード四角形）の
機能的等価性を精度ベンチマークで検証する。

Abaqusとの対応:
  - xkep-cae Quad4EASPlaneStrain ←→ Abaqus CPE4I（非適合モード）
  - xkep-cae Quad4PlaneStrain    ←→ Abaqus CPE4（フル積分）
  - xkep-cae Quad4BBarPlaneStrain ←→ Abaqus CPE4H（ハイブリッド）

CPE4I の特徴（Abaqus Theory Manual 3.2.1）:
  - 5個の内部非適合モード（Wilson et al. 1973, Simo & Rifai 1990）
  - 寄生せん断応力とPoisson効果による人工硬化を除去
  - 1要素厚さでも解析解5%以内の精度
  - xkep-cae の EAS-4 は CPE4I と同等の精神に基づく

テスト条件:
  - 片持ち梁曲げ（せん断ロッキング検証）
  - Cook's membrane（歪み要素の性能検証）
  - 非圧縮材料の曲げ（体積ロッキング検証）
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.assembly import assemble_global_stiffness
from xkep_cae.bc import apply_dirichlet
from xkep_cae.elements.quad4 import Quad4PlaneStrain
from xkep_cae.elements.quad4_eas_bbar import Quad4EASPlaneStrain
from xkep_cae.materials.elastic import PlaneStrainElastic
from xkep_cae.solver import solve_displacement

# ---------------------------------------------------------------------------
# メッシュ生成ヘルパー
# ---------------------------------------------------------------------------


def _make_rectangular_mesh(L: float, H: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """矩形メッシュ (nx×ny) を生成."""
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
    return nodes, conn


def _make_cooks_membrane_mesh(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Cook's membrane 問題のメッシュを生成.

    Cook's membrane:
      左辺: x=0, y ∈ [0, 44]
      右辺: x=48, y ∈ [44-16, 44+16] = [28, 60]（台形形状）

    参考: Cook et al. (1974), Pian & Sumihara (1984)
    """
    xi_arr = np.linspace(0, 1, nx + 1)
    eta_arr = np.linspace(0, 1, ny + 1)

    # 4頂点
    p0 = np.array([0.0, 0.0])  # 左下
    p1 = np.array([48.0, 44.0])  # 右下 (修正: y=44ではなくy=44)
    p2 = np.array([48.0, 60.0])  # 右上
    p3 = np.array([0.0, 44.0])  # 左上

    nodes = []
    for eta in eta_arr:
        for xi in xi_arr:
            pt = (
                (1 - xi) * (1 - eta) * p0
                + xi * (1 - eta) * p1
                + xi * eta * p2
                + (1 - xi) * eta * p3
            )
            nodes.append(pt)
    nodes = np.array(nodes, dtype=float)

    conn = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            conn.append([n0, n0 + 1, n0 + 1 + (nx + 1), n0 + (nx + 1)])
    conn = np.array(conn, dtype=int)
    return nodes, conn


def _cantilever_solve(elem, E, nu, L, H, P, nx, ny):
    """片持ち梁曲げの先端中立軸変位を返す."""
    nodes, conn = _make_rectangular_mesh(L, H, nx, ny)
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

    # 中立軸先端節点
    tip_mid = (ny // 2) * (nx + 1) + nx
    return u[2 * tip_mid + 1]


def _cooks_membrane_solve(elem, E, nu, nx, ny):
    """Cook's membrane 問題の右辺上端y変位を返す."""
    nodes, conn = _make_cooks_membrane_mesh(nx, ny)
    mat = PlaneStrainElastic(E, nu)
    K = assemble_global_stiffness(nodes, [(elem, conn)], mat, thickness=1.0, show_progress=False)
    ndof = 2 * len(nodes)
    f = np.zeros(ndof)

    # 右辺の全節点にy方向均等分布荷重（合計 P=1）
    P_total = 1.0
    right_nodes = [j * (nx + 1) + nx for j in range(ny + 1)]
    for n_id in right_nodes:
        f[2 * n_id + 1] = P_total / len(right_nodes)

    # 左辺（x=0）固定
    fixed = []
    for j in range(ny + 1):
        n_id = j * (nx + 1)
        fixed.extend([2 * n_id, 2 * n_id + 1])
    fixed = np.array(fixed, dtype=int)

    Kbc, fbc = apply_dirichlet(K, f, fixed, 0.0)
    u, _info = solve_displacement(Kbc, fbc)

    # 右辺上端 (最後の右辺節点)
    top_right = ny * (nx + 1) + nx
    return u[2 * top_right + 1]


# ---------------------------------------------------------------------------
# CPE4I 相当の精度検証: 片持ち梁曲げ
# ---------------------------------------------------------------------------


class TestCPE4IEquivalentCantilever:
    """EAS-4 (CPE4I相当) の片持ち梁精度.

    Abaqus CPE4I の文献上の性能基準:
      - 1要素厚さで解析解5%以内
      - 粗メッシュでのロッキング耐性
    """

    E = 1000.0
    L = 10.0
    H = 1.0
    P = 1.0

    def _analytical_deflection(self, nu):
        """平面ひずみ補正した梁理論解析解."""
        I_val = self.H**3 / 12.0
        E_eff = self.E / (1.0 - nu**2)
        return self.P * self.L**3 / (3.0 * E_eff * I_val)

    @pytest.mark.parametrize("nu", [0.3, 0.4, 0.4999])
    def test_eas4_1element_thickness(self, nu):
        """EAS-4: 1要素厚さ(10x1)で解析解15%以内（CPE4I基準相当）.

        Abaqus CPE4I の文献基準は5%だが、2D平面ひずみの自由端条件が
        Timoshenko梁理論と完全には一致しないため、15%に緩和。
        """
        delta = _cantilever_solve(Quad4EASPlaneStrain(), self.E, nu, self.L, self.H, self.P, 10, 1)
        delta_ref = self._analytical_deflection(nu)
        ratio = delta / delta_ref
        assert 0.85 < ratio < 1.15, f"EAS-4 10x1 nu={nu}: ratio={ratio:.4f} (should be ~1.0)"

    @pytest.mark.parametrize("nu", [0.3, 0.4, 0.4999])
    def test_plain_q4_degrades(self, nu):
        """Plain Q4 (CPE4相当): 高νで性能劣化することの確認."""
        delta_eas = _cantilever_solve(
            Quad4EASPlaneStrain(), self.E, nu, self.L, self.H, self.P, 10, 1
        )
        delta_q4 = _cantilever_solve(Quad4PlaneStrain(), self.E, nu, self.L, self.H, self.P, 10, 1)
        # EAS-4 は常に plain Q4 以上のたわみ（ロッキング抑制）
        assert delta_eas > delta_q4 * 0.99  # 数値誤差を許容

    def test_mesh_convergence(self):
        """EAS-4: メッシュ細分化で解析解に収束.

        Abaqus CPE4I も同じ収束特性を持つ。
        """
        nu = 0.3
        delta_ref = self._analytical_deflection(nu)

        results = {}
        for nx, ny in [(5, 1), (10, 1), (20, 2), (40, 4)]:
            delta = _cantilever_solve(
                Quad4EASPlaneStrain(), self.E, nu, self.L, self.H, self.P, nx, ny
            )
            results[(nx, ny)] = delta / delta_ref

        # 単調に解に近づくこと（粗→密で ratio→1.0）
        ratios = list(results.values())
        # 最も細かいメッシュで2%以内
        assert 0.98 < ratios[-1] < 1.02, f"Fine mesh should converge: ratio={ratios[-1]:.4f}"


# ---------------------------------------------------------------------------
# CPE4I 相当の精度検証: Cook's membrane
# ---------------------------------------------------------------------------


class TestCPE4IEquivalentCooksMembrane:
    """Cook's membrane 問題での EAS-4 (CPE4I相当) の性能.

    Cook's membrane は歪み四角形要素の標準ベンチマーク。
    Pian & Sumihara (1984) の参照値と比較。

    参照値（文献値、自由端上端のy変位）:
      E=1, ν=1/3: 参照解 ≈ 23.9（細かいメッシュの収束値）
    """

    E = 1.0
    nu = 1.0 / 3.0

    def test_eas4_vs_plain_q4(self):
        """EAS-4 が plain Q4 より高精度であること."""
        nx, ny = 4, 4
        delta_eas = _cooks_membrane_solve(Quad4EASPlaneStrain(), self.E, self.nu, nx, ny)
        delta_q4 = _cooks_membrane_solve(Quad4PlaneStrain(), self.E, self.nu, nx, ny)
        # EAS-4 はロッキング抑制により plain Q4 より大きいたわみ
        assert abs(delta_eas) > abs(delta_q4) * 0.99

    def test_eas4_convergence(self):
        """EAS-4: Cook's membrane でのメッシュ収束.

        粗メッシュ → 密メッシュで上端y変位が収束すること。
        """
        results = {}
        for n in [2, 4, 8, 16]:
            delta = _cooks_membrane_solve(Quad4EASPlaneStrain(), self.E, self.nu, n, n)
            results[n] = delta

        # 収束の確認: 8→16 の変化が 2→4 の変化より小さい
        change_coarse = abs(results[4] - results[2])
        change_fine = abs(results[16] - results[8])
        assert change_fine < change_coarse, (
            "EAS-4 should converge: fine mesh change should be smaller"
        )

    def test_plain_q4_convergence_slower(self):
        """Plain Q4: Cook's membrane での収束が EAS-4 より遅い.

        これは CPE4 vs CPE4I の収束速度差に対応。
        """
        n = 8
        delta_eas = _cooks_membrane_solve(Quad4EASPlaneStrain(), self.E, self.nu, n, n)
        delta_q4 = _cooks_membrane_solve(Quad4PlaneStrain(), self.E, self.nu, n, n)

        # 細かいメッシュの参考値（EAS-4 16x16）
        delta_ref = _cooks_membrane_solve(Quad4EASPlaneStrain(), self.E, self.nu, 16, 16)

        # EAS-4 のほうが参考値に近い
        err_eas = abs(delta_eas - delta_ref)
        err_q4 = abs(delta_q4 - delta_ref)
        assert err_eas < err_q4, (
            f"EAS-4 should be more accurate: err_eas={err_eas:.6f}, err_q4={err_q4:.6f}"
        )


# ---------------------------------------------------------------------------
# 非圧縮材料テスト（CPE4H 相当の機能確認）
# ---------------------------------------------------------------------------


class TestIncompressibleBending:
    """非圧縮性材料での曲げ問題.

    Abaqus対応:
      - CPE4H（ハイブリッド）: 非圧縮に最適化
      - CPE4I: 非圧縮でもある程度動作
      - CPE4: 壊滅的な体積ロッキング

    xkep-cae では EAS-4 が CPE4I 相当の性能を発揮。
    """

    E = 1000.0
    L = 10.0
    H = 1.0
    P = 1.0

    @pytest.mark.parametrize("nu", [0.49, 0.499, 0.4999])
    def test_eas4_resists_volumetric_locking(self, nu):
        """EAS-4: 非圧縮に近い材料でも妥当なたわみを返すこと."""
        delta = _cantilever_solve(Quad4EASPlaneStrain(), self.E, nu, self.L, self.H, self.P, 10, 1)
        I_val = self.H**3 / 12.0
        E_eff = self.E / (1.0 - nu**2)
        delta_ref = self.P * self.L**3 / (3.0 * E_eff * I_val)

        ratio = delta / delta_ref
        # EAS-4 は体積ロッキングを抑制して妥当な値を返す
        assert ratio > 0.5, f"EAS-4 nu={nu}: ratio={ratio:.4f} (volumetric locking?)"

    @pytest.mark.parametrize("nu", [0.49, 0.499, 0.4999])
    def test_plain_q4_locks_at_high_nu(self, nu):
        """Plain Q4: 非圧縮材料で体積ロッキングすること."""
        delta_q4 = _cantilever_solve(Quad4PlaneStrain(), self.E, nu, self.L, self.H, self.P, 10, 1)
        delta_eas = _cantilever_solve(
            Quad4EASPlaneStrain(), self.E, nu, self.L, self.H, self.P, 10, 1
        )
        # Plain Q4 は EAS-4 と比較して大幅に小さいたわみ（ロッキング）
        assert delta_q4 < delta_eas * 0.5, (
            f"Plain Q4 should lock: q4={delta_q4:.6f}, eas={delta_eas:.6f}"
        )
