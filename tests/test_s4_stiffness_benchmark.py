"""Phase S4: 剛性比較ベンチマーク（HEX8連続体 vs 梁モデル）.

同一断面の円柱をHEX8連続体要素と梁要素でモデル化し、
等価剛性（軸・曲げ・ねじり）を比較する。
HEX8モデルは矩形断面近似（円に内接する正方形）を使用。

結果はモデル選択指針の根拠とする。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.elements.beam_timo3d import timo_beam3d_ke_global
from xkep_cae.elements.hex8 import hex8_ke_sri_bbar
from xkep_cae.materials.elastic import constitutive_3d
from xkep_cae.sections.beam import BeamSection

pytestmark = pytest.mark.slow

# ====================================================================
# 共通パラメータ（撚線標準鋼線）
# ====================================================================

_E = 200e9  # Pa
_NU = 0.3
_G = _E / (2.0 * (1.0 + _NU))
_D = constitutive_3d(_E, _NU)

_WIRE_D = 0.002  # 直径 2mm
_SECTION = BeamSection.circle(_WIRE_D)
_KAPPA = 6.0 * (1.0 + _NU) / (7.0 + 6.0 * _NU)

_L = 0.040  # 40mm（1ピッチ長）


# ====================================================================
# ヘルパー: HEX8 矩形断面梁メッシュ
# ====================================================================


def _make_hex8_beam_mesh(
    L: float,
    h: float,
    w: float,
    ne_x: int,
    ne_y: int,
    ne_z: int,
):
    """z方向に伸びる直方体HEX8メッシュを生成.

    断面は xy 平面、長手方向が z。
    断面を原点中心に配置（-h/2～+h/2, -w/2～+w/2）。

    Returns:
        nodes: (n_nodes, 3)
        conn: (n_elems, 8)
    """
    xs = np.linspace(-w / 2, w / 2, ne_x + 1)
    ys = np.linspace(-h / 2, h / 2, ne_y + 1)
    zs = np.linspace(0, L, ne_z + 1)

    node_list = []
    for iz in range(ne_z + 1):
        for iy in range(ne_y + 1):
            for ix in range(ne_x + 1):
                node_list.append([xs[ix], ys[iy], zs[iz]])
    nodes = np.array(node_list)

    def node_id(ix, iy, iz):
        return iz * (ne_y + 1) * (ne_x + 1) + iy * (ne_x + 1) + ix

    conn_list = []
    for iz in range(ne_z):
        for iy in range(ne_y):
            for ix in range(ne_x):
                n0 = node_id(ix, iy, iz)
                n1 = node_id(ix + 1, iy, iz)
                n2 = node_id(ix + 1, iy + 1, iz)
                n3 = node_id(ix, iy + 1, iz)
                n4 = node_id(ix, iy, iz + 1)
                n5 = node_id(ix + 1, iy, iz + 1)
                n6 = node_id(ix + 1, iy + 1, iz + 1)
                n7 = node_id(ix, iy + 1, iz + 1)
                conn_list.append([n0, n1, n2, n3, n4, n5, n6, n7])
    conn = np.array(conn_list, dtype=int)
    return nodes, conn


def _assemble_hex8_stiffness(nodes, conn, D):
    """HEX8要素の全体剛性行列をアセンブリ."""
    n_nodes = len(nodes)
    ndof = 3 * n_nodes
    K = np.zeros((ndof, ndof))
    for elem in conn:
        coords_e = nodes[elem]
        ke = hex8_ke_sri_bbar(coords_e, D)
        dofs = np.array([3 * n + d for n in elem for d in range(3)])
        for ii in range(24):
            for jj in range(24):
                K[dofs[ii], dofs[jj]] += ke[ii, jj]
    return K


def _solve_hex8_statics(nodes, conn, D, fixed_dofs, f_ext):
    """HEX8モデルの静的解析."""
    K = _assemble_hex8_stiffness(nodes, conn, D)
    ndof = K.shape[0]
    free = np.array([d for d in range(ndof) if d not in fixed_dofs])
    K_ff = K[np.ix_(free, free)]
    F_f = f_ext[free]
    u_f = np.linalg.solve(K_ff, F_f)
    u = np.zeros(ndof)
    u[free] = u_f
    return u


# ====================================================================
# ヘルパー: 梁モデルの解析
# ====================================================================


def _assemble_beam_stiffness(L: float, ne_z: int, E, G, A, Iy, Iz, J, kappa):
    """z方向に伸びる梁モデルの全体剛性行列."""
    n_nodes = ne_z + 1
    ndof = 6 * n_nodes
    coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        coords[i, 2] = L * i / ne_z

    K = np.zeros((ndof, ndof))
    for e in range(ne_z):
        n1, n2 = e, e + 1
        elem_coords = coords[np.array([n1, n2])]
        ke = timo_beam3d_ke_global(elem_coords, E, G, A, Iy, Iz, J, kappa, kappa)
        edofs = np.array([6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)])
        for ii in range(12):
            for jj in range(12):
                K[edofs[ii], edofs[jj]] += ke[ii, jj]

    return K, n_nodes, coords


def _solve_beam_statics(L, ne_z, E, G, A, Iy, Iz, J, kappa, fixed_dofs, f_ext):
    """梁モデルの静的解析."""
    K, n_nodes, coords = _assemble_beam_stiffness(L, ne_z, E, G, A, Iy, Iz, J, kappa)
    ndof = K.shape[0]
    free = np.array([d for d in range(ndof) if d not in fixed_dofs])
    K_ff = K[np.ix_(free, free)]
    F_f = f_ext[free]
    u_f = np.linalg.solve(K_ff, F_f)
    u = np.zeros(ndof)
    u[free] = u_f
    return u


# ====================================================================
# 解析解
# ====================================================================


def _analytical_axial_stiffness(E, A, L):
    """軸剛性 k = EA/L."""
    return E * A / L


def _analytical_bending_stiffness(E, I, L):  # noqa: E741
    """曲げ剛性（片持ち先端集中荷重）k = 3EI/L³."""
    return 3 * E * I / L**3


def _analytical_torsion_stiffness(G, J, L):
    """ねじり剛性 k = GJ/L."""
    return G * J / L


# ====================================================================
# テストクラス
# ====================================================================


class TestS4StiffnessComparison:
    """HEX8連続体 vs 梁モデル 等価剛性比較."""

    # 矩形断面パラメータ（円に内接する正方形）
    _SIDE = _WIRE_D / np.sqrt(2)  # 正方形の辺長
    _A_RECT = _SIDE**2
    _I_RECT = _SIDE**4 / 12
    _J_RECT = 0.1406 * _SIDE**4  # 正方形のねじり定数（近似）

    # HEX8メッシュ分割数
    _NE_XY = 4  # 断面方向
    _NE_Z = 16  # 長手方向

    def test_axial_stiffness(self):
        """軸剛性（引張）: HEX8 vs 梁 vs 解析解.

        片端固定、もう片端にz方向単位力 → 先端z変位から剛性算出。
        """
        ne_xy = self._NE_XY
        ne_z = self._NE_Z
        side = self._SIDE

        # --- HEX8 ---
        nodes, conn = _make_hex8_beam_mesh(_L, side, side, ne_xy, ne_xy, ne_z)
        n_nodes_hex = len(nodes)
        ndof_hex = 3 * n_nodes_hex

        # 固定: z=0 面の全DOF
        fixed_hex = []
        for i in range(n_nodes_hex):
            if abs(nodes[i, 2]) < 1e-10:
                fixed_hex.extend([3 * i, 3 * i + 1, 3 * i + 2])

        # z=L 面にz方向力
        P = 1.0
        f_hex = np.zeros(ndof_hex)
        tip_nodes = [i for i in range(n_nodes_hex) if abs(nodes[i, 2] - _L) < 1e-10]
        f_per = P / len(tip_nodes)
        for nid in tip_nodes:
            f_hex[3 * nid + 2] = f_per

        u_hex = _solve_hex8_statics(nodes, conn, _D, fixed_hex, f_hex)
        tip_uz_hex = np.mean([u_hex[3 * i + 2] for i in tip_nodes])
        k_hex = P / tip_uz_hex

        # --- 梁モデル（矩形断面） ---
        ne_z_beam = ne_z
        n_nodes_beam = ne_z_beam + 1
        ndof_beam = 6 * n_nodes_beam

        fixed_beam = list(range(6))  # 節点0 全固定
        f_beam = np.zeros(ndof_beam)
        f_beam[6 * (n_nodes_beam - 1) + 2] = P  # 先端z

        A_rect = self._A_RECT
        I_rect = self._I_RECT
        J_rect = self._J_RECT
        u_beam = _solve_beam_statics(
            _L,
            ne_z_beam,
            _E,
            _G,
            A_rect,
            I_rect,
            I_rect,
            J_rect,
            _KAPPA,
            fixed_beam,
            f_beam,
        )
        tip_uz_beam = u_beam[6 * (n_nodes_beam - 1) + 2]
        k_beam = P / tip_uz_beam

        # --- 解析解 ---
        k_exact = _analytical_axial_stiffness(_E, A_rect, _L)

        # レポート
        print(f"\n{'=' * 60}")
        print("  S4 軸剛性比較")
        print(f"{'=' * 60}")
        print(f"  解析解 (EA/L):  {k_exact:.4e} N/m")
        print(
            f"  HEX8:           {k_hex:.4e} N/m (誤差: {abs(k_hex - k_exact) / k_exact * 100:.2f}%)"
        )
        print(
            f"  梁:             {k_beam:.4e} N/m (誤差: {abs(k_beam - k_exact) / k_exact * 100:.2f}%)"
        )
        print(f"  HEX8/梁比:      {k_hex / k_beam:.4f}")

        # 検証
        assert abs(k_hex - k_exact) / k_exact < 0.05, (
            f"HEX8軸剛性誤差5%超: {k_hex:.4e} vs {k_exact:.4e}"
        )
        assert abs(k_beam - k_exact) / k_exact < 0.01, (
            f"梁軸剛性誤差1%超: {k_beam:.4e} vs {k_exact:.4e}"
        )

    def test_bending_stiffness(self):
        """曲げ剛性: HEX8 vs 梁 vs 解析解.

        片端全固定、先端にy方向集中荷重 → 先端y変位から剛性算出。
        """
        ne_xy = self._NE_XY
        ne_z = self._NE_Z
        side = self._SIDE

        # --- HEX8 ---
        nodes, conn = _make_hex8_beam_mesh(_L, side, side, ne_xy, ne_xy, ne_z)
        n_nodes_hex = len(nodes)
        ndof_hex = 3 * n_nodes_hex

        fixed_hex = []
        for i in range(n_nodes_hex):
            if abs(nodes[i, 2]) < 1e-10:
                fixed_hex.extend([3 * i, 3 * i + 1, 3 * i + 2])

        P = 1.0
        f_hex = np.zeros(ndof_hex)
        tip_nodes = [i for i in range(n_nodes_hex) if abs(nodes[i, 2] - _L) < 1e-10]
        f_per = P / len(tip_nodes)
        for nid in tip_nodes:
            f_hex[3 * nid + 1] = f_per  # y方向

        u_hex = _solve_hex8_statics(nodes, conn, _D, fixed_hex, f_hex)
        tip_uy_hex = np.mean([u_hex[3 * i + 1] for i in tip_nodes])
        k_hex = P / tip_uy_hex

        # --- 梁モデル ---
        ne_z_beam = ne_z
        n_nodes_beam = ne_z_beam + 1
        ndof_beam = 6 * n_nodes_beam

        fixed_beam = list(range(6))
        f_beam = np.zeros(ndof_beam)
        f_beam[6 * (n_nodes_beam - 1) + 1] = P  # 先端y

        A_rect = self._A_RECT
        I_rect = self._I_RECT
        J_rect = self._J_RECT
        u_beam = _solve_beam_statics(
            _L,
            ne_z_beam,
            _E,
            _G,
            A_rect,
            I_rect,
            I_rect,
            J_rect,
            _KAPPA,
            fixed_beam,
            f_beam,
        )
        tip_uy_beam = u_beam[6 * (n_nodes_beam - 1) + 1]
        k_beam = P / tip_uy_beam

        # --- 解析解 ---
        k_exact = _analytical_bending_stiffness(_E, self._I_RECT, _L)

        print(f"\n{'=' * 60}")
        print("  S4 曲げ剛性比較")
        print(f"{'=' * 60}")
        print(f"  解析解 (3EI/L³): {k_exact:.4e} N/m")
        print(
            f"  HEX8:            {k_hex:.4e} N/m (誤差: {abs(k_hex - k_exact) / k_exact * 100:.2f}%)"
        )
        print(
            f"  梁:              {k_beam:.4e} N/m (誤差: {abs(k_beam - k_exact) / k_exact * 100:.2f}%)"
        )
        print(f"  HEX8/梁比:       {k_hex / k_beam:.4f}")

        # HEX8は曲げで10-20%の誤差が出うる（粗いメッシュ+矩形近似）
        assert abs(k_hex - k_exact) / k_exact < 0.25, "HEX8曲げ剛性誤差25%超"
        assert abs(k_beam - k_exact) / k_exact < 0.05, "梁曲げ剛性誤差5%超"

    def test_torsion_stiffness(self):
        """ねじり剛性: HEX8 vs 梁 vs 解析解.

        片端全固定、先端にz軸まわりねじりモーメント → 先端回転からねじり剛性算出。
        HEX8ではカップル力で近似。
        """
        ne_xy = self._NE_XY
        ne_z = self._NE_Z
        side = self._SIDE

        # --- HEX8 ---
        nodes, conn = _make_hex8_beam_mesh(_L, side, side, ne_xy, ne_xy, ne_z)
        n_nodes_hex = len(nodes)
        ndof_hex = 3 * n_nodes_hex

        fixed_hex = []
        for i in range(n_nodes_hex):
            if abs(nodes[i, 2]) < 1e-10:
                fixed_hex.extend([3 * i, 3 * i + 1, 3 * i + 2])

        # ねじりモーメント: 端面のカップル力で近似
        # T = sum(r × F) で T_z = 1 N·m を生成
        M_target = 0.001  # 1e-3 N·m
        f_hex = np.zeros(ndof_hex)
        tip_nodes = [i for i in range(n_nodes_hex) if abs(nodes[i, 2] - _L) < 1e-10]

        # 各節点に r × F_tangential を計算
        for nid in tip_nodes:
            x, y = nodes[nid, 0], nodes[nid, 1]
            r = np.sqrt(x**2 + y**2)
            if r < 1e-15:
                continue
            # 接線方向: (-y, x, 0) / r
            f_tangential = M_target / (len(tip_nodes) * r) if r > 0 else 0
            f_hex[3 * nid + 0] = -y / r * f_tangential  # Fx
            f_hex[3 * nid + 1] = x / r * f_tangential  # Fy

        u_hex = _solve_hex8_statics(nodes, conn, _D, fixed_hex, f_hex)

        # 先端の平均回転角を算出
        theta_sum = 0.0
        n_count = 0
        for nid in tip_nodes:
            x, y = nodes[nid, 0], nodes[nid, 1]
            r = np.sqrt(x**2 + y**2)
            if r < 1e-15:
                continue
            ux, uy = u_hex[3 * nid], u_hex[3 * nid + 1]
            # theta = (-y*ux + x*uy) / r^2
            theta = (-y * ux + x * uy) / r**2
            theta_sum += theta
            n_count += 1
        theta_avg = theta_sum / max(1, n_count)
        k_hex = M_target / theta_avg if abs(theta_avg) > 1e-20 else 0.0

        # --- 梁モデル ---
        ne_z_beam = ne_z
        n_nodes_beam = ne_z_beam + 1
        ndof_beam = 6 * n_nodes_beam

        fixed_beam = list(range(6))
        f_beam = np.zeros(ndof_beam)
        f_beam[6 * (n_nodes_beam - 1) + 5] = M_target  # θz（z軸梁のねじり DOF）

        A_rect = self._A_RECT
        I_rect = self._I_RECT
        J_rect = self._J_RECT
        u_beam = _solve_beam_statics(
            _L,
            ne_z_beam,
            _E,
            _G,
            A_rect,
            I_rect,
            I_rect,
            J_rect,
            _KAPPA,
            fixed_beam,
            f_beam,
        )
        theta_beam = u_beam[6 * (n_nodes_beam - 1) + 5]
        k_beam = M_target / theta_beam if abs(theta_beam) > 1e-20 else 0.0

        # --- 解析解 ---
        k_exact = _analytical_torsion_stiffness(_G, self._J_RECT, _L)

        print(f"\n{'=' * 60}")
        print("  S4 ねじり剛性比較")
        print(f"{'=' * 60}")
        print(f"  解析解 (GJ/L):  {k_exact:.4e} N·m/rad")
        print(
            f"  HEX8:           {k_hex:.4e} N·m/rad (誤差: {abs(k_hex - k_exact) / k_exact * 100:.2f}%)"
        )
        print(
            f"  梁:             {k_beam:.4e} N·m/rad (誤差: {abs(k_beam - k_exact) / k_exact * 100:.2f}%)"
        )
        print(f"  HEX8/梁比:      {k_hex / k_beam:.4f}")

        # ねじりはHEX8で大きな誤差が出る（断面変形の制約なし）
        # 定量的に結果を記録することが目的
        assert k_hex > 0, "HEX8ねじり剛性が正でない"
        assert k_beam > 0, "梁ねじり剛性が正でない"

    def test_summary_report(self):
        """全剛性の比較サマリーレポート."""
        side = self._SIDE
        A_rect = self._A_RECT
        I_rect = self._I_RECT
        J_rect = self._J_RECT

        # 円形断面パラメータ
        A_circ = _SECTION.A
        I_circ = _SECTION.Iy
        J_circ = _SECTION.J

        print(f"\n{'=' * 70}")
        print("  S4 剛性比較サマリー")
        print(f"{'=' * 70}")
        print(f"  素線直径: {_WIRE_D * 1000:.1f} mm")
        print(f"  長さ:     {_L * 1000:.1f} mm")
        print(f"  HEX8:     正方形断面 {side * 1000:.3f} mm × {side * 1000:.3f} mm")
        print(f"            （円に内接、面積比 = {A_rect / A_circ:.4f}）")
        print()
        print(f"  {'項目':<20} {'円形断面解析解':>16} {'矩形断面解析解':>16} {'比率':>8}")
        print(f"  {'-' * 64}")

        k_axial_circ = _analytical_axial_stiffness(_E, A_circ, _L)
        k_axial_rect = _analytical_axial_stiffness(_E, A_rect, _L)
        print(
            f"  {'軸剛性 EA/L':<20} {k_axial_circ:>16.4e} {k_axial_rect:>16.4e} {k_axial_rect / k_axial_circ:>7.4f}"
        )

        k_bend_circ = _analytical_bending_stiffness(_E, I_circ, _L)
        k_bend_rect = _analytical_bending_stiffness(_E, I_rect, _L)
        print(
            f"  {'曲げ剛性 3EI/L³':<20} {k_bend_circ:>16.4e} {k_bend_rect:>16.4e} {k_bend_rect / k_bend_circ:>7.4f}"
        )

        k_tors_circ = _analytical_torsion_stiffness(_G, J_circ, _L)
        k_tors_rect = _analytical_torsion_stiffness(_G, J_rect, _L)
        print(
            f"  {'ねじり剛性 GJ/L':<20} {k_tors_circ:>16.4e} {k_tors_rect:>16.4e} {k_tors_rect / k_tors_circ:>7.4f}"
        )

        print()
        print("  結論: 矩形断面（HEX8近似）は円形断面に比べて")
        print(f"    軸剛性:    {k_axial_rect / k_axial_circ:.1%}")
        print(f"    曲げ剛性:  {k_bend_rect / k_bend_circ:.1%}")
        print(f"    ねじり剛性: {k_tors_rect / k_tors_circ:.1%}")
        print("  の比率となる。梁モデルは断面特性を正確に反映できるため、")
        print("  撚線解析には梁モデルが適切。")
