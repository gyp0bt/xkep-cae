"""幾何学非線形基盤の物理テスト — CR定式化の物理的正しさを検証.

プログラムテストではなく「物理的に当然の性質」をコード化する。

テスト構成:
- TestCRStressPhysics: CR大変形後の応力分布の物理的妥当性
- TestCRCurvaturePhysics: CR大変形後の曲率分布の物理的妥当性
- TestCRLoadOrderPhysics: 荷重オーダーの検証（剛性×変位 ≈ 反力）
- TestCRSymmetryPhysics: 対称荷重に対する対称変形
- TestCRDeformationPhysics: 大変形の物理的性質（面積保存、せん断ロッキングなし等）
"""

from __future__ import annotations

import numpy as np
import pytest
from xkep_cae.elements.beam_timo3d import (
    assemble_cr_beam3d,
    beam3d_max_bending_stress,
    beam3d_section_forces,
)
from xkep_cae.sections.beam import BeamSection
from xkep_cae.solver import newton_raphson

pytestmark = pytest.mark.slow


# ====================================================================
# ヘルパー
# ====================================================================


def _build_cantilever_cr(
    n_elems: int = 20,
    L: float = 1.0,
    E: float = 2.1e11,
    nu: float = 0.3,
    d: float = 0.02,
):
    """CR用カンチレバー構築."""
    G = E / (2.0 * (1.0 + nu))
    sec = BeamSection.circle(d)
    kappa = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)

    nodes = np.zeros((n_elems + 1, 3))
    nodes[:, 0] = np.linspace(0, L, n_elems + 1)
    conn = np.array([[i, i + 1] for i in range(n_elems)])

    n_nodes = n_elems + 1
    ndof = 6 * n_nodes
    fixed_dofs = np.arange(6)

    return {
        "nodes": nodes,
        "conn": conn,
        "E": E,
        "G": G,
        "sec": sec,
        "A": sec.A,
        "Iy": sec.Iy,
        "Iz": sec.Iz,
        "J": sec.J,
        "kappa_y": kappa,
        "kappa_z": kappa,
        "d": d,
        "L": L,
        "ndof": ndof,
        "n_nodes": n_nodes,
        "n_elems": n_elems,
        "fixed_dofs": fixed_dofs,
    }


def _solve_cr_nonlinear(data, f_ext, n_load_steps=10, tol=1e-6):
    """CR非線形静解析."""
    nodes, conn = data["nodes"], data["conn"]
    E, G = data["E"], data["G"]
    A, Iy, Iz, J = data["A"], data["Iy"], data["Iz"], data["J"]
    ky, kz = data["kappa_y"], data["kappa_z"]

    def assemble_tangent(u):
        K, _ = assemble_cr_beam3d(
            nodes,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            ky,
            kz,
            stiffness=True,
            internal_force=False,
            sparse=True,
        )
        return K

    def assemble_fint(u):
        _, f = assemble_cr_beam3d(
            nodes,
            conn,
            u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            ky,
            kz,
            stiffness=False,
            internal_force=True,
            sparse=False,
        )
        return f

    result = newton_raphson(
        f_ext,
        data["fixed_dofs"],
        assemble_tangent,
        assemble_fint,
        n_load_steps=n_load_steps,
        tol_force=tol,
        tol_disp=tol,
        show_progress=False,
    )
    return result


def _extract_section_forces(data, u):
    """解析結果から全要素の断面力を抽出（線形ベース、小変形用）."""
    nodes, conn = data["nodes"], data["conn"]
    E, G = data["E"], data["G"]
    A, Iy, Iz, J = data["A"], data["Iy"], data["Iz"], data["J"]
    ky, kz = data["kappa_y"], data["kappa_z"]

    forces = []
    for n1, n2 in conn:
        coords = nodes[np.array([n1, n2])]
        edofs = np.array([6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)])
        u_elem = u[edofs]
        f1, f2 = beam3d_section_forces(
            coords,
            u_elem,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            ky,
            kz,
        )
        forces.append((f1, f2))
    return forces


def _extract_cr_element_moments(data, u):
    """CR要素ごとの corotated フレームでの断面力（モーメント）を抽出.

    timo_beam3d_cr_internal_force の中間結果 f_cr = K_local @ d_cr を
    各要素で計算し、node2 の曲げモーメント(My, Mz)を返す。
    """
    from xkep_cae.elements.beam_timo3d import (
        _beam3d_length_and_direction,
        _build_local_axes,
        _rotmat_to_rotvec,
        _rotvec_to_rotmat,
        timo_beam3d_ke_local,
    )

    nodes, conn = data["nodes"], data["conn"]
    E, G = data["E"], data["G"]
    A, Iy, Iz, J = data["A"], data["Iy"], data["Iz"], data["J"]
    ky, kz = data["kappa_y"], data["kappa_z"]

    My_list = []
    Mz_list = []
    N_list = []
    for n1, n2 in conn:
        coords = nodes[np.array([n1, n2])]
        edofs = np.array([6 * n1 + d for d in range(6)] + [6 * n2 + d for d in range(6)])
        u_elem = u[edofs]

        # CR kinematics: corotated frameでの自然変形を抽出
        L_0, e_x_0 = _beam3d_length_and_direction(coords)
        R_0 = _build_local_axes(e_x_0, None)
        v_ref_stable = R_0[1, :]

        x1_def = coords[0] + u_elem[0:3]
        x2_def = coords[1] + u_elem[6:9]
        coords_def = np.array([x1_def, x2_def])
        L_def, e_x_def = _beam3d_length_and_direction(coords_def)
        R_cr = _build_local_axes(e_x_def, v_ref_stable)

        R_node1 = _rotvec_to_rotmat(u_elem[3:6])
        R_node2 = _rotvec_to_rotmat(u_elem[9:12])
        R_def1 = R_cr @ R_node1 @ R_0.T
        R_def2 = R_cr @ R_node2 @ R_0.T
        theta_def1 = _rotmat_to_rotvec(R_def1)
        theta_def2 = _rotmat_to_rotvec(R_def2)

        d_cr = np.zeros(12, dtype=float)
        d_cr[3:6] = theta_def1
        d_cr[6] = L_def - L_0
        d_cr[9:12] = theta_def2

        Ke_local = timo_beam3d_ke_local(E, G, A, Iy, Iz, J, L_0, ky, kz)
        f_cr = Ke_local @ d_cr

        # f_cr: [Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2]
        N_list.append(f_cr[6])  # node2 axial force
        My_list.append(f_cr[10])  # node2 My
        Mz_list.append(f_cr[11])  # node2 Mz

    return np.array(N_list), np.array(My_list), np.array(Mz_list)


def _compute_curvature_from_forces(forces, E, Iy, Iz):
    """断面力から各要素の曲率を計算."""
    n = len(forces)
    kappa_y = np.zeros(n)
    kappa_z = np.zeros(n)
    for i, (_, f2) in enumerate(forces):
        kappa_y[i] = f2.My / (E * Iy) if E * Iy > 0 else 0.0
        kappa_z[i] = f2.Mz / (E * Iz) if E * Iz > 0 else 0.0
    return kappa_y, kappa_z


# ====================================================================
# TestCRStressPhysics: 応力の物理的妥当性
# ====================================================================


class TestCRStressPhysics:
    """CR非線形解析後の応力分布が物理的に妥当であることを検証."""

    def test_tip_load_stress_order_of_magnitude(self):
        """先端荷重: 最大曲げ応力が PL/Z のオーダーと一致.

        ここでZ=I/c（断面係数）。δ/L ~ 5%程度の中変形域で検証。
        """
        data = _build_cantilever_cr(n_elems=20)
        # 先端たわみ ≈ 5% L程度の荷重
        # δ = PL³/(3EI) → P = 3EIδ/L³
        delta_target = 0.05 * data["L"]
        P = 3.0 * data["E"] * data["Iy"] * delta_target / data["L"] ** 3

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"] + 1] = P  # y方向先端荷重

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=5)
        assert result.converged

        forces = _extract_section_forces(data, result.u)

        # 固定端（要素0）の最大曲げ応力を計算
        r = data["d"] / 2.0
        sigma_max = beam3d_max_bending_stress(forces[0][0], data["A"], data["Iy"], data["Iz"], r, r)

        # 解析解: σ = P*L / (I/c) = P*L*c/I
        sigma_analytical = P * data["L"] * r / data["Iy"]

        # 5%変形ではCR非線形性の影響は小さい。2倍以内であるべき
        ratio = sigma_max / sigma_analytical
        assert 0.5 < ratio < 2.0, (
            f"応力オーダーが解析解と乖離: σ_FEM={sigma_max:.2e}, "
            f"σ_analytical={sigma_analytical:.2e}, ratio={ratio:.2f}"
        )

    def test_moment_monotonic_decrease_from_root(self):
        """先端集中荷重: 要素曲げモーメント（CR断面力）は固定端から先端に向かって減少.

        CR corotatedフレームでの断面力を要素ごとに抽出して検証。
        M(x) = P(L-x) なので固定端が最大、先端がゼロ。
        """
        data = _build_cantilever_cr(n_elems=20)
        delta_target = 0.03 * data["L"]
        P = 3.0 * data["E"] * data["Iy"] * delta_target / data["L"] ** 3

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"] + 1] = P

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=5)
        assert result.converged

        _, _, Mz = _extract_cr_element_moments(data, result.u)
        moment_abs = np.abs(Mz)

        # 固定端側の要素 > 先端側の要素
        assert moment_abs[0] > moment_abs[-1], (
            f"モーメントが固定端→先端で減少していない: "
            f"M[0]={moment_abs[0]:.2e}, M[-1]={moment_abs[-1]:.2e}"
        )

        # 大部分で単調減少（数値誤差で若干の非単調は許容）
        n_violations = 0
        for i in range(len(moment_abs) - 1):
            if moment_abs[i] < moment_abs[i + 1] - 1e-6 * moment_abs[0]:
                n_violations += 1
        assert n_violations <= 2, (
            f"モーメントの単調減少違反が多すぎる: {n_violations}/{len(moment_abs) - 1}"
        )

    def test_axial_stress_uniform_under_tension(self):
        """軸引張: 全要素で軸応力が一様（P/A）."""
        data = _build_cantilever_cr(n_elems=10)
        P = 1000.0

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"]] = P  # x方向（軸方向）引張

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=3)
        assert result.converged

        forces = _extract_section_forces(data, result.u)
        axial_stress = np.array([f2.N / data["A"] for _, f2 in forces])

        # 全要素の軸応力がP/Aに近い
        sigma_expected = P / data["A"]
        for i, sigma in enumerate(axial_stress):
            assert abs(sigma - sigma_expected) / sigma_expected < 0.05, (
                f"要素{i}の軸応力が不均一: σ={sigma:.2e}, expected={sigma_expected:.2e}"
            )


# ====================================================================
# TestCRCurvaturePhysics: 曲率の物理的妥当性
# ====================================================================


class TestCRCurvaturePhysics:
    """CR非線形解析後の曲率分布が物理的に妥当であることを検証."""

    def test_curvature_linear_under_tip_load(self):
        """先端荷重: 曲率分布は（小変形域では）ほぼ線形.

        κ(x) = M(x)/(EI) = P(L-x)/(EI)
        """
        data = _build_cantilever_cr(n_elems=20)
        delta_target = 0.02 * data["L"]  # 2%変形（小変形域）
        P = 3.0 * data["E"] * data["Iy"] * delta_target / data["L"] ** 3

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"] + 1] = P

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=3)
        assert result.converged

        forces = _extract_section_forces(data, result.u)
        _, kappa_z = _compute_curvature_from_forces(forces, data["E"], data["Iy"], data["Iz"])
        kappa = np.abs(kappa_z)

        # 曲率の隣接差分（線形なら一定）
        diffs = np.abs(np.diff(kappa))
        nonzero = diffs[diffs > 1e-15]
        if len(nonzero) >= 2:
            ratio = nonzero.max() / nonzero.min()
            assert ratio < 2.0, f"曲率差分の比が大きすぎる（線形分布でない）: ratio={ratio:.2f}"

    def test_curvature_sign_consistency(self):
        """先端荷重: 全要素で曲率の符号が一致（一方向曲げ）."""
        data = _build_cantilever_cr(n_elems=20)
        delta_target = 0.05 * data["L"]
        P = 3.0 * data["E"] * data["Iy"] * delta_target / data["L"] ** 3

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"] + 1] = P

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=5)
        assert result.converged

        forces = _extract_section_forces(data, result.u)
        _, kappa_z = _compute_curvature_from_forces(forces, data["E"], data["Iy"], data["Iz"])

        # 先端に近い要素ではモーメントが小さくゼロに近いが、
        # 有意なモーメントを持つ要素では符号が一致するべき
        significant = kappa_z[np.abs(kappa_z) > 1e-10]
        if len(significant) > 0:
            signs = np.sign(significant)
            assert np.all(signs == signs[0]) or np.all(signs == -signs[0]), (
                f"曲率の符号が不一致: {signs}"
            )

    def test_curvature_zero_at_free_end(self):
        """先端荷重: 先端要素の曲率はほぼゼロ."""
        data = _build_cantilever_cr(n_elems=20)
        delta_target = 0.03 * data["L"]
        P = 3.0 * data["E"] * data["Iy"] * delta_target / data["L"] ** 3

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"] + 1] = P

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=5)
        assert result.converged

        forces = _extract_section_forces(data, result.u)
        _, kappa_z = _compute_curvature_from_forces(forces, data["E"], data["Iy"], data["Iz"])

        # 先端要素の曲率は固定端の曲率に比べて十分小さい
        kappa_root = abs(kappa_z[0])
        kappa_tip = abs(kappa_z[-1])
        if kappa_root > 1e-15:
            assert kappa_tip / kappa_root < 0.15, (
                f"先端曲率が大きすぎる: κ_tip/κ_root={kappa_tip / kappa_root:.4f}"
            )


# ====================================================================
# TestCRLoadOrderPhysics: 荷重オーダーの妥当性
# ====================================================================


class TestCRLoadOrderPhysics:
    """出力荷重が剛性×変位の解析解と同オーダーであることを検証."""

    def test_reaction_force_order(self):
        """固定端反力が外力と平衡.

        ΣF=0: 固定端反力 = 外力（力の釣り合い）
        """
        data = _build_cantilever_cr(n_elems=20)
        P = 500.0

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"] + 1] = P

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=5)
        assert result.converged

        # 内力の計算
        nodes, conn = data["nodes"], data["conn"]
        E, G = data["E"], data["G"]
        A, Iy, Iz, J = data["A"], data["Iy"], data["Iz"], data["J"]
        ky, kz = data["kappa_y"], data["kappa_z"]

        _, f_int = assemble_cr_beam3d(
            nodes,
            conn,
            result.u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            ky,
            kz,
            stiffness=False,
            internal_force=True,
            sparse=False,
        )

        # 固定端のy方向反力
        reaction_y = f_int[1]  # node0のy方向

        # 力の釣り合い: |reaction_y + P| ≈ 0（数値誤差許容）
        assert abs(reaction_y + P) / P < 0.05, (
            f"力の釣り合いが不成立: reaction={reaction_y:.4f}, P={P:.4f}"
        )

    def test_tip_displacement_order(self):
        """先端変位が解析解のオーダーと一致.

        δ = PL³/(3EI) （線形解析解）
        中変形域では非線形効果で若干異なるが、オーダーは同じ。
        """
        data = _build_cantilever_cr(n_elems=20)
        P = 100.0

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"] + 1] = P

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=5)
        assert result.converged

        # FEM先端y変位
        tip_y = result.u[6 * data["n_elems"] + 1]

        # 解析解
        delta_analytical = P * data["L"] ** 3 / (3.0 * data["E"] * data["Iy"])

        # δ/L が小さければ線形解と近い
        ratio = tip_y / delta_analytical
        assert 0.8 < ratio < 1.2, (
            f"先端変位が解析解と乖離: δ_FEM={tip_y:.6e}, "
            f"δ_analytical={delta_analytical:.6e}, ratio={ratio:.4f}"
        )


# ====================================================================
# TestCRSymmetryPhysics: 対称性の検証
# ====================================================================


class TestCRSymmetryPhysics:
    """対称荷重に対して対称変形が得られることを検証."""

    def test_simply_supported_symmetric_load(self):
        """単純支持梁の中央集中荷重 → 対称変形."""
        n_elems = 20
        L = 1.0
        E = 2.1e11
        nu = 0.3
        G = E / (2.0 * (1.0 + nu))
        d = 0.02
        sec = BeamSection.circle(d)
        kappa = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)

        nodes = np.zeros((n_elems + 1, 3))
        nodes[:, 0] = np.linspace(0, L, n_elems + 1)
        conn = np.array([[i, i + 1] for i in range(n_elems)])

        n_nodes = n_elems + 1
        ndof = 6 * n_nodes

        # 単純支持: 節点0の変位(x,y,z)固定、最終節点のy,z固定
        fixed_dofs = np.array([0, 1, 2, 6 * n_elems + 1, 6 * n_elems + 2])

        # 中央集中荷重（下向き）
        mid_node = n_elems // 2
        P = 200.0
        f_ext = np.zeros(ndof)
        f_ext[6 * mid_node + 1] = -P

        data = {
            "nodes": nodes,
            "conn": conn,
            "E": E,
            "G": G,
            "A": sec.A,
            "Iy": sec.Iy,
            "Iz": sec.Iz,
            "J": sec.J,
            "kappa_y": kappa,
            "kappa_z": kappa,
            "d": d,
            "L": L,
            "ndof": ndof,
            "n_nodes": n_nodes,
            "n_elems": n_elems,
            "fixed_dofs": fixed_dofs,
        }

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=5)
        assert result.converged

        # y方向変位の対称性チェック
        uy = np.array([result.u[6 * i + 1] for i in range(n_nodes)])
        n_half = n_elems // 2

        for i in range(1, n_half):
            left = uy[i]
            right = uy[n_elems - i]
            max_val = max(abs(left), abs(right))
            if max_val > 1e-15:
                sym_error = abs(left - right) / max_val
                assert sym_error < 0.02, (
                    f"対称性違反: uy[{i}]={left:.6e}, uy[{n_elems - i}]={right:.6e}, "
                    f"error={sym_error:.4f}"
                )

    def test_opposite_loads_give_opposite_displacements(self):
        """正負逆の荷重 → 正負逆の変位（小変形域）."""
        data = _build_cantilever_cr(n_elems=10)
        P = 50.0  # 小荷重（小変形域を確保）

        f_pos = np.zeros(data["ndof"])
        f_pos[6 * data["n_elems"] + 1] = P
        f_neg = np.zeros(data["ndof"])
        f_neg[6 * data["n_elems"] + 1] = -P

        r_pos = _solve_cr_nonlinear(data, f_pos, n_load_steps=3)
        r_neg = _solve_cr_nonlinear(data, f_neg, n_load_steps=3)
        assert r_pos.converged and r_neg.converged

        # u(+P) ≈ -u(-P) （小変形では厳密に成立）
        max_disp = np.max(np.abs(r_pos.u))
        diff = np.max(np.abs(r_pos.u + r_neg.u))
        if max_disp > 1e-15:
            sym_error = diff / max_disp
            assert sym_error < 0.05, f"正負荷重の反対称性が不成立: error={sym_error:.4f}"


# ====================================================================
# TestCRDeformationPhysics: 大変形の物理的性質
# ====================================================================


class TestCRDeformationPhysics:
    """CR大変形解析の物理的性質を検証."""

    def test_large_deformation_stiffening(self):
        """大変形で幾何学的剛性効果が現れる（非線形 < 線形の変位）.

        幾何学的非線形を考慮すると、大変形域では
        実際の変位は線形解析解よりも小さくなる（幾何学的剛性効果）。
        """
        data = _build_cantilever_cr(n_elems=20)
        # 大荷重（δ/L ~ 10%以上）
        delta_target = 0.15 * data["L"]
        P = 3.0 * data["E"] * data["Iy"] * delta_target / data["L"] ** 3

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"] + 1] = P

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=20, tol=1e-5)
        assert result.converged

        # FEM先端y変位
        tip_y_nl = abs(result.u[6 * data["n_elems"] + 1])

        # 線形解析解
        delta_linear = P * data["L"] ** 3 / (3.0 * data["E"] * data["Iy"])

        # 非線形解の変位 < 線形解（幾何学的剛性効果）
        assert tip_y_nl < delta_linear, (
            f"幾何学的剛性効果が見られない: δ_NL={tip_y_nl:.6e} >= δ_linear={delta_linear:.6e}"
        )

    def test_deformed_shape_smooth(self):
        """大変形後の変形形状が滑らか（隣接節点間の変位差に大きなジャンプがない）."""
        data = _build_cantilever_cr(n_elems=20)
        delta_target = 0.1 * data["L"]
        P = 3.0 * data["E"] * data["Iy"] * delta_target / data["L"] ** 3

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"] + 1] = P

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=10, tol=1e-5)
        assert result.converged

        # y方向変位の隣接差分
        uy = np.array([result.u[6 * i + 1] for i in range(data["n_nodes"])])
        diffs = np.abs(np.diff(uy))

        # 変位差分が滑らか（隣接差分の最大/平均比が小さい）
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        if mean_diff > 1e-15:
            ratio = max_diff / mean_diff
            assert ratio < 3.0, f"変形形状が不連続: max/mean ratio={ratio:.2f}"

    def test_load_path_independence_small_deformation(self):
        """小変形域: 荷重ステップ数に依存しない（経路独立）."""
        data = _build_cantilever_cr(n_elems=10)
        P = 30.0  # 小荷重

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"] + 1] = P

        # 3ステップと10ステップで比較
        r3 = _solve_cr_nonlinear(data, f_ext, n_load_steps=3)
        r10 = _solve_cr_nonlinear(data, f_ext, n_load_steps=10)
        assert r3.converged and r10.converged

        # 最終変位が一致（小変形では荷重ステップ数に依存しない）
        max_disp = max(np.max(np.abs(r3.u)), np.max(np.abs(r10.u)))
        if max_disp > 1e-15:
            diff = np.max(np.abs(r3.u - r10.u)) / max_disp
            assert diff < 0.01, f"荷重ステップ依存性あり: 差={diff:.6f}"

    def test_strain_energy_equals_external_work(self):
        """内部歪みエネルギー ≈ 外力仕事（エネルギーバランス）.

        U = ½ u^T f_int = ½ u^T f_ext（平衡状態では一致）
        """
        data = _build_cantilever_cr(n_elems=20)
        P = 200.0

        f_ext = np.zeros(data["ndof"])
        f_ext[6 * data["n_elems"] + 1] = P

        result = _solve_cr_nonlinear(data, f_ext, n_load_steps=10)
        assert result.converged

        # 内力を計算
        nodes, conn = data["nodes"], data["conn"]
        E, G = data["E"], data["G"]
        A, Iy, Iz, J = data["A"], data["Iy"], data["Iz"], data["J"]
        ky, kz = data["kappa_y"], data["kappa_z"]

        _, f_int = assemble_cr_beam3d(
            nodes,
            conn,
            result.u,
            E,
            G,
            A,
            Iy,
            Iz,
            J,
            ky,
            kz,
            stiffness=False,
            internal_force=True,
            sparse=False,
        )

        # 歪みエネルギー U_int = ½ u · f_int
        U_int = 0.5 * result.u @ f_int

        # 外力仕事 W_ext = ½ u · f_ext（線形荷重パスの場合近似的に成立）
        W_ext = 0.5 * result.u @ f_ext

        # エネルギーバランス（非線形では完全一致しないが同オーダー）
        if abs(W_ext) > 1e-15:
            ratio = U_int / W_ext
            assert 0.8 < ratio < 1.2, (
                f"エネルギーバランス不成立: U_int={U_int:.6e}, W_ext={W_ext:.6e}, ratio={ratio:.4f}"
            )
