"""Line contact + Mortar + Alart-Curnier 摩擦の3重統合テスト.

status-085 TODO: 3つの高精度接触機能を同時に有効化した際の収束性と物理的妥当性を検証。
- Line-to-line Gauss 積分（Phase C6-L1）
- Mortar 離散化（Phase C6-L5）
- Alart-Curnier 摩擦拡大鞍点系（Phase S1）
"""

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.pair import ContactConfig, ContactManager
from xkep_cae.contact.solver_ncp import (
    NCPSolveResult,
    newton_raphson_contact_ncp,
)


def _make_spring_system(
    k_spring: float = 1e4,
    ndof_per_node: int = 6,
    z_sep: float = 0.041,
    radii: float = 0.04,
):
    """2本の直交ばね梁モデル（test_mortar.py / test_ncp_friction_line.py と同一）."""
    node_coords_ref = np.array(
        [
            [0.0, 0.0, z_sep],
            [1.0, 0.0, z_sep],
            [0.5, -0.5, -z_sep],
            [0.5, 0.5, -z_sep],
        ]
    )
    connectivity = np.array([[0, 1], [2, 3]])
    n_nodes = 4
    ndof_total = n_nodes * ndof_per_node
    k_rot = k_spring * 0.01

    def assemble_tangent(u):
        K = sp.lil_matrix((ndof_total, ndof_total))
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * ndof_per_node + d
                d1 = n1 * ndof_per_node + d
                K[d0, d0] += k_spring
                K[d0, d1] -= k_spring
                K[d1, d0] -= k_spring
                K[d1, d1] += k_spring
            for d in range(3, ndof_per_node):
                d0 = n0 * ndof_per_node + d
                d1 = n1 * ndof_per_node + d
                K[d0, d0] += k_rot
                K[d0, d1] -= k_rot
                K[d1, d0] -= k_rot
                K[d1, d1] += k_rot
        return K.tocsr()

    def assemble_internal_force(u):
        f_int = np.zeros(ndof_total)
        for elem_nodes in connectivity:
            n0, n1 = elem_nodes
            for d in range(3):
                d0 = n0 * ndof_per_node + d
                d1 = n1 * ndof_per_node + d
                delta = u[d1] - u[d0]
                f_int[d0] -= k_spring * delta
                f_int[d1] += k_spring * delta
            for d in range(3, ndof_per_node):
                d0 = n0 * ndof_per_node + d
                d1 = n1 * ndof_per_node + d
                delta = u[d1] - u[d0]
                f_int[d0] -= k_rot * delta
                f_int[d1] += k_rot * delta
        return f_int

    fixed_dofs = np.array(
        [0 * ndof_per_node + d for d in range(ndof_per_node)]
        + [2 * ndof_per_node + d for d in range(ndof_per_node)],
        dtype=int,
    )

    return (
        node_coords_ref,
        connectivity,
        radii,
        ndof_total,
        ndof_per_node,
        assemble_tangent,
        assemble_internal_force,
        fixed_dofs,
    )


def _make_multi_segment_system(
    n_seg: int = 3,
    k_spring: float = 1e4,
    ndof_per_node: int = 6,
    z_sep: float = 0.041,
    radii: float = 0.04,
):
    """A/B各n_seg+1節点のマルチセグメントばね梁モデル."""
    n_per_beam = n_seg + 1
    n_nodes = 2 * n_per_beam
    ndof = n_nodes * ndof_per_node

    coords = np.zeros((n_nodes, 3))
    for i in range(n_per_beam):
        coords[i] = [float(i), 0.0, z_sep]
        coords[n_per_beam + i] = [float(i), 0.0, -z_sep]

    conn_a = [[i, i + 1] for i in range(n_per_beam - 1)]
    conn_b = [[n_per_beam + i, n_per_beam + i + 1] for i in range(n_per_beam - 1)]
    connectivity = np.array(conn_a + conn_b)

    K_diag = np.ones(ndof) * k_spring
    K_global = sp.diags(K_diag).tocsr()

    def assemble_tangent(u):
        return K_global.copy()

    def assemble_internal(u):
        return K_global @ u

    fixed_dofs = np.array(
        [i * ndof_per_node + d for i in range(n_nodes) for d in range(ndof_per_node) if d != 2],
        dtype=int,
    )

    f_ext = np.zeros(ndof)
    for i in range(n_per_beam):
        f_ext[i * ndof_per_node + 2] = -3.0
    for i in range(n_per_beam, n_nodes):
        f_ext[i * ndof_per_node + 2] = 3.0

    return {
        "node_coords_ref": coords,
        "connectivity": connectivity,
        "radii": radii,
        "ndof": ndof,
        "ndof_per_node": ndof_per_node,
        "assemble_tangent": assemble_tangent,
        "assemble_internal": assemble_internal,
        "fixed_dofs": fixed_dofs,
        "f_ext": f_ext,
    }


# ============================================================
# Line contact + Mortar + Alart-Curnier 摩擦の3重統合テスト
# ============================================================


class TestTripleIntegration:
    """Line contact + Mortar + Alart-Curnier 摩擦の3重統合."""

    def test_triple_converges(self):
        """Mortar + line_contact + friction の3重統合で NCP 収束."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.035, radii=0.04)

        config = ContactConfig(
            k_pen_scale=1e5,
            g_on=0.01,
            g_off=0.02,
            k_t_ratio=0.5,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )
        mgr = ContactManager(config=config)

        f_ext = np.zeros(ndof)
        f_ext[1 * ndof_per_node + 2] = -20.0
        f_ext[3 * ndof_per_node + 2] = 20.0

        result = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr,
            coords,
            conn,
            radii,
            n_load_steps=5,
            max_iter=50,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.2,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )

        assert isinstance(result, NCPSolveResult)
        assert result.converged, "Triple integration (Mortar+LC+friction) did not converge"

    def test_triple_lambda_nonneg(self):
        """3重統合でも λ >= 0 が保証される."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.035, radii=0.04)

        config = ContactConfig(
            k_pen_scale=1e5,
            g_on=0.01,
            g_off=0.02,
            k_t_ratio=0.5,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )
        mgr = ContactManager(config=config)

        f_ext = np.zeros(ndof)
        f_ext[1 * ndof_per_node + 2] = -20.0
        f_ext[3 * ndof_per_node + 2] = 20.0

        result = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr,
            coords,
            conn,
            radii,
            n_load_steps=5,
            max_iter=50,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.1,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )

        assert result.converged
        assert np.all(result.lambdas >= -1e-12), "λ must be non-negative"

    def test_triple_lateral_force(self):
        """3重統合 + 横方向荷重で Alart-Curnier 拡大系が収束."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.035, radii=0.04)

        config = ContactConfig(
            k_pen_scale=1e5,
            g_on=0.01,
            g_off=0.02,
            k_t_ratio=0.5,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )
        mgr = ContactManager(config=config)

        f_ext = np.zeros(ndof)
        # 法線方向 + 横方向荷重
        f_ext[1 * ndof_per_node + 2] = -20.0
        f_ext[3 * ndof_per_node + 2] = 20.0
        f_ext[1 * ndof_per_node + 0] = 5.0  # x方向横荷重

        result = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr,
            coords,
            conn,
            radii,
            n_load_steps=10,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.3,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )

        assert result.converged, "Triple + lateral force did not converge"

    def test_triple_friction_constrains_displacement(self):
        """3重統合で摩擦が横方向変位を拘束することを検証."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.035, radii=0.04)

        f_ext = np.zeros(ndof)
        f_ext[1 * ndof_per_node + 2] = -20.0
        f_ext[3 * ndof_per_node + 2] = 20.0
        f_ext[1 * ndof_per_node + 0] = 5.0

        # Mortar + line contact のみ（摩擦なし）
        mgr_no_fric = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.01,
                g_off=0.02,
                line_contact=True,
                n_gauss=2,
                use_mortar=True,
            ),
        )
        res_no_fric = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr_no_fric,
            coords,
            conn,
            radii,
            n_load_steps=5,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )

        # 3重統合（摩擦あり）
        mgr_fric = ContactManager(
            config=ContactConfig(
                k_pen_scale=1e5,
                g_on=0.01,
                g_off=0.02,
                k_t_ratio=0.5,
                line_contact=True,
                n_gauss=2,
                use_mortar=True,
            ),
        )
        res_fric = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr_fric,
            coords,
            conn,
            radii,
            n_load_steps=10,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.3,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )

        assert res_no_fric.converged
        assert res_fric.converged

        # 摩擦ありの方が横方向変位が小さい
        ux_no = res_no_fric.u[1 * ndof_per_node + 0]
        ux_yes = res_fric.u[1 * ndof_per_node + 0]
        assert abs(ux_yes) <= abs(ux_no) + 1e-10, (
            f"摩擦が横方向変位を拘束していない: |ux_yes|={abs(ux_yes):.6e} > |ux_no|={abs(ux_no):.6e}"
        )

    def test_triple_mu_ramp(self):
        """3重統合 + μランプで収束する."""
        (
            coords,
            conn,
            radii,
            ndof,
            ndof_per_node,
            K_fn,
            f_int_fn,
            fixed_dofs,
        ) = _make_spring_system(z_sep=0.035, radii=0.04)

        config = ContactConfig(
            k_pen_scale=1e5,
            g_on=0.01,
            g_off=0.02,
            k_t_ratio=0.5,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )
        mgr = ContactManager(config=config)

        f_ext = np.zeros(ndof)
        f_ext[1 * ndof_per_node + 2] = -20.0
        f_ext[3 * ndof_per_node + 2] = 20.0

        result = newton_raphson_contact_ncp(
            f_ext,
            fixed_dofs,
            K_fn,
            f_int_fn,
            mgr,
            coords,
            conn,
            radii,
            n_load_steps=10,
            max_iter=100,
            tol_force=1e-6,
            tol_ncp=1e-6,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.3,
            mu_ramp_steps=5,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )

        assert result.converged, "Triple + mu_ramp did not converge"


class TestTripleMultiSegment:
    """マルチセグメントでの3重統合テスト."""

    def test_multi_segment_triple_converges(self):
        """3セグメント梁で Mortar + line_contact + friction が収束."""
        sys = _make_multi_segment_system(n_seg=3, z_sep=0.041, radii=0.04)

        config = ContactConfig(
            k_pen_scale=1e5,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
            k_t_ratio=0.5,
            g_on=0.01,
            g_off=0.02,
        )
        mgr = ContactManager(config=config)

        result = newton_raphson_contact_ncp(
            sys["f_ext"],
            sys["fixed_dofs"],
            sys["assemble_tangent"],
            sys["assemble_internal"],
            mgr,
            sys["node_coords_ref"],
            sys["connectivity"],
            sys["radii"],
            n_load_steps=5,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.1,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )

        assert result.converged, "Multi-segment triple integration did not converge"

    def test_multi_segment_triple_vs_mortar_only(self):
        """マルチセグメントで Mortar+LC+摩擦 vs Mortar+LC のz変位方向が一致."""
        sys = _make_multi_segment_system(n_seg=3, z_sep=0.041, radii=0.04)

        # Mortar + LC のみ
        config_no_fric = ContactConfig(
            k_pen_scale=1e5,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )
        mgr_no_fric = ContactManager(config=config_no_fric)
        res_no_fric = newton_raphson_contact_ncp(
            sys["f_ext"],
            sys["fixed_dofs"],
            sys["assemble_tangent"],
            sys["assemble_internal"],
            mgr_no_fric,
            sys["node_coords_ref"],
            sys["connectivity"],
            sys["radii"],
            n_load_steps=3,
            max_iter=30,
            show_progress=False,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )

        # Mortar + LC + 摩擦
        config_fric = ContactConfig(
            k_pen_scale=1e5,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
            k_t_ratio=0.5,
            g_on=0.01,
            g_off=0.02,
        )
        mgr_fric = ContactManager(config=config_fric)
        res_fric = newton_raphson_contact_ncp(
            sys["f_ext"],
            sys["fixed_dofs"],
            sys["assemble_tangent"],
            sys["assemble_internal"],
            mgr_fric,
            sys["node_coords_ref"],
            sys["connectivity"],
            sys["radii"],
            n_load_steps=5,
            max_iter=50,
            show_progress=False,
            broadphase_margin=0.05,
            use_friction=True,
            mu=0.1,
            line_contact=True,
            n_gauss=2,
            use_mortar=True,
        )

        assert res_no_fric.converged
        assert res_fric.converged

        # z方向変位の符号が一致（A梁は下方、B梁は上方）
        ndof_per_node = sys["ndof_per_node"]
        uz_no = res_no_fric.u[0 * ndof_per_node + 2]
        uz_yes = res_fric.u[0 * ndof_per_node + 2]
        if abs(uz_no) > 1e-8 and abs(uz_yes) > 1e-8:
            assert np.sign(uz_no) == np.sign(uz_yes)
