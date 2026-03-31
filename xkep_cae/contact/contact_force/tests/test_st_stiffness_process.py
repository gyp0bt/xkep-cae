"""ContactForceStStiffnessProcess のテスト（status-256 B1）."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.contact_force import (
    ContactForceStStiffnessInput,
    ContactForceStStiffnessProcess,
)
from xkep_cae.core.testing import binds_to


def _make_pair(
    s: float = 0.5,
    t: float = 0.5,
    gap: float = -0.01,
    p_n: float = 1.0,
    radius_a: float = 0.05,
    radius_b: float = 0.05,
    nodes_a: tuple[int, int] = (0, 1),
    nodes_b: tuple[int, int] = (2, 3),
) -> SimpleNamespace:
    """テスト用の接触ペアを生成."""
    state = SimpleNamespace(
        s=s,
        t=t,
        gap=gap,
        p_n=p_n,
        normal=np.array([0.0, 1.0, 0.0]),
        tangent1=np.array([1.0, 0.0, 0.0]),
        tangent2=np.array([0.0, 0.0, 1.0]),
    )
    return SimpleNamespace(
        state=state,
        nodes_a=nodes_a,
        nodes_b=nodes_b,
        radius_a=radius_a,
        radius_b=radius_b,
    )


@binds_to(ContactForceStStiffnessProcess)
class TestContactForceStStiffnessProcess:
    """ContactForceStStiffnessProcess の単体テスト（status-256 B1）."""

    def test_empty_pairs(self):
        proc = ContactForceStStiffnessProcess()
        out = proc.process(
            ContactForceStStiffnessInput(
                pairs=[],
                node_coords=np.zeros((4, 3)),
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        assert isinstance(out.K_st, sp.csr_matrix)
        assert out.K_st.shape == (24, 24)
        assert out.K_st.nnz == 0

    def test_single_pair_nonzero(self):
        """近接平行セグメントで K_st が非ゼロ."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [1.0, 0.1, 0.0],
            ]
        )
        pair = _make_pair(gap=-0.01, p_n=1.0)
        proc = ContactForceStStiffnessProcess()
        out = proc.process(
            ContactForceStStiffnessInput(
                pairs=[pair],
                node_coords=coords,
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        assert out.K_st.shape == (24, 24)
        assert out.K_st.nnz > 0

    def test_inactive_pair_zero(self):
        """p_n=0 のペアは K_st に寄与しない."""
        coords = np.zeros((4, 3))
        pair = _make_pair(p_n=0.0)
        proc = ContactForceStStiffnessProcess()
        out = proc.process(
            ContactForceStStiffnessInput(
                pairs=[pair],
                node_coords=coords,
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        assert out.K_st.nnz == 0

    def test_meta(self):
        assert ContactForceStStiffnessProcess.meta.name == "ContactForceStStiffness"


class TestKstNonlocalFD:
    """K_st 隣接ノードDOF拡張の手動公式検証（status-272, status-275で非平行座標化）.

    3要素チェーン（0-1, 1-2, 2-3）+ 別の3要素チェーン（4-5, 5-6, 6-7）
    接触ペア: elem_a=0 (nodes 0,1), elem_b=5 (nodes 6,7)
    隣接ノード: A+2=2, B-1=5

    非平行座標配置にすることで StJacobian が非特異となり ds_du_adj ≠ None。
    """

    def _make_chain_coords(self):
        """2本の3要素チェーン（非平行配置でStJacobian非特異化）."""
        return np.array(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [2.0, 0.1, 0.0],  # 2
                [3.0, 0.3, 0.0],  # 3
                [0.0, 0.2, 0.0],  # 4
                [1.0, 0.15, 0.0],  # 5
                [2.0, 0.2, 0.0],  # 6
                [3.0, 0.35, 0.0],  # 7
            ]
        )

    def _make_connectivity(self):
        return np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7]])

    def _compute_kst(self, coords):
        """与えられた座標でK_stを計算."""
        from xkep_cae.contact.geometry._compute import (
            _compute_adj_node_map,
            _compute_node_counts,
            _compute_node_tangents,
        )

        conn = self._make_connectivity()
        node_tangents = _compute_node_tangents(coords, conn)
        node_counts = _compute_node_counts(len(coords), conn)
        adj_node_map = _compute_adj_node_map(conn)

        pair = SimpleNamespace(
            state=SimpleNamespace(
                s=0.5,
                t=0.5,
                gap=-0.01,
                p_n=1.0,
                normal=np.array([0.0, 1.0, 0.0]),
            ),
            nodes_a=(0, 1),
            nodes_b=(6, 7),
            radius_a=0.05,
            radius_b=0.05,
            elem_a=0,
            elem_b=5,
        )

        proc = ContactForceStStiffnessProcess()
        out = proc.process(
            ContactForceStStiffnessInput(
                pairs=[pair],
                node_coords=coords,
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=len(coords) * 6,
                ndof_per_node=6,
                use_hermite=True,
                node_tangents=node_tangents,
                node_counts=node_counts,
                adj_node_map=adj_node_map,
            )
        )
        return out.K_st

    def test_kst_adj_manual_formula(self):
        """K_st_adj が実コードと同じ df_ds/df_dt の公式と一致."""
        from xkep_cae.contact.contact_force.strategy import (
            _hermite_corrected_coeffs,
            _huber_deriv_scalar,
        )
        from xkep_cae.contact.geometry._compute import (
            _compute_dm_coeffs,
            _compute_dm_ext_coeffs,
            _compute_node_counts,
            _compute_node_tangents,
        )
        from xkep_cae.contact.geometry._st_jacobian import (
            ComputeStJacobianProcess,
            StJacobianInput,
            _hermite_deriv_scalar,
        )

        coords = self._make_chain_coords()
        K_st = self._compute_kst(coords)

        conn = self._make_connectivity()
        node_tangents = _compute_node_tangents(coords, conn)
        node_counts = _compute_node_counts(len(coords), conn)

        nodes_a = (0, 1)
        nodes_b = (6, 7)
        s, t = 0.5, 0.5
        gap = -0.01
        p_n = 1.0
        normal = np.array([0.0, 1.0, 0.0])
        k_pen = 1e4
        delta_h = 100.0
        rA, rB = 0.05, 0.05
        dist = gap + rA + rB

        # dm 係数
        dm_A = _compute_dm_coeffs(node_counts[nodes_a[0]], node_counts[nodes_a[1]])
        dm_B = _compute_dm_coeffs(node_counts[nodes_b[0]], node_counts[nodes_b[1]])
        dm_ext_A = _compute_dm_ext_coeffs(node_counts[nodes_a[0]], node_counts[nodes_a[1]])
        dm_ext_B = _compute_dm_ext_coeffs(node_counts[nodes_b[0]], node_counts[nodes_b[1]])
        coeffs, dc_ds, dc_dt = _hermite_corrected_coeffs(s, t, dm_A, dm_B)

        # ∂n/∂s, ∂n/∂t（Hermite）
        dpA = _hermite_deriv_scalar(
            s,
            coords[nodes_a[0]],
            coords[nodes_a[1]],
            node_tangents[nodes_a[0]],
            node_tangents[nodes_a[1]],
        )
        dpB = _hermite_deriv_scalar(
            t,
            coords[nodes_b[0]],
            coords[nodes_b[1]],
            node_tangents[nodes_b[0]],
            node_tangents[nodes_b[1]],
        )
        P_perp = np.eye(3) - np.outer(normal, normal)
        dn_ds = (1.0 / dist) * P_perp @ dpA
        dn_dt = -(1.0 / dist) * P_perp @ dpB

        # ∂p_n/∂s, ∂p_n/∂t
        x_p = k_pen * (-gap)
        h_d = _huber_deriv_scalar(x_p, delta_h)
        dgap_ds = float(np.dot(normal, dpA))
        dgap_dt = -float(np.dot(normal, dpB))
        dpn_ds = h_d * k_pen * (-dgap_ds)
        dpn_dt = h_d * k_pen * (-dgap_dt)

        # g_shape
        g_shape = np.zeros(12)
        for k in range(4):
            for i in range(3):
                g_shape[k * 3 + i] = coeffs[k] * normal[i]

        # 完全な df_ds, df_dt（3寄与: ∂p_n/∂s, dc/ds, dn/ds）
        df_ds = np.zeros(12)
        df_dt = np.zeros(12)
        for k in range(4):
            for i in range(3):
                li = k * 3 + i
                df_ds[li] = dpn_ds * g_shape[li] + p_n * (
                    dc_ds[k] * normal[i] + coeffs[k] * dn_ds[i]
                )
                df_dt[li] = dpn_dt * g_shape[li] + p_n * (
                    dc_dt[k] * normal[i] + coeffs[k] * dn_dt[i]
                )

        # StJacobian の隣接ノード微分を取得
        st_in = StJacobianInput(
            xA0=coords[nodes_a[0]],
            xA1=coords[nodes_a[1]],
            xB0=coords[nodes_b[0]],
            xB1=coords[nodes_b[1]],
            s=s,
            t=t,
            mA0=node_tangents[nodes_a[0]],
            mA1=node_tangents[nodes_a[1]],
            mB0=node_tangents[nodes_b[0]],
            mB1=node_tangents[nodes_b[1]],
            use_hermite=True,
            dm_A=dm_A,
            dm_B=dm_B,
            dm_ext_A=dm_ext_A,
            dm_ext_B=dm_ext_B,
        )
        st_out = ComputeStJacobianProcess().process(st_in)
        assert st_out.valid
        assert st_out.ds_du_adj is not None, "非平行配置で ds_du_adj が None"

        # K_st_adj = -(outer(df_ds, ds_du_adj) + outer(df_dt, dt_du_adj))
        K_adj_expected = -(np.outer(df_ds, st_out.ds_du_adj) + np.outer(df_dt, st_out.dt_du_adj))

        # K_st の隣接ノード列を抽出して比較
        ndpn = 6
        row_nodes = [0, 1, 6, 7]
        row_dofs = []
        for n in row_nodes:
            for d in range(3):
                row_dofs.append(n * ndpn + d)

        # ds_du_adj レイアウト: [A-1(3), A+2(3), B-1(3), B+2(3)] = 12列
        # adj_node_map[0] = (-1, 2), adj_node_map[5] = (5, -1)
        # A-1=-1(端点), A+2=2, B-1=5, B+2=-1(端点)
        adj_col_map = {2: slice(3, 6), 5: slice(6, 9)}  # adj列のスライス

        for adj_node, adj_sl in adj_col_map.items():
            for d in range(3):
                col_dof = adj_node * ndpn + d
                k_col = np.array([K_st[ri, col_dof] for ri in row_dofs])
                expected_col = K_adj_expected[:, adj_sl.start + d]
                np.testing.assert_allclose(
                    k_col,
                    expected_col,
                    atol=1e-10,
                    err_msg=f"K_st adj formula: adj_node={adj_node}, d={d}",
                )

    def test_kst_adj_nonzero(self):
        """非平行配置で隣接ノード列に非ゼロ値が存在."""
        coords = self._make_chain_coords()
        K_st = self._compute_kst(coords)

        ndpn = 6
        for adj_node in [2, 5]:
            cols = [adj_node * ndpn + d for d in range(3)]
            vals = []
            for c in cols:
                if sp.issparse(K_st):
                    vals.extend(K_st[:, c].toarray().ravel())
                else:
                    vals.extend(K_st[:, c])
            assert not np.allclose(vals, 0.0), (
                f"Node {adj_node}: K_st adj columns should be nonzero"
            )

    def test_kst_adj_endpoint_zero(self):
        """端点ノードの隣接DOFはK_stに寄与しない."""
        coords = self._make_chain_coords()
        K_st = self._compute_kst(coords)

        ndpn = 6
        for n in [3, 4]:
            for d in range(3):
                col = n * ndpn + d
                col_vals = K_st[:, col].toarray().ravel() if sp.issparse(K_st) else K_st[:, col]
                assert np.allclose(col_vals, 0.0), f"Node {n} d={d} should have zero K_st column"


class TestKcAdjFD:
    """K_c_adj（K_mat+K_geo 隣接ノード拡張）のFD検証（status-273: Step3, status-275修正）.

    3要素チェーン×2本。Hermite補間で隣接ノード位置が接触力に影響。
    consistent_st_tangent=False で K_st を除外し、K_mat+K_geo+K_c_adj のみを検証。

    status-275: 座標を非平行化 + elem_a=1,elem_b=4（中央要素、近接配置）に変更。
    旧テストは elem_a=0,elem_b=5 が空間的に離れており trivially passing だった。
    """

    def _make_chain_coords(self):
        """2本の3要素チェーン（非平行、中央要素が近接・接触あり）."""
        return np.array(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [2.0, 0.02, 0.0],  # 2
                [3.0, 0.06, 0.0],  # 3
                [0.0, 0.04, 0.0],  # 4
                [1.0, 0.01, 0.0],  # 5
                [2.0, 0.03, 0.0],  # 6
                [3.0, 0.08, 0.0],  # 7
            ]
        )

    def _make_connectivity(self):
        return np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7]])

    def _hermite_interp(self, s, x0, x1, m0, m1):
        """Hermite 補間点を計算."""
        s2, s3 = s * s, s * s * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h01 = -2.0 * s3 + 3.0 * s2
        h10 = s3 - 2.0 * s2 + s
        h11 = s3 - s2
        return h00 * x0 + h01 * x1 + h10 * m0 + h11 * m1

    def _compute_force_with_geom(self, coords):
        """座標から gap/normal を再計算して接触力を返す (12,).

        tangent() と整合するよう dm 凍結（_hermite_shape_coeffs を使用）。
        """
        from xkep_cae.contact.contact_force.strategy import (
            _hermite_shape_coeffs,
        )
        from xkep_cae.contact.geometry._compute import (
            _compute_node_tangents,
        )

        conn = self._make_connectivity()
        node_tangents = _compute_node_tangents(coords, conn)

        s, t = 0.5, 0.5
        k_pen = 1e4
        rA, rB = 0.05, 0.05
        nodes_a = (1, 2)
        nodes_b = (5, 6)

        # Hermite 補間点
        pA = self._hermite_interp(
            s,
            coords[nodes_a[0]],
            coords[nodes_a[1]],
            node_tangents[nodes_a[0]],
            node_tangents[nodes_a[1]],
        )
        pB = self._hermite_interp(
            t,
            coords[nodes_b[0]],
            coords[nodes_b[1]],
            node_tangents[nodes_b[0]],
            node_tangents[nodes_b[1]],
        )

        # gap, normal
        d = pA - pB
        dist = np.linalg.norm(d)
        normal = d / dist if dist > 1e-15 else np.array([0.0, 1.0, 0.0])
        gap = dist - rA - rB

        # Huber p_n
        delta_h = 100.0
        x_p = k_pen * (-gap)
        if delta_h <= 0.0:
            p_n = max(0.0, x_p)
        elif x_p < -delta_h:
            p_n = 0.0
        elif x_p > delta_h:
            p_n = x_p
        else:
            p_n = (x_p + delta_h) ** 2 / (4.0 * delta_h)

        # 形状関数係数（dm凍結: tangent()と一致させる）
        coeffs = _hermite_shape_coeffs(s, t)

        # f_c = p_n * c_k * n
        f_local = np.zeros(12)
        for k in range(4):
            for i in range(3):
                f_local[k * 3 + i] = p_n * coeffs[k] * normal[i]
        return f_local

    def _compute_tangent(self, coords):
        """座標から full tangent（K_mat+K_geo+K_c_adj、K_st除外）を計算."""
        from xkep_cae.contact.contact_force.strategy import (
            HuberContactForceProcess,
        )
        from xkep_cae.contact.geometry._compute import (
            _compute_node_tangents,
        )

        conn = self._make_connectivity()
        node_tangents = _compute_node_tangents(coords, conn)

        s, t = 0.5, 0.5
        k_pen = 1e4
        rA, rB = 0.05, 0.05

        # Hermite 補間で gap/normal 計算
        pA = self._hermite_interp(
            s,
            coords[1],
            coords[2],
            node_tangents[1],
            node_tangents[2],
        )
        pB = self._hermite_interp(
            t,
            coords[5],
            coords[6],
            node_tangents[5],
            node_tangents[6],
        )
        d = pA - pB
        dist = np.linalg.norm(d)
        normal = d / dist if dist > 1e-15 else np.array([0.0, 1.0, 0.0])
        gap = dist - rA - rB

        # Huber p_n
        delta_h = 100.0
        x_p = k_pen * (-gap)
        if delta_h <= 0.0:
            p_n = max(0.0, x_p)
        elif x_p < -delta_h:
            p_n = 0.0
        elif x_p > delta_h:
            p_n = x_p
        else:
            p_n = (x_p + delta_h) ** 2 / (4.0 * delta_h)

        pair = SimpleNamespace(
            state=SimpleNamespace(
                s=s,
                t=t,
                gap=gap,
                p_n=p_n,
                normal=normal,
            ),
            nodes_a=(1, 2),
            nodes_b=(5, 6),
            radius_a=rA,
            radius_b=rB,
            elem_a=1,
            elem_b=4,
        )

        manager = SimpleNamespace(
            pairs=[pair],
            connectivity=conn,
            config=SimpleNamespace(
                use_hermite_centerline=True,
                consistent_st_tangent=False,  # K_st 除外
            ),
        )

        ndof = len(coords) * 6
        proc = HuberContactForceProcess(
            ndof=ndof,
            ndof_per_node=6,
            smoothing_delta=5000.0,
            huber_delta_h=delta_h,
        )
        return proc.tangent(np.zeros(ndof), manager, k_pen, node_coords=coords)

    def test_kc_adj_fd(self):
        """K_c_adj の隣接ノードDOF列がFDと一致."""
        coords = self._make_chain_coords()
        K = self._compute_tangent(coords)

        ndpn = 6
        # 4接触ノード (1,2,5,6) の translational DOF
        row_nodes = [1, 2, 5, 6]
        row_dofs = []
        for n in row_nodes:
            for d in range(3):
                row_dofs.append(n * ndpn + d)

        # 隣接ノード: adj_node_map[1]=(0,3), adj_node_map[4]=(4,7)
        adj_nodes = [0, 3, 4, 7]

        eps = 1e-6
        for adj_node in adj_nodes:
            for d in range(3):
                col_dof = adj_node * ndpn + d

                k_col = np.array(
                    [K[ri, col_dof] if sp.issparse(K) else K[ri, col_dof] for ri in row_dofs]
                )

                # FD
                coords_p = coords.copy()
                coords_p[adj_node, d] += eps
                f_p = self._compute_force_with_geom(coords_p)

                coords_m = coords.copy()
                coords_m[adj_node, d] -= eps
                f_m = self._compute_force_with_geom(coords_m)

                fd_col = -(f_p - f_m) / (2.0 * eps)

                np.testing.assert_allclose(
                    k_col,
                    fd_col,
                    atol=1e-2,
                    err_msg=(f"K_c_adj FD mismatch: adj_node={adj_node}, d={d}"),
                )

    def test_kc_adj_nonzero(self):
        """非平行近接配置でK_c_adj列に非ゼロ値が存在."""
        coords = self._make_chain_coords()
        K = self._compute_tangent(coords)

        ndpn = 6
        for adj_node in [0, 3, 4, 7]:
            cols = [adj_node * ndpn + d for d in range(3)]
            vals = []
            for c in cols:
                if sp.issparse(K):
                    vals.extend(K[:, c].toarray().ravel())
                else:
                    vals.extend(K[:, c])
            assert not np.allclose(vals, 0.0), f"Node {adj_node}: K_c_adj columns should be nonzero"
