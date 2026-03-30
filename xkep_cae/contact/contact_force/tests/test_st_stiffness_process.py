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
    """K_st 隣接ノードDOF拡張のFD検証（status-272）.

    3要素チェーン（0-1, 1-2, 2-3）+ 別の3要素チェーン（4-5, 5-6, 6-7）
    接触ペア: elem_a=0 (nodes 0,1), elem_b=2 (nodes 6,7)（中央要素を避ける）
    隣接ノード: A+2=2（elem0のn1=1を共有するelem1の反対側）, B-1=5（elem2のn0=6を共有するelem_bの反対側）
    """

    def _make_chain_coords(self):
        """2本の3要素チェーン（y方向に近接）."""
        return np.array(
            [
                # チェーンA: x方向
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [2.0, 0.0, 0.0],  # 2
                [3.0, 0.0, 0.0],  # 3
                # チェーンB: x方向、y=0.1
                [0.0, 0.1, 0.0],  # 4
                [1.0, 0.1, 0.0],  # 5
                [2.0, 0.1, 0.0],  # 6
                [3.0, 0.1, 0.0],  # 7
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

    def _compute_force(self, coords):
        """与えられた座標で接触力（4ノードのローカル力）をFBHuberで計算."""
        from xkep_cae.contact.geometry._compute import (
            _compute_node_counts,
        )

        conn = self._make_connectivity()
        node_counts = _compute_node_counts(len(coords), conn)

        s, t = 0.5, 0.5
        k_pen = 1e4
        gap = -0.01
        normal = np.array([0.0, 1.0, 0.0])

        from xkep_cae.contact.contact_force.strategy import (
            _hermite_corrected_coeffs,
        )
        from xkep_cae.contact.geometry._compute import _compute_dm_coeffs

        nodes_a = (0, 1)
        nodes_b = (6, 7)
        dm_A = _compute_dm_coeffs(node_counts[nodes_a[0]], node_counts[nodes_a[1]])
        dm_B = _compute_dm_coeffs(node_counts[nodes_b[0]], node_counts[nodes_b[1]])
        coeffs, _, _ = _hermite_corrected_coeffs(s, t, dm_A, dm_B)

        x_p = k_pen * (-gap)
        delta_h = 100.0
        if delta_h <= 0.0:
            p_n = max(0.0, x_p)
        elif x_p < -delta_h:
            p_n = 0.0
        elif x_p > delta_h:
            p_n = x_p
        else:
            p_n = (x_p + delta_h) ** 2 / (4.0 * delta_h)

        # f_c = p_n * Σ c_k * n
        f_local = np.zeros(12)
        for k in range(4):
            for i in range(3):
                f_local[k * 3 + i] = p_n * coeffs[k] * normal[i]
        return f_local

    def test_kst_adj_nodes_fd(self):
        """K_st の隣接ノードDOF列がFDと一致."""
        coords = self._make_chain_coords()
        K_st = self._compute_kst(coords)

        # 4ノード(0,1,6,7)のDOF行インデックス
        ndpn = 6
        row_nodes = [0, 1, 6, 7]
        row_dofs = []
        for n in row_nodes:
            for d in range(3):
                row_dofs.append(n * ndpn + d)

        # 隣接ノード: elem_a=0のadj_right=2, elem_b=5のadj_left=5
        # adj_node_map[0] = (-1, 2), adj_node_map[5] = (5, -1)
        # ds_du_adj レイアウト: [A-1, A+2, B-1, B+2] → ノード[-1, 2, 5, -1]
        adj_nodes = [2, 5]  # 有効な隣接ノードのみ

        eps = 1e-6
        for adj_node in adj_nodes:
            for d in range(3):
                col_dof = adj_node * ndpn + d

                # K_st[row_dofs, col_dof] の解析値
                k_col = np.array([K_st[ri, col_dof] for ri in row_dofs])

                # FD: f(x + eps*e_j) - f(x - eps*e_j) / (2*eps)
                coords_p = coords.copy()
                coords_p[adj_node, d] += eps
                f_p = self._compute_force(coords_p)

                coords_m = coords.copy()
                coords_m[adj_node, d] -= eps
                f_m = self._compute_force(coords_m)

                fd_col = -(f_p - f_m) / (2.0 * eps)

                # K_st = -df/du なので fd_col = -df/du
                np.testing.assert_allclose(
                    k_col,
                    fd_col,
                    atol=1e-4,
                    err_msg=f"K_st adj FD mismatch: adj_node={adj_node}, d={d}",
                )

    def test_kst_adj_endpoint_zero(self):
        """端点ノードの隣接DOFはK_stに寄与しない."""
        coords = self._make_chain_coords()
        K_st = self._compute_kst(coords)

        # ノード0は端点（elem_a=0のadj_left=-1）→ 隣接ノードなし
        # ノード3は端点（adj_right of elem0=2, not 3）
        # A-1=-1, B+2=-1 → これらの隣接ノードDOFはK_stに現れない
        # 端点ノード0, 3, 4, 7はK_stの隣接列に現れてはいけない
        # （ただし0,1,6,7は4ノードの行/列として現れうる）
        ndpn = 6
        # ノード3とノード4はペアに含まれず、隣接ノードでもない
        for n in [3, 4]:
            for d in range(3):
                col = n * ndpn + d
                col_vals = K_st[:, col].toarray().ravel() if sp.issparse(K_st) else K_st[:, col]
                assert np.allclose(col_vals, 0.0), f"Node {n} d={d} should have zero K_st column"
