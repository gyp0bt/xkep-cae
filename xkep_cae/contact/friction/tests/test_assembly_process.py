"""摩擦アセンブリ Process のテスト（status-256 B2-B4, status-274 FD検証）.

B4: FrictionTangentStiffnessProcess
B2: FrictionGeometricStiffnessProcess
B3: FrictionStStiffnessProcess
FD: FrictionStStiffness 隣接ノードDOF拡張の検証
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.friction import (
    FrictionGeometricStiffnessInput,
    FrictionGeometricStiffnessProcess,
    FrictionStStiffnessInput,
    FrictionStStiffnessProcess,
    FrictionTangentStiffnessInput,
    FrictionTangentStiffnessProcess,
)
from xkep_cae.core.testing import binds_to


def _make_pair(
    s: float = 0.5,
    t: float = 0.5,
    gap: float = -0.01,
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


# ── B4: FrictionTangentStiffnessProcess ────────────────────


@binds_to(FrictionTangentStiffnessProcess)
class TestFrictionTangentStiffnessProcess:
    """FrictionTangentStiffnessProcess の単体テスト（status-256 B4）."""

    def test_empty_tangents(self):
        proc = FrictionTangentStiffnessProcess()
        out = proc.process(
            FrictionTangentStiffnessInput(
                contact_pairs=[],
                friction_tangents={},
                ndof_total=24,
            )
        )
        assert isinstance(out.K_mat, sp.csr_matrix)
        assert out.K_mat.shape == (24, 24)
        assert out.K_mat.nnz == 0

    def test_single_pair(self):
        pair = _make_pair()
        D_t = np.array([[100.0, 0.0], [0.0, 100.0]])
        proc = FrictionTangentStiffnessProcess()
        out = proc.process(
            FrictionTangentStiffnessInput(
                contact_pairs=[pair],
                friction_tangents={0: D_t},
                ndof_total=24,
            )
        )
        assert out.K_mat.shape == (24, 24)
        assert out.K_mat.nnz > 0

    def test_meta(self):
        assert FrictionTangentStiffnessProcess.meta.name == "FrictionTangentStiffness"


# ── B2: FrictionGeometricStiffnessProcess ──────────────────


@binds_to(FrictionGeometricStiffnessProcess)
class TestFrictionGeometricStiffnessProcess:
    """FrictionGeometricStiffnessProcess の単体テスト（status-256 B2）."""

    def test_empty_forces(self):
        proc = FrictionGeometricStiffnessProcess()
        out = proc.process(
            FrictionGeometricStiffnessInput(
                contact_pairs=[],
                friction_forces_local={},
                ndof_total=24,
            )
        )
        assert isinstance(out.K_geo, sp.csr_matrix)
        assert out.K_geo.shape == (24, 24)
        assert out.K_geo.nnz == 0

    def test_single_pair(self):
        pair = _make_pair()
        q = np.array([0.5, 0.3])
        proc = FrictionGeometricStiffnessProcess()
        out = proc.process(
            FrictionGeometricStiffnessInput(
                contact_pairs=[pair],
                friction_forces_local={0: q},
                ndof_total=24,
            )
        )
        assert out.K_geo.shape == (24, 24)
        assert out.K_geo.nnz > 0

    def test_meta(self):
        assert FrictionGeometricStiffnessProcess.meta.name == "FrictionGeometricStiffness"


# ── B3: FrictionStStiffnessProcess ─────────────────────────


@binds_to(FrictionStStiffnessProcess)
class TestFrictionStStiffnessProcess:
    """FrictionStStiffnessProcess の単体テスト（status-256 B3）."""

    def test_empty_forces(self):
        proc = FrictionStStiffnessProcess()
        out = proc.process(
            FrictionStStiffnessInput(
                contact_pairs=[],
                friction_forces_local={},
                ndof_total=24,
                node_coords=np.zeros((4, 3)),
            )
        )
        assert isinstance(out.K_st, sp.csr_matrix)
        assert out.K_st.shape == (24, 24)
        assert out.K_st.nnz == 0

    def test_single_pair(self):
        """近接平行セグメントでK_stが非ゼロ."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [1.0, 0.1, 0.0],
            ]
        )
        pair = _make_pair()
        q = np.array([0.5, 0.3])
        proc = FrictionStStiffnessProcess()
        out = proc.process(
            FrictionStStiffnessInput(
                contact_pairs=[pair],
                friction_forces_local={0: q},
                ndof_total=24,
                node_coords=coords,
            )
        )
        assert out.K_st.shape == (24, 24)
        assert out.K_st.nnz > 0

    def test_meta(self):
        assert FrictionStStiffnessProcess.meta.name == "FrictionStStiffness"


# ── 摩擦 K_st 隣接ノードDOF拡張検証（status-274） ──────────


class TestFrictionStStiffnessAdjFD:
    """摩擦 K_st 隣接ノードDOF拡張の検証（status-274）.

    3要素チェーン（0-1, 1-2, 2-3）+ 別の3要素チェーン（4-5, 5-6, 6-7）
    接触ペア: elem_a=0 (nodes 0,1), elem_b=5 (nodes 6,7)
    隣接ノード: A+2=2, B-1=5
    """

    def _make_chain_coords(self):
        """2本の3要素チェーン（非平行配置でStJacobian非特異化）."""
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.1, 0.0],
                [3.0, 0.3, 0.0],
                [0.0, 0.2, 0.0],
                [1.0, 0.15, 0.0],
                [2.0, 0.2, 0.0],
                [3.0, 0.35, 0.0],
            ]
        )

    def _make_connectivity(self):
        return np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7]])

    def _make_adj_pair(self):
        state = SimpleNamespace(
            s=0.5,
            t=0.5,
            gap=-0.01,
            normal=np.array([0.0, 1.0, 0.0]),
            tangent1=np.array([1.0, 0.0, 0.0]),
            tangent2=np.array([0.0, 0.0, 1.0]),
        )
        return SimpleNamespace(
            state=state,
            nodes_a=(0, 1),
            nodes_b=(6, 7),
            radius_a=0.05,
            radius_b=0.05,
            elem_a=0,
            elem_b=5,
        )

    def _compute_kst_fric(self, coords):
        from xkep_cae.contact.geometry._compute import (
            _compute_adj_node_map,
            _compute_node_counts,
            _compute_node_tangents,
        )

        conn = self._make_connectivity()
        node_tangents = _compute_node_tangents(coords, conn)
        node_counts = _compute_node_counts(len(coords), conn)
        adj_node_map = _compute_adj_node_map(conn)

        pair = self._make_adj_pair()
        q = np.array([0.5, 0.3])

        proc = FrictionStStiffnessProcess()
        out = proc.process(
            FrictionStStiffnessInput(
                contact_pairs=[pair],
                friction_forces_local={0: q},
                ndof_total=len(coords) * 6,
                node_coords=coords,
                ndof_per_node=6,
                use_hermite=True,
                node_tangents=node_tangents,
                node_counts=node_counts,
                adj_node_map=adj_node_map,
            )
        )
        return out.K_st

    def test_kst_fric_adj_manual_formula(self):
        """K_st_adj が outer(df_ds, ds_du_adj) + outer(df_dt, dt_du_adj) と一致."""
        from xkep_cae.contact.geometry._compute import (
            _compute_dm_coeffs,
            _compute_dm_ext_coeffs,
            _compute_node_counts,
            _compute_node_tangents,
        )
        from xkep_cae.contact.geometry._st_jacobian import (
            ComputeStJacobianProcess,
            StJacobianInput,
        )

        coords = self._make_chain_coords()
        K_st = self._compute_kst_fric(coords)

        # StJacobian を手動で計算して ds_du_adj, dt_du_adj を取得
        pair = self._make_adj_pair()
        conn = self._make_connectivity()
        node_tangents = _compute_node_tangents(coords, conn)
        node_counts = _compute_node_counts(len(coords), conn)

        st_kw = {
            "xA0": coords[0],
            "xA1": coords[1],
            "xB0": coords[6],
            "xB1": coords[7],
            "s": 0.5,
            "t": 0.5,
            "mA0": node_tangents[0],
            "mA1": node_tangents[1],
            "mB0": node_tangents[6],
            "mB1": node_tangents[7],
            "use_hermite": True,
        }
        dm_A = _compute_dm_coeffs(node_counts[0], node_counts[1])
        dm_B = _compute_dm_coeffs(node_counts[6], node_counts[7])
        st_kw["dm_A"] = dm_A
        st_kw["dm_B"] = dm_B
        dm_ext_A = _compute_dm_ext_coeffs(node_counts[0], node_counts[1])
        dm_ext_B = _compute_dm_ext_coeffs(node_counts[6], node_counts[7])
        st_kw["dm_ext_A"] = dm_ext_A
        st_kw["dm_ext_B"] = dm_ext_B

        st_proc = ComputeStJacobianProcess()
        out = st_proc.process(StJacobianInput(**st_kw))
        assert out.ds_du_adj is not None

        # df_ds, df_dt を手動計算
        q1, q2 = 0.5, 0.3
        t1 = pair.state.tangent1
        t2 = pair.state.tangent2
        dc_ds = [-1.0, 1.0, 0.0, 0.0]
        dc_dt = [0.0, 0.0, 1.0, -1.0]

        df_ds = np.zeros(12)
        df_dt = np.zeros(12)
        for qa, ta in [(q1, t1), (q2, t2)]:
            for k in range(4):
                for i in range(3):
                    li = k * 3 + i
                    df_ds[li] += qa * dc_ds[k] * ta[i]
                    df_dt[li] += qa * dc_dt[k] * ta[i]

        # K_fric_adj = outer(df_ds, ds_du_adj) + outer(df_dt, dt_du_adj)
        K_fric_adj = np.outer(df_ds, out.ds_du_adj) + np.outer(df_dt, out.dt_du_adj)

        # adj_node_map: elem_a=0 -> (-1,2), elem_b=5 -> (5,-1)
        # adj_global_nodes: [A-1=-1, A+2=2, B-1=5, B+2=-1]
        ndpn = 6
        row_nodes = [0, 1, 6, 7]
        adj_info = [(2, 1), (5, 2)]  # (adj_node, adj_idx in ds_du_adj layout)

        for adj_node, adj_idx in adj_info:
            for d in range(3):
                col_dof = adj_node * ndpn + d
                for li_idx, rn in enumerate(row_nodes):
                    for di in range(3):
                        ri = rn * ndpn + di
                        li = li_idx * 3 + di
                        adj_lj = adj_idx * 3 + d
                        expected = K_fric_adj[li, adj_lj]
                        actual = K_st[ri, col_dof]
                        np.testing.assert_allclose(
                            actual,
                            expected,
                            atol=1e-10,
                            err_msg=(
                                f"K_st_adj manual: row=node{rn}_d{di}, col=adj{adj_node}_d{d}"
                            ),
                        )

    def test_kst_fric_adj_nonzero(self):
        """Hermite隣接ノード拡張でK_stの隣接列に非ゼロ値が存在."""
        coords = self._make_chain_coords()
        K_st = self._compute_kst_fric(coords)

        ndpn = 6
        for adj_node in [2, 5]:
            col_sum = 0.0
            for d in range(3):
                col = adj_node * ndpn + d
                col_vals = K_st[:, col].toarray().ravel()
                col_sum += np.sum(np.abs(col_vals))
            assert col_sum > 1e-10, f"Adj node {adj_node} should have nonzero K_st columns"

    def test_kst_fric_adj_endpoint_zero(self):
        """端点ノードの隣接DOFは摩擦K_stに寄与しない."""
        coords = self._make_chain_coords()
        K_st = self._compute_kst_fric(coords)

        ndpn = 6
        for n in [3, 4]:
            for d in range(3):
                col = n * ndpn + d
                col_vals = K_st[:, col].toarray().ravel() if sp.issparse(K_st) else K_st[:, col]
                assert np.allclose(col_vals, 0.0), f"Node {n} d={d} should have zero K_st column"
