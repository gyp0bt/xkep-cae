"""接触接線剛性 K_st の有限差分検証テスト.

consistent_st_tangent=True のとき、接触力の完全接線剛性が
有限差分と一致することを検証する。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact._contact_pair import (
    _ContactConfigInput,
    _ContactManagerInput,
    _ContactPairOutput,
    _ContactStateOutput,
)
from xkep_cae.contact._types import ContactStatus
from xkep_cae.contact.contact_force.strategy import HuberContactForceProcess


def _make_manager_with_pair(xA0, xA1, xB0, xB1, radius=0.5, *, consistent_st_tangent=False):
    """テスト用の ContactManager を構築."""
    from xkep_cae.contact.geometry._compute import _closest_point_segments_batch

    s_arr, t_arr, _, _, dist_arr, normal_arr, _ = _closest_point_segments_batch(
        xA0[None], xA1[None], xB0[None], xB1[None]
    )
    gap = float(dist_arr[0]) - 2 * radius

    state = _ContactStateOutput(
        s=float(s_arr[0]),
        t=float(t_arr[0]),
        gap=gap,
        normal=normal_arr[0].copy(),
        status=ContactStatus.ACTIVE if gap < 0 else ContactStatus.APPROACHING,
        p_n=0.0,
    )
    pair = _ContactPairOutput(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array([0, 1]),
        nodes_b=np.array([2, 3]),
        state=state,
        radius_a=radius,
        radius_b=radius,
    )

    config = _ContactConfigInput(consistent_st_tangent=consistent_st_tangent)
    manager = _ContactManagerInput(pairs=[pair], config=config)
    return manager


def _node_coords_from_vecs(xA0, xA1, xB0, xB1):
    """4ノードの座標行列 (4, 3) を構築."""
    return np.array([xA0, xA1, xB0, xB1], dtype=float)


def _evaluate_contact_force(xA0, xA1, xB0, xB1, radius, k_pen, smoothing_delta):
    """与えられた座標で接触力を評価."""
    ndof = 4 * 6  # 4ノード × 6 DOF
    manager = _make_manager_with_pair(xA0, xA1, xB0, xB1, radius)
    proc = HuberContactForceProcess(ndof, ndof_per_node=6, smoothing_delta=smoothing_delta)
    u = np.zeros(ndof)
    f_c, _ = proc.evaluate(u, manager, k_pen)
    return f_c


def _compute_tangent_fd(xA0, xA1, xB0, xB1, radius, k_pen, smoothing_delta, eps=1e-7):
    """接触力の接線剛性を有限差分で計算.

    tangent() は -df_c_raw/du を返すため、FD も符号を揃える。
    """
    coords = [xA0.copy(), xA1.copy(), xB0.copy(), xB1.copy()]
    ndof = 4 * 6

    K_fd = np.zeros((ndof, ndof))
    for node in range(4):
        for dim in range(3):
            dof = node * 6 + dim
            coords_p = [c.copy() for c in coords]
            coords_p[node][dim] += eps
            f_p = _evaluate_contact_force(*coords_p, radius, k_pen, smoothing_delta)

            coords_m = [c.copy() for c in coords]
            coords_m[node][dim] -= eps
            f_m = _evaluate_contact_force(*coords_m, radius, k_pen, smoothing_delta)

            # tangent() returns -df_c_raw/du, so negate FD result
            K_fd[:, dof] = -(f_p - f_m) / (2 * eps)

    return K_fd


class TestConsistentStTangent:
    """接触接線剛性の整合性テスト."""

    def _compare_tangent(self, xA0, xA1, xB0, xB1, radius=0.5, k_pen=100.0, smoothing_delta=5000.0):
        """解析的接線と有限差分の一致を検証."""
        ndof = 4 * 6
        u = np.zeros(ndof)

        # 有限差分で接線を計算
        K_fd = _compute_tangent_fd(xA0, xA1, xB0, xB1, radius, k_pen, smoothing_delta)

        # consistent_st_tangent=True で解析的接線を計算
        manager = _make_manager_with_pair(xA0, xA1, xB0, xB1, radius, consistent_st_tangent=True)
        proc = HuberContactForceProcess(ndof, ndof_per_node=6, smoothing_delta=smoothing_delta)
        # p_n を設定するために一度 evaluate を呼ぶ
        proc.evaluate(u, manager, k_pen)
        node_coords = _node_coords_from_vecs(xA0, xA1, xB0, xB1)
        K_st = proc.tangent(u, manager, k_pen, node_coords=node_coords)

        # consistent_st_tangent=False で解析的接線を計算
        manager_no = _make_manager_with_pair(
            xA0, xA1, xB0, xB1, radius, consistent_st_tangent=False
        )
        proc_no = HuberContactForceProcess(ndof, ndof_per_node=6, smoothing_delta=smoothing_delta)
        proc_no.evaluate(u, manager_no, k_pen)
        K_no = proc_no.tangent(u, manager_no, k_pen)

        K_st_dense = K_st.toarray()
        K_no_dense = K_no.toarray()

        # K_st は K_no よりも有限差分に近いはず
        err_st = np.linalg.norm(K_st_dense - K_fd)
        err_no = np.linalg.norm(K_no_dense - K_fd)

        # K_st が有限差分と十分に一致
        K_fd_norm = np.linalg.norm(K_fd)
        if K_fd_norm > 1e-10:
            assert err_st / K_fd_norm < 0.01, (
                f"K_st rel error = {err_st / K_fd_norm:.4e}, "
                f"K_no rel error = {err_no / K_fd_norm:.4e}"
            )

        return K_st_dense, K_no_dense, K_fd

    def test_orthogonal_penetrating(self):
        """直交セグメント、貫入状態."""
        xA0 = np.array([0.0, 0.0, 0.4])
        xA1 = np.array([1.0, 0.0, 0.4])
        xB0 = np.array([0.3, -0.5, 0.0])
        xB1 = np.array([0.3, 0.5, 0.0])
        self._compare_tangent(xA0, xA1, xB0, xB1, radius=0.3)

    def test_skew_penetrating(self):
        """斜交セグメント、貫入状態."""
        xA0 = np.array([0.0, 0.0, 0.5])
        xA1 = np.array([2.0, 0.5, 0.5])
        xB0 = np.array([0.8, -0.3, 0.0])
        xB1 = np.array([1.2, 0.7, 0.0])
        self._compare_tangent(xA0, xA1, xB0, xB1, radius=0.35)

    def test_asymmetric_config(self):
        """非対称配置."""
        xA0 = np.array([0.1, 0.2, 0.45])
        xA1 = np.array([1.3, -0.1, 0.5])
        xB0 = np.array([0.6, 0.0, 0.0])
        xB1 = np.array([0.5, 0.6, 0.1])
        self._compare_tangent(xA0, xA1, xB0, xB1, radius=0.3)

    def test_kst_improves_accuracy(self):
        """K_st が接線精度を改善することの検証."""
        xA0 = np.array([0.0, 0.0, 0.4])
        xA1 = np.array([1.0, 0.0, 0.4])
        xB0 = np.array([0.3, -0.5, 0.0])
        xB1 = np.array([0.3, 0.5, 0.0])
        K_st, K_no, K_fd = self._compare_tangent(xA0, xA1, xB0, xB1, radius=0.3)

        K_fd_norm = np.linalg.norm(K_fd)
        if K_fd_norm > 1e-10:
            err_st = np.linalg.norm(K_st - K_fd) / K_fd_norm
            err_no = np.linalg.norm(K_no - K_fd) / K_fd_norm
            # K_st は K_no より精度が高い（または同等）
            assert err_st <= err_no + 1e-6, (
                f"K_st should be more accurate: err_st={err_st:.4e}, err_no={err_no:.4e}"
            )
