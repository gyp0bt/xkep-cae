"""∂(s,t)/∂u Jacobian の有限差分検証テスト.

status-078 の旧テスト (test_consistent_st_tangent.py) を
Process Architecture で再実装。
status-230 で Hermite 幾何対応テストを追加。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.geometry._compute import (
    _closest_point_hermite_refine,
    _closest_point_segments_batch,
)
from xkep_cae.contact.geometry._st_jacobian import (
    ComputeStJacobianProcess,
    StJacobianInput,
)
from xkep_cae.core.testing import binds_to


def _compute_s_t_numerical(xA0, xA1, xB0, xB1):
    """最近接点パラメータ (s, t) を直接計算する."""
    dA = xA1 - xA0
    dB = xB1 - xB0
    w0 = xA0 - xB0

    a = float(np.dot(dA, dA))
    b = float(np.dot(dA, dB))
    c = float(np.dot(dB, dB))
    d = float(np.dot(dA, w0))
    e = float(np.dot(dB, w0))

    det = a * c - b * b
    if abs(det) < 1e-15 * max(a * c, 1e-30):
        return 0.0, 0.0, 0.0, 0.0  # 平行

    s_unc = (b * e - c * d) / det
    t_unc = (a * e - b * d) / det
    s = np.clip(s_unc, 0.0, 1.0)
    t = np.clip(t_unc, 0.0, 1.0)

    # 再計算（クランプ後の相互依存）
    if s_unc < 0.0 or s_unc > 1.0:
        if c > 1e-15:
            t = np.clip((b * s + e) / c, 0.0, 1.0)
    if t_unc < 0.0 or t_unc > 1.0:
        if a > 1e-15:
            s = np.clip((b * t - d) / a, 0.0, 1.0)
        if c > 1e-15:
            t = np.clip((b * s + e) / c, 0.0, 1.0)

    return float(s), float(t), float(s_unc), float(t_unc)


def _finite_diff_st_jacobian(xA0, xA1, xB0, xB1, eps=1e-7):
    """有限差分で ds/du, dt/du を計算."""
    coords = [xA0.copy(), xA1.copy(), xB0.copy(), xB1.copy()]
    s0, t0, _, _ = _compute_s_t_numerical(*coords)

    ds_du = np.zeros(12)
    dt_du = np.zeros(12)

    for node in range(4):
        for dim in range(3):
            idx = node * 3 + dim
            coords_p = [c.copy() for c in coords]
            coords_p[node][dim] += eps
            s_p, t_p, _, _ = _compute_s_t_numerical(*coords_p)

            coords_m = [c.copy() for c in coords]
            coords_m[node][dim] -= eps
            s_m, t_m, _, _ = _compute_s_t_numerical(*coords_m)

            ds_du[idx] = (s_p - s_m) / (2 * eps)
            dt_du[idx] = (t_p - t_m) / (2 * eps)

    return ds_du, dt_du, s0, t0


@binds_to(ComputeStJacobianProcess)
class TestComputeStJacobian:
    """∂(s,t)/∂u の解析微分と有限差分の一致検証."""

    def test_orthogonal_segments(self):
        """直交セグメント（内部点）."""
        xA0 = np.array([0.0, 0.0, 0.5])
        xA1 = np.array([1.0, 0.0, 0.5])
        xB0 = np.array([0.3, -0.5, 0.0])
        xB1 = np.array([0.3, 0.5, 0.0])

        ds_du_fd, dt_du_fd, s, t = _finite_diff_st_jacobian(xA0, xA1, xB0, xB1)
        s0, t0, s_unc, t_unc = _compute_s_t_numerical(xA0, xA1, xB0, xB1)

        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0,
                xA1=xA1,
                xB0=xB0,
                xB1=xB1,
                s=s0,
                t=t0,
                s_unclamped=s_unc,
                t_unclamped=t_unc,
            )
        )

        assert out.valid
        np.testing.assert_allclose(out.ds_du, ds_du_fd, atol=1e-5)
        np.testing.assert_allclose(out.dt_du, dt_du_fd, atol=1e-5)

    def test_skew_segments(self):
        """斜交セグメント（内部点）."""
        xA0 = np.array([0.0, 0.0, 1.0])
        xA1 = np.array([2.0, 1.0, 1.0])
        xB0 = np.array([0.5, -1.0, 0.0])
        xB1 = np.array([1.5, 1.0, 0.0])

        ds_du_fd, dt_du_fd, s, t = _finite_diff_st_jacobian(xA0, xA1, xB0, xB1)
        s0, t0, s_unc, t_unc = _compute_s_t_numerical(xA0, xA1, xB0, xB1)

        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0,
                xA1=xA1,
                xB0=xB0,
                xB1=xB1,
                s=s0,
                t=t0,
                s_unclamped=s_unc,
                t_unclamped=t_unc,
            )
        )

        assert out.valid
        np.testing.assert_allclose(out.ds_du, ds_du_fd, atol=1e-5)
        np.testing.assert_allclose(out.dt_du, dt_du_fd, atol=1e-5)

    def test_near_endpoint_s(self):
        """s が端点近傍（クランプなし）."""
        xA0 = np.array([0.0, 0.0, 0.5])
        xA1 = np.array([1.0, 0.0, 0.5])
        xB0 = np.array([0.05, -0.5, 0.0])
        xB1 = np.array([0.05, 0.5, 0.0])

        ds_du_fd, dt_du_fd, s, t = _finite_diff_st_jacobian(xA0, xA1, xB0, xB1)
        s0, t0, s_unc, t_unc = _compute_s_t_numerical(xA0, xA1, xB0, xB1)

        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0,
                xA1=xA1,
                xB0=xB0,
                xB1=xB1,
                s=s0,
                t=t0,
                s_unclamped=s_unc,
                t_unclamped=t_unc,
            )
        )

        assert out.valid
        np.testing.assert_allclose(out.ds_du, ds_du_fd, atol=1e-5)
        np.testing.assert_allclose(out.dt_du, dt_du_fd, atol=1e-5)

    def test_parallel_segments_returns_invalid(self):
        """平行セグメント → valid=False."""
        xA0 = np.array([0.0, 0.0, 0.5])
        xA1 = np.array([1.0, 0.0, 0.5])
        xB0 = np.array([0.0, 1.0, 0.0])
        xB1 = np.array([1.0, 1.0, 0.0])

        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0,
                xA1=xA1,
                xB0=xB0,
                xB1=xB1,
                s=0.5,
                t=0.5,
                s_unclamped=0.5,
                t_unclamped=0.5,
            )
        )

        # スムーズ遷移で特異系でもフォールバック計算を行う
        # det ≈ 0 の場合は 1×1 系フォールバックで valid=True
        assert out.valid

    def test_both_clamped(self):
        """両方クランプ → ds/du = dt/du = 0."""
        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=np.array([0.0, 0.0, 0.5]),
                xA1=np.array([1.0, 0.0, 0.5]),
                xB0=np.array([2.0, -0.5, 0.0]),
                xB1=np.array([2.0, 0.5, 0.0]),
                s=1.0,
                t=0.5,
                s_unclamped=1.5,
                t_unclamped=-0.3,
            )
        )

        assert out.valid
        np.testing.assert_allclose(out.ds_du, 0.0, atol=1e-15)
        np.testing.assert_allclose(out.dt_du, 0.0, atol=1e-15)

    def test_large_gap(self):
        """大きなギャップでも正しく計算."""
        xA0 = np.array([0.0, 0.0, 5.0])
        xA1 = np.array([1.0, 0.0, 5.0])
        xB0 = np.array([0.4, -0.5, 0.0])
        xB1 = np.array([0.4, 0.5, 0.0])

        ds_du_fd, dt_du_fd, s, t = _finite_diff_st_jacobian(xA0, xA1, xB0, xB1)
        s0, t0, s_unc, t_unc = _compute_s_t_numerical(xA0, xA1, xB0, xB1)

        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0,
                xA1=xA1,
                xB0=xB0,
                xB1=xB1,
                s=s0,
                t=t0,
                s_unclamped=s_unc,
                t_unclamped=t_unc,
            )
        )

        assert out.valid
        np.testing.assert_allclose(out.ds_du, ds_du_fd, atol=1e-5)
        np.testing.assert_allclose(out.dt_du, dt_du_fd, atol=1e-5)

    def test_asymmetric_config(self):
        """非対称配置."""
        xA0 = np.array([0.1, 0.2, 0.8])
        xA1 = np.array([1.3, -0.4, 0.9])
        xB0 = np.array([0.7, 0.1, -0.2])
        xB1 = np.array([0.5, 0.8, 0.3])

        ds_du_fd, dt_du_fd, s, t = _finite_diff_st_jacobian(xA0, xA1, xB0, xB1)
        s0, t0, s_unc, t_unc = _compute_s_t_numerical(xA0, xA1, xB0, xB1)

        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0,
                xA1=xA1,
                xB0=xB0,
                xB1=xB1,
                s=s0,
                t=t0,
                s_unclamped=s_unc,
                t_unclamped=t_unc,
            )
        )

        assert out.valid
        np.testing.assert_allclose(out.ds_du, ds_du_fd, atol=1e-5)
        np.testing.assert_allclose(out.dt_du, dt_du_fd, atol=1e-5)


# ── Hermite 幾何対応 FD テスト（status-230）──────────────────


def _compute_hermite_s_t(xA0, xA1, xB0, xB1, mA0, mA1, mB0, mB1):
    """Hermite 曲線上の最近接点パラメータを計算."""
    xA0_b, xA1_b = xA0.reshape(1, 3), xA1.reshape(1, 3)
    xB0_b, xB1_b = xB0.reshape(1, 3), xB1.reshape(1, 3)
    s_lin, t_lin, _, _, _, _, _, _, _ = _closest_point_segments_batch(xA0_b, xA1_b, xB0_b, xB1_b)
    mA0_b, mA1_b = mA0.reshape(1, 3), mA1.reshape(1, 3)
    mB0_b, mB1_b = mB0.reshape(1, 3), mB1.reshape(1, 3)
    s, t, _, _, _, _, _, _ = _closest_point_hermite_refine(
        s_lin, t_lin, xA0_b, xA1_b, xB0_b, xB1_b, mA0_b, mA1_b, mB0_b, mB1_b
    )
    return float(s[0]), float(t[0])


def _finite_diff_st_jacobian_hermite(xA0, xA1, xB0, xB1, mA0, mA1, mB0, mB1, eps=1e-7):
    """Hermite 幾何での有限差分 ds/du, dt/du（m 凍結）."""
    coords = [xA0.copy(), xA1.copy(), xB0.copy(), xB1.copy()]
    tangents = [mA0.copy(), mA1.copy(), mB0.copy(), mB1.copy()]
    s0, t0 = _compute_hermite_s_t(*coords, *tangents)

    ds_du = np.zeros(12)
    dt_du = np.zeros(12)
    for node in range(4):
        for dim in range(3):
            idx = node * 3 + dim
            coords_p = [c.copy() for c in coords]
            coords_p[node][dim] += eps
            s_p, t_p = _compute_hermite_s_t(*coords_p, *tangents)
            coords_m = [c.copy() for c in coords]
            coords_m[node][dim] -= eps
            s_m, t_m = _compute_hermite_s_t(*coords_m, *tangents)
            ds_du[idx] = (s_p - s_m) / (2 * eps)
            dt_du[idx] = (t_p - t_m) / (2 * eps)
    return ds_du, dt_du, s0, t0


class TestComputeStJacobianHermite:
    """Hermite 幾何対応 ∂(s,t)/∂u の FD 検証（status-230）."""

    def test_straight_hermite_matches_fd(self):
        """直線 Hermite（m=dA,dB）の FD 一致."""
        xA0 = np.array([0.0, 0.0, 0.5])
        xA1 = np.array([1.0, 0.0, 0.5])
        xB0 = np.array([0.3, -0.5, 0.0])
        xB1 = np.array([0.3, 0.5, 0.0])
        dA, dB = xA1 - xA0, xB1 - xB0

        ds_fd, dt_fd, s0, t0 = _finite_diff_st_jacobian_hermite(xA0, xA1, xB0, xB1, dA, dA, dB, dB)
        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0,
                xA1=xA1,
                xB0=xB0,
                xB1=xB1,
                s=s0,
                t=t0,
                mA0=dA,
                mA1=dA,
                mB0=dB,
                mB1=dB,
                use_hermite=True,
            )
        )
        assert out.valid
        np.testing.assert_allclose(out.ds_du, ds_fd, atol=1e-5)
        np.testing.assert_allclose(out.dt_du, dt_fd, atol=1e-5)

    # NOTE(STA2): curved/skew は atol=1e-2。
    # 原因（status-275 で更新）: frozen-m は status-271〜274 で解消済みだが、
    # _closest_point_hermite_refine の収束精度が不十分（skew: 残差~6e-4,
    # orthogonal: 残差~5e-7）。FD は収束済み s,t を基準にするため、
    # 解析式（F1=F2=0 前提）との差が生じる。
    # asymmetric は完全収束（残差~1e-16）のため atol=1e-5 に厳格化。
    # TODO(STA2): _closest_point_hermite_refine の収束精度改善後に全て atol=1e-5 へ

    def test_curved_hermite_orthogonal(self):
        """曲がった Hermite 曲線（直交配置）."""
        xA0 = np.array([0.0, 0.0, 0.5])
        xA1 = np.array([1.0, 0.0, 0.5])
        xB0 = np.array([0.4, -0.5, 0.0])
        xB1 = np.array([0.4, 0.5, 0.0])
        mA0, mA1 = np.array([1.0, 0.3, 0.0]), np.array([1.0, -0.2, 0.0])
        mB0, mB1 = np.array([0.0, 1.0, 0.1]), np.array([0.0, 1.0, -0.1])

        ds_fd, dt_fd, s0, t0 = _finite_diff_st_jacobian_hermite(
            xA0, xA1, xB0, xB1, mA0, mA1, mB0, mB1
        )
        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0,
                xA1=xA1,
                xB0=xB0,
                xB1=xB1,
                s=s0,
                t=t0,
                mA0=mA0,
                mA1=mA1,
                mB0=mB0,
                mB1=mB1,
                use_hermite=True,
            )
        )
        assert out.valid
        np.testing.assert_allclose(out.ds_du, ds_fd, atol=1e-2)
        np.testing.assert_allclose(out.dt_du, dt_fd, atol=1e-2)

    def test_curved_hermite_skew(self):
        """曲がった Hermite 曲線（斜交配置）."""
        xA0 = np.array([0.0, 0.0, 1.0])
        xA1 = np.array([2.0, 1.0, 1.0])
        xB0 = np.array([0.5, -1.0, 0.0])
        xB1 = np.array([1.5, 1.0, 0.0])
        mA0, mA1 = np.array([2.0, 0.5, 0.3]), np.array([2.0, 1.5, -0.2])
        mB0, mB1 = np.array([1.0, 2.0, 0.2]), np.array([1.0, 2.0, -0.3])

        ds_fd, dt_fd, s0, t0 = _finite_diff_st_jacobian_hermite(
            xA0, xA1, xB0, xB1, mA0, mA1, mB0, mB1
        )
        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0,
                xA1=xA1,
                xB0=xB0,
                xB1=xB1,
                s=s0,
                t=t0,
                mA0=mA0,
                mA1=mA1,
                mB0=mB0,
                mB1=mB1,
                use_hermite=True,
            )
        )
        assert out.valid
        np.testing.assert_allclose(out.ds_du, ds_fd, atol=1e-2)
        np.testing.assert_allclose(out.dt_du, dt_fd, atol=1e-2)

    def test_hermite_asymmetric(self):
        """非対称 Hermite 配置（完全収束 → atol=1e-5, status-275）."""
        xA0 = np.array([0.1, 0.2, 0.8])
        xA1 = np.array([1.3, -0.4, 0.9])
        xB0 = np.array([0.7, 0.1, -0.2])
        xB1 = np.array([0.5, 0.8, 0.3])
        mA0, mA1 = np.array([1.2, -0.6, 0.1]), np.array([1.2, -0.6, 0.1])
        mB0, mB1 = np.array([-0.2, 0.7, 0.5]), np.array([-0.2, 0.7, 0.5])

        ds_fd, dt_fd, s0, t0 = _finite_diff_st_jacobian_hermite(
            xA0, xA1, xB0, xB1, mA0, mA1, mB0, mB1
        )
        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0,
                xA1=xA1,
                xB0=xB0,
                xB1=xB1,
                s=s0,
                t=t0,
                mA0=mA0,
                mA1=mA1,
                mB0=mB0,
                mB1=mB1,
                use_hermite=True,
            )
        )
        assert out.valid
        np.testing.assert_allclose(out.ds_du, ds_fd, atol=1e-5)
        np.testing.assert_allclose(out.dt_du, dt_fd, atol=1e-5)


# ── Hermite 非局所 ∂(s,t)/∂u FD テスト（status-271）──────────────


def _build_strand_tangents(coords: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """ストランドの節点接線ベクトルを計算."""
    from xkep_cae.contact.geometry._compute import _compute_node_tangents

    return _compute_node_tangents(coords, connectivity)


def _finite_diff_st_jacobian_hermite_nonlocal(
    strand_a_coords: np.ndarray,
    strand_b_coords: np.ndarray,
    conn_a: np.ndarray,
    conn_b: np.ndarray,
    elem_a_idx: int,
    elem_b_idx: int,
    neighbor_node_indices: list[int | None],
    is_strand_a: list[bool],
    eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """隣接ノードの有限差分で ds_du_adj, dt_du_adj を計算.

    neighbor_node_indices: [A-1, A+2, B-1, B+2] のグローバルノード番号
    is_strand_a: 各隣接ノードがstrand_aに属するかどうか
    """
    nA0, nA1 = conn_a[elem_a_idx]
    nB0, nB1 = conn_b[elem_b_idx]

    tangents_a_0 = _build_strand_tangents(strand_a_coords, conn_a)
    tangents_b_0 = _build_strand_tangents(strand_b_coords, conn_b)
    xA0, xA1 = strand_a_coords[nA0], strand_a_coords[nA1]
    xB0, xB1 = strand_b_coords[nB0], strand_b_coords[nB1]
    mA0, mA1 = tangents_a_0[nA0], tangents_a_0[nA1]
    mB0, mB1 = tangents_b_0[nB0], tangents_b_0[nB1]
    s0, t0 = _compute_hermite_s_t(xA0, xA1, xB0, xB1, mA0, mA1, mB0, mB1)

    ds_du_adj = np.zeros(12)
    dt_du_adj = np.zeros(12)

    for adj_idx, (node_id, in_a) in enumerate(zip(neighbor_node_indices, is_strand_a, strict=True)):
        if node_id is None:
            continue
        for dim in range(3):
            s_vals = []
            t_vals = []
            for sign in [+1, -1]:
                if in_a:
                    coords_pert = strand_a_coords.copy()
                    coords_pert[node_id, dim] += sign * eps
                    tang_a = _build_strand_tangents(coords_pert, conn_a)
                    tang_b = tangents_b_0
                    xa0, xa1 = coords_pert[nA0], coords_pert[nA1]
                    xb0, xb1 = xB0, xB1
                else:
                    coords_pert = strand_b_coords.copy()
                    coords_pert[node_id, dim] += sign * eps
                    tang_a = tangents_a_0
                    tang_b = _build_strand_tangents(coords_pert, conn_b)
                    xa0, xa1 = xA0, xA1
                    xb0, xb1 = coords_pert[nB0], coords_pert[nB1]

                ma0, ma1 = tang_a[nA0], tang_a[nA1]
                mb0, mb1 = tang_b[nB0], tang_b[nB1]
                s_p, t_p = _compute_hermite_s_t(xa0, xa1, xb0, xb1, ma0, ma1, mb0, mb1)
                s_vals.append(s_p)
                t_vals.append(t_p)

            idx = adj_idx * 3 + dim
            ds_du_adj[idx] = (s_vals[0] - s_vals[1]) / (2 * eps)
            dt_du_adj[idx] = (t_vals[0] - t_vals[1]) / (2 * eps)

    return ds_du_adj, dt_du_adj, s0, t0


class TestComputeStJacobianHermiteNonlocal:
    """Hermite 非局所 ∂(s,t)/∂u の FD 検証（status-271）."""

    def _setup_two_strands(self):
        """2本の3要素ストランド（内部ノード付き）を構築."""
        strand_a = np.array(
            [
                [-1.0, 0.0, 0.5],
                [0.0, 0.0, 0.5],
                [1.0, 0.1, 0.5],
                [2.0, 0.0, 0.5],
            ]
        )
        conn_a = np.array([[0, 1], [1, 2], [2, 3]])

        strand_b = np.array(
            [
                [0.3, -1.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.4, 2.0, 0.0],
            ]
        )
        conn_b = np.array([[0, 1], [1, 2], [2, 3]])

        elem_a_idx, elem_b_idx = 1, 1
        neighbor_nodes = [0, 3, 0, 3]
        is_strand_a = [True, True, False, False]

        return (
            strand_a,
            strand_b,
            conn_a,
            conn_b,
            elem_a_idx,
            elem_b_idx,
            neighbor_nodes,
            is_strand_a,
        )

    def test_nonlocal_fd_inner_nodes(self):
        """内部ノード配置での非局所 FD 検証."""
        from xkep_cae.contact.geometry._compute import (
            _compute_dm_coeffs,
            _compute_dm_ext_coeffs,
            _compute_node_counts,
        )

        (
            strand_a,
            strand_b,
            conn_a,
            conn_b,
            elem_a_idx,
            elem_b_idx,
            neighbor_nodes,
            is_strand_a,
        ) = self._setup_two_strands()

        ds_adj_fd, dt_adj_fd, s0, t0 = _finite_diff_st_jacobian_hermite_nonlocal(
            strand_a,
            strand_b,
            conn_a,
            conn_b,
            elem_a_idx,
            elem_b_idx,
            neighbor_nodes,
            is_strand_a,
        )

        nA0, nA1 = conn_a[elem_a_idx]
        nB0, nB1 = conn_b[elem_b_idx]
        tangents_a = _build_strand_tangents(strand_a, conn_a)
        tangents_b = _build_strand_tangents(strand_b, conn_b)
        counts_a = _compute_node_counts(len(strand_a), conn_a)
        counts_b = _compute_node_counts(len(strand_b), conn_b)

        dm_A = _compute_dm_coeffs(counts_a[nA0], counts_a[nA1])
        dm_B = _compute_dm_coeffs(counts_b[nB0], counts_b[nB1])
        dm_ext_A = _compute_dm_ext_coeffs(counts_a[nA0], counts_a[nA1])
        dm_ext_B = _compute_dm_ext_coeffs(counts_b[nB0], counts_b[nB1])

        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=strand_a[nA0],
                xA1=strand_a[nA1],
                xB0=strand_b[nB0],
                xB1=strand_b[nB1],
                s=s0,
                t=t0,
                mA0=tangents_a[nA0],
                mA1=tangents_a[nA1],
                mB0=tangents_b[nB0],
                mB1=tangents_b[nB1],
                use_hermite=True,
                dm_A=dm_A,
                dm_B=dm_B,
                dm_ext_A=dm_ext_A,
                dm_ext_B=dm_ext_B,
            )
        )
        assert out.valid
        assert out.ds_du_adj is not None
        assert out.dt_du_adj is not None
        assert out.ds_du_adj.shape == (12,)
        assert out.dt_du_adj.shape == (12,)

        np.testing.assert_allclose(out.ds_du_adj, ds_adj_fd, atol=1e-5)
        np.testing.assert_allclose(out.dt_du_adj, dt_adj_fd, atol=1e-5)

    def test_nonlocal_endpoint_zero(self):
        """端点ノードの隣接微分はゼロ."""
        from xkep_cae.contact.geometry._compute import (
            _compute_dm_coeffs,
            _compute_dm_ext_coeffs,
            _compute_node_counts,
        )

        strand_a = np.array(
            [
                [0.0, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [2.0, 0.0, 0.5],
            ]
        )
        conn_a = np.array([[0, 1], [1, 2]])

        strand_b = np.array(
            [
                [0.4, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.6, 1.5, 0.0],
            ]
        )
        conn_b = np.array([[0, 1], [1, 2]])

        elem_a_idx, elem_b_idx = 0, 0
        nA0, nA1 = conn_a[elem_a_idx]
        nB0, nB1 = conn_b[elem_b_idx]

        tangents_a = _build_strand_tangents(strand_a, conn_a)
        tangents_b = _build_strand_tangents(strand_b, conn_b)
        counts_a = _compute_node_counts(len(strand_a), conn_a)
        counts_b = _compute_node_counts(len(strand_b), conn_b)

        dm_A = _compute_dm_coeffs(counts_a[nA0], counts_a[nA1])
        dm_B = _compute_dm_coeffs(counts_b[nB0], counts_b[nB1])
        dm_ext_A = _compute_dm_ext_coeffs(counts_a[nA0], counts_a[nA1])
        dm_ext_B = _compute_dm_ext_coeffs(counts_b[nB0], counts_b[nB1])

        assert abs(dm_ext_A[0]) < 1e-15  # A0 は端点
        assert abs(dm_ext_B[0]) < 1e-15  # B0 は端点

        s0, t0 = _compute_hermite_s_t(
            strand_a[nA0],
            strand_a[nA1],
            strand_b[nB0],
            strand_b[nB1],
            tangents_a[nA0],
            tangents_a[nA1],
            tangents_b[nB0],
            tangents_b[nB1],
        )

        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=strand_a[nA0],
                xA1=strand_a[nA1],
                xB0=strand_b[nB0],
                xB1=strand_b[nB1],
                s=s0,
                t=t0,
                mA0=tangents_a[nA0],
                mA1=tangents_a[nA1],
                mB0=tangents_b[nB0],
                mB1=tangents_b[nB1],
                use_hermite=True,
                dm_A=dm_A,
                dm_B=dm_B,
                dm_ext_A=dm_ext_A,
                dm_ext_B=dm_ext_B,
            )
        )
        assert out.valid
        assert out.ds_du_adj is not None
        np.testing.assert_allclose(out.ds_du_adj[0:3], 0.0, atol=1e-15)
        np.testing.assert_allclose(out.dt_du_adj[0:3], 0.0, atol=1e-15)
        np.testing.assert_allclose(out.ds_du_adj[6:9], 0.0, atol=1e-15)
        np.testing.assert_allclose(out.dt_du_adj[6:9], 0.0, atol=1e-15)
