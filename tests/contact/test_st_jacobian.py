"""∂(s,t)/∂u Jacobian の有限差分検証テスト.

status-078 の旧テスト (test_consistent_st_tangent.py) を
Process Architecture で再実装。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.geometry._st_jacobian import (
    ComputeStJacobianProcess,
    StJacobianInput,
)


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

        assert not out.valid

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
