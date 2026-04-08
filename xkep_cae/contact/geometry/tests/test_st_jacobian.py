"""ComputeStJacobianProcess の C3 テスト紐付け.

@binds_to による 1:1 紐付け + API 適合テスト。
外部テスト tests/contact/test_st_jacobian.py に詳細な FD 検証あり。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.geometry._st_jacobian import (
    ComputeStJacobianProcess,
    StJacobianInput,
    StJacobianOutput,
)
from xkep_cae.core import SolverProcess
from xkep_cae.core.testing import binds_to


@binds_to(ComputeStJacobianProcess)
class TestComputeStJacobianProcessAPI:
    """ComputeStJacobianProcess の API 適合テスト."""

    def test_protocol_conformance(self):
        """SolverProcess を継承している."""
        assert issubclass(ComputeStJacobianProcess, SolverProcess)

    def test_meta_name(self):
        """meta.name が設定されている."""
        assert ComputeStJacobianProcess.meta.name == "ComputeStJacobian"

    def test_process_returns_output(self):
        """直交配置で StJacobianOutput を返す."""
        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=np.array([0.0, 0.0, 0.5]),
                xA1=np.array([1.0, 0.0, 0.5]),
                xB0=np.array([0.3, -0.5, 0.0]),
                xB1=np.array([0.3, 0.5, 0.0]),
                s=0.3,
                t=0.5,
                s_unclamped=0.3,
                t_unclamped=0.5,
            )
        )
        assert isinstance(out, StJacobianOutput)
        assert out.valid
        assert out.ds_du.shape == (12,)
        assert out.dt_du.shape == (12,)


class TestBatchStJacobianLinearAPI:
    """バッチ版 StJacobian（線形）の API テスト（status-309）."""

    def test_empty_input(self):
        """空入力で空配列を返す."""
        from xkep_cae.contact.geometry._st_jacobian import (
            _batch_st_jacobian_linear,
        )

        empty = np.zeros((0, 3))
        empty_s = np.zeros(0)
        ds, dt, valid = _batch_st_jacobian_linear(
            empty, empty, empty, empty, empty_s, empty_s, empty_s, empty_s
        )
        assert ds.shape == (0, 12)
        assert dt.shape == (0, 12)
        assert valid.shape == (0,)

    def test_single_pair_matches_scalar(self):
        """1ペアでスカラー版と一致."""
        from xkep_cae.contact.geometry._st_jacobian import (
            _batch_st_jacobian_linear,
        )

        xA0 = np.array([[0.0, 0.0, 0.5]])
        xA1 = np.array([[1.0, 0.0, 0.5]])
        xB0 = np.array([[0.3, -0.5, 0.0]])
        xB1 = np.array([[0.3, 0.5, 0.0]])
        s = np.array([0.3])
        t = np.array([0.5])

        ds_batch, dt_batch, valid_batch = _batch_st_jacobian_linear(xA0, xA1, xB0, xB1, s, t, s, t)

        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0[0],
                xA1=xA1[0],
                xB0=xB0[0],
                xB1=xB1[0],
                s=0.3,
                t=0.5,
                s_unclamped=0.3,
                t_unclamped=0.5,
            )
        )

        np.testing.assert_allclose(ds_batch[0], out.ds_du, atol=1e-12)
        np.testing.assert_allclose(dt_batch[0], out.dt_du, atol=1e-12)
        assert valid_batch[0] == out.valid

    def test_multi_pair_matches_scalar(self):
        """複数ペアでスカラー版と一致."""
        from xkep_cae.contact.geometry._st_jacobian import (
            _batch_st_jacobian_linear,
        )

        rng = np.random.default_rng(42)
        N = 20
        xA0 = rng.uniform(-1, 1, (N, 3))
        xA1 = xA0 + rng.uniform(0.5, 1.5, (N, 3))
        xB0 = rng.uniform(-1, 1, (N, 3))
        xB1 = xB0 + rng.uniform(0.5, 1.5, (N, 3))
        s = rng.uniform(0.1, 0.9, N)
        t = rng.uniform(0.1, 0.9, N)

        ds_batch, dt_batch, valid_batch = _batch_st_jacobian_linear(xA0, xA1, xB0, xB1, s, t, s, t)

        proc = ComputeStJacobianProcess()
        for i in range(N):
            out = proc.process(
                StJacobianInput(
                    xA0=xA0[i],
                    xA1=xA1[i],
                    xB0=xB0[i],
                    xB1=xB1[i],
                    s=float(s[i]),
                    t=float(t[i]),
                    s_unclamped=float(s[i]),
                    t_unclamped=float(t[i]),
                )
            )
            np.testing.assert_allclose(ds_batch[i], out.ds_du, atol=1e-12)
            np.testing.assert_allclose(dt_batch[i], out.dt_du, atol=1e-12)

    def test_clamped_boundary_fallback(self):
        """クランプ境界ペアが低速パスで正しく処理される."""
        from xkep_cae.contact.geometry._st_jacobian import (
            _batch_st_jacobian_linear,
        )

        xA0 = np.array([[0.0, 0.0, 0.0]])
        xA1 = np.array([[1.0, 0.0, 0.0]])
        xB0 = np.array([[0.5, 1.0, 0.0]])
        xB1 = np.array([[0.5, 2.0, 0.0]])
        s = np.array([0.0])  # クランプ済み
        t = np.array([0.5])
        s_unc = np.array([-0.5])  # クランプ前（境界外）
        t_unc = np.array([0.5])

        ds_batch, dt_batch, valid_batch = _batch_st_jacobian_linear(
            xA0, xA1, xB0, xB1, s, t, s_unc, t_unc
        )

        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0[0],
                xA1=xA1[0],
                xB0=xB0[0],
                xB1=xB1[0],
                s=0.0,
                t=0.5,
                s_unclamped=-0.5,
                t_unclamped=0.5,
            )
        )

        np.testing.assert_allclose(ds_batch[0], out.ds_du, atol=1e-12)
        np.testing.assert_allclose(dt_batch[0], out.dt_du, atol=1e-12)


class TestHermiteDerivBatchAPI:
    """バッチ版 Hermite 接線微分の API テスト（status-310）."""

    def test_single_pair_matches_scalar(self):
        """1ペアでスカラー版と一致."""
        from xkep_cae.contact.geometry._st_jacobian import (
            _hermite_deriv_batch,
            _hermite_deriv_scalar,
        )

        x0 = np.array([[0.0, 0.0, 0.0]])
        x1 = np.array([[1.0, 0.0, 0.0]])
        m0 = np.array([[1.0, 0.1, 0.0]])
        m1 = np.array([[1.0, -0.1, 0.0]])
        s = np.array([0.3])

        batch_result = _hermite_deriv_batch(s, x0, x1, m0, m1)
        scalar_result = _hermite_deriv_scalar(0.3, x0[0], x1[0], m0[0], m1[0])

        assert batch_result.shape == (1, 3)
        np.testing.assert_allclose(batch_result[0], scalar_result, atol=1e-14)

    def test_multi_pair_matches_scalar(self):
        """複数ペアでスカラー版と一致."""
        from xkep_cae.contact.geometry._st_jacobian import (
            _hermite_deriv_batch,
            _hermite_deriv_scalar,
        )

        rng = np.random.default_rng(77)
        N = 30
        x0 = rng.uniform(-1, 1, (N, 3))
        x1 = x0 + rng.uniform(0.5, 1.5, (N, 3))
        m0 = rng.uniform(-0.5, 0.5, (N, 3))
        m1 = rng.uniform(-0.5, 0.5, (N, 3))
        s = rng.uniform(0.0, 1.0, N)

        batch_result = _hermite_deriv_batch(s, x0, x1, m0, m1)
        assert batch_result.shape == (N, 3)

        for i in range(N):
            scalar_result = _hermite_deriv_scalar(float(s[i]), x0[i], x1[i], m0[i], m1[i])
            np.testing.assert_allclose(
                batch_result[i], scalar_result, atol=1e-14, err_msg=f"mismatch at pair {i}"
            )

    def test_boundary_values(self):
        """s=0 と s=1 の境界値でスカラー版と一致."""
        from xkep_cae.contact.geometry._st_jacobian import (
            _hermite_deriv_batch,
            _hermite_deriv_scalar,
        )

        x0 = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        x1 = np.array([[1.0, 0.0, 0.0], [4.0, 5.0, 6.0]])
        m0 = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        m1 = np.array([[1.0, 0.0, 0.0], [0.5, -0.5, 0.0]])
        s = np.array([0.0, 1.0])

        batch_result = _hermite_deriv_batch(s, x0, x1, m0, m1)
        for i in range(2):
            scalar_result = _hermite_deriv_scalar(float(s[i]), x0[i], x1[i], m0[i], m1[i])
            np.testing.assert_allclose(batch_result[i], scalar_result, atol=1e-14)


class TestBatchStJacobianHermiteAPI:
    """バッチ版 StJacobian（Hermite）の API テスト（status-309）."""

    def test_single_pair_matches_scalar(self):
        """1ペアでスカラー版と一致（Hermite + dm 補正）."""
        from xkep_cae.contact.geometry._st_jacobian import (
            _batch_st_jacobian_hermite,
        )

        xA0 = np.array([[0.0, 0.0, 0.0]])
        xA1 = np.array([[1.0, 0.0, 0.0]])
        xB0 = np.array([[0.5, 0.5, 0.0]])
        xB1 = np.array([[0.5, 1.5, 0.0]])
        mA0 = np.array([[1.0, 0.0, 0.0]])
        mA1 = np.array([[1.0, 0.0, 0.0]])
        mB0 = np.array([[0.0, 1.0, 0.0]])
        mB1 = np.array([[0.0, 1.0, 0.0]])
        dm_A = np.array([[[0.5, -0.5], [0.5, -0.5]]])  # (1, 2, 2)
        dm_B = np.array([[[0.5, -0.5], [0.5, -0.5]]])

        s = np.array([0.4])
        t = np.array([0.6])

        ds_batch, dt_batch, valid_batch = _batch_st_jacobian_hermite(
            xA0,
            xA1,
            xB0,
            xB1,
            s,
            t,
            s,
            t,
            mA0,
            mA1,
            mB0,
            mB1,
            dm_A,
            dm_B,
        )

        proc = ComputeStJacobianProcess()
        out = proc.process(
            StJacobianInput(
                xA0=xA0[0],
                xA1=xA1[0],
                xB0=xB0[0],
                xB1=xB1[0],
                s=0.4,
                t=0.6,
                s_unclamped=0.4,
                t_unclamped=0.6,
                mA0=mA0[0],
                mA1=mA1[0],
                mB0=mB0[0],
                mB1=mB1[0],
                use_hermite=True,
                dm_A=dm_A[0],
                dm_B=dm_B[0],
            )
        )

        np.testing.assert_allclose(ds_batch[0], out.ds_du, atol=1e-12)
        np.testing.assert_allclose(dt_batch[0], out.dt_du, atol=1e-12)
        assert valid_batch[0]

    def test_multi_pair_matches_scalar(self):
        """複数ペアでスカラー版と一致（Hermite）."""
        from xkep_cae.contact.geometry._st_jacobian import (
            _batch_st_jacobian_hermite,
        )

        rng = np.random.default_rng(123)
        N = 15
        xA0 = rng.uniform(-1, 1, (N, 3))
        xA1 = xA0 + rng.uniform(0.5, 1.5, (N, 3))
        xB0 = rng.uniform(-1, 1, (N, 3))
        xB1 = xB0 + rng.uniform(0.5, 1.5, (N, 3))
        mA0 = rng.uniform(-0.5, 0.5, (N, 3))
        mA1 = rng.uniform(-0.5, 0.5, (N, 3))
        mB0 = rng.uniform(-0.5, 0.5, (N, 3))
        mB1 = rng.uniform(-0.5, 0.5, (N, 3))
        s = rng.uniform(0.1, 0.9, N)
        t = rng.uniform(0.1, 0.9, N)
        dm_A = rng.uniform(-0.5, 0.5, (N, 2, 2))
        dm_B = rng.uniform(-0.5, 0.5, (N, 2, 2))

        ds_batch, dt_batch, valid_batch = _batch_st_jacobian_hermite(
            xA0,
            xA1,
            xB0,
            xB1,
            s,
            t,
            s,
            t,
            mA0,
            mA1,
            mB0,
            mB1,
            dm_A,
            dm_B,
        )

        proc = ComputeStJacobianProcess()
        for i in range(N):
            out = proc.process(
                StJacobianInput(
                    xA0=xA0[i],
                    xA1=xA1[i],
                    xB0=xB0[i],
                    xB1=xB1[i],
                    s=float(s[i]),
                    t=float(t[i]),
                    s_unclamped=float(s[i]),
                    t_unclamped=float(t[i]),
                    mA0=mA0[i],
                    mA1=mA1[i],
                    mB0=mB0[i],
                    mB1=mB1[i],
                    use_hermite=True,
                    dm_A=dm_A[i],
                    dm_B=dm_B[i],
                )
            )
            np.testing.assert_allclose(
                ds_batch[i],
                out.ds_du,
                atol=1e-10,
                err_msg=f"ds_du mismatch at pair {i}",
            )
            np.testing.assert_allclose(
                dt_batch[i],
                out.dt_du,
                atol=1e-10,
                err_msg=f"dt_du mismatch at pair {i}",
            )
