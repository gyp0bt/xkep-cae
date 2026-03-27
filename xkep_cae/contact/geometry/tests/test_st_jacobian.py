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
