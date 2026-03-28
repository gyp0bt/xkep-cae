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
