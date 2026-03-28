"""摩擦アセンブリ Process のテスト（status-256 B2-B4）.

B4: FrictionTangentStiffnessProcess
B2: FrictionGeometricStiffnessProcess
B3: FrictionStStiffnessProcess
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
