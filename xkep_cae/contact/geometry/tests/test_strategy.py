"""ContactGeometryStrategy 具象実装のテスト.

@binds_to による 1:1 紐付け + ContactGeometryStrategy Protocol 適合テスト。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact.geometry import (
    ContactGeometryInput,
    ContactGeometryOutput,
    LineToLineGaussProcess,
    MortarSegmentProcess,
    PointToPointProcess,
    create_contact_geometry_strategy,
)
from xkep_cae.core.strategies import ContactGeometryStrategy
from xkep_cae.core.testing import binds_to

# ── Protocol 準拠 ─────────────────────────────────────────


class TestContactGeometryProtocolConformance:
    """全 ContactGeometry 具象が Protocol を満たすことを検証."""

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (PointToPointProcess, {}),
            (LineToLineGaussProcess, {}),
            (MortarSegmentProcess, {}),
        ],
    )
    def test_protocol_conformance(self, cls, kwargs):
        instance = cls(**kwargs)
        assert isinstance(instance, ContactGeometryStrategy)


# ── PointToPoint ───────────────────────────────────────────


@binds_to(PointToPointProcess)
class TestPointToPointProcess:
    """PointToPointProcess の単体テスト."""

    def test_detect_returns_empty(self):
        proc = PointToPointProcess()
        pairs = proc.detect(np.zeros((4, 3)), np.zeros((2, 2), dtype=int), 1.0)
        assert pairs == []

    def test_compute_gap_no_state(self):
        proc = PointToPointProcess()

        class DummyPair:
            pass

        pair = DummyPair()
        assert proc.compute_gap(pair, np.zeros((4, 3))) == 0.0

    def test_compute_gap_with_state(self):
        proc = PointToPointProcess()

        class DummyState:
            gap = -0.5

        class DummyPair:
            state = DummyState()

        pair = DummyPair()
        assert proc.compute_gap(pair, np.zeros((4, 3))) == -0.5

    def test_update_geometry_empty_pairs(self):
        proc = PointToPointProcess()
        proc.update_geometry([], np.zeros((4, 3)))

    def test_build_constraint_jacobian_empty(self):
        proc = PointToPointProcess()
        G, indices = proc.build_constraint_jacobian([], ndof_total=12)
        assert G.shape == (0, 12)
        assert indices == []

    def test_process_returns_output(self):
        proc = PointToPointProcess()
        inp = ContactGeometryInput(
            node_coords=np.zeros((4, 3)),
            connectivity=np.zeros((2, 2), dtype=int),
            radii=1.0,
        )
        out = proc.process(inp)
        assert isinstance(out, ContactGeometryOutput)
        assert out.contact_pairs == []

    def test_exclude_same_layer_default(self):
        proc = PointToPointProcess()
        assert proc._exclude_same_layer is True

    def test_exclude_same_layer_false(self):
        proc = PointToPointProcess(exclude_same_layer=False)
        assert proc._exclude_same_layer is False


# ── LineToLineGauss ────────────────────────────────────────


@binds_to(LineToLineGaussProcess)
class TestLineToLineGaussProcess:
    """LineToLineGaussProcess の単体テスト."""

    def test_detect_returns_empty(self):
        proc = LineToLineGaussProcess()
        pairs = proc.detect(np.zeros((4, 3)), np.zeros((2, 2), dtype=int), 1.0)
        assert pairs == []

    def test_compute_gap_no_state(self):
        proc = LineToLineGaussProcess()

        class DummyPair:
            pass

        assert proc.compute_gap(DummyPair(), np.zeros((4, 3))) == 0.0

    def test_update_geometry_empty_pairs(self):
        proc = LineToLineGaussProcess()
        proc.update_geometry([], np.zeros((4, 3)))

    def test_build_constraint_jacobian_empty(self):
        proc = LineToLineGaussProcess()
        G, indices = proc.build_constraint_jacobian([], ndof_total=12)
        assert G.shape == (0, 12)
        assert indices == []

    def test_process_returns_output(self):
        proc = LineToLineGaussProcess()
        inp = ContactGeometryInput(
            node_coords=np.zeros((4, 3)),
            connectivity=np.zeros((2, 2), dtype=int),
            radii=1.0,
        )
        out = proc.process(inp)
        assert isinstance(out, ContactGeometryOutput)

    def test_n_gauss_default(self):
        proc = LineToLineGaussProcess()
        assert proc._n_gauss == 2

    def test_n_gauss_custom(self):
        proc = LineToLineGaussProcess(n_gauss=4)
        assert proc._n_gauss == 4

    def test_auto_gauss_flag(self):
        proc = LineToLineGaussProcess(auto_gauss=True)
        assert proc._auto_gauss is True


# ── MortarSegment ──────────────────────────────────────────


@binds_to(MortarSegmentProcess)
class TestMortarSegmentProcess:
    """MortarSegmentProcess の単体テスト."""

    def test_detect_returns_empty(self):
        proc = MortarSegmentProcess()
        pairs = proc.detect(np.zeros((4, 3)), np.zeros((2, 2), dtype=int), 1.0)
        assert pairs == []

    def test_compute_gap_no_state(self):
        proc = MortarSegmentProcess()

        class DummyPair:
            pass

        assert proc.compute_gap(DummyPair(), np.zeros((4, 3))) == 0.0

    def test_update_geometry_empty_pairs(self):
        proc = MortarSegmentProcess()
        proc.update_geometry([], np.zeros((4, 3)))

    def test_build_constraint_jacobian_empty(self):
        proc = MortarSegmentProcess()
        G, indices = proc.build_constraint_jacobian([], ndof_total=12)
        assert G.shape == (0, 12)
        assert indices == []

    def test_process_returns_output(self):
        proc = MortarSegmentProcess()
        inp = ContactGeometryInput(
            node_coords=np.zeros((4, 3)),
            connectivity=np.zeros((2, 2), dtype=int),
            radii=1.0,
        )
        out = proc.process(inp)
        assert isinstance(out, ContactGeometryOutput)

    def test_n_gauss_default(self):
        proc = MortarSegmentProcess()
        assert proc._n_gauss == 2


# ── ファクトリ ─────────────────────────────────────────────


class TestCreateContactGeometryStrategy:
    """ファクトリ関数のテスト."""

    def test_default_returns_ptp(self):
        s = create_contact_geometry_strategy()
        assert isinstance(s, PointToPointProcess)

    def test_point_to_point_mode(self):
        s = create_contact_geometry_strategy(mode="point_to_point")
        assert isinstance(s, PointToPointProcess)

    def test_line_to_line_mode(self):
        s = create_contact_geometry_strategy(mode="line_to_line")
        assert isinstance(s, LineToLineGaussProcess)

    def test_mortar_mode(self):
        s = create_contact_geometry_strategy(mode="mortar")
        assert isinstance(s, MortarSegmentProcess)

    def test_line_contact_flag(self):
        s = create_contact_geometry_strategy(line_contact=True)
        assert isinstance(s, LineToLineGaussProcess)

    def test_use_mortar_flag(self):
        s = create_contact_geometry_strategy(use_mortar=True)
        assert isinstance(s, MortarSegmentProcess)

    def test_mortar_overrides_line_contact(self):
        s = create_contact_geometry_strategy(line_contact=True, use_mortar=True)
        assert isinstance(s, MortarSegmentProcess)

    def test_custom_n_gauss(self):
        s = create_contact_geometry_strategy(mode="line_to_line", n_gauss=4)
        assert isinstance(s, LineToLineGaussProcess)
        assert s._n_gauss == 4

    def test_auto_gauss(self):
        s = create_contact_geometry_strategy(mode="line_to_line", auto_gauss=True)
        assert isinstance(s, LineToLineGaussProcess)
        assert s._auto_gauss is True
