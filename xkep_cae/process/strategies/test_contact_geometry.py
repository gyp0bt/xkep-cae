"""ContactGeometry Strategy 具象実装の1:1テスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.process.strategies.contact_geometry import (
    ContactGeometryInput,
    LineToLineGaussProcess,
    MortarSegmentProcess,
    PointToPointProcess,
)
from xkep_cae.process.strategies.protocols import ContactGeometryStrategy
from xkep_cae.process.testing import binds_to

# --- Protocol 準拠チェック ---


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


# --- PointToPoint ---


class _MockState:
    def __init__(self, gap=0.0):
        self.gap = gap


class _MockPair:
    def __init__(self, gap=0.0):
        self.state = _MockState(gap=gap)


@binds_to(PointToPointProcess)
class TestPointToPointProcess:
    """PointToPointProcess の単体テスト."""

    def test_detect_returns_empty(self):
        """Phase 3 前はスタブで空リスト."""
        proc = PointToPointProcess()
        pairs = proc.detect(np.zeros((4, 3)), np.array([[0, 1]]), 0.5)
        assert pairs == []

    def test_compute_gap_from_state(self):
        """pair.state.gap を返す."""
        proc = PointToPointProcess()
        pair = _MockPair(gap=-0.01)
        assert proc.compute_gap(pair, np.zeros((4, 3))) == pytest.approx(-0.01)

    def test_compute_gap_no_state(self):
        """state がない場合は 0.0."""
        proc = PointToPointProcess()
        assert proc.compute_gap(object(), np.zeros((4, 3))) == 0.0

    def test_exclude_same_layer_default(self):
        proc = PointToPointProcess()
        assert proc._exclude_same_layer is True

    def test_process_method(self):
        proc = PointToPointProcess()
        inp = ContactGeometryInput(
            node_coords=np.zeros((4, 3)),
            connectivity=np.array([[0, 1]]),
            radii=0.5,
        )
        out = proc.process(inp)
        assert out.contact_pairs == []

    def test_meta(self):
        assert PointToPointProcess.meta.name == "PointToPoint"
        assert not PointToPointProcess.meta.deprecated


# --- LineToLineGauss ---


@binds_to(LineToLineGaussProcess)
class TestLineToLineGaussProcess:
    """LineToLineGaussProcess の単体テスト."""

    def test_detect_returns_empty(self):
        proc = LineToLineGaussProcess()
        pairs = proc.detect(np.zeros((4, 3)), np.array([[0, 1]]), 0.5)
        assert pairs == []

    def test_n_gauss_default(self):
        proc = LineToLineGaussProcess()
        assert proc._n_gauss == 2

    def test_n_gauss_custom(self):
        proc = LineToLineGaussProcess(n_gauss=4)
        assert proc._n_gauss == 4

    def test_compute_gap(self):
        proc = LineToLineGaussProcess()
        pair = _MockPair(gap=-0.005)
        assert proc.compute_gap(pair, np.zeros((4, 3))) == pytest.approx(-0.005)

    def test_process_method(self):
        proc = LineToLineGaussProcess()
        inp = ContactGeometryInput(
            node_coords=np.zeros((4, 3)),
            connectivity=np.array([[0, 1]]),
            radii=0.5,
        )
        out = proc.process(inp)
        assert out.contact_pairs == []

    def test_meta(self):
        assert LineToLineGaussProcess.meta.name == "LineToLineGauss"
        assert not LineToLineGaussProcess.meta.deprecated


# --- MortarSegment ---


@binds_to(MortarSegmentProcess)
class TestMortarSegmentProcess:
    """MortarSegmentProcess の単体テスト."""

    def test_detect_returns_empty(self):
        proc = MortarSegmentProcess()
        pairs = proc.detect(np.zeros((4, 3)), np.array([[0, 1]]), 0.5)
        assert pairs == []

    def test_n_gauss_default(self):
        proc = MortarSegmentProcess()
        assert proc._n_gauss == 2

    def test_compute_gap(self):
        proc = MortarSegmentProcess()
        pair = _MockPair(gap=-0.02)
        assert proc.compute_gap(pair, np.zeros((4, 3))) == pytest.approx(-0.02)

    def test_process_method(self):
        proc = MortarSegmentProcess()
        inp = ContactGeometryInput(
            node_coords=np.zeros((4, 3)),
            connectivity=np.array([[0, 1]]),
            radii=0.5,
        )
        out = proc.process(inp)
        assert out.contact_pairs == []

    def test_meta(self):
        assert MortarSegmentProcess.meta.name == "MortarSegment"
        assert not MortarSegmentProcess.meta.deprecated
