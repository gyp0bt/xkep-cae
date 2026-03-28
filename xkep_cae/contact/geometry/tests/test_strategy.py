"""ContactGeometryStrategy 具象実装のテスト.

@binds_to による 1:1 紐付け + ContactGeometryStrategy Protocol 適合テスト。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact.geometry import (
    BatchUpdateGeometryInput,
    BatchUpdateGeometryProcess,
    ContactFrameInput,
    ContactFrameProcess,
    ContactGeometryInput,
    ContactGeometryOutput,
    LineToLineGaussProcess,
    MortarSegmentProcess,
    PointToPointProcess,
)
from xkep_cae.contact.geometry.strategy import (
    _create_contact_geometry_strategy,
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

    def test_exclude_same_strand_default(self):
        proc = PointToPointProcess()
        assert proc._exclude_same_strand is True

    def test_exclude_same_strand_false(self):
        proc = PointToPointProcess(exclude_same_strand=False)
        assert proc._exclude_same_strand is False


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
        s = _create_contact_geometry_strategy()
        assert isinstance(s, PointToPointProcess)

    def test_point_to_point_mode(self):
        s = _create_contact_geometry_strategy(mode="point_to_point")
        assert isinstance(s, PointToPointProcess)

    def test_line_to_line_mode(self):
        s = _create_contact_geometry_strategy(mode="line_to_line")
        assert isinstance(s, LineToLineGaussProcess)

    def test_mortar_mode(self):
        s = _create_contact_geometry_strategy(mode="mortar")
        assert isinstance(s, MortarSegmentProcess)

    def test_line_contact_flag(self):
        s = _create_contact_geometry_strategy(line_contact=True)
        assert isinstance(s, LineToLineGaussProcess)

    def test_use_mortar_flag(self):
        s = _create_contact_geometry_strategy(use_mortar=True)
        assert isinstance(s, MortarSegmentProcess)

    def test_mortar_overrides_line_contact(self):
        s = _create_contact_geometry_strategy(line_contact=True, use_mortar=True)
        assert isinstance(s, MortarSegmentProcess)

    def test_custom_n_gauss(self):
        s = _create_contact_geometry_strategy(mode="line_to_line", n_gauss=4)
        assert isinstance(s, LineToLineGaussProcess)
        assert s._n_gauss == 4

    def test_auto_gauss(self):
        s = _create_contact_geometry_strategy(mode="line_to_line", auto_gauss=True)
        assert isinstance(s, LineToLineGaussProcess)
        assert s._auto_gauss is True


# ── ContactFrameProcess (C3) ──────────────────────────��──────


@binds_to(ContactFrameProcess)
class TestContactFrameProcess:
    """ContactFrameProcess の単体テスト（status-255 C3）."""

    def test_empty_input(self):
        proc = ContactFrameProcess()
        out = proc.process(ContactFrameInput(normals=np.empty((0, 3))))
        assert out.n.shape == (0, 3)
        assert out.t1.shape == (0, 3)
        assert out.t2.shape == (0, 3)

    def test_single_normal(self):
        proc = ContactFrameProcess()
        normals = np.array([[0.0, 0.0, 1.0]])
        out = proc.process(ContactFrameInput(normals=normals))
        assert out.n.shape == (1, 3)
        np.testing.assert_allclose(out.n[0], [0, 0, 1], atol=1e-10)
        # t1, t2 は n に直交
        assert abs(np.dot(out.t1[0], out.n[0])) < 1e-10
        assert abs(np.dot(out.t2[0], out.n[0])) < 1e-10
        # t1 × t2 ≈ n (右手系)
        cross = np.cross(out.t1[0], out.t2[0])
        np.testing.assert_allclose(np.abs(np.dot(cross, out.n[0])), 1.0, atol=1e-10)

    def test_batch_normals(self):
        proc = ContactFrameProcess()
        normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        out = proc.process(ContactFrameInput(normals=normals))
        assert out.n.shape == (3, 3)
        for i in range(3):
            # 正規直交性
            assert abs(np.dot(out.t1[i], out.n[i])) < 1e-10
            assert abs(np.dot(out.t2[i], out.n[i])) < 1e-10
            assert abs(np.dot(out.t1[i], out.t2[i])) < 1e-10

    def test_parallel_transport(self):
        proc = ContactFrameProcess()
        normals = np.array([[0.0, 0.0, 1.0]])
        prev_normals = np.array([[0.0, 0.1, 0.995]])
        prev_t1 = np.array([[1.0, 0.0, 0.0]])
        out = proc.process(
            ContactFrameInput(
                normals=normals,
                prev_tangent1s=prev_t1,
                prev_normals=prev_normals,
                has_prev_mask=np.array([True]),
                has_prev_n_mask=np.array([True]),
            )
        )
        # t1 は前ステップから連続的に輸送されている
        assert abs(np.dot(out.t1[0], out.n[0])) < 1e-10
        # t1 が前ステップの t1 と近い方向を向く
        assert np.dot(out.t1[0], prev_t1[0]) > 0.9


# ── BatchUpdateGeometryProcess (C2) ──────────────────────────


@binds_to(BatchUpdateGeometryProcess)
class TestBatchUpdateGeometryProcess:
    """BatchUpdateGeometryProcess の単体テスト（status-255 C2）."""

    def test_empty_pairs(self):
        proc = BatchUpdateGeometryProcess()
        out = proc.process(
            BatchUpdateGeometryInput(
                pairs=[],
                node_coords=np.zeros((4, 3)),
            )
        )
        assert out.pairs == []

    def test_returns_updated_pairs(self):
        """2本近接セグメントでギャップが正しく計算される."""
        proc = BatchUpdateGeometryProcess()
        # 2本の近接平行セグメント: A=(0,0,0)→(1,0,0), B=(0,0.15,0)→(1,0.15,0)
        # 距離=0.15, 半径=0.1 → ギャップ = 0.15 - 0.2 = -0.05（貫入）
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.15, 0.0],
                [1.0, 0.15, 0.0],
            ]
        )
        conn = np.array([[0, 1], [2, 3]], dtype=int)

        # PointToPoint で候補検出して初期ペアを生成
        ptp = PointToPointProcess(exclude_same_strand=False)
        pairs = ptp.detect(coords, conn, 0.1)
        assert len(pairs) > 0

        # BatchUpdateGeometryProcess でペアを更新
        out = proc.process(
            BatchUpdateGeometryInput(
                pairs=pairs,
                node_coords=coords,
            )
        )
        assert len(out.pairs) > 0
        for p in out.pairs:
            assert hasattr(p, "state")
            # ギャップ = 0.15 - 0.2 = -0.05（貫入）
            assert p.state.gap < 0.0
