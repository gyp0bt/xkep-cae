"""ContactGeometry Strategy 具象実装の1:1テスト."""

from __future__ import annotations

import numpy as np
import pytest

from __xkep_cae_deprecated.contact.pair import ContactPair, ContactStatus
from __xkep_cae_deprecated.process.strategies.contact_geometry import (
    ContactGeometryInput,
    LineToLineGaussProcess,
    MortarSegmentProcess,
    PointToPointProcess,
    create_contact_geometry_strategy,
)
from __xkep_cae_deprecated.process.strategies.protocols import ContactGeometryStrategy
from __xkep_cae_deprecated.process.testing import binds_to

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


# --- ヘルパー ---


def _make_pair(
    elem_a: int,
    elem_b: int,
    nodes_a: list[int],
    nodes_b: list[int],
    radius: float = 0.5,
) -> ContactPair:
    """テスト用 ContactPair を生成."""
    pair = ContactPair(
        elem_a=elem_a,
        elem_b=elem_b,
        nodes_a=np.array(nodes_a, dtype=int),
        nodes_b=np.array(nodes_b, dtype=int),
        radius_a=radius,
        radius_b=radius,
    )
    return pair


def _make_two_segment_system():
    """2セグメント（4ノード）の十字配置.

    セグメントA: ノード0=[0,0,0] → ノード1=[1,0,0]  (X軸方向)
    セグメントB: ノード2=[0.5,-0.5,0.3] → ノード3=[0.5,0.5,0.3]  (Y軸方向)
    半径 r=0.1 → 中心間距離 ≈ 0.3 → gap ≈ 0.3 - 0.2 = 0.1
    """
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -0.5, 0.3],
            [0.5, 0.5, 0.3],
        ]
    )
    pair = _make_pair(0, 1, [0, 1], [2, 3], radius=0.1)
    return node_coords, pair


def _make_penetrating_system():
    """貫入状態の2セグメント.

    セグメントA: ノード0=[0,0,0] → ノード1=[1,0,0]
    セグメントB: ノード2=[0.5,-0.5,0.05] → ノード3=[0.5,0.5,0.05]
    半径 r=0.1 → gap ≈ 0.05 - 0.2 = -0.15
    """
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -0.5, 0.05],
            [0.5, 0.5, 0.05],
        ]
    )
    pair = _make_pair(0, 1, [0, 1], [2, 3], radius=0.1)
    return node_coords, pair


# --- PointToPoint ---


@binds_to(PointToPointProcess)
class TestPointToPointProcess:
    """PointToPointProcess の単体テスト."""

    def test_detect_returns_empty(self):
        proc = PointToPointProcess()
        pairs = proc.detect(np.zeros((4, 3)), np.array([[0, 1]]), 0.5)
        assert pairs == []

    def test_compute_gap_from_state(self):
        proc = PointToPointProcess()
        _, pair = _make_two_segment_system()
        pair.state.gap = -0.01
        assert proc.compute_gap(pair, np.zeros((4, 3))) == pytest.approx(-0.01)

    def test_compute_gap_no_state(self):
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

    def test_update_geometry_basic(self):
        """update_geometry で s, t, gap, normal が更新される."""
        proc = PointToPointProcess()
        node_coords, pair = _make_two_segment_system()
        proc.update_geometry([pair], node_coords)

        # s ≈ 0.5（A上の中点付近）, t ≈ 0.5（B上の中点付近）
        assert 0.0 <= pair.state.s <= 1.0
        assert 0.0 <= pair.state.t <= 1.0
        # gap ≈ 0.3 - 0.2 = 0.1 (正 = 離間)
        assert pair.state.gap > 0.0
        # 法線は非ゼロ
        assert np.linalg.norm(pair.state.normal) > 0.5

    def test_update_geometry_penetration(self):
        """貫入状態で gap < 0 かつ ACTIVE."""
        proc = PointToPointProcess()
        node_coords, pair = _make_penetrating_system()
        proc.update_geometry([pair], node_coords)

        assert pair.state.gap < 0.0
        assert pair.state.status == ContactStatus.ACTIVE

    def test_update_geometry_separated(self):
        """離間状態で gap > 0 かつ INACTIVE."""
        proc = PointToPointProcess()
        node_coords, pair = _make_two_segment_system()
        proc.update_geometry([pair], node_coords)

        assert pair.state.gap > 0.0
        assert pair.state.status == ContactStatus.INACTIVE

    def test_update_geometry_empty_pairs(self):
        """空リストで例外なし."""
        proc = PointToPointProcess()
        proc.update_geometry([], np.zeros((4, 3)))

    def test_update_geometry_frame_orthogonality(self):
        """接触フレーム (n, t1, t2) が正規直交."""
        proc = PointToPointProcess()
        node_coords, pair = _make_two_segment_system()
        proc.update_geometry([pair], node_coords)

        n = pair.state.normal
        t1 = pair.state.tangent1
        t2 = pair.state.tangent2

        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-10)
        assert np.linalg.norm(t1) == pytest.approx(1.0, abs=1e-10)
        assert np.linalg.norm(t2) == pytest.approx(1.0, abs=1e-10)
        assert abs(n @ t1) < 1e-10
        assert abs(n @ t2) < 1e-10
        assert abs(t1 @ t2) < 1e-10

    def test_update_geometry_multiple_pairs(self):
        """複数ペアの一括更新."""
        proc = PointToPointProcess()
        node_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, -0.5, 0.3],
                [0.5, 0.5, 0.3],
                [0.3, -0.5, 0.05],
                [0.3, 0.5, 0.05],
            ]
        )
        pair1 = _make_pair(0, 1, [0, 1], [2, 3], radius=0.1)
        pair2 = _make_pair(2, 3, [0, 1], [4, 5], radius=0.1)
        proc.update_geometry([pair1, pair2], node_coords)

        # 両ペアとも幾何情報が更新される
        assert pair1.state.gap != 0.0
        assert pair2.state.gap != 0.0

    def test_build_constraint_jacobian_empty(self):
        """アクティブペアなしで空行列."""
        proc = PointToPointProcess()
        G, active_idx = proc.build_constraint_jacobian([], ndof_total=24)
        assert G.shape == (0, 24)
        assert len(active_idx) == 0

    def test_build_constraint_jacobian_active_pair(self):
        """アクティブペアで G 行列が構築される."""
        proc = PointToPointProcess()
        node_coords, pair = _make_penetrating_system()
        proc.update_geometry([pair], node_coords)

        # ペアがACTIVEになったことを確認
        assert pair.state.status == ContactStatus.ACTIVE

        ndof = 24  # 4ノード × 6DOF
        G, active_idx = proc.build_constraint_jacobian([pair], ndof_total=ndof)
        assert G.shape == (1, ndof)
        assert len(active_idx) == 1
        assert active_idx[0] == 0
        # G は非ゼロ
        assert G.nnz > 0

    def test_build_constraint_jacobian_inactive_pair(self):
        """INACTIVEペアは G に含まれない."""
        proc = PointToPointProcess()
        node_coords, pair = _make_two_segment_system()
        proc.update_geometry([pair], node_coords)

        assert pair.state.status == ContactStatus.INACTIVE

        G, active_idx = proc.build_constraint_jacobian([pair], ndof_total=24)
        assert G.shape == (0, 24)
        assert len(active_idx) == 0


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

    def test_auto_gauss_flag(self):
        proc = LineToLineGaussProcess(auto_gauss=True)
        assert proc._auto_gauss is True

    def test_compute_gap(self):
        proc = LineToLineGaussProcess()
        _, pair = _make_two_segment_system()
        pair.state.gap = -0.005
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

    def test_update_geometry_basic(self):
        """update_geometry で幾何情報が更新される."""
        proc = LineToLineGaussProcess()
        node_coords, pair = _make_penetrating_system()
        proc.update_geometry([pair], node_coords)

        assert pair.state.gap < 0.0
        assert pair.state.status == ContactStatus.ACTIVE

    def test_update_geometry_empty_pairs(self):
        """空リストで例外なし."""
        proc = LineToLineGaussProcess()
        proc.update_geometry([], np.zeros((4, 3)))

    def test_build_constraint_jacobian(self):
        """制約ヤコビアンが構築される."""
        proc = LineToLineGaussProcess()
        node_coords, pair = _make_penetrating_system()
        proc.update_geometry([pair], node_coords)

        G, active_idx = proc.build_constraint_jacobian([pair], ndof_total=24)
        assert G.shape == (1, 24)
        assert len(active_idx) == 1


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
        _, pair = _make_two_segment_system()
        pair.state.gap = -0.02
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

    def test_update_geometry_basic(self):
        """update_geometry で幾何情報が更新される."""
        proc = MortarSegmentProcess()
        node_coords, pair = _make_penetrating_system()
        proc.update_geometry([pair], node_coords)

        assert pair.state.gap < 0.0
        assert pair.state.status == ContactStatus.ACTIVE

    def test_update_geometry_empty_pairs(self):
        """空リストで例外なし."""
        proc = MortarSegmentProcess()
        proc.update_geometry([], np.zeros((4, 3)))

    def test_build_constraint_jacobian(self):
        """制約ヤコビアンが構築される."""
        proc = MortarSegmentProcess()
        node_coords, pair = _make_penetrating_system()
        proc.update_geometry([pair], node_coords)

        G, active_idx = proc.build_constraint_jacobian([pair], ndof_total=24)
        assert G.shape == (1, 24)
        assert len(active_idx) == 1


# --- create_contact_geometry_strategy ファクトリ ---


class TestCreateContactGeometryStrategy:
    """create_contact_geometry_strategy ファクトリのテスト."""

    def test_default_ptp(self):
        """デフォルトは PointToPoint."""
        strategy = create_contact_geometry_strategy()
        assert isinstance(strategy, PointToPointProcess)

    def test_explicit_ptp(self):
        """mode='point_to_point' → PointToPointProcess."""
        strategy = create_contact_geometry_strategy(mode="point_to_point")
        assert isinstance(strategy, PointToPointProcess)

    def test_line_to_line(self):
        """mode='line_to_line' → LineToLineGaussProcess."""
        strategy = create_contact_geometry_strategy(mode="line_to_line")
        assert isinstance(strategy, LineToLineGaussProcess)

    def test_mortar(self):
        """mode='mortar' → MortarSegmentProcess."""
        strategy = create_contact_geometry_strategy(mode="mortar")
        assert isinstance(strategy, MortarSegmentProcess)

    def test_line_contact_flag(self):
        """line_contact=True → LineToLineGaussProcess."""
        strategy = create_contact_geometry_strategy(line_contact=True)
        assert isinstance(strategy, LineToLineGaussProcess)

    def test_use_mortar_flag(self):
        """use_mortar=True → MortarSegmentProcess."""
        strategy = create_contact_geometry_strategy(use_mortar=True)
        assert isinstance(strategy, MortarSegmentProcess)

    def test_n_gauss_propagation(self):
        """n_gauss パラメータの伝播."""
        strategy = create_contact_geometry_strategy(mode="line_to_line", n_gauss=5)
        assert isinstance(strategy, LineToLineGaussProcess)
        assert strategy._n_gauss == 5

    def test_auto_gauss_propagation(self):
        """auto_gauss パラメータの伝播."""
        strategy = create_contact_geometry_strategy(mode="line_to_line", auto_gauss=True)
        assert isinstance(strategy, LineToLineGaussProcess)
        assert strategy._auto_gauss is True

    def test_exclude_same_layer(self):
        """exclude_same_layer パラメータの伝播."""
        strategy = create_contact_geometry_strategy(exclude_same_layer=False)
        assert isinstance(strategy, PointToPointProcess)
        assert strategy._exclude_same_layer is False

    def test_use_mortar_overrides_line_contact(self):
        """use_mortar は line_contact より優先."""
        strategy = create_contact_geometry_strategy(line_contact=True, use_mortar=True)
        assert isinstance(strategy, MortarSegmentProcess)
