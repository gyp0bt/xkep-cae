"""Coating Strategy 具象実装の1:1テスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae_deprecated.contact.pair import ContactConfig, ContactPair, ContactState, ContactStatus
from xkep_cae_deprecated.process.strategies.coating import (
    KelvinVoigtCoatingProcess,
    NoCoatingProcess,
    create_coating_strategy,
)
from xkep_cae_deprecated.process.strategies.protocols import CoatingStrategy
from xkep_cae_deprecated.process.testing import binds_to

# --- Protocol 準拠チェック ---


class TestCoatingProtocolConformance:
    """全 Coating 具象が CoatingStrategy Protocol を満たすことを検証."""

    @pytest.mark.parametrize("cls", [NoCoatingProcess, KelvinVoigtCoatingProcess])
    def test_protocol_conformance(self, cls):
        instance = cls()
        assert isinstance(instance, CoatingStrategy)


# --- NoCoatingProcess ---


@binds_to(NoCoatingProcess)
class TestNoCoatingProcess:
    """NoCoatingProcess のテスト."""

    def test_forces_zero(self):
        proc = NoCoatingProcess()
        config = ContactConfig()
        coords = np.zeros((4, 3))
        result = proc.forces([], coords, config, dt=1.0)
        assert np.allclose(result, 0.0)

    def test_stiffness_zero(self):
        proc = NoCoatingProcess()
        config = ContactConfig()
        coords = np.zeros((4, 3))
        K = proc.stiffness([], coords, config, ndof_total=24, dt=1.0)
        assert K.shape == (24, 24)
        assert K.nnz == 0

    def test_friction_forces_zero(self):
        proc = NoCoatingProcess()
        config = ContactConfig()
        coords = np.zeros((4, 3))
        u = np.zeros(24)
        result = proc.friction_forces([], coords, config, u, u)
        assert np.allclose(result, 0.0)

    def test_friction_stiffness_zero(self):
        proc = NoCoatingProcess()
        config = ContactConfig()
        coords = np.zeros((4, 3))
        K = proc.friction_stiffness([], coords, config, ndof_total=24)
        assert K.shape == (24, 24)
        assert K.nnz == 0


# --- KelvinVoigtCoatingProcess ---


def _make_coating_pair(compression: float = 0.1) -> ContactPair:
    """テスト用の被膜接触ペアを作成."""
    pair = ContactPair(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array([0, 1]),
        nodes_b=np.array([2, 3]),
        radius_a=0.5,
        radius_b=0.5,
    )
    pair.state = ContactState(
        s=0.5,
        t=0.5,
        gap=-0.01,
        normal=np.array([0.0, 1.0, 0.0]),
        tangent1=np.array([1.0, 0.0, 0.0]),
        tangent2=np.array([0.0, 0.0, 1.0]),
        status=ContactStatus.ACTIVE,
        coating_compression=compression,
        coating_compression_prev=compression * 0.5,
    )
    return pair


@binds_to(KelvinVoigtCoatingProcess)
class TestKelvinVoigtCoatingProcess:
    """KelvinVoigtCoatingProcess のテスト."""

    def test_forces_nonzero(self):
        proc = KelvinVoigtCoatingProcess()
        config = ContactConfig(coating_stiffness=1e6, coating_damping=0.0)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        pair = _make_coating_pair(compression=0.1)
        result = proc.forces([pair], coords, config, dt=1.0)
        assert result.shape == (24,)
        assert not np.allclose(result, 0.0)

    def test_forces_zero_when_no_compression(self):
        proc = KelvinVoigtCoatingProcess()
        config = ContactConfig(coating_stiffness=1e6)
        coords = np.zeros((4, 3))
        pair = _make_coating_pair(compression=0.0)
        pair.state.coating_compression_prev = 0.0
        result = proc.forces([pair], coords, config, dt=1.0)
        assert np.allclose(result, 0.0)

    def test_stiffness_nonzero(self):
        proc = KelvinVoigtCoatingProcess()
        config = ContactConfig(coating_stiffness=1e6)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        pair = _make_coating_pair(compression=0.1)
        K = proc.stiffness([pair], coords, config, ndof_total=24, dt=1.0)
        assert K.shape == (24, 24)
        assert K.nnz > 0

    def test_damping_contribution(self):
        """粘性ダッシュポットが力に寄与することを確認."""
        proc = KelvinVoigtCoatingProcess()
        config_no_damp = ContactConfig(coating_stiffness=1e6, coating_damping=0.0)
        config_damp = ContactConfig(coating_stiffness=1e6, coating_damping=1e3)
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        pair1 = _make_coating_pair(compression=0.1)
        pair2 = _make_coating_pair(compression=0.1)
        f_no_damp = proc.forces([pair1], coords, config_no_damp, dt=0.01)
        f_damp = proc.forces([pair2], coords, config_damp, dt=0.01)
        # 粘性ありの方が大きい力を生成する
        assert np.linalg.norm(f_damp) > np.linalg.norm(f_no_damp)

    def test_friction_forces(self):
        """被膜摩擦力が非ゼロを確認."""
        proc = KelvinVoigtCoatingProcess()
        config = ContactConfig(
            coating_stiffness=1e6,
            coating_mu=0.3,
            coating_k_t_ratio=0.5,
        )
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        pair = _make_coating_pair(compression=0.1)
        u_cur = np.zeros(24)
        u_ref = np.zeros(24)
        # 接線方向の相対変位を作る
        u_cur[0] = 0.01  # node A0 の x変位
        result = proc.friction_forces([pair], coords, config, u_cur, u_ref)
        assert result.shape == (24,)
        assert not np.allclose(result, 0.0)

    def test_friction_stiffness(self):
        """被膜摩擦剛性行列が非ゼロを確認."""
        proc = KelvinVoigtCoatingProcess()
        config = ContactConfig(
            coating_stiffness=1e6,
            coating_mu=0.3,
            coating_k_t_ratio=0.5,
        )
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        pair = _make_coating_pair(compression=0.1)
        # friction_forces で状態更新が必要
        u_cur = np.zeros(24)
        u_ref = np.zeros(24)
        u_cur[0] = 0.01
        proc.friction_forces([pair], coords, config, u_cur, u_ref)

        K = proc.friction_stiffness([pair], coords, config, ndof_total=24)
        assert K.shape == (24, 24)
        assert K.nnz > 0


# --- ファクトリ ---


class TestCreateCoatingStrategy:
    """create_coating_strategy ファクトリのテスト."""

    def test_zero_returns_no_coating(self):
        s = create_coating_strategy(coating_stiffness=0.0)
        assert isinstance(s, NoCoatingProcess)

    def test_positive_returns_kelvin_voigt(self):
        s = create_coating_strategy(coating_stiffness=1e6)
        assert isinstance(s, KelvinVoigtCoatingProcess)
