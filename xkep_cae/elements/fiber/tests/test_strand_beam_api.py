"""StrandFiberBeamProcess API テスト.

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F4
完了判定: 弾性 EI 一致 < 0.1%、線形梁モードとの切替動作
[← README](../../../../README.md)
"""

from __future__ import annotations

import math

import numpy as np

from xkep_cae.core import binds_to
from xkep_cae.elements.fiber.materials import Elastic1D
from xkep_cae.elements.fiber.section import CircularFiberSection
from xkep_cae.elements.fiber.state import Fiber1DState, SectionState
from xkep_cae.elements.fiber.strand_beam import (
    StrandFiberBeamConfig,
    StrandFiberBeamProcess,
    StrandFiberBeamResult,
)


def _make_simple_beam(
    L: float = 10.0,
    d: float = 17.0,
    E: float = 200_000.0,
    G: float = 80_000.0,
    n_fiber: int = 80,
) -> tuple[StrandFiberBeamConfig, float, float, float, float]:
    """テスト用の単純梁構成を生成.

    Returns:
        (config_template, A, Iy, Iz, J)
        config_template は u_elem=0 の状態。
    """
    r = d / 2.0
    A = math.pi * r**2
    Iy = math.pi * r**4 / 4.0
    Iz = Iy
    J = Iy + Iz

    sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)
    mat = Elastic1D(E=E)
    state = SectionState(fibers=tuple(Fiber1DState() for _ in range(n_fiber)))

    kappa = 6.0 / 7.0  # 円形断面のせん断補正係数

    coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
    u_elem = np.zeros(12)

    cfg = StrandFiberBeamConfig(
        coords_init=coords,
        u_elem=u_elem,
        section=sec,
        material=mat,
        section_state=state,
        G=G,
        J=J,
        kappa_y=kappa,
        kappa_z=kappa,
    )
    return cfg, A, Iy, Iz, J


@binds_to(StrandFiberBeamProcess)
class TestStrandFiberBeamAPI:
    """StrandFiberBeamProcess の API テスト."""

    def test_returns_result_type(self) -> None:
        """process() が StrandFiberBeamResult を返す."""
        cfg, *_ = _make_simple_beam()
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)
        assert isinstance(result, StrandFiberBeamResult)

    def test_f_int_shape(self) -> None:
        """f_int は (12,) の配列."""
        cfg, *_ = _make_simple_beam()
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)
        assert result.f_int.shape == (12,)

    def test_k_elem_shape(self) -> None:
        """K_elem は (12, 12) の配列."""
        cfg, *_ = _make_simple_beam()
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)
        assert result.K_elem.shape == (12, 12)

    def test_k_elem_symmetric(self) -> None:
        """K_elem は対称行列."""
        cfg, *_ = _make_simple_beam()
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)
        np.testing.assert_allclose(result.K_elem, result.K_elem.T, atol=1e-6)

    def test_zero_displacement_zero_force(self) -> None:
        """変位ゼロなら内力もゼロ."""
        cfg, *_ = _make_simple_beam()
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)
        np.testing.assert_allclose(result.f_int, 0.0, atol=1e-10)

    def test_state_preserved_at_zero(self) -> None:
        """変位ゼロでは状態変化なし（弾性材料）."""
        cfg, *_ = _make_simple_beam()
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)
        assert len(result.section_state_new.fibers) == cfg.section.n_fiber

    def test_axial_force_sign(self) -> None:
        """引張変位で正の軸力."""
        cfg, *_ = _make_simple_beam()
        u = np.zeros(12)
        u[6] = 0.01  # ノード2を+x方向に引張
        cfg = StrandFiberBeamConfig(
            coords_init=cfg.coords_init,
            u_elem=u,
            section=cfg.section,
            material=cfg.material,
            section_state=cfg.section_state,
            G=cfg.G,
            J=cfg.J,
            kappa_y=cfg.kappa_y,
            kappa_z=cfg.kappa_z,
        )
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)
        # ノード2のx方向内力は正（引張方向の反力）
        assert result.f_int[6] > 0

    def test_uses_declaration(self) -> None:
        """uses に FiberSectionIntegratorProcess が宣言されている."""
        from xkep_cae.elements.fiber.integrator import FiberSectionIntegratorProcess

        assert FiberSectionIntegratorProcess in StrandFiberBeamProcess.uses
