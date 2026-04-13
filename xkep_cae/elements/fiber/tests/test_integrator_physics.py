"""FiberSectionIntegratorProcess 物理テスト.

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F3
完了判定: FD 接線 atol=1e-5
[← README](../../../../README.md)
"""

from __future__ import annotations

import math

import numpy as np

from xkep_cae.elements.fiber.integrator import (
    FiberIntegratorConfig,
    FiberSectionIntegratorProcess,
)
from xkep_cae.elements.fiber.materials import (
    BilinearKinematicHardening1D,
    Elastic1D,
    MultiLayerFrictionDegrading1D,
)
from xkep_cae.elements.fiber.section import CircularFiberSection
from xkep_cae.elements.fiber.state import Fiber1DState, SectionState


def _make_config(
    mat: object,
    state: SectionState,
    sec: CircularFiberSection,
    eps0: float = 0.0,
    kappa_y: float = 0.0,
    kappa_z: float = 0.0,
) -> FiberIntegratorConfig:
    return FiberIntegratorConfig(
        eps0=eps0,
        kappa_y=kappa_y,
        kappa_z=kappa_z,
        section=sec,
        material=mat,
        state=state,
    )


def _fd_tangent(
    proc: FiberSectionIntegratorProcess,
    mat: object,
    state: SectionState,
    sec: CircularFiberSection,
    base: np.ndarray,
    h: float = 1e-6,
) -> np.ndarray:
    """中央差分で接線行列を数値計算."""
    C_fd = np.zeros((3, 3))
    for j in range(3):
        perturb = np.zeros(3)
        perturb[j] = h
        p_plus = base + perturb
        p_minus = base - perturb

        cfg_p = _make_config(mat, state, sec, eps0=p_plus[0], kappa_y=p_plus[1], kappa_z=p_plus[2])
        cfg_m = _make_config(
            mat, state, sec, eps0=p_minus[0], kappa_y=p_minus[1], kappa_z=p_minus[2]
        )
        r_p = proc.process(cfg_p)
        r_m = proc.process(cfg_m)

        C_fd[0, j] = (r_p.N - r_m.N) / (2 * h)
        C_fd[1, j] = (r_p.M_y - r_m.M_y) / (2 * h)
        C_fd[2, j] = (r_p.M_z - r_m.M_z) / (2 * h)
    return C_fd


class TestFiberSectionConvergence:
    """ファイバー断面積分の物理的整合性テスト."""

    def test_elastic_ei_convergence(self) -> None:
        """弾性極限で EI が解析値 E*pi*d^4/64 と一致（誤差 < 1%）."""
        d = 17.0
        E = 200_000.0
        n_fiber = 60

        sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)
        mat = Elastic1D(E=E)
        state = SectionState(
            fibers=tuple(Fiber1DState() for _ in range(n_fiber)),
        )

        proc = FiberSectionIntegratorProcess()
        cfg = _make_config(mat, state, sec, kappa_y=0.001)
        result = proc.process(cfg)

        ei_computed = result.C_section[1, 1]
        ei_exact = E * math.pi * d**4 / 64.0
        rel_err = abs(ei_computed - ei_exact) / ei_exact
        assert rel_err < 0.01, f"EI 相対誤差 {rel_err:.4%} > 1%"

    def test_elastic_ea_convergence(self) -> None:
        """弾性極限で EA が解析値 E*pi*d^2/4 と一致（誤差 < 1%）."""
        d = 17.0
        E = 200_000.0
        n_fiber = 60

        sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)
        mat = Elastic1D(E=E)
        state = SectionState(
            fibers=tuple(Fiber1DState() for _ in range(n_fiber)),
        )

        proc = FiberSectionIntegratorProcess()
        cfg = _make_config(mat, state, sec, eps0=0.001)
        result = proc.process(cfg)

        ea_computed = result.C_section[0, 0]
        ea_exact = E * math.pi * d**2 / 4.0
        rel_err = abs(ea_computed - ea_exact) / ea_exact
        assert rel_err < 0.01, f"EA 相対誤差 {rel_err:.4%} > 1%"

    def test_elastic_coupling_zero(self) -> None:
        """弾性均質断面では軸-曲げカップリングがゼロ."""
        d = 17.0
        E = 200_000.0
        n_fiber = 60

        sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)
        mat = Elastic1D(E=E)
        state = SectionState(
            fibers=tuple(Fiber1DState() for _ in range(n_fiber)),
        )

        proc = FiberSectionIntegratorProcess()
        cfg = _make_config(mat, state, sec, kappa_y=0.001)
        result = proc.process(cfg)

        ea = result.C_section[0, 0]
        coupling = abs(result.C_section[0, 1])
        assert coupling / ea < 1e-10

    def test_n_fiber_convergence(self) -> None:
        """ファイバー数 20→40→60→80 で EI が収束."""
        d = 17.0
        E = 200_000.0
        proc = FiberSectionIntegratorProcess()

        ei_values = []
        for n in [20, 40, 60, 80]:
            sec = CircularFiberSection.strip(diameter=d, n_fiber=n)
            mat = Elastic1D(E=E)
            state = SectionState(
                fibers=tuple(Fiber1DState() for _ in range(n)),
            )
            cfg = _make_config(mat, state, sec, kappa_y=0.001)
            result = proc.process(cfg)
            ei_values.append(result.C_section[1, 1])

        rel_change = abs(ei_values[2] - ei_values[3]) / ei_values[3]
        assert rel_change < 0.05, f"EI 相対変化 {rel_change:.4%} > 5%"

    def test_tangent_fd_elastic(self) -> None:
        """弾性材料での接線行列 FD 検証.

        対称断面・弾性ではオフダイアゴナルが数値的にゼロ。
        対角成分の相対誤差で接線整合性を検証する。
        """
        d = 17.0
        E = 200_000.0
        n_fiber = 60
        proc = FiberSectionIntegratorProcess()

        sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)
        mat = Elastic1D(E=E)
        state = SectionState(
            fibers=tuple(Fiber1DState() for _ in range(n_fiber)),
        )

        base = np.array([0.001, 0.002, 0.0])

        cfg = _make_config(mat, state, sec, eps0=base[0], kappa_y=base[1], kappa_z=base[2])
        result = proc.process(cfg)
        C_analytic = result.C_section

        C_fd = _fd_tangent(proc, mat, state, sec, base, h=1e-6)

        # 対角成分: EA, EI は大きな値 → rtol で検証
        scale = np.max(np.abs(C_analytic))
        np.testing.assert_allclose(C_analytic, C_fd, atol=scale * 1e-5)

    def test_tangent_fd_bilinear_kh(self) -> None:
        """BilinearKH 材料（部分塑性域）での接線行列 FD 検証.

        塑性化後に弾性域内の微小変動で接線を検証する。
        """
        d = 17.0
        n_fiber = 60
        proc = FiberSectionIntegratorProcess()

        sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)
        mat = BilinearKinematicHardening1D(E=200_000.0, sigma_y=300.0, H=2000.0)
        state = SectionState(
            fibers=tuple(Fiber1DState() for _ in range(n_fiber)),
        )

        # 塑性化させる
        cfg_pre = _make_config(mat, state, sec, kappa_y=0.01)
        result_pre = proc.process(cfg_pre)
        state_after = result_pre.state_new

        # 弾性域内で FD 検証（摂動が新たな降伏を起こさない範囲）
        base = np.array([0.0005, 0.005, 0.0])

        cfg = _make_config(mat, state_after, sec, eps0=base[0], kappa_y=base[1], kappa_z=base[2])
        result = proc.process(cfg)
        C_analytic = result.C_section

        C_fd = _fd_tangent(proc, mat, state_after, sec, base, h=1e-6)

        scale = np.max(np.abs(C_analytic))
        np.testing.assert_allclose(C_analytic, C_fd, atol=scale * 1e-5)

    def test_tangent_fd_multilayer_friction(self) -> None:
        """MultiLayerFriction（部分スリップ域）での接線行列 FD 検証."""
        d = 17.0
        n_fiber = 40
        proc = FiberSectionIntegratorProcess()

        sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)

        N_layers = 5
        E = 200_000.0
        H = 2000.0
        sigma_y = 300.0
        E_base = E * H / (E + H)
        k_total = E**2 / (E + H)
        f_y_arr = np.linspace(0.5, 1.5, N_layers) * sigma_y * E / (E + H)
        k_arr = np.full(N_layers, k_total / N_layers)

        mat = MultiLayerFrictionDegrading1D(
            E_base=E_base,
            k_virgin=k_arr,
            k_degraded=k_arr * 0.5,
            f_y=f_y_arr,
        )

        fiber_states = tuple(mat.initial_state() for _ in range(n_fiber))
        state = SectionState(fibers=fiber_states)

        # スリップ発生
        cfg_pre = _make_config(mat, state, sec, kappa_y=0.005)
        result_pre = proc.process(cfg_pre)
        state_after = result_pre.state_new

        # 弾性域内の微小変動で FD 検証
        base = np.array([0.0001, 0.003, 0.0])

        cfg = _make_config(mat, state_after, sec, eps0=base[0], kappa_y=base[1], kappa_z=base[2])
        result = proc.process(cfg)
        C_analytic = result.C_section

        C_fd = _fd_tangent(proc, mat, state_after, sec, base, h=1e-6)

        scale = np.max(np.abs(C_analytic))
        np.testing.assert_allclose(C_analytic, C_fd, atol=scale * 1e-5)

    def test_pure_bending_moment_sign(self) -> None:
        """正の曲率 kappa_y > 0 で正のモーメント M_y > 0."""
        d = 17.0
        E = 200_000.0
        n_fiber = 60

        sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)
        mat = Elastic1D(E=E)
        state = SectionState(
            fibers=tuple(Fiber1DState() for _ in range(n_fiber)),
        )

        proc = FiberSectionIntegratorProcess()
        cfg = _make_config(mat, state, sec, kappa_y=0.001)
        result = proc.process(cfg)

        assert result.M_y > 0
        assert abs(result.N) < 1e-6

    def test_plastic_coupling_nonzero(self) -> None:
        """部分降伏時は軸-曲げカップリングが非ゼロ.

        eps0 で中立軸をシフトし、片側のみ降伏させることで
        非対称な接線分布を生成する。
        """
        d = 17.0
        n_fiber = 60
        E = 200_000.0
        sigma_y = 300.0
        H = 2000.0

        sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)
        mat = BilinearKinematicHardening1D(E=E, sigma_y=sigma_y, H=H)
        state = SectionState(
            fibers=tuple(Fiber1DState() for _ in range(n_fiber)),
        )

        proc = FiberSectionIntegratorProcess()

        # eps_y = sigma_y / E = 0.0015
        # eps_i = eps0 - kappa * y_i
        # eps0 = 0.001, kappa = 0.00025:
        #   y = +8.5: eps = 0.001 - 0.00213 = -0.00113 (弾性)
        #   y = -8.5: eps = 0.001 + 0.00213 = +0.00313 (塑性)
        # → 片側のみ降伏 → 非対称接線
        cfg = _make_config(mat, state, sec, eps0=0.001, kappa_y=0.00025)
        result = proc.process(cfg)

        coupling = abs(result.C_section[0, 1])
        assert coupling > 1.0, f"カップリング {coupling:.2e} が小さすぎる"

    def test_moment_matches_prototype(self) -> None:
        """05_smooth_teardrop.py の FiberSection.moment() と一致."""
        d = 17.0
        n_fiber = 60
        E = 200_000.0
        kappa = 0.001

        sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)
        mat = Elastic1D(E=E)
        state = SectionState(
            fibers=tuple(Fiber1DState() for _ in range(n_fiber)),
        )

        proc = FiberSectionIntegratorProcess()
        cfg = _make_config(mat, state, sec, kappa_y=kappa)
        result = proc.process(cfg)

        M_proto = sum(-mat.E * (-kappa * sec.y[i]) * sec.y[i] * sec.area[i] for i in range(n_fiber))

        assert abs(result.M_y - M_proto) / abs(M_proto) < 1e-10
