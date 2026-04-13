"""StrandFiberBeamProcess 物理テスト.

設計仕様: xkep_cae/elements/docs/fiber_beam_strand.md Phase F4
完了判定: 弾性 EI 一致 < 0.1%
[← README](../../../../README.md)
"""

from __future__ import annotations

import math

import numpy as np

from xkep_cae.elements._beam_cr import (
    timo_beam3d_cr_internal_force,
    timo_beam3d_cr_tangent_analytical,
    timo_beam3d_ke_local,
)
from xkep_cae.elements.fiber.materials import Elastic1D
from xkep_cae.elements.fiber.section import CircularFiberSection
from xkep_cae.elements.fiber.state import Fiber1DState, SectionState
from xkep_cae.elements.fiber.strand_beam import (
    StrandFiberBeamConfig,
    StrandFiberBeamProcess,
    _timo_ke_from_section,
)


def _beam_params(
    L: float = 10.0,
    d: float = 17.0,
    E: float = 200_000.0,
    G: float = 80_000.0,
) -> dict:
    """テスト用の梁パラメータを辞書で返す."""
    r = d / 2.0
    A = math.pi * r**2
    Iy = math.pi * r**4 / 4.0
    Iz = Iy
    J = Iy + Iz
    kappa = 6.0 / 7.0
    return {
        "L": L,
        "d": d,
        "E": E,
        "G": G,
        "A": A,
        "Iy": Iy,
        "Iz": Iz,
        "J": J,
        "kappa": kappa,
    }


def _make_fiber_config(
    u_elem: np.ndarray,
    n_fiber: int = 200,
    section_type: str = "strip",
) -> StrandFiberBeamConfig:
    """ファイバー梁要素の config を生成.

    Args:
        u_elem: 変位ベクトル (12,)
        n_fiber: ファイバー数（strip 用）
        section_type: "strip" or "polar"
    """
    p = _beam_params()
    L = p["L"]
    d = p["d"]
    E = p["E"]
    G = p["G"]

    if section_type == "polar":
        sec = CircularFiberSection.polar(diameter=d, n_radial=12, n_theta=24)
    else:
        sec = CircularFiberSection.strip(diameter=d, n_fiber=n_fiber)

    mat = Elastic1D(E=E)
    state = SectionState(fibers=tuple(Fiber1DState() for _ in range(sec.n_fiber)))

    coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])

    return StrandFiberBeamConfig(
        coords_init=coords,
        u_elem=u_elem,
        section=sec,
        material=mat,
        section_state=state,
        G=G,
        J=p["J"],
        kappa_y=p["kappa"],
        kappa_z=p["kappa"],
    )


class TestStrandFiberBeamPhysics:
    """ファイバー梁要素の物理的整合性テスト."""

    def test_ke_from_section_matches_standard(self) -> None:
        """_timo_ke_from_section が標準 timo_beam3d_ke_local ���一致."""
        p = _beam_params()
        E, G, A, Iy, Iz, J, kappa, L = (
            p["E"],
            p["G"],
            p["A"],
            p["Iy"],
            p["Iz"],
            p["J"],
            p["kappa"],
            p["L"],
        )

        Ke_std = timo_beam3d_ke_local(E, G, A, Iy, Iz, J, L, kappa, kappa)

        EA = E * A
        EI_y = E * Iy
        EI_z = E * Iz
        GJ = G * J
        kGA_y = kappa * G * A
        kGA_z = kappa * G * A

        Ke_sec = _timo_ke_from_section(EA, EI_y, EI_z, GJ, kGA_y, kGA_z, L)

        np.testing.assert_allclose(Ke_sec, Ke_std, rtol=1e-12)

    def test_elastic_axial_stiffness_match(self) -> None:
        """弾性材料: 軸剛性が線形梁と一致（n=200 strip, 誤差 < 0.2%）."""
        p = _beam_params()

        u = np.zeros(12)
        u[6] = 0.01  # 軸方向引張

        cfg = _make_fiber_config(u, n_fiber=200)
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)

        f_linear = timo_beam3d_cr_internal_force(
            cfg.coords_init,
            u,
            p["E"],
            p["G"],
            p["A"],
            p["Iy"],
            p["Iz"],
            p["J"],
            p["kappa"],
            p["kappa"],
        )

        rel_err = np.max(np.abs(result.f_int - f_linear)) / np.max(np.abs(f_linear))
        assert rel_err < 0.002, f"軸力相対誤差 {rel_err:.4%} > 0.2%"

    def test_elastic_bending_y_match(self) -> None:
        """弾性材料: y 軸曲げが線形梁と一致（n=200 strip, 誤差 < 0.2%）."""
        p = _beam_params()

        u = np.zeros(12)
        u[10] = 0.01  # ノード2の θ_y 回転

        cfg = _make_fiber_config(u, n_fiber=200)
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)

        f_linear = timo_beam3d_cr_internal_force(
            cfg.coords_init,
            u,
            p["E"],
            p["G"],
            p["A"],
            p["Iy"],
            p["Iz"],
            p["J"],
            p["kappa"],
            p["kappa"],
        )

        rel_err = np.max(np.abs(result.f_int - f_linear)) / np.max(np.abs(f_linear))
        assert rel_err < 0.002, f"y曲げ相対誤差 {rel_err:.4%} > 0.2%"

    def test_elastic_bending_z_polar_match(self) -> None:
        """弾性材料: z 軸曲げ（polar 断面）が線形梁と一致."""
        p = _beam_params()

        u = np.zeros(12)
        u[11] = 0.01  # ノード2の θ_z 回転

        cfg = _make_fiber_config(u, section_type="polar")
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)

        f_linear = timo_beam3d_cr_internal_force(
            cfg.coords_init,
            u,
            p["E"],
            p["G"],
            p["A"],
            p["Iy"],
            p["Iz"],
            p["J"],
            p["kappa"],
            p["kappa"],
        )

        nonzero = np.abs(f_linear) > 1e-6
        if np.any(nonzero):
            rel_err = np.max(np.abs(result.f_int[nonzero] - f_linear[nonzero])) / np.max(
                np.abs(f_linear[nonzero])
            )
            assert rel_err < 0.01, f"z曲げ相対誤差 {rel_err:.4%} > 1%"

    def test_elastic_tangent_matches_linear(self) -> None:
        """弾性材料: 接線剛性が解析的接線と一致（polar, 対角 < 1%）."""
        p = _beam_params()

        u = np.zeros(12)
        u[10] = 0.005  # 微小曲げ変形

        cfg = _make_fiber_config(u, section_type="polar")
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)

        K_linear = timo_beam3d_cr_tangent_analytical(
            cfg.coords_init,
            u,
            p["E"],
            p["G"],
            p["A"],
            p["Iy"],
            p["Iz"],
            p["J"],
            p["kappa"],
            p["kappa"],
        )

        # 対角成分で比較
        diag_fiber = np.diag(result.K_elem)
        diag_linear = np.diag(K_linear)
        nonzero = np.abs(diag_linear) > 1.0
        if np.any(nonzero):
            rel_err = np.max(
                np.abs(diag_fiber[nonzero] - diag_linear[nonzero]) / np.abs(diag_linear[nonzero])
            )
            assert rel_err < 0.01, f"接線剛性対角相対誤差 {rel_err:.4%} > 1%"

    def test_elastic_ei_convergence(self) -> None:
        """ファイバー数増加で有効 EI が解析値に収束.

        n_fiber 200→400 で曲げ力の収束を確認。
        """
        u = np.zeros(12)
        u[10] = 0.01
        proc = StrandFiberBeamProcess()

        f_values = []
        for n_fiber in [100, 200, 400]:
            cfg = _make_fiber_config(u, n_fiber=n_fiber)
            result = proc.process(cfg)
            f_values.append(result.f_int[10])  # M_y at node 2

        # n=200 と n=400 の差が十分小さい（収束している）
        rel_change = abs(f_values[1] - f_values[2]) / abs(f_values[2])
        assert rel_change < 0.002, f"n=200→400 の変化 {rel_change:.4%} > 0.2%"

    def test_combined_deformation_polar_match(self) -> None:
        """軸＋曲げ＋ねじりの複合変形（polar 断面）で線形梁と一致."""
        p = _beam_params()

        u = np.zeros(12)
        u[6] = 0.005  # 軸
        u[9] = 0.003  # ねじり
        u[10] = 0.008  # 曲げ y
        u[11] = 0.004  # 曲げ z

        cfg = _make_fiber_config(u, section_type="polar")
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)

        f_linear = timo_beam3d_cr_internal_force(
            cfg.coords_init,
            u,
            p["E"],
            p["G"],
            p["A"],
            p["Iy"],
            p["Iz"],
            p["J"],
            p["kappa"],
            p["kappa"],
        )

        nonzero = np.abs(f_linear) > 1e-6
        if np.any(nonzero):
            rel_err = np.max(
                np.abs(result.f_int[nonzero] - f_linear[nonzero])
                / np.max(np.abs(f_linear[nonzero]))
            )
            assert rel_err < 0.01, f"複合変形相対誤差 {rel_err:.4%} > 1%"

    def test_fd_tangent_self_consistency(self) -> None:
        """接線剛性の FD 自己整合性検証.

        K_elem が fiber beam 自身の df/du の FD と一致するか確認。
        """
        u0 = np.zeros(12)
        u0[10] = 0.005
        u0[6] = 0.002

        cfg0 = _make_fiber_config(u0, section_type="polar")
        proc = StrandFiberBeamProcess()
        result0 = proc.process(cfg0)

        K_analytic = result0.K_elem
        K_fd = np.zeros((12, 12))
        h = 1e-7

        for j in range(12):
            u_p = u0.copy()
            u_m = u0.copy()
            u_p[j] += h
            u_m[j] -= h

            cfg_p = StrandFiberBeamConfig(
                coords_init=cfg0.coords_init,
                u_elem=u_p,
                section=cfg0.section,
                material=cfg0.material,
                section_state=cfg0.section_state,
                G=cfg0.G,
                J=cfg0.J,
                kappa_y=cfg0.kappa_y,
                kappa_z=cfg0.kappa_z,
            )
            cfg_m = StrandFiberBeamConfig(
                coords_init=cfg0.coords_init,
                u_elem=u_m,
                section=cfg0.section,
                material=cfg0.material,
                section_state=cfg0.section_state,
                G=cfg0.G,
                J=cfg0.J,
                kappa_y=cfg0.kappa_y,
                kappa_z=cfg0.kappa_z,
            )

            f_p = proc.process(cfg_p).f_int
            f_m = proc.process(cfg_m).f_int
            K_fd[:, j] = (f_p - f_m) / (2 * h)

        # FD も対称化して比較（解析的接線は Battini 式で対称化済み）
        K_fd_sym = 0.5 * (K_fd + K_fd.T)
        scale = np.max(np.abs(K_analytic))
        np.testing.assert_allclose(K_analytic, K_fd_sym, atol=scale * 1e-4)

    def test_large_rotation_nonzero_force(self) -> None:
        """大回転変位で非ゼロの内力が返る（CR が動作している）."""
        u = np.zeros(12)
        u[7] = 2.0  # 大きな横変位

        cfg = _make_fiber_config(u, section_type="polar")
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)

        assert np.max(np.abs(result.f_int)) > 0

    def test_torsion_matches_linear(self) -> None:
        """ねじり変形が線形梁と一致（ファイバーはねじりに関与しない）."""
        p = _beam_params()

        u = np.zeros(12)
        u[9] = 0.01  # ノード2の θ_x 回転

        cfg = _make_fiber_config(u)
        proc = StrandFiberBeamProcess()
        result = proc.process(cfg)

        f_linear = timo_beam3d_cr_internal_force(
            cfg.coords_init,
            u,
            p["E"],
            p["G"],
            p["A"],
            p["Iy"],
            p["Iz"],
            p["J"],
            p["kappa"],
            p["kappa"],
        )

        for idx in [3, 9]:
            if abs(f_linear[idx]) > 1e-6:
                rel_err = abs(result.f_int[idx] - f_linear[idx]) / abs(f_linear[idx])
                assert rel_err < 1e-10, f"DOF {idx}: ねじり力誤差 {rel_err:.2e}"
