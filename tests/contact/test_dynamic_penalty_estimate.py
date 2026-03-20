"""DynamicPenaltyEstimateProcess のユニットテスト.

動的解析用 k_pen の c0*M_ii ベース自動推定が正しいスケールを返すことを検証。

[← README](../../README.md)
"""

from __future__ import annotations

import math

import pytest

from xkep_cae.contact.penalty.strategy import (
    DynamicPenaltyEstimateInput,
    DynamicPenaltyEstimateOutput,
    DynamicPenaltyEstimateProcess,
)


class TestDynamicPenaltyEstimateAPI:
    """API・計算正確性のテスト."""

    def test_output_type(self):
        """出力が DynamicPenaltyEstimateOutput であること."""
        proc = DynamicPenaltyEstimateProcess()
        result = proc.process(
            DynamicPenaltyEstimateInput(rho_inf=0.9, dt=1e-4, rho=7.85e-9, A=math.pi, L_elem=5.0)
        )
        assert isinstance(result, DynamicPenaltyEstimateOutput)

    def test_c0_calculation(self):
        """c0 = 1/(beta*dt^2) が正しく計算されること."""
        rho_inf = 0.9
        dt = 1e-4
        alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
        alpha_f = rho_inf / (rho_inf + 1.0)
        beta = 0.25 * (1.0 - alpha_m + alpha_f) ** 2
        expected_c0 = 1.0 / (beta * dt**2)

        proc = DynamicPenaltyEstimateProcess()
        result = proc.process(
            DynamicPenaltyEstimateInput(rho_inf=rho_inf, dt=dt, rho=7.85e-9, A=math.pi, L_elem=5.0)
        )
        assert result.c0 == pytest.approx(expected_c0, rel=1e-10)

    def test_m_ii_calculation(self):
        """m_ii = rho*A*L_elem/2 が正しく計算されること."""
        rho = 7.85e-9
        A = math.pi
        L_elem = 5.0
        expected_m_ii = rho * A * L_elem / 2.0

        proc = DynamicPenaltyEstimateProcess()
        result = proc.process(
            DynamicPenaltyEstimateInput(rho_inf=0.9, dt=1e-4, rho=rho, A=A, L_elem=L_elem)
        )
        assert result.m_ii == pytest.approx(expected_m_ii, rel=1e-10)

    def test_k_pen_scale(self):
        """k_pen = scale * c0 * m_ii が正しくスケーリングされること."""
        proc = DynamicPenaltyEstimateProcess()
        result_50 = proc.process(
            DynamicPenaltyEstimateInput(
                rho_inf=0.9, dt=1e-4, rho=7.85e-9, A=math.pi, L_elem=5.0, scale=0.5
            )
        )
        result_100 = proc.process(
            DynamicPenaltyEstimateInput(
                rho_inf=0.9, dt=1e-4, rho=7.85e-9, A=math.pi, L_elem=5.0, scale=1.0
            )
        )
        assert result_100.k_pen == pytest.approx(2.0 * result_50.k_pen, rel=1e-10)

    def test_c0_m_ii_field(self):
        """c0_m_ii = c0 * m_ii であること."""
        proc = DynamicPenaltyEstimateProcess()
        result = proc.process(
            DynamicPenaltyEstimateInput(rho_inf=0.9, dt=1e-4, rho=7.85e-9, A=math.pi, L_elem=5.0)
        )
        assert result.c0_m_ii == pytest.approx(result.c0 * result.m_ii, rel=1e-10)

    def test_default_scale_is_half(self):
        """デフォルト scale=0.5 で k_pen = 0.5 * c0 * m_ii."""
        proc = DynamicPenaltyEstimateProcess()
        result = proc.process(
            DynamicPenaltyEstimateInput(rho_inf=0.9, dt=1e-4, rho=7.85e-9, A=math.pi, L_elem=5.0)
        )
        assert result.k_pen == pytest.approx(0.5 * result.c0_m_ii, rel=1e-10)


class TestDynamicPenaltyEstimatePhysics:
    """物理的妥当性テスト."""

    def test_k_pen_order_of_magnitude(self):
        """三点曲げ典型条件で k_pen が数十 N/mm オーダーになること.

        L=100mm, d=2mm, E=200GPa, rho=7.85e-9, dt=T1/40
        f1 = π/(2L²)√(EI/ρA) ≈ 396 Hz, T1 ≈ 2.52e-3 s
        c0*M_ii ≈ 56 N/mm → k_pen(50%) ≈ 28 N/mm
        """
        E = 200e3  # MPa
        d = 2.0  # mm
        Iy = math.pi * d**4 / 64
        A = math.pi * d**2 / 4
        L = 100.0  # mm
        rho = 7.85e-9  # ton/mm³

        # f1 = pi/(2L^2) * sqrt(EI/(rho*A))  -- simply supported beam 1st mode
        f1 = (math.pi / (2.0 * L**2)) * math.sqrt(E * Iy / (rho * A))
        T1 = 1.0 / f1
        dt = T1 / 40.0  # DynamicThreePointBendContactJigProcess のデフォルト

        proc = DynamicPenaltyEstimateProcess()
        result = proc.process(
            DynamicPenaltyEstimateInput(
                rho_inf=0.9,
                dt=dt,
                rho=rho,
                A=A,
                L_elem=L / 20,  # 20要素
                scale=0.5,
            )
        )

        # k_pen は 10-100 N/mm のオーダー（dt=T1/40 時）
        assert 10.0 < result.k_pen < 100.0, f"k_pen={result.k_pen:.1f} が期待範囲 [10, 100] 外"
        # c0*M_ii は 30-100 のオーダー
        assert 30.0 < result.c0_m_ii < 100.0, (
            f"c0_m_ii={result.c0_m_ii:.1f} が期待範囲 [30, 100] 外"
        )

    def test_k_pen_larger_than_static(self):
        """動的 k_pen が静的梁剛性 48EI/L³ より大きいこと."""
        E = 200e3
        d = 2.0
        Iy = math.pi * d**4 / 64
        A = math.pi * d**2 / 4
        L = 100.0
        rho = 7.85e-9

        k_static = 48.0 * E * Iy / L**3  # ≈ 7.54 N/mm

        f1 = (math.pi / (2.0 * L**2)) * math.sqrt(E * Iy / (rho * A))
        dt = 1.0 / f1 / 40.0

        proc = DynamicPenaltyEstimateProcess()
        result = proc.process(
            DynamicPenaltyEstimateInput(rho_inf=0.9, dt=dt, rho=rho, A=A, L_elem=L / 20, scale=0.5)
        )

        ratio = result.k_pen / k_static
        assert ratio > 2.0, f"動的/静的比 = {ratio:.1f} < 2: k_pen が静的剛性に近すぎる"

    def test_positive_definiteness_margin(self):
        """k_pen < (1-alpha_m)*c0*M_ii で K_eff 正定値を保つこと."""
        rho_inf = 0.9
        alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)

        proc = DynamicPenaltyEstimateProcess()
        result = proc.process(
            DynamicPenaltyEstimateInput(
                rho_inf=rho_inf,
                dt=1e-5,
                rho=7.85e-9,
                A=math.pi,
                L_elem=5.0,
                scale=0.5,
            )
        )

        # (1-alpha_m)*c0*m_ii > k_pen であること
        effective_mass_stiffness = (1.0 - alpha_m) * result.c0_m_ii
        assert result.k_pen < effective_mass_stiffness, (
            f"k_pen={result.k_pen:.1f} >= (1-alpha_m)*c0*m_ii={effective_mass_stiffness:.1f}: "
            "exact_tangent で K_eff が不定値になる"
        )
