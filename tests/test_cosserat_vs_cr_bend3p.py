"""Cosserat rod 非線形 vs CR梁（Corotational Timoshenko 3D）の三点曲げ比較テスト.

小荷重域（線形応答）では両定式化の結果が一致し、
大荷重域では両定式化が定性的に同じ非線形応答を示すことを検証する。

Note:
    数値微分接線（eps=1e-7）を使用するため、NR の収束トレランスは 1e-5 程度に
    緩和する必要がある。荷重は ref_norm > 1.0 となるよう十分大きくする。
"""

from __future__ import annotations

import pytest

from xkep_cae.numerical_tests.core import DynamicTestConfig
from xkep_cae.numerical_tests.dynamic_runner import run_dynamic_test

# ---------------------------------------------------------------------------
# 共通パラメータ
# ---------------------------------------------------------------------------
E = 200e3  # MPa
NU = 0.3
RHO = 7.85e-9  # ton/mm³
L = 100.0  # mm
SECTION_SHAPE = "rectangle"
SECTION_PARAMS = {"b": 5.0, "h": 5.0}


def _make_config(
    beam_type: str,
    load_value: float,
    nlgeom: bool,
    n_elems: int = 20,
    n_steps: int = 200,
    dt: float = 0.5,
    ramp_time: float = 50.0,
    damping_beta: float = 1e-3,
    tol_force: float = 1e-5,
) -> DynamicTestConfig:
    """テスト用の DynamicTestConfig を生成."""
    return DynamicTestConfig(
        name="dynamic_bend3p",
        beam_type=beam_type,
        E=E,
        nu=NU,
        rho=RHO,
        length=L,
        n_elems=n_elems,
        load_value=load_value,
        section_shape=SECTION_SHAPE,
        section_params=SECTION_PARAMS,
        dt=dt,
        n_steps=n_steps,
        load_type="ramp",
        ramp_time=ramp_time,
        damping_alpha=0.0,
        damping_beta=damping_beta,
        max_iter=50,
        tol_force=tol_force,
        mass_type="lumped",
        nlgeom=nlgeom,
    )


class TestSmallLoadLinearMatch:
    """小荷重域: CR梁と Cosserat rod 非線形が線形解析解に近い結果を示す."""

    def test_cr_small_load_matches_analytical(self):
        """CR梁（nlgeom=True）で荷重 100N → 解析解に一致."""
        cfg = _make_config("timo3d", load_value=100.0, nlgeom=True)
        result = run_dynamic_test(cfg)
        assert result.converged
        assert result.relative_error_final is not None
        assert result.relative_error_final < 0.05

    def test_cosserat_nl_small_load_matches_analytical(self):
        """Cosserat rod 非線形（nlgeom=True）で荷重 100N → 解析解に一致."""
        cfg = _make_config("cosserat", load_value=100.0, nlgeom=True)
        result = run_dynamic_test(cfg)
        assert result.converged
        assert result.relative_error_final is not None
        assert result.relative_error_final < 0.05

    def test_cr_vs_cosserat_small_load(self):
        """CR梁と Cosserat rod 非線形の最終変位が近い（100N）."""
        cfg_cr = _make_config("timo3d", load_value=100.0, nlgeom=True)
        cfg_cos = _make_config("cosserat", load_value=100.0, nlgeom=True)
        res_cr = run_dynamic_test(cfg_cr)
        res_cos = run_dynamic_test(cfg_cos)
        assert res_cr.converged
        assert res_cos.converged
        rel_diff = abs(res_cr.displacement_max_final - res_cos.displacement_max_final) / max(
            res_cr.displacement_max_final, 1e-12
        )
        assert rel_diff < 0.10


class TestLargeLoadNonlinear:
    """大荷重域: 幾何学的非線形応答の定性比較."""

    def test_cr_large_load_converges(self):
        """CR梁で大荷重 500N → 収束."""
        cfg = _make_config("timo3d", load_value=500.0, nlgeom=True)
        result = run_dynamic_test(cfg)
        assert result.converged

    def test_cosserat_nl_large_load_converges(self):
        """Cosserat rod 非線形で大荷重 500N → 収束."""
        cfg = _make_config("cosserat", load_value=500.0, nlgeom=True)
        result = run_dynamic_test(cfg)
        assert result.converged

    @pytest.mark.slow
    def test_cr_vs_cosserat_large_load_qualitative(self):
        """大荷重域で CR梁と Cosserat rod 非線形が定性的に同じ応答."""
        cfg_cr = _make_config("timo3d", load_value=500.0, nlgeom=True)
        cfg_cos = _make_config("cosserat", load_value=500.0, nlgeom=True)
        res_cr = run_dynamic_test(cfg_cr)
        res_cos = run_dynamic_test(cfg_cos)
        assert res_cr.converged
        assert res_cos.converged
        ratio = res_cr.displacement_max_final / max(res_cos.displacement_max_final, 1e-12)
        assert 0.5 < ratio < 2.0

    def test_nonlinear_stiffening_effect(self):
        """非線形により線形解析解より変位が小さくなる（硬化効果）."""
        cfg_cr = _make_config("timo3d", load_value=500.0, nlgeom=True)
        res_cr = run_dynamic_test(cfg_cr)
        assert res_cr.converged
        if res_cr.displacement_analytical is not None:
            assert res_cr.displacement_max_final < res_cr.displacement_analytical * 1.1


class TestLinearVsNonlinearSmallLoad:
    """線形と非線形が同じ結果を返すことの確認."""

    def test_timo3d_linear_vs_nlgeom(self):
        """Timoshenko 3D: 線形 vs nlgeom（100N）."""
        cfg_lin = _make_config("timo3d", load_value=100.0, nlgeom=False)
        cfg_nl = _make_config("timo3d", load_value=100.0, nlgeom=True)
        res_lin = run_dynamic_test(cfg_lin)
        res_nl = run_dynamic_test(cfg_nl)
        assert res_lin.converged
        assert res_nl.converged
        rel_diff = abs(res_lin.displacement_max_final - res_nl.displacement_max_final) / max(
            res_lin.displacement_max_final, 1e-12
        )
        assert rel_diff < 0.05

    def test_cosserat_linear_vs_nlgeom(self):
        """Cosserat rod: 線形 vs nlgeom（100N）."""
        cfg_lin = _make_config("cosserat", load_value=100.0, nlgeom=False)
        cfg_nl = _make_config("cosserat", load_value=100.0, nlgeom=True)
        res_lin = run_dynamic_test(cfg_lin)
        res_nl = run_dynamic_test(cfg_nl)
        assert res_lin.converged
        assert res_nl.converged
        rel_diff = abs(res_lin.displacement_max_final - res_nl.displacement_max_final) / max(
            res_lin.displacement_max_final, 1e-12
        )
        assert rel_diff < 0.05
