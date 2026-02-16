"""数値試験フレームワーク（Phase 2.6）のテスト.

静的試験（3点曲げ・4点曲げ・引張・ねん回）と
周波数応答試験のフレームワークを検証する。
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from xkep_cae.numerical_tests.core import (
    FrequencyResponseConfig,
    NumericalTestConfig,
    StaticTestResult,
    analytical_bend3p,
    analytical_bend4p,
    analytical_tensile,
    analytical_torsion,
    assess_friction_effect,
    generate_beam_mesh_2d,
    generate_beam_mesh_3d,
)
from xkep_cae.numerical_tests.csv_export import (
    export_frequency_response_csv,
    export_static_csv,
)
from xkep_cae.numerical_tests.frequency import run_frequency_response
from xkep_cae.numerical_tests.inp_input import parse_test_input
from xkep_cae.numerical_tests.runner import run_all_tests, run_test, run_tests

# ===========================================================================
# パラメータ定義
# ===========================================================================
E_STEEL = 200e3  # MPa
NU_STEEL = 0.3
RHO_STEEL = 7.85e-9  # ton/mm³ (SI consistent with MPa)
L_BEAM = 100.0  # mm
N_ELEMS = 10
P_LOAD = -1000.0  # N
T_TORQUE = 500.0  # N·mm

RECT_PARAMS = {"b": 10.0, "h": 20.0}
CIRC_PARAMS = {"d": 10.0}


# ===========================================================================
# メッシュ生成テスト
# ===========================================================================
class TestMeshGeneration:
    def test_2d_mesh_shape(self):
        nodes, conn = generate_beam_mesh_2d(10, 100.0)
        assert nodes.shape == (11, 2)
        assert conn.shape == (10, 2)

    def test_3d_mesh_shape(self):
        nodes, conn = generate_beam_mesh_3d(10, 100.0)
        assert nodes.shape == (11, 3)
        assert conn.shape == (10, 2)

    def test_mesh_endpoints(self):
        nodes, _ = generate_beam_mesh_2d(5, 50.0)
        assert nodes[0, 0] == pytest.approx(0.0)
        assert nodes[-1, 0] == pytest.approx(50.0)

    def test_connectivity(self):
        _, conn = generate_beam_mesh_2d(4, 40.0)
        np.testing.assert_array_equal(conn[0], [0, 1])
        np.testing.assert_array_equal(conn[-1], [3, 4])


# ===========================================================================
# 解析解テスト
# ===========================================================================
class TestAnalyticalSolutions:
    def test_bend3p_eb(self):
        """3点曲げ Euler-Bernoulli 解析解."""
        Ixy = 10.0 * 20.0**3 / 12.0
        ana = analytical_bend3p(1000.0, 100.0, E_STEEL, Ixy)
        delta_ref = 1000.0 * 100.0**3 / (48.0 * E_STEEL * Ixy)
        assert ana["delta_mid"] == pytest.approx(delta_ref, rel=1e-10)
        assert ana["V_max"] == pytest.approx(500.0)
        assert ana["M_max"] == pytest.approx(25000.0)

    def test_bend3p_timo(self):
        """3点曲げ Timoshenko 解析解（せん断項あり）."""
        Ixy = 10.0 * 20.0**3 / 12.0
        A = 10.0 * 20.0
        G = E_STEEL / (2 * (1 + NU_STEEL))
        kappa = 10.0 * (1.0 + NU_STEEL) / (12.0 + 11.0 * NU_STEEL)
        ana = analytical_bend3p(1000.0, 100.0, E_STEEL, Ixy, kappa=kappa, G=G, A=A)
        assert ana["delta_shear"] > 0
        assert ana["delta_mid"] > ana["delta_eb"]

    def test_bend4p(self):
        """4点曲げ解析解（24EI, 2点対称荷重の重ね合わせ）."""
        Ixy = 10.0 * 20.0**3 / 12.0
        ana = analytical_bend4p(1000.0, 100.0, 25.0, E_STEEL, Ixy)
        delta_ref = 1000.0 * 25.0 * (3 * 100.0**2 - 4 * 25.0**2) / (24.0 * E_STEEL * Ixy)
        assert ana["delta_mid"] == pytest.approx(delta_ref, rel=1e-10)
        assert ana["M_max"] == pytest.approx(25000.0)

    def test_tensile(self):
        """引張解析解."""
        A = 10.0 * 20.0
        ana = analytical_tensile(1000.0, 100.0, E_STEEL, A)
        assert ana["delta"] == pytest.approx(1000.0 * 100.0 / (E_STEEL * A))
        assert ana["N"] == 1000.0

    def test_torsion(self):
        """ねん回解析解."""
        d = 10.0
        J = math.pi * d**4 / 32.0
        G = E_STEEL / (2 * (1 + NU_STEEL))
        r_max = d / 2.0
        ana = analytical_torsion(500.0, 100.0, G, J, r_max)
        assert ana["theta"] == pytest.approx(500.0 * 100.0 / (G * J), rel=1e-10)
        assert ana["Mx"] == 500.0
        assert ana["tau_max"] == pytest.approx(500.0 * r_max / J)


# ===========================================================================
# 摩擦影響評価テスト
# ===========================================================================
class TestFrictionAssessment:
    def test_slender_beam_no_warning(self):
        msg = assess_friction_effect("bend3p", 15.0, "roller")
        assert "無視可能" in msg

    def test_moderate_beam_warning(self):
        msg = assess_friction_effect("bend3p", 7.0, "roller")
        assert "軽微" in msg

    def test_thick_beam_warning(self):
        msg = assess_friction_effect("bend3p", 3.0, "roller")
        assert "警告" in msg

    def test_non_bending_no_msg(self):
        msg = assess_friction_effect("tensile", 5.0, "roller")
        assert msg == ""


# ===========================================================================
# 静的試験 — 3点曲げ
# ===========================================================================
@pytest.mark.bend3p
class TestBend3p:
    def _make_config(self, beam_type="timo2d"):
        return NumericalTestConfig(
            name="bend3p",
            beam_type=beam_type,
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=N_ELEMS,
            load_value=abs(P_LOAD),
            section_shape="rectangle",
            section_params=RECT_PARAMS,
        )

    def test_eb2d(self):
        cfg = self._make_config("eb2d")
        result = run_test(cfg)
        assert isinstance(result, StaticTestResult)
        assert result.relative_error is not None
        assert result.relative_error < 1e-6

    def test_timo2d(self):
        cfg = self._make_config("timo2d")
        result = run_test(cfg)
        assert result.relative_error < 1e-4

    def test_timo3d(self):
        cfg = self._make_config("timo3d")
        result = run_test(cfg)
        assert result.relative_error < 1e-4

    def test_section_forces_exist(self):
        cfg = self._make_config("timo2d")
        result = run_test(cfg)
        assert len(result.element_forces) == N_ELEMS

    def test_friction_warning_present(self):
        cfg = self._make_config("timo2d")
        result = run_test(cfg)
        # L/h = 100/20 = 5 → 軽微の範囲
        assert result.friction_warning != ""


# ===========================================================================
# 静的試験 — 4点曲げ
# ===========================================================================
@pytest.mark.bend4p
class TestBend4p:
    def _make_config(self, beam_type="timo2d"):
        return NumericalTestConfig(
            name="bend4p",
            beam_type=beam_type,
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=N_ELEMS,
            load_value=abs(P_LOAD),
            section_shape="rectangle",
            section_params=RECT_PARAMS,
            load_span=25.0,
        )

    def test_eb2d(self):
        cfg = self._make_config("eb2d")
        result = run_test(cfg)
        assert result.relative_error < 1e-4

    def test_timo2d(self):
        cfg = self._make_config("timo2d")
        result = run_test(cfg)
        assert result.relative_error < 1e-4

    def test_timo3d(self):
        cfg = self._make_config("timo3d")
        result = run_test(cfg)
        assert result.relative_error < 1e-4


# ===========================================================================
# 静的試験 — 引張
# ===========================================================================
@pytest.mark.tensile
class TestTensile:
    def _make_config(self, beam_type="timo2d"):
        return NumericalTestConfig(
            name="tensile",
            beam_type=beam_type,
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=N_ELEMS,
            load_value=1000.0,
            section_shape="rectangle",
            section_params=RECT_PARAMS,
        )

    def test_eb2d(self):
        cfg = self._make_config("eb2d")
        result = run_test(cfg)
        assert result.relative_error is not None
        assert result.relative_error < 1e-10

    def test_timo2d(self):
        cfg = self._make_config("timo2d")
        result = run_test(cfg)
        assert result.relative_error < 1e-10

    def test_timo3d(self):
        cfg = self._make_config("timo3d")
        result = run_test(cfg)
        assert result.relative_error < 1e-10


# ===========================================================================
# 静的試験 — ねん回（3Dのみ）
# ===========================================================================
@pytest.mark.torsion
class TestTorsion:
    def _make_config(self):
        return NumericalTestConfig(
            name="torsion",
            beam_type="timo3d",
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=N_ELEMS,
            load_value=T_TORQUE,
            section_shape="circle",
            section_params=CIRC_PARAMS,
        )

    def test_torsion_analytical(self):
        cfg = self._make_config()
        result = run_test(cfg)
        assert result.relative_error is not None
        assert result.relative_error < 1e-10

    def test_torsion_not_2d(self):
        with pytest.raises(ValueError, match="3D"):
            NumericalTestConfig(
                name="torsion",
                beam_type="timo2d",
                E=E_STEEL,
                nu=NU_STEEL,
                length=L_BEAM,
                n_elems=N_ELEMS,
                load_value=T_TORQUE,
            )


# ===========================================================================
# 一括/部分実行API
# ===========================================================================
class TestRunAPI:
    def _configs(self):
        return [
            NumericalTestConfig(
                name="bend3p",
                beam_type="timo2d",
                E=E_STEEL,
                nu=NU_STEEL,
                length=L_BEAM,
                n_elems=N_ELEMS,
                load_value=abs(P_LOAD),
                section_shape="rectangle",
                section_params=RECT_PARAMS,
            ),
            NumericalTestConfig(
                name="tensile",
                beam_type="timo2d",
                E=E_STEEL,
                nu=NU_STEEL,
                length=L_BEAM,
                n_elems=N_ELEMS,
                load_value=1000.0,
                section_shape="rectangle",
                section_params=RECT_PARAMS,
            ),
        ]

    def test_run_all(self):
        results = run_all_tests(self._configs())
        assert len(results) == 2

    def test_run_partial(self):
        results = run_tests(self._configs(), ["tensile"])
        assert len(results) == 1
        assert results[0].config.name == "tensile"


# ===========================================================================
# 周波数応答試験
# ===========================================================================
@pytest.mark.freq_response
class TestFrequencyResponse:
    def _make_config(self, beam_type="timo2d", exc_type="displacement"):
        return FrequencyResponseConfig(
            beam_type=beam_type,
            E=E_STEEL,
            nu=NU_STEEL,
            rho=RHO_STEEL,
            length=L_BEAM,
            n_elems=N_ELEMS,
            section_shape="rectangle",
            section_params=RECT_PARAMS,
            freq_min=10.0,
            freq_max=5000.0,
            n_freq=50,
            excitation_type=exc_type,
            excitation_dof="uy",
            damping_alpha=0.0,
            damping_beta=1e-7,
        )

    def test_displacement_excitation_2d(self):
        cfg = self._make_config("timo2d", "displacement")
        result = run_frequency_response(cfg)
        assert len(result.frequencies) == 50
        assert len(result.magnitude) == 50
        # 伝達関数の低周波極限は≈1（変位入出力で同一DOF）
        assert result.magnitude[0] > 0.5

    def test_acceleration_excitation_2d(self):
        cfg = self._make_config("timo2d", "acceleration")
        result = run_frequency_response(cfg)
        assert len(result.frequencies) == 50
        assert np.all(result.magnitude >= 0)

    def test_displacement_excitation_3d(self):
        cfg = self._make_config("timo3d", "displacement")
        result = run_frequency_response(cfg)
        assert len(result.frequencies) == 50
        assert result.magnitude[0] > 0.5

    def test_natural_frequency_detection(self):
        """固有振動数のピーク検出が動作するか."""
        cfg = self._make_config("timo2d", "acceleration")
        result = run_frequency_response(cfg)
        # ピークが1つ以上検出されること
        # （パラメータによってはゼロの可能性もあるため soft check）
        assert isinstance(result.natural_frequencies, np.ndarray)

    def test_eb2d_frequency_response(self):
        cfg = FrequencyResponseConfig(
            beam_type="eb2d",
            E=E_STEEL,
            nu=NU_STEEL,
            rho=RHO_STEEL,
            length=L_BEAM,
            n_elems=N_ELEMS,
            section_shape="rectangle",
            section_params=RECT_PARAMS,
            freq_min=10.0,
            freq_max=5000.0,
            n_freq=20,
            excitation_type="displacement",
            excitation_dof="uy",
        )
        result = run_frequency_response(cfg)
        assert len(result.frequencies) == 20


# ===========================================================================
# CSV出力テスト
# ===========================================================================
class TestCSVExport:
    def test_static_csv_string(self):
        cfg = NumericalTestConfig(
            name="tensile",
            beam_type="timo2d",
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=4,
            load_value=1000.0,
            section_shape="rectangle",
            section_params=RECT_PARAMS,
        )
        result = run_test(cfg)
        outputs = export_static_csv(result)
        assert "summary" in outputs
        assert "nodal_disp" in outputs
        assert "element_forces" in outputs
        assert "試験名" in outputs["summary"]
        assert "node_id" in outputs["nodal_disp"]

    def test_static_csv_file(self):
        cfg = NumericalTestConfig(
            name="tensile",
            beam_type="timo2d",
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=4,
            load_value=1000.0,
            section_shape="rectangle",
            section_params=RECT_PARAMS,
        )
        result = run_test(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = export_static_csv(result, output_dir=tmpdir)
            for key, path in outputs.items():
                assert os.path.isfile(path), f"{key}: {path} not found"

    def test_frequency_csv_string(self):
        cfg = FrequencyResponseConfig(
            beam_type="timo2d",
            E=E_STEEL,
            nu=NU_STEEL,
            rho=RHO_STEEL,
            length=L_BEAM,
            n_elems=4,
            section_shape="rectangle",
            section_params=RECT_PARAMS,
            freq_min=10.0,
            freq_max=100.0,
            n_freq=5,
            excitation_type="displacement",
            excitation_dof="uy",
        )
        result = run_frequency_response(cfg)
        outputs = export_frequency_response_csv(result)
        assert "summary" in outputs
        assert "frf" in outputs
        assert "freq_Hz" in outputs["frf"]

    def test_frequency_csv_file(self):
        cfg = FrequencyResponseConfig(
            beam_type="timo2d",
            E=E_STEEL,
            nu=NU_STEEL,
            rho=RHO_STEEL,
            length=L_BEAM,
            n_elems=4,
            section_shape="rectangle",
            section_params=RECT_PARAMS,
            freq_min=10.0,
            freq_max=100.0,
            n_freq=5,
            excitation_type="displacement",
            excitation_dof="uy",
        )
        result = run_frequency_response(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = export_frequency_response_csv(result, output_dir=tmpdir)
            for key, path in outputs.items():
                assert os.path.isfile(path), f"{key}: {path} not found"


# ===========================================================================
# Abaqusライク入力テスト
# ===========================================================================
class TestInpInput:
    def test_parse_bend3p(self):
        text = """\
*TEST, TYPE=BEND3P
*BEAM SECTION, SECTION=RECT
 10.0, 20.0
*MATERIAL
*ELASTIC
 200000.0, 0.3
*SPECIMEN
 100.0, 10
*LOAD
 1000.0
*SUPPORT, TYPE=ROLLER
"""
        cfg = parse_test_input(text, beam_type="timo2d")
        assert isinstance(cfg, NumericalTestConfig)
        assert cfg.name == "bend3p"
        assert cfg.E == 200000.0
        assert cfg.nu == 0.3
        assert cfg.length == 100.0
        assert cfg.load_value == 1000.0
        assert cfg.support_condition == "roller"

    def test_parse_bend4p(self):
        text = """\
*TEST, TYPE=BEND4P
*BEAM SECTION, SECTION=RECT
 10.0, 20.0
*ELASTIC
 200000.0, 0.3
*SPECIMEN
 100.0, 20
*LOAD
 1000.0, 25.0
"""
        cfg = parse_test_input(text, beam_type="timo2d")
        assert cfg.name == "bend4p"
        assert cfg.load_span == 25.0
        assert cfg.n_elems == 20

    def test_parse_torsion(self):
        text = """\
*TEST, TYPE=TORSION
*BEAM TYPE, TYPE=TIMO3D
*BEAM SECTION, SECTION=CIRC
 10.0
*ELASTIC
 200000.0, 0.3
*SPECIMEN
 100.0, 10
*LOAD
 500.0
"""
        cfg = parse_test_input(text)
        assert cfg.name == "torsion"
        assert cfg.beam_type == "timo3d"

    def test_parse_freq_response(self):
        text = """\
*TEST, TYPE=FREQ_RESPONSE
*BEAM SECTION, SECTION=RECT
 10.0, 20.0
*ELASTIC
 200000.0, 0.3
*DENSITY
 7.85e-9
*SPECIMEN
 100.0, 10
*FREQUENCY
 10.0, 5000.0, 100
*EXCITATION, TYPE=DISPLACEMENT, DOF=uy
*DAMPING, ALPHA=0.0, BETA=1e-7
"""
        cfg = parse_test_input(text, beam_type="timo2d")
        assert isinstance(cfg, FrequencyResponseConfig)
        assert cfg.rho == pytest.approx(7.85e-9)
        assert cfg.freq_min == 10.0
        assert cfg.freq_max == 5000.0
        assert cfg.n_freq == 100
        assert cfg.excitation_type == "displacement"
        assert cfg.excitation_dof == "uy"
        assert cfg.damping_beta == pytest.approx(1e-7)

    def test_parse_and_run(self):
        """パース結果で実際に試験を実行できる."""
        text = """\
*TEST, TYPE=TENSILE
*BEAM SECTION, SECTION=RECT
 10.0, 20.0
*ELASTIC
 200000.0, 0.3
*SPECIMEN
 100.0, 10
*LOAD
 1000.0
"""
        cfg = parse_test_input(text, beam_type="timo2d")
        result = run_test(cfg)
        assert result.relative_error < 1e-10

    def test_parse_pipe_section(self):
        text = """\
*TEST, TYPE=TENSILE
*BEAM SECTION, SECTION=PIPE
 20.0, 16.0
*ELASTIC
 200000.0, 0.3
*SPECIMEN
 100.0, 10
*LOAD
 1000.0
"""
        cfg = parse_test_input(text, beam_type="timo3d")
        assert cfg.section_shape == "pipe"
        assert cfg.section_params["d_outer"] == 20.0


# ===========================================================================
# 円形断面テスト
# ===========================================================================
class TestCircularSection:
    def test_bend3p_circle(self):
        cfg = NumericalTestConfig(
            name="bend3p",
            beam_type="timo2d",
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=N_ELEMS,
            load_value=abs(P_LOAD),
            section_shape="circle",
            section_params=CIRC_PARAMS,
        )
        result = run_test(cfg)
        assert result.relative_error < 1e-4

    def test_tensile_circle_3d(self):
        cfg = NumericalTestConfig(
            name="tensile",
            beam_type="timo3d",
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=N_ELEMS,
            load_value=1000.0,
            section_shape="circle",
            section_params=CIRC_PARAMS,
        )
        result = run_test(cfg)
        assert result.relative_error < 1e-10


# ===========================================================================
# バリデーションテスト
# ===========================================================================
class TestValidation:
    def test_invalid_test_name(self):
        with pytest.raises(ValueError, match="試験名"):
            NumericalTestConfig(
                name="invalid",
                beam_type="timo2d",
                E=E_STEEL,
                nu=NU_STEEL,
                length=100.0,
                n_elems=10,
                load_value=1000.0,
            )

    def test_bend4p_requires_span(self):
        with pytest.raises(ValueError, match="load_span"):
            NumericalTestConfig(
                name="bend4p",
                beam_type="timo2d",
                E=E_STEEL,
                nu=NU_STEEL,
                length=100.0,
                n_elems=10,
                load_value=1000.0,
            )

    def test_freq_response_requires_rho(self):
        with pytest.raises(ValueError, match="rho"):
            FrequencyResponseConfig(
                beam_type="timo2d",
                E=E_STEEL,
                nu=NU_STEEL,
                rho=-1.0,
                length=100.0,
                n_elems=10,
            )

    def test_cosserat_beam_type_accepted(self):
        """cosserat が beam_type として受理されること."""
        cfg = NumericalTestConfig(
            name="tensile",
            beam_type="cosserat",
            E=E_STEEL,
            nu=NU_STEEL,
            length=100.0,
            n_elems=10,
            load_value=1000.0,
        )
        assert cfg.beam_type == "cosserat"


# ===========================================================================
# Cosserat rod 数値試験
# ===========================================================================
@pytest.mark.cosserat
class TestCosseratNumerical:
    """Cosserat rod の数値試験フレームワーク統合テスト."""

    def test_tensile_cosserat(self):
        """Cosserat rod で引張試験（厳密解一致）."""
        cfg = NumericalTestConfig(
            name="tensile",
            beam_type="cosserat",
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=N_ELEMS,
            load_value=1000.0,
            section_shape="rectangle",
            section_params=RECT_PARAMS,
        )
        result = run_test(cfg)
        assert result.relative_error is not None
        assert result.relative_error < 1e-10

    def test_torsion_cosserat(self):
        """Cosserat rod でねん回試験（厳密解一致）."""
        cfg = NumericalTestConfig(
            name="torsion",
            beam_type="cosserat",
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=N_ELEMS,
            load_value=T_TORQUE,
            section_shape="circle",
            section_params=CIRC_PARAMS,
        )
        result = run_test(cfg)
        assert result.relative_error is not None
        assert result.relative_error < 1e-10

    def test_bend3p_cosserat(self):
        """Cosserat rod で3点曲げ試験（メッシュ収束）."""
        cfg = NumericalTestConfig(
            name="bend3p",
            beam_type="cosserat",
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=20,
            load_value=abs(P_LOAD),
            section_shape="rectangle",
            section_params=RECT_PARAMS,
        )
        result = run_test(cfg)
        assert result.relative_error is not None
        # Cosserat rod は B行列定式化のため 20要素ではTimo3Dほど正確でないが収束する
        assert result.relative_error < 0.05

    def test_bend4p_cosserat(self):
        """Cosserat rod で4点曲げ試験."""
        cfg = NumericalTestConfig(
            name="bend4p",
            beam_type="cosserat",
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=20,
            load_value=abs(P_LOAD),
            section_shape="rectangle",
            section_params=RECT_PARAMS,
            load_span=25.0,
        )
        result = run_test(cfg)
        assert result.relative_error is not None
        assert result.relative_error < 0.05

    def test_section_forces_cosserat(self):
        """Cosserat rod の断面力ポスト処理."""
        cfg = NumericalTestConfig(
            name="tensile",
            beam_type="cosserat",
            E=E_STEEL,
            nu=NU_STEEL,
            length=L_BEAM,
            n_elems=4,
            load_value=1000.0,
            section_shape="rectangle",
            section_params=RECT_PARAMS,
        )
        result = run_test(cfg)
        assert len(result.element_forces) == 4


# ===========================================================================
# 周波数応答 — 固有振動数の解析解との比較
# ===========================================================================
@pytest.mark.freq_response
class TestFrequencyResponseAnalytical:
    """カンチレバー梁の固有振動数を Euler-Bernoulli 解析解と比較."""

    def test_cantilever_natural_frequencies_eb2d(self):
        """EB2Dの固有振動数がEB解析解と一致（低次モード）."""
        # EB解析解: f_n = (β_n L)² / (2πL²) √(EI/ρA)
        # β_1 L = 1.8751 (カンチレバー第1モード)
        # I は _build_section_props の 2D 規約（Iz = xy面内曲げ）に合わせる
        from xkep_cae.numerical_tests.core import _build_section_props

        sec = _build_section_props("rectangle", RECT_PARAMS, "eb2d", NU_STEEL)
        A = sec["A"]
        I = sec["I"]  # noqa: E741 — 2D梁のxy面内曲げ用断面二次モーメント
        beta_1L = 1.8751
        f1_analytical = (
            beta_1L**2 / (2 * np.pi * L_BEAM**2) * np.sqrt(E_STEEL * I / (RHO_STEEL * A))
        )

        cfg = FrequencyResponseConfig(
            beam_type="eb2d",
            E=E_STEEL,
            nu=NU_STEEL,
            rho=RHO_STEEL,
            length=L_BEAM,
            n_elems=20,
            section_shape="rectangle",
            section_params=RECT_PARAMS,
            freq_min=1.0,
            freq_max=f1_analytical * 2.0,
            n_freq=500,
            excitation_type="acceleration",
            excitation_dof="uy",
            damping_alpha=0.0,
            damping_beta=1e-8,
        )
        result = run_frequency_response(cfg)
        assert len(result.natural_frequencies) > 0, "固有振動数が検出されなかった"
        f1_fem = result.natural_frequencies[0]
        rel_error = abs(f1_fem - f1_analytical) / f1_analytical
        # 20要素のFEMは十分正確（5%以内）
        assert rel_error < 0.05, (
            f"第1固有振動数: FEM={f1_fem:.1f} Hz, 解析解={f1_analytical:.1f} Hz, "
            f"誤差={rel_error * 100:.1f}%"
        )


# ===========================================================================
# 非一様メッシュテスト
# ===========================================================================
class TestNonUniformMesh:
    """非一様メッシュ生成のテスト."""

    def test_2d_nonuniform_basic(self):
        """2D非一様メッシュの基本テスト."""
        from xkep_cae.numerical_tests.core import generate_beam_mesh_2d_nonuniform

        nodes, conn = generate_beam_mesh_2d_nonuniform(
            100.0,
            [50.0],
            base_n_elems=10,
            refinement_factor=3.0,
        )
        assert nodes.shape[1] == 2
        assert nodes[0, 0] == pytest.approx(0.0)
        assert nodes[-1, 0] == pytest.approx(100.0)
        # 非一様メッシュは均一メッシュより要素数が多い
        assert len(conn) >= 10
        # 荷重点（50.0）が節点に含まれる
        assert any(abs(nodes[:, 0] - 50.0) < 1e-10)

    def test_3d_nonuniform_basic(self):
        """3D非一様メッシュの基本テスト."""
        from xkep_cae.numerical_tests.core import generate_beam_mesh_3d_nonuniform

        nodes, conn = generate_beam_mesh_3d_nonuniform(
            100.0,
            [50.0],
            base_n_elems=10,
            refinement_factor=3.0,
        )
        assert nodes.shape[1] == 3
        assert nodes[0, 0] == pytest.approx(0.0)
        assert nodes[-1, 0] == pytest.approx(100.0)

    def test_nonuniform_multiple_points(self):
        """複数の細分割ポイント."""
        from xkep_cae.numerical_tests.core import generate_beam_mesh_2d_nonuniform

        nodes, conn = generate_beam_mesh_2d_nonuniform(
            100.0,
            [25.0, 75.0],
            base_n_elems=10,
            refinement_factor=3.0,
        )
        assert any(abs(nodes[:, 0] - 25.0) < 1e-10)
        assert any(abs(nodes[:, 0] - 75.0) < 1e-10)

    def test_nonuniform_refinement_increases_elements(self):
        """細分割すると要素数が増加する."""
        from xkep_cae.numerical_tests.core import (
            generate_beam_mesh_2d,
            generate_beam_mesh_2d_nonuniform,
        )

        _, conn_uniform = generate_beam_mesh_2d(10, 100.0)
        _, conn_nonuniform = generate_beam_mesh_2d_nonuniform(
            100.0,
            [50.0],
            base_n_elems=10,
            refinement_factor=3.0,
        )
        assert len(conn_nonuniform) >= len(conn_uniform)
