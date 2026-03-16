"""inp_parser ラウンドトリップテスト.

write_abaqus_model → read_abaqus_inp → 全フィールド検証。
新パーサー（AbstractKeywordParser フレームワーク）の網羅的テスト。

[← README](../README.md)
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest
from xkep_cae.io.abaqus_inp import (
    AbaqusMesh,
    InpContactDef,
    InpInitialCondition,
    InpStep,
    InpSurfaceDef,
    InpSurfaceInteraction,
    read_abaqus_inp,
    write_abaqus_inp,
    write_abaqus_model,
)
from xkep_cae.io.inp_parser import (
    AbstractKeywordParser,
)

# ====================================================================
# フレームワーク基盤テスト
# ====================================================================


class TestParserFramework:
    """AbstractKeywordParser フレームワークのテスト."""

    def test_registry_contains_all_keywords(self):
        """全主要キーワードが登録されていること."""
        registry = AbstractKeywordParser._registry
        required = [
            "*NODE",
            "*ELEMENT",
            "*NSET",
            "*ELSET",
            "*BOUNDARY",
            "*MATERIAL",
            "*ELASTIC",
            "*DENSITY",
            "*PLASTIC",
            "*BEAM SECTION",
            "*TRANSVERSE SHEAR STIFFNESS",
            "*STEP",
            "*END STEP",
            "*STATIC",
            "*DYNAMIC",
            "*SURFACE",
            "*SURFACE INTERACTION",
            "*SURFACE BEHAVIOR",
            "*FRICTION",
            "*CONTACT",
            "*CONTACT INCLUSIONS",
            "*CONTACT PROPERTY ASSIGNMENT",
            "*INITIAL CONDITIONS",
            "*CLOAD",
            "*DLOAD",
            "*OUTPUT",
            "*NODE OUTPUT",
            "*ELEMENT OUTPUT",
            "*ENERGY OUTPUT",
            "*HEADING",
            "*ANIMATION",
        ]
        for kw in required:
            assert kw in registry, f"{kw} がレジストリに未登録"

    def test_aliases_registered(self):
        """スペース無し別名が登録されていること."""
        registry = AbstractKeywordParser._registry
        aliases = [
            "*BEAMSECTION",
            "*TRANSVERSESHEARSTIFFNESS",
            "*ENDSTEP",
            "*SURFACEINTERACTION",
            "*SURFACEBEHAVIOR",
            "*CONTACTINCLUSIONS",
            "*CONTACTPROPERTYASSIGNMENT",
            "*INITIALCONDITIONS",
            "*NODEOUTPUT",
            "*ELEMENTOUTPUT",
            "*ENERGYOUTPUT",
        ]
        for alias in aliases:
            assert alias in registry, f"別名 {alias} がレジストリに未登録"

    def test_get_parser_returns_instance(self):
        """get_parser がインスタンスを返すこと."""
        parser = AbstractKeywordParser.get_parser("*NODE")
        assert parser is not None
        assert hasattr(parser, "parse")

    def test_get_parser_unknown_returns_none(self):
        """未知のキーワードに対して None を返すこと."""
        assert AbstractKeywordParser.get_parser("*UNKNOWN_KEYWORD") is None


# ====================================================================
# *INCLUDE テスト
# ====================================================================


class TestInclude:
    """*INCLUDE 再帰読み込みテスト."""

    def test_include_basic(self, tmp_path):
        """*INCLUDE でノード定義を分離ファイルから読み込む."""
        # 分離ファイル: ノード定義
        nodes_inp = tmp_path / "nodes.inp"
        nodes_inp.write_text("*NODE\n1, 0.0, 0.0\n2, 1.0, 0.0\n")

        # メインファイル
        main_inp = tmp_path / "main.inp"
        main_inp.write_text(
            "*HEADING\nTest include\n*INCLUDE, INPUT=nodes.inp\n*ELEMENT, TYPE=B21\n1, 1, 2\n"
        )

        mesh = read_abaqus_inp(main_inp)
        assert len(mesh.nodes) == 2
        assert len(mesh.element_groups) == 1
        assert mesh.heading == "Test include"

    def test_include_nested(self, tmp_path):
        """ネストした *INCLUDE."""
        mat_inp = tmp_path / "material.inp"
        mat_inp.write_text("*MATERIAL, NAME=STEEL\n*ELASTIC\n200e9, 0.3\n")

        sub_inp = tmp_path / "sub.inp"
        sub_inp.write_text("*INCLUDE, INPUT=material.inp\n*NODE\n1, 0.0, 0.0\n")

        main_inp = tmp_path / "main.inp"
        main_inp.write_text("*INCLUDE, INPUT=sub.inp\n")

        mesh = read_abaqus_inp(main_inp)
        assert len(mesh.materials) == 1
        assert mesh.materials[0].elastic == (200e9, 0.3)
        assert len(mesh.nodes) == 1

    def test_include_missing_file_warning(self, tmp_path):
        """存在しないファイルの *INCLUDE は警告のみでエラーにならない."""
        main_inp = tmp_path / "main.inp"
        main_inp.write_text("*INCLUDE, INPUT=nonexistent.inp\n*NODE\n1, 0.0, 0.0\n")
        mesh = read_abaqus_inp(main_inp)
        assert len(mesh.nodes) == 1  # INCLUDE は無視され、NODE は読める


# ====================================================================
# ステップ・プロシージャテスト
# ====================================================================


class TestStepParsing:
    """*STEP 関連のパーステスト."""

    def _write_and_read(self, tmp_path, content):
        p = tmp_path / "test.inp"
        p.write_text(textwrap.dedent(content))
        return read_abaqus_inp(p)

    def test_static_step(self, tmp_path):
        """*STEP + *STATIC のパース."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            *STEP, INC=500, NLGEOM=YES
            Step-Bending
            *STATIC
            0.1, 1.0, 1e-6, 0.5
            *END STEP
            """,
        )
        assert len(mesh.steps) == 1
        step = mesh.steps[0]
        assert step.name == "Step-Bending"
        assert step.procedure == "STATIC"
        assert step.inc == 500
        assert step.nlgeom is True
        assert step.time_params == (0.1, 1.0, 1e-6, 0.5)

    def test_dynamic_step(self, tmp_path):
        """*STEP + *DYNAMIC のパース."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *NODE
            1, 0.0, 0.0
            *STEP, INC=1000
            *DYNAMIC
            0.001, 1.0, 1e-8, 0.01
            *END STEP
            """,
        )
        assert len(mesh.steps) == 1
        step = mesh.steps[0]
        assert step.procedure == "DYNAMIC"
        assert step.time_params == (0.001, 1.0, 1e-8, 0.01)

    def test_multiple_steps(self, tmp_path):
        """複数ステップのパース."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *NODE
            1, 0.0, 0.0
            *STEP
            Bending
            *STATIC
            0.1, 1.0
            *END STEP
            *STEP
            Oscillation
            *DYNAMIC
            0.001, 1.0
            *END STEP
            """,
        )
        assert len(mesh.steps) == 2
        assert mesh.steps[0].name == "Bending"
        assert mesh.steps[0].procedure == "STATIC"
        assert mesh.steps[1].name == "Oscillation"
        assert mesh.steps[1].procedure == "DYNAMIC"

    def test_step_boundary(self, tmp_path):
        """ステップ内の *BOUNDARY パース."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *NODE
            1, 0.0, 0.0
            *STEP
            *STATIC
            1.0, 1.0
            *BOUNDARY
            1, 1, 3, 0.0
            1, 4, 4, 0.5
            *END STEP
            """,
        )
        step = mesh.steps[0]
        assert len(step.boundaries) == 2
        assert step.boundaries[0] == (1, 1, 3, 0.0)
        assert step.boundaries[1] == (1, 4, 4, 0.5)

    def test_step_cload(self, tmp_path):
        """ステップ内の *CLOAD パース."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *NODE
            1, 0.0, 0.0
            *STEP
            *STATIC
            1.0, 1.0
            *CLOAD
            1, 2, 1000.0
            *END STEP
            """,
        )
        step = mesh.steps[0]
        assert len(step.cloads) == 1
        assert step.cloads[0] == (1, 2, 1000.0)

    def test_step_dload(self, tmp_path):
        """ステップ内の *DLOAD パース."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *NODE
            1, 0.0, 0.0
            *STEP
            *STATIC
            1.0, 1.0
            *DLOAD
            ALL_ELEMS, GRAV, 9.81
            *END STEP
            """,
        )
        step = mesh.steps[0]
        assert len(step.dloads) == 1
        assert step.dloads[0] == ("ALL_ELEMS", "GRAV", 9.81)


# ====================================================================
# サーフェス・接触テスト
# ====================================================================


class TestContactParsing:
    """サーフェス・接触定義のパーステスト."""

    def _write_and_read(self, tmp_path, content):
        p = tmp_path / "test.inp"
        p.write_text(textwrap.dedent(content))
        return read_abaqus_inp(p)

    def test_surface_element(self, tmp_path):
        """*SURFACE, TYPE=ELEMENT のパース."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *SURFACE, TYPE=ELEMENT, NAME=SURF1
            STRAND_0,
            """,
        )
        assert len(mesh.surfaces) == 1
        surf = mesh.surfaces[0]
        assert surf.name == "SURF1"
        assert surf.type == "ELEMENT"
        assert surf.elset == "STRAND_0"

    def test_surface_interaction(self, tmp_path):
        """*SURFACE INTERACTION + *SURFACE BEHAVIOR + *FRICTION."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *SURFACE INTERACTION, NAME=CONTACT_PROP
            *SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR
            1e6, 0.
            *FRICTION
            0.3
            """,
        )
        assert len(mesh.surface_interactions) == 1
        si = mesh.surface_interactions[0]
        assert si.name == "CONTACT_PROP"
        assert si.pressure_overclosure == "LINEAR"
        assert si.k_pen == 1e6
        assert si.friction == 0.3

    def test_surface_behavior_hard(self, tmp_path):
        """*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD（データ行なし）."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *SURFACE INTERACTION, NAME=HARD_CONTACT
            *SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD
            """,
        )
        si = mesh.surface_interactions[0]
        assert si.pressure_overclosure == "HARD"
        assert si.k_pen == 0.0

    def test_general_contact(self, tmp_path):
        """General Contact (model level) のパース."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *SURFACE, TYPE=ELEMENT, NAME=ALLSURF
            ALL_ELEMS,
            *SURFACE INTERACTION, NAME=CPROP
            *SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD
            *CONTACT
            *CONTACT INCLUSIONS
            ALLEXT, ALLEXT
            *CONTACT PROPERTY ASSIGNMENT
            , , CPROP
            """,
        )
        assert len(mesh.contact_defs) == 1
        cdef = mesh.contact_defs[0]
        assert cdef.inclusions == [("ALLEXT", "ALLEXT")]
        assert cdef.interaction == "CPROP"

    def test_xkep_comment_exclude_same_layer(self, tmp_path):
        """** XKEP-CAE: EXCLUDE_SAME_LAYER=YES コメントのパース."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *SURFACE INTERACTION, NAME=CPROP
            *SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD
            ** XKEP-CAE: ALGORITHM=NCP, MORTAR=YES
            *CONTACT
            *CONTACT INCLUSIONS
            ALLEXT, ALLEXT
            *CONTACT PROPERTY ASSIGNMENT
            , , CPROP
            ** XKEP-CAE: EXCLUDE_SAME_LAYER=YES
            """,
        )
        si = mesh.surface_interactions[0]
        assert si.algorithm == "NCP"
        assert si.mortar is True
        cdef = mesh.contact_defs[0]
        assert cdef.exclude_same_layer is True


# ====================================================================
# 初期条件テスト
# ====================================================================


class TestInitialConditions:
    """*INITIAL CONDITIONS のパーステスト."""

    def test_velocity_ic(self, tmp_path):
        p = tmp_path / "test.inp"
        p.write_text(
            textwrap.dedent(
                """\
                *INITIAL CONDITIONS, TYPE=VELOCITY
                1, 1, 10.0
                1, 2, 0.0
                2, 1, 10.0
                """
            )
        )
        mesh = read_abaqus_inp(p)
        assert len(mesh.initial_conditions) == 1
        ic = mesh.initial_conditions[0]
        assert ic.type == "VELOCITY"
        assert len(ic.data) == 3
        assert ic.data[0] == (1, 1, 10.0)

    def test_temperature_ic(self, tmp_path):
        p = tmp_path / "test.inp"
        p.write_text(
            textwrap.dedent(
                """\
                *INITIAL CONDITIONS, TYPE=TEMPERATURE
                ALL_NODES, 1, 293.15
                """
            )
        )
        mesh = read_abaqus_inp(p)
        ic = mesh.initial_conditions[0]
        assert ic.type == "TEMPERATURE"
        assert ic.data[0] == ("ALL_NODES", 1, 293.15)


# ====================================================================
# 出力要求テスト
# ====================================================================


class TestOutputParsing:
    """*OUTPUT / *NODE OUTPUT 等のパーステスト."""

    def _write_and_read(self, tmp_path, content):
        p = tmp_path / "test.inp"
        p.write_text(textwrap.dedent(content))
        return read_abaqus_inp(p)

    def test_output_field(self, tmp_path):
        """*OUTPUT, FIELD + *NODE OUTPUT."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *NODE
            1, 0.0, 0.0
            *STEP
            *STATIC
            1.0, 1.0
            *OUTPUT, FIELD, FREQUENCY=5
            *NODE OUTPUT
            U, RF, COORD
            *END STEP
            """,
        )
        step = mesh.steps[0]
        assert len(step.output_requests) >= 1
        # FIELD 出力要求に変数が設定されていること
        field_reqs = [r for r in step.output_requests if r.domain == "FIELD"]
        assert len(field_reqs) >= 1
        assert "U" in field_reqs[-1].variables

    def test_element_output(self, tmp_path):
        """*ELEMENT OUTPUT."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *NODE
            1, 0.0, 0.0
            *STEP
            *STATIC
            1.0, 1.0
            *OUTPUT, FIELD
            *ELEMENT OUTPUT
            S, E, COORD
            *END STEP
            """,
        )
        step = mesh.steps[0]
        field_reqs = [r for r in step.output_requests if r.domain == "FIELD"]
        assert any("S" in r.variables for r in field_reqs)

    def test_animation_in_step(self, tmp_path):
        """ステップ内 *ANIMATION のパース."""
        mesh = self._write_and_read(
            tmp_path,
            """\
            *NODE
            1, 0.0, 0.0
            *STEP
            *STATIC
            1.0, 1.0
            *ANIMATION, DIR=output_anim, FREQUENCY=10
            yz, xz
            *END STEP
            """,
        )
        step = mesh.steps[0]
        assert step.animation is not None
        assert step.animation.output_dir == "output_anim"
        assert step.animation.frequency == 10
        assert step.animation.views == ["yz", "xz"]


# ====================================================================
# ラウンドトリップテスト（write → read）
# ====================================================================


class TestRoundTrip:
    """write_abaqus_model → read_abaqus_inp のラウンドトリップ検証."""

    def test_basic_beam_model(self, tmp_path):
        """基本的な梁モデルのラウンドトリップ."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        conn = np.array([[0, 1], [1, 2]], dtype=int)

        out_path = tmp_path / "beam.inp"
        write_abaqus_model(
            out_path,
            coords,
            conn,
            elem_type="B31",
            material_name="STEEL",
            E=200e9,
            nu=0.3,
            density=7850.0,
            beam_section_type="CIRC",
            beam_section_dims=[0.001],
            beam_section_direction=[0, 1, 0],
        )

        mesh = read_abaqus_inp(out_path)

        # ノード
        assert len(mesh.nodes) == 3
        assert mesh.nodes[0].x == pytest.approx(0.0)
        assert mesh.nodes[2].x == pytest.approx(2.0)

        # 要素
        assert len(mesh.element_groups) == 1
        assert mesh.element_groups[0].elem_type == "B31"
        assert len(mesh.element_groups[0].elements) == 2

        # 材料
        assert len(mesh.materials) == 1
        mat = mesh.materials[0]
        assert mat.name == "STEEL"
        assert mat.elastic == pytest.approx((200e9, 0.3))
        assert mat.density == pytest.approx(7850.0)

        # 梁断面
        assert len(mesh.beam_sections) == 1
        bsec = mesh.beam_sections[0]
        assert bsec.section_type == "CIRC"
        assert bsec.dimensions == pytest.approx([0.001])
        assert bsec.direction == pytest.approx([0, 1, 0])

    def test_model_with_steps(self, tmp_path):
        """ステップ付きモデルのラウンドトリップ."""
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        conn = np.array([[0, 1]], dtype=int)

        step1 = InpStep(
            name="Bending",
            procedure="STATIC",
            nlgeom=True,
            inc=500,
            time_params=(0.1, 1.0, 1e-6, 0.5),
            boundaries=[(1, 1, 6, 0.0)],
        )
        step2 = InpStep(
            name="Oscillation",
            procedure="DYNAMIC",
            nlgeom=True,
            inc=1000,
            time_params=(0.001, 1.0, 1e-8, 0.01),
            boundaries=[(2, 3, 3, 0.005)],
        )

        out_path = tmp_path / "steps.inp"
        write_abaqus_model(
            out_path,
            coords,
            conn,
            E=200e9,
            nu=0.3,
            steps=[step1, step2],
        )

        mesh = read_abaqus_inp(out_path)

        assert len(mesh.steps) == 2
        s1 = mesh.steps[0]
        assert s1.procedure == "STATIC"
        assert s1.time_params == pytest.approx((0.1, 1.0, 1e-6, 0.5))
        assert len(s1.boundaries) >= 1

        s2 = mesh.steps[1]
        assert s2.procedure == "DYNAMIC"
        assert s2.time_params == pytest.approx((0.001, 1.0, 1e-8, 0.01))

    def test_model_with_contact(self, tmp_path):
        """接触定義付きモデルのラウンドトリップ."""
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        conn = np.array([[0, 1]], dtype=int)

        si = InpSurfaceInteraction(
            name="CPROP",
            pressure_overclosure="HARD",
            friction=0.3,
            algorithm="NCP",
            mortar=True,
        )
        surf = InpSurfaceDef(
            name="STRAND_SURF",
            type="ELEMENT",
            elset="ALL_ELEMS",
        )
        contact = InpContactDef(
            interaction="CPROP",
            inclusions=[],
            exclude_same_layer=True,
            surfaces=[surf],
            surface_interactions=[si],
        )

        step = InpStep(
            procedure="STATIC",
            time_params=(1.0, 1.0),
            contact=contact,
        )

        out_path = tmp_path / "contact.inp"
        write_abaqus_model(
            out_path,
            coords,
            conn,
            E=200e9,
            nu=0.3,
            steps=[step],
        )

        mesh = read_abaqus_inp(out_path)

        # サーフェス
        assert len(mesh.surfaces) >= 1
        # サーフェスインタラクション
        assert len(mesh.surface_interactions) >= 1
        si_read = mesh.surface_interactions[0]
        assert si_read.name == "CPROP"
        assert si_read.pressure_overclosure == "HARD"
        assert si_read.friction == pytest.approx(0.3)

    def test_model_with_initial_conditions(self, tmp_path):
        """初期条件付きモデルのラウンドトリップ."""
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        conn = np.array([[0, 1]], dtype=int)

        ic = InpInitialCondition(
            type="VELOCITY",
            data=[(1, 1, 10.0), (2, 1, 10.0)],
        )

        out_path = tmp_path / "ic.inp"
        write_abaqus_model(
            out_path,
            coords,
            conn,
            E=200e9,
            nu=0.3,
            initial_conditions=[ic],
        )

        mesh = read_abaqus_inp(out_path)

        assert len(mesh.initial_conditions) == 1
        ic_read = mesh.initial_conditions[0]
        assert ic_read.type == "VELOCITY"
        assert len(ic_read.data) == 2
        assert ic_read.data[0] == (1, 1, 10.0)

    def test_nset_elset_roundtrip(self, tmp_path):
        """NSET/ELSET のラウンドトリップ."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        conn = np.array([[0, 1], [1, 2], [2, 3]], dtype=int)

        out_path = tmp_path / "sets.inp"
        write_abaqus_model(
            out_path,
            coords,
            conn,
            nsets={"FIXED_END": [1], "FREE_END": [4]},
            elsets={"STRAND_0": [1, 2, 3]},
            E=200e9,
            nu=0.3,
        )

        mesh = read_abaqus_inp(out_path)

        assert "FIXED_END" in mesh.nsets
        assert 1 in mesh.nsets["FIXED_END"]
        assert "FREE_END" in mesh.nsets
        assert 4 in mesh.nsets["FREE_END"]
        assert "STRAND_0" in mesh.elsets
        assert mesh.elsets["STRAND_0"] == [1, 2, 3]


# ====================================================================
# ヘッダテスト
# ====================================================================


class TestHeading:
    """*HEADING パーステスト."""

    def test_heading_parse(self, tmp_path):
        p = tmp_path / "test.inp"
        p.write_text(
            "*HEADING\n** My Model Title\n** Generated by xkep-cae\n**\n*NODE\n1, 0.0, 0.0\n"
        )
        mesh = read_abaqus_inp(p)
        assert "My Model Title" in mesh.heading


# ====================================================================
# 後方互換テスト
# ====================================================================


class TestBackwardCompatibility:
    """既存 API との後方互換テスト."""

    def test_read_abaqus_inp_still_works(self, tmp_path):
        """read_abaqus_inp が InpReader に委譲しても同じ結果を返す."""
        coords = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        conn = np.array([[0, 1], [1, 2]], dtype=int)

        p = tmp_path / "test.inp"
        write_abaqus_inp(
            p,
            coords,
            conn,
            elem_type="B21",
            E=100e9,
            nu=0.25,
            density=2700.0,
            beam_section_type="RECT",
            beam_section_dims=[0.1, 0.2],
        )

        mesh = read_abaqus_inp(p)
        assert len(mesh.nodes) == 3
        assert len(mesh.element_groups) == 1
        assert mesh.materials[0].elastic == pytest.approx((100e9, 0.25))
        assert mesh.materials[0].density == pytest.approx(2700.0)
        assert mesh.beam_sections[0].section_type == "RECT"
        assert mesh.beam_sections[0].dimensions == pytest.approx([0.1, 0.2])

    def test_new_fields_have_defaults(self):
        """新しい AbaqusMesh フィールドにデフォルト値があること."""
        mesh = AbaqusMesh()
        assert mesh.heading == ""
        assert mesh.steps == []
        assert mesh.initial_conditions == []
        assert mesh.surfaces == []
        assert mesh.surface_interactions == []
        assert mesh.contact_defs == []
