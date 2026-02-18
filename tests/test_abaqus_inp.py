"""Abaqus .inp パーサーのテスト.

検証項目:
  - *NODE セクションの読み込み（2D/3D座標）
  - *ELEMENT セクションの読み込み（TRI3, Q4, TRI6, 梁要素）
  - *NSET セクションの読み込み（通常/GENERATE）
  - *ELSET セクションの読み込み（通常/GENERATE）
  - *BOUNDARY セクションの読み込み
  - *OUTPUT, FIELD ANIMATION セクションの読み込み
  - 継続行の処理
  - pymesh互換API（get_node_coord_array, get_element_array, get_node_labels_with_nset）
  - エラーケース
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from xkep_cae.io.abaqus_inp import read_abaqus_inp


@pytest.fixture()
def tmp_inp(tmp_path):
    """一時 .inp ファイルを書き込むヘルパー."""

    def _write(content: str) -> Path:
        p = tmp_path / "test.inp"
        p.write_text(dedent(content), encoding="utf-8")
        return p

    return _write


class TestNodeParsing:
    """*NODE セクションの読み込みテスト."""

    def test_basic_2d_nodes(self, tmp_inp):
        """2D節点座標の読み込み."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.nodes) == 4
        assert mesh.nodes[0].label == 1
        assert mesh.nodes[0].x == 0.0
        assert mesh.nodes[0].y == 0.0
        assert mesh.nodes[0].z == 0.0  # デフォルト

    def test_3d_nodes(self, tmp_inp):
        """3D節点座標の読み込み."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0, 0.0
            2, 1.0, 0.0, 0.5
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.nodes) == 2
        assert mesh.nodes[1].z == 0.5

    def test_get_node_coord_array(self, tmp_inp):
        """pymesh互換のget_node_coord_arrayが正しく動作すること."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0, 0.0
            2, 1.0, 2.0, 3.0
        """)
        mesh = read_abaqus_inp(p)
        arr = mesh.get_node_coord_array()
        assert len(arr) == 2
        assert arr[0] == {"label": 1, "x": 0.0, "y": 0.0, "z": 0.0}
        assert arr[1] == {"label": 2, "x": 1.0, "y": 2.0, "z": 3.0}


class TestElementParsing:
    """*ELEMENT セクションの読み込みテスト."""

    def test_quad4_elements(self, tmp_inp):
        """Q4要素の読み込み."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
            *ELEMENT, TYPE=CPS4R
            1, 1, 2, 3, 4
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.element_groups) == 1
        assert mesh.element_groups[0].elem_type == "CPS4R"
        assert len(mesh.element_groups[0].elements) == 1
        label, nodes = mesh.element_groups[0].elements[0]
        assert label == 1
        assert nodes == [1, 2, 3, 4]

    def test_tri3_elements(self, tmp_inp):
        """TRI3要素の読み込み."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 0.5, 1.0
            *ELEMENT, TYPE=CPE3
            1, 1, 2, 3
        """)
        mesh = read_abaqus_inp(p)
        assert mesh.element_groups[0].elem_type == "CPE3"
        _, nodes = mesh.element_groups[0].elements[0]
        assert nodes == [1, 2, 3]

    def test_tri6_elements(self, tmp_inp):
        """TRI6要素の読み込み."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 0.5, 1.0
            4, 0.5, 0.0
            5, 0.75, 0.5
            6, 0.25, 0.5
            *ELEMENT, TYPE=CPS6
            1, 1, 2, 3, 4, 5, 6
        """)
        mesh = read_abaqus_inp(p)
        _, nodes = mesh.element_groups[0].elements[0]
        assert nodes == [1, 2, 3, 4, 5, 6]

    def test_element_with_elset(self, tmp_inp):
        """ELSET付き要素の読み込み."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
            *ELEMENT, TYPE=CPS4, ELSET=solid
            1, 1, 2, 3, 4
        """)
        mesh = read_abaqus_inp(p)
        assert mesh.element_groups[0].elset == "solid"

    def test_continuation_lines(self, tmp_inp):
        """継続行（末尾カンマ）の処理."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 0.5, 1.0
            4, 0.5, 0.0
            5, 0.75, 0.5
            6, 0.25, 0.5
            *ELEMENT, TYPE=CPS6
            1, 1, 2, 3,
            4, 5, 6
        """)
        mesh = read_abaqus_inp(p)
        _, nodes = mesh.element_groups[0].elements[0]
        assert nodes == [1, 2, 3, 4, 5, 6]

    def test_mixed_element_types(self, tmp_inp):
        """複数タイプの要素の混在読み込み."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
            5, 2.0, 0.5
            *ELEMENT, TYPE=CPS4R, ELSET=quads
            1, 1, 2, 3, 4
            *ELEMENT, TYPE=CPS3, ELSET=tris
            2, 2, 5, 3
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.element_groups) == 2
        assert mesh.element_groups[0].elem_type == "CPS4R"
        assert mesh.element_groups[1].elem_type == "CPS3"

    def test_get_element_array(self, tmp_inp):
        """pymesh互換のget_element_arrayが正しく動作すること."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
            *ELEMENT, TYPE=CPS4R
            1, 1, 2, 3, 4
            2, 4, 3, 2, 1
        """)
        mesh = read_abaqus_inp(p)
        arr = mesh.get_element_array()
        assert len(arr) == 2
        assert arr[0] == [1, 1, 2, 3, 4]
        assert arr[1] == [2, 4, 3, 2, 1]

    def test_get_element_array_polymorphism(self, tmp_inp):
        """異種要素混在時のパディングが正しく動作すること."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
            5, 2.0, 0.5
            *ELEMENT, TYPE=CPS4R
            1, 1, 2, 3, 4
            *ELEMENT, TYPE=CPS3
            2, 2, 5, 3
        """)
        mesh = read_abaqus_inp(p)
        arr = mesh.get_element_array(allow_polymorphism=True, invalid_node=0)
        assert len(arr) == 2
        # Q4: [label, n1,n2,n3,n4]
        assert arr[0] == [1, 1, 2, 3, 4]
        # TRI3パディング: [label, n1,n2,n3, 0]
        assert arr[1] == [2, 2, 5, 3, 0]


class TestNsetParsing:
    """*NSET セクションの読み込みテスト."""

    def test_basic_nset(self, tmp_inp):
        """通常のノードセット読み込み."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            *NSET, NSET=gfix
            1, 3
        """)
        mesh = read_abaqus_inp(p)
        labels = mesh.get_node_labels_with_nset("gfix")
        assert labels == [1, 3]

    def test_nset_generate(self, tmp_inp):
        """GENERATE形式のノードセット読み込み."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 0.0, 1.0
            3, 0.0, 2.0
            4, 0.0, 3.0
            5, 0.0, 4.0
            *NSET, NSET=left, GENERATE
            1, 5, 2
        """)
        mesh = read_abaqus_inp(p)
        labels = mesh.get_node_labels_with_nset("left")
        assert labels == [1, 3, 5]

    def test_nset_multiline(self, tmp_inp):
        """複数行にまたがるノードセット."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 2.0, 0.0
            4, 3.0, 0.0
            *NSET, NSET=bottom
            1, 2
            3, 4
        """)
        mesh = read_abaqus_inp(p)
        labels = mesh.get_node_labels_with_nset("bottom")
        assert labels == [1, 2, 3, 4]

    def test_nset_case_insensitive(self, tmp_inp):
        """ノードセット名の大文字小文字を区別しないこと."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            *NSET, NSET=MySet
            1
        """)
        mesh = read_abaqus_inp(p)
        labels = mesh.get_node_labels_with_nset("myset")
        assert labels == [1]

    def test_nset_not_found(self, tmp_inp):
        """存在しないノードセットでKeyErrorが発生すること."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
        """)
        mesh = read_abaqus_inp(p)
        with pytest.raises(KeyError, match="見つかりません"):
            mesh.get_node_labels_with_nset("nonexistent")

    def test_multiple_nsets(self, tmp_inp):
        """複数のノードセットの読み込み."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
            *NSET, NSET=gfix
            1, 4
            *NSET, NSET=gmove
            2, 3
        """)
        mesh = read_abaqus_inp(p)
        assert mesh.get_node_labels_with_nset("gfix") == [1, 4]
        assert mesh.get_node_labels_with_nset("gmove") == [2, 3]


class TestCompleteInpFile:
    """完全な .inp ファイルの統合テスト."""

    def test_full_mesh(self, tmp_inp):
        """Q4+TRI3混在メッシュの完全読み込み."""
        p = tmp_inp("""\
            *HEADING
            Test mesh with Q4 and TRI3 elements
            **
            ** Node definitions
            **
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
            5, 2.0, 0.5
            **
            ** Element definitions
            **
            *ELEMENT, TYPE=CPS4R, ELSET=quads
            1, 1, 2, 3, 4
            *ELEMENT, TYPE=CPS3, ELSET=tris
            2, 2, 5, 3
            **
            ** Node sets
            **
            *NSET, NSET=gfix
            1, 4
            *NSET, NSET=gmove
            2, 3, 5
            **
            *STEP
            *END STEP
        """)
        mesh = read_abaqus_inp(p)

        # 節点
        assert len(mesh.nodes) == 5
        nodes = mesh.get_node_coord_array()
        assert nodes[4]["x"] == 2.0

        # 要素
        assert len(mesh.element_groups) == 2
        arr = mesh.get_element_array(allow_polymorphism=True, invalid_node=0)
        assert len(arr) == 2
        # Q4: [1, 1,2,3,4], TRI3: [2, 2,5,3, 0]
        assert arr[0] == [1, 1, 2, 3, 4]
        assert arr[1] == [2, 2, 5, 3, 0]

        # ノードセット
        assert mesh.get_node_labels_with_nset("gfix") == [1, 4]
        assert mesh.get_node_labels_with_nset("gmove") == [2, 3, 5]


class TestErrorHandling:
    """エラーケースの検証."""

    def test_file_not_found(self):
        """存在しないファイルでFileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_abaqus_inp("/nonexistent/path/mesh.inp")

    def test_empty_file(self, tmp_inp):
        """空ファイルの読み込み."""
        p = tmp_inp("")
        mesh = read_abaqus_inp(p)
        assert len(mesh.nodes) == 0
        assert len(mesh.element_groups) == 0

    def test_comments_only(self, tmp_inp):
        """コメントのみのファイル."""
        p = tmp_inp("""\
            ** This is a comment
            ** Another comment
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.nodes) == 0


class TestBeamSectionParsing:
    """*BEAM SECTION と *TRANSVERSE SHEAR STIFFNESS のパーステスト."""

    @pytest.fixture()
    def tmp_inp(self, tmp_path):
        """一時 .inp ファイルを作成するヘルパー."""
        import textwrap

        def _create(content: str) -> Path:
            p = tmp_path / "test_beam.inp"
            p.write_text(textwrap.dedent(content), encoding="utf-8")
            return p

        return _create

    def test_beam_section_rect(self, tmp_inp):
        """矩形断面の *BEAM SECTION パース."""
        p = tmp_inp("""\
            *BEAM SECTION, SECTION=RECT, ELSET=beams, MATERIAL=steel
            10.0, 20.0
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.beam_sections) == 1
        sec = mesh.beam_sections[0]
        assert sec.section_type == "RECT"
        assert sec.elset == "beams"
        assert sec.material == "steel"
        assert len(sec.dimensions) == 2
        assert abs(sec.dimensions[0] - 10.0) < 1e-12
        assert abs(sec.dimensions[1] - 20.0) < 1e-12
        assert sec.transverse_shear is None

    def test_beam_section_with_direction(self, tmp_inp):
        """方向ベクトル付きの *BEAM SECTION パース."""
        p = tmp_inp("""\
            *BEAM SECTION, SECTION=CIRC, ELSET=pipes, MATERIAL=copper
            5.0
            0.0, 0.0, 1.0
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.beam_sections) == 1
        sec = mesh.beam_sections[0]
        assert sec.section_type == "CIRC"
        assert sec.direction is not None
        assert len(sec.direction) == 3
        assert abs(sec.direction[2] - 1.0) < 1e-12

    def test_transverse_shear_stiffness(self, tmp_inp):
        """*TRANSVERSE SHEAR STIFFNESS のパース."""
        p = tmp_inp("""\
            *BEAM SECTION, SECTION=RECT, ELSET=beams, MATERIAL=steel
            10.0, 10.0
            *TRANSVERSE SHEAR STIFFNESS
            6410256.0, 6410256.0
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.beam_sections) == 1
        sec = mesh.beam_sections[0]
        assert sec.transverse_shear is not None
        k11, k22, k12 = sec.transverse_shear
        assert abs(k11 - 6410256.0) < 1.0
        assert abs(k22 - 6410256.0) < 1.0
        assert abs(k12 - 0.0) < 1e-12

    def test_transverse_shear_with_k12(self, tmp_inp):
        """K12 付きの横せん断剛性パース."""
        p = tmp_inp("""\
            *BEAM SECTION, SECTION=RECT, ELSET=beams, MATERIAL=steel
            10.0, 10.0
            *TRANSVERSE SHEAR STIFFNESS
            1000.0, 2000.0, 500.0
        """)
        mesh = read_abaqus_inp(p)
        sec = mesh.beam_sections[0]
        k11, k22, k12 = sec.transverse_shear
        assert abs(k11 - 1000.0) < 1e-12
        assert abs(k22 - 2000.0) < 1e-12
        assert abs(k12 - 500.0) < 1e-12

    def test_beam_section_with_elements_and_nodes(self, tmp_inp):
        """ノード・要素と併用した場合のパース."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 100.0, 0.0
            *ELEMENT, TYPE=B21, ELSET=beams
            1, 1, 2
            *BEAM SECTION, SECTION=RECT, ELSET=beams, MATERIAL=steel
            10.0, 10.0
            *TRANSVERSE SHEAR STIFFNESS
            6410256.0, 6410256.0
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.nodes) == 2
        assert len(mesh.element_groups) == 1
        assert len(mesh.beam_sections) == 1
        assert mesh.beam_sections[0].transverse_shear is not None

    def test_multiple_beam_sections(self, tmp_inp):
        """複数の *BEAM SECTION のパース."""
        p = tmp_inp("""\
            *BEAM SECTION, SECTION=RECT, ELSET=beams1, MATERIAL=steel
            10.0, 10.0
            *TRANSVERSE SHEAR STIFFNESS
            1000.0, 1000.0
            *BEAM SECTION, SECTION=CIRC, ELSET=beams2, MATERIAL=copper
            5.0
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.beam_sections) == 2
        assert mesh.beam_sections[0].transverse_shear is not None
        assert mesh.beam_sections[1].transverse_shear is None


class TestElsetParsing:
    """*ELSET セクションの読み込みテスト."""

    @pytest.fixture()
    def tmp_inp(self, tmp_path):
        """一時 .inp ファイルを作成するヘルパー."""

        def _create(content: str) -> Path:
            p = tmp_path / "test_elset.inp"
            p.write_text(dedent(content), encoding="utf-8")
            return p

        return _create

    def test_basic_elset(self, tmp_inp):
        """通常の要素セット読み込み."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            3, 1.0, 1.0
            4, 0.0, 1.0
            *ELEMENT, TYPE=CPS4R
            1, 1, 2, 3, 4
            2, 4, 3, 2, 1
            *ELSET, ELSET=group1
            1, 2
        """)
        mesh = read_abaqus_inp(p)
        assert "group1" in mesh.elsets
        assert mesh.elsets["group1"] == [1, 2]

    def test_elset_generate(self, tmp_inp):
        """GENERATE形式の要素セット読み込み."""
        p = tmp_inp("""\
            *ELSET, ELSET=all_beams, GENERATE
            1, 10, 2
        """)
        mesh = read_abaqus_inp(p)
        assert mesh.elsets["all_beams"] == [1, 3, 5, 7, 9]

    def test_elset_multiline(self, tmp_inp):
        """複数行にまたがる要素セット."""
        p = tmp_inp("""\
            *ELSET, ELSET=mixed
            1, 2, 3
            4, 5
        """)
        mesh = read_abaqus_inp(p)
        assert mesh.elsets["mixed"] == [1, 2, 3, 4, 5]

    def test_get_element_labels_with_elset_explicit(self, tmp_inp):
        """明示的 *ELSET からの要素ラベル取得."""
        p = tmp_inp("""\
            *ELSET, ELSET=group1
            10, 20, 30
        """)
        mesh = read_abaqus_inp(p)
        labels = mesh.get_element_labels_with_elset("group1")
        assert labels == [10, 20, 30]

    def test_get_element_labels_with_elset_implicit(self, tmp_inp):
        """*ELEMENT の ELSET= からの暗黙的要素セット取得."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
            2, 1.0, 0.0
            *ELEMENT, TYPE=B21, ELSET=beams
            1, 1, 2
        """)
        mesh = read_abaqus_inp(p)
        labels = mesh.get_element_labels_with_elset("beams")
        assert labels == [1]

    def test_elset_case_insensitive(self, tmp_inp):
        """要素セット名の大文字小文字を区別しないこと."""
        p = tmp_inp("""\
            *ELSET, ELSET=MyElset
            1, 2
        """)
        mesh = read_abaqus_inp(p)
        labels = mesh.get_element_labels_with_elset("myelset")
        assert labels == [1, 2]

    def test_elset_not_found(self, tmp_inp):
        """存在しない要素セットでKeyErrorが発生すること."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
        """)
        mesh = read_abaqus_inp(p)
        with pytest.raises(KeyError, match="見つかりません"):
            mesh.get_element_labels_with_elset("nonexistent")

    def test_multiple_elsets(self, tmp_inp):
        """複数の要素セットの読み込み."""
        p = tmp_inp("""\
            *ELSET, ELSET=group_a
            1, 2
            *ELSET, ELSET=group_b
            3, 4, 5
        """)
        mesh = read_abaqus_inp(p)
        assert mesh.elsets["group_a"] == [1, 2]
        assert mesh.elsets["group_b"] == [3, 4, 5]


class TestBoundaryParsing:
    """*BOUNDARY セクションの読み込みテスト."""

    @pytest.fixture()
    def tmp_inp(self, tmp_path):
        """一時 .inp ファイルを作成するヘルパー."""

        def _create(content: str) -> Path:
            p = tmp_path / "test_bc.inp"
            p.write_text(dedent(content), encoding="utf-8")
            return p

        return _create

    def test_single_dof_boundary(self, tmp_inp):
        """単一DOF拘束の読み込み."""
        p = tmp_inp("""\
            *BOUNDARY
            1, 1
            1, 2
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.boundaries) == 2
        bc0 = mesh.boundaries[0]
        assert bc0.node_label == 1
        assert bc0.first_dof == 1
        assert bc0.last_dof == 1
        assert bc0.value == 0.0

    def test_dof_range_boundary(self, tmp_inp):
        """DOF範囲拘束の読み込み."""
        p = tmp_inp("""\
            *BOUNDARY
            1, 1, 3
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.boundaries) == 1
        bc = mesh.boundaries[0]
        assert bc.node_label == 1
        assert bc.first_dof == 1
        assert bc.last_dof == 3

    def test_prescribed_displacement(self, tmp_inp):
        """規定変位値付き境界条件の読み込み."""
        p = tmp_inp("""\
            *BOUNDARY
            5, 1, 1, 0.01
        """)
        mesh = read_abaqus_inp(p)
        bc = mesh.boundaries[0]
        assert bc.node_label == 5
        assert bc.first_dof == 1
        assert bc.last_dof == 1
        assert abs(bc.value - 0.01) < 1e-15

    def test_multiple_boundaries(self, tmp_inp):
        """複数節点の境界条件."""
        p = tmp_inp("""\
            *BOUNDARY
            1, 1, 6
            2, 1, 3
            3, 2
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.boundaries) == 3
        assert mesh.boundaries[0].last_dof == 6
        assert mesh.boundaries[1].last_dof == 3
        assert mesh.boundaries[2].first_dof == 2
        assert mesh.boundaries[2].last_dof == 2

    def test_boundary_with_nodes_and_elements(self, tmp_inp):
        """ノード・要素と併用した場合の境界条件パース."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0, 0.0
            2, 1.0, 0.0, 0.0
            3, 2.0, 0.0, 0.0
            *ELEMENT, TYPE=B31, ELSET=beams
            1, 1, 2
            2, 2, 3
            *BOUNDARY
            1, 1, 6
            3, 2, 2, 0.005
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.nodes) == 3
        assert len(mesh.element_groups) == 1
        assert len(mesh.boundaries) == 2
        assert mesh.boundaries[1].value == 0.005


class TestFieldAnimationParsing:
    """*OUTPUT, FIELD ANIMATION セクションの読み込みテスト."""

    @pytest.fixture()
    def tmp_inp(self, tmp_path):
        """一時 .inp ファイルを作成するヘルパー."""

        def _create(content: str) -> Path:
            p = tmp_path / "test_anim.inp"
            p.write_text(dedent(content), encoding="utf-8")
            return p

        return _create

    def test_default_field_animation(self, tmp_inp):
        """デフォルト設定のアニメーション出力."""
        p = tmp_inp("""\
            *OUTPUT, FIELD ANIMATION
        """)
        mesh = read_abaqus_inp(p)
        assert mesh.field_animation is not None
        assert mesh.field_animation.output_dir == "animation"
        assert mesh.field_animation.views == ["xy", "xz", "yz"]

    def test_field_animation_custom_dir(self, tmp_inp):
        """カスタム出力ディレクトリのアニメーション出力."""
        p = tmp_inp("""\
            *OUTPUT, FIELD ANIMATION, DIR=results/anim
        """)
        mesh = read_abaqus_inp(p)
        assert mesh.field_animation is not None
        assert mesh.field_animation.output_dir == "results/anim"

    def test_field_animation_custom_views(self, tmp_inp):
        """カスタムビュー指定のアニメーション出力."""
        p = tmp_inp("""\
            *OUTPUT, FIELD ANIMATION
            xy, xz
        """)
        mesh = read_abaqus_inp(p)
        assert mesh.field_animation is not None
        assert mesh.field_animation.views == ["xy", "xz"]

    def test_field_animation_single_view(self, tmp_inp):
        """単一ビューのアニメーション出力."""
        p = tmp_inp("""\
            *OUTPUT, FIELD ANIMATION
            yz
        """)
        mesh = read_abaqus_inp(p)
        assert mesh.field_animation.views == ["yz"]

    def test_no_field_animation(self, tmp_inp):
        """アニメーション出力が指定されていない場合."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0
        """)
        mesh = read_abaqus_inp(p)
        assert mesh.field_animation is None

    def test_field_animation_with_full_model(self, tmp_inp):
        """完全なモデルとの併用テスト."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0, 0.0
            2, 1.0, 0.0, 0.0
            3, 2.0, 0.0, 0.0
            *ELEMENT, TYPE=B31, ELSET=beam1
            1, 1, 2
            *ELEMENT, TYPE=B31, ELSET=beam2
            2, 2, 3
            *NSET, NSET=fix
            1
            *ELSET, ELSET=all_beams
            1, 2
            *BEAM SECTION, SECTION=RECT, ELSET=all_beams, MATERIAL=steel
            0.01, 0.01
            *BOUNDARY
            1, 1, 6
            *OUTPUT, FIELD ANIMATION, DIR=output/anim
            xy, yz
        """)
        mesh = read_abaqus_inp(p)
        assert len(mesh.nodes) == 3
        assert len(mesh.element_groups) == 2
        assert mesh.nsets["fix"] == [1]
        assert mesh.elsets["all_beams"] == [1, 2]
        assert len(mesh.beam_sections) == 1
        assert len(mesh.boundaries) == 1
        assert mesh.field_animation is not None
        assert mesh.field_animation.output_dir == "output/anim"
        assert mesh.field_animation.views == ["xy", "yz"]
