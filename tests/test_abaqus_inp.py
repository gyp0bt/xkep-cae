"""Abaqus .inp パーサーのテスト.

検証項目:
  - *NODE セクションの読み込み（2D/3D座標）
  - *ELEMENT セクションの読み込み（TRI3, Q4, TRI6, 梁要素）
  - *NSET セクションの読み込み（通常/GENERATE）
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
