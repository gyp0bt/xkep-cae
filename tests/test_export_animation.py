"""FIELD ANIMATION エクスポートのテスト.

検証項目:
  - 梁要素セグメントの収集（要素セット別）
  - 各ビュー方向（xy, xz, yz）での描画
  - 全要素が画面に収まるビュー設定
  - 要素セット凡例の表示
  - 変形後座標でのフレーム描画
  - PNG画像ファイルの出力
  - GIFアニメーション出力
  - AbaqusMesh からの統合テスト
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

try:
    import matplotlib  # noqa: F401

    _has_matplotlib = True
except ImportError:
    _has_matplotlib = False

needs_matplotlib = pytest.mark.skipif(not _has_matplotlib, reason="matplotlib is not installed")

from xkep_cae.io.abaqus_inp import (  # noqa: E402
    AbaqusElementGroup,
    AbaqusMesh,
    AbaqusNode,
    read_abaqus_inp,
)
from xkep_cae.output.export_animation import (  # noqa: E402
    _collect_beam_segments,
    export_field_animation,
    export_field_animation_gif,
    render_beam_animation_frame,
)

try:
    from PIL import Image as PILImage  # noqa: F401

    _has_pillow = True
except ImportError:
    _has_pillow = False

needs_pillow = pytest.mark.skipif(not _has_pillow, reason="Pillow is not installed")
needs_matplotlib_and_pillow = pytest.mark.skipif(
    not (_has_matplotlib and _has_pillow),
    reason="matplotlib and/or Pillow is not installed",
)


@pytest.fixture()
def simple_beam_mesh():
    """2要素の単純な3D梁メッシュ."""
    mesh = AbaqusMesh()
    mesh.nodes = [
        AbaqusNode(label=1, x=0.0, y=0.0, z=0.0),
        AbaqusNode(label=2, x=1.0, y=0.0, z=0.0),
        AbaqusNode(label=3, x=2.0, y=0.0, z=0.0),
    ]
    group = AbaqusElementGroup(elem_type="B31", elset="beam1")
    group.elements = [(1, [1, 2]), (2, [2, 3])]
    mesh.element_groups = [group]
    return mesh


@pytest.fixture()
def multi_elset_mesh():
    """複数要素セットの3D梁メッシュ."""
    mesh = AbaqusMesh()
    mesh.nodes = [
        AbaqusNode(label=1, x=0.0, y=0.0, z=0.0),
        AbaqusNode(label=2, x=1.0, y=0.0, z=0.0),
        AbaqusNode(label=3, x=2.0, y=0.0, z=0.0),
        AbaqusNode(label=4, x=2.0, y=1.0, z=0.0),
    ]
    group1 = AbaqusElementGroup(elem_type="B31", elset="horizontal")
    group1.elements = [(1, [1, 2]), (2, [2, 3])]
    group2 = AbaqusElementGroup(elem_type="B31", elset="vertical")
    group2.elements = [(3, [3, 4])]
    mesh.element_groups = [group1, group2]
    return mesh


@pytest.fixture()
def beam_3d_mesh():
    """3次元配置の梁メッシュ."""
    mesh = AbaqusMesh()
    mesh.nodes = [
        AbaqusNode(label=1, x=0.0, y=0.0, z=0.0),
        AbaqusNode(label=2, x=1.0, y=0.0, z=0.0),
        AbaqusNode(label=3, x=1.0, y=1.0, z=0.0),
        AbaqusNode(label=4, x=1.0, y=1.0, z=1.0),
    ]
    group = AbaqusElementGroup(elem_type="B31", elset="frame")
    group.elements = [(1, [1, 2]), (2, [2, 3]), (3, [3, 4])]
    mesh.element_groups = [group]
    return mesh


class TestCollectBeamSegments:
    """梁要素セグメント収集のテスト."""

    def test_basic_segments(self, simple_beam_mesh):
        """基本的なセグメント収集."""
        segments = _collect_beam_segments(simple_beam_mesh)
        assert "beam1" in segments
        assert len(segments["beam1"]) == 2

    def test_segment_coordinates(self, simple_beam_mesh):
        """セグメント座標の正しさ."""
        segments = _collect_beam_segments(simple_beam_mesh)
        seg0_start, seg0_end = segments["beam1"][0]
        assert np.allclose(seg0_start, [0.0, 0.0, 0.0])
        assert np.allclose(seg0_end, [1.0, 0.0, 0.0])

    def test_multi_elset(self, multi_elset_mesh):
        """複数要素セットのセグメント収集."""
        segments = _collect_beam_segments(multi_elset_mesh)
        assert "horizontal" in segments
        assert "vertical" in segments
        assert len(segments["horizontal"]) == 2
        assert len(segments["vertical"]) == 1

    def test_with_deformed_coords(self, simple_beam_mesh):
        """変形後座標でのセグメント収集."""
        deformed = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [2.0, 0.3, 0.0],
            ]
        )
        segments = _collect_beam_segments(simple_beam_mesh, node_coords=deformed)
        seg1_start, seg1_end = segments["beam1"][1]
        assert np.allclose(seg1_start, [1.0, 0.1, 0.0])
        assert np.allclose(seg1_end, [2.0, 0.3, 0.0])

    def test_non_beam_elements_ignored(self):
        """非梁要素は無視されること."""
        mesh = AbaqusMesh()
        mesh.nodes = [
            AbaqusNode(label=1, x=0.0, y=0.0),
            AbaqusNode(label=2, x=1.0, y=0.0),
            AbaqusNode(label=3, x=1.0, y=1.0),
            AbaqusNode(label=4, x=0.0, y=1.0),
        ]
        group = AbaqusElementGroup(elem_type="CPS4R", elset="solid")
        group.elements = [(1, [1, 2, 3, 4])]
        mesh.element_groups = [group]
        segments = _collect_beam_segments(mesh)
        assert len(segments) == 0

    def test_2d_deformed_coords(self, simple_beam_mesh):
        """2D変形後座標（z省略）でのセグメント収集."""
        deformed = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.2],
                [2.0, 0.5],
            ]
        )
        segments = _collect_beam_segments(simple_beam_mesh, node_coords=deformed)
        seg0_start, seg0_end = segments["beam1"][0]
        assert np.allclose(seg0_start, [0.0, 0.0, 0.0])
        assert np.allclose(seg0_end, [1.0, 0.2, 0.0])


@needs_matplotlib
class TestRenderBeamAnimationFrame:
    """フレーム描画のテスト."""

    def test_xy_view(self, simple_beam_mesh):
        """XYビューの描画."""
        fig, ax = render_beam_animation_frame(simple_beam_mesh, view="xy")
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_xz_view(self, beam_3d_mesh):
        """XZビューの描画."""
        fig, ax = render_beam_animation_frame(beam_3d_mesh, view="xz")
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Z"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_yz_view(self, beam_3d_mesh):
        """YZビューの描画."""
        fig, ax = render_beam_animation_frame(beam_3d_mesh, view="yz")
        assert ax.get_xlabel() == "Y"
        assert ax.get_ylabel() == "Z"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_invalid_view(self, simple_beam_mesh):
        """無効なビュー方向でValueError."""
        with pytest.raises(ValueError, match="未対応のビュー方向"):
            render_beam_animation_frame(simple_beam_mesh, view="ab")

    def test_elset_legend(self, multi_elset_mesh):
        """要素セットの凡例が表示されること."""
        fig, ax = render_beam_animation_frame(multi_elset_mesh, view="xy")
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert "horizontal" in texts
        assert "vertical" in texts
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_title(self, simple_beam_mesh):
        """カスタムタイトルの設定."""
        fig, ax = render_beam_animation_frame(simple_beam_mesh, view="xy", title="Test Title")
        assert ax.get_title() == "Test Title"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_equal_aspect(self, simple_beam_mesh):
        """アスペクト比が等しいこと."""
        fig, ax = render_beam_animation_frame(simple_beam_mesh, view="xy")
        # matplotlib はset_aspect("equal")後にget_aspect()で1.0を返す場合がある
        aspect = ax.get_aspect()
        assert aspect == "equal" or aspect == 1.0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_deformed_coords(self, simple_beam_mesh):
        """変形後座標での描画."""
        deformed = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [2.0, 0.3, 0.0],
            ]
        )
        fig, ax = render_beam_animation_frame(simple_beam_mesh, view="xy", node_coords=deformed)
        # 描画が完了すること自体が検証
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


@needs_matplotlib
class TestExportFieldAnimation:
    """PNG出力のテスト."""

    def test_single_frame_default_views(self, simple_beam_mesh, tmp_path):
        """デフォルトビュー（3方向）の初期フレーム出力."""
        output_dir = tmp_path / "anim"
        files = export_field_animation(
            simple_beam_mesh,
            output_dir=output_dir,
        )
        assert len(files) == 3
        for f in files:
            assert f.exists()
            assert f.suffix == ".png"
        # ファイル名の検証
        names = sorted(f.name for f in files)
        assert "frame_0000_xy.png" in names
        assert "frame_0000_xz.png" in names
        assert "frame_0000_yz.png" in names

    def test_single_view(self, simple_beam_mesh, tmp_path):
        """単一ビューの出力."""
        output_dir = tmp_path / "anim"
        files = export_field_animation(
            simple_beam_mesh,
            output_dir=output_dir,
            views=["xy"],
        )
        assert len(files) == 1
        assert files[0].name == "frame_0000_xy.png"

    def test_multiple_frames(self, simple_beam_mesh, tmp_path):
        """複数フレームの出力."""
        frames = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.3, 0.0]]),
        ]
        output_dir = tmp_path / "anim"
        files = export_field_animation(
            simple_beam_mesh,
            output_dir=output_dir,
            views=["xy"],
            node_coords_frames=frames,
        )
        assert len(files) == 2
        assert files[0].name == "frame_0000_xy.png"
        assert files[1].name == "frame_0001_xy.png"

    def test_custom_frame_labels(self, simple_beam_mesh, tmp_path):
        """カスタムフレームラベルの出力."""
        output_dir = tmp_path / "anim"
        files = export_field_animation(
            simple_beam_mesh,
            output_dir=output_dir,
            views=["xy"],
            frame_labels=["t=0.0s"],
        )
        assert len(files) == 1

    def test_output_dir_creation(self, simple_beam_mesh, tmp_path):
        """存在しない出力ディレクトリの自動作成."""
        output_dir = tmp_path / "deep" / "nested" / "anim"
        assert not output_dir.exists()
        files = export_field_animation(
            simple_beam_mesh,
            output_dir=output_dir,
            views=["xy"],
        )
        assert output_dir.exists()
        assert len(files) == 1


@needs_matplotlib
class TestIntegrationWithInpParser:
    """パーサーとアニメーション出力の統合テスト."""

    @pytest.fixture()
    def tmp_inp(self, tmp_path):
        """一時 .inp ファイルを作成するヘルパー."""

        def _create(content: str) -> Path:
            p = tmp_path / "test_model.inp"
            p.write_text(dedent(content), encoding="utf-8")
            return p

        return _create

    def test_full_pipeline(self, tmp_inp, tmp_path):
        """パーサー → アニメーション出力のフルパイプライン."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0, 0.0
            2, 0.5, 0.0, 0.0
            3, 1.0, 0.0, 0.0
            4, 1.0, 0.5, 0.0
            5, 1.0, 1.0, 0.0
            *ELEMENT, TYPE=B31, ELSET=horiz
            1, 1, 2
            2, 2, 3
            *ELEMENT, TYPE=B31, ELSET=vert
            3, 3, 4
            4, 4, 5
            *ELSET, ELSET=all_beams
            1, 2, 3, 4
            *BOUNDARY
            1, 1, 6
            *OUTPUT, FIELD ANIMATION
            xy, xz
        """)
        mesh = read_abaqus_inp(p)

        # パーサーの検証
        assert len(mesh.nodes) == 5
        assert len(mesh.element_groups) == 2
        assert mesh.elsets["all_beams"] == [1, 2, 3, 4]
        assert len(mesh.boundaries) == 1
        assert mesh.field_animation is not None
        assert mesh.field_animation.views == ["xy", "xz"]

        # アニメーション出力
        output_dir = tmp_path / "anim_out"
        files = export_field_animation(
            mesh,
            output_dir=output_dir,
            views=mesh.field_animation.views,
        )
        assert len(files) == 2  # xy + xz
        for f in files:
            assert f.exists()
            assert f.stat().st_size > 0

    def test_pipeline_with_deformation(self, tmp_inp, tmp_path):
        """変形後座標を使用したアニメーション出力."""
        p = tmp_inp("""\
            *NODE
            1, 0.0, 0.0, 0.0
            2, 1.0, 0.0, 0.0
            3, 2.0, 0.0, 0.0
            *ELEMENT, TYPE=B31, ELSET=cantilever
            1, 1, 2
            2, 2, 3
            *BOUNDARY
            1, 1, 6
            *OUTPUT, FIELD ANIMATION
        """)
        mesh = read_abaqus_inp(p)

        # 3フレームの変形アニメーション
        frames = [
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
            np.array([[0, 0, 0], [1, 0.05, 0], [2, 0.15, 0]], dtype=float),
            np.array([[0, 0, 0], [1, 0.10, 0], [2, 0.30, 0]], dtype=float),
        ]
        output_dir = tmp_path / "deform_anim"
        files = export_field_animation(
            mesh,
            output_dir=output_dir,
            views=mesh.field_animation.views,
            node_coords_frames=frames,
            frame_labels=["t=0.0", "t=0.5", "t=1.0"],
        )
        # 3フレーム × 3ビュー = 9ファイル
        assert len(files) == 9
        for f in files:
            assert f.exists()


@needs_matplotlib_and_pillow
class TestExportFieldAnimationGif:
    """GIFアニメーション出力のテスト."""

    def test_single_frame_default_views(self, simple_beam_mesh, tmp_path):
        """デフォルトビュー（3方向）の初期フレームGIF出力."""
        output_dir = tmp_path / "gif_anim"
        files = export_field_animation_gif(
            simple_beam_mesh,
            output_dir=output_dir,
        )
        assert len(files) == 3
        for f in files:
            assert f.exists()
            assert f.suffix == ".gif"
            assert f.stat().st_size > 0
        names = sorted(f.name for f in files)
        assert "animation_xy.gif" in names
        assert "animation_xz.gif" in names
        assert "animation_yz.gif" in names

    def test_single_view_gif(self, simple_beam_mesh, tmp_path):
        """単一ビューのGIF出力."""
        output_dir = tmp_path / "gif_anim"
        files = export_field_animation_gif(
            simple_beam_mesh,
            output_dir=output_dir,
            views=["xy"],
        )
        assert len(files) == 1
        assert files[0].name == "animation_xy.gif"

    def test_multiple_frames_gif(self, simple_beam_mesh, tmp_path):
        """複数フレームのGIFアニメーション出力."""
        frames = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.3, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.6, 0.0]]),
        ]
        output_dir = tmp_path / "gif_anim"
        files = export_field_animation_gif(
            simple_beam_mesh,
            output_dir=output_dir,
            views=["xy"],
            node_coords_frames=frames,
            frame_labels=["t=0.0", "t=0.5", "t=1.0"],
        )
        assert len(files) == 1
        gif_path = files[0]
        assert gif_path.exists()
        # GIFファイルをPILで開いてフレーム数を検証
        from PIL import Image as PILImage

        with PILImage.open(gif_path) as img:
            n_frames = getattr(img, "n_frames", 1)
            assert n_frames == 3

    def test_gif_is_animated(self, simple_beam_mesh, tmp_path):
        """複数フレームのGIFがアニメーションGIFであること."""
        frames = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.15, 0.0], [2.0, 0.4, 0.0]]),
        ]
        output_dir = tmp_path / "gif_anim"
        files = export_field_animation_gif(
            simple_beam_mesh,
            output_dir=output_dir,
            views=["xy"],
            node_coords_frames=frames,
        )
        from PIL import Image as PILImage

        with PILImage.open(files[0]) as img:
            assert getattr(img, "is_animated", False)

    def test_custom_duration(self, simple_beam_mesh, tmp_path):
        """カスタムフレーム間隔のGIF出力."""
        frames = [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.3, 0.0]]),
        ]
        output_dir = tmp_path / "gif_anim"
        files = export_field_animation_gif(
            simple_beam_mesh,
            output_dir=output_dir,
            views=["xy"],
            node_coords_frames=frames,
            duration=500,
        )
        assert len(files) == 1
        assert files[0].exists()

    def test_output_dir_creation(self, simple_beam_mesh, tmp_path):
        """存在しない出力ディレクトリの自動作成."""
        output_dir = tmp_path / "deep" / "nested" / "gif_anim"
        assert not output_dir.exists()
        files = export_field_animation_gif(
            simple_beam_mesh,
            output_dir=output_dir,
            views=["xy"],
        )
        assert output_dir.exists()
        assert len(files) == 1

    def test_multi_elset_gif(self, multi_elset_mesh, tmp_path):
        """複数要素セットのGIF出力."""
        output_dir = tmp_path / "gif_anim"
        files = export_field_animation_gif(
            multi_elset_mesh,
            output_dir=output_dir,
            views=["xy"],
        )
        assert len(files) == 1
        assert files[0].exists()
        assert files[0].stat().st_size > 0
