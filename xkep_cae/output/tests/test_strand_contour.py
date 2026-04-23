"""Strand3DContourProcess API + 基本動作テスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xkep_cae.core import (
    MeshData,
    PostProcess,
    SolverResultData,
)
from xkep_cae.core.data import ContactPairSnapshotEntry
from xkep_cae.core.testing import binds_to
from xkep_cae.output.strand_contour import (
    Strand3DContourConfig,
    Strand3DContourProcess,
    Strand3DContourResult,
    _element_chattering_score,
    _element_contact_mask,
    _element_curvature,
)


def _make_two_strand_mesh(n_nodes_per_strand: int = 17) -> MeshData:
    """2 本の直線素線（3D ベンチ用）."""
    coords_list = []
    conn_list = []
    strand_ids_list = []
    for strand_id in range(2):
        y_offset = 0.1 * strand_id
        for i in range(n_nodes_per_strand):
            x = i * 1.0 / (n_nodes_per_strand - 1)
            coords_list.append([x, y_offset, 0.0])
        base = strand_id * n_nodes_per_strand
        for i in range(n_nodes_per_strand - 1):
            conn_list.append([base + i, base + i + 1])
            strand_ids_list.append(strand_id)

    return MeshData(
        node_coords=np.array(coords_list, dtype=float),
        connectivity=np.array(conn_list, dtype=int),
        radii=0.05,
        n_strands=2,
        strand_ids=np.array(strand_ids_list, dtype=int),
    )


def _make_result(
    mesh: MeshData,
    *,
    pair_history: tuple = (),
    load_frac: float = 1.0,
) -> SolverResultData:
    """最小限の SolverResultData を構築."""
    n_nodes = mesh.node_coords.shape[0]
    u = np.zeros(n_nodes * 6)
    # 上方向にわずかに変形
    for i in range(n_nodes):
        u[i * 6 + 2] = 0.01 * mesh.node_coords[i, 0]
    return SolverResultData(
        u=u,
        converged=True,
        n_increments=len(pair_history) if pair_history else 1,
        n_cutbacks=0,
        total_attempts=1,
        load_history=(load_frac,),
        contact_pair_history=pair_history,
    )


@binds_to(Strand3DContourProcess)
class TestStrand3DContourProcessAPI:
    """Strand3DContourProcess の API テスト."""

    def test_protocol_conformance(self):
        assert issubclass(Strand3DContourProcess, PostProcess)

    def test_basic_rendering(self, tmp_path: Path):
        """基本的なレンダリング（pair_history なし、chattering skip）."""
        pytest.importorskip("matplotlib")
        mesh = _make_two_strand_mesh()
        result = _make_result(mesh)
        cfg = Strand3DContourConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="test_basic",
            requested_fields=("contact", "stress", "curvature"),
        )
        out = Strand3DContourProcess().process(cfg)
        assert isinstance(out, Strand3DContourResult)
        # 3 フィールド指定 → 3 画像生成
        assert len(out.image_paths) == 3
        for p in out.image_paths:
            assert Path(p).exists()
            assert Path(p).stat().st_size > 0
        # 統計が格納されている
        assert set(out.field_stats.keys()) == {"contact", "stress", "curvature"}

    def test_chatter_skip_without_history(self, tmp_path: Path):
        """pair_history が無ければ chatter フィールドは自動 skip."""
        pytest.importorskip("matplotlib")
        mesh = _make_two_strand_mesh()
        result = _make_result(mesh)  # pair_history=()
        cfg = Strand3DContourConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="test_noskip",
            requested_fields=("contact", "chatter_binary", "chatter_score"),
        )
        out = Strand3DContourProcess().process(cfg)
        # chatter 系は skip、contact のみ生成
        assert len(out.image_paths) == 1
        assert "contact" in out.image_paths[0]

    def test_chattering_with_history(self, tmp_path: Path):
        """pair_history あり時のチャタリング検出."""
        pytest.importorskip("matplotlib")
        mesh = _make_two_strand_mesh()
        # 3 increment: pair (0,17) が active → inactive → active の flip
        history = (
            (
                0.3,
                (
                    ContactPairSnapshotEntry(
                        elem_a=0,
                        elem_b=17,
                        p_n=1.0,
                        gap=-0.01,
                        slip_s=0.0,
                        slip_t=0.0,
                        stick=True,
                        dissipation=0.0,
                    ),
                ),
            ),
            (0.6, ()),  # inactive
            (
                1.0,
                (
                    ContactPairSnapshotEntry(
                        elem_a=0,
                        elem_b=17,
                        p_n=2.0,
                        gap=-0.02,
                        slip_s=0.0,
                        slip_t=0.0,
                        stick=False,  # stick flip
                        dissipation=0.0,
                    ),
                ),
            ),
        )
        result = _make_result(mesh, pair_history=history)
        cfg = Strand3DContourConfig(
            solver_result=result,
            mesh=mesh,
            output_dir=str(tmp_path),
            prefix="test_chatter",
            requested_fields=("chatter_binary", "chatter_score"),
        )
        out = Strand3DContourProcess().process(cfg)
        assert len(out.image_paths) == 2
        # pair (0,17) は少なくとも 1 回 flip → chattering detected
        assert out.n_chattering_elements >= 2  # elem 0 + elem 17


class TestPureHelpers:
    """ヘルパ関数の純粋ロジックテスト."""

    def test_element_contact_mask_empty(self):
        """manager が None の場合は全要素 False."""

        class _NullManager:
            pairs = []

        mask, force = _element_contact_mask(5, _NullManager())
        assert not mask.any()
        assert float(force.sum()) == 0.0

    def test_element_curvature_straight_beam(self):
        """直線梁では曲率ゼロ."""
        n_elems = 4
        node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)
        connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=int)
        strand_ids = np.array([0] * n_elems, dtype=int)
        curv = _element_curvature(node_coords, connectivity, strand_ids)
        assert np.allclose(curv, 0.0, atol=1e-10)

    def test_element_chattering_score_empty_history(self):
        """履歴 0 件 or 1 件なら全要素 score=0."""
        score, mask = _element_chattering_score(10, ())
        assert np.allclose(score, 0.0)
        assert not mask.any()

    def test_element_chattering_score_two_flips(self):
        """3 increment で pair (0,5) が active→inactive→active の flip."""
        history = (
            (
                0.3,
                (
                    ContactPairSnapshotEntry(
                        elem_a=0,
                        elem_b=5,
                        p_n=1.0,
                        gap=-0.01,
                        slip_s=0.0,
                        slip_t=0.0,
                        stick=True,
                        dissipation=0.0,
                    ),
                ),
            ),
            (0.6, ()),
            (
                1.0,
                (
                    ContactPairSnapshotEntry(
                        elem_a=0,
                        elem_b=5,
                        p_n=1.0,
                        gap=-0.01,
                        slip_s=0.0,
                        slip_t=0.0,
                        stick=True,
                        dissipation=0.0,
                    ),
                ),
            ),
        )
        score, mask = _element_chattering_score(10, history)
        # 2 flips / 3 increments = 0.667 が elem 0 と 5 に加算
        assert mask[0] and mask[5]
        assert not mask[1]  # 関係ない要素
        assert float(score[0]) == pytest.approx(2.0 / 3.0, abs=1e-6)
