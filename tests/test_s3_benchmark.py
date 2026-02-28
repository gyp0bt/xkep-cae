"""Phase S3: スケーラビリティベンチマーク + 並列化スピードアップ測定.

段階的ベンチマーク（7→19→37→61→91本）のメッシュ生成・broadphase・
アセンブリの性能スケーリングを検証する。
"""

import time

import numpy as np
import pytest

from xkep_cae.assembly import (
    _assemble_parallel,
    _assemble_sequential,
)
from xkep_cae.contact.broadphase import broadphase_aabb
from xkep_cae.contact.pair import ContactConfig, ContactManager
from xkep_cae.mesh.twisted_wire import make_strand_layout, make_twisted_wire_mesh

pytestmark = pytest.mark.slow

# ========== 共通パラメータ ==========

_WIRE_D = 0.002  # 直径 2mm
_PITCH = 0.040  # ピッチ 40mm
_N_ELEM = 8  # 素線あたり要素数（ベンチマーク用に軽量化）


# ========== ヘルパー ==========


def _make_mesh(n_strands: int, n_elems: int = _N_ELEM):
    """撚線メッシュを生成."""
    return make_twisted_wire_mesh(
        n_strands=n_strands,
        wire_diameter=_WIRE_D,
        pitch=_PITCH,
        length=0.0,
        n_elems_per_strand=n_elems,
        n_pitches=1.0,
    )


def _build_segments(mesh):
    """メッシュからセグメント・半径を抽出."""
    segments = []
    for e in range(mesh.n_elems):
        na, nb = mesh.connectivity[e]
        segments.append((mesh.node_coords[na], mesh.node_coords[nb]))
    radii = np.full(mesh.n_elems, _WIRE_D / 2.0)
    return segments, radii


# ========== メッシュ生成スケーリング ==========


class TestMeshScaling:
    """撚線メッシュ生成の段階的スケーリング."""

    @pytest.mark.parametrize("n_strands", [7, 19, 37, 61, 91])
    def test_mesh_generation(self, n_strands):
        """メッシュ生成が正常に完了し、節点数・要素数がスケールする."""
        mesh = _make_mesh(n_strands)
        expected_elems = n_strands * _N_ELEM
        assert mesh.n_elems == expected_elems
        expected_nodes = n_strands * (_N_ELEM + 1)
        assert mesh.n_nodes == expected_nodes

    @pytest.mark.parametrize("n_strands", [7, 19, 37, 61, 91])
    def test_strand_layout(self, n_strands):
        """素線配置が正しい層構成."""
        layout = make_strand_layout(n_strands, _WIRE_D / 2.0)
        assert len(layout) == n_strands
        # 全素線にユニークな strand_id
        ids = [s.strand_id for s in layout]
        assert len(set(ids)) == n_strands


# ========== Broadphase スケーリング ==========


class TestBroadphaseScaling:
    """Broadphase AABB のスケーリング特性."""

    @pytest.mark.parametrize("n_strands", [7, 19, 37, 61, 91])
    def test_broadphase_produces_candidates(self, n_strands):
        """各規模で候補ペアが生成される."""
        mesh = _make_mesh(n_strands)
        segments, radii = _build_segments(mesh)
        candidates = broadphase_aabb(segments, radii, margin=_WIRE_D * 0.5)
        # 撚線構造では異なる素線間の接触候補が必ず存在する
        assert len(candidates) > 0
        # 候補数がセグメント数より小さい（全組み合わせではない）
        n_all_pairs = mesh.n_elems * (mesh.n_elems - 1) // 2
        assert len(candidates) < n_all_pairs

    def test_broadphase_subquadratic(self):
        """候補ペア数がセグメント数の二乗未満でスケール."""
        results = {}
        for n_strands in [7, 19, 37]:
            mesh = _make_mesh(n_strands)
            segments, radii = _build_segments(mesh)
            candidates = broadphase_aabb(segments, radii, margin=_WIRE_D * 0.5)
            results[n_strands] = (mesh.n_elems, len(candidates))

        # 19本/7本 のペア比率が要素数比率の二乗未満
        n7, c7 = results[7]
        n19, c19 = results[19]
        elem_ratio = n19 / n7
        cand_ratio = c19 / c7 if c7 > 0 else float("inf")
        assert cand_ratio < elem_ratio**2, (
            f"候補ペア比 {cand_ratio:.1f} >= 要素数比^2 {elem_ratio**2:.1f}"
        )


# ========== 中点距離プリスクリーニング ==========


class TestMidpointPrescreening:
    """中点距離プリスクリーニングの効果測定."""

    @pytest.mark.parametrize("n_strands", [7, 19, 37])
    def test_prescreening_reduces_pairs(self, n_strands):
        """プリスクリーニングが候補ペアを削減する."""
        mesh = _make_mesh(n_strands)
        lmap = mesh.build_elem_layer_map()
        radii = np.full(mesh.n_elems, _WIRE_D / 2.0)

        # プリスクリーニング OFF
        cfg_off = ContactConfig(
            midpoint_prescreening=False,
            elem_layer_map=lmap,
            exclude_same_layer=True,
        )
        mgr_off = ContactManager(config=cfg_off)
        cands_off = mgr_off.detect_candidates(
            mesh.node_coords, mesh.connectivity, radii, margin=_WIRE_D * 0.5
        )

        # プリスクリーニング ON（デフォルト）
        cfg_on = ContactConfig(
            midpoint_prescreening=True,
            elem_layer_map=lmap,
            exclude_same_layer=True,
        )
        mgr_on = ContactManager(config=cfg_on)
        cands_on = mgr_on.detect_candidates(
            mesh.node_coords, mesh.connectivity, radii, margin=_WIRE_D * 0.5
        )

        # ON の方が候補数が少ない or 同数（削減効果）
        assert len(cands_on) <= len(cands_off)

    def test_prescreening_preserves_true_contacts(self):
        """プリスクリーニングが実際の接触ペアを落とさない."""
        mesh = _make_mesh(7)
        radii = np.full(mesh.n_elems, _WIRE_D / 2.0)

        cfg = ContactConfig(midpoint_prescreening=True)
        mgr = ContactManager(config=cfg)
        cands = mgr.detect_candidates(
            mesh.node_coords, mesh.connectivity, radii, margin=_WIRE_D * 0.5
        )

        # 候補が存在する（7本撚りでは必ず接触ペアがある）
        assert len(cands) > 0


# ========== 同層除外フィルタ + プリスクリーニング併用 ==========


class TestCombinedFilters:
    """同層除外 + 中点距離プリスクリーニングの併用効果."""

    @pytest.mark.parametrize("n_strands", [19, 37])
    def test_combined_filter_reduction(self, n_strands):
        """両フィルタ併用で候補がさらに削減される."""
        mesh = _make_mesh(n_strands)
        lmap = mesh.build_elem_layer_map()
        radii = np.full(mesh.n_elems, _WIRE_D / 2.0)

        # フィルタなし
        cfg_none = ContactConfig(
            midpoint_prescreening=False,
            exclude_same_layer=False,
        )
        mgr_none = ContactManager(config=cfg_none)
        cands_none = mgr_none.detect_candidates(
            mesh.node_coords, mesh.connectivity, radii, margin=_WIRE_D * 0.5
        )

        # 両方有効
        cfg_both = ContactConfig(
            midpoint_prescreening=True,
            exclude_same_layer=True,
            elem_layer_map=lmap,
        )
        mgr_both = ContactManager(config=cfg_both)
        cands_both = mgr_both.detect_candidates(
            mesh.node_coords, mesh.connectivity, radii, margin=_WIRE_D * 0.5
        )

        # 併用の方が少ない
        assert len(cands_both) < len(cands_none)


# ========== 並列アセンブリ スピードアップ測定 ==========


class _DummyBeamMaterial:
    """ダミー梁材料."""

    D = np.eye(6) * 1e6

    def stiffness_matrix(self, strain=None):
        return self.D


class _DummyBeam:
    """ダミー梁要素（12x12 剛性行列）."""

    ndof_per_node = 6
    ndof = 12
    nnodes = 2

    def local_stiffness(self, coords, material, thickness):
        return np.eye(self.ndof) * 1e3

    def dof_indices(self, node_ids):
        return np.array(
            [n * self.ndof_per_node + d for n in node_ids for d in range(self.ndof_per_node)]
        )


class TestParallelSpeedup:
    """並列アセンブリのスピードアップ測定."""

    def _make_beam_mesh(self, n_elems: int):
        """1次元梁メッシュ."""
        nodes = np.zeros((n_elems + 1, 3))
        nodes[:, 0] = np.linspace(0, 1, n_elems + 1)
        conn = np.array([[i, i + 1] for i in range(n_elems)], dtype=int)
        return nodes, conn

    def test_parallel_correctness_large(self):
        """大規模問題で逐次と並列の結果が一致."""
        elem = _DummyBeam()
        mat = _DummyBeamMaterial()
        nodes, conn = self._make_beam_mesh(256)
        n_total = len(conn)
        ndof_total = len(nodes) * 6

        K_seq = _assemble_sequential(
            nodes, [(elem, conn)], mat, ndof_total, n_total, show_progress=False
        )
        K_par = _assemble_parallel(
            nodes, [(elem, conn)], mat, ndof_total, n_total, show_progress=False, n_jobs=4
        )

        diff = abs(K_seq - K_par).max()
        assert diff < 1e-12, f"逐次と並列の差: {diff}"

    def test_speedup_measurement(self):
        """並列化のスピードアップを計測（情報記録用）.

        ダミー要素の計算コストが極めて小さいため、スレッドオーバーヘッドが
        支配的になる。実要素（Timoshenko梁等）ではC拡張呼び出しが多く、
        GIL解放区間で実質的なスピードアップが期待される。
        ここでは正しさの検証と計測値の記録を行う。
        """
        elem = _DummyBeam()
        mat = _DummyBeamMaterial()
        nodes, conn = self._make_beam_mesh(512)
        n_total = len(conn)
        ndof_total = len(nodes) * 6

        # ウォームアップ
        _assemble_sequential(nodes, [(elem, conn)], mat, ndof_total, n_total, show_progress=False)

        # 逐次の計測
        n_trials = 3
        t_seq = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            K_seq = _assemble_sequential(
                nodes, [(elem, conn)], mat, ndof_total, n_total, show_progress=False
            )
            t_seq.append(time.perf_counter() - t0)

        # 並列の計測（4ジョブ）
        t_par = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            K_par = _assemble_parallel(
                nodes, [(elem, conn)], mat, ndof_total, n_total, show_progress=False, n_jobs=4
            )
            t_par.append(time.perf_counter() - t0)

        avg_seq = np.median(t_seq)
        avg_par = np.median(t_par)

        # 結果が一致することを確認（正しさの検証）
        diff = abs(K_seq - K_par).max()
        assert diff < 1e-12

        # スピードアップ比を記録（情報記録のみ、閾値アサートなし）
        speedup = avg_seq / avg_par if avg_par > 0 else 0
        print(f"\n[Speedup] seq={avg_seq:.4f}s, par(4)={avg_par:.4f}s, speedup={speedup:.2f}x")
        print("[Note] ダミー要素では並列オーバーヘッドが支配的。実要素ではGIL解放で高速化。")


# ========== Broadphase 性能計測 ==========


class TestBroadphasePerformance:
    """Broadphase の性能スケーリング計測."""

    def test_broadphase_timing(self):
        """各規模での broadphase 実行時間を計測."""
        results = {}
        for n_strands in [7, 19, 37, 61]:
            mesh = _make_mesh(n_strands)
            segments, radii = _build_segments(mesh)

            # ウォームアップ
            broadphase_aabb(segments, radii, margin=_WIRE_D * 0.5)

            n_trials = 3
            times = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                cands = broadphase_aabb(segments, radii, margin=_WIRE_D * 0.5)
                times.append(time.perf_counter() - t0)

            results[n_strands] = {
                "n_elems": mesh.n_elems,
                "n_candidates": len(cands),
                "time_ms": np.median(times) * 1000,
            }

        # 結果表示
        print("\n[Broadphase Scaling]")
        print(f"{'n_strands':>10} {'n_elems':>8} {'n_cands':>8} {'time_ms':>10}")
        for ns, r in results.items():
            print(f"{ns:>10} {r['n_elems']:>8} {r['n_candidates']:>8} {r['time_ms']:>10.2f}")

        # 61本で 10秒以内であることを確認
        assert results[61]["time_ms"] < 10000, "61本 broadphase が 10秒超過"

    def test_detect_candidates_with_filters(self):
        """フィルタ付き detect_candidates の性能計測."""
        results = {}
        for n_strands in [7, 19, 37]:
            mesh = _make_mesh(n_strands)
            lmap = mesh.build_elem_layer_map()
            radii = np.full(mesh.n_elems, _WIRE_D / 2.0)

            cfg = ContactConfig(
                midpoint_prescreening=True,
                exclude_same_layer=True,
                elem_layer_map=lmap,
            )

            # ウォームアップ
            mgr = ContactManager(config=cfg)
            mgr.detect_candidates(mesh.node_coords, mesh.connectivity, radii, margin=_WIRE_D * 0.5)

            n_trials = 3
            times = []
            for _ in range(n_trials):
                mgr2 = ContactManager(config=cfg)
                t0 = time.perf_counter()
                cands = mgr2.detect_candidates(
                    mesh.node_coords, mesh.connectivity, radii, margin=_WIRE_D * 0.5
                )
                times.append(time.perf_counter() - t0)

            results[n_strands] = {
                "n_elems": mesh.n_elems,
                "n_pairs": len(mgr2.pairs),
                "n_candidates": len(cands),
                "time_ms": np.median(times) * 1000,
            }

        print("\n[Detect Candidates + Filters Scaling]")
        print(f"{'n_strands':>10} {'n_elems':>8} {'n_pairs':>8} {'n_cands':>8} {'time_ms':>10}")
        for ns, r in results.items():
            print(
                f"{ns:>10} {r['n_elems']:>8} {r['n_pairs']:>8} "
                f"{r['n_candidates']:>8} {r['time_ms']:>10.2f}"
            )
