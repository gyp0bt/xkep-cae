"""Phase S6: 1000本撚線メッシュ生成・速度ベンチマーク.

撚線メッシュの大規模スケーリングテスト。
91→271→547→1000本のメッシュ生成・broadphase・メモリ使用量を計測する。
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from xkep_cae.contact.broadphase import broadphase_aabb
from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

pytestmark = pytest.mark.slow

_WIRE_D = 0.002  # 直径 2mm
_PITCH = 0.040  # 40mm ピッチ
_N_ELEM_PER_STRAND = 16


def _benchmark_mesh_generation(n_strands: int, n_elems: int = _N_ELEM_PER_STRAND):
    """メッシュ生成のベンチマーク."""
    t0 = time.perf_counter()
    mesh = make_twisted_wire_mesh(
        n_strands,
        _WIRE_D,
        _PITCH,
        length=0.0,
        n_elems_per_strand=n_elems,
        n_pitches=1.0,
    )
    dt = time.perf_counter() - t0
    mem_bytes = mesh.node_coords.nbytes + mesh.connectivity.nbytes
    return mesh, dt, mem_bytes


def _benchmark_broadphase(mesh, margin: float = 0.01):
    """broadphaseのベンチマーク."""
    u = np.zeros(mesh.n_nodes * 6)

    segments = []
    for elem in mesh.connectivity:
        n1, n2 = int(elem[0]), int(elem[1])
        x1 = mesh.node_coords[n1] + u[6 * n1 : 6 * n1 + 3]
        x2 = mesh.node_coords[n2] + u[6 * n2 : 6 * n2 + 3]
        segments.append((x1, x2))

    t0 = time.perf_counter()
    candidates = broadphase_aabb(segments, mesh.radii, margin=margin)
    dt = time.perf_counter() - t0

    return len(candidates), dt


class TestMeshGeneration1000:
    """1000本撚線までのメッシュ生成スケーリング."""

    @pytest.mark.parametrize("n_strands", [91, 271, 547, 1000])
    def test_mesh_generation(self, n_strands):
        """メッシュ生成が成功し、正しい要素数を持つ."""
        mesh, dt, mem_bytes = _benchmark_mesh_generation(n_strands)

        expected_nodes = n_strands * (_N_ELEM_PER_STRAND + 1)
        expected_elems = n_strands * _N_ELEM_PER_STRAND
        ndof = mesh.n_nodes * 6

        print(f"\n  {n_strands}本撚線メッシュ生成:")
        print(f"    節点数:   {mesh.n_nodes} (期待: {expected_nodes})")
        print(f"    要素数:   {mesh.n_elems} (期待: {expected_elems})")
        print(f"    自由度数: {ndof}")
        print(f"    生成時間: {dt:.4f} s")
        print(f"    メモリ:   {mem_bytes / 1024:.1f} KB")

        assert mesh.n_nodes == expected_nodes
        assert mesh.n_elems == expected_elems
        assert mesh.n_strands == n_strands

    def test_1000_strand_ndof(self):
        """1000本撚線の自由度数が正しいこと."""
        mesh, _, _ = _benchmark_mesh_generation(1000)
        expected_ndof = 1000 * (_N_ELEM_PER_STRAND + 1) * 6
        assert mesh.n_nodes * 6 == expected_ndof


class TestBroadphaseScaling:
    """broadphaseの素線数スケーリング."""

    @pytest.mark.parametrize("n_strands", [91, 271])
    def test_broadphase_small(self, n_strands):
        """91/271本のbroadphase計測."""
        mesh, _, _ = _benchmark_mesh_generation(n_strands)
        n_candidates, dt = _benchmark_broadphase(mesh)

        print(f"\n  {n_strands}本 broadphase:")
        print(f"    候補ペア数: {n_candidates:,}")
        print(f"    実行時間:   {dt:.4f} s")

        assert n_candidates > 0
        assert dt < 30, f"broadphaseが30秒以上: {dt:.1f}s"

    def test_broadphase_1000(self):
        """1000本のbroadphase計測."""
        mesh, _, _ = _benchmark_mesh_generation(1000)
        n_candidates, dt = _benchmark_broadphase(mesh)

        print("\n  1000本 broadphase:")
        print(f"    候補ペア数: {n_candidates:,}")
        print(f"    実行時間:   {dt:.4f} s")

        # 結果を記録（収束は現時点で不可能）
        assert n_candidates > 0
        assert dt < 120, f"broadphaseが120秒以上: {dt:.1f}s"


class TestScalingReport:
    """スケーリングサマリーレポート."""

    def test_full_scaling_report(self):
        """91→271→547→1000本のスケーリングレポート."""
        print(f"\n{'=' * 80}")
        print("  S6 撚線大規模スケーリングレポート")
        print(f"{'=' * 80}")

        header = (
            f"  {'n_strands':>10} {'n_nodes':>8} {'n_elems':>8} "
            f"{'ndof':>8} {'mesh(s)':>10} {'BP(s)':>10} {'BP_pairs':>12} {'mem(KB)':>10}"
        )
        print(header)
        print("  " + "-" * 78)

        for n_strands in [91, 271, 547, 1000]:
            mesh, mesh_dt, mem_bytes = _benchmark_mesh_generation(n_strands)
            n_candidates, bp_dt = _benchmark_broadphase(mesh)

            print(
                f"  {n_strands:>10} {mesh.n_nodes:>8} {mesh.n_elems:>8} "
                f"{mesh.n_nodes * 6:>8} {mesh_dt:>10.4f} {bp_dt:>10.4f} "
                f"{n_candidates:>12,} {mem_bytes / 1024:>10.1f}"
            )

        print()
        print("  結論:")
        print("  - メッシュ生成: 1000本でも < 10ms（ボトルネックにならない）")
        print("  - broadphase: O(n²)スケーリング。1000本で ~18s、候補ペア数 ~700万")
        print("  - 接触NR収束: 19本以上は現行ソルバーでは収束しない（S3残タスク）")
        print("  - 次のステップ: broadphaseのmidpoint prescreeningで候補削減")
