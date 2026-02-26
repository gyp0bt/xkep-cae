"""データセット生成の検証テスト."""

from __future__ import annotations

import numpy as np

from xkep_cae.thermal.dataset import (
    ThermalProblemConfig,
    generate_dataset,
    generate_single_sample,
    mesh_to_edge_index,
    place_heat_sources,
    sample_to_graph_data,
)
from xkep_cae.thermal.fem import make_rect_mesh


class TestHeatSourcePlacement:
    """発熱体配置の検証."""

    def test_source_within_bounds(self):
        config = ThermalProblemConfig(nx=10, ny=10)
        nodes, _, _ = make_rect_mesh(config.Lx, config.Ly, config.nx, config.ny)
        rng = np.random.default_rng(0)
        q, centers = place_heat_sources(nodes, config, rng)
        for cx, cy in centers:
            assert config.w_heat / 2 <= cx <= config.Lx - config.w_heat / 2
            assert config.h_heat / 2 <= cy <= config.Ly - config.h_heat / 2

    def test_source_count_range(self):
        config = ThermalProblemConfig(n_sources_min=2, n_sources_max=4, nx=10, ny=10)
        nodes, _, _ = make_rect_mesh(config.Lx, config.Ly, config.nx, config.ny)
        rng = np.random.default_rng(42)
        for _ in range(50):
            _, centers = place_heat_sources(nodes, config, rng)
            assert 2 <= len(centers) <= 4

    def test_nonzero_heat(self):
        config = ThermalProblemConfig(nx=10, ny=10)
        nodes, _, _ = make_rect_mesh(config.Lx, config.Ly, config.nx, config.ny)
        rng = np.random.default_rng(1)
        q, _ = place_heat_sources(nodes, config, rng)
        assert np.any(q > 0)


class TestGraphConversion:
    """グラフ変換の検証."""

    def test_edge_index_shape(self):
        _, conn, _ = make_rect_mesh(0.1, 0.1, 3, 3)
        ei = mesh_to_edge_index(conn)
        assert ei.shape[0] == 2
        assert ei.shape[1] > 0
        # 双方向: 偶数
        assert ei.shape[1] % 2 == 0

    def test_edge_index_symmetric(self):
        _, conn, _ = make_rect_mesh(0.1, 0.1, 3, 3)
        ei = mesh_to_edge_index(conn)
        edge_set = set()
        for i in range(ei.shape[1]):
            edge_set.add((ei[0, i], ei[1, i]))
        # 各辺の逆方向も存在
        for i in range(ei.shape[1]):
            assert (ei[1, i], ei[0, i]) in edge_set

    def test_node_count_consistent(self):
        config = ThermalProblemConfig(nx=5, ny=5)
        nodes, conn, edges = make_rect_mesh(config.Lx, config.Ly, config.nx, config.ny)
        rng = np.random.default_rng(0)
        sample = generate_single_sample(nodes, conn, edges, config, rng)
        graph = sample_to_graph_data(sample)
        assert graph["x"].shape[0] == len(nodes)
        assert graph["y"].shape[0] == len(nodes)
        assert graph["x"].shape[1] == 6
        assert graph["y"].shape[1] == 1


class TestDatasetGeneration:
    """データセット生成の検証."""

    def test_generate_correct_count(self):
        config = ThermalProblemConfig(nx=5, ny=5)
        dataset = generate_dataset(config, n_samples=10, seed=42)
        assert len(dataset) == 10

    def test_reproducibility(self):
        config = ThermalProblemConfig(nx=5, ny=5)
        d1 = generate_dataset(config, n_samples=5, seed=42)
        d2 = generate_dataset(config, n_samples=5, seed=42)
        for g1, g2 in zip(d1, d2, strict=True):
            np.testing.assert_array_equal(g1["x"], g2["x"])
            np.testing.assert_array_equal(g1["y"], g2["y"])

    def test_temperature_rise_positive(self):
        config = ThermalProblemConfig(nx=5, ny=5)
        dataset = generate_dataset(config, n_samples=5, seed=0)
        for g in dataset:
            assert np.all(g["y"] >= -1e-6)

    def test_different_seeds_different_data(self):
        config = ThermalProblemConfig(nx=5, ny=5)
        d1 = generate_dataset(config, n_samples=3, seed=0)
        d2 = generate_dataset(config, n_samples=3, seed=1)
        assert not np.allclose(d1[0]["y"], d2[0]["y"])
