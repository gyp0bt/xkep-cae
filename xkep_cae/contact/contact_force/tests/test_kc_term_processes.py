"""KcNormalStiffnessProcess / KcGeoStiffnessProcess のテスト (status-350 Phase C-1).

MCDD Phase C-1 で導入された K_c 項別 Process の単体検証。
``tangent_components()`` は orchestrator として各 Process を呼び出すため、
本テストで Process 単体のアクティブ判定・組み立てセマンティクスを固定する。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xkep_cae.contact.contact_force import (
    ContactForceStStiffnessInput,
    ContactForceStStiffnessProcess,
    HuberContactForceProcess,
    KcClosestPointStiffnessOutput,
    KcClosestPointStiffnessProcess,
    KcGeoStiffnessOutput,
    KcGeoStiffnessProcess,
    KcHermiteNonlocalStiffnessOutput,
    KcHermiteNonlocalStiffnessProcess,
    KcNormalStiffnessOutput,
    KcNormalStiffnessProcess,
    KcTermAssemblyInput,
)
from xkep_cae.core.testing import binds_to
from xkep_cae.mathematics import TermExpansionContract


def _make_pair(
    *,
    s: float = 0.5,
    t: float = 0.5,
    gap: float = -0.01,
    p_n: float = 1.0,
    radius_a: float = 0.05,
    radius_b: float = 0.05,
    nodes_a: tuple[int, int] = (0, 1),
    nodes_b: tuple[int, int] = (2, 3),
    normal: tuple[float, float, float] = (0.0, 1.0, 0.0),
    elem_a: int = 0,
    elem_b: int = 1,
) -> SimpleNamespace:
    """テスト用の接触ペアを生成."""
    state = SimpleNamespace(
        s=s,
        t=t,
        gap=gap,
        p_n=p_n,
        normal=np.array(normal, dtype=float),
    )
    return SimpleNamespace(
        state=state,
        nodes_a=nodes_a,
        nodes_b=nodes_b,
        radius_a=radius_a,
        radius_b=radius_b,
        elem_a=elem_a,
        elem_b=elem_b,
    )


@binds_to(KcNormalStiffnessProcess)
class TestKcNormalStiffnessProcess:
    """K_c 法線材料剛性項 K_mat = w_mat * (n⊗n) の単体検証."""

    def test_meta(self):
        assert KcNormalStiffnessProcess.meta.name == "KcNormalStiffness"
        assert KcNormalStiffnessProcess.meta.module == "solve"

    def test_contract_declared(self):
        """ClassVar contracts に TermExpansionContract が宣言されている."""
        contracts = KcNormalStiffnessProcess.contracts
        assert len(contracts) >= 1
        term_contracts = [c for c in contracts if isinstance(c, TermExpansionContract)]
        assert len(term_contracts) == 1
        tc = term_contracts[0]
        assert tc.name == "K_c_term_expansion"
        assert tc.total_name == "K_c"
        assert "KcNormalStiffnessProcess" in tc.providers

    def test_empty_pairs(self):
        proc = KcNormalStiffnessProcess()
        out = proc.process(
            KcTermAssemblyInput(
                pairs=[],
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        assert isinstance(out, KcNormalStiffnessOutput)
        assert out.K_mat.shape == (24, 24)
        assert out.K_mat.nnz == 0

    def test_inactive_pair_zero(self):
        """p_n=0 かつ gap>0 のペアは K_mat に寄与しない."""
        pair = _make_pair(gap=1.0, p_n=0.0)
        proc = KcNormalStiffnessProcess()
        out = proc.process(
            KcTermAssemblyInput(
                pairs=[pair],
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        assert out.K_mat.nnz == 0

    def test_single_pair_nonzero(self):
        """アクティブペアで K_mat が非ゼロかつ対称寄与を持つ."""
        pair = _make_pair(gap=-0.01, p_n=1.0)
        proc = KcNormalStiffnessProcess()
        out = proc.process(
            KcTermAssemblyInput(
                pairs=[pair],
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        K = out.K_mat
        assert K.shape == (24, 24)
        assert K.nnz > 0
        # w_mat > 0 なので 4 ノード (0,1,2,3) の y 方向対角成分が非ゼロ
        dense = K.toarray()
        for n in (0, 1, 2, 3):
            # n⊗n = diag(0,1,0) なので y 方向(1)対角のみ非ゼロ
            assert dense[n * 6 + 1, n * 6 + 1] > 0.0


@binds_to(KcGeoStiffnessProcess)
class TestKcGeoStiffnessProcess:
    """K_c 幾何補正項 K_geo = (p_n/d) * (I - n⊗n) の単体検証."""

    def test_meta(self):
        assert KcGeoStiffnessProcess.meta.name == "KcGeoStiffness"

    def test_contract_declared(self):
        contracts = KcGeoStiffnessProcess.contracts
        term_contracts = [c for c in contracts if isinstance(c, TermExpansionContract)]
        assert len(term_contracts) == 1
        tc = term_contracts[0]
        assert tc.name == "K_c_term_expansion"
        assert "KcGeoStiffnessProcess" in tc.providers

    def test_empty_pairs(self):
        proc = KcGeoStiffnessProcess()
        out = proc.process(
            KcTermAssemblyInput(
                pairs=[],
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        assert isinstance(out, KcGeoStiffnessOutput)
        assert out.K_geo.shape == (24, 24)
        assert out.K_geo.nnz == 0

    def test_inactive_pair_zero(self):
        pair = _make_pair(gap=1.0, p_n=0.0)
        proc = KcGeoStiffnessProcess()
        out = proc.process(
            KcTermAssemblyInput(
                pairs=[pair],
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        assert out.K_geo.nnz == 0

    def test_single_pair_nonzero(self):
        """アクティブペアで K_geo が接線成分 (I - n⊗n) を持つ."""
        pair = _make_pair(gap=-0.01, p_n=1.0, normal=(0.0, 1.0, 0.0))
        proc = KcGeoStiffnessProcess()
        out = proc.process(
            KcTermAssemblyInput(
                pairs=[pair],
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        K = out.K_geo
        assert K.shape == (24, 24)
        assert K.nnz > 0
        # I-n⊗n = diag(1,0,1) なので x,z 成分のみ寄与、y は寄与しない
        dense = K.toarray()
        for n in (0, 1, 2, 3):
            assert dense[n * 6 + 0, n * 6 + 0] > 0.0  # x 方向
            assert abs(dense[n * 6 + 1, n * 6 + 1]) < 1e-14  # y 方向（法線）
            assert dense[n * 6 + 2, n * 6 + 2] > 0.0  # z 方向


class TestKcTermExpansionContract:
    """TermExpansionContract（K_c_term_expansion）の静的検証 (status-351 5 項)."""

    def test_providers_registered_on_all_five_processes(self):
        """K_mat_nn / K_closest / K_hermite_adj / K_geo / K_st 5 Process が同一契約."""
        from xkep_cae.contact.contact_force import (
            ContactForceStStiffnessProcess,
            KcClosestPointStiffnessProcess,
            KcHermiteNonlocalStiffnessProcess,
        )

        for cls in (
            KcNormalStiffnessProcess,
            KcHermiteNonlocalStiffnessProcess,
            KcClosestPointStiffnessProcess,
            KcGeoStiffnessProcess,
            ContactForceStStiffnessProcess,
        ):
            term_contracts = [c for c in cls.contracts if isinstance(c, TermExpansionContract)]
            assert len(term_contracts) == 1, f"{cls.__name__} は TermExpansionContract を宣言すべき"
            assert term_contracts[0].name == "K_c_term_expansion"

    def test_contract_structure(self):
        """契約の骨格: 5 項 term_names + providers + combinator='add_sub'."""
        tc = [
            c for c in KcNormalStiffnessProcess.contracts if isinstance(c, TermExpansionContract)
        ][0]
        assert tc.term_names == ("K_mat_nn", "K_closest", "K_hermite_adj", "K_geo", "K_st")
        assert tc.providers == (
            "KcNormalStiffnessProcess",
            "KcClosestPointStiffnessProcess",
            "KcHermiteNonlocalStiffnessProcess",
            "KcGeoStiffnessProcess",
            "ContactForceStStiffnessProcess",
        )
        assert tc.combinator == "add_sub"
        assert tc.total_name == "K_c"
        assert tc.equation_ref.startswith("03_huber_contact_penalty.md")


@binds_to(KcHermiteNonlocalStiffnessProcess)
class TestKcHermiteNonlocalStiffnessProcess:
    """K_c Hermite 非局所材料剛性項 K_hermite_adj の単体検証 (status-351 Phase C-2)."""

    def test_meta(self):
        assert KcHermiteNonlocalStiffnessProcess.meta.name == "KcHermiteNonlocalStiffness"
        assert KcHermiteNonlocalStiffnessProcess.meta.module == "solve"

    def test_contract_declared(self):
        contracts = KcHermiteNonlocalStiffnessProcess.contracts
        term_contracts = [c for c in contracts if isinstance(c, TermExpansionContract)]
        assert len(term_contracts) == 1
        assert "KcHermiteNonlocalStiffnessProcess" in term_contracts[0].providers

    def test_empty_pairs(self):
        out = KcHermiteNonlocalStiffnessProcess().process(
            KcTermAssemblyInput(
                pairs=[],
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        assert isinstance(out, KcHermiteNonlocalStiffnessOutput)
        assert out.K_hermite_adj.shape == (24, 24)
        assert out.K_hermite_adj.nnz == 0

    def test_zero_when_adj_map_not_provided(self):
        """adj_node_map / adj_node_counts が None のとき K_hermite_adj はゼロ."""
        pair = _make_pair(gap=-0.01, p_n=1.0)
        out = KcHermiteNonlocalStiffnessProcess().process(
            KcTermAssemblyInput(
                pairs=[pair],
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
                adj_node_map=None,
                adj_node_counts=None,
            )
        )
        assert out.K_hermite_adj.nnz == 0

    def test_nonzero_with_adj_map(self):
        """隣接ノードが存在するとき K_hermite_adj が非ゼロで隣接 DOF 列に配置."""
        pair = _make_pair(gap=-0.01, p_n=1.0, elem_a=1, elem_b=4)
        # elem_a=1 → (node_prev, node_next) 隣接
        # elem_b=4 → (node_prev, node_next) 隣接
        adj_node_map = {1: (9, 10), 4: (11, 12)}  # 隣接ノード (ダミー)
        adj_node_counts = np.array(
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            dtype=int,  # 全ノード 2 要素参照
        )
        out = KcHermiteNonlocalStiffnessProcess().process(
            KcTermAssemblyInput(
                pairs=[pair],
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=13 * 6,  # 13 ノード * 6 DOF
                adj_node_map=adj_node_map,
                adj_node_counts=adj_node_counts,
            )
        )
        assert out.K_hermite_adj.nnz > 0


@binds_to(KcClosestPointStiffnessProcess)
class TestKcClosestPointStiffnessProcess:
    """K_c 最近接点追従項 K_closest の単体検証 (status-351 Phase C-2)."""

    def test_meta(self):
        assert KcClosestPointStiffnessProcess.meta.name == "KcClosestPointStiffness"

    def test_contract_declared(self):
        contracts = KcClosestPointStiffnessProcess.contracts
        term_contracts = [c for c in contracts if isinstance(c, TermExpansionContract)]
        assert len(term_contracts) == 1
        assert "KcClosestPointStiffnessProcess" in term_contracts[0].providers

    def test_empty_pairs(self):
        out = KcClosestPointStiffnessProcess().process(
            ContactForceStStiffnessInput(
                pairs=[],
                node_coords=np.zeros((4, 3)),
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        assert isinstance(out, KcClosestPointStiffnessOutput)
        assert out.K_closest.shape == (24, 24)
        assert out.K_closest.nnz == 0

    def test_inactive_pair_zero(self):
        """p_n=0 のペアは K_closest に寄与しない."""
        coords = np.zeros((4, 3))
        pair = _make_pair(p_n=0.0)
        out = KcClosestPointStiffnessProcess().process(
            ContactForceStStiffnessInput(
                pairs=[pair],
                node_coords=coords,
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        assert out.K_closest.nnz == 0

    def test_single_pair_nonzero(self):
        """近接平行セグメントで K_closest が非ゼロ."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [1.0, 0.1, 0.0],
            ]
        )
        pair = _make_pair(gap=-0.01, p_n=1.0)
        out = KcClosestPointStiffnessProcess().process(
            ContactForceStStiffnessInput(
                pairs=[pair],
                node_coords=coords,
                k_pen=1e4,
                delta_h=100.0,
                ndof_total=24,
            )
        )
        assert out.K_closest.shape == (24, 24)
        assert out.K_closest.nnz > 0


class TestTangentComponentsOrchestration:
    """tangent_components() が 5 Process の出力を orchestrate することを検証."""

    def _make_manager(self, pair) -> SimpleNamespace:
        return SimpleNamespace(
            pairs=[pair],
            connectivity=np.array([[0, 1], [2, 3]]),
            config=SimpleNamespace(
                use_hermite_centerline=False,
                consistent_st_tangent=False,
            ),
        )

    def test_orchestrator_matches_process_outputs(self):
        """tangent_components() 直接呼び出しと 5 Process 個別呼び出しが一致."""
        pair = _make_pair(gap=-0.01, p_n=1.0)
        manager = self._make_manager(pair)

        ndof = 24
        proc = HuberContactForceProcess(
            ndof=ndof,
            ndof_per_node=6,
            smoothing_delta=5000.0,
            huber_delta_h=100.0,
        )
        K_mat_o, K_geo_o, K_st_o = proc.tangent_components(np.zeros(ndof), manager, k_pen=1e4)

        # 同じ入力で Kc* を直接呼ぶ
        term_input = KcTermAssemblyInput(
            pairs=manager.pairs,
            k_pen=1e4,
            delta_h=100.0,
            ndof_total=ndof,
            ndof_per_node=6,
            penalty_exponent=1.0,
            use_hermite=False,
            node_counts=None,
            adj_node_map=None,
            adj_node_counts=None,
        )
        K_mat_nn = KcNormalStiffnessProcess().process(term_input).K_mat
        K_hermite_adj = KcHermiteNonlocalStiffnessProcess().process(term_input).K_hermite_adj
        K_geo_d = KcGeoStiffnessProcess().process(term_input).K_geo

        # orchestrator の K_mat は K_mat_nn + K_hermite_adj (adj_map=None なので 0)
        diff_mat = (K_mat_o - (K_mat_nn + K_hermite_adj)).toarray()
        diff_geo = (K_geo_o - K_geo_d).toarray()
        np.testing.assert_allclose(diff_mat, 0.0, atol=1e-14)
        np.testing.assert_allclose(diff_geo, 0.0, atol=1e-14)

        # K_st は consistent_st_tangent=False なのでゼロ
        assert K_st_o.nnz == 0

    def test_orchestrator_k_st_equals_kst_residual_plus_closest(self):
        """K_st (orchestrator) == K_st_residual + K_closest (直接呼び出し)."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [1.0, 0.1, 0.0],
            ]
        )
        pair = _make_pair(gap=-0.01, p_n=1.0)
        manager = SimpleNamespace(
            pairs=[pair],
            connectivity=np.array([[0, 1], [2, 3]]),
            config=SimpleNamespace(
                use_hermite_centerline=False,
                consistent_st_tangent=True,
            ),
        )
        ndof = 24
        proc = HuberContactForceProcess(
            ndof=ndof,
            ndof_per_node=6,
            smoothing_delta=5000.0,
            huber_delta_h=100.0,
        )
        _, _, K_st_o = proc.tangent_components(
            np.zeros(ndof), manager, k_pen=1e4, node_coords=coords
        )

        st_input = ContactForceStStiffnessInput(
            pairs=manager.pairs,
            node_coords=coords,
            k_pen=1e4,
            delta_h=100.0,
            ndof_total=ndof,
            ndof_per_node=6,
            use_hermite=False,
        )
        K_closest = KcClosestPointStiffnessProcess().process(st_input).K_closest
        K_st_res = ContactForceStStiffnessProcess().process(st_input).K_st
        expected = (K_closest + K_st_res).toarray()
        np.testing.assert_allclose(K_st_o.toarray(), expected, atol=1e-14)

    def test_empty_manager_returns_zeros(self):
        """manager.pairs が空のとき 3 ブロックとも zero."""
        manager = SimpleNamespace(pairs=[], config=SimpleNamespace())
        ndof = 12
        proc = HuberContactForceProcess(ndof=ndof, ndof_per_node=6)
        K_mat, K_geo, K_st = proc.tangent_components(np.zeros(ndof), manager, 1e4)
        assert K_mat.shape == (ndof, ndof)
        assert K_mat.nnz == 0
        assert K_geo.nnz == 0
        assert K_st.nnz == 0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
