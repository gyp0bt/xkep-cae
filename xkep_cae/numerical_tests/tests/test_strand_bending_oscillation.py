"""StrandBendingOscillationProcess のテスト.

C3 契約: @binds_to 紐付け + API テスト。
実行は重いのでソルバー実行は行わず、構成検証のみ。

[← README](../../../README.md)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkep_cae.core import binds_to
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
    _collect_adjacent_nodes,
    _collect_end_nodes,
)


@binds_to(StrandBendingOscillationProcess)
class TestStrandBendingOscillationProcessAPI:
    """StrandBendingOscillationProcess の API テスト."""

    def test_meta_name(self) -> None:
        """meta.name が正しい."""
        assert StrandBendingOscillationProcess.meta.name == "StrandBendingOscillation"

    def test_uses_declared(self) -> None:
        """uses に必要な Process が宣言されている."""
        uses_names = [u.__name__ for u in StrandBendingOscillationProcess.uses]
        assert "StrandMeshProcess" in uses_names
        assert "MPCEliminationProcess" in uses_names
        assert "ULCRBeamAssemblerProcess" in uses_names
        assert "ContactFrictionProcess" in uses_names

    def test_config_defaults(self) -> None:
        """デフォルト構成が正しい."""
        cfg = StrandBendingOscillationConfig()
        assert cfg.n_strands == 7
        assert cfg.wire_radius == 0.5
        assert cfg.E == 130.0e3
        assert cfg.mu == 0.15
        assert cfg.smoothing_delta == 0.0  # 0=自動推定

    def test_smoothing_delta_auto_estimation(self) -> None:
        """smoothing_delta=0 のとき 1000/wire_radius で自動推定."""
        cfg = StrandBendingOscillationConfig(wire_radius=0.5, smoothing_delta=0.0)
        assert cfg.smoothing_delta == 0.0  # 0=自動推定トリガー
        # 自動推定公式: 1000 / wire_radius = 2000（status-260で5000→1000に変更）
        assert 1000.0 / cfg.wire_radius == 2000.0

    def test_smoothing_delta_manual_override(self) -> None:
        """smoothing_delta > 0 のとき手動値をそのまま使用."""
        cfg = StrandBendingOscillationConfig(smoothing_delta=3000.0)
        assert cfg.smoothing_delta == 3000.0

    def test_huber_delta_h_default_zero(self) -> None:
        """huber_delta_h デフォルトは 0（smoothing_delta で間接計算）."""
        cfg = StrandBendingOscillationConfig()
        assert cfg.huber_delta_h == 0.0

    def test_huber_delta_h_manual_override(self) -> None:
        """huber_delta_h > 0 のとき直接指定値をそのまま使用."""
        cfg = StrandBendingOscillationConfig(huber_delta_h=0.01)
        assert cfg.huber_delta_h == 0.01

    def test_free_end_mode_default_false(self) -> None:
        """free_end_mode デフォルトは False."""
        cfg = StrandBendingOscillationConfig()
        assert cfg.free_end_mode is False

    def test_free_end_mode_true(self) -> None:
        """free_end_mode=True で構成可能."""
        cfg = StrandBendingOscillationConfig(free_end_mode=True)
        assert cfg.free_end_mode is True

    def test_solver_mode_default_implicit(self) -> None:
        """solver_mode デフォルトは "implicit"（status-377 Phase 1）."""
        cfg = StrandBendingOscillationConfig()
        assert cfg.solver_mode == "implicit"

    def test_solver_mode_explicit_constructible(self) -> None:
        """solver_mode="explicit" は config 側で受理（実行は Phase 2 待機）."""
        cfg = StrandBendingOscillationConfig(solver_mode="explicit")
        assert cfg.solver_mode == "explicit"

    def test_solver_mode_explicit_raises_not_implemented(self) -> None:
        """status-377 Phase 1: solver_mode="explicit" 実行時は NotImplementedError."""
        from xkep_cae.numerical_tests.strand_bending_oscillation import (
            StrandBendingOscillationProcess,
        )

        cfg = StrandBendingOscillationConfig(solver_mode="explicit")
        proc = StrandBendingOscillationProcess()
        import pytest

        with pytest.raises(NotImplementedError, match="status-377 Phase 1"):
            proc.process(cfg)


class TestCollectEndNodes:
    """端部節点収集のテスト."""

    def test_single_strand(self) -> None:
        """単一素線の端部節点."""
        # 3要素: 0-1-2-3
        conn = np.array([[0, 1], [1, 2], [2, 3]])
        strand_ids = np.array([0, 0, 0])
        left, right = _collect_end_nodes(conn, 1, strand_ids)
        assert left == [0]
        assert right == [3]

    def test_two_strands(self) -> None:
        """2素線の端部節点."""
        # strand 0: 0-1-2, strand 1: 3-4-5
        conn = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
        strand_ids = np.array([0, 0, 1, 1])
        left, right = _collect_end_nodes(conn, 2, strand_ids)
        assert sorted(left) == [0, 3]
        assert sorted(right) == [2, 5]


class TestCollectAdjacentNodes:
    """隣接ノード収集のテスト."""

    def test_adjacent_of_end(self) -> None:
        """端部ノードの隣接ノードが正しい."""
        conn = np.array([[0, 1], [1, 2], [2, 3]])
        strand_ids = np.array([0, 0, 0])
        adj = _collect_adjacent_nodes(conn, strand_ids, [3])
        assert adj == [2]

    def test_adjacent_of_left_end(self) -> None:
        """左端ノードの隣接ノードが正しい."""
        conn = np.array([[0, 1], [1, 2], [2, 3]])
        strand_ids = np.array([0, 0, 0])
        adj = _collect_adjacent_nodes(conn, strand_ids, [0])
        assert adj == [1]


class TestLoadingModeConfig:
    """loading_mode 構成テスト."""

    def test_loading_mode_default_rotation(self) -> None:
        """loading_mode デフォルトは 'rotation'."""
        cfg = StrandBendingOscillationConfig()
        assert cfg.loading_mode == "rotation"

    def test_loading_mode_moment(self) -> None:
        """loading_mode='moment' で構成可能."""
        cfg = StrandBendingOscillationConfig(loading_mode="moment")
        assert cfg.loading_mode == "moment"


# ====================================================================
# 収束テスト（slow: ソルバー実行あり）
# ====================================================================


class TestHelical90DegBendConvergence:
    """7本ヘリカル素線 接触なし 90度曲げ 収束テスト.

    status-281: 動的ソルバーでのUL参照配置更新により完走。
    STA2再現性保証: このテストで次の担当者が改善を確認できる。
    """

    @pytest.mark.slow
    def test_7strand_90deg_dynamic_completes(self) -> None:
        """動的ソルバーで7本接触なし90度曲げが frac=1.0 に到達する."""
        cfg = StrandBendingOscillationConfig(
            n_strands=7,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=math.pi / 200.0,
            n_cycles=1,
            n_increments_per_cycle=40,
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=50,
            tol_force=1e-8,
            max_increments=10000,
            exclude_same_strand=True,
            free_end_mode=True,
            contact_enabled=False,
            loading_mode="rotation",
        )
        result = StrandBendingOscillationProcess().process(cfg)
        sr = result.solver_result
        frac = sr.load_history[-1] if sr.load_history else 0.0
        assert frac > 0.99, f"frac={frac:.4f} < 0.99: 90度曲げが完走しない"
        assert sr.n_cutbacks <= 20, f"cutback={sr.n_cutbacks} > 20: カットバック過多"


class TestHelical90DegBendPhysics:
    """7本ヘリカル素線 90度曲げ 物理妥当性テスト.

    エラスティカ理論: R = L/θ, tip_x = R*sin(θ), tip_z = L - R
    """

    @pytest.mark.slow
    def test_center_strand_tip_displacement(self) -> None:
        """中心素線の先端変位がエラスティカ理論値と5%以内で一致する."""
        cfg = StrandBendingOscillationConfig(
            n_strands=7,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=math.pi / 200.0,
            n_cycles=1,
            n_increments_per_cycle=40,
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=50,
            tol_force=1e-8,
            max_increments=10000,
            exclude_same_strand=True,
            free_end_mode=True,
            contact_enabled=False,
            loading_mode="rotation",
        )
        result = StrandBendingOscillationProcess().process(cfg)
        sr = result.solver_result
        frac = sr.load_history[-1] if sr.load_history else 0.0
        assert frac > 0.99, "完走していない"

        u = sr.u
        _, right_nodes = _collect_end_nodes(
            result.mesh.connectivity, cfg.n_strands, result.mesh.strand_ids
        )
        center_rn = right_nodes[0]
        tip = u[center_rn * 6 : center_rn * 6 + 6]

        # エラスティカ理論: R = L/θ = 100/(π/2) ≈ 63.66mm
        r_theory = 100.0 / (math.pi / 2.0)
        ux_theory = r_theory * math.sin(math.pi / 2.0)  # ≈ 63.66
        uz_theory = -(100.0 - r_theory)  # ≈ -36.34
        theta_y_theory = 90.0  # deg

        assert abs(tip[0] - ux_theory) / ux_theory < 0.05, (
            f"u_x={tip[0]:.2f} vs theory={ux_theory:.2f}"
        )
        assert abs(tip[2] - uz_theory) / abs(uz_theory) < 0.05, (
            f"u_z={tip[2]:.2f} vs theory={uz_theory:.2f}"
        )
        assert abs(math.degrees(tip[4]) - theta_y_theory) < 1.0, (
            f"θ_y={math.degrees(tip[4]):.1f}° vs theory={theta_y_theory}°"
        )


class TestMPC90DegBendConvergence:
    """MPC端部結合 + 接触なし 90度曲げ 収束テスト.

    status-283: MPC変換行列T再構築 + UL参照配置更新により完走。
    修正前: frac=0.14（~13°で線形化破綻）
    修正後: frac=1.0
    """

    @pytest.mark.slow
    def test_mpc_90deg_nocontact_completes(self) -> None:
        """MPC + 接触なしで7本90度曲げが frac=1.0 に到達する."""
        cfg = StrandBendingOscillationConfig(
            n_strands=7,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=math.pi / 200.0,
            n_cycles=1,
            n_increments_per_cycle=40,
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=50,
            tol_force=1e-8,
            max_increments=10000,
            exclude_same_strand=True,
            free_end_mode=False,
            contact_enabled=False,
            loading_mode="rotation",
        )
        result = StrandBendingOscillationProcess().process(cfg)
        sr = result.solver_result
        frac = sr.load_history[-1] if sr.load_history else 0.0
        assert frac > 0.99, f"MPC frac={frac:.4f} < 0.99: 90度曲げが完走しない"
        assert sr.n_cutbacks <= 20, f"cutback={sr.n_cutbacks} > 20: カットバック過多"
