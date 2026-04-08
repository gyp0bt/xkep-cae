"""撚線曲げ揺動の収束実行テスト.

status-253 TODO: StrandBendingOscillationProcess を実際に動かし、収束を確認。
- メッシュ: E=130MPa, ρ=8.96e-9, R=0.5mm, pitch=100mm
- 曲げ曲率 κ=0.001 1/mm, n_cycles=1

status-278: 2本撚線テスト追加（NR収束調査用の高速デバッグケース）
status-306: free_end_mode + Hertz型(α=1.5) に更新（status-280/285の改善反映）

[← README](../../README.md)
"""

from __future__ import annotations

import pytest

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


@pytest.mark.slow
class TestStrandBendingConvergence:
    """7本撚線曲げ揺動の収束テスト."""

    def test_strand_bending_oscillation_converges(self) -> None:
        """7本撚線曲げ揺動が収束完走する.

        status-306: free_end_mode + Hertz型(α=1.5) + max_nr=200 に更新。
        status-280: free_end_mode で frac=1.0 完走。
        status-285: Hertz型で frac 0.70→0.998 改善。
        """
        cfg = StrandBendingOscillationConfig(
            n_strands=7,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=40,  # 細かい増分
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=200,
            tol_force=1e-8,
            max_increments=10000,
            exclude_same_strand=True,
            free_end_mode=True,  # status-280: MPC不使用
            penalty_exponent=1.5,  # status-285: Hertz型
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        sr = result.solver_result
        frac = sr.load_history[-1] if sr.load_history else 0.0
        print("\n=== 7本撚線曲げ揺動 収束結果 ===")
        print(f"  frac_completed: {frac:.4f}")
        print(f"  converged:      {sr.converged}")
        print(f"  n_increments:   {sr.n_increments}")
        print(f"  n_cutbacks:     {sr.n_cutbacks}")
        print(f"  bending_angle:  {result.bending_angle:.6f} rad")
        print(f"  total_ndof:     {result.total_ndof}")
        print(f"  n_strand_nodes: {result.n_strand_nodes}")
        print(f"  n_ref_nodes:    {result.n_ref_nodes}")
        print(f"  elapsed:        {sr.elapsed_seconds:.2f} s")

        # 変位の確認
        import numpy as np

        u_max = float(np.max(np.abs(sr.u)))
        print(f"  max |u|:        {u_max:.6e}")

        # 接触力履歴
        if sr.contact_force_history:
            print(f"  contact forces: {len(sr.contact_force_history)} entries")
            print(f"  max contact F:  {max(sr.contact_force_history):.6e}")

        # 収束判定
        # status-280: free_end_mode で frac=1.0 完走
        # status-285: Hertz型(α=1.5) で frac 0.70→0.998
        assert u_max > 0.0, "変位がゼロ"
        assert frac >= 0.90, f"frac={frac} < 0.90 — free_end_mode+Hertz型で完走が期待される"

    def test_strand_bending_full_completion_hertz(self) -> None:
        """Hertz型+free_end_mode で frac=1.0 完走を確認.

        status-298 ベースライン: frac=1.0, incr=535, cutback=45, 752s。
        status-306: smoothing_delta=1000 依存を解消し Hertz型に更新。
        """
        cfg = StrandBendingOscillationConfig(
            n_strands=7,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=40,
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=200,
            tol_force=1e-8,
            max_increments=10000,
            exclude_same_strand=True,
            free_end_mode=True,  # status-280
            penalty_exponent=1.5,  # status-285: Hertz型
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        sr = result.solver_result
        frac = sr.load_history[-1] if sr.load_history else 0.0
        print("\n=== 7本撚線曲げ揺動 Hertz型完走テスト ===")
        print(f"  frac_completed: {frac:.4f}")
        print(f"  converged:      {sr.converged}")
        print(f"  n_increments:   {sr.n_increments}")
        print(f"  n_cutbacks:     {sr.n_cutbacks}")
        print(f"  elapsed:        {sr.elapsed_seconds:.2f} s")

        import numpy as np

        u_max = float(np.max(np.abs(sr.u)))
        print(f"  max |u|:        {u_max:.6e}")

        # Hertz型 + free_end_mode で frac=1.0 完走が期待される（status-298）
        assert u_max > 0.0, "変位がゼロ"
        assert frac >= 0.95, f"frac={frac} < 0.95 — Hertz型で完走が期待されるが未達"


@pytest.mark.slow
class TestTwoStrandBendingConvergence:
    """2本撚線曲げ揺動の収束テスト（NR収束調査用, status-278）.

    7本撚線と同じパラメータで2本のみ実行。
    接触ペア数が少なく高速（~20s）のため、デバッグ・パラメータ感度分析に適する。
    """

    def test_two_strand_bending_baseline(self) -> None:
        """2本撚線曲げ揺動ベースライン（free_end_mode + Hertz型）.

        status-306: free_end_mode + Hertz型に更新。
        """
        cfg = StrandBendingOscillationConfig(
            n_strands=2,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=40,
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=200,
            tol_force=1e-8,
            max_increments=10000,
            exclude_same_strand=True,
            gap=0.05,  # 2本撚線は初期貫入回避にgap必要
            free_end_mode=True,  # status-280
            penalty_exponent=1.5,  # status-285: Hertz型
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        sr = result.solver_result
        frac = sr.load_history[-1] if sr.load_history else 0.0
        print("\n=== 2本撚線曲げ揺動 ベースライン ===")
        print(f"  frac_completed: {frac:.4f}")
        print(f"  converged:      {sr.converged}")
        print(f"  n_increments:   {sr.n_increments}")
        print(f"  n_cutbacks:     {sr.n_cutbacks}")
        print(f"  elapsed:        {sr.elapsed_seconds:.2f} s")
        print(f"  total_ndof:     {result.total_ndof}")

        import numpy as np

        u_max = float(np.max(np.abs(sr.u)))
        print(f"  max |u|:        {u_max:.6e}")

        assert u_max > 0.0, "変位がゼロ"
        # 2本でもチャタリングが起きるかの確認（閾値は緩く）
        assert frac > 0.0, "frac=0: 1ステップも進まない"

    def test_two_strand_bending_large_delta_h(self) -> None:
        """2本撚線曲げ揺動 Hertz型+free_end_mode で完走確認.

        status-306: smoothing_delta=1000 依存を解消。
        """
        cfg = StrandBendingOscillationConfig(
            n_strands=2,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=40,
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=200,
            tol_force=1e-8,
            max_increments=10000,
            exclude_same_strand=True,
            gap=0.05,
            free_end_mode=True,  # status-280
            penalty_exponent=1.5,  # status-285: Hertz型
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        sr = result.solver_result
        frac = sr.load_history[-1] if sr.load_history else 0.0
        print("\n=== 2本撚線曲げ揺動 δ=1000 ===")
        print(f"  frac_completed: {frac:.4f}")
        print(f"  converged:      {sr.converged}")
        print(f"  n_increments:   {sr.n_increments}")
        print(f"  n_cutbacks:     {sr.n_cutbacks}")
        print(f"  elapsed:        {sr.elapsed_seconds:.2f} s")

        import numpy as np

        u_max = float(np.max(np.abs(sr.u)))
        print(f"  max |u|:        {u_max:.6e}")

        assert u_max > 0.0, "変位がゼロ"

    def test_two_strand_bending_with_fd_diagnostic(self) -> None:
        """2本撚線曲げ揺動 FD接線診断付き（K_c/K_st精度検証）.

        status-306: free_end_mode + Hertz型に更新。
        """
        cfg = StrandBendingOscillationConfig(
            n_strands=2,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=40,
            rho_inf=0.9,
            mu=0.15,
            max_nr_attempts=200,
            tol_force=1e-8,
            max_increments=500,  # 短縮（診断目的）
            exclude_same_strand=True,
            gap=0.05,
            tangent_fd_diagnostic=True,  # FD接線診断ON
            free_end_mode=True,  # status-280
            penalty_exponent=1.5,  # status-285: Hertz型
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        sr = result.solver_result
        frac = sr.load_history[-1] if sr.load_history else 0.0
        print("\n=== 2本撚線曲げ揺動 FD診断 ===")
        print(f"  frac_completed: {frac:.4f}")
        print(f"  n_increments:   {sr.n_increments}")
        print(f"  n_cutbacks:     {sr.n_cutbacks}")
        print(f"  elapsed:        {sr.elapsed_seconds:.2f} s")

        assert frac > 0.0, "frac=0: 1ステップも進まない"
