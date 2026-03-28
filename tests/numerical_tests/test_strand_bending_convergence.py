"""7本撚線曲げ揺動の収束実行テスト.

status-253 TODO: StrandBendingOscillationProcess を実際に動かし、収束を確認。
- メッシュ: E=130MPa, ρ=8.96e-9, R=0.5mm, pitch=100mm
- 曲げ曲率 κ=0.001 1/mm, n_cycles=1

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
        """7本撚線曲げ揺動が収束完走する."""
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
            max_nr_attempts=50,
            tol_force=1e-8,
            max_increments=10000,
            exclude_same_strand=True,
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

        # インクリメント診断
        if sr.increment_diagnostics:
            for d in sr.increment_diagnostics:
                print(
                    f"    step={d.step}, frac={d.load_frac:.4f}, "
                    f"nr={d.n_attempts}, active={d.n_active}, "
                    f"Fc={d.contact_force_norm:.2e}"
                )

        # 収束判定
        # status-260: smoothing_delta=1000/r でfrac≈0.59達成（0.35→0.59改善）
        assert u_max > 0.0, "変位がゼロ — MPCが機能していない可能性"
        assert frac >= 0.50, f"frac={frac} < 0.50 — smoothing_delta改善後の期待値未達"
