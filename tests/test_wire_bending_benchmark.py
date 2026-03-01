"""撚線曲げ + サイクル変位ベンチマーク テスト.

少数素線（3/7本）で撚線曲げベンチマークの動作確認と速度計測を行う。

[← README](../README.md)
"""

import time

import pytest

from xkep_cae.numerical_tests.wire_bending_benchmark import (
    WireBendingBenchmarkResult,
    print_benchmark_report,
    run_wire_bending_benchmark,
)

pytestmark = pytest.mark.slow


class TestWireBendingBenchmarkSmall:
    """少数素線での曲げベンチマーク動作確認."""

    def test_7_strand_bending_benchmark(self):
        """7本撚線の曲げ+サイクルベンチマーク（軽量版）."""
        result = run_wire_bending_benchmark(
            n_strands=7,
            n_elems_per_strand=4,
            n_pitches=0.5,
            bend_angle_deg=45.0,
            n_bending_steps=5,
            cyclic_amplitude_mm=2.0,
            n_cycles=1,
            n_steps_per_quarter=2,
            max_iter=20,
            n_outer_max=3,
            tol_force=1e-4,
            show_progress=True,
        )

        assert isinstance(result, WireBendingBenchmarkResult)
        assert result.n_strands == 7
        assert result.n_elems == 7 * 4
        assert result.ndof == 7 * 5 * 6  # 7 strands * 5 nodes * 6 DOF
        assert result.total_time_s > 0

        # タイミングデータの記録確認
        assert result.timing is not None
        totals = result.timing.phase_totals()
        assert len(totals) > 0
        assert "mesh_generation" in totals

        # レポート出力
        report = print_benchmark_report(result)
        assert "撚線曲げベンチマーク" in report

    def test_7_strand_full_benchmark(self):
        """7本撚線の曲げ+サイクルベンチマーク（フル版）.

        90°曲げ + ±5mm 2周期。ベンチマーク速度の参照値として使用。
        """
        t0 = time.perf_counter()
        result = run_wire_bending_benchmark(
            n_strands=7,
            n_elems_per_strand=8,
            n_pitches=1.0,
            bend_angle_deg=90.0,
            n_bending_steps=10,
            cyclic_amplitude_mm=5.0,
            n_cycles=2,
            n_steps_per_quarter=3,
            max_iter=30,
            n_outer_max=5,
            tol_force=1e-6,
            show_progress=True,
        )
        elapsed = time.perf_counter() - t0

        assert isinstance(result, WireBendingBenchmarkResult)
        assert result.total_time_s > 0

        # タイミングデータの検証
        assert result.timing is not None
        totals = result.timing.phase_totals()
        assert "broadphase" in totals or "structural_tangent" in totals

        # Phase 2 結果が4半周期分
        assert len(result.phase2_results) == 4

        # 計時レポート
        print(f"\n[7本撚線ベンチマーク] 総計算時間: {elapsed:.2f} s")
        print(print_benchmark_report(result))


class TestWireBendingBenchmarkMedium:
    """中規模素線でのベンチマーク（19/37本）."""

    def test_19_strand_benchmark(self):
        """19本撚線ベンチマーク（収束は保証しない）."""
        result = run_wire_bending_benchmark(
            n_strands=19,
            n_elems_per_strand=8,
            n_pitches=1.0,
            bend_angle_deg=45.0,
            n_bending_steps=10,
            cyclic_amplitude_mm=3.0,
            n_cycles=1,
            n_steps_per_quarter=3,
            max_iter=20,
            n_outer_max=3,
            tol_force=1e-4,
            show_progress=True,
        )

        assert isinstance(result, WireBendingBenchmarkResult)
        assert result.n_strands == 19
        assert result.total_time_s > 0

        print(f"\n[19本撚線ベンチマーク] 総計算時間: {result.total_time_s:.2f} s")
        print(print_benchmark_report(result))

    def test_37_strand_benchmark(self):
        """37本撚線ベンチマーク（収束は保証しない）."""
        result = run_wire_bending_benchmark(
            n_strands=37,
            n_elems_per_strand=8,
            n_pitches=1.0,
            bend_angle_deg=30.0,
            n_bending_steps=10,
            cyclic_amplitude_mm=2.0,
            n_cycles=1,
            n_steps_per_quarter=3,
            max_iter=20,
            n_outer_max=3,
            tol_force=1e-4,
            show_progress=True,
        )

        assert isinstance(result, WireBendingBenchmarkResult)
        assert result.n_strands == 37
        assert result.total_time_s > 0

        print(f"\n[37本撚線ベンチマーク] 総計算時間: {result.total_time_s:.2f} s")
        print(print_benchmark_report(result))


class TestWireBendingTimingReport:
    """ベンチマーク計時レポートのテスト."""

    def test_timing_phases_recorded(self):
        """タイミングの主要工程が記録されること."""
        result = run_wire_bending_benchmark(
            n_strands=7,
            n_elems_per_strand=4,
            n_pitches=0.5,
            bend_angle_deg=30.0,
            n_bending_steps=3,
            cyclic_amplitude_mm=2.0,
            n_cycles=1,
            n_steps_per_quarter=2,
            max_iter=15,
            n_outer_max=3,
            tol_force=1e-3,
            show_progress=False,
        )

        assert result.timing is not None
        totals = result.timing.phase_totals()
        counts = result.timing.phase_counts()

        # セットアップ工程
        assert "mesh_generation" in totals
        assert "assembler_setup" in totals

        # 各工程に正の時間
        for phase in ["mesh_generation", "assembler_setup"]:
            assert totals[phase] > 0, f"{phase} の時間が 0"
            assert counts[phase] > 0, f"{phase} の呼び出し回数が 0"

        # 合計時間が正
        assert result.timing.total_time() > 0

    def test_report_format(self):
        """レポート文字列が正しい形式で出力されること."""
        result = run_wire_bending_benchmark(
            n_strands=7,
            n_elems_per_strand=4,
            n_pitches=0.5,
            bend_angle_deg=30.0,
            n_bending_steps=3,
            n_cycles=1,
            n_steps_per_quarter=2,
            max_iter=15,
            tol_force=1e-3,
            show_progress=False,
        )

        report = print_benchmark_report(result)
        assert "撚線曲げベンチマーク" in report
        assert "要素数" in report
        assert "自由度数" in report
        assert "総計算時間" in report
