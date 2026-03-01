"""曲げ揺動ベンチマーク テスト.

少数素線（7本）で曲げ揺動ベンチマークの動作確認と速度計測を行う。
Phase 2 は変位制御（z方向サイクル変位）。

[← README](../README.md)
"""

import time

import pytest

from xkep_cae.numerical_tests.wire_bending_benchmark import (
    BendingOscillationResult,
    print_benchmark_report,
    run_bending_oscillation,
)

pytestmark = pytest.mark.slow


class TestBendingOscillationSmall:
    """少数素線での曲げ揺動ベンチマーク動作確認."""

    def test_7_strand_lightweight(self):
        """7本撚線の曲げ揺動（軽量版: 45°, 1周期）."""
        result = run_bending_oscillation(
            n_strands=7,
            n_elems_per_strand=4,
            n_pitches=0.5,
            bend_angle_deg=45.0,
            n_bending_steps=5,
            oscillation_amplitude_mm=2.0,
            n_cycles=1,
            n_steps_per_quarter=2,
            max_iter=20,
            n_outer_max=3,
            tol_force=1e-4,
            show_progress=True,
        )

        assert isinstance(result, BendingOscillationResult)
        assert result.n_strands == 7
        assert result.n_elems == 7 * 4
        assert result.total_time_s > 0

        # Phase 2 結果が 1周期 = 4*2 = 8 ステップ
        assert len(result.phase2_results) == 8

        # スナップショットが記録されていること
        assert len(result.displacement_snapshots) > 0
        assert len(result.snapshot_labels) == len(result.displacement_snapshots)

        # タイミングデータの記録確認
        assert result.timing is not None
        totals = result.timing.phase_totals()
        assert len(totals) > 0
        assert "mesh_generation" in totals

        # レポート出力
        report = print_benchmark_report(result)
        assert "曲げ揺動ベンチマーク" in report

    def test_7_strand_full(self):
        """7本撚線の曲げ揺動ベンチマーク（フル版: 90°曲げ + ±3mm 1周期）.

        速度参照値として使用。
        """
        t0 = time.perf_counter()
        result = run_bending_oscillation(
            n_strands=7,
            n_elems_per_strand=8,
            n_pitches=1.0,
            bend_angle_deg=90.0,
            n_bending_steps=10,
            oscillation_amplitude_mm=3.0,
            n_cycles=1,
            n_steps_per_quarter=3,
            max_iter=30,
            n_outer_max=5,
            tol_force=1e-6,
            show_progress=True,
        )
        elapsed = time.perf_counter() - t0

        assert isinstance(result, BendingOscillationResult)
        assert result.total_time_s > 0

        # Phase 2: 1周期 × 4 × 3 = 12 ステップ
        assert len(result.phase2_results) == 12

        # タイミングデータの検証
        assert result.timing is not None
        totals = result.timing.phase_totals()
        assert "broadphase" in totals or "structural_tangent" in totals

        # 計時レポート
        print(f"\n[7本フルBM] 総計算時間: {elapsed:.2f} s")
        print(print_benchmark_report(result))


class TestBendingOscillationMedium:
    """中規模素線でのベンチマーク."""

    def test_19_strand(self):
        """19本撚線ベンチマーク（収束は保証しない）."""
        result = run_bending_oscillation(
            n_strands=19,
            n_elems_per_strand=4,
            n_pitches=0.5,
            bend_angle_deg=30.0,
            n_bending_steps=5,
            oscillation_amplitude_mm=2.0,
            n_cycles=1,
            n_steps_per_quarter=2,
            max_iter=15,
            n_outer_max=3,
            tol_force=1e-3,
            show_progress=True,
        )

        assert isinstance(result, BendingOscillationResult)
        assert result.n_strands == 19
        assert result.total_time_s > 0

        print(f"\n[19本BM] 総計算時間: {result.total_time_s:.2f} s")
        print(print_benchmark_report(result))

    def test_37_strand(self):
        """37本撚線ベンチマーク（収束は保証しない）."""
        result = run_bending_oscillation(
            n_strands=37,
            n_elems_per_strand=4,
            n_pitches=0.5,
            bend_angle_deg=20.0,
            n_bending_steps=5,
            oscillation_amplitude_mm=1.0,
            n_cycles=1,
            n_steps_per_quarter=2,
            max_iter=15,
            n_outer_max=3,
            tol_force=1e-3,
            show_progress=True,
        )

        assert isinstance(result, BendingOscillationResult)
        assert result.n_strands == 37
        assert result.total_time_s > 0

        print(f"\n[37本BM] 総計算時間: {result.total_time_s:.2f} s")
        print(print_benchmark_report(result))


class TestBendingOscillationTiming:
    """タイミング記録テスト."""

    def test_timing_phases_recorded(self):
        """タイミングの主要工程が記録されること."""
        result = run_bending_oscillation(
            n_strands=7,
            n_elems_per_strand=4,
            n_pitches=0.5,
            bend_angle_deg=30.0,
            n_bending_steps=3,
            oscillation_amplitude_mm=2.0,
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

        for phase in ["mesh_generation", "assembler_setup"]:
            assert totals[phase] > 0
            assert counts[phase] > 0

        assert result.timing.total_time() > 0

    def test_report_format(self):
        """レポート文字列が正しい形式で出力されること."""
        result = run_bending_oscillation(
            n_strands=7,
            n_elems_per_strand=4,
            n_pitches=0.5,
            bend_angle_deg=30.0,
            n_bending_steps=3,
            oscillation_amplitude_mm=2.0,
            n_cycles=1,
            n_steps_per_quarter=2,
            max_iter=15,
            tol_force=1e-3,
            show_progress=False,
        )

        report = print_benchmark_report(result)
        assert "曲げ揺動ベンチマーク" in report
        assert "要素数" in report
        assert "自由度数" in report
        assert "総計算時間" in report
        assert "Phase2ステップ数" in report


class TestBendingOscillationGIF:
    """GIF出力テスト."""

    def test_gif_output(self, tmp_path):
        """GIF出力が正しく生成されること."""
        result = run_bending_oscillation(
            n_strands=7,
            n_elems_per_strand=4,
            n_pitches=0.5,
            bend_angle_deg=30.0,
            n_bending_steps=3,
            oscillation_amplitude_mm=2.0,
            n_cycles=1,
            n_steps_per_quarter=2,
            max_iter=15,
            tol_force=1e-3,
            show_progress=False,
            gif_output_dir=str(tmp_path / "gif"),
            gif_snapshot_interval=2,
        )

        # スナップショットが記録されていること
        assert len(result.displacement_snapshots) > 0

        # GIF ファイルが生成されていること（matplotlib/PIL が利用可能な場合）
        try:
            import matplotlib  # noqa: F401
            from PIL import Image  # noqa: F401

            gif_dir = tmp_path / "gif"
            gif_files = list(gif_dir.glob("*.gif"))
            assert len(gif_files) > 0, "GIF ファイルが生成されていない"
            for gif_file in gif_files:
                assert gif_file.stat().st_size > 0
                assert "bending_oscillation" in gif_file.name
                assert "7strand" in gif_file.name
        except ImportError:
            pytest.skip("matplotlib/PIL が利用不可")
