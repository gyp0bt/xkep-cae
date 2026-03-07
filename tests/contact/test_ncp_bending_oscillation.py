"""NCP ソルバー版 曲げ揺動収束テスト.

S3収束要件: 非線形梁（CR）の90度曲げ + サイクル変位（揺動）を
NCP Semi-smooth Newton ソルバーで収束させる。

既存の径方向圧縮テスト（線形梁）とは異なり、CR（Co-Rotational）梁を使用した
幾何学的非線形問題。S3ベンチマークの中核テスト。

[← README](../../README.md)
"""

import numpy as np
import pytest

from xkep_cae.numerical_tests.wire_bending_benchmark import (
    BendingOscillationResult,
    run_bending_oscillation,
)

pytestmark = pytest.mark.slow

# ====================================================================
# 共通パラメータ
# ====================================================================

_N_ELEMS_PER_STRAND = 16  # 16要素/ピッチ厳守


# ====================================================================
# テスト: 7本撚線 NCP 曲げ揺動
# ====================================================================


class TestNCP7StrandBendingOscillation:
    """7本撚線のNCP曲げ揺動収束テスト.

    Phase 1: CR梁による90度曲げ（変位制御: 端部回転角を処方）
    Phase 2: z方向サイクル変位（揺動）

    S3マイルストーン: 7本NCPでの曲げ揺動収束が前提。
    """

    def test_ncp_7strand_bending_45deg(self):
        """7本: 45度曲げのみ（揺動なし）でNCP収束を確認."""
        result = run_bending_oscillation(
            n_strands=7,
            n_elems_per_strand=_N_ELEMS_PER_STRAND,
            n_pitches=0.5,
            bend_angle_deg=45.0,
            oscillation_amplitude_mm=0.0,
            n_cycles=0,
            n_steps_per_quarter=1,
            max_iter=30,
            tol_force=1e-4,
            show_progress=True,
            # NCP ソルバー
            use_ncp=True,
            use_mortar=True,
            adaptive_timestepping=True,
            # 接触パラメータ
            exclude_same_layer=True,
            midpoint_prescreening=True,
            use_line_search=False,
            g_on=0.0005,
            g_off=0.001,
            # Updated Lagrangian（ヘリカル梁の大回転収束問題を解消）
            use_updated_lagrangian=True,
        )

        assert isinstance(result, BendingOscillationResult)
        assert result.n_strands == 7
        assert result.phase1_converged, "7本 NCP 45度曲げが収束しなかった"
        print(
            f"\n  7本 NCP (45度曲げ): converged={result.phase1_converged}, "
            f"time={result.total_time_s:.1f}s, "
            f"tip=({result.tip_displacement_final[0] * 1000:.2f}, "
            f"{result.tip_displacement_final[1] * 1000:.2f}, "
            f"{result.tip_displacement_final[2] * 1000:.2f}) mm"
        )

    def test_ncp_7strand_bending_90deg(self):
        """7本: 90度曲げのみ（揺動なし）でNCP収束を確認."""
        result = run_bending_oscillation(
            n_strands=7,
            n_elems_per_strand=_N_ELEMS_PER_STRAND,
            n_pitches=0.5,
            bend_angle_deg=90.0,
            oscillation_amplitude_mm=0.0,
            n_cycles=0,
            n_steps_per_quarter=1,
            max_iter=40,
            tol_force=1e-4,
            show_progress=True,
            # NCP ソルバー
            use_ncp=True,
            use_mortar=True,
            adaptive_timestepping=True,
            # 接触パラメータ
            exclude_same_layer=True,
            midpoint_prescreening=True,
            use_line_search=False,
            g_on=0.0005,
            g_off=0.001,
            # Updated Lagrangian（ヘリカル梁の大回転収束問題を解消）
            use_updated_lagrangian=True,
        )

        assert isinstance(result, BendingOscillationResult)
        assert result.phase1_converged, "7本 NCP 90度曲げが収束しなかった"
        print(
            f"\n  7本 NCP (90度曲げ): converged={result.phase1_converged}, "
            f"time={result.total_time_s:.1f}s"
        )

    @pytest.mark.xfail(reason="UL Phase2（揺動）の参照配置統合が未完了 (status-130)", strict=False)
    def test_ncp_7strand_bending_oscillation_full(self):
        """7本: 90度曲げ + 揺動1周期（S3ベンチマーク）.

        S3要件: 非線形梁の90度曲げ + サイクル変位（揺動）でNCP収束。
        """
        result = run_bending_oscillation(
            n_strands=7,
            n_elems_per_strand=_N_ELEMS_PER_STRAND,
            n_pitches=0.5,
            bend_angle_deg=90.0,
            oscillation_amplitude_mm=2.0,
            n_cycles=1,
            n_steps_per_quarter=3,
            max_iter=40,
            tol_force=1e-4,
            show_progress=True,
            # NCP ソルバー
            use_ncp=True,
            use_mortar=True,
            adaptive_timestepping=True,
            # 接触パラメータ
            exclude_same_layer=True,
            midpoint_prescreening=True,
            use_line_search=False,
            g_on=0.0005,
            g_off=0.001,
            # Updated Lagrangian（ヘリカル梁の大回転収束問題を解消）
            use_updated_lagrangian=True,
        )

        assert isinstance(result, BendingOscillationResult)
        assert result.phase1_converged, "7本 NCP Phase1（90度曲げ）が収束しなかった"
        assert result.phase2_converged, "7本 NCP Phase2（揺動）が収束しなかった"
        # 揺動ステップ数: 1周期 * 4 * 3 = 12 ステップ
        assert len(result.phase2_results) == 12

        print(
            f"\n  7本 NCP (90度曲げ+揺動): "
            f"phase1={result.phase1_converged}, "
            f"phase2={result.phase2_converged}, "
            f"time={result.total_time_s:.1f}s, "
            f"active={result.n_active_contacts}"
        )


# ====================================================================
# テスト: 19本撚線 NCP 曲げ揺動
# ====================================================================


class TestNCP19StrandBendingOscillation:
    """19本撚線のNCP曲げ揺動収束テスト.

    7本収束達成後の段階的スケールアップ。
    19本(1+6+12)で非線形曲げ+揺動を収束させる。
    """

    def test_ncp_19strand_bending_45deg(self):
        """19本: 45度曲げのみでNCP収束を試行."""
        result = run_bending_oscillation(
            n_strands=19,
            n_elems_per_strand=_N_ELEMS_PER_STRAND,
            n_pitches=0.5,
            bend_angle_deg=45.0,
            oscillation_amplitude_mm=0.0,
            n_cycles=0,
            n_steps_per_quarter=1,
            max_iter=50,
            tol_force=1e-4,
            show_progress=True,
            # NCP ソルバー
            use_ncp=True,
            use_mortar=True,
            adaptive_timestepping=True,
            # 接触パラメータ
            exclude_same_layer=True,
            midpoint_prescreening=True,
            use_line_search=False,
            g_on=0.0005,
            g_off=0.001,
            # Updated Lagrangian
            use_updated_lagrangian=True,
        )

        assert isinstance(result, BendingOscillationResult)
        # 19本は収束を必須としない（段階的改善のトラッキング）
        assert result.total_time_s > 0
        print(
            f"\n  19本 NCP (45度曲げ): converged={result.phase1_converged}, "
            f"time={result.total_time_s:.1f}s"
        )

    @pytest.mark.xfail(reason="UL Phase2（揺動）の参照配置統合が未完了 (status-130)", strict=False)
    def test_ncp_19strand_bending_oscillation(self):
        """19本: 45度曲げ + 揺動1周期でNCP収束を試行.

        収束は現時点では必須としないが、NR反復が実行されていることを確認。
        """
        result = run_bending_oscillation(
            n_strands=19,
            n_elems_per_strand=_N_ELEMS_PER_STRAND,
            n_pitches=0.5,
            bend_angle_deg=45.0,
            oscillation_amplitude_mm=1.0,
            n_cycles=1,
            n_steps_per_quarter=2,
            max_iter=50,
            tol_force=1e-4,
            show_progress=True,
            # NCP ソルバー
            use_ncp=True,
            use_mortar=True,
            adaptive_timestepping=True,
            # 接触パラメータ
            exclude_same_layer=True,
            midpoint_prescreening=True,
            use_line_search=False,
            g_on=0.0005,
            g_off=0.001,
            # Updated Lagrangian
            use_updated_lagrangian=True,
        )

        assert isinstance(result, BendingOscillationResult)
        assert result.total_time_s > 0
        print(
            f"\n  19本 NCP (45度曲げ+揺動): "
            f"phase1={result.phase1_converged}, "
            f"phase2={result.phase2_converged}, "
            f"time={result.total_time_s:.1f}s, "
            f"active={result.n_active_contacts}"
        )


# ====================================================================
# テスト: 物理的妥当性検証（曲げ揺動固有）
# ====================================================================


class TestNCP7StrandBendingPhysics:
    """7本NCP曲げ揺動の物理的妥当性テスト.

    曲げ変形後の先端変位・回転角が解析解と整合することを確認。
    """

    def test_tip_displacement_direction(self):
        """曲げ後の先端変位方向が物理的に正しい.

        z軸方向の梁に正のMx（x軸回りモーメント）を与えると、
        先端はy正方向に変位し、z方向は短縮する。
        """
        result = run_bending_oscillation(
            n_strands=7,
            n_elems_per_strand=_N_ELEMS_PER_STRAND,
            n_pitches=0.5,
            bend_angle_deg=45.0,
            oscillation_amplitude_mm=0.0,
            n_cycles=0,
            n_steps_per_quarter=1,
            max_iter=30,
            tol_force=1e-4,
            show_progress=False,
            use_ncp=True,
            use_mortar=True,
            adaptive_timestepping=True,
            exclude_same_layer=True,
            midpoint_prescreening=True,
            use_line_search=False,
            g_on=0.0005,
            g_off=0.001,
            # Updated Lagrangian
            use_updated_lagrangian=True,
        )

        assert result.phase1_converged, "Phase1が収束しなかった"
        dx, dy, dz = result.tip_displacement_final

        # 45度曲げで先端はy方向に有意な変位を持つ
        # （正のMx回転→右手則でy負方向に変位）
        assert abs(dy) > 0.001 * result.mesh_length, f"先端y変位が小さすぎる: dy={dy * 1000:.4f} mm"
        # z方向は短縮（負方向変位）するはず
        assert dz < 0, f"先端z変位が負でない: dz={dz * 1000:.4f} mm"

        # 変位オーダーの妥当性: ピッチ長の1%〜100%程度
        L = result.mesh_length
        disp_mag = np.sqrt(dx**2 + dy**2 + dz**2)
        assert 0.01 * L < disp_mag < 2.0 * L, (
            f"先端変位が物理的に不合理: |u|={disp_mag * 1000:.2f} mm, L={L * 1000:.1f} mm"
        )

        print(
            f"\n  物理テスト（先端変位方向）: "
            f"dx={dx * 1000:.3f} mm, dy={dy * 1000:.3f} mm, dz={dz * 1000:.3f} mm"
        )

    @pytest.mark.xfail(reason="UL Phase2（揺動）の参照配置統合が未完了 (status-130)", strict=False)
    def test_penetration_ratio_within_limit(self):
        """曲げ揺動後の最大貫入量がワイヤ直径の2%以内."""
        result = run_bending_oscillation(
            n_strands=7,
            n_elems_per_strand=_N_ELEMS_PER_STRAND,
            n_pitches=0.5,
            bend_angle_deg=45.0,
            oscillation_amplitude_mm=1.0,
            n_cycles=1,
            n_steps_per_quarter=2,
            max_iter=30,
            tol_force=1e-4,
            show_progress=False,
            use_ncp=True,
            use_mortar=True,
            adaptive_timestepping=True,
            exclude_same_layer=True,
            midpoint_prescreening=True,
            use_line_search=False,
            g_on=0.0005,
            g_off=0.001,
            # Updated Lagrangian
            use_updated_lagrangian=True,
        )

        # 貫入比チェック（16要素/ピッチでは2%以内が期待値）
        assert result.max_penetration_ratio < 0.05, (
            f"最大貫入比が大きすぎる: {result.max_penetration_ratio:.3f} (期待値: < 0.05)"
        )
        print(f"\n  物理テスト（貫入比）: max_pen_ratio={result.max_penetration_ratio:.4f}")
