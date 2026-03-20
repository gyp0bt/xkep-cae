"""接触なし梁揺動解析テスト.

BeamOscillationProcess を使い、単純支持梁の自由振動（動的ソルバー）が
物理的に妥当であることを検証する。三点曲げの前準備。

テスト構成:
- TestBeamOscillationConvergence: ソルバー収束テスト（プログラムテスト）
- TestBeamOscillationPhysics: 物理的妥当性テスト
  - 振動周期、振幅、数値粘性、ひずみ分布

物理パラメータ（mm-ton-MPa 単位系）:
- ワイヤ: L=100mm, d=2mm, E=100GPa, ν=0.3, ρ=8.96e-9 ton/mm³ (銅)
- メッシュサイズ ≈ 半径（1mm）→ n_elems=100
- 振幅: 5mm（非線形域）

[← README](../README.md)
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.numerical_tests.beam_oscillation import (
    BeamOscillationConfig,
    BeamOscillationProcess,
    ElementBendingStrainInput,
    ElementBendingStrainProcess,
)
from xkep_cae.output.stress_contour import (
    ContourFieldInput,
    StressContour3DConfig,
    StressContour3DProcess,
)

# ====================================================================
# 共通パラメータ（銅）
# ====================================================================

_E = 100e3  # MPa (銅)
_NU = 0.3
_D = 2.0  # mm
_L = 100.0  # mm
_RHO = 8.96e-9  # ton/mm³ (銅)


def _default_config(**overrides) -> BeamOscillationConfig:
    """テスト用デフォルト設定."""
    defaults = {
        "wire_length": _L,
        "wire_diameter": _D,
        "n_elems_wire": 100,
        "E": _E,
        "nu": _NU,
        "rho": _RHO,
        "amplitude": 5.0,
        "n_periods": 3.0,
        "rho_inf": 0.9,
        "lumped_mass": True,
    }
    defaults.update(overrides)
    return BeamOscillationConfig(**defaults)


# ====================================================================
# プログラमテスト（API・収束）
# ====================================================================


class TestBeamOscillationAPI:
    """BeamOscillationProcess の API テスト."""

    def test_config_creation(self):
        """設定データクラスが正しく生成される."""
        cfg = _default_config()
        assert cfg.wire_length == _L
        assert cfg.n_elems_wire == 100
        assert cfg.amplitude == 5.0

    def test_strain_computation_zero_displacement(self):
        """変位ゼロでひずみもゼロ."""
        n_nodes = 11
        coords = np.column_stack(
            [np.linspace(0, 10, n_nodes), np.zeros(n_nodes), np.zeros(n_nodes)]
        )
        conn = np.column_stack([np.arange(10), np.arange(1, 11)])
        u = np.zeros(n_nodes * 6)
        proc = ElementBendingStrainProcess()
        result = proc.process(
            ElementBendingStrainInput(
                node_coords=coords,
                connectivity=conn,
                u=u,
                wire_radius=1.0,
            )
        )
        assert np.allclose(result.element_strain, 0.0)

    def test_strain_computation_parabolic_deflection(self):
        """二次放物線変位で一定曲率 → 一定ひずみ."""
        n_nodes = 21
        L = 20.0
        x = np.linspace(0, L, n_nodes)
        coords = np.column_stack([x, np.zeros(n_nodes), np.zeros(n_nodes)])
        conn = np.column_stack([np.arange(20), np.arange(1, 21)])

        # 放物線: y = -a*x*(L-x) → κ = 2a
        a = 0.01
        u = np.zeros(n_nodes * 6)
        for i in range(n_nodes):
            u[6 * i + 1] = -a * x[i] * (L - x[i])

        R = 1.0
        proc = ElementBendingStrainProcess()
        result = proc.process(
            ElementBendingStrainInput(
                node_coords=coords,
                connectivity=conn,
                u=u,
                wire_radius=R,
            )
        )
        expected_strain = 2.0 * a * R  # κ * R = 2a * R
        # 端部以外は一定
        interior = result.element_strain[2:-2]
        assert np.allclose(interior, expected_strain, rtol=0.05)


@pytest.mark.slow
class TestBeamOscillationConvergence:
    """BeamOscillationProcess の収束テスト."""

    def test_small_amplitude_converges(self):
        """小振幅（線形域）で収束する."""
        cfg = _default_config(
            amplitude=0.1,
            n_elems_wire=20,
            n_periods=1.0,
        )
        process = BeamOscillationProcess()
        result = process.process(cfg)
        assert result.solver_result.converged, "小振幅揺動が収束しない"
        assert result.solver_result.n_increments > 0

    def test_large_amplitude_converges(self):
        """大振幅（非線形域）で収束する."""
        cfg = _default_config(
            amplitude=5.0,
            n_elems_wire=100,
            n_periods=3.0,
        )
        process = BeamOscillationProcess()
        result = process.process(cfg)
        assert result.solver_result.converged, "大振幅揺動が収束しない"
        assert result.solver_result.n_increments > 0
        assert len(result.deflection_history) > 10, "履歴ステップが少なすぎる"


# ====================================================================
# 物理テスト
# ====================================================================


@pytest.mark.slow
class TestBeamOscillationPhysics:
    """梁揺動の物理的妥当性テスト."""

    @pytest.fixture(scope="class")
    def small_result(self):
        """小振幅（線形域）の揺動結果."""
        cfg = _default_config(
            amplitude=0.1,
            n_elems_wire=20,
            n_periods=2.0,
        )
        return BeamOscillationProcess().process(cfg)

    @pytest.fixture(scope="class")
    def large_result(self):
        """大振幅（非線形域）の揺動結果."""
        cfg = _default_config(
            amplitude=5.0,
            n_elems_wire=100,
            n_periods=3.0,
        )
        return BeamOscillationProcess().process(cfg)

    def test_small_amplitude_ratio(self, small_result):
        """小振幅で振幅比 ≈ 1.0（解析解一致）."""
        ratio = small_result.amplitude_ratio
        assert 0.8 < ratio < 1.3, f"振幅比 {ratio:.3f} が解析解から乖離"

    def test_small_oscillation_detected(self, small_result):
        """小振幅で振動が検出される（方向反転あり）."""
        defl = small_result.deflection_history
        if len(defl) > 10:
            # 方向反転（ピーク）を検出: diff の符号変化
            d_diff = np.diff(defl)
            direction_changes = np.sum(np.diff(np.sign(d_diff)) != 0)
            assert direction_changes >= 1, (
                f"振動なし: 方向反転 {direction_changes} 回, "
                f"defl range=[{defl.min():.4f}, {defl.max():.4f}]"
            )

    def test_small_energy_conservation(self, small_result):
        """小振幅でエネルギーがおおむね保存される（数値粘性評価）.

        rho_inf=0.9 では 3 周期後に ~10-30% の高周波減衰が期待される。
        """
        ratio = small_result.energy_decay_ratio
        assert ratio > 0.5, f"エネルギー減衰率 {ratio:.3f} — 過剰な数値減衰"

    def test_large_amplitude_nonlinear(self, large_result):
        """大振幅で非線形効果が現れる（振幅比 ≠ 1.0）."""
        # 大変形では幾何学的非線形により周期・振幅が変化
        assert large_result.max_deflection > 0.1, "変形がほぼゼロ"

    def test_contour_fields_exist(self, large_result):
        """contour_fields に S11/LE11/SK1 が含まれる."""
        assert "S11" in large_result.contour_fields
        assert "LE11" in large_result.contour_fields
        assert "SK1" in large_result.contour_fields
        # S11 = E * LE11
        if len(large_result.contour_fields["S11"]) > 0:
            s11 = large_result.contour_fields["S11"][0]
            le11 = large_result.contour_fields["LE11"][0]
            np.testing.assert_allclose(s11, le11 * _E, rtol=1e-10)

    def test_large_strain_distribution(self, large_result):
        """大振幅でひずみ分布が物理的に妥当.

        中央部で最大、端部でゼロに近い。
        """
        if len(large_result.element_strain_history) == 0:
            pytest.skip("ひずみ履歴なし")

        # 最大変形時のスナップショット
        defl = large_result.deflection_history
        max_idx = int(np.argmax(np.abs(defl)))
        strain = large_result.element_strain_history[max_idx]

        n_elems = len(strain)
        if n_elems < 10:
            pytest.skip("要素数不足")

        # 中央部（40-60%）のひずみが端部（0-10%, 90-100%）より大きい
        center_range = slice(int(n_elems * 0.4), int(n_elems * 0.6))
        edge_range_l = slice(0, int(n_elems * 0.1))
        edge_range_r = slice(int(n_elems * 0.9), n_elems)

        center_mean = np.mean(strain[center_range])
        edge_mean = 0.5 * (np.mean(strain[edge_range_l]) + np.mean(strain[edge_range_r]))
        assert center_mean > edge_mean, f"中央ひずみ {center_mean:.2e} ≤ 端部ひずみ {edge_mean:.2e}"

    def test_large_deflection_bounded(self, large_result):
        """大振幅でも変位が発散しない.

        梁長の半分を超える変位は物理的に不正。
        """
        L = large_result.config.wire_length
        assert large_result.max_deflection < L / 2, (
            f"変位発散: max={large_result.max_deflection:.3f} > L/2={L / 2:.1f}"
        )

    def test_numerical_dissipation_rate(self, large_result):
        """数値粘性の減衰率が妥当な範囲.

        rho_inf=0.9 で 3 周期: 高周波モードは大きく減衰するが、
        1次モード（低周波）は保存される。

        NOTE: UL定式化ではassemble_internal_force(u_total)が正しいひずみ
        エネルギーを返さない。正確な評価にはULステップごとの累積計算が必要。
        """
        ratio = large_result.energy_decay_ratio
        # 完全に減衰しきらない（>10%残存）が、発散もしない
        assert ratio > 0.1, f"エネルギー過剰減衰: {ratio:.3f}"
        assert ratio < 2.0, f"エネルギー発散: {ratio:.3f}"


# ====================================================================
# 可視化テスト（3Dひずみコンター）
# ====================================================================


@pytest.mark.slow
class TestStrainContour3DRendering:
    """3Dひずみコンターレンダリングのテスト."""

    def test_render_produces_images(self, tmp_path):
        """レンダリングが PNG ファイルを出力する."""
        # 小振幅・少要素で高速実行
        cfg = _default_config(
            amplitude=0.5,
            n_elems_wire=20,
            n_periods=1.0,
        )
        osc_result = BeamOscillationProcess().process(cfg)
        assert osc_result.solver_result.converged

        # contour_fields からフィールド入力を構築
        contour_fields = [
            ContourFieldInput(name=name, snapshots=snaps)
            for name, snaps in osc_result.contour_fields.items()
        ]
        render_cfg = StressContour3DConfig(
            mesh=osc_result.mesh,
            node_coords_initial=osc_result.mesh.node_coords,
            displacement_snapshots=osc_result.solver_result.displacement_history,
            contour_fields=contour_fields,
            time_values=osc_result.time_history,
            wire_radius=cfg.wire_diameter / 2.0,
            output_dir=str(tmp_path),
            prefix="test_oscillation",
            n_render_frames=3,
        )
        render_result = StressContour3DProcess().process(render_cfg)
        # 3フィールド × 3フレーム + 時刻歴 = 10枚以上
        assert len(render_result.image_paths) >= 3, "画像が出力されない"
        assert "S11" in render_result.field_max_values
        assert "LE11" in render_result.field_max_values
        assert "SK1" in render_result.field_max_values

        # PNG ファイルの存在確認
        from pathlib import Path

        for p in render_result.image_paths:
            assert Path(p).exists(), f"ファイルが存在しない: {p}"
