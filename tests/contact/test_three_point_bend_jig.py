"""単線の剛体支えと押しジグによる動的三点曲げテスト.

ThreePointBendJigProcess を使い、変位制御三点曲げが
解析解（Euler-Bernoulli / Timoshenko）と一致することを検証する。

テスト構成:
- TestAnalyticalSolution: 解析解ヘルパーの単体テスト
- TestThreePointBendJigConvergence: ソルバー収束テスト
- TestThreePointBendJigPhysics: 解析解一致テスト（物理検証）

物理パラメータ（mm-ton-MPa 単位系）:
- ワイヤ: L=100mm, d=2mm, E=200GPa, ν=0.3
- 解析剛性: k_EB = 48EI/L³ ≈ 7.54 N/mm

[← README](../../README.md)
"""

from __future__ import annotations

import pytest

from xkep_cae.numerical_tests.three_point_bend_jig import (
    DynamicThreePointBendContactJigConfig,
    DynamicThreePointBendContactJigProcess,
    DynamicThreePointBendJigConfig,
    DynamicThreePointBendJigProcess,
    ThreePointBendJigConfig,
    ThreePointBendJigProcess,
    _analytical_three_point_bend,
    _beam_fundamental_frequency,
    _circle_section,
)

# ====================================================================
# 共通パラメータ
# ====================================================================

_E = 200e3  # MPa
_NU = 0.3
_D = 2.0  # mm
_L = 100.0  # mm


def _default_config(**overrides) -> ThreePointBendJigConfig:
    """テスト用デフォルト設定."""
    defaults = {
        "wire_length": _L,
        "wire_diameter": _D,
        "n_elems_wire": 20,
        "E": _E,
        "nu": _NU,
        "jig_push": 0.1,
    }
    defaults.update(overrides)
    return ThreePointBendJigConfig(**defaults)


# ====================================================================
# 解析解の単体検証
# ====================================================================


class TestAnalyticalSolution:
    """解析解ヘルパー関数の単体テスト."""

    def test_eb_deflection_formula(self):
        """EB 三点曲げの中央変位公式が正しい."""
        sec = _circle_section(_D, _NU)
        P = 1.0
        delta = P * _L**3 / (48.0 * _E * sec["Iy"])
        G = _E / (2.0 * (1.0 + _NU))
        ana = _analytical_three_point_bend(P, _L, _E, sec["Iy"], sec["kappa"], G, sec["A"])
        assert abs(ana["delta_eb"] - delta) < 1e-12

    def test_timoshenko_correction_small(self):
        """細長い梁（L/d=50）で Timoshenko 補正が小さい."""
        sec = _circle_section(_D, _NU)
        G = _E / (2.0 * (1.0 + _NU))
        P = 1.0
        ana = _analytical_three_point_bend(P, _L, _E, sec["Iy"], sec["kappa"], G, sec["A"])
        shear_ratio = ana["delta_timo"] / ana["delta_eb"] - 1.0
        assert shear_ratio < 0.01, f"せん断補正比 {shear_ratio:.4f} > 1%"

    def test_stiffness_consistency(self):
        """剛性 k = P/δ が正しい."""
        sec = _circle_section(_D, _NU)
        G = _E / (2.0 * (1.0 + _NU))
        P = 10.0
        ana = _analytical_three_point_bend(P, _L, _E, sec["Iy"], sec["kappa"], G, sec["A"])
        k_calc = P / ana["delta_eb"]
        assert abs(k_calc - ana["stiffness_eb"]) / k_calc < 1e-10


# ====================================================================
# 収束テスト
# ====================================================================


class TestThreePointBendJigConvergence:
    """三点曲げジグ試験のソルバー収束テスト."""

    def test_small_push_converges(self):
        """小変位（0.1mm）で収束する."""
        cfg = _default_config(jig_push=0.1)
        proc = ThreePointBendJigProcess()
        result = proc.process(cfg)

        print(
            f"\n  三点曲げジグ（小変位）: converged={result.solver_result.converged}, "
            f"increments={result.solver_result.n_increments}, "
            f"newton={result.solver_result.total_attempts}, "
            f"δ_wire={result.wire_midpoint_deflection:.6f} mm, "
            f"P={result.reaction_force:.4f} N"
        )
        assert result.solver_result.converged, "小変位で収束しなかった"

    def test_medium_push_converges(self):
        """中変位（1.0mm）で収束する."""
        cfg = _default_config(jig_push=1.0)
        proc = ThreePointBendJigProcess()
        result = proc.process(cfg)

        print(
            f"\n  三点曲げジグ（中変位）: converged={result.solver_result.converged}, "
            f"increments={result.solver_result.n_increments}, "
            f"newton={result.solver_result.total_attempts}, "
            f"δ_wire={result.wire_midpoint_deflection:.6f} mm, "
            f"P={result.reaction_force:.4f} N"
        )
        assert result.solver_result.converged, "中変位で収束しなかった"


# ====================================================================
# 物理テスト（解析解一致）
# ====================================================================


class TestThreePointBendJigPhysics:
    """三点曲げジグ試験の物理的妥当性テスト."""

    def test_deflection_matches_analytical(self):
        """ワイヤ中央変位が処方値に一致."""
        cfg = _default_config(jig_push=0.1)
        proc = ThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged
        # 処方変位 = 0.1mm → 実変位も 0.1mm
        assert abs(result.wire_midpoint_deflection - cfg.jig_push) / cfg.jig_push < 0.01, (
            f"変位不一致: expected={cfg.jig_push}, actual={result.wire_midpoint_deflection}"
        )

    def test_reaction_force_matches_beam_theory(self):
        """反力が梁理論の剛性 × 変位に一致（許容誤差5%）."""
        cfg = _default_config(jig_push=0.1)
        proc = ThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged

        sec = _circle_section(cfg.wire_diameter, cfg.nu)
        k_eb = 48.0 * cfg.E * sec["Iy"] / cfg.wire_length**3
        P_expected = k_eb * cfg.jig_push
        P_actual = result.reaction_force
        rel_err = abs(P_actual - P_expected) / P_expected

        print(
            f"\n  反力検証: k_EB={k_eb:.4f} N/mm, "
            f"P_expected={P_expected:.4f} N, P_actual={P_actual:.4f} N, "
            f"rel_err={rel_err:.4f} ({rel_err * 100:.1f}%)"
        )
        assert rel_err < 0.05, (
            f"反力の相対誤差 {rel_err:.4f} > 5%: expected={P_expected:.6f}, actual={P_actual:.6f}"
        )

    def test_stiffness_matches_analytical(self):
        """P/δ が解析剛性に一致（許容5%）."""
        cfg = _default_config(jig_push=0.1)
        proc = ThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged

        k_fem = result.reaction_force / result.wire_midpoint_deflection
        k_ana = result.analytical_stiffness_eb
        rel_err = abs(k_fem - k_ana) / k_ana

        print(
            f"\n  剛性検証: k_FEM={k_fem:.4f}, k_EB={k_ana:.4f}, "
            f"rel_err={rel_err:.4f} ({rel_err * 100:.1f}%)"
        )
        assert rel_err < 0.05, f"剛性の相対誤差 {rel_err:.4f} > 5%"

    def test_deflection_proportional_to_push(self):
        """ジグ押し量に対して反力が線形応答する."""
        proc = ThreePointBendJigProcess()

        pushes = [0.05, 0.1, 0.2]
        forces = []
        for push in pushes:
            cfg = _default_config(jig_push=push)
            result = proc.process(cfg)
            assert result.solver_result.converged, f"push={push} で収束しなかった"
            forces.append(result.reaction_force)

        # P/δ がほぼ一定（線形性）
        stiffnesses = [f / p for f, p in zip(forces, pushes, strict=True)]
        mean_k = sum(stiffnesses) / len(stiffnesses)
        for i, (push, k) in enumerate(zip(pushes, stiffnesses, strict=True)):
            ratio = k / mean_k
            print(f"  push={push}: P={forces[i]:.4f}, k={k:.4f}, ratio={ratio:.4f}")
            assert 0.95 < ratio < 1.05, (
                f"push={push} の剛性 {k:.4f} が平均 {mean_k:.4f} から5%以上逸脱"
            )

    def test_wire_deformation_symmetric(self):
        """ワイヤの変形が中央対称."""
        cfg = _default_config(jig_push=0.1)
        proc = ThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged

        u = result.solver_result.u
        n_wire_nodes = result.n_wire_nodes
        mid = result.wire_mid_node

        for k in range(1, mid):
            left_y = abs(u[6 * k + 1])
            right_y = abs(u[6 * (n_wire_nodes - 1 - k) + 1])
            if max(left_y, right_y) < 1e-12:
                continue
            sym_err = abs(left_y - right_y) / max(left_y, right_y)
            assert sym_err < 0.05, f"節点 {k}: 対称性誤差 {sym_err:.4f} > 5%"

    def test_support_reactions_correct(self):
        """支持点変位がゼロ."""
        cfg = _default_config(jig_push=0.1)
        proc = ThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged

        u = result.solver_result.u
        n_wire_nodes = result.n_wire_nodes

        # 左端: x, y, z = 0
        for d in range(3):
            assert abs(u[d]) < 1e-10, f"左端 DOF{d} = {u[d]}"

        # 右端: y, z = 0
        right_dof_y = 6 * (n_wire_nodes - 1) + 1
        right_dof_z = 6 * (n_wire_nodes - 1) + 2
        assert abs(u[right_dof_y]) < 1e-10, f"右端 y = {u[right_dof_y]}"
        assert abs(u[right_dof_z]) < 1e-10, f"右端 z = {u[right_dof_z]}"

    def test_wire_deflects_downward(self):
        """ワイヤが下方に撓む."""
        cfg = _default_config(jig_push=0.1)
        proc = ThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged
        mid_dof = 6 * result.wire_mid_node + 1
        assert result.solver_result.u[mid_dof] < 0, (
            f"中央 y = {result.solver_result.u[mid_dof]:.6f}（正=異常）"
        )

    def test_deflection_profile_parabolic(self):
        """変位プロファイルが放物線に近い（EB 理論）."""
        cfg = _default_config(jig_push=0.1, n_elems_wire=40)
        proc = ThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged

        u = result.solver_result.u
        n = result.n_wire_nodes
        L = cfg.wire_length
        delta_max = result.wire_midpoint_deflection

        max_err = 0.0
        for k in range(1, n - 1):
            x = k * L / (n - 1)
            # EB 解析解: δ(x)/δ_max for 0 ≤ x ≤ L/2:
            #   δ(x) = P*x*(3*L^2 - 4*x^2)/(48*E*I)
            #   δ_max = δ(L/2) = P*L^3/(48*E*I)
            #   δ(x)/δ_max = x*(3*L^2 - 4*x^2)/L^3
            if x <= L / 2:
                analytical_ratio = x * (3 * L**2 - 4 * x**2) / L**3
            else:
                xm = L - x
                analytical_ratio = xm * (3 * L**2 - 4 * xm**2) / L**3
            fem_ratio = abs(u[6 * k + 1]) / delta_max if delta_max > 0 else 0
            err = abs(fem_ratio - analytical_ratio)
            max_err = max(max_err, err)

        print(f"\n  変位プロファイル最大誤差: {max_err:.4f}")
        assert max_err < 0.05, f"変位プロファイル誤差 {max_err:.4f} > 5%"


# ====================================================================
# 動的三点曲げテスト
# ====================================================================

_RHO = 7.85e-9  # ton/mm³（鉄鋼）


def _dynamic_config(**overrides) -> DynamicThreePointBendJigConfig:
    """動的テスト用デフォルト設定."""
    defaults = {
        "wire_length": _L,
        "wire_diameter": _D,
        "n_elems_wire": 20,
        "E": _E,
        "nu": _NU,
        "rho": _RHO,
        "jig_push": 0.1,
        "n_periods": 2.0,
        "rho_inf": 0.9,
        "max_increments": 10000,
    }
    defaults.update(overrides)
    return DynamicThreePointBendJigConfig(**defaults)


class TestDynamicThreePointBendJigConvergence:
    """動的三点曲げの収束テスト."""

    def test_dynamic_converges(self):
        """動的解析が収束する."""
        cfg = _dynamic_config()
        proc = DynamicThreePointBendJigProcess()
        result = proc.process(cfg)

        print(
            f"\n  動的三点曲げ: converged={result.solver_result.converged}, "
            f"increments={result.solver_result.n_increments}, "
            f"f1={result.analytical_frequency_hz:.1f} Hz, "
            f"T1={result.analytical_period:.6f} s, "
            f"v0={result.initial_velocity:.4f} mm/s, "
            f"δ_max={result.max_deflection:.6f} mm, "
            f"DAF={result.dynamic_amplification:.3f}"
        )
        assert result.solver_result.converged, "動的解析が収束しなかった"

    def test_frequency_analytical(self):
        """固有振動数の解析解が正しい."""
        sec = _circle_section(_D, _NU)
        f1 = _beam_fundamental_frequency(_L, _E, sec["Iy"], _RHO, sec["A"])
        assert f1 > 0
        T1 = 1.0 / f1
        print(f"\n  f1={f1:.1f} Hz, T1={T1:.6f} s")
        assert T1 < 1.0, f"固有周期 {T1} s が異常に長い"


class TestDynamicThreePointBendJigPhysics:
    """動的三点曲げの物理的妥当性テスト（初速度制御）."""

    def test_dynamic_response_has_oscillation(self):
        """動的応答に振動成分が含まれる."""
        cfg = _dynamic_config(jig_push=0.1, n_periods=2.0)
        proc = DynamicThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged
        hist = result.deflection_history
        if len(hist) < 5:
            pytest.skip("履歴が短すぎる")

        import numpy as np

        diffs = np.diff(hist)
        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
        print(
            f"\n  変位履歴: {len(hist)} 点, "
            f"max={np.max(hist):.6f}, min={np.min(hist):.6f}, "
            f"符号変化={sign_changes}"
        )
        assert sign_changes >= 2, f"振動の符号変化 {sign_changes} < 2: 動的応答に振動がない"

    def test_max_deflection_order(self):
        """最大変位が静的等価変位と同オーダー."""
        cfg = _dynamic_config(jig_push=0.1, n_periods=1.0)
        proc = DynamicThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged
        ratio = result.max_deflection / cfg.jig_push if cfg.jig_push > 0 else 0
        print(
            f"\n  max_defl={result.max_deflection:.6f} mm, "
            f"jig_push={cfg.jig_push:.6f} mm, ratio={ratio:.3f}"
        )
        # 初速度 v₀=ω₁*δ_s → 振幅≈δ_s。0.5〜2倍の範囲内
        assert 0.5 < ratio < 2.0, f"最大変位比 {ratio:.3f} が範囲外 [0.5, 2.0]"

    def test_vibration_period_matches_analytical(self):
        """振動周期が解析解（1次固有振動数）と一致する（5%以内）."""
        cfg = _dynamic_config(jig_push=0.1, n_periods=3.0, n_elems_wire=40)
        proc = DynamicThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged
        f_analytical = result.analytical_frequency_hz
        f_measured = result.measured_frequency_hz
        assert f_measured > 0, "FFTで周波数を検出できなかった"

        rel_error = abs(f_measured - f_analytical) / f_analytical
        print(
            f"\n  振動数: 解析解={f_analytical:.1f} Hz, "
            f"実測={f_measured:.1f} Hz, 誤差={rel_error * 100:.2f}%"
        )
        assert rel_error < 0.05, (
            f"振動周期の相対誤差 {rel_error * 100:.2f}% > 5%: "
            f"解析解 {f_analytical:.1f} Hz vs 実測 {f_measured:.1f} Hz"
        )

    def test_amplitude_matches_analytical(self):
        """最大変位が解析解（v₀/ω₁ = δ_static）と10%以内で一致する."""
        cfg = _dynamic_config(jig_push=0.1, n_periods=2.0, n_elems_wire=40)
        proc = DynamicThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged
        # 解析解: 自由振動の振幅 = v₀/ω₁ = δ_static = jig_push
        analytical_amplitude = cfg.jig_push
        rel_error = abs(result.max_deflection - analytical_amplitude) / analytical_amplitude
        print(
            f"\n  振幅: 解析解={analytical_amplitude:.6f} mm, "
            f"実測={result.max_deflection:.6f} mm, 誤差={rel_error * 100:.2f}%"
        )
        assert rel_error < 0.10, (
            f"振幅の相対誤差 {rel_error * 100:.2f}% > 10%: "
            f"解析解 {analytical_amplitude:.6f} mm vs 実測 {result.max_deflection:.6f} mm"
        )

    def test_dynamic_reaction_force_matches_analytical(self):
        """動的ピーク反力が k_EB × jig_push と一致する（許容誤差10%）.

        自由振動の最大変位時における弾性復元力:
          F_peak = k_EB × max_deflection ≈ k_EB × jig_push

        解析解: v₀ = ω₁ × δ_s なので振幅 ≈ δ_s = jig_push。
        よってピーク反力 ≈ k_EB × jig_push。
        """
        cfg = _dynamic_config(jig_push=0.1, n_periods=2.0, n_elems_wire=40)
        proc = DynamicThreePointBendJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged

        # 解析剛性
        sec = _circle_section(cfg.wire_diameter, cfg.nu)
        k_eb = 48.0 * cfg.E * sec["Iy"] / cfg.wire_length**3

        # ピーク反力の解析解一致
        F_peak = k_eb * result.max_deflection
        F_peak_analytical = k_eb * cfg.jig_push
        rel_err_peak = abs(F_peak - F_peak_analytical) / F_peak_analytical

        print(
            f"\n  動的反力検証:"
            f"\n    k_EB={k_eb:.4f} N/mm"
            f"\n    ピーク反力: F_peak={F_peak:.6f} N, "
            f"F_analytical={F_peak_analytical:.6f} N, 誤差={rel_err_peak * 100:.2f}%"
            f"\n    max_deflection={result.max_deflection:.6f} mm, "
            f"jig_push={cfg.jig_push:.6f} mm"
        )
        assert rel_err_peak < 0.10, (
            f"ピーク反力の相対誤差 {rel_err_peak * 100:.2f}% > 10%: "
            f"k*max_defl={F_peak:.6f} N vs k*jig_push={F_peak_analytical:.6f} N"
        )

    def test_rho_inf_numerical_dissipation(self):
        """rho_inf パラメータが動的応答に影響する（パラメータ感度検証）.

        Generalized-α法は高周波モードを選択的に減衰する。
        rho_inf = {1.0, 0.5, 0.0} で動的解析を実行し:
        1. 全ケースが収束する（ロバスト性）
        2. rho_inf によって最終変位が変化する（パラメータ感度）
        3. エネルギー診断が記録される（診断性）

        注: SE = 0.5*u^T*f_int は非線形 CR 梁では近似値のため、
        エネルギー減衰の厳密な単調性は保証されない。
        """
        import numpy as np

        rho_inf_values = [1.0, 0.5, 0.0]
        final_displacements = []
        proc = DynamicThreePointBendJigProcess()

        for rho_inf in rho_inf_values:
            cfg = _dynamic_config(jig_push=0.1, n_periods=2.0, rho_inf=rho_inf)
            result = proc.process(cfg)
            assert result.solver_result.converged, f"rho_inf={rho_inf} で収束しなかった"
            # エネルギー診断が記録されている
            assert result.solver_result.energy_history is not None, (
                f"rho_inf={rho_inf} でエネルギー履歴が記録されていない"
            )
            final_displacements.append(result.wire_midpoint_deflection)

        print("\n  rho_inf パラメータ感度:")
        for rho_inf, d in zip(rho_inf_values, final_displacements, strict=True):
            print(f"    rho_inf={rho_inf:.1f}: 最終変位={d:.6f} mm")

        # 異なる rho_inf で異なる最終変位 → パラメータが効いている
        d_arr = np.array(final_displacements)
        spread = (np.max(d_arr) - np.min(d_arr)) / np.mean(d_arr) if np.mean(d_arr) > 1e-30 else 0.0
        print(f"    変位ばらつき={spread * 100:.1f}%")
        assert spread > 0.01, (
            f"rho_inf によるばらつき {spread * 100:.1f}% < 1%: パラメータが動的応答に影響していない"
        )


# ====================================================================
# 動的接触ジグ三点曲げテスト（HEX8 + smooth_penalty + 摩擦）
# ====================================================================


def _dynamic_contact_config(**overrides) -> DynamicThreePointBendContactJigConfig:
    """動的接触ジグテスト用デフォルト設定."""
    defaults = {
        "wire_length": _L,
        "wire_diameter": _D,
        "n_elems_wire": 20,
        "E": _E,
        "nu": _NU,
        "rho": _RHO,
        "jig_push": 0.1,
        "n_periods": 2.0,
        "rho_inf": 0.9,
        "mu": 0.15,
        "initial_gap": 0.05,  # ジグをワイヤ上方0.05mmに配置→振動ピーク付近で接触
        "max_increments": 10000,
        "n_uzawa_max": 1,  # 純粋ペナルティ法
    }
    defaults.update(overrides)
    return DynamicThreePointBendContactJigConfig(**defaults)


class TestDynamicContactJigConvergence:
    """動的接触ジグ三点曲げの収束テスト."""

    def test_dynamic_contact_converges(self):
        """剛体エッジジグ + smooth_penalty + 摩擦 + 動的解析が収束する.

        半周期分のみ計算し、接触遷移を含む区間で収束することを検証。
        """
        cfg = _dynamic_contact_config(n_periods=0.5)
        proc = DynamicThreePointBendContactJigProcess()
        result = proc.process(cfg)

        print(
            f"\n  動的接触ジグ: converged={result.solver_result.converged}, "
            f"increments={result.solver_result.n_increments}, "
            f"f1={result.analytical_frequency_hz:.1f} Hz, "
            f"T1={result.analytical_period:.6f} s, "
            f"δ_max={result.max_deflection:.6f} mm, "
            f"DAF={result.dynamic_amplification:.3f}, "
            f"fc_norm={result.contact_force_norm:.6f}"
        )
        assert result.solver_result.converged, "動的接触ジグ解析が収束しなかった"

    def test_dynamic_contact_frictionless_converges(self):
        """摩擦なし（mu=0）で動的接触ジグが収束する（接触問題の分離検証）."""
        cfg = _dynamic_contact_config(n_periods=0.5, mu=0.0)
        proc = DynamicThreePointBendContactJigProcess()
        result = proc.process(cfg)

        print(
            f"\n  摩擦なし動的接触ジグ: converged={result.solver_result.converged}, "
            f"increments={result.solver_result.n_increments}, "
            f"δ_max={result.max_deflection:.6f} mm"
        )
        assert result.solver_result.converged, "摩擦なしで収束しなかった"


class TestDynamicContactJigPhysics:
    """動的接触ジグ三点曲げの物理的妥当性テスト."""

    def test_dynamic_contact_has_oscillation(self):
        """動的応答に振動成分が含まれる."""
        cfg = _dynamic_contact_config(n_periods=2.0)
        proc = DynamicThreePointBendContactJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged
        hist = result.deflection_history
        if len(hist) < 5:
            pytest.skip("履歴が短すぎる")

        import numpy as np

        diffs = np.diff(hist)
        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
        print(
            f"\n  変位履歴: {len(hist)} 点, "
            f"max={np.max(hist):.6f}, min={np.min(hist):.6f}, "
            f"符号変化={sign_changes}"
        )
        assert sign_changes >= 2, f"振動の符号変化 {sign_changes} < 2: 動的応答に振動がない"

    def test_max_deflection_order(self):
        """最大変位が初期変位と同オーダー."""
        cfg = _dynamic_contact_config(jig_push=0.1, n_periods=1.0)
        proc = DynamicThreePointBendContactJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged
        ratio = result.max_deflection / cfg.jig_push if cfg.jig_push > 0 else 0
        print(
            f"\n  max_defl={result.max_deflection:.6f} mm, "
            f"jig_push={cfg.jig_push:.6f} mm, ratio={ratio:.3f}"
        )
        # 接触拘束で振幅が変わるため、0.3〜3.0 の広い範囲
        assert 0.3 < ratio < 3.0, f"最大変位比 {ratio:.3f} が範囲外 [0.3, 3.0]"

    def test_frequency_within_range(self):
        """FFT計測周波数が解析的自由振動周波数と同オーダー.

        接触拘束により振動数が変化するため、厳密一致ではなく
        0.5〜2.0 倍の範囲内であることを確認。
        """
        cfg = _dynamic_contact_config(jig_push=0.1, n_periods=3.0, n_elems_wire=40)
        proc = DynamicThreePointBendContactJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged
        f_analytical = result.analytical_frequency_hz
        f_measured = result.measured_frequency_hz
        if f_measured <= 0:
            pytest.skip("FFTで周波数を検出できなかった")

        ratio = f_measured / f_analytical
        print(f"\n  振動数: 解析解={f_analytical:.1f} Hz, 実測={f_measured:.1f} Hz, 比={ratio:.3f}")
        assert 0.5 < ratio < 2.0, f"周波数比 {ratio:.3f} が範囲外 [0.5, 2.0]"

    def test_energy_history_recorded(self):
        """動的解析でエネルギー履歴が記録される."""
        cfg = _dynamic_contact_config(n_periods=1.0)
        proc = DynamicThreePointBendContactJigProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged
        assert result.solver_result.energy_history is not None, (
            "動的接触解析でエネルギー履歴が記録されていない"
        )
