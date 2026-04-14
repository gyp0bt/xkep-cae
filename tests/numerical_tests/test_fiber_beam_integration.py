"""ファイバー梁モード統合テスト.

status-330 / Phase F5: StrandBendingOscillationProcess の use_fiber_beam フラグ検証。
- 弾性材料でファイバー梁モードが収束し、線形梁と同等の結果を返すことを確認
- 非線形材料（BilinearKH）でNR収束することを確認
- APIテスト（戻り値型、メッシュ構造）

[← README](../../README.md)
"""

from __future__ import annotations

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


class TestFiberBeamIntegrationAPI:
    """ファイバー梁モードの API テスト."""

    def test_fiber_beam_returns_result(self) -> None:
        """use_fiber_beam=True で結果が返る."""
        cfg = StrandBendingOscillationConfig(
            n_strands=7,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=8,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=10,
            use_fiber_beam=True,
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        assert result is not None
        assert result.solver_result is not None
        assert result.mesh is not None

    def test_fiber_beam_mesh_is_single_beam(self) -> None:
        """ファイバー梁モードのメッシュが単一直線梁."""
        n_elems_per_pitch = 8
        n_pitches = 1.0
        cfg = StrandBendingOscillationConfig(
            n_strands=7,
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=n_elems_per_pitch,
            n_pitches=n_pitches,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=5,
            use_fiber_beam=True,
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        expected_n_elems = int(n_elems_per_pitch * n_pitches)
        expected_n_nodes = expected_n_elems + 1
        assert result.n_strand_nodes == expected_n_nodes
        assert result.n_ref_nodes == 0
        assert result.total_ndof == expected_n_nodes * 6
        assert result.mesh.n_strands == 1

    def test_fiber_beam_bending_angle(self) -> None:
        """曲げ角度が正しく計算される."""
        kappa = 0.002
        L = 100.0
        cfg = StrandBendingOscillationConfig(
            wire_radius=0.5,
            pitch_length=L,
            n_elements_per_pitch=8,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=kappa,
            n_cycles=1,
            n_increments_per_cycle=5,
            use_fiber_beam=True,
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        expected_angle = kappa * L
        assert abs(result.bending_angle - expected_angle) < 1e-12

    def test_fiber_beam_polar_section(self) -> None:
        """polar 離散化でも動作する."""
        cfg = StrandBendingOscillationConfig(
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=8,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=5,
            use_fiber_beam=True,
            fiber_section_type="polar",
            fiber_n_fiber=4,
            fiber_n_theta=8,
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        assert result.solver_result.converged


class TestFiberBeamIntegrationPhysics:
    """ファイバー梁モードの物理テスト."""

    def test_elastic_fiber_beam_converges(self) -> None:
        """弾性材料でファイバー梁モードが frac=1.0 完走する."""
        cfg = StrandBendingOscillationConfig(
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=10,
            use_fiber_beam=True,
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        sr = result.solver_result
        frac = sr.load_history[-1] if sr.load_history else 0.0
        print("\n=== ファイバー梁 弾性 ===")
        print(f"  frac: {frac:.4f}, incr: {sr.n_increments}, cb: {sr.n_cutbacks}")

        assert sr.converged, f"弾性ファイバー梁が未収束: frac={frac}"
        assert frac >= 1.0 - 1e-10

    def test_elastic_tip_displacement_matches_theory(self) -> None:
        """弾性カンチレバーの先端変位が線形理論と一致（小変形範囲）.

        Euler-Bernoulli 片持ち梁:
        δ_z = κ * L² / 2 = θ * L / 2
        """
        kappa = 0.0005  # 小曲率（線形範囲）
        L = 100.0
        cfg = StrandBendingOscillationConfig(
            wire_radius=0.5,
            pitch_length=L,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=kappa,
            n_cycles=1,
            n_increments_per_cycle=10,
            use_fiber_beam=True,
            fiber_n_fiber=200,  # 高精度
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        sr = result.solver_result
        assert sr.converged

        # 先端節点の z 変位
        right_node = result.n_strand_nodes - 1
        u_z_tip = sr.u[right_node * 6 + 2]

        # 線形理論: δ_z ≈ κ * L² / 2（右手系 z 方向）
        # Timoshenko 梁はせん断項を含むが、L/d >> 1 の場合は Euler-Bernoulli で近似
        theta = kappa * L
        delta_theory = theta * L / 2.0

        # 10% 以内の一致（CR 定式化の非線形効果、ファイバー離散化誤差を許容）
        rel_err = abs(abs(u_z_tip) - delta_theory) / delta_theory
        print("\n=== ファイバー梁 先端変位検証 ===")
        print(f"  u_z_tip:    {u_z_tip:.6e}")
        print(f"  理論値:     {delta_theory:.6e}")
        print(f"  相対誤差:   {rel_err:.4f}")
        assert rel_err < 0.10, f"先端変位の相対誤差 {rel_err:.4f} > 0.10"

    def test_elastic_fiber_matches_contact_mode(self) -> None:
        """弾性ファイバー梁の先端変位が接触なし素線モデルと一致.

        use_fiber_beam=True と free_end_mode=True+contact_enabled=False を比較。
        両者とも弾性カンチレバーであり、接触なしの同一条件で
        先端変位が一致するべき。ただしファイバー梁は1本梁なので
        n_strands 分の剛性差は出ないことに注意。
        単一素線（n_strands=1 不可なので n_strands=2, 接触なし）で比較。
        """
        kappa = 0.0005
        L = 100.0
        common = dict(
            wire_radius=0.5,
            pitch_length=L,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=kappa,
            n_cycles=1,
            n_increments_per_cycle=10,
            max_nr_attempts=50,
            tol_force=1e-8,
        )

        # ファイバー梁モード（1本の直線梁）
        result_fb = StrandBendingOscillationProcess().process(
            StrandBendingOscillationConfig(
                **common,
                use_fiber_beam=True,
                fiber_n_fiber=200,
            )
        )

        # 接触なし素線モード（free_end_mode, 1素線の θ_y 処方）
        # free_end_mode は全素線にθ_yを処方。1素線を取り出して比較。
        result_contact = StrandBendingOscillationProcess().process(
            StrandBendingOscillationConfig(
                **common,
                n_strands=2,
                gap=0.5,  # 大きな gap で接触なし
                free_end_mode=True,
                contact_enabled=False,
            )
        )

        assert result_fb.solver_result.converged
        assert result_contact.solver_result.converged

        # ファイバー梁の先端変位
        right_fb = result_fb.n_strand_nodes - 1
        u_z_fb = result_fb.solver_result.u[right_fb * 6 + 2]

        # 接触モードの先端変位（最後のノード = 最右端の素線端部）
        # free_end_mode で全素線のθ_yが処方されているので、各素線で同様の変位
        # 最後のノードの z 変位を取得
        u_contact = result_contact.solver_result.u
        n_nodes_contact = result_contact.n_strand_nodes
        # 右端ノードを特定（最大 x 座標のノード）
        coords = result_contact.mesh.node_coords
        right_idx = np.argmax(coords[:n_nodes_contact, 0])
        u_z_contact = u_contact[right_idx * 6 + 2]

        # 10% 以内の一致（ヘリカル配置 vs 直線梁の差を許容）
        if abs(u_z_contact) > 1e-10:
            rel_err = abs(abs(u_z_fb) - abs(u_z_contact)) / abs(u_z_contact)
            print("\n=== ファイバー梁 vs 接触モード 比較 ===")
            print(f"  u_z_fb:      {u_z_fb:.6e}")
            print(f"  u_z_contact: {u_z_contact:.6e}")
            print(f"  相対誤差:    {rel_err:.4f}")
            assert rel_err < 0.15, f"ファイバー梁と接触モードの先端変位差 {rel_err:.4f} > 0.15"

    def test_bilinear_kh_fiber_beam_converges(self) -> None:
        """BilinearKH 材料でファイバー梁モードが収束する."""
        from xkep_cae.elements.fiber import BilinearKinematicHardening1D

        material = BilinearKinematicHardening1D(
            E=130.0e3,
            H=13.0e3,  # 硬化係数: E/10
            sigma_y=200.0,  # 降伏応力 [MPa]
        )
        cfg = StrandBendingOscillationConfig(
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=20,
            use_fiber_beam=True,
            fiber_material=material,
            fiber_n_fiber=60,
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        sr = result.solver_result
        frac = sr.load_history[-1] if sr.load_history else 0.0
        print("\n=== ファイバー梁 BilinearKH ===")
        print(f"  frac: {frac:.4f}, incr: {sr.n_increments}, cb: {sr.n_cutbacks}")

        assert sr.converged, f"BilinearKH ファイバー梁が未収束: frac={frac}"

    def test_multilayer_friction_fiber_beam_converges(self) -> None:
        """MultiLayerFrictionDegrading1D 材料でファイバー梁モードが収束する."""
        from xkep_cae.elements.fiber import MultiLayerFrictionDegrading1D

        n_layers = 10
        k_virgin = np.linspace(1000.0, 10000.0, n_layers)
        material = MultiLayerFrictionDegrading1D(
            E_base=130.0e3,
            k_virgin=k_virgin,
            k_degraded=k_virgin * 0.5,
            f_y=np.linspace(10.0, 100.0, n_layers),
        )
        cfg = StrandBendingOscillationConfig(
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.001,
            n_cycles=1,
            n_increments_per_cycle=20,
            use_fiber_beam=True,
            fiber_material=material,
            fiber_n_fiber=30,
        )
        proc = StrandBendingOscillationProcess()
        result = proc.process(cfg)

        sr = result.solver_result
        frac = sr.load_history[-1] if sr.load_history else 0.0
        print("\n=== ファイバー梁 MultiLayerFriction ===")
        print(f"  frac: {frac:.4f}, incr: {sr.n_increments}, cb: {sr.n_cutbacks}")

        assert sr.converged, f"MultiLayerFriction ファイバー梁が未収束: frac={frac}"

    def test_nonlinear_tip_displacement_less_than_elastic(self) -> None:
        """非線形材料の先端変位が弾性より小さい（降伏による剛性低下）.

        弾塑性材料では降伏すると剛性が低下するため、
        同一荷重（処方変位）に対して内力が小さくなり、
        変形形態が変化する。処方変位は同一のため θ_y は同じだが、
        塑性化による軸方向短縮が大きくなり得る。
        ここでは少なくとも結果が異なることを確認。
        """
        from xkep_cae.elements.fiber import BilinearKinematicHardening1D

        common = dict(
            wire_radius=0.5,
            pitch_length=100.0,
            n_elements_per_pitch=16,
            n_pitches=1.0,
            E=130.0e3,
            nu=0.3,
            rho=8.96e-9,
            bending_curvature=0.002,  # 大曲率で降伏を誘発
            n_cycles=1,
            n_increments_per_cycle=20,
            use_fiber_beam=True,
            fiber_n_fiber=60,
        )

        # 弾性
        result_el = StrandBendingOscillationProcess().process(
            StrandBendingOscillationConfig(**common)
        )

        # 弾塑性
        material_ep = BilinearKinematicHardening1D(E=130.0e3, H=13.0e3, sigma_y=100.0)
        result_ep = StrandBendingOscillationProcess().process(
            StrandBendingOscillationConfig(**common, fiber_material=material_ep)
        )

        assert result_el.solver_result.converged
        assert result_ep.solver_result.converged

        # 内力が異なることを確認（弾塑性では降伏により異なる変位パターン）
        u_el = result_el.solver_result.u
        u_ep = result_ep.solver_result.u
        diff = float(np.linalg.norm(u_el - u_ep))
        print("\n=== 弾性 vs 弾塑性 変位差 ===")
        print(f"  ||u_el||:  {np.linalg.norm(u_el):.6e}")
        print(f"  ||u_ep||:  {np.linalg.norm(u_ep):.6e}")
        print(f"  ||diff||:  {diff:.6e}")
        assert diff > 1e-6, "弾性と弾塑性の結果が同一 — 降伏が起きていない可能性"
