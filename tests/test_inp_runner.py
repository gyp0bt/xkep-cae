"""inp_runner（.inp → 梁解析モデル構築）のテスト.

examples/*.inp ファイルを読み込んでモデル構築→線形静解析→解析解比較を行う。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xkep_cae.io import build_beam_model_from_inp, node_dof, read_abaqus_inp, solve_beam_static

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


class TestBuildBeamModel:
    """build_beam_model_from_inp の基本テスト."""

    def test_cantilever_3d_model_structure(self):
        """3D片持ち梁: モデル構造が正しく構築される."""
        mesh = read_abaqus_inp(EXAMPLES_DIR / "cantilever_beam_3d.inp")
        model = build_beam_model_from_inp(mesh)

        assert model.is_3d is True
        assert model.ndof_per_node == 6
        assert model.nodes.shape == (11, 3)
        assert model.ndof_total == 66
        assert len(model.element_groups) == 1
        assert len(model.fixed_dofs) == 6  # node 1, DOF 1-6

    def test_three_point_bending_2d_model(self):
        """2D 3点曲げ: モデル構造が正しく構築される."""
        mesh = read_abaqus_inp(EXAMPLES_DIR / "three_point_bending.inp")
        model = build_beam_model_from_inp(mesh)

        assert model.is_3d is False
        assert model.ndof_per_node == 3
        assert model.nodes.shape == (21, 2)
        assert model.ndof_total == 63

    def test_portal_frame_multiple_sections(self):
        """門型フレーム: 複数断面が正しく処理される."""
        mesh = read_abaqus_inp(EXAMPLES_DIR / "portal_frame.inp")
        model = build_beam_model_from_inp(mesh)

        assert model.is_3d is True
        assert len(model.element_groups) == 3  # 左柱 + 梁 + 右柱
        assert len(model.sections) == 3
        assert model.ndof_total == 23 * 6  # 23 nodes × 6 DOF

    def test_pipe_section(self):
        """L型フレーム: パイプ断面が正しく構築される."""
        mesh = read_abaqus_inp(EXAMPLES_DIR / "l_frame_3d.inp")
        model = build_beam_model_from_inp(mesh)

        assert model.is_3d is True
        assert len(model.element_groups) == 2  # 垂直 + 水平

    def test_node_dof_mapping(self):
        """ノードラベルからDOFインデックスへの変換が正しい."""
        mesh = read_abaqus_inp(EXAMPLES_DIR / "cantilever_beam_3d.inp")
        model = build_beam_model_from_inp(mesh)

        # node 1 (index 0): DOF 0-5
        assert node_dof(model, 1, 0) == 0
        assert node_dof(model, 1, 5) == 5
        # node 2 (index 1): DOF 6-11
        assert node_dof(model, 2, 0) == 6

    def test_nsets_converted(self):
        """ノードセットがインデックスベースで変換される."""
        mesh = read_abaqus_inp(EXAMPLES_DIR / "cantilever_beam_3d.inp")
        model = build_beam_model_from_inp(mesh)

        assert "FIX" in model.nsets
        assert "LOAD" in model.nsets
        assert model.nsets["FIX"][0] == 0  # node 1 → index 0
        assert model.nsets["LOAD"][0] == 10  # node 11 → index 10


class TestSolveBeamStatic:
    """線形静解析のテスト — 解析解との比較."""

    def test_cantilever_3d_tip_load(self):
        """3D 片持ち梁先端荷重: 解析解との一致（< 0.1%）."""
        mesh = read_abaqus_inp(EXAMPLES_DIR / "cantilever_beam_3d.inp")
        model = build_beam_model_from_inp(mesh)

        P = -1000.0
        f = np.zeros(model.ndof_total)
        f[node_dof(model, 11, 1)] = P

        result = solve_beam_static(model, f, show_progress=False)
        u = result.u

        # 解析解
        mat = model.material
        sec = model.sections[0]
        E, nu = mat.E, mat.nu
        G = E / (2.0 * (1.0 + nu))
        L = 1.0
        Iz = sec.Iz
        A = sec.A
        kappa = sec.cowper_kappa_z(nu)
        delta_analytical = abs(P) * L**3 / (3.0 * E * Iz) + abs(P) * L / (kappa * G * A)

        tip_uy = abs(u[node_dof(model, 11, 1)])
        error = abs(tip_uy - delta_analytical) / delta_analytical
        assert error < 1e-3, f"相対誤差 {error:.6e} > 0.1%"

    def test_three_point_bending_2d(self):
        """2D 3点曲げ: 解析解との一致（< 0.1%）."""
        mesh = read_abaqus_inp(EXAMPLES_DIR / "three_point_bending.inp")
        model = build_beam_model_from_inp(mesh)

        P = -1000.0
        f = np.zeros(model.ndof_total)
        f[node_dof(model, 11, 1)] = P

        result = solve_beam_static(model, f, show_progress=False)
        u = result.u

        mat = model.material
        sec = model.sections[0]
        E, nu = mat.E, mat.nu
        G = E / (2.0 * (1.0 + nu))
        L = 0.5
        Ival = sec.I
        A = sec.A
        kappa = sec.cowper_kappa(nu)
        delta_analytical = abs(P) * L**3 / (48.0 * E * Ival) + abs(P) * L / (4.0 * kappa * G * A)

        mid_uy = abs(u[node_dof(model, 11, 1)])
        error = abs(mid_uy - delta_analytical) / delta_analytical
        assert error < 1e-3, f"相対誤差 {error:.6e} > 0.1%"

    def test_portal_frame_runs(self):
        """門型フレーム: 解析が完了し妥当な結果を返す."""
        mesh = read_abaqus_inp(EXAMPLES_DIR / "portal_frame.inp")
        model = build_beam_model_from_inp(mesh)

        P = 500.0
        f = np.zeros(model.ndof_total)
        f[node_dof(model, 7, 0)] = P

        result = solve_beam_static(model, f, show_progress=False)
        u = result.u

        # 水平荷重 → x変位が正
        assert u[node_dof(model, 7, 0)] > 0
        # 対称構造 → y変位は小さい
        assert abs(u[node_dof(model, 7, 1)]) < abs(u[node_dof(model, 7, 0)]) * 0.1

    def test_l_frame_runs(self):
        """L型フレーム: 解析が完了し妥当な結果を返す."""
        mesh = read_abaqus_inp(EXAMPLES_DIR / "l_frame_3d.inp")
        model = build_beam_model_from_inp(mesh)

        P = -100.0
        f = np.zeros(model.ndof_total)
        f[node_dof(model, 11, 1)] = P

        result = solve_beam_static(model, f, show_progress=False)
        u = result.u

        # 下向き荷重 → y変位が負
        assert u[node_dof(model, 11, 1)] < 0
