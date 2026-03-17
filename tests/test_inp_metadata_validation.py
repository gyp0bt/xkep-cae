"""inp メタデータ整合検証テスト.

solve_from_inp の .inp 標準データ読み取り・メタデータ整合検証をテストする。

カテゴリ A〜E の問題（status-105）に対応:
- A: .inp 標準Abaqusデータの読み取り・検証
- B+C: E/nu のメタデータ→ソルバー受渡し
- D: 接触パラメータのメタデータ記録
- E: broadphase_margin 等の数値パラメータ記録

[← README](../README.md)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip(
    "xkep_cae.numerical_tests",
    reason="xkep_cae.numerical_tests 未移行のため無効化 (status-193)",
)

from xkep_cae.numerical_tests.wire_bending_benchmark import (  # noqa: E402
    _DEFAULT_E,
    _DEFAULT_NU,
    _compute_G,
    _compute_kappa,
)

# プログラムテスト: 16要素/ピッチ以上を厳守
_TEST_PARAMS = {"n_elems_per_strand": 16}


class TestMaterialParameterization:
    """カテゴリB+C: E/nu パラメータ化テスト."""

    def test_compute_G(self):
        """せん断弾性係数の計算."""
        G = _compute_G(200e3, 0.3)
        expected = 200e3 / (2.0 * 1.3)
        assert abs(G - expected) < 1e-6

    def test_compute_kappa(self):
        """Cowper せん断補正係数の計算."""
        kappa = _compute_kappa(0.3)
        expected = 6.0 * 1.3 / (7.0 + 1.8)
        assert abs(kappa - expected) < 1e-10

    def test_default_values_unchanged(self):
        """デフォルト値が鋼線値と一致（mm-ton-MPa単位系）."""
        assert _DEFAULT_E == 200e3  # MPa
        assert _DEFAULT_NU == 0.3

    def test_custom_material(self):
        """カスタム材料の G/kappa 計算."""
        E_aluminum = 70e3  # MPa
        nu_aluminum = 0.33
        G = _compute_G(E_aluminum, nu_aluminum)
        kappa = _compute_kappa(nu_aluminum)
        assert G == pytest.approx(E_aluminum / (2.0 * (1.0 + nu_aluminum)))
        assert kappa > 0.0 and kappa < 1.0


class TestMetadataExportImport:
    """メタデータの export → import ラウンドトリップテスト."""

    def test_metadata_roundtrip(self, tmp_path):
        """export_bending_oscillation_inp → load_metadata_from_inp のラウンドトリップ."""
        from scripts.run_bending_oscillation import (
            export_bending_oscillation_inp,
            load_metadata_from_inp,
        )

        out_dir = tmp_path / "test_roundtrip"
        inp_path = export_bending_oscillation_inp(3, out_dir, params=_TEST_PARAMS)
        meta = load_metadata_from_inp(inp_path)

        # 基本パラメータ
        assert meta["problem_type"] == "bending_oscillation"
        assert meta["n_strands"] == 3
        assert meta["xkep_version"] == "3.0"

        # カテゴリB+C: E, nu がメタデータに記録されている
        assert meta["E"] == 200e9
        assert meta["nu"] == 0.3

        # カテゴリD: 接触パラメータがメタデータに記録されている
        assert "k_t_ratio" in meta
        assert "g_on" in meta
        assert "g_off" in meta
        assert "use_line_search" in meta
        assert "line_search_max_steps" in meta
        assert "use_geometric_stiffness" in meta
        assert "tol_penetration_ratio" in meta
        assert "k_pen_max" in meta
        assert "exclude_same_layer" in meta
        assert "midpoint_prescreening" in meta
        assert "linear_solver" in meta
        assert "line_contact" in meta

        # カテゴリE: 数値パラメータがメタデータに記録されている
        assert "broadphase_margin" in meta
        assert meta["broadphase_margin"] == 0.01

    def test_metadata_custom_E_nu(self, tmp_path):
        """カスタム E/nu がメタデータに正しく記録される."""
        from scripts.run_bending_oscillation import (
            export_bending_oscillation_inp,
            load_metadata_from_inp,
        )

        custom_params = {"E": 70e9, "nu": 0.33, **_TEST_PARAMS}
        out_dir = tmp_path / "custom_mat"
        inp_path = export_bending_oscillation_inp(3, out_dir, params=custom_params)
        meta = load_metadata_from_inp(inp_path)
        assert meta["E"] == 70e9
        assert meta["nu"] == 0.33

    def test_metadata_category_d_defaults(self, tmp_path):
        """カテゴリD パラメータのデフォルト値が正しい."""
        from scripts.run_bending_oscillation import (
            export_bending_oscillation_inp,
            load_metadata_from_inp,
        )

        out_dir = tmp_path / "cat_d"
        inp_path = export_bending_oscillation_inp(3, out_dir, params=_TEST_PARAMS)
        meta = load_metadata_from_inp(inp_path)

        assert meta["k_t_ratio"] == 0.1
        assert meta["g_on"] == 0.0
        assert meta["g_off"] == 1e-5
        assert meta["use_line_search"] is True
        assert meta["line_search_max_steps"] == 5
        assert meta["use_geometric_stiffness"] is True
        assert meta["tol_penetration_ratio"] == 0.02
        assert meta["k_pen_max"] == 1e12
        assert meta["exclude_same_layer"] is True
        assert meta["midpoint_prescreening"] is True
        assert meta["linear_solver"] == "auto"
        assert meta["line_contact"] is True


class TestOutputSettingsMetadata:
    """出力設定のメタデータ記録テスト."""

    def test_output_settings_roundtrip(self, tmp_path):
        """出力設定がメタデータにラウンドトリップする."""
        from scripts.run_bending_oscillation import (
            export_bending_oscillation_inp,
            load_metadata_from_inp,
        )

        out_dir = tmp_path / "output_rt"
        inp_path = export_bending_oscillation_inp(3, out_dir, params=_TEST_PARAMS)
        meta = load_metadata_from_inp(inp_path)

        # 全出力設定がメタデータに記録されている
        assert meta["output_vtk"] is True
        assert meta["output_vtk_prefix"] == "result"
        assert meta["output_gif"] is True
        assert meta["output_gif_views"] == ["isometric", "end_yz"]
        assert meta["output_gif_figsize"] == [10.0, 8.0]
        assert meta["output_gif_dpi"] == 80
        assert meta["output_gif_duration"] == 300
        assert meta["output_contact_graph"] is True
        assert meta["output_contact_graph_fps"] == 2
        assert meta["output_contact_graph_figsize"] == [8, 6]
        assert meta["output_contact_graph_dpi"] == 80
        assert meta["output_summary"] is True

    def test_custom_output_settings(self, tmp_path):
        """カスタム出力設定がメタデータに正しく記録される."""
        from scripts.run_bending_oscillation import (
            export_bending_oscillation_inp,
            load_metadata_from_inp,
        )

        custom_params = {
            "output_vtk": False,
            "output_vtk_prefix": "custom_out",
            "output_gif_views": ["xy"],
            "output_gif_dpi": 150,
            "output_gif_duration": 500,
            "output_contact_graph_fps": 5,
            **_TEST_PARAMS,
        }
        out_dir = tmp_path / "custom_output"
        inp_path = export_bending_oscillation_inp(3, out_dir, params=custom_params)
        meta = load_metadata_from_inp(inp_path)

        assert meta["output_vtk"] is False
        assert meta["output_vtk_prefix"] == "custom_out"
        assert meta["output_gif_views"] == ["xy"]
        assert meta["output_gif_dpi"] == 150
        assert meta["output_gif_duration"] == 500
        assert meta["output_contact_graph_fps"] == 5
        # 未指定のものはデフォルト値
        assert meta["output_gif"] is True
        assert meta["output_summary"] is True

    def test_animation_request_in_inp(self, tmp_path):
        """InpAnimationRequest が .inp ファイルに記録される."""
        from scripts.run_bending_oscillation import export_bending_oscillation_inp

        out_dir = tmp_path / "animation"
        inp_path = export_bending_oscillation_inp(3, out_dir, params=_TEST_PARAMS)
        content = inp_path.read_text()
        assert "*OUTPUT, FIELD ANIMATION" in content


class TestInpValidation:
    """カテゴリA: .inp 標準データ検証テスト."""

    def test_validate_consistent_inp(self, tmp_path):
        """整合的な .inp はエラーなしで検証パス."""
        from scripts.run_bending_oscillation import (
            _validate_inp_vs_metadata,
            export_bending_oscillation_inp,
            load_metadata_from_inp,
        )

        out_dir = tmp_path / "consistent"
        inp_path = export_bending_oscillation_inp(3, out_dir, params=_TEST_PARAMS)
        meta = load_metadata_from_inp(inp_path)
        overrides = _validate_inp_vs_metadata(inp_path, meta)

        # .inp と メタデータが一致しているので上書きは発生しない
        public_overrides = {k: v for k, v in overrides.items() if not k.startswith("_inp_")}
        assert len(public_overrides) == 0

    def test_validate_detects_E_mismatch(self, tmp_path):
        """.inp の E をメタデータと不一致にした場合に検出."""
        from scripts.run_bending_oscillation import (
            _validate_inp_vs_metadata,
            export_bending_oscillation_inp,
            load_metadata_from_inp,
        )

        out_dir = tmp_path / "mismatch_E"
        inp_path = export_bending_oscillation_inp(3, out_dir, params=_TEST_PARAMS)

        # メタデータの E を書き換え（.inp の *ELASTIC は 200e9 のまま）
        meta = load_metadata_from_inp(inp_path)
        meta["E"] = 100e9  # 意図的に不一致

        overrides = _validate_inp_vs_metadata(inp_path, meta)
        assert "E" in overrides
        assert overrides["E"] == 200e9  # .inp の値が truth

    def test_validate_detects_nu_mismatch(self, tmp_path):
        """.inp の nu をメタデータと不一致にした場合に検出."""
        from scripts.run_bending_oscillation import (
            _validate_inp_vs_metadata,
            export_bending_oscillation_inp,
            load_metadata_from_inp,
        )

        out_dir = tmp_path / "mismatch_nu"
        inp_path = export_bending_oscillation_inp(3, out_dir, params=_TEST_PARAMS)
        meta = load_metadata_from_inp(inp_path)
        meta["nu"] = 0.4  # 意図的に不一致

        overrides = _validate_inp_vs_metadata(inp_path, meta)
        assert "nu" in overrides
        assert overrides["nu"] == 0.3  # .inp の値が truth

    def test_validate_records_node_info(self, tmp_path):
        """.inp から節点情報が読み取られる."""
        from scripts.run_bending_oscillation import (
            _validate_inp_vs_metadata,
            export_bending_oscillation_inp,
            load_metadata_from_inp,
        )

        out_dir = tmp_path / "nodes"
        inp_path = export_bending_oscillation_inp(3, out_dir, params=_TEST_PARAMS)
        meta = load_metadata_from_inp(inp_path)
        overrides = _validate_inp_vs_metadata(inp_path, meta)

        assert "_inp_n_nodes" in overrides
        assert "_inp_n_elems" in overrides
        # 3素線 × 16要素/素線 = 48要素, 51節点
        assert overrides["_inp_n_elems"] == 48
        assert overrides["_inp_n_nodes"] == 51


class TestNoHardcodedConstants:
    """ハードコード定数が除去されていることを検証."""

    def test_no_old_constants_in_benchmark(self):
        """wire_bending_benchmark.py に旧ハードコード定数が残っていない."""
        import inspect

        from xkep_cae.numerical_tests import wire_bending_benchmark as mod

        src = inspect.getsource(mod)
        # 旧モジュール定数名がソースに残っていないことを確認
        # （ただし _DEFAULT_E, _DEFAULT_NU は許可）
        assert "_E " not in src.replace("_DEFAULT_E", "").replace("_KAPPA", "")
        assert "_G " not in src.replace("_compute_G", "")
        assert "_NU " not in src.replace("_DEFAULT_NU", "")
        # _KAPPA は完全に除去されたはず
        assert "_KAPPA" not in src
