"""収束判定の力/モーメント分離テスト.

並進 DOF [N] と回転 DOF [N·mm] を分離し、
力収束を並進残差のみで判定することを検証する。

status-240 で追加。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.solver._newton_steps import (
    ConvergenceCheckInput,
    ConvergenceCheckOutput,
    ConvergenceCheckProcess,
    ConvergenceType,
)


def _make_manager_stub(n_pairs: int = 0):
    """接触ペアなしのスタブマネージャー."""

    class _Stub:
        pairs: list = []

    return _Stub()


class TestConvergenceSeparationAPI:
    """力/モーメント分離の API テスト."""

    def test_output_has_trans_rot_norms(self):
        """出力に res_trans_norm, res_rot_norm が含まれる."""
        R_u = np.zeros(12)  # 2 nodes * 6 DOF
        R_u[0] = 1.0  # 並進 DOF
        R_u[3] = 100.0  # 回転 DOF
        proc = ConvergenceCheckProcess()
        out = proc.process(
            ConvergenceCheckInput(
                R_u=R_u,
                du=None,
                u=np.ones(12),
                f_ext_ref_norm=1e6,
                tol_force=1e-6,
                tol_disp=1e-8,
                dynamic_ref=False,
                is_first_attempt=True,
                energy_ref=None,
                manager=_make_manager_stub(),
                ndof_per_node=6,
            )
        )
        assert isinstance(out, ConvergenceCheckOutput)
        assert hasattr(out, "res_trans_norm")
        assert hasattr(out, "res_rot_norm")

    def test_separation_correct_values(self):
        """並進/回転の分離が正しい値を返す."""
        R_u = np.zeros(12)
        # node 0: trans=[3,0,0], rot=[0,4,0]
        R_u[0] = 3.0
        R_u[4] = 4.0
        # node 1: trans=[0,0,0], rot=[0,0,0]

        proc = ConvergenceCheckProcess()
        out = proc.process(
            ConvergenceCheckInput(
                R_u=R_u,
                du=None,
                u=np.ones(12),
                f_ext_ref_norm=1e6,
                tol_force=1e-6,
                tol_disp=1e-8,
                dynamic_ref=False,
                is_first_attempt=True,
                energy_ref=None,
                manager=_make_manager_stub(),
                ndof_per_node=6,
            )
        )
        np.testing.assert_allclose(out.res_trans_norm, 3.0, atol=1e-12)
        np.testing.assert_allclose(out.res_rot_norm, 4.0, atol=1e-12)

    def test_force_convergence_uses_translation_only(self):
        """力収束は並進残差のみで判定."""
        R_u = np.zeros(12)
        # 並進残差: ゼロ → 収束
        # 回転残差: 大きい → 収束に影響しない
        R_u[3] = 1e6  # 回転 DOF に大きな残差
        R_u[9] = 1e6

        proc = ConvergenceCheckProcess()
        out = proc.process(
            ConvergenceCheckInput(
                R_u=R_u,
                du=None,
                u=np.ones(12),
                f_ext_ref_norm=1.0,
                tol_force=1e-3,
                tol_disp=1e-8,
                dynamic_ref=False,
                is_first_attempt=False,
                energy_ref=None,
                manager=_make_manager_stub(),
                ndof_per_node=6,
            )
        )
        # 並進残差ゼロなので力収束するはず
        assert out.converged
        assert out.convergence_type == ConvergenceType.FORCE

    def test_large_translation_residual_not_converged(self):
        """並進残差が大きいと収束しない."""
        R_u = np.zeros(12)
        R_u[0] = 100.0  # 並進 DOF に大きな残差
        # 回転 DOF はゼロ

        proc = ConvergenceCheckProcess()
        out = proc.process(
            ConvergenceCheckInput(
                R_u=R_u,
                du=None,
                u=np.ones(12),
                f_ext_ref_norm=1.0,
                tol_force=1e-3,
                tol_disp=1e-8,
                dynamic_ref=False,
                is_first_attempt=False,
                energy_ref=None,
                manager=_make_manager_stub(),
                ndof_per_node=6,
            )
        )
        assert not out.converged

    def test_dynamic_ref_uses_translation_norm(self):
        """dynamic_ref=True のとき参照値は並進残差ノルム."""
        R_u = np.zeros(12)
        R_u[0] = 10.0  # 並進
        R_u[3] = 1000.0  # 回転（大きい）

        proc = ConvergenceCheckProcess()
        out = proc.process(
            ConvergenceCheckInput(
                R_u=R_u,
                du=None,
                u=np.ones(12),
                f_ext_ref_norm=1.0,
                tol_force=1e-6,
                tol_disp=1e-8,
                dynamic_ref=True,
                is_first_attempt=True,
                energy_ref=None,
                manager=_make_manager_stub(),
                ndof_per_node=6,
            )
        )
        # dynamic_ref=True の初回: f_ref = res_trans_norm = 10.0
        np.testing.assert_allclose(out.f_ref, 10.0, atol=1e-12)

    def test_ndof_per_node_3_no_rotation(self):
        """ndof_per_node=3 の場合は全て並進（回転ゼロ）."""
        R_u = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        proc = ConvergenceCheckProcess()
        out = proc.process(
            ConvergenceCheckInput(
                R_u=R_u,
                du=None,
                u=np.ones(6),
                f_ext_ref_norm=1e6,
                tol_force=1e-6,
                tol_disp=1e-8,
                dynamic_ref=False,
                is_first_attempt=False,
                energy_ref=None,
                manager=_make_manager_stub(),
                ndof_per_node=3,
            )
        )
        # ndof_per_node < 6 → 分離なし、trans = total
        np.testing.assert_allclose(out.res_trans_norm, np.linalg.norm(R_u), atol=1e-12)
        np.testing.assert_allclose(out.res_rot_norm, 0.0, atol=1e-12)

    def test_backward_compatible_default(self):
        """ndof_per_node デフォルト=6 で後方互換."""
        inp = ConvergenceCheckInput(
            R_u=np.zeros(12),
            du=None,
            u=np.ones(12),
            f_ext_ref_norm=1.0,
            tol_force=1e-6,
            tol_disp=1e-8,
            dynamic_ref=False,
            is_first_attempt=False,
            energy_ref=None,
            manager=_make_manager_stub(),
        )
        assert inp.ndof_per_node == 6
