"""TangentFDDiagnosticProcess のテスト（status-256）."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact.solver import (
    TangentFDDiagnosticInput,
    TangentFDDiagnosticProcess,
)
from xkep_cae.core.testing import binds_to


@binds_to(TangentFDDiagnosticProcess)
class TestTangentFDDiagnosticProcessAPI:
    """TangentFDDiagnosticProcess の単体テスト（status-256）."""

    def test_zero_du(self):
        """du=0 の場合はスキップ."""
        proc = TangentFDDiagnosticProcess()
        K = sp.eye(6, format="csr")
        out = proc.process(
            TangentFDDiagnosticInput(
                u=np.zeros(6),
                du=np.zeros(6),
                R_u=np.ones(6),
                K_T=K,
            )
        )
        assert out.directional_ratio == 1.0
        assert "zero" in out.report.lower() or "skip" in out.report.lower()

    def test_consistent_system_with_residual(self):
        """K_T が正定値で compute_residual ありの場合、FD と解析が一致."""
        ndof = 6
        K = sp.diags([10.0] * ndof, format="csr")
        R_u = np.array([10.0, -20.0, 30.0, 1.0, -2.0, 3.0])
        du = np.array([-1.0, 2.0, -3.0, -0.1, 0.2, -0.3])

        def compute_residual(u_pert):
            return K @ u_pert + R_u

        proc = TangentFDDiagnosticProcess()
        out = proc.process(
            TangentFDDiagnosticInput(
                u=np.zeros(ndof),
                du=du,
                R_u=R_u,
                K_T=K,
                compute_residual=compute_residual,
            )
        )
        # 線形系: FD と解析は一致
        assert out.deriv_agreement < 0.01
        assert "接線剛性" in out.report

    def test_with_compute_residual(self):
        """compute_residual 付きで FD 検証が走る."""
        ndof = 4
        K = sp.diags([5.0, 10.0, 5.0, 10.0], format="csr")
        R_u = np.array([1.0, -1.0, 0.5, -0.5])
        du = np.array([-0.2, 0.1, -0.1, 0.05])

        def compute_residual(u_pert):
            # 線形系: R(u) = K @ u - f_ext, f_ext = K@0 - R_u = -R_u
            return K @ u_pert + R_u

        proc = TangentFDDiagnosticProcess()
        out = proc.process(
            TangentFDDiagnosticInput(
                u=np.zeros(ndof),
                du=du,
                R_u=R_u,
                K_T=K,
                compute_residual=compute_residual,
            )
        )
        # 線形系なので FD と解析は一致するはず
        assert out.deriv_agreement < 0.01  # 1%未満の差

    def test_with_mpc_transform(self):
        """MPC変換行列ありの診断."""
        ndof = 6
        ndof_red = 4
        # T: (6, 4) — slave DOF 2つを master に結合
        T = sp.lil_matrix((ndof, ndof_red))
        T[0, 0] = 1.0
        T[1, 1] = 1.0
        T[2, 2] = 1.0
        T[3, 3] = 1.0
        T[4, 0] = 1.0  # slave DOF 4 = master DOF 0
        T[5, 1] = 1.0  # slave DOF 5 = master DOF 1
        T = T.tocsc()

        K = sp.eye(ndof, format="csr") * 10.0
        R_u = np.array([1.0, -1.0, 0.5, -0.5, 0.3, -0.3])
        du = np.array([-0.1, 0.1, -0.05, 0.05, -0.1, 0.1])

        proc = TangentFDDiagnosticProcess()
        out = proc.process(
            TangentFDDiagnosticInput(
                u=np.zeros(ndof),
                du=du,
                R_u=R_u,
                K_T=K,
                mpc_transform=T,
            )
        )
        assert "R_red" in out.report
        assert "cos" in out.report

    def test_mpc_with_compute_residual(self):
        """MPC変換行列 + compute_residual ありの場合、縮退系FD検証が走る."""
        ndof = 6
        ndof_red = 4
        T = sp.lil_matrix((ndof, ndof_red))
        T[0, 0] = 1.0
        T[1, 1] = 1.0
        T[2, 2] = 1.0
        T[3, 3] = 1.0
        T[4, 0] = 1.0  # slave DOF 4 = master DOF 0
        T[5, 1] = 1.0  # slave DOF 5 = master DOF 1
        T = T.tocsc()

        K = sp.eye(ndof, format="csr") * 10.0
        R_u = np.array([1.0, -1.0, 0.5, -0.5, 0.3, -0.3])
        du = np.array([-0.1, 0.1, -0.05, 0.05, -0.1, 0.1])

        def compute_residual(u_pert):
            return K @ u_pert + R_u

        proc = TangentFDDiagnosticProcess()
        out = proc.process(
            TangentFDDiagnosticInput(
                u=np.zeros(ndof),
                du=du,
                R_u=R_u,
                K_T=K,
                mpc_transform=T,
                compute_residual=compute_residual,
            )
        )
        # 線形系: MPC縮退系でもFDと解析は一致
        assert out.deriv_agreement < 0.01
        assert "MPC縮退系" in out.report

    def test_inconsistent_tangent_detected(self):
        """意図的に不整合な接線剛性を与え、検出されることを確認."""
        ndof = 4
        # 実際の系は K_true で動くが、ソルバーには K_wrong を渡す
        K_true = sp.diags([5.0, 10.0, 5.0, 10.0], format="csr")
        K_wrong = sp.diags([50.0, 10.0, 5.0, 10.0], format="csr")  # DOF 0 が10倍
        R_u = np.array([1.0, -1.0, 0.5, -0.5])
        du = np.array([-0.2, 0.1, -0.1, 0.05])

        def compute_residual(u_pert):
            return K_true @ u_pert + R_u

        proc = TangentFDDiagnosticProcess()
        out = proc.process(
            TangentFDDiagnosticInput(
                u=np.zeros(ndof),
                du=du,
                R_u=R_u,
                K_T=K_wrong,
                compute_residual=compute_residual,
            )
        )
        # FD と解析の不整合が検出される
        assert out.deriv_agreement > 0.1  # 10%以上の不整合

    def test_active_contact_dofs_consistent(self):
        """活性DOFフィルタリング: 線形系では整合（status-260）."""
        ndof = 6
        K = sp.diags([5.0, 10.0, 5.0, 10.0, 5.0, 10.0], format="csr")
        R_u = np.array([1.0, -1.0, 0.5, -0.5, 0.3, -0.3])
        du = np.array([-0.2, 0.1, -0.1, 0.05, -0.05, 0.02])

        def compute_residual(u_pert):
            return K @ u_pert + R_u

        proc = TangentFDDiagnosticProcess()
        out = proc.process(
            TangentFDDiagnosticInput(
                u=np.zeros(ndof),
                du=du,
                R_u=R_u,
                K_T=K,
                compute_residual=compute_residual,
                active_contact_dofs=np.array([0, 1, 2, 3]),
            )
        )
        assert out.active_dof_rel_err >= 0.0
        assert out.active_dof_rel_err < 0.01
        assert "活性DOF" in out.report

    def test_active_contact_dofs_detects_mismatch(self):
        """活性DOFで不整合検出（status-260）."""
        ndof = 4
        K_true = sp.diags([5.0, 10.0, 5.0, 10.0], format="csr")
        K_wrong = sp.diags([50.0, 10.0, 5.0, 10.0], format="csr")
        R_u = np.array([1.0, -1.0, 0.5, -0.5])
        du = np.array([-0.2, 0.1, -0.1, 0.05])

        def compute_residual(u_pert):
            return K_true @ u_pert + R_u

        proc = TangentFDDiagnosticProcess()
        # DOF 0（不整合あり）を含む
        out = proc.process(
            TangentFDDiagnosticInput(
                u=np.zeros(ndof),
                du=du,
                R_u=R_u,
                K_T=K_wrong,
                compute_residual=compute_residual,
                active_contact_dofs=np.array([0, 1]),
            )
        )
        assert out.active_dof_rel_err > 0.1

    def test_active_contact_dofs_none_skipped(self):
        """active_contact_dofs=None の場合は活性DOF診断スキップ（status-260）."""
        ndof = 4
        K = sp.diags([5.0, 10.0, 5.0, 10.0], format="csr")
        R_u = np.array([1.0, -1.0, 0.5, -0.5])
        du = np.array([-0.2, 0.1, -0.1, 0.05])

        def compute_residual(u_pert):
            return K @ u_pert + R_u

        proc = TangentFDDiagnosticProcess()
        out = proc.process(
            TangentFDDiagnosticInput(
                u=np.zeros(ndof),
                du=du,
                R_u=R_u,
                K_T=K,
                compute_residual=compute_residual,
                active_contact_dofs=None,
            )
        )
        assert out.active_dof_rel_err == -1.0
        assert "活性DOF" not in out.report

    def test_meta(self):
        assert TangentFDDiagnosticProcess.meta.name == "TangentFDDiagnostic"
