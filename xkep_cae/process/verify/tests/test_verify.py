"""VerifyProcess 具象クラスの1:1テスト."""

from __future__ import annotations

import numpy as np

from xkep_cae.process.data import SolverResultData, VerifyResult
from xkep_cae.process.testing import binds_to
from xkep_cae.process.verify.contact import (
    ContactVerifyInput,
    ContactVerifyProcess,
)
from xkep_cae.process.verify.convergence import (
    ConvergenceVerifyInput,
    ConvergenceVerifyProcess,
)
from xkep_cae.process.verify.energy import (
    EnergyBalanceVerifyInput,
    EnergyBalanceVerifyProcess,
)


def _make_converged_result() -> SolverResultData:
    return SolverResultData(
        u=np.zeros(12),
        converged=True,
        n_increments=5,
        total_newton_iterations=20,
    )


def _make_diverged_result() -> SolverResultData:
    return SolverResultData(
        u=np.full(12, np.nan),
        converged=False,
        n_increments=0,
        total_newton_iterations=200,
    )


# --- ConvergenceVerifyProcess ---


@binds_to(ConvergenceVerifyProcess)
class TestConvergenceVerifyProcess:
    """ConvergenceVerifyProcess の単体テスト."""

    def test_meta(self):
        assert ConvergenceVerifyProcess.meta.name == "ConvergenceVerify"
        assert ConvergenceVerifyProcess.meta.module == "verify"

    def test_converged_result_passes(self):
        inp = ConvergenceVerifyInput(solver_result=_make_converged_result())
        proc = ConvergenceVerifyProcess()
        out = proc.process(inp)
        assert isinstance(out, VerifyResult)
        assert out.passed is True

    def test_diverged_result_fails(self):
        inp = ConvergenceVerifyInput(
            solver_result=_make_diverged_result(),
            max_iterations_threshold=100,
        )
        proc = ConvergenceVerifyProcess()
        out = proc.process(inp)
        assert out.passed is False

    def test_report_contains_markdown(self):
        inp = ConvergenceVerifyInput(solver_result=_make_converged_result())
        proc = ConvergenceVerifyProcess()
        out = proc.process(inp)
        assert "収束検証レポート" in out.report_markdown


# --- EnergyBalanceVerifyProcess ---


@binds_to(EnergyBalanceVerifyProcess)
class TestEnergyBalanceVerifyProcess:
    """EnergyBalanceVerifyProcess の単体テスト."""

    def test_meta(self):
        assert EnergyBalanceVerifyProcess.meta.name == "EnergyBalanceVerify"
        assert EnergyBalanceVerifyProcess.meta.module == "verify"

    def test_finite_displacement_passes(self):
        inp = EnergyBalanceVerifyInput(solver_result=_make_converged_result())
        proc = EnergyBalanceVerifyProcess()
        out = proc.process(inp)
        assert out.passed is True

    def test_nan_displacement_fails(self):
        inp = EnergyBalanceVerifyInput(solver_result=_make_diverged_result())
        proc = EnergyBalanceVerifyProcess()
        out = proc.process(inp)
        assert out.passed is False

    def test_with_external_force(self):
        result = _make_converged_result()
        result.u = np.ones(12) * 0.01
        f_ext = np.ones(12) * 100.0
        inp = EnergyBalanceVerifyInput(solver_result=result, f_ext=f_ext)
        proc = EnergyBalanceVerifyProcess()
        out = proc.process(inp)
        assert "external_work" in out.checks


# --- ContactVerifyProcess ---


@binds_to(ContactVerifyProcess)
class TestContactVerifyProcess:
    """ContactVerifyProcess の単体テスト."""

    def test_meta(self):
        assert ContactVerifyProcess.meta.name == "ContactVerify"
        assert ContactVerifyProcess.meta.module == "verify"

    def test_converged_without_diagnostics_passes(self):
        """diagnostics なしでも converged なら PASS."""
        inp = ContactVerifyInput(solver_result=_make_converged_result())
        proc = ContactVerifyProcess()
        out = proc.process(inp)
        assert out.passed is True

    def test_diverged_without_diagnostics_fails(self):
        inp = ContactVerifyInput(solver_result=_make_diverged_result())
        proc = ContactVerifyProcess()
        out = proc.process(inp)
        assert out.passed is False

    def test_report_contains_markdown(self):
        inp = ContactVerifyInput(solver_result=_make_converged_result())
        proc = ContactVerifyProcess()
        out = proc.process(inp)
        assert "接触検証レポート" in out.report_markdown
