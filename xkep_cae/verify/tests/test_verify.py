"""VerifyProcess 群のテスト."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from xkep_cae.core import SolverResultData, VerifyProcess, VerifyResult
from xkep_cae.core.testing import binds_to
from xkep_cae.verify.contact import ContactVerifyInput, ContactVerifyProcess
from xkep_cae.verify.convergence import ConvergenceVerifyInput, ConvergenceVerifyProcess
from xkep_cae.verify.energy import EnergyBalanceVerifyInput, EnergyBalanceVerifyProcess


def _converged_result() -> SolverResultData:
    return SolverResultData(
        u=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        converged=True,
        n_increments=5,
        total_newton_iterations=20,
    )


def _diverged_result() -> SolverResultData:
    return SolverResultData(
        u=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        converged=False,
        n_increments=0,
        total_newton_iterations=200,
    )


# ── ConvergenceVerifyProcess ──


@binds_to(ConvergenceVerifyProcess)
class TestConvergenceVerifyProcess:
    """ConvergenceVerifyProcess の単体テスト."""

    def test_is_verify_process(self):
        proc = ConvergenceVerifyProcess()
        assert isinstance(proc, VerifyProcess)

    def test_meta_name(self):
        assert ConvergenceVerifyProcess.meta.name == "ConvergenceVerify"

    def test_converged_passes(self):
        proc = ConvergenceVerifyProcess()
        inp = ConvergenceVerifyInput(solver_result=_converged_result())
        result = proc.process(inp)
        assert isinstance(result, VerifyResult)
        assert result.passed is True

    def test_diverged_fails(self):
        proc = ConvergenceVerifyProcess()
        inp = ConvergenceVerifyInput(solver_result=_diverged_result())
        result = proc.process(inp)
        assert result.passed is False

    def test_iterations_threshold(self):
        proc = ConvergenceVerifyProcess()
        inp = ConvergenceVerifyInput(
            solver_result=_converged_result(),
            max_iterations_threshold=10,
        )
        result = proc.process(inp)
        # 20 > 10 なので反復数チェックが FAIL
        assert result.checks["total_newton_iterations"][2] is False

    def test_report_markdown_generated(self):
        proc = ConvergenceVerifyProcess()
        inp = ConvergenceVerifyInput(solver_result=_converged_result())
        result = proc.process(inp)
        assert "収束検証レポート" in result.report_markdown

    def test_config_frozen(self):
        inp = ConvergenceVerifyInput(solver_result=_converged_result())
        try:
            inp.max_iterations_threshold = 50  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass


# ── EnergyBalanceVerifyProcess ──


@binds_to(EnergyBalanceVerifyProcess)
class TestEnergyBalanceVerifyProcess:
    """EnergyBalanceVerifyProcess の単体テスト."""

    def test_is_verify_process(self):
        proc = EnergyBalanceVerifyProcess()
        assert isinstance(proc, VerifyProcess)

    def test_meta_name(self):
        assert EnergyBalanceVerifyProcess.meta.name == "EnergyBalanceVerify"

    def test_finite_displacement_passes(self):
        proc = EnergyBalanceVerifyProcess()
        inp = EnergyBalanceVerifyInput(solver_result=_converged_result())
        result = proc.process(inp)
        assert result.passed is True

    def test_external_work_check(self):
        proc = EnergyBalanceVerifyProcess()
        f_ext = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        inp = EnergyBalanceVerifyInput(
            solver_result=_converged_result(),
            f_ext=f_ext,
        )
        result = proc.process(inp)
        assert "external_work" in result.checks
        assert result.passed is True

    def test_nan_displacement_fails(self):
        proc = EnergyBalanceVerifyProcess()
        bad_result = SolverResultData(
            u=np.array([float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]),
            converged=True,
            n_increments=1,
            total_newton_iterations=5,
        )
        inp = EnergyBalanceVerifyInput(solver_result=bad_result)
        result = proc.process(inp)
        assert result.passed is False

    def test_energy_balance_from_diagnostics(self):
        proc = EnergyBalanceVerifyProcess()
        diag = SimpleNamespace(energy_balance=0.01)
        sr = SolverResultData(
            u=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            converged=True,
            n_increments=5,
            total_newton_iterations=20,
            diagnostics=diag,
        )
        inp = EnergyBalanceVerifyInput(solver_result=sr, tolerance=0.1)
        result = proc.process(inp)
        assert result.checks["energy_balance"][2] is True

    def test_report_markdown_generated(self):
        proc = EnergyBalanceVerifyProcess()
        inp = EnergyBalanceVerifyInput(solver_result=_converged_result())
        result = proc.process(inp)
        assert "エネルギー収支検証レポート" in result.report_markdown


# ── ContactVerifyProcess ──


@binds_to(ContactVerifyProcess)
class TestContactVerifyProcess:
    """ContactVerifyProcess の単体テスト."""

    def test_is_verify_process(self):
        proc = ContactVerifyProcess()
        assert isinstance(proc, VerifyProcess)

    def test_meta_name(self):
        assert ContactVerifyProcess.meta.name == "ContactVerify"

    def test_converged_fallback(self):
        proc = ContactVerifyProcess()
        inp = ContactVerifyInput(solver_result=_converged_result())
        result = proc.process(inp)
        # diagnostics なしの場合は converged で判定
        assert result.passed is True
        assert "solver_converged" in result.checks

    def test_penetration_check(self):
        proc = ContactVerifyProcess()
        diag = SimpleNamespace(max_penetration=0.0005, chattering_ratio=0.1)
        sr = SolverResultData(
            u=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            converged=True,
            n_increments=5,
            total_newton_iterations=20,
            diagnostics=diag,
        )
        inp = ContactVerifyInput(solver_result=sr, max_penetration=1e-3)
        result = proc.process(inp)
        assert result.passed is True

    def test_excessive_penetration_fails(self):
        proc = ContactVerifyProcess()
        diag = SimpleNamespace(max_penetration=0.01, chattering_ratio=0.1)
        sr = SolverResultData(
            u=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            converged=True,
            n_increments=5,
            total_newton_iterations=20,
            diagnostics=diag,
        )
        inp = ContactVerifyInput(solver_result=sr, max_penetration=1e-3)
        result = proc.process(inp)
        assert result.checks["max_penetration"][2] is False

    def test_chattering_check(self):
        proc = ContactVerifyProcess()
        diag = SimpleNamespace(max_penetration=0.0001, chattering_ratio=0.5)
        sr = SolverResultData(
            u=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            converged=True,
            n_increments=5,
            total_newton_iterations=20,
            diagnostics=diag,
        )
        inp = ContactVerifyInput(solver_result=sr, max_chattering_ratio=0.3)
        result = proc.process(inp)
        assert result.checks["chattering_ratio"][2] is False

    def test_report_markdown_generated(self):
        proc = ContactVerifyProcess()
        inp = ContactVerifyInput(solver_result=_converged_result())
        result = proc.process(inp)
        assert "接触検証レポート" in result.report_markdown

    def test_config_frozen(self):
        inp = ContactVerifyInput(solver_result=_converged_result())
        try:
            inp.max_penetration = 0.1  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass
