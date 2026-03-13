"""NCPContactSolverProcess の1:1テスト."""

from __future__ import annotations

import pytest

from xkep_cae.process.concrete.solve_ncp import NCPContactSolverProcess
from xkep_cae.process.testing import binds_to


@binds_to(NCPContactSolverProcess)
class TestNCPContactSolverProcess:
    """NCPContactSolverProcess の単体テスト."""

    def test_meta(self):
        assert NCPContactSolverProcess.meta.name == "NCP接触ソルバー"
        assert NCPContactSolverProcess.meta.module == "solve"
        assert not NCPContactSolverProcess.meta.deprecated

    def test_default_strategies(self):
        """デフォルト strategies でインスタンス化."""
        proc = NCPContactSolverProcess()
        assert proc.strategies is not None
        assert proc.strategies.penalty is not None
        assert proc.strategies.friction is not None
        assert proc.strategies.time_integration is not None

    def test_runtime_uses_populated(self):
        """_runtime_uses が strategies から構築される."""
        proc = NCPContactSolverProcess()
        assert len(proc._runtime_uses) >= 3  # penalty, friction, time_integration

    def test_effective_uses(self):
        """effective_uses() が _runtime_uses を含む."""
        proc = NCPContactSolverProcess()
        effective = proc.effective_uses()
        assert len(effective) >= 3

    def test_instance_dependency_tree(self):
        proc = NCPContactSolverProcess()
        tree = proc.get_instance_dependency_tree()
        assert tree["name"] == "NCPContactSolverProcess"
        assert len(tree["uses"]) >= 3

    def test_registry_registered(self):
        from xkep_cae.process.base import AbstractProcess

        assert "NCPContactSolverProcess" in AbstractProcess._registry
