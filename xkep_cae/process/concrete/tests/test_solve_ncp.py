"""NCPContactSolverProcess の1:1テスト."""

from __future__ import annotations

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

    def test_effective_uses_from_slots(self):
        """effective_uses() が StrategySlot の型を含む（Phase 9-B）."""
        from xkep_cae.process.slots import collect_strategy_types

        proc = NCPContactSolverProcess()
        slot_types = collect_strategy_types(proc)
        effective = proc.effective_uses()
        assert len(effective) >= 3
        for t in slot_types:
            assert t in effective

    def test_instance_dependency_tree(self):
        proc = NCPContactSolverProcess()
        tree = proc.get_instance_dependency_tree()
        assert tree["name"] == "NCPContactSolverProcess"
        assert len(tree["uses"]) >= 3

    def test_registry_registered(self):
        from xkep_cae.process.base import AbstractProcess

        assert "NCPContactSolverProcess" in AbstractProcess._registry

    def test_strategy_slots_populated(self):
        """StrategySlot が正しく設定されていること（Phase 8-E）."""
        proc = NCPContactSolverProcess()
        assert proc.penalty_slot is not None
        assert proc.friction_slot is not None
        assert proc.time_integration_slot is not None

    def test_strategy_slots_match_strategies(self):
        """StrategySlot の値が strategies の値と一致すること."""
        proc = NCPContactSolverProcess()
        assert proc.penalty_slot is proc.strategies.penalty
        assert proc.friction_slot is proc.strategies.friction
        assert proc.time_integration_slot is proc.strategies.time_integration

    def test_no_runtime_uses_attribute(self):
        """Phase 9-B: _runtime_uses が廃止されていること."""
        proc = NCPContactSolverProcess()
        assert not hasattr(proc, "_runtime_uses")
