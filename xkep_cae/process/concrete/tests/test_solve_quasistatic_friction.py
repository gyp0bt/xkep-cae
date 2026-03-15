"""NCPQuasiStaticContactFrictionProcess の1:1テスト（deprecated）."""

from __future__ import annotations

import warnings

from xkep_cae.process.concrete.solve_quasistatic_friction import (
    NCPQuasiStaticContactFrictionProcess,
)
from xkep_cae.process.testing import binds_to


@binds_to(NCPQuasiStaticContactFrictionProcess)
class TestNCPQuasiStaticContactFrictionProcess:
    """NCPQuasiStaticContactFrictionProcess の単体テスト."""

    def test_meta(self):
        assert NCPQuasiStaticContactFrictionProcess.meta.name == "準静的摩擦接触ソルバー"
        assert NCPQuasiStaticContactFrictionProcess.meta.module == "solve"
        assert NCPQuasiStaticContactFrictionProcess.meta.deprecated
        assert NCPQuasiStaticContactFrictionProcess.meta.deprecated_by == "ContactFrictionProcess"

    def test_default_strategies(self):
        """デフォルト strategies でインスタンス化（DeprecationWarning 発生）."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            proc = NCPQuasiStaticContactFrictionProcess()
        assert proc.strategies is not None
        assert proc.strategies.penalty is not None
        assert proc.strategies.friction is not None
        assert proc.strategies.time_integration is not None

    def test_deprecation_warning(self):
        """インスタンス化時に DeprecationWarning が出ること."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NCPQuasiStaticContactFrictionProcess()
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_effective_uses_from_slots(self):
        """effective_uses() が StrategySlot の型を含む."""
        from xkep_cae.process.slots import collect_strategy_types

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            proc = NCPQuasiStaticContactFrictionProcess()
        slot_types = collect_strategy_types(proc)
        effective = proc.effective_uses()
        assert len(effective) >= 3
        for t in slot_types:
            assert t in effective

    def test_registry_registered(self):
        from xkep_cae.process.base import AbstractProcess

        assert "NCPQuasiStaticContactFrictionProcess" in AbstractProcess._registry
