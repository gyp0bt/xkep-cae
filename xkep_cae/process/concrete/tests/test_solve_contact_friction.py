"""ContactFrictionProcess の1:1テスト."""

from __future__ import annotations

from xkep_cae.process.concrete.solve_contact_friction import (
    ContactFrictionProcess,
)
from xkep_cae.process.testing import binds_to


@binds_to(ContactFrictionProcess)
class TestContactFrictionProcess:
    """ContactFrictionProcess の単体テスト."""

    def test_meta(self):
        assert ContactFrictionProcess.meta.name == "摩擦接触ソルバー"
        assert ContactFrictionProcess.meta.module == "solve"
        assert not ContactFrictionProcess.meta.deprecated

    def test_default_strategies(self):
        """デフォルト strategies でインスタンス化."""
        proc = ContactFrictionProcess()
        assert proc.strategies is not None
        assert proc.strategies.penalty is not None
        assert proc.strategies.friction is not None
        assert proc.strategies.time_integration is not None

    def test_effective_uses_from_slots(self):
        """effective_uses() が StrategySlot の型を含む."""
        from xkep_cae.process.slots import collect_strategy_types

        proc = ContactFrictionProcess()
        slot_types = collect_strategy_types(proc)
        effective = proc.effective_uses()
        assert len(effective) >= 3
        for t in slot_types:
            assert t in effective

    def test_instance_dependency_tree(self):
        proc = ContactFrictionProcess()
        tree = proc.get_instance_dependency_tree()
        assert tree["name"] == "ContactFrictionProcess"
        assert len(tree["uses"]) >= 3

    def test_registry_registered(self):
        from xkep_cae.process.base import AbstractProcess

        assert "ContactFrictionProcess" in AbstractProcess._registry

    def test_strategy_slots_populated(self):
        """StrategySlot が正しく設定されていること."""
        proc = ContactFrictionProcess()
        assert proc.penalty_slot is not None
        assert proc.friction_slot is not None
        assert proc.time_integration_slot is not None

    def test_strategy_slots_match_strategies(self):
        """StrategySlot の値が strategies の値と一致すること."""
        proc = ContactFrictionProcess()
        assert proc.penalty_slot is proc.strategies.penalty
        assert proc.friction_slot is proc.strategies.friction
        assert proc.time_integration_slot is proc.strategies.time_integration

    def test_input_data_is_dynamic_property(self):
        """ContactFrictionInputData.is_dynamic が正しく判定すること."""
        import numpy as np
        import scipy.sparse as sp

        # 最小限のモックデータ
        from xkep_cae.process.data import (
            AssembleCallbacks,
            BoundaryData,
            ContactFrictionInputData,
            ContactSetupData,
            MeshData,
        )

        mesh = MeshData(
            node_coords=np.zeros((2, 3)),
            connectivity=np.array([[0, 1]]),
            radii=0.1,
            n_strands=1,
        )
        boundary = BoundaryData(
            f_ext_total=np.zeros(12),
            fixed_dofs=np.array([0, 1, 2]),
        )
        contact = ContactSetupData(manager=None, k_pen=1.0, use_friction=True)
        callbacks = AssembleCallbacks(
            assemble_tangent=lambda u: sp.eye(12),
            assemble_internal_force=lambda u: np.zeros(12),
        )

        # 準静的: mass_matrix なし
        qs_input = ContactFrictionInputData(
            mesh=mesh, boundary=boundary, contact=contact, callbacks=callbacks
        )
        assert not qs_input.is_dynamic

        # 動的: mass_matrix + dt_physical > 0
        dyn_input = ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact,
            callbacks=callbacks,
            mass_matrix=sp.eye(12),
            dt_physical=0.01,
        )
        assert dyn_input.is_dynamic
