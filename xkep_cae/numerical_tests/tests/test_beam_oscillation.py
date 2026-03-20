"""BeamOscillationProcess / ElementBendingStrainProcess の @binds_to 紐付けテスト.

C3 契約: 全 concrete Process に対し @binds_to テストが必須。

[← README](../../../README.md)
"""

from __future__ import annotations

import numpy as np

from xkep_cae.core.testing import binds_to
from xkep_cae.numerical_tests.beam_oscillation import (
    BeamOscillationConfig,
    BeamOscillationProcess,
    ElementBendingStrainInput,
    ElementBendingStrainProcess,
)


@binds_to(BeamOscillationProcess)
class TestBeamOscillationProcessAPI:
    """BeamOscillationProcess の基本動作確認."""

    def test_process_runs(self):
        """小振幅で ProcessMeta + 入出力契約を満たす."""
        cfg = BeamOscillationConfig(
            amplitude=0.05,
            n_elems_wire=20,
            n_periods=0.5,
        )
        proc = BeamOscillationProcess()
        result = proc.process(cfg)
        assert result.solver_result.converged
        assert result.max_deflection > 0
        assert result.analytical_frequency_hz > 0
        assert len(result.deflection_history) > 0


@binds_to(ElementBendingStrainProcess)
class TestElementBendingStrainProcessAPI:
    """ElementBendingStrainProcess の基本動作確認."""

    def test_zero_displacement(self):
        """変位ゼロでひずみゼロ."""
        n_nodes = 11
        coords = np.column_stack(
            [np.linspace(0, 10, n_nodes), np.zeros(n_nodes), np.zeros(n_nodes)]
        )
        conn = np.column_stack([np.arange(10), np.arange(1, 11)])
        u = np.zeros(n_nodes * 6)
        proc = ElementBendingStrainProcess()
        result = proc.process(
            ElementBendingStrainInput(
                node_coords=coords,
                connectivity=conn,
                u=u,
                wire_radius=1.0,
            )
        )
        assert np.allclose(result.element_strain, 0.0)
        assert len(result.element_strain) == 10
