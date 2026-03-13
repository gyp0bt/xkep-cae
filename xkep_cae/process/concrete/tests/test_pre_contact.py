"""ContactSetupProcess の1:1テスト."""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.process.concrete.pre_contact import (
    ContactSetupConfig,
    ContactSetupProcess,
)
from xkep_cae.process.data import ContactSetupData, MeshData
from xkep_cae.process.testing import binds_to


def _make_simple_mesh() -> MeshData:
    """2本の平行梁（4要素）の最小メッシュ."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [2.0, 0.5, 0.0],
        ]
    )
    conn = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
    return MeshData(
        node_coords=coords,
        connectivity=conn,
        radii=0.1,
        n_strands=2,
        layer_ids=np.array([0, 0, 0, 1, 1, 1]),
    )


@binds_to(ContactSetupProcess)
class TestContactSetupProcess:
    """ContactSetupProcess の単体テスト."""

    def test_meta(self):
        assert ContactSetupProcess.meta.name == "ContactSetup"
        assert ContactSetupProcess.meta.module == "pre"
        assert not ContactSetupProcess.meta.deprecated

    def test_process_returns_contact_setup_data(self):
        mesh = _make_simple_mesh()
        config = ContactSetupConfig(mesh=mesh, k_pen=1e4)
        proc = ContactSetupProcess()
        result = proc.process(config)
        assert isinstance(result, ContactSetupData)
        assert result.k_pen == 1e4
        assert result.use_friction is True

    def test_registry_registered(self):
        from xkep_cae.process.base import AbstractProcess

        assert "ContactSetupProcess" in AbstractProcess._registry
