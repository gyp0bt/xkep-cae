"""ContactSetupProcess のテスト."""

from __future__ import annotations

import numpy as np

from xkep_cae.contact.setup.process import ContactSetupConfig, ContactSetupProcess
from xkep_cae.core import ContactSetupData, MeshData, PreProcess
from xkep_cae.core.testing import binds_to


def _make_simple_mesh() -> MeshData:
    """テスト用の簡易2本梁メッシュ."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [1.0, 0.1, 0.0],
        ]
    )
    conn = np.array([[0, 1], [2, 3]])
    return MeshData(
        node_coords=coords,
        connectivity=conn,
        radii=0.05,
        n_strands=2,
    )


@binds_to(ContactSetupProcess)
class TestContactSetupProcess:
    """ContactSetupProcess の単体テスト."""

    def test_is_pre_process(self):
        proc = ContactSetupProcess()
        assert isinstance(proc, PreProcess)

    def test_meta_name(self):
        assert ContactSetupProcess.meta.name == "ContactSetup"

    def test_meta_module(self):
        assert ContactSetupProcess.meta.module == "pre"

    def test_config_defaults(self):
        mesh = _make_simple_mesh()
        config = ContactSetupConfig(mesh=mesh)
        assert config.mu == 0.15

    def test_config_frozen(self):
        mesh = _make_simple_mesh()
        config = ContactSetupConfig(mesh=mesh)
        try:
            config.k_pen = 999.0  # type: ignore[misc]
            raise AssertionError("frozen dataclass should not allow mutation")
        except AttributeError:
            pass

    def test_process_returns_contact_setup_data(self):
        mesh = _make_simple_mesh()
        config = ContactSetupConfig(mesh=mesh)
        proc = ContactSetupProcess()
        result = proc.process(config)
        assert isinstance(result, ContactSetupData)

    def test_process_preserves_parameters(self):
        mesh = _make_simple_mesh()
        config = ContactSetupConfig(mesh=mesh, k_pen=100.0, mu=0.3)
        proc = ContactSetupProcess()
        result = proc.process(config)
        assert result.k_pen == 100.0
        assert result.mu == 0.3

    def test_process_manager_created(self):
        mesh = _make_simple_mesh()
        config = ContactSetupConfig(mesh=mesh)
        proc = ContactSetupProcess()
        result = proc.process(config)
        assert result.manager is not None
