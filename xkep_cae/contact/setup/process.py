"""ContactSetupProcess — 接触設定の PreProcess.

旧 __xkep_cae_deprecated/process/concrete/pre_contact.py の完全書き直し。
設計仕様: docs/contact_setup.md
"""

from __future__ import annotations

from dataclasses import dataclass

from xkep_cae.contact._contact_pair import _ContactConfig, _ContactManager
from xkep_cae.core import ContactSetupData, MeshData, PreProcess, ProcessMeta


@dataclass(frozen=True)
class ContactSetupConfig:
    """接触設定の入力パラメータ."""

    mesh: MeshData
    k_pen: float = 0.0
    use_friction: bool = True
    mu: float = 0.15
    contact_mode: str = "smooth_penalty"
    broadphase_margin: float = 0.0
    broadphase_cell_size: float | None = None
    exclude_same_layer: bool = True
    line_contact: bool = False
    n_gauss: int = 2
    use_mortar: bool = False
    coating_stiffness: float = 0.0


class ContactSetupProcess(PreProcess[ContactSetupConfig, ContactSetupData]):
    """接触設定プロセス.

    ContactManager の初期化と broadphase 候補検出を実行する。
    """

    meta = ProcessMeta(
        name="ContactSetup",
        module="pre",
        version="1.0.0",
        document_path="docs/contact_setup.md",
    )

    def process(self, input_data: ContactSetupConfig) -> ContactSetupData:
        """接触設定の実行."""
        config = _ContactConfig(
            use_friction=input_data.use_friction,
            mu=input_data.mu,
            exclude_same_layer=input_data.exclude_same_layer,
            line_contact=input_data.line_contact,
            n_gauss=input_data.n_gauss,
            use_mortar=input_data.use_mortar,
            coating_stiffness=input_data.coating_stiffness,
        )
        manager = _ContactManager(config=config)

        mesh = input_data.mesh
        manager.detect_candidates(
            mesh.node_coords,
            mesh.connectivity,
            mesh.radii,
            margin=input_data.broadphase_margin,
            cell_size=input_data.broadphase_cell_size,
        )

        return ContactSetupData(
            manager=manager,
            k_pen=input_data.k_pen,
            use_friction=input_data.use_friction,
            mu=input_data.mu,
            contact_mode=input_data.contact_mode,
        )
