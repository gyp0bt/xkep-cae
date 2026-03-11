"""ContactSetupProcess — 接触設定の PreProcess.

設計仕様: process-architecture.md §3.1
ContactManager 初期化 + broadphase 探索を PreProcess として管理する。
"""

from __future__ import annotations

from dataclasses import dataclass

from xkep_cae.process.base import ProcessMeta
from xkep_cae.process.categories import PreProcess
from xkep_cae.process.data import ContactSetupData, MeshData


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
        version="0.1.0",
        document_path="xkep_cae/process/docs/process-architecture.md",
    )

    def process(self, input_data: ContactSetupConfig) -> ContactSetupData:
        """接触設定の実行."""
        from xkep_cae.contact.pair import ContactConfig, ContactManager

        config = ContactConfig(
            use_friction=input_data.use_friction,
            mu=input_data.mu,
            exclude_same_layer=input_data.exclude_same_layer,
            line_contact=input_data.line_contact,
            n_gauss=input_data.n_gauss,
            use_mortar=input_data.use_mortar,
            coating_stiffness=input_data.coating_stiffness,
        )
        manager = ContactManager(config=config)

        # broadphase 候補検出
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
