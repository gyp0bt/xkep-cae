"""ContactSetupProcess — 接触設定の PreProcess.

設計仕様: docs/contact_setup.md
"""

from __future__ import annotations

from dataclasses import dataclass

from xkep_cae.contact._contact_pair import _ContactConfigInput, _ContactManagerInput
from xkep_cae.contact._manager_process import DetectCandidatesInput, DetectCandidatesProcess
from xkep_cae.core import ContactSetupData, MeshData, PreProcess, ProcessMeta


@dataclass(frozen=True)
class ContactSetupConfig:
    """接触設定の入力パラメータ."""

    mesh: MeshData
    k_pen: float = 0.0
    mu: float = 0.15
    broadphase_margin: float = 0.0
    broadphase_cell_size: float | None = None
    exclude_same_strand: bool = True
    line_contact: bool = False
    n_gauss: int = 2
    use_mortar: bool = False
    coating_stiffness: float = 0.0
    smoothing_delta: float = 0.0
    huber_delta_h: float = 0.0  # >0: Huber遷移幅を直接指定（status-261）


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
    uses = [DetectCandidatesProcess]

    def process(self, input_data: ContactSetupConfig) -> ContactSetupData:
        """接触設定の実行."""
        # strand_ids → elem_strand_map 変換
        mesh = input_data.mesh
        elem_strand_map: dict[int, int] | None = None
        if mesh.strand_ids is not None and input_data.exclude_same_strand:
            import numpy as np

            strand_arr = np.asarray(mesh.strand_ids)
            elem_strand_map = {int(i): int(strand_arr[i]) for i in range(len(strand_arr))}

        config = _ContactConfigInput(
            mu=input_data.mu,
            exclude_same_strand=input_data.exclude_same_strand,
            elem_strand_map=elem_strand_map,
            line_contact=input_data.line_contact,
            n_gauss=input_data.n_gauss,
            use_mortar=input_data.use_mortar,
            coating_stiffness=input_data.coating_stiffness,
            smoothing_delta=input_data.smoothing_delta,
            huber_delta_h=input_data.huber_delta_h,
        )
        manager = _ContactManagerInput(config=config)

        mesh = input_data.mesh
        _dc_out = DetectCandidatesProcess().process(
            DetectCandidatesInput(
                manager=manager,
                node_coords=mesh.node_coords,
                connectivity=mesh.connectivity,
                radii=mesh.radii,
                margin=input_data.broadphase_margin,
                cell_size=input_data.broadphase_cell_size,
            )
        )
        manager = _dc_out.manager

        return ContactSetupData(
            manager=manager,
            k_pen=input_data.k_pen,
            mu=input_data.mu,
        )
