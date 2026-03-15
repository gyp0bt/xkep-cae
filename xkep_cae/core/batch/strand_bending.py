"""StrandBendingBatchProcess — 撚線曲げ揺動ワークフロー.

旧 xkep_cae_deprecated/process/batch/strand_bending.py の完全書き直し。
設計仕様: process-architecture.md §6

Phase 2 時点では Strategy Process のみが移行済みのため、
ワークフロー実行は未実装。uses 宣言で依存関係を表明する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xkep_cae.contact.contact_force.strategy import (
    NCPContactForceProcess,
    SmoothPenaltyContactForceProcess,
)
from xkep_cae.contact.friction.strategy import (
    CoulombReturnMappingProcess,
    NoFrictionProcess,
    SmoothPenaltyFrictionProcess,
)
from xkep_cae.contact.geometry.strategy import (
    LineToLineGaussProcess,
    PointToPointProcess,
)
from xkep_cae.contact.penalty.strategy import (
    AutoBeamEIPenalty,
)
from xkep_cae.core import BatchProcess, ProcessMeta
from xkep_cae.core.time_integration.strategy import (
    GeneralizedAlphaProcess,
    QuasiStaticProcess,
)

# ── Input / Output ─────────────────────────────────────────


@dataclass(frozen=True)
class StrandBatchConfig:
    """撚線曲げ揺動バッチの設定.

    Phase 2 時点では Strategy 選択パラメータのみ。
    concrete プロセス（メッシュ生成、エクスポート等）の移行完了後に拡張する。
    """

    k_pen: float = 0.0
    use_friction: bool = True
    mu: float = 0.15
    contact_mode: str = "smooth_penalty"
    geometry_mode: str = "point_to_point"
    output_dir: str = "output"
    run_export: bool = True
    run_render: bool = False
    run_verify: bool = True


@dataclass
class StrandBatchResult:
    """撚線曲げ揺動バッチの結果."""

    solver_converged: bool = False
    elapsed_seconds: float = 0.0
    process_log: list[str] = field(default_factory=list)


# ── BatchProcess ───────────────────────────────────────────


class StrandBendingBatchProcess(
    BatchProcess[StrandBatchConfig, StrandBatchResult],
):
    """撚線曲げ揺動ワークフロー.

    実行ツリー（process-architecture.md §6）:
      MeshProcess → ContactSetupProcess → ContactFrictionProcess
        → [ExportProcess] → [RenderProcess] → [VerifyProcess]

    Phase 2 時点の uses: 移行済み Strategy プロセスのみ宣言。
    concrete プロセス（Mesh/Setup/Export/Render/Verify）は Phase 3 で追加予定。
    """

    meta = ProcessMeta(
        name="StrandBendingBatch",
        module="batch",
        version="1.0.0",
        document_path="docs/strand_bending.md",
    )

    uses = [
        # Strategy プロセス（Phase 2 移行済み）
        AutoBeamEIPenalty,
        NoFrictionProcess,
        CoulombReturnMappingProcess,
        SmoothPenaltyFrictionProcess,
        QuasiStaticProcess,
        GeneralizedAlphaProcess,
        PointToPointProcess,
        LineToLineGaussProcess,
        NCPContactForceProcess,
        SmoothPenaltyContactForceProcess,
    ]

    def process(self, input_data: StrandBatchConfig) -> StrandBatchResult:
        """ワークフロー実行.

        Phase 2 時点ではスタブ実装。
        concrete プロセス移行完了後にフル実装する。
        """
        import time

        t0 = time.perf_counter()
        log: list[str] = []
        result = StrandBatchResult()

        log.append("StrandBendingBatchProcess: Phase 2 stub")
        log.append(f"  contact_mode={input_data.contact_mode}")
        log.append(f"  geometry_mode={input_data.geometry_mode}")
        log.append(f"  use_friction={input_data.use_friction}")

        result.elapsed_seconds = time.perf_counter() - t0
        result.process_log = log
        return result
