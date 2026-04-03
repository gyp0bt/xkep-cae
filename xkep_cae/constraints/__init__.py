"""拘束条件パッケージ.

剛体結合、MPC（Multi-Point Constraint）などの拘束条件 Process を提供する。

[← README](../../README.md)
"""

from xkep_cae.constraints.mpc_elimination import (
    MPCEliminationConfig,
    MPCEliminationProcess,
    MPCEliminationResult,
    MPCGroup,
    RebuildMPCTransformInput,
    RebuildMPCTransformProcess,
)
from xkep_cae.constraints.rigid_assembler import (
    RigidEdgeAssemblerConfig,
    RigidEdgeAssemblerProcess,
    RigidEdgeAssemblerResult,
)

__all__ = [
    "MPCEliminationConfig",
    "MPCEliminationProcess",
    "MPCEliminationResult",
    "MPCGroup",
    "RebuildMPCTransformInput",
    "RebuildMPCTransformProcess",
    "RigidEdgeAssemblerConfig",
    "RigidEdgeAssemblerResult",
    "RigidEdgeAssemblerProcess",
]
