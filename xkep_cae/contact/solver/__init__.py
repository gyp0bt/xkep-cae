"""摩擦接触ソルバー."""

from xkep_cae.contact.solver._energy_diagnostics import (
    StepEnergyDiagnosticsProcess,
    StepEnergyInput,
    StepEnergyOutput,
)
from xkep_cae.contact.solver._newton_steps import (
    ContactBacktrackingLineSearchInput,
    ContactBacktrackingLineSearchOutput,
    ContactBacktrackingLineSearchProcess,
    TangentFDDiagnosticInput,
    TangentFDDiagnosticOutput,
    TangentFDDiagnosticProcess,
)
from xkep_cae.contact.solver._unified_time_controller import (
    TimeStepQueryInput,
    TimeStepResultOutput,
    UnifiedTimeStepInput,
    UnifiedTimeStepProcess,
)

__all__ = [
    # 接触 backtracking line search（status-362: 仮説 C (c)）
    "ContactBacktrackingLineSearchInput",
    "ContactBacktrackingLineSearchOutput",
    "ContactBacktrackingLineSearchProcess",
    "StepEnergyDiagnosticsProcess",
    "StepEnergyInput",
    "StepEnergyOutput",
    # FD診断（status-256）
    "TangentFDDiagnosticProcess",
    "TangentFDDiagnosticInput",
    "TangentFDDiagnosticOutput",
    "TimeStepQueryInput",
    "TimeStepResultOutput",
    "UnifiedTimeStepInput",
    "UnifiedTimeStepProcess",
]
