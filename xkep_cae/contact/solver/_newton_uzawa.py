"""Newton-Raphson + Uzawa イテレーション（プライベート）.

Static/Dynamic 完全分離版への統合エントリポイント。
Dynamic を正統 NewtonUzawaProcess とする（status-191）。
Static は保存用。
"""

from __future__ import annotations

from xkep_cae.contact.solver._newton_uzawa_dynamic import (  # noqa: F401
    DynamicStepResult,
    NewtonUzawaDynamicConfig,
    NewtonUzawaDynamicProcess,
    NewtonUzawaDynamicStepInput,
)
from xkep_cae.contact.solver._newton_uzawa_static import (  # noqa: F401
    NewtonUzawaStaticConfig,
    NewtonUzawaStaticProcess,
    NewtonUzawaStaticStepInput,
    StaticStepResult,
)

# --- Dynamic を正統 Process とする ---
NewtonUzawaProcess = NewtonUzawaDynamicProcess
NewtonUzawaConfig = NewtonUzawaDynamicConfig
NewtonUzawaStepInput = NewtonUzawaDynamicStepInput
StepResult = DynamicStepResult
