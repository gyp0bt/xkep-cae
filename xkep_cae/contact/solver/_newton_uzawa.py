"""Newton-Raphson + Uzawa イテレーション（プライベート）.

Static/Dynamic 完全分離版への統合エントリポイント。
後方互換のため NewtonUzawaProcess = NewtonUzawaStaticProcess として維持。
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

# --- 後方互換エイリアス ---
NewtonUzawaProcess = NewtonUzawaStaticProcess
NewtonUzawaConfig = NewtonUzawaStaticConfig
NewtonUzawaStepInput = NewtonUzawaStaticStepInput
StepResult = StaticStepResult
