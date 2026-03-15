"""TimeIntegration Strategy サブパッケージ.

準静的・動的解析の時間積分方法。
"""

from xkep_cae.core.time_integration.strategy import (
    GeneralizedAlphaProcess,
    QuasiStaticProcess,
    TimeIntegrationInput,
    TimeIntegrationOutput,
)
from xkep_cae.core.time_integration.strategy import (
    _create_time_integration_strategy as create_time_integration_strategy,
)

__all__ = [
    "QuasiStaticProcess",
    "GeneralizedAlphaProcess",
    "TimeIntegrationInput",
    "TimeIntegrationOutput",
    "create_time_integration_strategy",
]
