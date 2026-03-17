"""数値試験フレームワーク — Process Architecture 移行中.

静的試験（3点曲げ・4点曲げ・引張・ねん回）、周波数応答試験、
動的試験をProcess Architectureで統合する。

旧 ``__xkep_cae_deprecated.numerical_tests`` からの移植。
純粋関数のre-exportは廃止 (status-200)。
純粋関数は _ prefix で private 化済み (status-201)。
"""

from xkep_cae.numerical_tests.core import (
    TEST_TYPES_ALL,
    TEST_TYPES_DYNAMIC,
    TEST_TYPES_STATIC,
    BeamType,
    DynamicTestConfig,
    DynamicTestResult,
    FrequencyResponseConfig,
    FrequencyResponseResult,
    NumericalTestConfig,
    StaticTestResult,
    SupportCondition,
)

__all__ = [
    "TEST_TYPES_ALL",
    "TEST_TYPES_DYNAMIC",
    "TEST_TYPES_STATIC",
    "BeamType",
    "DynamicTestConfig",
    "DynamicTestResult",
    "FrequencyResponseConfig",
    "FrequencyResponseResult",
    "NumericalTestConfig",
    "StaticTestResult",
    "SupportCondition",
]
