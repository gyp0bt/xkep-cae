"""数値試験フレームワーク.

静的試験（3点曲げ・4点曲げ・引張・ねん回）、周波数応答試験、
動的試験をProcess Architectureで統合する。
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
