"""数値試験フレームワーク — Process Architecture 版.

静的試験（3点曲げ・4点曲げ・引張・ねん回）、周波数応答試験、
動的試験をProcess Architectureで統合する。

旧 ``__xkep_cae_deprecated.numerical_tests`` からの移植。
要素剛性・ソルバー等の deprecated 依存はコールバック注入で分離。
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
    _build_section_props,
    analytical_bend3p,
    analytical_bend4p,
    analytical_tensile,
    analytical_torsion,
    assess_friction_effect,
    generate_beam_mesh_2d,
    generate_beam_mesh_2d_nonuniform,
    generate_beam_mesh_3d,
    generate_beam_mesh_3d_nonuniform,
)
from xkep_cae.numerical_tests.csv_export import (
    export_frequency_response_csv,
    export_static_csv,
)
from xkep_cae.numerical_tests.dynamic_runner import run_dynamic_test
from xkep_cae.numerical_tests.frequency import run_frequency_response
from xkep_cae.numerical_tests.inp_input import parse_test_input
from xkep_cae.numerical_tests.runner import run_all_tests, run_test, run_tests

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
    "_build_section_props",
    "analytical_bend3p",
    "analytical_bend4p",
    "analytical_tensile",
    "analytical_torsion",
    "assess_friction_effect",
    "export_frequency_response_csv",
    "export_static_csv",
    "generate_beam_mesh_2d",
    "generate_beam_mesh_2d_nonuniform",
    "generate_beam_mesh_3d",
    "generate_beam_mesh_3d_nonuniform",
    "parse_test_input",
    "run_all_tests",
    "run_dynamic_test",
    "run_frequency_response",
    "run_test",
    "run_tests",
]
