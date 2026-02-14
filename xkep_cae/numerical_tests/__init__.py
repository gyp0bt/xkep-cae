"""数値試験フレームワーク（Phase 2.6）.

材料試験（3点曲げ・4点曲げ・引張・ねん回・周波数応答）の
数値シミュレーションを統一インタフェースで定義・実行・比較する。

[← README](../../README.md) | [ロードマップ](../../docs/roadmap.md)
"""

from xkep_cae.numerical_tests.core import (
    FrequencyResponseConfig,
    FrequencyResponseResult,
    NumericalTestConfig,
    StaticTestResult,
    analytical_bend3p,
    analytical_bend4p,
    analytical_tensile,
    analytical_torsion,
    generate_beam_mesh_2d_nonuniform,
    generate_beam_mesh_3d_nonuniform,
)
from xkep_cae.numerical_tests.csv_export import (
    export_frequency_response_csv,
    export_static_csv,
)
from xkep_cae.numerical_tests.frequency import (
    run_frequency_response,
)
from xkep_cae.numerical_tests.inp_input import (
    parse_test_input,
)
from xkep_cae.numerical_tests.runner import (
    run_all_tests,
    run_test,
    run_tests,
)

__all__ = [
    "NumericalTestConfig",
    "StaticTestResult",
    "FrequencyResponseConfig",
    "FrequencyResponseResult",
    "analytical_bend3p",
    "analytical_bend4p",
    "analytical_tensile",
    "analytical_torsion",
    "run_test",
    "run_all_tests",
    "run_tests",
    "run_frequency_response",
    "export_static_csv",
    "export_frequency_response_csv",
    "parse_test_input",
    "generate_beam_mesh_2d_nonuniform",
    "generate_beam_mesh_3d_nonuniform",
]
