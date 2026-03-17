"""過渡応答出力インターフェース（Abaqus 準拠）.

Step / Increment / Frame の階層構造で動解析の計算結果を管理し、
CSV・JSON・VTK（ParaView対応）・アニメーション（PNG）の4形式でエクスポートする。

主要クラス:
    Step: 解析ステップの定義
    HistoryOutputRequest: ヒストリ出力要求（時系列プロファイル）
    FieldOutputRequest: フィールド出力要求（空間分布スナップショット）
    InitialConditions: 初期条件
    OutputDatabase: 全結果のデータベース

主要関数:
    build_output_database: ソルバー結果から OutputDatabase を構築
    export_history_csv: ヒストリ出力を CSV にエクスポート
    export_frames_csv: フレーム出力を CSV にエクスポート
    export_json: 全データを JSON にエクスポート
    export_vtk: フレームデータを VTK (ParaView) にエクスポート
    export_field_animation: 梁要素のアニメーション出力（PNG画像）
    render_beam_animation_frame: 梁要素の1フレーム描画
"""

from __xkep_cae_deprecated.output.database import (
    OutputDatabase,
    build_output_database,
    mesh_from_abaqus_inp,
    run_transient_steps,
)
from __xkep_cae_deprecated.output.export_animation import (
    export_3d_animation,
    export_3d_animation_gif,
    export_field_animation,
    export_field_animation_gif,
    render_beam_animation_frame,
)
from __xkep_cae_deprecated.output.export_csv import export_frames_csv, export_history_csv
from __xkep_cae_deprecated.output.export_json import export_json
from __xkep_cae_deprecated.output.export_vtk import (
    VTK_LINE,
    VTK_QUAD,
    VTK_QUADRATIC_TRIANGLE,
    VTK_TRIANGLE,
    VTK_VERTEX,
    export_vtk,
)
from __xkep_cae_deprecated.output.initial_conditions import (
    InitialConditionEntry,
    InitialConditions,
    InitialConditionType,
)
from __xkep_cae_deprecated.output.render_beam_3d import (
    VIEW_PRESETS,
    render_beam_3d,
    render_multiview_3d,
    render_twisted_wire_3d,
)
from __xkep_cae_deprecated.output.request import (
    ALL_VARIABLES,
    ENERGY_VARIABLES,
    NODAL_VARIABLES,
    FieldOutputRequest,
    HistoryOutputRequest,
)
from __xkep_cae_deprecated.output.step import Frame, IncrementResult, Step, StepResult

__all__ = [
    # Step / Increment / Frame
    "Step",
    "IncrementResult",
    "Frame",
    "StepResult",
    # Output Requests
    "HistoryOutputRequest",
    "FieldOutputRequest",
    "NODAL_VARIABLES",
    "ENERGY_VARIABLES",
    "ALL_VARIABLES",
    # Initial Conditions
    "InitialConditions",
    "InitialConditionEntry",
    "InitialConditionType",
    # Database
    "OutputDatabase",
    "build_output_database",
    "run_transient_steps",
    "mesh_from_abaqus_inp",
    # Export
    "export_history_csv",
    "export_frames_csv",
    "export_json",
    "export_vtk",
    "export_field_animation",
    "export_field_animation_gif",
    "render_beam_animation_frame",
    # 3D rendering
    "render_beam_3d",
    "render_twisted_wire_3d",
    "render_multiview_3d",
    "VIEW_PRESETS",
    "export_3d_animation",
    "export_3d_animation_gif",
    # VTK constants
    "VTK_VERTEX",
    "VTK_LINE",
    "VTK_TRIANGLE",
    "VTK_QUAD",
    "VTK_QUADRATIC_TRIANGLE",
]
