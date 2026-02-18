"""JSON エクスポート.

OutputDatabase の全データを構造化 JSON 形式でエクスポートする。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from xkep_cae.output.database import OutputDatabase


class _NumpyEncoder(json.JSONEncoder):
    """NumPy 配列を JSON シリアライズ可能にするエンコーダー."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def export_json(
    db: OutputDatabase,
    output_dir: str | Path,
    *,
    filename: str = "output_database.json",
    indent: int = 2,
) -> str:
    """OutputDatabase を JSON ファイルにエクスポートする.

    Args:
        db: 出力データベース
        output_dir: 出力ディレクトリ
        filename: 出力ファイル名
        indent: JSON インデント

    Returns:
        生成されたファイルパス
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / filename

    data = _build_json_dict(db)

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, cls=_NumpyEncoder, indent=indent, ensure_ascii=False)

    return str(filepath)


def _build_json_dict(db: OutputDatabase) -> dict[str, Any]:
    """OutputDatabase を辞書に変換する."""
    result: dict[str, Any] = {
        "metadata": {
            "n_steps": db.n_steps,
            "n_nodes": db.n_nodes,
            "ndim": db.ndim,
            "ndof_per_node": db.ndof_per_node,
            "total_time": db.total_time(),
        },
        "node_sets": {name: indices.tolist() for name, indices in db.node_sets.items()},
    }

    if db.node_coords is not None:
        result["node_coords"] = db.node_coords.tolist()

    # ステップ結果
    steps_data: list[dict[str, Any]] = []
    for sr in db.step_results:
        step_dict: dict[str, Any] = {
            "name": sr.step.name,
            "step_index": sr.step_index,
            "start_time": sr.start_time,
            "total_time": sr.step.total_time,
            "dt": sr.step.dt,
            "converged": sr.converged,
            "n_increments": len(sr.increments),
            "n_frames": len(sr.frames),
        }

        # フレーム
        if sr.frames:
            frames_data = []
            for frame in sr.frames:
                fd: dict[str, Any] = {
                    "frame_index": frame.frame_index,
                    "time": frame.time,
                    "displacement": frame.displacement.tolist(),
                }
                if frame.velocity is not None:
                    fd["velocity"] = frame.velocity.tolist()
                if frame.acceleration is not None:
                    fd["acceleration"] = frame.acceleration.tolist()
                frames_data.append(fd)
            step_dict["frames"] = frames_data

        # ヒストリ
        if sr.history:
            history_data: dict[str, Any] = {
                "times": sr.history_times.tolist(),
            }
            for nset_name, var_data in sr.history.items():
                nset_dict: dict[str, Any] = {}
                for var_name, arr in var_data.items():
                    nset_dict[var_name] = arr.tolist()
                history_data[nset_name] = nset_dict
            step_dict["history"] = history_data

        steps_data.append(step_dict)

    result["steps"] = steps_data

    return result


__all__ = [
    "export_json",
]
