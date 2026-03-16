"""CSV エクスポート.

OutputDatabase のヒストリ出力・フレーム出力を CSV 形式でエクスポートする。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xkep_cae.output.database import OutputDatabase
    from xkep_cae.output.step import Frame


def export_history_csv(
    db: OutputDatabase,
    output_dir: str | Path,
) -> list[str]:
    """ヒストリ出力を CSV ファイルにエクスポートする.

    各ステップ・各節点集合ごとに1ファイル生成:
        {step_name}_history_{nset_name}.csv

    Args:
        db: 出力データベース
        output_dir: 出力ディレクトリ

    Returns:
        生成されたファイルパスのリスト
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    for sr in db.step_results:
        if not sr.history:
            continue

        for nset_name, var_data in sr.history.items():
            filename = f"{sr.step.name}_history_{nset_name}.csv"
            filepath = out / filename

            # ヘッダー構築
            header = ["time"]
            data_columns: list[np.ndarray] = []

            for var_name, arr in var_data.items():
                if arr.ndim == 1:
                    # スカラー変数（ALLIE, ALLKE）
                    header.append(var_name)
                    data_columns.append(arr.reshape(-1, 1))
                else:
                    # ベクトル変数（U, V, A, RF, CF）
                    n_components = arr.shape[1]
                    nset_nodes = np.asarray(
                        sr.step.history_output.node_sets.get(nset_name, []), dtype=int
                    )
                    for c in range(n_components):
                        node_idx = c // db.ndof_per_node
                        dof_idx = c % db.ndof_per_node
                        if node_idx < len(nset_nodes):
                            node_label = int(nset_nodes[node_idx])
                        else:
                            node_label = c
                        header.append(f"{var_name}_{node_label}_d{dof_idx + 1}")
                    data_columns.append(arr)

            # データ行の書き出し
            times = sr.history_times
            n_rows = len(times)
            all_data = np.hstack(data_columns) if data_columns else np.zeros((n_rows, 0))

            with open(filepath, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                for i in range(n_rows):
                    row = [f"{times[i]:.10g}"]
                    for j in range(all_data.shape[1]):
                        row.append(f"{all_data[i, j]:.10g}")
                    writer.writerow(row)

            files.append(str(filepath))

    return files


def export_frames_csv(
    db: OutputDatabase,
    output_dir: str | Path,
) -> list[str]:
    """フレーム出力を CSV ファイルにエクスポートする.

    各ステップのフレーム一覧:
        {step_name}_frames_summary.csv

    各フレームの節点データ:
        {step_name}_frame_{frame_index:04d}.csv

    Args:
        db: 出力データベース
        output_dir: 出力ディレクトリ

    Returns:
        生成されたファイルパスのリスト
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    for sr in db.step_results:
        if not sr.frames:
            continue

        # フレーム一覧 CSV
        summary_path = out / f"{sr.step.name}_frames_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frame_index", "time"])
            for frame in sr.frames:
                writer.writerow([frame.frame_index, f"{frame.time:.10g}"])
        files.append(str(summary_path))

        # 各フレームの節点データ
        for frame in sr.frames:
            frame_path = out / f"{sr.step.name}_frame_{frame.frame_index:04d}.csv"
            _write_frame_csv(frame_path, frame, db)
            files.append(str(frame_path))

    return files


def _write_frame_csv(
    filepath: Path,
    frame: Frame,
    db: OutputDatabase,
) -> None:
    """1フレームの節点データを CSV ファイルに書き出す."""
    n_nodes = db.n_nodes if db.n_nodes > 0 else len(frame.displacement) // max(db.ndof_per_node, 1)
    ndpn = db.ndof_per_node

    # ヘッダー
    header = ["node_id"]
    if db.node_coords is not None:
        ndim = db.ndim
        for d in range(ndim):
            header.append(f"x{d + 1}")

    for d in range(ndpn):
        header.append(f"U{d + 1}")

    if frame.velocity is not None:
        for d in range(ndpn):
            header.append(f"V{d + 1}")

    if frame.acceleration is not None:
        for d in range(ndpn):
            header.append(f"A{d + 1}")

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for i in range(n_nodes):
            row: list[str] = [str(i)]

            if db.node_coords is not None:
                for d in range(db.ndim):
                    row.append(f"{db.node_coords[i, d]:.10g}")

            for d in range(ndpn):
                dof = i * ndpn + d
                row.append(f"{frame.displacement[dof]:.10g}")

            if frame.velocity is not None:
                for d in range(ndpn):
                    dof = i * ndpn + d
                    row.append(f"{frame.velocity[dof]:.10g}")

            if frame.acceleration is not None:
                for d in range(ndpn):
                    dof = i * ndpn + d
                    row.append(f"{frame.acceleration[dof]:.10g}")

            writer.writerow(row)


__all__ = [
    "export_history_csv",
    "export_frames_csv",
]
