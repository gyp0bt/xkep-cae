"""数値試験フレームワーク — CSV出力.

静的試験と周波数応答試験の結果をCSV形式で出力する。
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from xkep_cae.numerical_tests.core import (
    FrequencyResponseResult,
    StaticTestResult,
)


def export_static_csv(
    result: StaticTestResult,
    output_dir: str | Path | None = None,
    prefix: str = "",
) -> dict[str, str]:
    """静的試験結果をCSVファイルに出力する.

    出力ファイル:
    - {prefix}{name}_summary.csv: サマリ情報
    - {prefix}{name}_nodal_disp.csv: 節点変位
    - {prefix}{name}_element_forces.csv: 要素断面力

    Args:
        result: 静的試験結果
        output_dir: 出力ディレクトリ (None=文字列として返す)
        prefix: ファイル名プレフィックス

    Returns:
        dict: {ファイル種別: CSV文字列 or ファイルパス}
    """
    cfg = result.config
    name = cfg.name
    is_3d = cfg.beam_type == "timo3d"
    dof_per_node = 6 if is_3d else 3

    outputs = {}

    # --- サマリ CSV ---
    summary_rows = [
        ["項目", "値"],
        ["試験名", name],
        ["梁タイプ", cfg.beam_type],
        ["ヤング率 E", f"{cfg.E:.6g}"],
        ["ポアソン比 nu", f"{cfg.nu:.4f}"],
        ["試料長 L", f"{cfg.length:.6g}"],
        ["要素数", str(cfg.n_elems)],
        ["荷重値", f"{cfg.load_value:.6g}"],
        ["断面形状", cfg.section_shape],
        ["断面パラメータ", str(cfg.section_params)],
        ["FEM最大変位/回転", f"{result.displacement_max:.10g}"],
    ]
    if result.displacement_analytical is not None:
        summary_rows.append(["解析解", f"{result.displacement_analytical:.10g}"])
    if result.relative_error is not None:
        summary_rows.append(["相対誤差", f"{result.relative_error:.6e}"])
    if result.max_bending_stress > 0:
        summary_rows.append(["最大曲げ応力", f"{result.max_bending_stress:.6g}"])
    if result.max_shear_stress > 0:
        summary_rows.append(["最大せん断応力", f"{result.max_shear_stress:.6g}"])
    if result.friction_warning:
        summary_rows.append(["摩擦影響注記", result.friction_warning])

    outputs["summary"] = _write_csv(summary_rows, output_dir, f"{prefix}{name}_summary.csv")

    # --- 節点変位 CSV ---
    n_nodes = len(result.node_coords)
    u = result.displacement
    disp_header = ["node_id"]
    if is_3d:
        disp_header += ["x", "y", "z", "ux", "uy", "uz", "theta_x", "theta_y", "theta_z"]
    else:
        disp_header += ["x", "y", "ux", "uy", "theta_z"]

    disp_rows = [disp_header]
    for i in range(n_nodes):
        row = [str(i)]
        # 座標
        for c in range(result.node_coords.shape[1]):
            row.append(f"{result.node_coords[i, c]:.10g}")
        # 変位
        for d in range(dof_per_node):
            row.append(f"{u[dof_per_node * i + d]:.10g}")
        disp_rows.append(row)

    outputs["nodal_disp"] = _write_csv(disp_rows, output_dir, f"{prefix}{name}_nodal_disp.csv")

    # --- 要素断面力 CSV ---
    if is_3d:
        force_header = [
            "elem_id",
            "node_pos",
            "N",
            "Vy",
            "Vz",
            "Mx",
            "My",
            "Mz",
        ]
    else:
        force_header = ["elem_id", "node_pos", "N", "V", "M"]

    force_rows = [force_header]
    for e_idx, (f1, f2) in enumerate(result.element_forces):
        if is_3d:
            force_rows.append(
                [
                    str(e_idx),
                    "node1",
                    f"{f1.N:.10g}",
                    f"{f1.Vy:.10g}",
                    f"{f1.Vz:.10g}",
                    f"{f1.Mx:.10g}",
                    f"{f1.My:.10g}",
                    f"{f1.Mz:.10g}",
                ]
            )
            force_rows.append(
                [
                    str(e_idx),
                    "node2",
                    f"{f2.N:.10g}",
                    f"{f2.Vy:.10g}",
                    f"{f2.Vz:.10g}",
                    f"{f2.Mx:.10g}",
                    f"{f2.My:.10g}",
                    f"{f2.Mz:.10g}",
                ]
            )
        else:
            force_rows.append(
                [
                    str(e_idx),
                    "node1",
                    f"{f1.N:.10g}",
                    f"{f1.V:.10g}",
                    f"{f1.M:.10g}",
                ]
            )
            force_rows.append(
                [
                    str(e_idx),
                    "node2",
                    f"{f2.N:.10g}",
                    f"{f2.V:.10g}",
                    f"{f2.M:.10g}",
                ]
            )

    outputs["element_forces"] = _write_csv(
        force_rows, output_dir, f"{prefix}{name}_element_forces.csv"
    )

    return outputs


def export_frequency_response_csv(
    result: FrequencyResponseResult,
    output_dir: str | Path | None = None,
    prefix: str = "",
) -> dict[str, str]:
    """周波数応答試験結果をCSVファイルに出力する.

    出力ファイル:
    - {prefix}freq_response_frf.csv: 伝達関数データ
    - {prefix}freq_response_summary.csv: サマリ情報

    Args:
        result: 周波数応答試験結果
        output_dir: 出力ディレクトリ (None=文字列として返す)
        prefix: ファイル名プレフィックス

    Returns:
        dict: {ファイル種別: CSV文字列 or ファイルパス}
    """
    cfg = result.config
    outputs = {}

    # --- サマリ CSV ---
    summary_rows = [
        ["項目", "値"],
        ["試験種別", "freq_response"],
        ["梁タイプ", cfg.beam_type],
        ["ヤング率 E", f"{cfg.E:.6g}"],
        ["ポアソン比 nu", f"{cfg.nu:.4f}"],
        ["密度 rho", f"{cfg.rho:.6g}"],
        ["試料長 L", f"{cfg.length:.6g}"],
        ["要素数", str(cfg.n_elems)],
        ["周波数範囲", f"{cfg.freq_min:.1f} - {cfg.freq_max:.1f} Hz"],
        ["周波数点数", str(cfg.n_freq)],
        ["励起タイプ", cfg.excitation_type],
        ["励起DOF", cfg.excitation_dof],
        ["Rayleigh α", f"{cfg.damping_alpha:.6g}"],
        ["Rayleigh β", f"{cfg.damping_beta:.6g}"],
    ]
    if len(result.natural_frequencies) > 0:
        for i, fn in enumerate(result.natural_frequencies):
            summary_rows.append([f"推定固有振動数 {i + 1}", f"{fn:.4f} Hz"])

    outputs["summary"] = _write_csv(summary_rows, output_dir, f"{prefix}freq_response_summary.csv")

    # --- FRF CSV ---
    frf_header = [
        "freq_Hz",
        "omega_rad_s",
        "H_real",
        "H_imag",
        "magnitude",
        "phase_deg",
    ]
    frf_rows = [frf_header]
    for i in range(len(result.frequencies)):
        frf_rows.append(
            [
                f"{result.frequencies[i]:.6f}",
                f"{2 * 3.141592653589793 * result.frequencies[i]:.6f}",
                f"{result.transfer_function[i].real:.10g}",
                f"{result.transfer_function[i].imag:.10g}",
                f"{result.magnitude[i]:.10g}",
                f"{result.phase_deg[i]:.6f}",
            ]
        )

    outputs["frf"] = _write_csv(frf_rows, output_dir, f"{prefix}freq_response_frf.csv")

    return outputs


# ---------------------------------------------------------------------------
# CSV書き込みヘルパー
# ---------------------------------------------------------------------------
def _write_csv(
    rows: list[list[str]],
    output_dir: str | Path | None,
    filename: str,
) -> str:
    """CSV行データをファイルまたは文字列に書き出す.

    Args:
        rows: CSV行リスト
        output_dir: 出力ディレクトリ (None=文字列として返す)
        filename: ファイル名

    Returns:
        str: ファイルパス or CSV文字列
    """
    if output_dir is not None:
        out_path = Path(output_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows(rows)
        return str(out_path)
    else:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerows(rows)
        return buf.getvalue()
