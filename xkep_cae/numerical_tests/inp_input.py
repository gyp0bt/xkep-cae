"""数値試験フレームワーク — Abaqusライクテキスト入力パーサー.

Abaqusの *KEYWORD 形式に準拠したテキストから
NumericalTestConfig / FrequencyResponseConfig を生成する。

サポートするキーワード:
  *TEST, TYPE={bend3p|bend4p|tensile|torsion|freq_response}
  *BEAM SECTION, SECTION={RECT|CIRC|PIPE}
    <断面パラメータ行>
  *MATERIAL
  *ELASTIC
    <E>, <nu>
  *DENSITY
    <rho>
  *SPECIMEN
    <length>, <n_elems>
  *LOAD
    <load_value>[, <load_span>]
  *SUPPORT, TYPE={ROLLER|PIN}
  *FREQUENCY
    <freq_min>, <freq_max>, <n_freq>
  *EXCITATION, TYPE={DISPLACEMENT|ACCELERATION}, DOF={uy|uz|...}
  *DAMPING, ALPHA=<val>, BETA=<val>

使用例:
    *TEST, TYPE=BEND3P
    *BEAM SECTION, SECTION=RECT
     10.0, 20.0
    *MATERIAL
    *ELASTIC
     200000.0, 0.3
    *SPECIMEN
     100.0, 10
    *LOAD
     -1000.0
    *SUPPORT, TYPE=ROLLER
"""

from __future__ import annotations

import re
from typing import Any

from xkep_cae.numerical_tests.core import (
    FrequencyResponseConfig,
    NumericalTestConfig,
)


def parse_test_input(
    text: str,
    beam_type: str = "timo2d",
) -> NumericalTestConfig | FrequencyResponseConfig:
    """Abaqusライクなテキスト入力から試験コンフィグを生成する.

    Args:
        text: 入力テキスト（Abaqusライク形式）
        beam_type: デフォルトの梁タイプ

    Returns:
        NumericalTestConfig or FrequencyResponseConfig
    """
    lines = text.strip().splitlines()
    params: dict[str, Any] = {
        "beam_type": beam_type,
        "test_type": None,
        "section_shape": "rectangle",
        "section_params": {},
        "E": None,
        "nu": 0.3,
        "rho": None,
        "length": None,
        "n_elems": 10,
        "load_value": None,
        "load_span": None,
        "support_condition": "roller",
        "freq_min": 1.0,
        "freq_max": 1000.0,
        "n_freq": 200,
        "excitation_type": "displacement",
        "excitation_dof": "uy",
        "response_dof": None,
        "damping_alpha": 0.0,
        "damping_beta": 0.0,
    }

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # コメント行・空行スキップ
        if not line or line.startswith("**"):
            i += 1
            continue

        upper = line.upper()

        if upper.startswith("*TEST"):
            params["test_type"] = _extract_param(line, "TYPE").lower()
            i += 1

        elif upper.startswith("*BEAM SECTION"):
            sec_type = _extract_param(line, "SECTION").upper()
            i += 1
            if i < len(lines):
                data_line = lines[i].strip()
                vals = _parse_data_line(data_line)
                if sec_type == "RECT":
                    params["section_shape"] = "rectangle"
                    params["section_params"] = {"b": vals[0], "h": vals[1]}
                elif sec_type == "CIRC":
                    params["section_shape"] = "circle"
                    params["section_params"] = {"d": vals[0]}
                elif sec_type == "PIPE":
                    params["section_shape"] = "pipe"
                    params["section_params"] = {"d_outer": vals[0], "d_inner": vals[1]}
                i += 1

        elif upper.startswith("*ELASTIC"):
            i += 1
            if i < len(lines):
                vals = _parse_data_line(lines[i].strip())
                params["E"] = vals[0]
                if len(vals) > 1:
                    params["nu"] = vals[1]
                i += 1

        elif upper.startswith("*DENSITY"):
            i += 1
            if i < len(lines):
                vals = _parse_data_line(lines[i].strip())
                params["rho"] = vals[0]
                i += 1

        elif upper.startswith("*MATERIAL"):
            i += 1  # ヘッダのみ、データは後続キーワードで読む

        elif upper.startswith("*SPECIMEN"):
            i += 1
            if i < len(lines):
                vals = _parse_data_line(lines[i].strip())
                params["length"] = vals[0]
                if len(vals) > 1:
                    params["n_elems"] = int(vals[1])
                i += 1

        elif upper.startswith("*LOAD"):
            i += 1
            if i < len(lines):
                vals = _parse_data_line(lines[i].strip())
                params["load_value"] = vals[0]
                if len(vals) > 1:
                    params["load_span"] = vals[1]
                i += 1

        elif upper.startswith("*SUPPORT"):
            sup_type = _extract_param(line, "TYPE").upper()
            params["support_condition"] = "pin" if sup_type == "PIN" else "roller"
            i += 1

        elif upper.startswith("*FREQUENCY"):
            i += 1
            if i < len(lines):
                vals = _parse_data_line(lines[i].strip())
                params["freq_min"] = vals[0]
                if len(vals) > 1:
                    params["freq_max"] = vals[1]
                if len(vals) > 2:
                    params["n_freq"] = int(vals[2])
                i += 1

        elif upper.startswith("*EXCITATION"):
            exc_type = _extract_param(line, "TYPE")
            if exc_type:
                params["excitation_type"] = exc_type.lower()
            exc_dof = _extract_param(line, "DOF")
            if exc_dof:
                params["excitation_dof"] = exc_dof.lower()
            resp_dof = _extract_param(line, "RESPONSE")
            if resp_dof:
                params["response_dof"] = resp_dof.lower()
            i += 1

        elif upper.startswith("*DAMPING"):
            alpha = _extract_param(line, "ALPHA")
            if alpha:
                params["damping_alpha"] = float(alpha)
            beta = _extract_param(line, "BETA")
            if beta:
                params["damping_beta"] = float(beta)
            i += 1

        elif upper.startswith("*BEAM TYPE"):
            bt = _extract_param(line, "TYPE")
            if bt:
                params["beam_type"] = bt.lower()
            i += 1

        else:
            i += 1

    # コンフィグ生成
    if params["test_type"] == "freq_response":
        if params["rho"] is None:
            raise ValueError("周波数応答試験には *DENSITY が必要です。")
        return FrequencyResponseConfig(
            beam_type=params["beam_type"],
            E=params["E"],
            nu=params["nu"],
            rho=params["rho"],
            length=params["length"],
            n_elems=params["n_elems"],
            section_shape=params["section_shape"],
            section_params=params["section_params"],
            freq_min=params["freq_min"],
            freq_max=params["freq_max"],
            n_freq=params["n_freq"],
            excitation_type=params["excitation_type"],
            excitation_dof=params["excitation_dof"],
            response_dof=params["response_dof"],
            damping_alpha=params["damping_alpha"],
            damping_beta=params["damping_beta"],
        )
    else:
        return NumericalTestConfig(
            name=params["test_type"],
            beam_type=params["beam_type"],
            E=params["E"],
            nu=params["nu"],
            length=params["length"],
            n_elems=params["n_elems"],
            load_value=params["load_value"],
            section_shape=params["section_shape"],
            section_params=params["section_params"],
            load_span=params["load_span"],
            support_condition=params["support_condition"],
        )


# ---------------------------------------------------------------------------
# パーサーヘルパー
# ---------------------------------------------------------------------------
def _extract_param(line: str, param_name: str) -> str | None:
    """*KEYWORD, PARAM=VALUE 形式からパラメータ値を抽出する."""
    pattern = rf"{param_name}\s*=\s*(\S+)"
    match = re.search(pattern, line, re.IGNORECASE)
    if match:
        return match.group(1).strip(",").strip()
    return None


def _parse_data_line(line: str) -> list[float]:
    """カンマ区切りのデータ行をパースする."""
    if not line or line.startswith("*") or line.startswith("**"):
        return []
    parts = [p.strip() for p in line.split(",") if p.strip()]
    return [float(p) for p in parts]
