#!/usr/bin/env python3
"""撚線曲げ揺動 — プリ/ソルブ分離スクリプト（7〜91本）.

曲げ揺動計算を **プリ処理（.inp エクスポート）** と **ソルバー実行** の
2工程に分離する。これにより、メッシュ・条件を .inp で固定したまま
ソルバーパラメータだけ変えた再実行が容易になる。

== モード ==

  export : メッシュ・材料・BC・荷重・ソルバーパラメータを .inp 形式で書き出す
  solve  : .inp を読み込んで曲げ揺動を実行し、VTK/GIF/接触グラフを出力
  all    : export → solve を連続実行（デフォルト）

== 使い方 ==

  # プリ処理のみ（.inp 生成）
  python scripts/run_bending_oscillation.py export --strands 7,19

  # .inp から計算（結果出力フック付き）
  python scripts/run_bending_oscillation.py solve --inpdir scripts/results/bending

  # 一気通貫
  python scripts/run_bending_oscillation.py all --strands 7,19,37,61,91

== .inp 構造 ==

  標準 Abaqus キーワード:
    *NODE, *ELEMENT (B31), *NSET, *MATERIAL, *ELASTIC,
    *BEAM SECTION, *BOUNDARY

  xkep-cae 独自拡張（コメントブロック）:
    ** XKEP-CAE METADATA BEGIN
    ** {JSON: ソルバー・接触・荷重パラメータ}
    ** XKEP-CAE METADATA END

== 出力 ==

  {outdir}/{N}strand/
    model_{N}strand.inp          ← プリ処理
    bending_oscillation_*_yz.gif ← 変位 GIF
    bending_oscillation_*_xz.gif
    contact_graph_*strand.gif    ← 接触グラフ GIF
    vtk/result.pvd + *.vtu       ← VTK
    summary.txt                  ← テキストサマリー

[← README](../README.md)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from xkep_cae.contact.graph import save_contact_graph_gif
from xkep_cae.io.abaqus_inp import (
    InpContactDef,
    InpOutputRequest,
    InpStep,
    InpSurfaceDef,
    InpSurfaceInteraction,
    read_abaqus_inp,
    write_abaqus_model,
)
from xkep_cae.mesh.twisted_wire import (
    TwistedWireMesh,
    make_twisted_wire_mesh,
    minimum_strand_diameter,
)
from xkep_cae.numerical_tests.wire_bending_benchmark import (
    BendingOscillationResult,
    export_bending_oscillation_gif,
    run_bending_oscillation,
)
from xkep_cae.output.database import OutputDatabase
from xkep_cae.output.export_vtk import VTK_LINE, export_vtk
from xkep_cae.output.step import Frame, Step, StepResult

# ====================================================================
# 共通パラメータ
# ====================================================================

# 軽量パラメータ（分析確認用 — 本番では値を上げる）
DEFAULT_PARAMS = {
    # メッシュ
    "wire_diameter": 0.002,
    "pitch": 0.040,
    "n_elems_per_strand": 4,
    "n_pitches": 0.5,
    "strand_diameter": "auto",  # "auto" = minimum_strand_diameter で自動計算
    # 材料
    "E": 200e9,
    "nu": 0.3,
    # 曲げ
    "bend_angle_deg": 45.0,
    "n_bending_steps": 5,
    # 揺動
    "oscillation_amplitude_mm": 2.0,
    "n_cycles": 1,
    "n_steps_per_quarter": 2,
    # ソルバー
    "max_iter": 20,
    "n_outer_max": 3,
    "tol_force": 1e-4,
    # 接触
    "auto_kpen": True,
    "use_friction": False,
    "mu": 0.0,
    "k_pen_scaling": "sqrt",
    "penalty_growth_factor": 4.0,
    # NCP ソルバー（デフォルトで有効）
    "use_ncp": True,
    "use_mortar": True,
    "n_gauss": 2,
    "ncp_k_pen": 0.0,
    "augmented_threshold": 20,
    "saddle_regularization": 0.0,
    "ncp_active_threshold": 0.0,
    "lambda_relaxation": 1.0,
    "max_step_cuts": 3,
    "modified_nr_threshold": 5,
    # カテゴリD: 接触パラメータ（従来ハードコード）
    "k_t_ratio": 0.1,
    "g_on": 0.0,
    "g_off": 1e-5,
    "use_line_search": True,
    "line_search_max_steps": 5,
    "use_geometric_stiffness": True,
    "tol_penetration_ratio": 0.02,
    "k_pen_max": 1e12,
    "exclude_same_layer": True,
    "midpoint_prescreening": True,
    "linear_solver": "auto",
    "line_contact": True,
    # カテゴリE: 数値パラメータ（従来ハードコード）
    "broadphase_margin": 0.01,
}

STRAND_COUNTS_DEFAULT = [7, 19, 37, 61, 91]

_NDOF_PER_NODE = 6


# ====================================================================
# Phase 1: .inp エクスポート（プリ処理）
# ====================================================================


def export_bending_oscillation_inp(
    n_strands: int,
    out_dir: Path,
    params: dict | None = None,
) -> Path:
    """曲げ揺動問題を .inp 形式でエクスポートする.

    Args:
        n_strands: 素線本数
        out_dir: 出力ディレクトリ
        params: パラメータ辞書（None なら DEFAULT_PARAMS）

    Returns:
        書き出した .inp ファイルのパス
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 非貫入配置: strand_diameter 計算 ---
    sd_param = p.get("strand_diameter")
    if sd_param == "auto":
        sd_value = minimum_strand_diameter(n_strands, p["wire_diameter"])
    elif sd_param is not None and sd_param != 0:
        sd_value = float(sd_param)
    else:
        sd_value = None

    # --- メッシュ生成 ---
    mesh = make_twisted_wire_mesh(
        n_strands,
        p["wire_diameter"],
        p["pitch"],
        length=0.0,
        n_elems_per_strand=p["n_elems_per_strand"],
        n_pitches=p["n_pitches"],
        strand_diameter=sd_value,
    )

    # --- ノードセット: 素線ごと + 固定端 + 自由端 ---
    nsets: dict[str, list[int]] = {}
    fixed_nodes: list[int] = []
    free_end_nodes: list[int] = []

    for sid in range(mesh.n_strands):
        nodes = mesh.strand_nodes(sid)
        nsets[f"STRAND_{sid}"] = [int(n) + 1 for n in nodes]  # 1-based
        fixed_nodes.append(int(nodes[0]) + 1)
        free_end_nodes.append(int(nodes[-1]) + 1)

    nsets["FIXED_END"] = sorted(set(fixed_nodes))
    nsets["FREE_END"] = sorted(set(free_end_nodes))

    # --- 要素セット: 素線ごと ---
    elsets: dict[str, list[int]] = {}
    for sid in range(mesh.n_strands):
        elems = mesh.strand_elems(sid)
        elsets[f"STRAND_{sid}"] = [int(e) + 1 for e in elems]  # 1-based

    # --- 断面寸法 ---
    r = p["wire_diameter"] / 2.0
    beam_section_dims = [r]

    # --- 接触定義（Abaqus General Contact 互換）---
    contact_alg = "NCP" if p["use_ncp"] else "AL"

    # サーフェス定義: 各素線を ELSET ベースのサーフェスとして定義
    contact_surfaces: list[InpSurfaceDef] = []
    for sid in range(mesh.n_strands):
        contact_surfaces.append(
            InpSurfaceDef(
                name=f"SURF_STRAND_{sid}",
                type="ELEMENT",
                elset=f"STRAND_{sid}",
            )
        )

    # サーフェスインタラクション定義
    interaction_name = "CONTACT_PROP"
    contact_interaction = InpSurfaceInteraction(
        name=interaction_name,
        pressure_overclosure="HARD",
        k_pen=p.get("ncp_k_pen", 0.0),
        friction=p.get("mu", 0.0),
        algorithm=contact_alg,
        mortar=p.get("use_mortar", True),
    )

    # 接触ペア: 異なる素線間のみ（同層除外は独自拡張で処理）
    contact_def = InpContactDef(
        interaction=interaction_name,
        inclusions=[],  # 空 = ALLEXT, ALLEXT（全自己接触）
        exclude_same_layer=True,
        surfaces=contact_surfaces,
        surface_interactions=[contact_interaction],
    )

    # --- Step 1: 曲げ（静解析）---
    bending_bcs: list[tuple[int | str, int, int, float]] = []
    for node_label in nsets["FIXED_END"]:
        bending_bcs.append((node_label, 1, 6, 0.0))

    step_bend = InpStep(
        name="Bending",
        procedure="STATIC",
        nlgeom=True,
        unsymm=True,
        inc=1000,
        time_params=(1.0 / p["n_bending_steps"], 1.0, 1e-8, 1.0),
        boundaries=bending_bcs,
        contact=contact_def,
        output_requests=[
            InpOutputRequest(
                domain="FIELD",
                frequency=1,
                variables=["U", "RF", "S", "COORD"],
            ),
            InpOutputRequest(
                domain="HISTORY",
                frequency=1,
                variables=["ALLKE", "ALLIE", "ETOTAL"],
            ),
        ],
    )

    # --- Step 2: 揺動（動解析）---
    osc_bcs: list[tuple[int | str, int, int, float]] = []
    for node_label in nsets["FIXED_END"]:
        osc_bcs.append((node_label, 1, 6, 0.0))

    n_quarter = p["n_steps_per_quarter"]
    n_osc_steps = 4 * p["n_cycles"] * n_quarter
    total_osc_time = float(n_osc_steps)
    dt_osc = 1.0

    step_osc = InpStep(
        name="Oscillation",
        procedure="DYNAMIC",
        nlgeom=True,
        unsymm=True,
        inc=n_osc_steps * 10,
        time_params=(dt_osc, total_osc_time, 1e-8, dt_osc),
        boundaries=osc_bcs,
        boundary_type="DISPLACEMENT",
        contact=contact_def,
        output_requests=[
            InpOutputRequest(
                domain="FIELD",
                frequency=1,
                variables=["U", "V", "A", "RF", "S", "COORD"],
            ),
            InpOutputRequest(
                domain="HISTORY",
                frequency=1,
                variables=["ALLKE", "ALLIE", "ETOTAL"],
            ),
        ],
    )

    steps = [step_bend, step_osc]

    # --- xkep-cae メタデータ（ソルバー・荷重パラメータ）---
    metadata = {
        "xkep_version": "3.0",
        "problem_type": "bending_oscillation",
        "n_strands": n_strands,
        "wire_diameter": p["wire_diameter"],
        "pitch": p["pitch"],
        "strand_diameter": sd_value,
        "n_elems_per_strand": p["n_elems_per_strand"],
        "n_pitches": p["n_pitches"],
        # 材料（カテゴリB+C: .inp *ELASTIC と整合）
        "E": p["E"],
        "nu": p["nu"],
        # 荷重・ステップ
        "bend_angle_deg": p["bend_angle_deg"],
        "n_bending_steps": p["n_bending_steps"],
        "oscillation_amplitude_mm": p["oscillation_amplitude_mm"],
        "n_cycles": p["n_cycles"],
        "n_steps_per_quarter": p["n_steps_per_quarter"],
        # NR パラメータ
        "max_iter": p["max_iter"],
        "n_outer_max": p["n_outer_max"],
        "tol_force": p["tol_force"],
        # 接触基本パラメータ
        "auto_kpen": p["auto_kpen"],
        "use_friction": p["use_friction"],
        "mu": p["mu"],
        "k_pen_scaling": p["k_pen_scaling"],
        "penalty_growth_factor": p["penalty_growth_factor"],
        # NCP ソルバーパラメータ
        "use_ncp": p["use_ncp"],
        "use_mortar": p["use_mortar"],
        "n_gauss": p["n_gauss"],
        "ncp_k_pen": p["ncp_k_pen"],
        "augmented_threshold": p["augmented_threshold"],
        "saddle_regularization": p["saddle_regularization"],
        "ncp_active_threshold": p["ncp_active_threshold"],
        "lambda_relaxation": p["lambda_relaxation"],
        "max_step_cuts": p["max_step_cuts"],
        "modified_nr_threshold": p["modified_nr_threshold"],
        # カテゴリD: 接触パラメータ（従来ハードコード → メタデータ記録）
        "k_t_ratio": p["k_t_ratio"],
        "g_on": p["g_on"],
        "g_off": p["g_off"],
        "use_line_search": p["use_line_search"],
        "line_search_max_steps": p["line_search_max_steps"],
        "use_geometric_stiffness": p["use_geometric_stiffness"],
        "tol_penetration_ratio": p["tol_penetration_ratio"],
        "k_pen_max": p["k_pen_max"],
        "exclude_same_layer": p["exclude_same_layer"],
        "midpoint_prescreening": p["midpoint_prescreening"],
        "linear_solver": p["linear_solver"],
        "line_contact": p["line_contact"],
        # カテゴリE: 数値パラメータ（従来ハードコード → メタデータ記録）
        "broadphase_margin": p["broadphase_margin"],
        # メッシュ位相
        "strand_node_ranges": mesh.strand_node_ranges,
        "strand_elem_ranges": mesh.strand_elem_ranges,
    }

    # --- 鋼線密度 ---
    steel_density = 7850.0  # kg/m³

    # --- .inp 書き出し ---
    inp_path = write_abaqus_model(
        out_dir / f"model_{n_strands}strand.inp",
        mesh.node_coords,
        mesh.connectivity,
        elem_type="B31",
        title=f"xkep-cae bending oscillation {n_strands}-strand twisted wire",
        nsets=nsets,
        elsets=elsets,
        material_name="STEEL",
        E=p["E"],
        nu=p["nu"],
        density=steel_density,
        beam_section_type="CIRC",
        beam_section_dims=beam_section_dims,
        beam_section_direction=[0.0, 1.0, 0.0],
        steps=steps,
    )

    # --- メタデータをファイル末尾に追記 ---
    with open(inp_path, "a", encoding="utf-8") as f:
        f.write("**\n")
        f.write("** XKEP-CAE METADATA BEGIN\n")
        meta_json = json.dumps(metadata, indent=2, ensure_ascii=False)
        for line in meta_json.splitlines():
            f.write(f"** {line}\n")
        f.write("** XKEP-CAE METADATA END\n")

    return inp_path


# ====================================================================
# Phase 2: .inp からソルバー実行
# ====================================================================


def load_metadata_from_inp(inp_path: Path) -> dict:
    """Abaqus .inp ファイルから xkep-cae メタデータを読み出す."""
    text = inp_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    in_meta = False
    meta_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "** XKEP-CAE METADATA BEGIN":
            in_meta = True
            continue
        if stripped == "** XKEP-CAE METADATA END":
            break
        if in_meta and stripped.startswith("**"):
            # "** " プレフィクスを除去
            content = stripped[2:].strip() if len(stripped) > 2 else ""
            meta_lines.append(content)

    if not meta_lines:
        raise ValueError(f"{inp_path} に XKEP-CAE メタデータが見つかりません")

    return json.loads("\n".join(meta_lines))


def _validate_inp_vs_metadata(inp_path: Path, meta: dict) -> dict:
    """標準 Abaqus データとメタデータの整合を検証.

    .inp の *NODE, *ELEMENT, *ELASTIC 等を読み取り、メタデータと比較する。
    不整合がある場合は警告を出し、.inp の値を truth として返す。

    Returns:
        overrides: メタデータを上書きすべきパラメータ辞書
    """
    overrides: dict = {}
    warnings: list[str] = []

    try:
        abaqus_mesh = read_abaqus_inp(inp_path)
    except Exception as e:
        print(f"  [WARN] .inp 標準データの読み取りに失敗: {e}")
        return overrides

    # --- 材料: *ELASTIC → E, nu ---
    if abaqus_mesh.materials:
        mat = abaqus_mesh.materials[0]
        if mat.elastic is not None:
            inp_E, inp_nu = mat.elastic
            meta_E = meta.get("E", 200e9)
            meta_nu = meta.get("nu", 0.3)
            if abs(inp_E - meta_E) / max(abs(meta_E), 1.0) > 1e-10:
                warnings.append(f"E: .inp={inp_E:.6e} ≠ metadata={meta_E:.6e} → .inp を採用")
                overrides["E"] = inp_E
            if abs(inp_nu - meta_nu) > 1e-10:
                warnings.append(f"nu: .inp={inp_nu} ≠ metadata={meta_nu} → .inp を採用")
                overrides["nu"] = inp_nu

    # --- 節点数・要素数の整合チェック ---
    n_nodes_inp = len(abaqus_mesh.nodes)
    n_elems_inp = sum(len(g.elements) for g in abaqus_mesh.element_groups)
    n_strands = meta.get("n_strands", 0)
    n_elems_per_strand = meta.get("n_elems_per_strand", 0)
    expected_elems = n_strands * n_elems_per_strand if n_strands and n_elems_per_strand else 0
    expected_nodes = n_strands * (n_elems_per_strand + 1) if n_strands and n_elems_per_strand else 0

    if expected_elems > 0 and n_elems_inp != expected_elems:
        warnings.append(f"要素数: .inp={n_elems_inp} ≠ expected={expected_elems}")
    if expected_nodes > 0 and n_nodes_inp != expected_nodes:
        warnings.append(f"節点数: .inp={n_nodes_inp} ≠ expected={expected_nodes}")

    # --- 境界条件の整合チェック ---
    n_bc_inp = len(abaqus_mesh.boundaries)
    if n_bc_inp > 0:
        fixed_nodes = {bc.node_label for bc in abaqus_mesh.boundaries}
        overrides["_inp_n_fixed_nodes"] = len(fixed_nodes)

    # --- 断面: *BEAM SECTION ---
    if abaqus_mesh.beam_sections:
        bsec = abaqus_mesh.beam_sections[0]
        if bsec.section_type.upper() == "CIRC" and bsec.dimensions:
            inp_radius = bsec.dimensions[0]
            meta_d = meta.get("wire_diameter", 0.002)
            if abs(inp_radius - meta_d / 2.0) / max(meta_d / 2.0, 1e-10) > 1e-6:
                warnings.append(f"断面半径: .inp={inp_radius} ≠ metadata d/2={meta_d / 2.0}")

    # --- 節点座標の差分チェック（サンプリング検証）---
    if n_nodes_inp > 0 and abaqus_mesh.nodes:
        # 最初と最後の節点座標を検証用に記録
        first_node = abaqus_mesh.nodes[0]
        last_node = abaqus_mesh.nodes[-1]
        overrides["_inp_first_node"] = (first_node.x, first_node.y, first_node.z)
        overrides["_inp_last_node"] = (last_node.x, last_node.y, last_node.z)
        overrides["_inp_n_nodes"] = n_nodes_inp
        overrides["_inp_n_elems"] = n_elems_inp

    if warnings:
        print("  [.inp 検証] 不整合検出:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  [.inp 検証] メタデータと標準データの整合: OK")

    return overrides


def solve_from_inp(
    inp_path: Path,
    out_dir: Path,
    *,
    show_progress: bool = True,
) -> BendingOscillationResult | None:
    """Abaqus .inp ファイルから曲げ揺動を実行.

    .inp のメッシュ + メタデータを読み込み、
    run_bending_oscillation に橋渡しする。
    標準 Abaqus データ（*ELASTIC 等）とメタデータの整合を検証し、
    不整合時は .inp の値を truth として採用する。

    Args:
        inp_path: .inp ファイルパス
        out_dir: 結果出力ディレクトリ
        show_progress: 進捗表示

    Returns:
        BendingOscillationResult（失敗時 None）
    """
    meta = load_metadata_from_inp(inp_path)

    if meta.get("problem_type") != "bending_oscillation":
        raise ValueError(
            f"problem_type が 'bending_oscillation' ではありません: {meta.get('problem_type')}"
        )

    # カテゴリA: .inp 標準データの読み取りとメタデータ検証
    overrides = _validate_inp_vs_metadata(inp_path, meta)
    # .inp の値で上書き（_inp_ プレフィックスの内部データは除外）
    for k, v in overrides.items():
        if not k.startswith("_inp_"):
            meta[k] = v

    n_strands = meta["n_strands"]
    print(f"\n  .inp ロード: {inp_path}")
    print(f"  素線数: {n_strands}, 要素/素線: {meta['n_elems_per_strand']}")

    t0 = time.perf_counter()
    try:
        result = run_bending_oscillation(
            n_strands=n_strands,
            wire_diameter=meta["wire_diameter"],
            pitch=meta["pitch"],
            n_elems_per_strand=meta["n_elems_per_strand"],
            n_pitches=meta["n_pitches"],
            strand_diameter=meta.get("strand_diameter"),
            # 材料パラメータ（カテゴリB+C: メタデータから読み取り）
            E=meta.get("E", 200e9),
            nu=meta.get("nu", 0.3),
            # 荷重・ステップ
            bend_angle_deg=meta["bend_angle_deg"],
            n_bending_steps=meta["n_bending_steps"],
            oscillation_amplitude_mm=meta["oscillation_amplitude_mm"],
            n_cycles=meta["n_cycles"],
            n_steps_per_quarter=meta["n_steps_per_quarter"],
            # NR パラメータ
            max_iter=meta["max_iter"],
            n_outer_max=meta["n_outer_max"],
            tol_force=meta["tol_force"],
            # 接触基本パラメータ
            auto_kpen=meta.get("auto_kpen", True),
            use_friction=meta.get("use_friction", False),
            mu=meta.get("mu", 0.0),
            k_pen_scaling=meta.get("k_pen_scaling", "sqrt"),
            penalty_growth_factor=meta.get("penalty_growth_factor", 4.0),
            show_progress=show_progress,
            gif_output_dir=None,
            gif_snapshot_interval=1,
            # NCP ソルバーパラメータ
            use_ncp=meta.get("use_ncp", False),
            use_mortar=meta.get("use_mortar", True),
            n_gauss=meta.get("n_gauss", 2),
            ncp_k_pen=meta.get("ncp_k_pen", 0.0),
            augmented_threshold=meta.get("augmented_threshold", 20),
            saddle_regularization=meta.get("saddle_regularization", 0.0),
            ncp_active_threshold=meta.get("ncp_active_threshold", 0.0),
            lambda_relaxation=meta.get("lambda_relaxation", 1.0),
            max_step_cuts=meta.get("max_step_cuts", 3),
            modified_nr_threshold=meta.get("modified_nr_threshold", 5),
            # カテゴリD: 接触パラメータ（メタデータから読み取り）
            k_t_ratio=meta.get("k_t_ratio", 0.1),
            g_on=meta.get("g_on", 0.0),
            g_off=meta.get("g_off", 1e-5),
            use_line_search=meta.get("use_line_search", True),
            line_search_max_steps=meta.get("line_search_max_steps", 5),
            use_geometric_stiffness=meta.get("use_geometric_stiffness", True),
            tol_penetration_ratio=meta.get("tol_penetration_ratio", 0.02),
            k_pen_max=meta.get("k_pen_max", 1e12),
            exclude_same_layer=meta.get("exclude_same_layer", True),
            midpoint_prescreening=meta.get("midpoint_prescreening", True),
            linear_solver=meta.get("linear_solver", "auto"),
            line_contact=meta.get("line_contact", True),
            # カテゴリE: 数値パラメータ（メタデータから読み取り）
            broadphase_margin=meta.get("broadphase_margin", 0.01),
        )
    except Exception as e:
        print(f"\n  [ERROR] ソルバー失敗: {e}")
        import traceback

        traceback.print_exc()
        return None

    elapsed = time.perf_counter() - t0
    print(f"\n  計算完了: {elapsed:.1f}s")
    print(f"  Phase1 収束={result.phase1_converged}, Phase2 収束={result.phase2_converged}")

    # メッシュ再構築（エクスポートフック用）
    mesh = make_twisted_wire_mesh(
        n_strands,
        meta["wire_diameter"],
        meta["pitch"],
        length=0.0,
        n_elems_per_strand=meta["n_elems_per_strand"],
        n_pitches=meta["n_pitches"],
        strand_diameter=meta.get("strand_diameter"),
    )

    # --- エクスポートフック ---
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n  エクスポート:")
    _hook_summary(result, out_dir)
    _hook_vtk(result, mesh, out_dir)
    _hook_gif(result, mesh, out_dir)
    _hook_contact_graph(result, mesh, out_dir)

    return result


# ====================================================================
# エクスポートフック
# ====================================================================


def _hook_vtk(
    result: BendingOscillationResult,
    mesh: TwistedWireMesh,
    out_dir: Path,
) -> str | None:
    """VTK (.vtu + .pvd) エクスポート."""
    vtk_dir = out_dir / "vtk"
    vtk_dir.mkdir(parents=True, exist_ok=True)

    db = OutputDatabase(
        node_coords=mesh.node_coords,
        connectivity=[(VTK_LINE, mesh.connectivity)],
        ndof_per_node=6,
    )

    frames = []
    for i, (u, _label) in enumerate(
        zip(result.displacement_snapshots, result.snapshot_labels, strict=True)
    ):
        frames.append(Frame(frame_index=i, time=float(i), displacement=u))

    n_frames = len(frames)
    if n_frames == 0:
        print("    [VTK] スナップショットなし — スキップ")
        return None

    step = Step(name="bending_oscillation", total_time=float(n_frames), dt=1.0)
    sr = StepResult(step=step, step_index=0, frames=frames)
    db.step_results.append(sr)

    pvd_path = export_vtk(db, vtk_dir, prefix="result")
    print(f"    [VTK] {pvd_path} ({n_frames} frames)")
    return pvd_path


def _hook_gif(
    result: BendingOscillationResult,
    mesh: TwistedWireMesh,
    out_dir: Path,
) -> list[Path]:
    """変位 GIF エクスポート."""
    if not result.displacement_snapshots:
        print("    [GIF] スナップショットなし — スキップ")
        return []

    try:
        gif_paths = export_bending_oscillation_gif(
            mesh,
            result.displacement_snapshots,
            result.snapshot_labels,
            out_dir,
            prefix=f"bending_oscillation_{result.n_strands}strand",
            views=["yz", "xz"],
            figsize=(10.0, 8.0),
            dpi=80,
            duration=300,
        )
        for p in gif_paths:
            print(f"    [GIF] {p}")
        return gif_paths
    except ImportError:
        print("    [GIF] matplotlib/Pillow 未インストール — スキップ")
        return []


def _hook_contact_graph(
    result: BendingOscillationResult,
    mesh: TwistedWireMesh,
    out_dir: Path,
) -> Path | None:
    """接触グラフ GIF エクスポート."""
    graph_history = None

    if hasattr(result.phase1_result, "graph_history"):
        graph_history = result.phase1_result.graph_history

    for r2 in result.phase2_results:
        if hasattr(r2, "graph_history") and r2.graph_history is not None:
            if graph_history is None:
                graph_history = r2.graph_history
            else:
                for snap in r2.graph_history.snapshots:
                    graph_history.add_snapshot(snap)

    if graph_history is None or graph_history.n_steps == 0:
        print("    [接触グラフ] データなし — スキップ")
        return None

    gif_path = out_dir / f"contact_graph_{result.n_strands}strand.gif"
    try:
        save_contact_graph_gif(
            graph_history,
            str(gif_path),
            fps=2,
            figsize=(8, 6),
            dpi=80,
        )
        print(f"    [接触グラフ] {gif_path} ({graph_history.n_steps} snapshots)")
        return gif_path
    except ImportError:
        print("    [接触グラフ] matplotlib/Pillow 未インストール — スキップ")
        return None
    except Exception as e:
        print(f"    [接触グラフ] エラー: {e}")
        return None


def _hook_summary(
    result: BendingOscillationResult,
    out_dir: Path,
) -> Path:
    """テキストサマリー出力."""
    summary_path = out_dir / "summary.txt"
    lines = [
        f"撚線曲げ揺動結果サマリー: {result.n_strands}本",
        f"{'=' * 50}",
        f"要素数:       {result.n_elems}",
        f"節点数:       {result.n_nodes}",
        f"自由度数:     {result.ndof}",
        f"モデル長さ:   {result.mesh_length * 1000:.1f} mm",
        "",
        "Phase 1（曲げ）:",
        f"  収束:       {result.phase1_converged}",
        f"  NR反復:     {result.phase1_result.total_newton_iterations}",
        "",
        "Phase 2（揺動）:",
        f"  収束:       {result.phase2_converged}",
        f"  ステップ数: {len(result.phase2_results)}",
        "",
        f"先端変位:     ({result.tip_displacement_final[0] * 1000:.3f}, "
        f"{result.tip_displacement_final[1] * 1000:.3f}, "
        f"{result.tip_displacement_final[2] * 1000:.3f}) mm",
        f"最大貫入比:   {result.max_penetration_ratio:.6f}",
        f"活性接触ペア: {result.n_active_contacts}",
        f"計算時間:     {result.total_time_s:.1f} s",
        "",
        f"スナップショット数: {len(result.displacement_snapshots)}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"    [Summary] {summary_path}")
    return summary_path


# ====================================================================
# CLI
# ====================================================================


def cmd_export(args):
    """export サブコマンド: .inp 生成."""
    strand_counts = [int(s.strip()) for s in args.strands.split(",")]
    out_root = Path(args.outdir)
    params = dict(DEFAULT_PARAMS)
    _apply_ncp_params(args, params)

    print("== export モード: .inp 生成 ==")
    print(f"  素線数: {strand_counts}")
    print(f"  出力先: {out_root.resolve()}")
    if params.get("use_ncp"):
        print("  ソルバー: NCP")

    for n in strand_counts:
        out_dir = out_root / f"{n}strand"
        inp_path = export_bending_oscillation_inp(n, out_dir, params=params)
        print(f"  [{n}本] {inp_path}")


def cmd_solve(args):
    """solve サブコマンド: .inp からソルバー実行."""
    inp_dir = Path(args.inpdir)
    out_root = Path(args.outdir) if args.outdir else inp_dir

    # .inp ファイルを検索
    inp_files = sorted(inp_dir.rglob("model_*strand.inp"))
    if not inp_files:
        print(f"  [ERROR] {inp_dir} に model_*strand.inp が見つかりません")
        sys.exit(1)

    print("== solve モード: .inp からソルバー実行 ==")
    print(f"  入力: {inp_dir.resolve()}")
    print(f"  出力: {out_root.resolve()}")
    print(f"  .inp ファイル: {len(inp_files)}個")

    results: list[tuple[str, BendingOscillationResult | None]] = []
    for inp_path in inp_files:
        # 出力先は .inp と同じディレクトリ（or --outdir 配下）
        rel = inp_path.parent.relative_to(inp_dir)
        solve_out = out_root / rel

        r = solve_from_inp(inp_path, solve_out, show_progress=True)
        results.append((inp_path.name, r))

    _print_summary(results)


def cmd_all(args):
    """all サブコマンド: export → solve 連続実行."""
    strand_counts = [int(s.strip()) for s in args.strands.split(",")]
    out_root = Path(args.outdir)
    params = dict(DEFAULT_PARAMS)
    _apply_ncp_params(args, params)

    print("== all モード: export → solve ==")
    print(f"  素線数: {strand_counts}")
    print(f"  出力先: {out_root.resolve()}")
    if params.get("use_ncp"):
        print("  ソルバー: NCP")

    results: list[tuple[str, BendingOscillationResult | None]] = []
    for n in strand_counts:
        out_dir = out_root / f"{n}strand"

        # Phase 1: export
        print(f"\n{'#' * 70}")
        print(f"# {n}本撚線 — export")
        print(f"{'#' * 70}")
        inp_path = export_bending_oscillation_inp(n, out_dir, params=params)
        print(f"  .inp: {inp_path}")

        # Phase 2: solve
        print(f"\n{'#' * 70}")
        print(f"# {n}本撚線 — solve")
        print(f"{'#' * 70}")
        r = solve_from_inp(inp_path, out_dir, show_progress=True)
        results.append((inp_path.name, r))

    _print_summary(results)


def _print_summary(results: list[tuple[str, BendingOscillationResult | None]]):
    """全体サマリーテーブルを表示."""
    print(f"\n{'=' * 70}")
    print("  全体サマリー")
    print(f"{'=' * 70}")
    print(f"{'ファイル':<30} {'要素':>6} {'DOF':>8} {'Phase1':>8} {'Phase2':>8} {'時間[s]':>10}")
    print(f"{'-' * 70}")
    for name, r in results:
        if r is None:
            print(f"{name:<30} {'FAILED':>6}")
            continue
        p1 = "OK" if r.phase1_converged else "NG"
        p2 = "OK" if r.phase2_converged else "NG"
        print(f"{name:<30} {r.n_elems:>6} {r.ndof:>8} {p1:>8} {p2:>8} {r.total_time_s:>10.1f}")


def _add_ncp_args(p):
    """argparse サブパーサーに NCP ソルバー・非貫入配置オプションを追加."""
    g = p.add_argument_group("NCP ソルバーオプション")
    g.add_argument(
        "--ncp", action="store_true", default=True, help="NCP ソルバーを使用（デフォルト: 有効）"
    )
    g.add_argument(
        "--no-ncp", action="store_true", default=False, help="NCP を無効化し AL 法を使用"
    )
    g.add_argument("--mortar", action="store_true", default=True, help="Mortar 積分")
    g.add_argument("--n-gauss", type=int, default=2, help="Gauss 積分点数")
    g.add_argument("--ncp-k-pen", type=float, default=0.0, help="NCP ペナルティ剛性（0=自動）")
    g.add_argument("--max-step-cuts", type=int, default=3, help="ステップ二分法の最大深度")
    g.add_argument("--modified-nr-threshold", type=int, default=5, help="修正NR法の閾値")
    g.add_argument("--k-pen-scaling", type=str, default="sqrt", help="k_pen スケーリング")
    g.add_argument("--penalty-growth", type=float, default=4.0, help="ペナルティ成長係数")
    g2 = p.add_argument_group("非貫入配置オプション")
    g2.add_argument(
        "--strand-diameter",
        type=str,
        default="auto",
        help="撚線外径 [m]（'auto'=最小外径自動計算, '0'=従来配置, 数値=指定値）",
    )


def _apply_ncp_params(args, params):
    """CLI の NCP・非貫入配置引数を params 辞書に反映."""
    # NCP: --no-ncp で明示無効化、それ以外はデフォルト（True）
    if hasattr(args, "no_ncp") and args.no_ncp:
        params["use_ncp"] = False
    elif hasattr(args, "ncp"):
        params["use_ncp"] = args.ncp
    if hasattr(args, "mortar"):
        params["use_mortar"] = args.mortar
    if hasattr(args, "n_gauss"):
        params["n_gauss"] = args.n_gauss
    if hasattr(args, "ncp_k_pen"):
        params["ncp_k_pen"] = args.ncp_k_pen
    if hasattr(args, "max_step_cuts"):
        params["max_step_cuts"] = args.max_step_cuts
    if hasattr(args, "modified_nr_threshold"):
        params["modified_nr_threshold"] = args.modified_nr_threshold
    if hasattr(args, "k_pen_scaling"):
        params["k_pen_scaling"] = args.k_pen_scaling
    if hasattr(args, "penalty_growth"):
        params["penalty_growth_factor"] = args.penalty_growth
    # 非貫入配置
    if hasattr(args, "strand_diameter"):
        params["strand_diameter"] = args.strand_diameter


def main():
    parser = argparse.ArgumentParser(
        description="撚線曲げ揺動 — プリ/ソルブ分離スクリプト",
    )
    subparsers = parser.add_subparsers(dest="command", help="実行モード")

    # --- export ---
    p_export = subparsers.add_parser("export", help=".inp ファイルを生成")
    p_export.add_argument(
        "--strands",
        type=str,
        default=",".join(str(s) for s in STRAND_COUNTS_DEFAULT),
        help="素線数リスト（カンマ区切り）",
    )
    p_export.add_argument("--outdir", type=str, default="scripts/results/bending", help="出力先")
    _add_ncp_args(p_export)
    p_export.set_defaults(func=cmd_export)

    # --- solve ---
    p_solve = subparsers.add_parser("solve", help=".inp からソルバー実行")
    p_solve.add_argument(
        "--inpdir",
        type=str,
        default="scripts/results/bending",
        help=".inp 検索ディレクトリ",
    )
    p_solve.add_argument(
        "--outdir", type=str, default=None, help="結果出力先（省略で inpdir と同じ）"
    )
    p_solve.set_defaults(func=cmd_solve)

    # --- all ---
    p_all = subparsers.add_parser("all", help="export → solve 連続実行（デフォルト）")
    p_all.add_argument(
        "--strands",
        type=str,
        default=",".join(str(s) for s in STRAND_COUNTS_DEFAULT),
        help="素線数リスト（カンマ区切り）",
    )
    p_all.add_argument("--outdir", type=str, default="scripts/results/bending", help="出力先")
    _add_ncp_args(p_all)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()

    if args.command is None:
        # デフォルトは all
        args.command = "all"
        args.strands = ",".join(str(s) for s in STRAND_COUNTS_DEFAULT)
        args.outdir = "scripts/results/bending"
        cmd_all(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
