"""三点曲げ非線形動解析 — Abaqus比較スクリプト.

Abaqus三点曲げモデル（assets/test_assets/Abaqus/1-bend3p/ idx1: 弾性）と同条件で
xkep-cae の非線形動解析を実施し、結果を GIF アニメーションと比較プロットで出力する。

モデル仕様（Abaqus idx1 準拠）:
  - yz対称1/2モデル（L_half = 50mm, support at x = 25mm）
  - 円形断面 d = 1mm
  - E = 100 GPa, ν = 0.3, ρ = 8.96e-9 ton/mm³（mm-ton-s 整合単位）
  - ランプ荷重（中央節点 y方向、準静的相当）

出力 (examples/output/nonlinear_bend3p/):
  - animation_xy.gif    — 変形アニメーション（xy ビュー）
  - force_displacement.png — 荷重-変位曲線比較
  - displacement_time.png  — 変位時刻歴
  - summary.txt         — 解析サマリー

Usage:
    python examples/run_nonlinear_bend3p.py
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = PROJECT_ROOT / "assets" / "test_assets" / "Abaqus" / "1-bend3p"
OUTPUT_DIR = PROJECT_ROOT / "examples" / "output" / "nonlinear_bend3p"

# ---------------------------------------------------------------------------
# xkep-cae imports
# ---------------------------------------------------------------------------
from xkep_cae.assembly import assemble_global_stiffness  # noqa: E402
from xkep_cae.dynamics import (  # noqa: E402
    NonlinearTransientConfig,
    solve_nonlinear_transient,
)
from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D  # noqa: E402
from xkep_cae.materials.beam_elastic import BeamElastic1D  # noqa: E402
from xkep_cae.numerical_tests.frequency import (  # noqa: E402
    _assemble_lumped_mass_3d,
)
from xkep_cae.sections.beam import BeamSection  # noqa: E402


# ---------------------------------------------------------------------------
# Abaqus データ読み込み
# ---------------------------------------------------------------------------
def load_abaqus_rf(csv_path: Path) -> dict[str, np.ndarray]:
    """Abaqus RF CSV を辞書として読み込む."""
    data: dict[str, list[float]] = {}
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, []).append(float(val))
    return {k: np.array(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# モデル構築
# ---------------------------------------------------------------------------
def build_model(
    n_elems: int = 100,
    L_half: float = 50.0,
    diameter: float = 1.0,
    E: float = 100e3,
    nu: float = 0.3,
    rho: float = 8.96e-9,
) -> dict:
    """Abaqus三点曲げ半モデルに対応する xkep-cae モデルを構築する.

    Returns:
        モデル辞書（nodes, conn, section, material, beam, fixed_dofs 等）
    """
    n_nodes = n_elems + 1
    nodes = np.zeros((n_nodes, 3), dtype=float)
    nodes[:, 0] = np.linspace(0.0, L_half, n_nodes)

    conn = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])

    section = BeamSection.circle(diameter)
    material = BeamElastic1D(E=E, nu=nu)
    G = E / (2.0 * (1.0 + nu))
    beam = TimoshenkoBeam3D(
        section=section,
        kappa_y="cowper",
        kappa_z="cowper",
        v_ref=np.array([0.0, 1.0, 0.0]),
    )

    ndof_per_node = 6
    ndof_total = ndof_per_node * n_nodes

    dx = L_half / n_elems
    support_idx = int(round(25.0 / dx))
    center_idx = 0

    # 境界条件
    fixed_dofs = set()
    # 対称BC (x=0): ux=0, uz=0, θx=0, θy=0, θz=0 (uy は自由)
    for dof in [0, 2, 3, 4, 5]:
        fixed_dofs.add(center_idx * ndof_per_node + dof)
    # 支持BC (x=25): uy=0
    fixed_dofs.add(support_idx * ndof_per_node + 1)

    center_uy_dof = center_idx * ndof_per_node + 1
    support_uy_dof = support_idx * ndof_per_node + 1

    return {
        "nodes": nodes,
        "conn": conn,
        "section": section,
        "material": material,
        "beam": beam,
        "G": G,
        "rho": rho,
        "n_nodes": n_nodes,
        "n_elems": n_elems,
        "ndof_per_node": ndof_per_node,
        "ndof_total": ndof_total,
        "fixed_dofs": np.array(sorted(fixed_dofs), dtype=int),
        "center_idx": center_idx,
        "support_idx": support_idx,
        "center_uy_dof": center_uy_dof,
        "support_uy_dof": support_uy_dof,
    }


# ---------------------------------------------------------------------------
# 非線形動解析の実行
# ---------------------------------------------------------------------------
def run_nonlinear_dynamic_bend3p(
    model: dict,
    *,
    F_target: float = -5.0,
    total_time: float = 100.0,
    n_steps: int = 200,
    damping_beta: float = 1e-3,
) -> dict:
    """三点曲げ半モデルの非線形動解析を実行する.

    ランプ荷重を中央節点に印加し、Newmark-β + NR で時間積分する。

    Args:
        model: build_model() の戻り値
        F_target: ランプ荷重の最終値 [N]（負=下向き）
        total_time: 解析時間 [s]
        n_steps: 時間ステップ数
        damping_beta: Rayleigh 減衰 β（剛性比例）

    Returns:
        結果辞書
    """
    nodes = model["nodes"]
    conn = model["conn"]
    beam = model["beam"]
    material = model["material"]
    section = model["section"]
    rho = model["rho"]
    ndof_total = model["ndof_total"]
    fixed_dofs = model["fixed_dofs"]
    center_uy_dof = model["center_uy_dof"]
    dt = total_time / n_steps

    # --- 剛性行列 ---
    import scipy.sparse as sp

    K_sp = assemble_global_stiffness(
        nodes,
        [(beam, conn)],
        material,
        show_progress=False,
    )
    K = K_sp.toarray() if sp.issparse(K_sp) else np.array(K_sp, dtype=float)

    # --- 質量行列（集中質量: HRZ法、対角） ---
    M = _assemble_lumped_mass_3d(
        nodes,
        conn,
        rho,
        section.A,
        section.Iy,
        section.Iz,
    )

    # --- Rayleigh 減衰（剛性比例のみ: 準静的安定化） ---
    C = damping_beta * K

    # --- 荷重関数（ランプ荷重） ---
    ramp_time = total_time  # 全時間かけて線形増加

    def f_ext(t: float) -> np.ndarray:
        f = np.zeros(ndof_total)
        scale = min(t / ramp_time, 1.0)
        f[center_uy_dof] = F_target * scale
        return f

    # --- 線形梁アセンブラー ---
    def assemble_internal_force(u: np.ndarray) -> np.ndarray:
        return K @ u

    def assemble_tangent(_u: np.ndarray) -> np.ndarray:
        return K

    # --- 初期条件 ---
    u0 = np.zeros(ndof_total)
    v0 = np.zeros(ndof_total)

    # --- 非線形動解析コンフィグ ---
    config = NonlinearTransientConfig(
        dt=dt,
        n_steps=n_steps,
        beta=0.25,  # 平均加速度法
        gamma=0.5,
        alpha_hht=0.0,  # 数値減衰なし
        max_iter=30,
        tol_force=1e-8,
    )

    print(f"  非線形動解析開始: dt={dt:.4f}s, {n_steps}ステップ, F_target={F_target:.3f}N")
    result = solve_nonlinear_transient(
        M=M,
        f_ext=f_ext,
        u0=u0,
        v0=v0,
        config=config,
        assemble_internal_force=assemble_internal_force,
        assemble_tangent=assemble_tangent,
        C=C,
        fixed_dofs=fixed_dofs,
        show_progress=True,
    )

    # --- 結果の抽出 ---
    center_uy = result.displacement[:, center_uy_dof]
    support_uy_dof_val = model["support_uy_dof"]

    # 支持点反力: R = K·u での support_uy 成分
    n_frames = result.displacement.shape[0]
    support_reaction = np.zeros(n_frames)
    for i in range(n_frames):
        R = K @ result.displacement[i]
        support_reaction[i] = R[support_uy_dof_val]

    # 印加荷重の時刻歴
    applied_force = np.array([F_target * min(t / ramp_time, 1.0) for t in result.time])

    return {
        "result": result,
        "time": result.time,
        "center_uy": center_uy,
        "support_reaction": support_reaction,
        "applied_force": applied_force,
        "converged": result.converged,
        "dt": dt,
        "K": K,
    }


# ---------------------------------------------------------------------------
# GIF アニメーション生成
# ---------------------------------------------------------------------------
def generate_gif(
    model: dict,
    analysis: dict,
    output_path: Path,
    *,
    n_gif_frames: int = 40,
    duration: int = 150,
) -> Path:
    """変形アニメーション GIF を生成する.

    Args:
        model: モデル辞書
        analysis: 解析結果辞書
        output_path: GIF 出力パス
        n_gif_frames: GIF のフレーム数
        duration: フレーム間隔 [ms]

    Returns:
        GIF ファイルパス
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    nodes = model["nodes"]
    result = analysis["result"]
    n_total_steps = result.displacement.shape[0]
    ndof_per_node = model["ndof_per_node"]
    n_nodes = model["n_nodes"]

    # フレーム間引き
    frame_indices = np.linspace(0, n_total_steps - 1, n_gif_frames, dtype=int)
    frame_indices = np.unique(frame_indices)

    # 全フレームの変形後座標を事前計算（描画範囲固定用）
    all_x, all_y = [], []
    deformed_frames = []
    for fi in frame_indices:
        u = result.displacement[fi]
        coords = nodes.copy()
        for k in range(n_nodes):
            coords[k, 0] += u[ndof_per_node * k + 0]  # ux
            coords[k, 1] += u[ndof_per_node * k + 1]  # uy
        deformed_frames.append(coords)
        all_x.extend(coords[:, 0].tolist())
        all_y.extend(coords[:, 1].tolist())

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    margin = 0.15
    xlim = (x_min - x_range * margin, x_max + x_range * margin)
    ylim = (y_min - y_range * margin, y_max + y_range * margin)

    # フレーム描画
    pil_images: list[PILImage.Image] = []
    for _i, (fi, coords) in enumerate(zip(frame_indices, deformed_frames, strict=True)):
        t = result.time[fi]
        uy_center = analysis["center_uy"][fi]

        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        # 初期形状（灰色破線）
        ax.plot(nodes[:, 0], nodes[:, 1], "k--", alpha=0.3, linewidth=1, label="Initial")
        # 変形形状（青）
        ax.plot(coords[:, 0], coords[:, 1], "b-", linewidth=2.0, label="Deformed")
        # 支持点マーカー
        si = model["support_idx"]
        ax.plot(coords[si, 0], coords[si, 1], "r^", markersize=10, label="Support")
        # 中央マーカー
        ci = model["center_idx"]
        ax.plot(coords[ci, 0], coords[ci, 1], "go", markersize=8, label="Center")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        ax.set_title(
            f"3-Point Bend NL Dynamic  t={t:.1f}s  uy_center={uy_center:.4f}mm",
        )
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pil_images.append(PILImage.open(buf).convert("RGB"))

    # GIF 書き出し
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(pil_images) > 1:
        pil_images[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0,
        )
    else:
        pil_images[0].save(output_path, format="GIF")

    return output_path


# ---------------------------------------------------------------------------
# 比較プロット生成
# ---------------------------------------------------------------------------
def generate_comparison_plots(
    model: dict,
    analysis: dict,
    output_dir: Path,
) -> list[Path]:
    """xkep-cae vs Abaqus の比較プロットを生成する.

    Returns:
        出力ファイルパスのリスト
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    # --- Abaqus データ読み込み ---
    abaqus_rf = None
    rf_path = ASSET_DIR / "go_idx1_RF.csv"
    if rf_path.exists():
        abaqus_rf = load_abaqus_rf(rf_path)

    # --- 1. 荷重-変位曲線 ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # xkep-cae: 中央変位 vs 支持点反力
    disp_xkep = -analysis["center_uy"]  # 符号反転で正値（下向き変位を正に）
    force_xkep = analysis["support_reaction"]

    ax.plot(disp_xkep, force_xkep, "b-", linewidth=2, label="xkep-cae (linear beam)")

    if abaqus_rf is not None:
        # Abaqus: stroke = 0.3 * t [mm], RF2 = support reaction
        t_abaqus = abaqus_rf["t"]
        disp_abaqus = 0.3 * t_abaqus  # center disp [mm]
        force_abaqus = -abaqus_rf["RF2"]  # sign flip
        ax.plot(disp_abaqus, force_abaqus, "r--", linewidth=2, label="Abaqus (NLGEOM)")

    ax.set_xlabel("Center displacement [mm]")
    ax.set_ylabel("Support reaction [N]")
    ax.set_title("3-Point Bend: Force-Displacement Comparison")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path_fd = output_dir / "force_displacement.png"
    fig.savefig(path_fd, bbox_inches="tight")
    plt.close(fig)
    output_files.append(path_fd)

    # --- 2. 変位時刻歴 ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    ax.plot(analysis["time"], analysis["center_uy"], "b-", linewidth=2, label="xkep-cae uy_center")
    ax.plot(
        analysis["time"], analysis["applied_force"], "r--", linewidth=1.5, label="F_applied [N]"
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [mm] / Force [N]")
    ax.set_title("3-Point Bend: Center Displacement & Force History")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path_dt = output_dir / "displacement_time.png"
    fig.savefig(path_dt, bbox_inches="tight")
    plt.close(fig)
    output_files.append(path_dt)

    return output_files


# ---------------------------------------------------------------------------
# サマリー出力
# ---------------------------------------------------------------------------
def write_summary(
    model: dict,
    analysis: dict,
    output_dir: Path,
) -> Path:
    """解析サマリーをテキスト出力する."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.txt"

    # 線形剛性
    K_linear = (
        abs(analysis["support_reaction"][-1]) / abs(analysis["center_uy"][-1])
        if abs(analysis["center_uy"][-1]) > 1e-15
        else 0.0
    )

    # Abaqus データとの比較
    abaqus_info = ""
    rf_path = ASSET_DIR / "go_idx1_RF.csv"
    if rf_path.exists():
        abaqus_rf = load_abaqus_rf(rf_path)
        t_abq = abaqus_rf["t"]
        rf2_abq = abaqus_rf["RF2"]
        # 初期線形領域の剛性（t=5~15s）
        mask = (t_abq >= 5) & (t_abq <= 15)
        if np.any(mask):
            disp_lin = 0.3 * t_abq[mask]
            rf_lin = np.abs(rf2_abq[mask])
            K_abaqus = np.mean(rf_lin / disp_lin)
            diff = abs(K_linear - K_abaqus) / K_abaqus * 100
            abaqus_info = (
                f"\nAbaqus 比較:\n"
                f"  Abaqus 線形剛性: {K_abaqus:.4f} N/mm\n"
                f"  相対差異:         {diff:.2f}%\n"
                f"  Abaqus 最大 RF2:  {np.max(np.abs(rf2_abq)):.4f} N (t=100s)\n"
            )

    lines = [
        "=" * 60,
        "三点曲げ非線形動解析 サマリー",
        "=" * 60,
        "",
        "モデル:",
        f"  半モデル長: {model['nodes'][-1, 0]:.1f} mm",
        f"  要素数:     {model['n_elems']}",
        f"  節点数:     {model['n_nodes']}",
        f"  総DOF:      {model['ndof_total']}",
        f"  断面:       円形 d={2 * math.sqrt(model['section'].A / math.pi):.3f} mm",
        f"  材料:       E={model['material'].E:.0f} MPa, nu={model['material'].nu}",
        f"  密度:       {model['rho']:.2e} ton/mm^3",
        "",
        "解析条件:",
        f"  解析時間:   {analysis['time'][-1]:.1f} s",
        f"  時間刻み:   {analysis['dt']:.4f} s",
        f"  ステップ数: {len(analysis['time']) - 1}",
        f"  荷重:       ランプ {analysis['applied_force'][-1]:.4f} N",
        f"  収束:       {'OK' if analysis['converged'] else 'NG'}",
        "",
        "結果:",
        f"  最終中央変位: {analysis['center_uy'][-1]:.6f} mm",
        f"  最終支持反力: {analysis['support_reaction'][-1]:.6f} N",
        f"  線形剛性:     {K_linear:.4f} N/mm",
        abaqus_info,
        "=" * 60,
    ]
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    print(text)
    return path


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main() -> None:
    """三点曲げ非線形動解析を実行し、GIF と比較プロットを出力する."""
    print("=" * 60)
    print("三点曲げ非線形動解析 — xkep-cae")
    print("=" * 60)

    # --- モデル構築 ---
    print("\n[1/5] モデル構築...")
    model = build_model(
        n_elems=100,
        L_half=50.0,
        diameter=1.0,
        E=100e3,
        nu=0.3,
        rho=8.96e-9,
    )
    print(f"  節点数: {model['n_nodes']}, 要素数: {model['n_elems']}, DOF: {model['ndof_total']}")

    # --- 非線形動解析 ---
    print("\n[2/5] 非線形動解析...")
    # Abaqus 準拠: 100秒、準静的ランプ荷重
    # F_target を Abaqus 最大 RF2 に合わせる（約 2.44 N）
    # xkep-cae は線形梁なので、小変形範囲で比較するため -5.0 N 程度
    analysis = run_nonlinear_dynamic_bend3p(
        model,
        F_target=-5.0,
        total_time=100.0,
        n_steps=200,
        damping_beta=1e-3,
    )
    print(f"  収束: {'OK' if analysis['converged'] else 'NG'}")
    print(f"  最終中央変位: {analysis['center_uy'][-1]:.6f} mm")

    # --- GIF アニメーション ---
    print("\n[3/5] GIF アニメーション生成...")
    gif_path = generate_gif(
        model,
        analysis,
        OUTPUT_DIR / "animation_xy.gif",
        n_gif_frames=40,
        duration=150,
    )
    print(f"  出力: {gif_path}")

    # --- 比較プロット ---
    print("\n[4/5] 比較プロット生成...")
    plot_paths = generate_comparison_plots(model, analysis, OUTPUT_DIR)
    for p in plot_paths:
        print(f"  出力: {p}")

    # --- サマリー ---
    print("\n[5/5] サマリー出力...")
    write_summary(model, analysis, OUTPUT_DIR)

    print(f"\n全出力ファイル: {OUTPUT_DIR}/")
    print("完了。")


if __name__ == "__main__":
    main()
