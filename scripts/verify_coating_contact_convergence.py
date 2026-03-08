"""被膜接触モデルの収束検証スクリプト.

status-137 TODO: 被膜スプリングモデルの実問題での収束検証。
3ケースの接触パターンを7本撚線45度曲げで検証する。

ケース:
  1. 被膜なし同士（ベースライン）
  2. 被膜あり同士（被膜スプリングモデル有効）
  3. 被膜なし＋あり混合ペア

Usage:
  python scripts/verify_coating_contact_convergence.py 2>&1 | tee /tmp/verify_coating_contact.log

[← README](../README.md)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# --- ログ tee設定 ---
log_path = f"/tmp/verify_coating_contact_{int(time.time())}.log"


class TeeWriter:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


tee = TeeWriter(log_path)
sys.stdout = tee

print("=" * 70)
print("  被膜接触モデル 収束検証")
print("  status-137 TODO: 被膜スプリングモデルの実問題での収束検証")
print("=" * 70)
print(f"ログ出力先: {log_path}")
print(f"日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

from xkep_cae.numerical_tests.wire_bending_benchmark import run_bending_oscillation

# 共通パラメータ
COMMON_PARAMS = dict(
    n_strands=7,
    n_pitches=0.5,
    bend_angle_deg=45,
    use_ncp=True,
    use_mortar=True,
    n_gauss=2,
    max_iter=30,
    tol_force=1e-6,
    adaptive_timestepping=True,
    use_updated_lagrangian=True,
    show_progress=True,
    n_cycles=0,
)

results = {}

# ==================================================================
# ケース1: 被膜なし同士（ベースライン）
# ==================================================================
print("=" * 70)
print("  ケース1: 被膜なし同士（ベースライン）")
print("=" * 70)

t0 = time.perf_counter()
result_bare = run_bending_oscillation(
    **COMMON_PARAMS,
    coating_thickness=0.0,
    coating_stiffness=0.0,
)
t_bare = time.perf_counter() - t0

print(f"\n結果: converged={result_bare.phase1_converged}, 時間={t_bare:.2f}s")
print(f"  NR反復: {result_bare.phase1_result.total_newton_iterations}")
print(f"  最大貫入比: {result_bare.max_penetration_ratio:.6f}")
print(f"  活性接触ペア: {result_bare.n_active_contacts}")
results["bare"] = (result_bare, t_bare)

# ==================================================================
# ケース2: 被膜あり同士（被膜スプリングモデル有効）
# ==================================================================
print()
print("=" * 70)
print("  ケース2: 被膜あり同士（coating_thickness=0.1mm, k_coat=1e8）")
print("=" * 70)

COAT_T = 0.0001  # 100μm
COAT_K = 1e8  # Pa/m

t0 = time.perf_counter()
result_coated = run_bending_oscillation(
    **COMMON_PARAMS,
    coating_thickness=COAT_T,
    coating_stiffness=COAT_K,
)
t_coated = time.perf_counter() - t0

print(f"\n結果: converged={result_coated.phase1_converged}, 時間={t_coated:.2f}s")
print(f"  NR反復: {result_coated.phase1_result.total_newton_iterations}")
print(f"  最大貫入比: {result_coated.max_penetration_ratio:.6f}")
print(f"  活性接触ペア: {result_coated.n_active_contacts}")
results["coated"] = (result_coated, t_coated)

# ==================================================================
# ケース3: 被膜なし＋あり混合
# 被膜パラメータありでメッシュを作るが、coating_stiffnessを低くして
# 被膜接触の柔軟性を確認
# ==================================================================
print()
print("=" * 70)
print("  ケース3: 被膜あり（低剛性, k_coat=1e6）")
print("=" * 70)

t0 = time.perf_counter()
result_soft_coat = run_bending_oscillation(
    **COMMON_PARAMS,
    coating_thickness=COAT_T,
    coating_stiffness=1e6,  # 低剛性被膜
)
t_soft = time.perf_counter() - t0

print(f"\n結果: converged={result_soft_coat.phase1_converged}, 時間={t_soft:.2f}s")
print(f"  NR反復: {result_soft_coat.phase1_result.total_newton_iterations}")
print(f"  最大貫入比: {result_soft_coat.max_penetration_ratio:.6f}")
print(f"  活性接触ペア: {result_soft_coat.n_active_contacts}")
results["soft_coat"] = (result_soft_coat, t_soft)

# ==================================================================
# 3D梁表面の2D投影スナップショット
# ==================================================================
print()
print("=" * 70)
print("  3D梁表面の2D投影スナップショット生成")
print("=" * 70)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection

    from xkep_cae.math.quaternion import (
        quat_from_axis_angle,
        quat_multiply,
        quat_to_rotation_matrix,
    )
    from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh

    def _project_3d_to_2d(coords_3d, elev_deg=25.0, azim_deg=45.0):
        """四元数で3D座標を任意視点に回転し、XY平面に投影."""
        elev = np.deg2rad(elev_deg)
        azim = np.deg2rad(azim_deg)
        q_azim = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), -azim)
        q_elev = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), -elev)
        q_view = quat_multiply(q_elev, q_azim)
        R = quat_to_rotation_matrix(q_view)
        rotated = (R @ coords_3d.T).T
        return rotated[:, :2], rotated[:, 2]

    def _beam_surface_polys_2d(coords, radius, n_circ=12, elev_deg=25.0, azim_deg=45.0):
        """梁中心線から円管表面メッシュを生成し、2D投影された四角形リストを返す."""
        n_pts = len(coords)
        theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
        surface_pts = []
        for i in range(n_pts):
            if i == 0:
                tang = coords[1] - coords[0]
            elif i == n_pts - 1:
                tang = coords[-1] - coords[-2]
            else:
                tang = coords[i + 1] - coords[i - 1]
            tang = tang / (np.linalg.norm(tang) + 1e-30)
            if abs(tang[2]) < 0.9:
                up = np.array([0.0, 0.0, 1.0])
            else:
                up = np.array([1.0, 0.0, 0.0])
            n1 = np.cross(tang, up)
            n1 = n1 / (np.linalg.norm(n1) + 1e-30)
            n2 = np.cross(tang, n1)
            for th in theta:
                offset = radius * (np.cos(th) * n1 + np.sin(th) * n2)
                surface_pts.append(coords[i] + offset)
        surface_pts = np.array(surface_pts)
        proj_2d, depth = _project_3d_to_2d(surface_pts, elev_deg, azim_deg)
        polys = []
        depths = []
        for i in range(n_pts - 1):
            for j in range(n_circ):
                j_next = (j + 1) % n_circ
                idx00 = i * n_circ + j
                idx01 = i * n_circ + j_next
                idx10 = (i + 1) * n_circ + j
                idx11 = (i + 1) * n_circ + j_next
                poly_verts = np.array(
                    [
                        proj_2d[idx00],
                        proj_2d[idx01],
                        proj_2d[idx11],
                        proj_2d[idx10],
                    ]
                )
                avg_depth = (depth[idx00] + depth[idx01] + depth[idx10] + depth[idx11]) / 4.0
                polys.append(poly_verts)
                depths.append(avg_depth)
        return polys, depths

    output_dir = Path("docs/verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 層ごとの色分け
    LAYER_COLORS = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#984ea3"}
    ELEV, AZIM = 25.0, 45.0
    N_CIRC = 12
    R_WIRE = 0.001  # 1mm radius

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    case_names = ["被膜なし", "被膜あり(k=1e8)", "被膜あり(k=1e6)"]
    case_keys = ["bare", "coated", "soft_coat"]

    for ax, name, key in zip(axes, case_names, case_keys, strict=True):
        result, t_elapsed = results[key]
        coat_t = 0.0 if key == "bare" else COAT_T
        mesh = make_twisted_wire_mesh(
            7,
            0.002,
            0.040,
            length=0.0,
            n_elems_per_strand=8,
            n_pitches=0.5,
            coating_thickness=coat_t,
        )

        # 変形後座標
        u = result.phase1_result.u
        deformed = mesh.node_coords.copy()
        for i in range(mesh.n_nodes):
            deformed[i, 0] += u[6 * i]
            deformed[i, 1] += u[6 * i + 1]
            deformed[i, 2] += u[6 * i + 2]

        # 全ストランドのポリゴンを収集（深度ソート用）
        all_polys = []
        all_depths = []
        all_colors = []
        for sid in range(mesh.n_strands):
            ns, ne = mesh.strand_node_ranges[sid]
            strand_coords = deformed[ns:ne]
            layer = mesh.strand_infos[sid].layer
            color = LAYER_COLORS.get(layer, "#999999")
            polys, depths = _beam_surface_polys_2d(strand_coords, R_WIRE, N_CIRC, ELEV, AZIM)
            all_polys.extend(polys)
            all_depths.extend(depths)
            all_colors.extend([color] * len(polys))

        # 深度順にソート（奥→手前）
        sorted_idx = np.argsort(all_depths)
        sorted_polys = [all_polys[i] for i in sorted_idx]
        sorted_colors = [all_colors[i] for i in sorted_idx]

        pc = PolyCollection(
            sorted_polys,
            facecolors=sorted_colors,
            edgecolors="none",
            alpha=0.8,
        )
        ax.add_collection(pc)

        # 軸範囲設定
        all_verts = np.concatenate(sorted_polys, axis=0)
        xr = all_verts[:, 0].max() - all_verts[:, 0].min()
        yr = all_verts[:, 1].max() - all_verts[:, 1].min()
        margin = max(xr, yr) * 0.05
        ax.set_xlim(all_verts[:, 0].min() - margin, all_verts[:, 0].max() + margin)
        ax.set_ylim(all_verts[:, 1].min() - margin, all_verts[:, 1].max() + margin)
        ax.set_aspect("equal")

        conv_str = "PASS" if result.phase1_converged else "FAIL"
        nr_iter = result.phase1_result.total_newton_iterations
        pen_ratio = result.max_penetration_ratio
        ax.set_title(
            f"{name}\n{conv_str} | NR={nr_iter} | pen={pen_ratio:.4f} | {t_elapsed:.1f}s",
            fontsize=9,
        )
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x' [m]", fontsize=8)
        ax.set_ylabel("y' [m]", fontsize=8)

    fig.suptitle("被膜接触モデル 収束比較（7本45°曲げ, 3D梁表面2D投影）", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_dir / "coating_contact_convergence.png"), dpi=150)
    plt.close(fig)
    print("  保存: docs/verification/coating_contact_convergence.png")

except ImportError:
    print("  matplotlib が利用不可能。2D投影スキップ。")

# ==================================================================
# サマリー
# ==================================================================
print()
print("=" * 70)
print("  検証サマリー")
print("=" * 70)

all_pass = True
for key, label in [
    ("bare", "被膜なし"),
    ("coated", "被膜あり(k=1e8)"),
    ("soft_coat", "被膜あり(k=1e6)"),
]:
    result, t_elapsed = results[key]
    status = "PASS" if result.phase1_converged else "FAIL"
    if not result.phase1_converged:
        all_pass = False
    print(
        f"  {label:20s}: {status}  "
        f"NR={result.phase1_result.total_newton_iterations:3d}  "
        f"pen_ratio={result.max_penetration_ratio:.6f}  "
        f"time={t_elapsed:.1f}s"
    )

print(f"\n  総合: {'ALL PASS' if all_pass else 'FAIL'}")
print(f"  ログ: {log_path}")

sys.stdout = tee.terminal
tee.close()
print(f"\n検証完了。ログ: {log_path}")

if not all_pass:
    sys.exit(1)
