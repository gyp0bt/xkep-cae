"""3Dパイプメッシュの視野違い2D投影画像を生成し、ギャラリーMDを各フォルダに書き出す。

時間ステップではなく、異なる視角からの投影図を出力する。
- 正面3面（XY, XZ, YZ）
- 斜め投影（Isometric, 30°/60°回転）
- 断面図（z=25%, 50%, 75%）

Usage:
  python scripts/generate_multiview_gallery.py 2>&1 | tee /tmp/log-multiview-$(date +%s).log

[<- README](../README.md)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

log_path = f"/tmp/generate_multiview_{int(time.time())}.log"


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

print("=== Multi-view gallery generation ===")
print(f"Log: {log_path}")
print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh  # noqa: E402


# ---------------------------------------------------------------
# 回転行列ユーティリティ
# ---------------------------------------------------------------
def rotation_matrix_x(theta_deg: float) -> np.ndarray:
    t = np.radians(theta_deg)
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(t), -np.sin(t)],
            [0, np.sin(t), np.cos(t)],
        ]
    )


def rotation_matrix_y(theta_deg: float) -> np.ndarray:
    t = np.radians(theta_deg)
    return np.array(
        [
            [np.cos(t), 0, np.sin(t)],
            [0, 1, 0],
            [-np.sin(t), 0, np.cos(t)],
        ]
    )


def rotation_matrix_z(theta_deg: float) -> np.ndarray:
    t = np.radians(theta_deg)
    return np.array(
        [
            [np.cos(t), -np.sin(t), 0],
            [np.sin(t), np.cos(t), 0],
            [0, 0, 1],
        ]
    )


def project_coords(coords: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """回転行列を適用して (x', y') の2D座標を返す."""
    rotated = coords @ rot.T
    return rotated[:, :2]  # 投影面 = 最初の2軸


# ---------------------------------------------------------------
# 描画関数
# ---------------------------------------------------------------
def plot_projected_view(mesh_obj, coords_3d, ax, rot, title="", label_x="u", label_y="v"):
    """回転行列で3D座標を変換し、最初の2軸を2D投影としてプロットする."""
    proj = project_coords(coords_3d, rot)
    for sid in range(mesh_obj.n_strands):
        nodes = mesh_obj.strand_nodes(sid)
        c = f"C{sid % 10}"
        ax.plot(
            proj[nodes, 0] * 1000,
            proj[nodes, 1] * 1000,
            "-",
            color=c,
            linewidth=1.0,
            alpha=0.7,
        )
    ax.set_aspect("equal")
    ax.set_xlabel(f"{label_x} [mm]", fontsize=7)
    ax.set_ylabel(f"{label_y} [mm]", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)


def plot_cross_section(mesh_obj, coords, ax, z_frac=0.5, title=""):
    """指定z位置での断面図."""
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    z_target = z_min + (z_max - z_min) * z_frac

    for sid in range(mesh_obj.n_strands):
        nodes = mesh_obj.strand_nodes(sid)
        strand_coords = coords[nodes]
        z_vals = strand_coords[:, 2]
        for j in range(len(nodes) - 1):
            z0, z1 = z_vals[j], z_vals[j + 1]
            if (z0 <= z_target <= z1) or (z1 <= z_target <= z0):
                if abs(z1 - z0) < 1e-15:
                    t = 0.5
                else:
                    t = (z_target - z0) / (z1 - z0)
                x_i = strand_coords[j, 0] * (1 - t) + strand_coords[j + 1, 0] * t
                y_i = strand_coords[j, 1] * (1 - t) + strand_coords[j + 1, 1] * t
                c = f"C{sid % 10}"
                ax.plot(x_i * 1000, y_i * 1000, "o", color=c, markersize=5)
                break
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]", fontsize=7)
    ax.set_ylabel("y [mm]", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------
# 視角定義
# ---------------------------------------------------------------
VIEWS = {
    "front_xy": {
        "rot": np.eye(3),
        "title": "Front (XY)",
        "lx": "X",
        "ly": "Y",
    },
    "side_xz": {
        "rot": rotation_matrix_x(-90),
        "title": "Side (XZ)",
        "lx": "X",
        "ly": "Z",
    },
    "end_yz": {
        "rot": rotation_matrix_y(90),
        "title": "End (YZ)",
        "lx": "Y",
        "ly": "Z",
    },
    "iso_30": {
        "rot": rotation_matrix_y(30) @ rotation_matrix_x(-20),
        "title": "Oblique 30deg",
        "lx": "u",
        "ly": "v",
    },
    "iso_45": {
        "rot": rotation_matrix_y(45) @ rotation_matrix_x(-35.264),
        "title": "Isometric",
        "lx": "u",
        "ly": "v",
    },
    "iso_60": {
        "rot": rotation_matrix_y(60) @ rotation_matrix_x(-20),
        "title": "Oblique 60deg",
        "lx": "u",
        "ly": "v",
    },
    "top_down": {
        "rot": rotation_matrix_x(-90),
        "title": "Top-down (XZ)",
        "lx": "X",
        "ly": "Z",
    },
    "bottom_up": {
        "rot": rotation_matrix_x(90),
        "title": "Bottom-up",
        "lx": "X",
        "ly": "-Z",
    },
}


def generate_views(mesh_obj, coords, out_dir: Path, prefix: str):
    """全視角の個別画像と総合マルチビュー画像を生成."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # --- 個別視角画像 ---
    for view_name, vdef in VIEWS.items():
        fname = f"{prefix}_{view_name}.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_projected_view(
            mesh_obj,
            coords,
            ax,
            vdef["rot"],
            title=f"{prefix} - {vdef['title']}",
            label_x=vdef["lx"],
            label_y=vdef["ly"],
        )
        fig.tight_layout()
        fig.savefig(str(out_dir / fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append((fname, vdef["title"]))
        print(f"  Saved: {out_dir / fname}")

    # --- 断面図 (3箇所) ---
    for z_frac, label in [(0.25, "z25"), (0.5, "z50"), (0.75, "z75")]:
        fname = f"{prefix}_cross_{label}.png"
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_cross_section(
            mesh_obj,
            coords,
            ax,
            z_frac,
            title=f"{prefix} - Cross-section {label}",
        )
        fig.tight_layout()
        fig.savefig(str(out_dir / fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append((fname, f"Cross-section {label}"))
        print(f"  Saved: {out_dir / fname}")

    # --- 総合マルチビュー (3x3) ---
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    view_list = list(VIEWS.items())
    for idx in range(min(9, len(view_list))):
        r, c = idx // 3, idx % 3
        vn, vd = view_list[idx]
        plot_projected_view(
            mesh_obj,
            coords,
            axes[r, c],
            vd["rot"],
            title=vd["title"],
            label_x=vd["lx"],
            label_y=vd["ly"],
        )

    # 残りの空パネルを断面図で埋める
    if len(view_list) < 9:
        for extra_idx, (z_frac, label) in enumerate(
            [(0.25, "z=25%"), (0.5, "z=50%"), (0.75, "z=75%")]
        ):
            idx = len(view_list) + extra_idx
            if idx >= 9:
                break
            r, c = idx // 3, idx % 3
            plot_cross_section(mesh_obj, coords, axes[r, c], z_frac, f"Cross-section {label}")

    multiview_fname = f"{prefix}_multiview_angles.png"
    fig.suptitle(f"{prefix} - Multi-angle Views", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(out_dir / multiview_fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append((multiview_fname, "Multi-angle Views (Overview)"))
    print(f"  Saved: {out_dir / multiview_fname}")

    return generated


def write_gallery_md(out_dir: Path, prefix: str, title: str, generated: list[tuple[str, str]]):
    """ギャラリーMDファイルを出力."""
    md_path = out_dir / "gallery.md"
    lines = [
        f"# {title}",
        "",
        "[<- README](../../../README.md) | [<- gallery](../../gallery.md)",
        "",
        f"> {prefix} の視角別2D投影ギャラリー（時間ステップなし、視野違いのみ）",
        "",
        "## 総合マルチビュー",
        "",
    ]
    # multiview が最後にある
    multiview = [g for g in generated if "multiview" in g[0]]
    if multiview:
        lines.append(f"![{multiview[0][1]}]({multiview[0][0]})")
        lines.append("")

    lines.append("## 個別視角")
    lines.append("")
    for fname, desc in generated:
        if "multiview" in fname:
            continue
        lines.append(f"### {desc}")
        lines.append("")
        lines.append(f"![{desc}]({fname})")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Gallery MD: {md_path}")


# ---------------------------------------------------------------
# メイン: メッシュ構成ごとに生成
# ---------------------------------------------------------------
CONFIGS = [
    {
        "name": "7strand",
        "title": "7本撚線パイプ — 視角別投影",
        "n_strands": 7,
        "wire_diameter": 0.002,
        "pitch": 0.040,
        "n_elems_per_strand": 16,
        "n_pitches": 0.5,
        "out_dir": "docs/verification/7strand",
    },
    {
        "name": "19strand",
        "title": "19本撚線パイプ — 視角別投影",
        "n_strands": 19,
        "wire_diameter": 0.002,
        "pitch": 0.040,
        "n_elems_per_strand": 16,
        "n_pitches": 0.5,
        "out_dir": "docs/verification/19strand/views",
    },
    {
        "name": "37strand",
        "title": "37本撚線パイプ — 視角別投影",
        "n_strands": 37,
        "wire_diameter": 0.002,
        "pitch": 0.040,
        "n_elems_per_strand": 16,
        "n_pitches": 0.5,
        "out_dir": "docs/verification/37strand",
    },
]

for cfg in CONFIGS:
    print("=" * 70)
    print(f"  Generating views: {cfg['name']} ({cfg['n_strands']} strands)")
    print("=" * 70)

    mesh = make_twisted_wire_mesh(
        n_strands=cfg["n_strands"],
        wire_diameter=cfg["wire_diameter"],
        pitch=cfg["pitch"],
        length=0.0,
        n_elems_per_strand=cfg["n_elems_per_strand"],
        n_pitches=cfg["n_pitches"],
    )
    coords = mesh.node_coords.copy()
    out = Path(cfg["out_dir"])

    generated = generate_views(mesh, coords, out, cfg["name"])
    write_gallery_md(out, cfg["name"], cfg["title"], generated)
    print()

# ---------------------------------------------------------------
# ルートギャラリーMDに視角別セクションを追加
# ---------------------------------------------------------------
root_gallery_addition = """
## 視角別マルチビュー（初期メッシュ形状）

### 7本撚線

![7strand multiview](7strand/7strand_multiview_angles.png)

[-> 7strand 詳細ギャラリー](7strand/gallery.md)

### 19本撚線

![19strand multiview](19strand/views/19strand_multiview_angles.png)

[-> 19strand 詳細ギャラリー](19strand/views/gallery.md)

### 37本撚線

![37strand multiview](37strand/37strand_multiview_angles.png)

[-> 37strand 詳細ギャラリー](37strand/gallery.md)
"""

print("=" * 70)
print("  All multi-view galleries generated successfully.")
print("=" * 70)
print(f"\nTotal configurations: {len(CONFIGS)}")
print(f"Views per config: {len(VIEWS)} + 3 cross-sections + 1 multiview = {len(VIEWS) + 4}")

tee.close()
sys.stdout = tee.terminal
