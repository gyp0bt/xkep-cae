"""3D梁チューブレンダリングのマルチビューギャラリーを生成する.

従来の2D投影線図を置き換え、円形断面が見える3Dサーフェスレンダリングで
撚線構成ごとのギャラリーを出力する。

Usage:
  python scripts/generate_3d_beam_gallery.py 2>&1 | tee /tmp/log-3d-gallery-$(date +%s).log

[<- README](../README.md)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

log_path = f"/tmp/generate_3d_gallery_{int(time.time())}.log"


class TeeWriter:
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.file = open(filepath, "w")  # noqa: SIM115

    def write(self, msg: str) -> int:
        self.terminal.write(msg)
        self.file.write(msg)
        return len(msg)

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


tee = TeeWriter(log_path)
sys.stdout = tee

print("=== 3D Beam Gallery Generation ===")
print(f"Log: {log_path}")
print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh  # noqa: E402
from xkep_cae.output.render_beam_3d import (  # noqa: E402
    VIEW_PRESETS,
    render_multiview_3d,
    render_twisted_wire_3d,
)

# ---------------------------------------------------------------
# 撚線構成
# ---------------------------------------------------------------
CONFIGS = [
    {
        "name": "7strand",
        "title": "7-strand wire",
        "n_strands": 7,
        "wire_diameter": 0.002,
        "pitch": 0.040,
        "n_elems_per_strand": 16,
        "n_pitches": 0.5,
        "out_dir": "docs/verification/7strand",
    },
    {
        "name": "19strand",
        "title": "19-strand wire",
        "n_strands": 19,
        "wire_diameter": 0.002,
        "pitch": 0.040,
        "n_elems_per_strand": 16,
        "n_pitches": 0.5,
        "out_dir": "docs/verification/19strand/views",
    },
    {
        "name": "37strand",
        "title": "37-strand wire",
        "n_strands": 37,
        "wire_diameter": 0.002,
        "pitch": 0.040,
        "n_elems_per_strand": 16,
        "n_pitches": 0.5,
        "out_dir": "docs/verification/37strand",
    },
]

# 主要視角（ギャラリー用）
GALLERY_VIEWS = [
    "isometric",
    "front_xy",
    "side_xz",
    "end_yz",
    "bird_eye",
    "oblique_30",
    "oblique_60",
    "top_down",
]


def generate_gallery(cfg: dict) -> list[tuple[str, str]]:
    """1撚線構成のギャラリーを生成する."""
    name = cfg["name"]
    title = cfg["title"]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  メッシュ生成: {name} ({cfg['n_strands']}本)")
    mesh = make_twisted_wire_mesh(
        n_strands=cfg["n_strands"],
        wire_diameter=cfg["wire_diameter"],
        pitch=cfg["pitch"],
        length=0.0,
        n_elems_per_strand=cfg["n_elems_per_strand"],
        n_pitches=cfg["n_pitches"],
    )
    print(f"    節点数: {mesh.n_nodes}, 要素数: {mesh.n_elems}")
    print(f"    素線半径: {mesh.wire_radius * 1000:.2f} mm")

    generated: list[tuple[str, str]] = []

    # --- 個別視角画像 ---
    print(f"  個別視角レンダリング ({len(GALLERY_VIEWS)}視角)...")
    results = render_multiview_3d(
        mesh,
        views=GALLERY_VIEWS,
        title_prefix=f"{title}",
        dpi=150,
        n_circ=16,
    )

    for vname, fig, _ax in results:
        fname = f"{name}_3d_{vname}.png"
        fpath = out_dir / fname
        fig.savefig(str(fpath), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        label = VIEW_PRESETS[vname]["label"]
        generated.append((fname, str(label)))
        print(f"    Saved: {fpath}")

    # --- 総合マルチビュー (2x4 パネル) ---
    print("  総合マルチビューパネル生成...")
    n_views = len(GALLERY_VIEWS)
    n_cols = 4
    n_rows = (n_views + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(24, 6 * n_rows), dpi=120)
    for idx, vname in enumerate(GALLERY_VIEWS):
        preset = VIEW_PRESETS[vname]
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")

        # 直接描画（render_beam_3dの内部ロジックを簡略利用）
        _fig_tmp, _ax_tmp = render_twisted_wire_3d(
            mesh,
            elev=float(preset["elev"]),
            azim=float(preset["azim"]),
            title=str(preset["label"]),
            figsize=(8, 7),
            dpi=80,
            n_circ=12,
        )
        # 個別figは不要（マルチビューにはaxだけ欲しいが mplot3dの制約上
        # 別figで生成してclose）
        plt.close(_fig_tmp)

        # マルチビューパネルに再描画
        coords_mm = mesh.node_coords * 1000.0
        r_mm = mesh.wire_radius * 1000.0
        from xkep_cae.output.render_beam_3d import (
            _STRAND_COLORS,
            _make_tube_mesh,
            _set_equal_aspect_3d,
        )

        for sid in range(mesh.n_strands):
            color = _STRAND_COLORS[sid % len(_STRAND_COLORS)]
            ns, ne = mesh.strand_node_ranges[sid]
            # この素線の要素
            for eidx in range(len(mesh.connectivity)):
                n0, n1 = mesh.connectivity[eidx]
                if ns <= n0 < ne and ns <= n1 < ne:
                    p0 = coords_mm[n0]
                    p1 = coords_mm[n1]
                    X, Y, Z = _make_tube_mesh(p0, p1, r_mm, 10)
                    ax.plot_surface(
                        X,
                        Y,
                        Z,
                        color=color,
                        alpha=0.85,
                        shade=True,
                        linewidth=0,
                        antialiased=True,
                    )

        ax.view_init(elev=float(preset["elev"]), azim=float(preset["azim"]))
        ax.set_title(str(preset["label"]), fontsize=10)
        ax.set_xlabel("X [mm]", fontsize=7)
        ax.set_ylabel("Y [mm]", fontsize=7)
        ax.set_zlabel("Z [mm]", fontsize=7)
        ax.tick_params(labelsize=6)
        _set_equal_aspect_3d(ax, coords_mm)

    fig.suptitle(f"{title} — 3D Multi-angle Views", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    multiview_fname = f"{name}_3d_multiview.png"
    fig.savefig(str(out_dir / multiview_fname), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    generated.append((multiview_fname, "3D Multi-angle Views (Overview)"))
    print(f"    Saved: {out_dir / multiview_fname}")

    return generated


def write_gallery_md(
    out_dir: Path, prefix: str, title: str, generated: list[tuple[str, str]]
) -> None:
    """ギャラリーMDファイルを出力する."""
    md_path = out_dir / "gallery.md"
    lines = [
        f"# {title} — 3D Rendering Gallery",
        "",
        "[<- README](../../../README.md) | [<- gallery](../../gallery.md)",
        "",
        f"> {prefix}: 3D tube rendering with circular cross-sections",
        "",
        "## 総合マルチビュー",
        "",
    ]
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
# メイン実行
# ---------------------------------------------------------------
for cfg in CONFIGS:
    print("=" * 70)
    print(f"  Generating 3D gallery: {cfg['name']} ({cfg['n_strands']} strands)")
    print("=" * 70)

    t0 = time.time()
    generated = generate_gallery(cfg)
    out_dir = Path(cfg["out_dir"])
    write_gallery_md(out_dir, cfg["name"], cfg["title"], generated)
    elapsed = time.time() - t0
    print(f"  完了: {elapsed:.1f}秒")
    print()

print("=" * 70)
print("  All 3D galleries generated successfully.")
print("=" * 70)
print(f"\nTotal configurations: {len(CONFIGS)}")
print(f"Views per config: {len(GALLERY_VIEWS)} + 1 multiview = {len(GALLERY_VIEWS) + 1}")

tee.close()
sys.stdout = tee.terminal
