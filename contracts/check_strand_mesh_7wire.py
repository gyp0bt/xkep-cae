"""7本撚線メッシュ生成 + 貫入チェック + 可視化.

E=130MPa, ρ=8.96e-9 t/mm³, 線径10mm（半径5mm）, ピッチ100mm, 3ピッチ分。
梁表面の貫入なき初期メッシュを作成し、3パネル可視化画像を出力する。

[← README](../README.md)
"""

from __future__ import annotations

import sys

import numpy as np

from xkep_cae.contact.setup.process import ContactSetupConfig, ContactSetupProcess
from xkep_cae.contact.solver._initial_penetration import (
    InitialPenetrationInput,
    InitialPenetrationProcess,
)
from xkep_cae.mesh.process import StrandMeshConfig, StrandMeshProcess

# ====================================================================
# パラメータ
# ====================================================================

WIRE_RADIUS = 5.0  # mm（線径10mm）
PITCH_LENGTH = 100.0  # mm（10倍径）
N_STRANDS = 7
N_ELEMENTS_PER_PITCH = 32
N_PITCHES = 3.0
GAP = 0.5  # 明示的ギャップ（自動安全ギャップより大きめ）

# ====================================================================
# メッシュ生成
# ====================================================================

print("=" * 60)
print("  7本撚線メッシュ生成 + 貫入チェック")
print("=" * 60)
print(f"  wire_radius     = {WIRE_RADIUS} mm")
print(f"  pitch_length    = {PITCH_LENGTH} mm")
print(f"  n_strands       = {N_STRANDS}")
print(f"  n_elems/pitch   = {N_ELEMENTS_PER_PITCH}")
print(f"  n_pitches       = {N_PITCHES}")
print(f"  total length    = {PITCH_LENGTH * N_PITCHES} mm")
print()

mesh_config = StrandMeshConfig(
    n_strands=N_STRANDS,
    wire_radius=WIRE_RADIUS,
    pitch_length=PITCH_LENGTH,
    n_elements_per_pitch=N_ELEMENTS_PER_PITCH,
    n_pitches=N_PITCHES,
    gap=GAP,
)

mesh_result = StrandMeshProcess().process(mesh_config)
mesh = mesh_result.mesh

n_nodes = mesh.node_coords.shape[0]
n_elems = mesh.connectivity.shape[0]
n_nodes_per_strand = n_nodes // N_STRANDS
n_elems_per_strand = n_elems // N_STRANDS

print(f"  生成完了:")
print(f"    節点数         = {n_nodes}")
print(f"    要素数         = {n_elems}")
print(f"    節点/素線      = {n_nodes_per_strand}")
print(f"    要素/素線      = {n_elems_per_strand}")
print()

# ====================================================================
# 接触セットアップ
# ====================================================================

print("  接触セットアップ...")
contact_config = ContactSetupConfig(
    mesh=mesh,
    k_pen=1e6,
    mu=0.0,
)
contact_result = ContactSetupProcess().process(contact_config)
n_pairs = len(contact_result.manager.pairs)
print(f"    接触ペア数     = {n_pairs}")
print()

# ====================================================================
# 初期貫入チェック
# ====================================================================

print("  初期貫入チェック...")
pen_proc = InitialPenetrationProcess()
pen_out = pen_proc.process(
    InitialPenetrationInput(
        pairs=contact_result.manager.pairs,
        node_coords=mesh.node_coords,
    )
)
n_pen = pen_out.n_penetrations
print(f"    初期貫入ペア数 = {n_pen}")

if n_pen > 0:
    print(f"  ⚠ 初期貫入あり: {n_pen} ペア — 座標調整を試行...")
    pen_adj = pen_proc.process(
        InitialPenetrationInput(
            pairs=contact_result.manager.pairs,
            node_coords=mesh.node_coords,
            adjust=True,
            max_iterations=50,
        )
    )
    if pen_adj.adjusted_coords is not None:
        adjusted_coords = pen_adj.adjusted_coords
        print(f"    座標調整: {pen_adj.n_pen_fixed} ペア修正, {pen_adj.n_gap_closed} ギャップ閉鎖")
        # 再チェック
        pen_recheck = pen_proc.process(
            InitialPenetrationInput(
                pairs=contact_result.manager.pairs,
                node_coords=adjusted_coords,
            )
        )
        n_pen_after = pen_recheck.n_penetrations
        print(f"    調整後貫入ペア数 = {n_pen_after}")
        if n_pen_after == 0:
            print(f"  ✓ 座標調整で貫入解消 — メッシュ品質OK")
            # 調整後座標を使用
            coords = adjusted_coords
        else:
            print(f"  ❌ 座標調整後も貫入残: {n_pen_after} ペア")
            print("  gap を増やしてリトライしてください。")
            coords = adjusted_coords  # 可視化は調整後で行う
    else:
        print(f"  ❌ 座標調整不要（n_pen=0）だが初期チェックでは{n_pen}ペア検出")
        coords = mesh.node_coords
else:
    print(f"  ✓ 初期貫入なし — メッシュ品質OK")
    coords = mesh.node_coords
print()

# ====================================================================
# 可視化
# ====================================================================

print("  可視化...")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 素線ごとの色
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
conn = mesh.connectivity
coords = mesh.node_coords

# 素線ごとの要素範囲を計算
strand_elem_ranges = []
for s in range(N_STRANDS):
    start = s * n_elems_per_strand
    end = (s + 1) * n_elems_per_strand
    strand_elem_ranges.append((start, end))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: XY投影（側面図）
ax = axes[0]
for s in range(N_STRANDS):
    start, end = strand_elem_ranges[s]
    color = colors[s % len(colors)]
    for e in range(start, end):
        n0, n1 = conn[e]
        ax.plot(
            [coords[n0, 0], coords[n1, 0]],
            [coords[n0, 1], coords[n1, 1]],
            color=color,
            linewidth=0.8,
        )
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_title("XY Projection (Side View)")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# Panel 2: XZ投影（上面図）
ax = axes[1]
for s in range(N_STRANDS):
    start, end = strand_elem_ranges[s]
    color = colors[s % len(colors)]
    for e in range(start, end):
        n0, n1 = conn[e]
        ax.plot(
            [coords[n0, 0], coords[n1, 0]],
            [coords[n0, 2], coords[n1, 2]],
            color=color,
            linewidth=0.8,
        )
ax.set_xlabel("X [mm]")
ax.set_ylabel("Z [mm]")
ax.set_title("XZ Projection (Top View)")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# Panel 3: YZ投影（端面図）+ 梁半径を円で表現
ax = axes[2]
# 端面の節点（x=0 付近、各素線の最初の節点）
for s in range(N_STRANDS):
    start_node = s * n_nodes_per_strand
    y0 = coords[start_node, 1]
    z0 = coords[start_node, 2]
    color = colors[s % len(colors)]

    # 梁半径の円
    theta = np.linspace(0, 2 * np.pi, 64)
    cy = y0 + WIRE_RADIUS * np.cos(theta)
    cz = z0 + WIRE_RADIUS * np.sin(theta)
    ax.fill(cy, cz, color=color, alpha=0.3)
    ax.plot(cy, cz, color=color, linewidth=1.0)
    ax.plot(y0, z0, "o", color=color, markersize=3)
    ax.annotate(f"S{s}", (y0, z0), fontsize=8, ha="center", va="center")

ax.set_xlabel("Y [mm]")
ax.set_ylabel("Z [mm]")
ax.set_title("YZ End View (x=0, beam radius shown)")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"7-strand mesh: R={WIRE_RADIUS}mm, pitch={PITCH_LENGTH}mm, "
    f"{N_PITCHES} pitches, {n_elems} elems, penetration={n_pen}",
    fontsize=12,
)
fig.tight_layout()

out_path = "docs/verification/strand_mesh_7wire.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)

print(f"  ✓ 保存: {out_path}")
print()
print("  完了。ユーザーに目視確認を依頼してください。")
