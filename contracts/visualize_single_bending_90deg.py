"""1本素線 θ_y 90度曲げの3Dパイプレンダリング（物理妥当性確認）.

status-280: 1本素線は3-4反復で90度曲げ完走。変形形状を確認する。

Usage:
    python contracts/visualize_single_bending_90deg.py

[← README](../README.md)
"""

import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from xkep_cae.contact._contact_pair import _ContactConfigInput, _ContactManagerInput
from xkep_cae.contact.solver.process import ContactFrictionProcess
from xkep_cae.core import (
    AssembleCallbacks,
    BoundaryData,
    ContactFrictionInputData,
    ContactSetupData,
    MeshData,
)
from xkep_cae.elements._beam_assembler import ULCRBeamAssemblerInput, ULCRBeamAssemblerProcess
from xkep_cae.mesh.process import StrandMeshConfig, StrandMeshProcess
from xkep_cae.numerical_tests.three_point_bend_jig import _circle_section


def _draw_tube(ax, p0, p1, radius, color, n_seg=8, alpha=0.6):
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-15:
        return
    axis_dir = axis / length
    if abs(axis_dir[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis_dir, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis_dir, e1)
    theta = np.linspace(0, 2 * np.pi, n_seg + 1)
    circle = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
    ring0 = p0 + radius * circle
    ring1 = p1 + radius * circle
    for j in range(n_seg):
        verts = [[ring0[j], ring0[j + 1], ring1[j + 1], ring1[j]]]
        poly = Poly3DCollection(verts, alpha=alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor("none")
        ax.add_collection3d(poly)


def main():
    # 1本素線メッシュ
    mesh_result = StrandMeshProcess().process(
        StrandMeshConfig(
            n_strands=1, wire_radius=0.5, pitch_length=100.0, n_elements_per_pitch=16, n_pitches=1.0
        )
    )
    mesh = MeshData(
        node_coords=mesh_result.mesh.node_coords,
        connectivity=mesh_result.mesh.connectivity,
        radii=0.0,
        n_strands=1,
        strand_ids=mesh_result.mesh.strand_ids,
    )
    coords = mesh.node_coords
    conn = mesh.connectivity
    n_nodes = len(coords)
    ndof = n_nodes * 6

    sec = _circle_section(1.0, 0.3)
    G = 130.0e3 / (2.0 * 1.3)
    asm = (
        ULCRBeamAssemblerProcess()
        .process(
            ULCRBeamAssemblerInput(
                node_coords=coords,
                connectivity=conn,
                E=130.0e3,
                G=G,
                A=sec["A"],
                Iy=sec["Iy"],
                Iz=sec["Iz"],
                J=sec["J"],
                kappa_y=sec["kappa"],
                kappa_z=sec["kappa"],
            )
        )
        .assembler
    )
    M = asm.assemble_mass(8.96e-9, lumped=True)

    # 左端固定、右端θ_y処方 + u_x/u_z自由
    fixed = []
    for k in range(6):
        fixed.append(k)  # node 0 全固定
    right = n_nodes - 1
    fixed.append(right * 6 + 1)  # u_y
    fixed.append(right * 6 + 3)  # θ_x
    fixed.append(right * 6 + 5)  # θ_z

    bending_angle = math.pi / 2.0
    prescribed_dofs = np.array([right * 6 + 4], dtype=int)
    prescribed_values = np.array([bending_angle])

    strand_length = 100.0
    f1 = (math.pi / (2.0 * strand_length**2)) * math.sqrt(
        130.0e3 * sec["Iy"] / (8.96e-9 * sec["A"])
    )
    T1 = 1.0 / f1
    t_total = max(10.0 * T1, 1.0)
    dt_initial = t_total / 40

    boundary = BoundaryData(
        fixed_dofs=np.array(sorted(fixed), dtype=int),
        prescribed_dofs=prescribed_dofs,
        prescribed_values=prescribed_values,
        f_ext_total=np.zeros(ndof),
    )
    contact_config = _ContactConfigInput(
        beam_E=130.0e3,
        beam_I=sec["Iy"],
        mu=0.15,
        adaptive_timestepping=True,
        dt_min_fraction=dt_initial / (t_total * 64),
        dt_max_fraction=dt_initial / t_total,
        exclude_same_strand=True,
    )
    contact_setup = ContactSetupData(
        manager=_ContactManagerInput(config=contact_config), k_pen=0.0, mu=0.15
    )

    print("Solving single strand θ_y=π/2 bending...")
    sr = ContactFrictionProcess().process(
        ContactFrictionInputData(
            mesh=mesh,
            boundary=boundary,
            contact=contact_setup,
            callbacks=AssembleCallbacks(
                assemble_tangent=asm.assemble_tangent,
                assemble_internal_force=asm.assemble_internal_force,
                ul_assembler=asm,
            ),
            mass_matrix=M,
            dt_physical=t_total,
            rho_inf=0.9,
            max_nr_attempts=200,
            tol_force=1e-8,
            max_increments=10000,
        )
    )

    frac = sr.load_history[-1] if sr.load_history else 0.0
    print(f"frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}")

    if not sr.displacement_history:
        print("ERROR: no displacement history")
        return

    # 5フレームレンダリング
    n_snaps = len(sr.displacement_history)
    snap_indices = np.linspace(0, n_snaps - 1, 5, dtype=int)

    fig = plt.figure(figsize=(30, 6))
    fig.suptitle(
        f"Single strand θ_y bending (90°) — frac={frac:.4f}\n"
        f"incr={sr.n_increments}, cutback={sr.n_cutbacks}",
        fontsize=14,
    )

    for panel_idx, snap_idx in enumerate(snap_indices):
        u = sr.displacement_history[snap_idx]
        f_val = sr.load_history[snap_idx] if snap_idx < len(sr.load_history) else 0.0

        coords_cur = coords.copy()
        for nd in range(n_nodes):
            coords_cur[nd, 0] += u[6 * nd + 0]
            coords_cur[nd, 1] += u[6 * nd + 1]
            coords_cur[nd, 2] += u[6 * nd + 2]

        ax = fig.add_subplot(1, 5, panel_idx + 1, projection="3d")
        for e in range(len(conn)):
            n0, n1 = int(conn[e, 0]), int(conn[e, 1])
            _draw_tube(ax, coords_cur[n0], coords_cur[n1], 0.5, "#1f77b4", n_seg=10, alpha=0.8)

        angle_deg = f_val * 90.0
        ax.set_title(f"frac={f_val:.3f} ({angle_deg:.1f}°)", fontsize=10)
        ax.view_init(elev=5, azim=-90)  # x-z面から見る
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-10, 80)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-10, 110)

    plt.tight_layout()
    out_path = "docs/verification/single_strand_bending_90deg.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
