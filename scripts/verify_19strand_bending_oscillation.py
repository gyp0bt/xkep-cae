"""19-strand bending oscillation convergence verification.

Verification flow:
  1. Phase1 (bending) + Phase2 (oscillation) convergence check
  2. Convergence log with cutback, contact state, energy info (tee to file)
  3. Deformed mesh 2D projection snapshots for physical plausibility

Usage:
  python scripts/verify_19strand_bending_oscillation.py 2>&1 | tee /tmp/verify_19strand.log

[<- README](../README.md)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# --- Log tee setup ---
log_path = f"/tmp/verify_19strand_{int(time.time())}.log"


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

print("=== 19-strand bending oscillation verification ===")
print(f"Log: {log_path}")
print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xkep_cae.mesh.twisted_wire import make_twisted_wire_mesh  # noqa: E402
from xkep_cae.numerical_tests.wire_bending_benchmark import run_bending_oscillation  # noqa: E402
from xkep_cae.output.render_beam_3d import (  # noqa: E402
    _STRAND_COLORS,
    _make_tube_mesh,
    _set_equal_aspect_3d,
    render_twisted_wire_3d,
)


def deformed_coords(mesh_obj, u_snap):
    """Compute deformed coordinates from displacement snapshot."""
    coords = mesh_obj.node_coords.copy()
    for i in range(mesh_obj.n_nodes):
        coords[i, 0] += u_snap[6 * i]
        coords[i, 1] += u_snap[6 * i + 1]
        coords[i, 2] += u_snap[6 * i + 2]
    return coords


def _render_3d_on_ax(mesh_obj, coords, ax, title="", elev=25.0, azim=-60.0):
    """3Dチューブを指定axに描画する（マルチパネル用）."""
    coords_mm = coords * 1000.0
    r_mm = mesh_obj.wire_radius * 1000.0
    for sid in range(mesh_obj.n_strands):
        color = _STRAND_COLORS[sid % len(_STRAND_COLORS)]
        ns, ne = mesh_obj.strand_node_ranges[sid]
        for eidx in range(len(mesh_obj.connectivity)):
            n0, n1 = mesh_obj.connectivity[eidx]
            if ns <= n0 < ne and ns <= n1 < ne:
                X, Y, Z = _make_tube_mesh(coords_mm[n0], coords_mm[n1], r_mm, 8)
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
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("X [mm]", fontsize=6)
    ax.set_ylabel("Y [mm]", fontsize=6)
    ax.set_zlabel("Z [mm]", fontsize=6)
    ax.tick_params(labelsize=5)
    _set_equal_aspect_3d(ax, coords_mm)


def save_multi_view(mesh_obj, u_snap, out_path, suptitle=""):
    """Save 2x3 panel: 3D views from 6 angles."""
    coords = deformed_coords(mesh_obj, u_snap)
    view_list = [
        ("isometric", 25.0, -60.0),
        ("front_xy", 0.0, -90.0),
        ("side_xz", 0.0, 0.0),
        ("end_yz", 0.0, -180.0),
        ("bird_eye", 45.0, -45.0),
        ("top_down", 90.0, -90.0),
    ]

    fig = plt.figure(figsize=(18, 12))
    for idx, (label, elev, azim) in enumerate(view_list):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        _render_3d_on_ax(mesh_obj, coords, ax, label, elev, azim)

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_increment_gallery(mesh_obj, snapshots, labels, out_dir, prefix, plane="xy"):
    """Save individual 3D increment images + gallery overview.

    plane引数は後方互換性のため残すが、3Dレンダリングでは無視される。
    """
    if not snapshots:
        return

    n = len(snapshots)
    # Individual 3D images
    for i, (u_snap, label) in enumerate(zip(snapshots, labels, strict=True)):
        coords = deformed_coords(mesh_obj, u_snap)
        fig, _ax = render_twisted_wire_3d(
            mesh_obj,
            node_coords=coords,
            elev=25.0,
            azim=-60.0,
            title=f"{prefix} - {label}",
            figsize=(10, 8),
            dpi=100,
            n_circ=10,
        )
        fig.savefig(
            str(out_dir / f"{prefix}_incr{i:03d}_3d.png"),
            dpi=120,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    # Gallery (up to 12 panels)
    n_show = min(n, 12)
    indices = np.linspace(0, n - 1, n_show, dtype=int)
    ncols = min(n_show, 4)
    nrows = (n_show + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 5.5 * nrows))
    for plot_idx, snap_idx in enumerate(indices):
        coords = deformed_coords(mesh_obj, snapshots[snap_idx])
        lbl = labels[snap_idx] if snap_idx < len(labels) else f"Step {snap_idx}"
        ax = fig.add_subplot(nrows, ncols, plot_idx + 1, projection="3d")
        _render_3d_on_ax(mesh_obj, coords, ax, lbl)

    fig.suptitle(f"{prefix} gallery (3D)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        str(out_dir / f"{prefix}_gallery_3d.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"  Saved gallery: {out_dir / f'{prefix}_gallery_3d.png'}")


# ==================================================================
# 1. 45-degree bending (Phase1 only)
# ==================================================================
print("=" * 70)
print("  Test 1: 19-strand 45deg bending (UL+NCP, adaptive)")
print("=" * 70)

mesh = make_twisted_wire_mesh(19, 0.002, 0.040, length=0.0, n_elems_per_strand=16, n_pitches=0.5)
out_base = Path("docs/verification/19strand")
out_base.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
result_45 = run_bending_oscillation(
    n_strands=19,
    n_elems_per_strand=16,
    n_pitches=0.5,
    bend_angle_deg=45,
    use_ncp=True,
    use_mortar=True,
    n_gauss=2,
    max_iter=50,
    tol_force=1e-4,
    adaptive_timestepping=True,
    use_updated_lagrangian=True,
    show_progress=True,
    n_cycles=0,
    exclude_same_layer=True,
    midpoint_prescreening=True,
    use_line_search=False,
    g_on=0.0005,
    g_off=0.001,
)
t_45 = time.perf_counter() - t0
print(f"\nResult: converged={result_45.phase1_converged}, time={t_45:.2f}s")
print(f"  NR iterations: {result_45.phase1_result.total_newton_iterations}")
print(f"  Max penetration ratio: {result_45.max_penetration_ratio:.6f}")
print(f"  Active contact pairs: {result_45.n_active_contacts}")

# Save images for 45-degree bending
out_45 = out_base / "bend45"
out_45.mkdir(parents=True, exist_ok=True)

if result_45.displacement_snapshots:
    # Multi-view of final state
    save_multi_view(
        mesh,
        result_45.displacement_snapshots[-1],
        out_45 / "final_multiview.png",
        "19-strand 45deg bending - final state",
    )
    # Increment gallery for all 3 planes
    save_increment_gallery(
        mesh, result_45.displacement_snapshots, result_45.snapshot_labels, out_45, "bend45"
    )

# ==================================================================
# 2. 90-degree bending + oscillation
# ==================================================================
print()
print("=" * 70)
print("  Test 2: 19-strand 90deg bending + 1-cycle oscillation (UL+NCP)")
print("=" * 70)

t0 = time.perf_counter()
result_90 = run_bending_oscillation(
    n_strands=19,
    n_elems_per_strand=16,
    n_pitches=0.5,
    bend_angle_deg=90.0,
    oscillation_amplitude_mm=2.0,
    n_cycles=1,
    n_steps_per_quarter=3,
    max_iter=50,
    tol_force=1e-4,
    show_progress=True,
    use_ncp=True,
    use_mortar=True,
    adaptive_timestepping=True,
    exclude_same_layer=True,
    midpoint_prescreening=True,
    use_line_search=False,
    g_on=0.0005,
    g_off=0.001,
    use_updated_lagrangian=True,
)
t_90 = time.perf_counter() - t0
print("\nResult:")
print(f"  Phase1(bending): converged={result_90.phase1_converged}")
print(f"  Phase2(oscillation): converged={result_90.phase2_converged}")
print(f"  Total time: {t_90:.2f}s")
print(f"  Active contact pairs: {result_90.n_active_contacts}")
print(f"  Max penetration ratio: {result_90.max_penetration_ratio:.6f}")
print(f"  Phase2 steps: {len(result_90.phase2_results)}")

# Save images for 90-degree bending + oscillation
out_90 = out_base / "bend90_osc"
out_90.mkdir(parents=True, exist_ok=True)

if result_90.displacement_snapshots:
    save_multi_view(
        mesh,
        result_90.displacement_snapshots[-1],
        out_90 / "final_multiview.png",
        "19-strand 90deg + oscillation - final state",
    )
    save_increment_gallery(
        mesh, result_90.displacement_snapshots, result_90.snapshot_labels, out_90, "bend90_osc"
    )

    # Cross-section evolution — end view (YZ) showing circular cross-sections
    n_show = min(len(result_90.displacement_snapshots), 8)
    indices = np.linspace(0, len(result_90.displacement_snapshots) - 1, n_show, dtype=int)
    ncols = min(n_show, 4)
    nrows = (n_show + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 5.5 * nrows))
    for plot_idx, snap_idx in enumerate(indices):
        coords = deformed_coords(mesh, result_90.displacement_snapshots[snap_idx])
        lbl = (
            result_90.snapshot_labels[snap_idx] if snap_idx < len(result_90.snapshot_labels) else ""
        )
        ax = fig.add_subplot(nrows, ncols, plot_idx + 1, projection="3d")
        _render_3d_on_ax(mesh, coords, ax, f"End view - {lbl}", elev=0.0, azim=-180.0)

    fig.suptitle("19-strand cross-section evolution (End view)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    cs_path = out_90 / "cross_section_evolution_3d.png"
    fig.savefig(str(cs_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {cs_path}")

# ==================================================================
# 3. Summary
# ==================================================================
print()
print("=" * 70)
print("  Verification Summary")
print("=" * 70)
print(f"  45deg bending:   {'PASS' if result_45.phase1_converged else 'FAIL'} ({t_45:.1f}s)")
print(f"  90deg bending:   {'PASS' if result_90.phase1_converged else 'FAIL'}")
print(f"  90deg oscillation: {'PASS' if result_90.phase2_converged else 'FAIL'} ({t_90:.1f}s)")
print(f"  Log: {log_path}")

all_pass = result_45.phase1_converged and result_90.phase1_converged and result_90.phase2_converged
print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAIL'}")

sys.stdout = tee.terminal
tee.close()
print(f"\nVerification complete. Log: {log_path}")
print(f"Images: {out_base.resolve()}")

if not all_pass:
    sys.exit(1)
