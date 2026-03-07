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


def deformed_coords(mesh_obj, u_snap):
    """Compute deformed coordinates from displacement snapshot."""
    coords = mesh_obj.node_coords.copy()
    for i in range(mesh_obj.n_nodes):
        coords[i, 0] += u_snap[6 * i]
        coords[i, 1] += u_snap[6 * i + 1]
        coords[i, 2] += u_snap[6 * i + 2]
    return coords


def plot_side_view(mesh_obj, coords, ax, plane="xy", title=""):
    """Plot side view of deformed mesh (beam centerlines)."""
    idx_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ix, iy = idx_map[plane]
    labels = {0: "x", 1: "y", 2: "z"}

    for sid in range(mesh_obj.n_strands):
        nodes = mesh_obj.strand_nodes(sid)
        c = f"C{sid % 10}"
        ax.plot(
            coords[nodes, ix] * 1000,
            coords[nodes, iy] * 1000,
            "-", color=c, linewidth=1.0, alpha=0.7,
        )
    ax.set_aspect("equal")
    ax.set_xlabel(f"{labels[ix]} [mm]", fontsize=7)
    ax.set_ylabel(f"{labels[iy]} [mm]", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)


def plot_cross_section(mesh_obj, coords, ax, z_frac=0.5, title=""):
    """Plot cross-section at given z fraction (interpolated node positions)."""
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    z_target = z_min + (z_max - z_min) * z_frac

    for sid in range(mesh_obj.n_strands):
        nodes = mesh_obj.strand_nodes(sid)
        strand_coords = coords[nodes]
        z_vals = strand_coords[:, 2]

        # Find closest segment to z_target
        for j in range(len(nodes) - 1):
            z0, z1 = z_vals[j], z_vals[j + 1]
            if (z0 <= z_target <= z1) or (z1 <= z_target <= z0):
                if abs(z1 - z0) < 1e-15:
                    t = 0.5
                else:
                    t = (z_target - z0) / (z1 - z0)
                x_interp = strand_coords[j, 0] * (1 - t) + strand_coords[j + 1, 0] * t
                y_interp = strand_coords[j, 1] * (1 - t) + strand_coords[j + 1, 1] * t
                c = f"C{sid % 10}"
                ax.plot(x_interp * 1000, y_interp * 1000, "o", color=c, markersize=4)
                break

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]", fontsize=7)
    ax.set_ylabel("y [mm]", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)


def save_multi_view(mesh_obj, u_snap, out_path, suptitle=""):
    """Save 6-panel figure: 3 side views + 3 cross-sections."""
    coords = deformed_coords(mesh_obj, u_snap)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Side views
    plot_side_view(mesh_obj, coords, axes[0, 0], "xy", "Side view (XY)")
    plot_side_view(mesh_obj, coords, axes[0, 1], "xz", "Side view (XZ)")
    plot_side_view(mesh_obj, coords, axes[0, 2], "yz", "Side view (YZ)")

    # Cross-sections at 25%, 50%, 75%
    for j, (frac, label) in enumerate([(0.25, "z=25%"), (0.5, "z=50%"), (0.75, "z=75%")]):
        plot_cross_section(mesh_obj, coords, axes[1, j], frac, f"Cross-section {label}")

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_increment_gallery(mesh_obj, snapshots, labels, out_dir, prefix, plane="xy"):
    """Save individual increment images + gallery overview."""
    if not snapshots:
        return

    n = len(snapshots)
    # Individual images
    for i, (u_snap, label) in enumerate(zip(snapshots, labels, strict=True)):
        coords = deformed_coords(mesh_obj, u_snap)
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_side_view(mesh_obj, coords, ax, plane, f"{prefix} - {label}")
        fig.tight_layout()
        fig.savefig(str(out_dir / f"{prefix}_incr{i:03d}_{plane}.png"), dpi=120)
        plt.close(fig)

    # Gallery (up to 12 panels)
    n_show = min(n, 12)
    indices = np.linspace(0, n - 1, n_show, dtype=int)
    ncols = min(n_show, 4)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for plot_idx, snap_idx in enumerate(indices):
        r, c = plot_idx // ncols, plot_idx % ncols
        coords = deformed_coords(mesh_obj, snapshots[snap_idx])
        lbl = labels[snap_idx] if snap_idx < len(labels) else f"Step {snap_idx}"
        plot_side_view(mesh_obj, coords, axes[r, c], plane, lbl)

    for plot_idx in range(len(indices), nrows * ncols):
        r, c = plot_idx // ncols, plot_idx % ncols
        axes[r, c].set_visible(False)

    fig.suptitle(f"{prefix} gallery ({plane.upper()} plane)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_dir / f"{prefix}_gallery_{plane}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved gallery: {out_dir / f'{prefix}_gallery_{plane}.png'}")


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
    for plane in ["xy", "xz", "yz"]:
        save_increment_gallery(mesh, result_45.displacement_snapshots,
                               result_45.snapshot_labels, out_45, "bend45", plane)

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
    for plane in ["xy", "xz", "yz"]:
        save_increment_gallery(mesh, result_90.displacement_snapshots,
                               result_90.snapshot_labels, out_90, "bend90_osc", plane)

    # Cross-section evolution (at z=50%) across increments
    n_show = min(len(result_90.displacement_snapshots), 8)
    indices = np.linspace(0, len(result_90.displacement_snapshots) - 1, n_show, dtype=int)
    ncols = min(n_show, 4)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for plot_idx, snap_idx in enumerate(indices):
        r, c = plot_idx // ncols, plot_idx % ncols
        coords = deformed_coords(mesh, result_90.displacement_snapshots[snap_idx])
        lbl = result_90.snapshot_labels[snap_idx] if snap_idx < len(result_90.snapshot_labels) else ""
        plot_cross_section(mesh, coords, axes[r, c], 0.5, f"z=50% - {lbl}")

    for plot_idx in range(len(indices), nrows * ncols):
        r, c = plot_idx // ncols, plot_idx % ncols
        axes[r, c].set_visible(False)

    fig.suptitle("19-strand cross-section evolution (z=50%)", fontsize=12)
    fig.tight_layout()
    cs_path = out_90 / "cross_section_evolution.png"
    fig.savefig(str(cs_path), dpi=150, bbox_inches="tight")
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
