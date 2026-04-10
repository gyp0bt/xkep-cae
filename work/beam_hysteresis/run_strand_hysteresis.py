"""
7本撚線 曲げ+揺動ヒステリシス観測
==================================

StrandBendingOscillationProcess を利用し、曲げ後に揺動サイクルを実行。
接触あり/なしの比較で、ストランド間摩擦によるヒステリシスを観測。

比較:
  1. contact OFF (弾性、ヒステリシスなし)
  2. contact ON, mu=0.15 (摩擦あり、ヒステリシスあり期待)

[<- README](../../README.md)
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/home/user/xkep-cae")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xkep_cae.mesh.process import StrandMeshConfig, StrandMeshProcess
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
    _collect_end_nodes,
)


def run_case(label, contact_enabled, mu, bending_curvature=0.005, n_osc=0, osc_amp=0.0):
    """1ケース実行してθ履歴と接触診断を返す."""
    cfg = StrandBendingOscillationConfig(
        n_strands=7,
        wire_radius=0.5,
        pitch_length=100.0,  # テスト実績と同じ
        gap=0.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=bending_curvature,
        n_cycles=1,
        n_increments_per_cycle=40,
        rho_inf=0.9,
        mu=mu,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        contact_enabled=contact_enabled,
        free_end_mode=True,  # status-280: MPC不使用（実績あり）
        loading_mode="rotation",
        penalty_exponent=1.5,  # Hertz型
        exclude_end_elements=0,
        n_oscillation_cycles=n_osc,
        oscillation_amplitude=osc_amp,
    )

    strand_length = cfg.pitch_length * cfg.n_pitches
    bending_angle = bending_curvature * strand_length
    print(f"\n=== {label} ===")
    print(f"  contact={'ON' if contact_enabled else 'OFF'}, mu={mu}, "
          f"theta_max={math.degrees(bending_angle):.1f}deg, osc_amp={osc_amp}mm")

    result = StrandBendingOscillationProcess().process(cfg)
    sr = result.solver_result

    print(f"  converged={sr.converged}, incr={sr.n_increments}, "
          f"cutback={sr.n_cutbacks}, time={sr.elapsed_seconds:.1f}s")

    # θ_y 履歴を抽出
    # free_end_mode: 各素線右端ノードのθ_yを平均
    # MPC mode: 右端参照点のθ_y
    if cfg.free_end_mode:
        mesh_result2 = StrandMeshProcess().process(
            StrandMeshConfig(
                n_strands=cfg.n_strands, wire_radius=cfg.wire_radius,
                pitch_length=cfg.pitch_length, gap=cfg.gap,
                n_elements_per_pitch=cfg.n_elements_per_pitch,
                n_pitches=cfg.n_pitches,
            )
        )
        _, right_nodes = _collect_end_nodes(
            mesh_result2.mesh.connectivity, cfg.n_strands, mesh_result2.mesh.strand_ids)
        ref_right_dofs = [n * 6 + 4 for n in right_nodes]  # θ_y of each right end node
    else:
        ref_right_dof = (result.n_strand_nodes + 1) * 6 + 4
    thetas = []
    for u in sr.displacement_history:
        if cfg.free_end_mode:
            thetas.append(np.mean([u[d] for d in ref_right_dofs]))
        else:
            thetas.append(u[ref_right_dof] if len(u) > ref_right_dof else 0.0)
    thetas = np.array(thetas)

    # 接触診断
    fracs = np.array(sr.load_history)
    cf_norms = np.array(sr.contact_force_history) if sr.contact_force_history else np.zeros(len(fracs))

    # increment_diagnostics から活性ペア数・滑りペア数
    n_active = []
    n_sliding = []
    n_sticking = []
    strain_energies = []
    for diag in sr.increment_diagnostics:
        n_active.append(diag.n_active_pairs)
        n_sliding.append(diag.n_sliding_pairs)
        n_sticking.append(diag.n_sticking_pairs)
        strain_energies.append(diag.strain_energy)

    return {
        "label": label,
        "fracs": fracs,
        "thetas": thetas,
        "cf_norms": cf_norms,
        "n_active": np.array(n_active),
        "n_sliding": np.array(n_sliding),
        "n_sticking": np.array(n_sticking),
        "strain_energies": np.array(strain_energies),
        "result": sr,
        "bending_angle": bending_angle,
    }


def main():
    out_dir = "/home/user/xkep-cae/work/beam_hysteresis"

    # テスト実績と同じ構成: pitch=100mm, kappa=0.001 (5.7度)
    # n_osc=2: 曲げ後に2サイクル揺動（θ=0まで完全除荷含む）
    kappa = 0.001

    cases = [
        run_case("Elastic (no contact)", contact_enabled=False, mu=0.0,
                 bending_curvature=kappa, n_osc=2),
        run_case("Contact mu=0.15", contact_enabled=True, mu=0.15,
                 bending_curvature=kappa, n_osc=2),
    ]

    # ── プロット ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) θ vs frac
    ax = axes[0, 0]
    for c in cases:
        ax.plot(c["fracs"], np.degrees(c["thetas"]), '-', lw=1.5, label=c["label"])
    ax.set_xlabel("Load fraction")
    ax.set_ylabel("theta_y [deg]")
    ax.set_title("(a) Rotation history (bend + oscillation)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Contact force norm vs θ
    ax = axes[0, 1]
    for c in cases:
        if len(c["cf_norms"]) > 0 and len(c["cf_norms"]) == len(c["thetas"]):
            ax.plot(np.degrees(c["thetas"]), c["cf_norms"], '-', lw=1.5, label=c["label"])
    ax.set_xlabel("theta_y [deg]")
    ax.set_ylabel("Contact force norm [N]")
    ax.set_title("(b) Contact force vs rotation\n(hysteresis = different loading/unloading paths)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Strain energy vs θ (M ≈ ΔE/Δθ)
    ax = axes[1, 0]
    for c in cases:
        se = c["strain_energies"]
        thetas = c["thetas"]
        n = min(len(se), len(thetas))
        if n > 0:
            ax.plot(np.degrees(thetas[:n]), se[:n], '-', lw=1.5, label=c["label"])
    ax.set_xlabel("theta_y [deg]")
    ax.set_ylabel("Strain energy [N*mm]")
    ax.set_title("(c) Strain energy vs rotation\n(loop area = dissipated energy)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Active/sliding/sticking pairs
    ax = axes[1, 1]
    for c in cases:
        n = min(len(c["n_sliding"]), len(c["thetas"]))
        if n > 0 and c["n_sliding"].max() > 0:
            ax.plot(np.degrees(c["thetas"][:n]), c["n_sliding"][:n],
                    '-', lw=1.5, label=f'{c["label"]} (sliding)')
            ax.plot(np.degrees(c["thetas"][:n]), c["n_sticking"][:n],
                    '--', lw=1.5, label=f'{c["label"]} (sticking)')
    ax.set_xlabel("theta_y [deg]")
    ax.set_ylabel("Number of contact pairs")
    ax.set_title("(d) Contact pair states vs rotation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"7-strand bending hysteresis (free_end_mode, Hertz penalty)\n"
        f"kappa={kappa}, pitch=100mm, E=130GPa",
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = f"{out_dir}/08_strand_hysteresis.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
