"""work/beam_hysteresis/35_explicit_accuracy_validation.py — explicit 解の精度検証.

[← README](README.md) | [← project README](../../README.md)

status-381 §7 の引継ぎ TODO「explicit 解を implicit / 解析解と一致させる」の検証
スクリプト。status-381 で発覚した「解析解 73.3mm に対し explicit が 50% アンダー」
問題を、status-382 で導入した:

- (p3) 質量比例 Rayleigh damping `explicit_mass_proportional_damping_alpha`
- (p1) BC 完了後の動的緩和ステップ `explicit_relax_steps`

の組合せで掃引し、効果を実測する。

**status-382 結論**: (p3) + (p1) の組合せでは精度 gate (5) を満たさない。
UL 定式化の `update_reference` が各増分の dynamic lag を「reference」に
凍結するため、後続の relax phase では `f_int(u_incr) ≈ 0` となり、
構造を quasi-static 平衡へ駆動できない。詳細は status-382 §3 を参照。

Gate (MCDD 凍結解除条件 (5)、status-381 §5):
- `|max|u_explicit| − max|u_analytical|| / max|u_analytical| < 0.10`
- 単梁 90° カンチレバー曲げ: 解析解 max|u| = 73.30 mm
  （quarter circle、L=100mm、R = 2L/π ≈ 63.66 mm、|u| = √((L−R)² + R²)）

実行:
    uv run --extra dev python work/beam_hysteresis/35_explicit_accuracy_validation.py \\
        2>&1 | tee /tmp/accuracy_$(date +%s).log
"""

from __future__ import annotations

import time

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _make_base_cfg() -> dict:
    """単梁 90° 曲げ baseline（接触なし、L=100 mm、E=130 GPa、撚線径=1mm）."""
    return dict(
        n_strands=1,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.015,
        n_cycles=1,
        n_increments_per_cycle=20,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=20000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=False,
    )


def _run_one(label: str, **overrides) -> dict:
    base = _make_base_cfg()
    base.update(overrides)
    cfg = StrandBendingOscillationConfig(**base)

    print()
    print("─" * 72)
    print(f"[run] {label}")
    print("─" * 72)

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = float(sr.load_history[-1]) if sr.load_history else 0.0
    u = sr.u
    n_total_nodes = u.shape[0] // 6
    u_trans = u.reshape(n_total_nodes, 6)[:, :3]
    max_u_trans = float(np.max(np.linalg.norm(u_trans, axis=1)))

    print(
        f"  frac={frac:.4f}, conv={sr.converged}, "
        f"incr={sr.n_increments}, cb={sr.n_cutbacks}, t={elapsed:.2f}s, "
        f"max|u|={max_u_trans:.3e} mm"
    )
    return dict(
        label=label,
        frac=frac,
        converged=bool(sr.converged),
        n_increments=int(sr.n_increments),
        n_cutbacks=int(sr.n_cutbacks),
        elapsed=elapsed,
        max_u_trans=max_u_trans,
    )


def main() -> int:
    print("=" * 72)
    print("status-382 候補 (p3) damping + (p1) relax の効果実測")
    print("単梁 90° カンチレバー曲げ（L=100mm）")
    print("解析解 max|u| = 73.30 mm（quarter circle）")
    print("=" * 72)

    L_strand = 100.0
    R_analytical = 2.0 * L_strand / np.pi  # ≈ 63.66 mm
    u_analytical = float(np.sqrt((L_strand - R_analytical) ** 2 + R_analytical**2))
    print(f"\n解析解: R = 2L/π = {R_analytical:.3f} mm, max|u| = {u_analytical:.3f} mm")

    runs: list[dict] = []
    runs.append(_run_one("implicit_baseline", solver_mode="implicit"))

    # status-381 baseline — damping/relax なし
    runs.append(_run_one(
        "exp_baseline_no_damp_no_relax",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_mass_scaling_max_growth_per_update=4.0,
    ))

    # 候補 (p3) damping のみ
    runs.append(_run_one(
        "exp_alpha0.5_no_relax",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_mass_scaling_max_growth_per_update=4.0,
        explicit_mass_proportional_damping_alpha=0.5,
    ))

    # 候補 (p1) relax のみ（damping なし）
    runs.append(_run_one(
        "exp_no_damp_relax500",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_mass_scaling_max_growth_per_update=4.0,
        explicit_relax_steps=500,
        explicit_relax_tol=1.0e-4,
    ))

    # 候補 (p3) + (p1) 同時適用
    runs.append(_run_one(
        "exp_alpha0.5_relax500",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_mass_scaling_max_growth_per_update=4.0,
        explicit_mass_proportional_damping_alpha=0.5,
        explicit_relax_steps=500,
        explicit_relax_tol=1.0e-4,
    ))

    # 仮説 (p4): n_increments 増 → BC ramp が遅くなり explicit が追従するか
    runs.append(_run_one(
        "exp_ninc200_no_damp",
        solver_mode="explicit",
        n_increments_per_cycle=200,
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_mass_scaling_max_growth_per_update=4.0,
    ))

    print()
    print("=" * 72)
    print(f"{'label':32s} | frac  |   max|u|     | err_anal | err_imp | gate")
    print("─" * 72)
    u_implicit = next(r["max_u_trans"] for r in runs if r["label"] == "implicit_baseline")
    for r in runs:
        err_anal = abs(r["max_u_trans"] - u_analytical) / u_analytical
        err_imp = abs(r["max_u_trans"] - u_implicit) / u_implicit if u_implicit > 0 else 1.0
        gate_disp = r["max_u_trans"] < L_strand * 10.0
        gate_acc = (err_imp < 0.10) or (err_anal < 0.10)
        gate = "PASS" if (r["frac"] >= 0.999 and gate_disp and gate_acc) else "FAIL"
        print(
            f"  {r['label']:30s} | {r['frac']:5.3f} | {r['max_u_trans']:.3e} | "
            f"{err_anal*100:6.2f}% | {err_imp*100:6.2f}% | {gate}"
        )

    print()
    print("Gate (MCDD 凍結解除条件 (5)):")
    print("  - frac=1.0 完走 + max|u| < L_strand × 10 + 精度 < 10%")
    print(f"  - 解析解 max|u| = {u_analytical:.3f} mm（quarter circle）")
    print()
    print("status-382 結論:")
    print("  (p3) damping + (p1) relax 両 API 実装は正しいが、UL 定式化の")
    print("  update_reference が各増分の dynamic lag を reference に凍結するため、")
    print("  後続 relax phase では f_int(u_incr) ≈ 0 となり構造を平衡へ駆動できない。")
    print("  詳細は docs/status/status-382.md §3 を参照。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
