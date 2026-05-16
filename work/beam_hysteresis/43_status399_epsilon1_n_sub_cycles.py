"""work/beam_hysteresis/43_status399_epsilon1_n_sub_cycles.py — status-399 ε-1 再検証: `explicit_n_sub_cycles_per_increment` の効果実機計測.

[← README](README.md) | [← project README](../../README.md)

status-398 で確定した hypothesis 1（stepwise prescribed BC × mass scaling auto-tune の
interaction）に対する architectural fix（`explicit_n_sub_cycles_per_increment`）の
実機検証スクリプト。

**目的**: status-398 §5.2 で設計した sub-cycle 内部ループの実装が ε-1 sub-experiment
（n_strands=1 straight, free_end_mode=True, contact_enabled=False,
explicit_ul_disable_update=True）で u_x rel_err < 10% を達成するか確認.

**比較対象**:

- implicit baseline: u_x ≈ 4.996 mm（解析 cantilever 解と機械精度級一致）
- explicit-TL baseline (status-398 同設定): u_x ≈ 0.186 mm（96.29% under-deformation）
- explicit-TL + N=1000 sub-cycles（本実装の目標）: rel_err < 10% を期待

**実装メカニズム**:

`process.py` の `solver_mode=="explicit"` 経路で 1 QUERY を N 個の sub-step に分割し、
- `dt_inner = dt_sub / N`
- prescribed BC を `frac_k = frac_prev + (k/N)·(frac − frac_prev)` で線形補間
- f_ext も同係数で補間
- 各 sub-step で `_explicit_proc.process()` を呼ぶ

`ExplicitDynamicProcess` の mass scaling auto-tune は dt_inner と dt_critical_raw を
比較するため、N が大きいほど β_inner が縮小し T_1_scaled = β · T_1_raw も縮小、
t_total / T_1_scaled 比が十分大きくなり quasi-static 応答が成立する.

**実行**:

    uv run --extra dev python work/beam_hysteresis/43_status399_epsilon1_n_sub_cycles.py \\
        2>&1 | tee /tmp/status399_epsilon1_$(date +%s).log

**判定**:

- N=1000 で rel_err < 10% → 仮説 1 fix 成功、status-400 (ε-2 接触あり 3 strand) へ進行
- 達成不能 → hypothesis 2 / 3 を本格検証
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
    _collect_end_nodes,
)


@dataclass
class RunSummary:
    label: str
    frac: float
    n_increments: int
    n_cutbacks: int
    elapsed: float
    max_u_trans: float
    tip_disp: np.ndarray
    e_strain: float
    e_kin: float


def _extract_tip(u: np.ndarray, mesh: object) -> np.ndarray:
    _, right = _collect_end_nodes(
        mesh.connectivity, int(mesh.n_strands), np.asarray(mesh.strand_ids)
    )
    if not right:
        return np.zeros(3)
    um = u.reshape(-1, 6)
    return np.mean(um[right, :3], axis=0)


def _base_cfg() -> dict:
    """ε-1 sub-experiment (n_strands=1 straight) の基本 config (status-398 と同)."""
    return dict(
        n_strands=1,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.001,
        n_cycles=1,
        n_increments_per_cycle=20,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=False,
        penalty_exponent=1.5,
    )


def _explicit_overrides() -> dict:
    return dict(
        solver_mode="explicit",
        explicit_ul_disable_update=True,
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e5,
        explicit_kinetic_energy_budget_ratio=0.05,
    )


def _run(label: str, **overrides) -> RunSummary:
    base = _base_cfg()
    base.update(overrides)
    cfg = StrandBendingOscillationConfig(**base)

    print()
    print("─" * 72)
    print(f"[run] {label}")
    print(
        f"  n_inc={base['n_increments_per_cycle']}, "
        f"N_sub={base.get('explicit_n_sub_cycles_per_increment', 1)}, "
        f"β_max={base.get('explicit_mass_scaling_max_beta', 'N/A')}"
    )
    print("─" * 72)

    t0 = time.perf_counter()
    try:
        result = StrandBendingOscillationProcess().process(cfg)
        sr = result.solver_result
        frac = float(sr.load_history[-1]) if sr.load_history else 0.0
        e_hist = sr.energy_history
        e_kin = float(e_hist.entries[-1].kinetic_energy) if e_hist and e_hist.entries else 0.0
        e_str = float(e_hist.entries[-1].strain_energy) if e_hist and e_hist.entries else 0.0
        u = sr.u
        u_trans = u.reshape(-1, 6)[:, :3]
        max_ut = float(np.max(np.linalg.norm(u_trans, axis=1)))
        tip = _extract_tip(u, result.mesh)
        n_incr = int(sr.n_increments)
        n_cb = int(sr.n_cutbacks)
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        return RunSummary(
            label=label,
            frac=0.0,
            n_increments=0,
            n_cutbacks=0,
            elapsed=time.perf_counter() - t0,
            max_u_trans=float("nan"),
            tip_disp=np.full(3, float("nan")),
            e_strain=float("nan"),
            e_kin=float("nan"),
        )

    elapsed = time.perf_counter() - t0
    print(f"  frac={frac:.4f}, incr={n_incr}, cb={n_cb}, elapsed={elapsed:.2f}s")
    print(f"  max|u| = {max_ut:.4e} mm, tip = ({tip[0]:+.4e}, {tip[1]:+.4e}, {tip[2]:+.4e})")
    print(f"  E_strain={e_str:.4e}, E_kin={e_kin:.4e}")

    return RunSummary(
        label=label,
        frac=frac,
        n_increments=n_incr,
        n_cutbacks=n_cb,
        elapsed=elapsed,
        max_u_trans=max_ut,
        tip_disp=tip,
        e_strain=e_str,
        e_kin=e_kin,
    )


def main() -> int:
    print("=" * 72)
    print("status-399 ε-1 再検証: explicit_n_sub_cycles_per_increment の効果")
    print("=" * 72)
    print("基準: n_strands=1 straight, free_end_mode=True, contact_enabled=False,")
    print("      bending_curvature=0.001 (0.1 rad), explicit_ul_disable_update=True")
    print()
    print("status-398 reference:")
    print("  implicit:                  u_x = +4.996e+00 mm (ref)")
    print("  explicit-TL baseline:      u_x = +1.855e-01 mm (96.29% under)")
    print("  explicit-TL + n_inc=200:   u_x = +7.477e-01 mm (85.03% under)")
    print("  explicit-TL + n_inc=2000:  u_x = +2.252e+00 mm (54.93% under)")
    print("  explicit-TL + n_inc=20000: u_x = +5.268e+00 mm (5.45% over)")
    print()
    print("**status-399 期待**: N_sub=N で n_inc=N の n_inc 掃引と等価精度")
    print()

    # implicit baseline
    imp = _run("implicit_baseline", solver_mode="implicit")
    # explicit-TL default (N=1, n_inc=20) — status-398 と同設定で再現確認
    exp_base = _run("explicit_TL_default_N=1", **_explicit_overrides())

    # status-399 fix: N_sub 掃引（n_inc=20 固定）
    exp_n10 = _run(
        "explicit_TL_N=10",
        explicit_n_sub_cycles_per_increment=10,
        **_explicit_overrides(),
    )
    exp_n100 = _run(
        "explicit_TL_N=100",
        explicit_n_sub_cycles_per_increment=100,
        **_explicit_overrides(),
    )
    exp_n1000 = _run(
        "explicit_TL_N=1000",
        explicit_n_sub_cycles_per_increment=1000,
        **_explicit_overrides(),
    )

    # 結果サマリ
    print()
    print("=" * 72)
    print("結果サマリ（u_x_tip vs implicit reference）")
    print("=" * 72)
    cases = [imp, exp_base, exp_n10, exp_n100, exp_n1000]
    print(
        f"  {'label':<28} {'frac':>6} {'incr':>5} {'cb':>4} {'u_x [mm]':>12} {'rel_err':>10} {'elapsed':>10}"
    )
    for c in cases:
        rel = abs(c.tip_disp[0] - imp.tip_disp[0]) / max(abs(imp.tip_disp[0]), 1e-30)
        print(
            f"  {c.label:<28} {c.frac:>6.4f} {c.n_increments:>5d} "
            f"{c.n_cutbacks:>4d} {c.tip_disp[0]:>+12.4e} {rel:>9.2%} "
            f"{c.elapsed:>9.2f}s"
        )

    print()
    print("判定:")
    err_base = abs(exp_base.tip_disp[0] - imp.tip_disp[0]) / max(abs(imp.tip_disp[0]), 1e-30)
    print(f"  baseline (N=1) rel_err: {err_base:.2%}")
    target_gate = 0.10  # MCDD 凍結解除条件 (5): rel_err < 10%
    for c in [exp_n10, exp_n100, exp_n1000]:
        rel = abs(c.tip_disp[0] - imp.tip_disp[0]) / max(abs(imp.tip_disp[0]), 1e-30)
        passed = rel < target_gate
        verdict = "PASS (< 10%)" if passed else "FAIL (>= 10%)"
        print(f"  {c.label}: rel_err={rel:.2%} → {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
