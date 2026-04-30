"""work/beam_hysteresis/32_explicit_single_beam_no_contact.py — 接触なし単梁 explicit + mass scaling smoke test.

[← README](README.md) | [← project README](../../README.md)

status-380 物理的妥当性検証の **追加切り分け**。

7 本/19 本撚線で explicit + mass scaling auto-tune が数値発散することを発見した
（status-380 §1.1, §1.2）が、原因が **mass scaling 実装そのものの問題** なのか、
**接触ペナルティとの相互作用** なのかが切り分けられていない。

本スクリプトは **接触なしの単梁 90° 曲げ** で implicit / explicit を比較し、
explicit + mass scaling auto-tune が**接触なし系で物理的に妥当な解を返すか**を確認する:

- 接触なし単梁で explicit も妥当 → 接触ペナルティとの相互作用が破綻原因
- 接触なし単梁でも explicit 発散 → mass scaling 実装そのものに問題

設定:
- `n_strands=1` + `contact_enabled=False`: 接触検出スキップ、単一梁のみ
- `bending_curvature=0.015` で 90° 曲げ
- `free_end_mode=True` で右端に処方変位
- 他は status-380 7 本撚線比較と同条件

Gate（接触なしの単梁ベース）:
- 両 solver_mode で frac=1.0 完走
- max |u_trans| < L_strand × 10（撚線長 100mm に対し最大変位 1m 以内）
- 並進変位 L2 相対誤差 < 5%

実行:
    uv run --extra dev python work/beam_hysteresis/32_explicit_single_beam_no_contact.py \\
        2>&1 | tee /tmp/no_contact_single_$(date +%s).log
"""

from __future__ import annotations

import sys
import time

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _make_base_cfg() -> dict:
    """接触なし単梁の基本設定."""
    return dict(
        n_strands=1,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.015,  # 90° 曲げ
        n_cycles=1,
        n_increments_per_cycle=20,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=False,  # ★ 接触なし
    )


def _run_one(label: str, **overrides) -> dict:
    base = _make_base_cfg()
    base.update(overrides)
    cfg = StrandBendingOscillationConfig(**base)

    print()
    print("─" * 72)
    print(f"[run] label={label}, solver_mode={base.get('solver_mode', 'implicit')}")
    print("─" * 72)

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = float(sr.load_history[-1]) if sr.load_history else 0.0
    e_hist = sr.energy_history
    e_kin = float(e_hist.entries[-1].kinetic_energy) if e_hist and e_hist.entries else 0.0
    e_str = float(e_hist.entries[-1].strain_energy) if e_hist and e_hist.entries else 0.0
    e_ratio = e_kin / max(e_str, 1e-30)

    u = sr.u
    n_total_nodes = u.shape[0] // 6
    u_trans = u.reshape(n_total_nodes, 6)[:, :3]
    max_u_trans = float(np.max(np.linalg.norm(u_trans, axis=1)))

    print(
        f"  frac={frac:.4f}, converged={sr.converged}, "
        f"incr={sr.n_increments}, cb={sr.n_cutbacks}, elapsed={elapsed:.2f}s"
    )
    print(f"  max |u_trans| = {max_u_trans:.4e} mm")
    print(f"  E_kin={e_kin:.3e}, E_strain={e_str:.3e}, ratio={e_ratio:.3e}")

    return dict(
        label=label,
        frac=frac,
        converged=bool(sr.converged),
        n_increments=int(sr.n_increments),
        n_cutbacks=int(sr.n_cutbacks),
        elapsed=elapsed,
        u=u.copy(),
        max_u_trans=max_u_trans,
        e_kin=e_kin,
        e_strain=e_str,
        e_ratio=e_ratio,
    )


def main() -> int:
    print("=" * 72)
    print("status-380 追加: 接触なし単梁 90° 曲げ implicit vs explicit 切り分け")
    print("=" * 72)
    print(
        "目的: explicit + mass scaling 発散の原因が mass scaling 実装そのもの"
        "なのか、\n      接触ペナルティとの相互作用なのかを切り分ける。"
    )
    print("Gate: 両 solver_mode で frac=1.0 + max |u_trans| < 1m + L2 相対誤差 < 5%")

    imp = _run_one("implicit", solver_mode="implicit")

    exp = _run_one(
        "explicit_mass_scaling_auto",
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_kinetic_energy_budget_ratio=0.05,
    )

    # 比較
    L_strand = 100.0
    diff = imp["u"] - exp["u"]
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(imp["u"]), 1e-30))

    print()
    print("=" * 72)
    print("差分統計（接触なし単梁、implicit ↔ explicit）")
    print("=" * 72)
    print(f"  並進 + 回転 DOF L2 相対誤差 = {rel_l2:.3e}")
    print(f"  imp max |u_trans|          = {imp['max_u_trans']:.3e} mm")
    print(f"  exp max |u_trans|          = {exp['max_u_trans']:.3e} mm")

    g_frac = imp["frac"] >= 0.999 and exp["frac"] >= 0.999
    g_phys_imp = imp["max_u_trans"] < L_strand * 10
    g_phys_exp = exp["max_u_trans"] < L_strand * 10
    g_l2 = rel_l2 < 0.05

    print()
    print("─" * 72)
    print("Gate 判定（接触なし単梁）")
    print("─" * 72)
    print(f"  両 frac=1.0 完走         : {'PASS' if g_frac else 'FAIL'}")
    print(f"  imp 物理的妥当性 (|u|<1m): {'PASS' if g_phys_imp else 'FAIL'}")
    print(f"  exp 物理的妥当性 (|u|<1m): {'PASS' if g_phys_exp else 'FAIL'}")
    print(f"  L2 相対誤差 < 5%         : {'PASS' if g_l2 else 'FAIL'}")
    overall = g_frac and g_phys_imp and g_phys_exp and g_l2

    print()
    print("=" * 72)
    print("切り分け判定")
    print("=" * 72)
    if g_phys_exp:
        print("  → exp 接触なし単梁で**物理的に妥当**")
        print("  → 7 本/19 本撚線 explicit 発散の原因は **接触ペナルティとの相互作用**")
        print("    （mass scaling 実装そのものは健全、接触系で慣性 vs ペナルティ剛性")
        print("    バランスが破綻する）")
    else:
        print("  → exp 接触なし単梁でも**発散**")
        print("  → 7 本/19 本撚線 explicit 発散の原因は **mass scaling 実装そのもの**")
        print("    （β² 倍化が単梁系でも準静的近似を壊している）")
    print("=" * 72)

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
