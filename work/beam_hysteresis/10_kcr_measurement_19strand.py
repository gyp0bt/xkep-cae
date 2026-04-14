"""work/beam_hysteresis/10_kcr_measurement_19strand.py — 19本撚線 κ_cr 実測.

[← README](README.md) | [← project README](../../README.md)

status-338 で確立した 7本撚線 κ_cr 実測手順を 19本撚線（1+6+12 構造）へ
スケールアップする。

7本撚線ベースライン（status-338）:
    κ_cr mean=5.80e-3, CV=0.30, n_slipped=24/26, 90°曲げ frac=1.0, 281s

19本撚線で期待される変化:
    - n_unique_pairs が 26 → ~100 程度（外層-外層ペア増、2層化）
    - 層間（内層 vs 外層）で κ_cr 分布がバイモーダルになる可能性
    - 計算時間 ~1000-2000s（n² 接触ペア + K_st 準線形スケール、status-326）

実行:
    python work/beam_hysteresis/10_kcr_measurement_19strand.py 2>&1 \
        | tee /tmp/kcr_meas_19_$(date +%s).log
"""

from __future__ import annotations

import sys
import time

import numpy as np

from xkep_cae.numerical_tests.contact_pair_analysis import (
    ContactPairAnalysisInput,
    ContactPairAnalysisProcess,
)
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    """19本撚線曲げ + 接触ペア解析."""
    print("=" * 70)
    print("19本撚線 κ_cr 実測（status-338 の 7本撚線スケールアップ）")
    print("=" * 70)

    cfg = StrandBendingOscillationConfig(
        n_strands=19,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=0.015,  # 90度曲げ相当（status-338 と同条件）
        n_cycles=1,
        n_increments_per_cycle=20,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        penalty_exponent=1.5,
        track_contact_mk=True,
        track_contact_pairs=True,
    )
    print(
        f"cfg: n_strands={cfg.n_strands}, n_pitches={cfg.n_pitches}, "
        f"κ_max={cfg.bending_curvature}, mu={cfg.mu}, "
        f"n_incr_per_cycle={cfg.n_increments_per_cycle}"
    )
    print()

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result

    frac = sr.load_history[-1] if sr.load_history else 0.0
    print()
    print("=" * 70)
    print("ソルバー結果")
    print("=" * 70)
    print(f"  frac_completed: {frac:.4f}")
    print(f"  converged:      {sr.converged}")
    print(f"  n_increments:   {sr.n_increments}")
    print(f"  n_cutbacks:     {sr.n_cutbacks}")
    print(f"  elapsed:        {elapsed:.2f} s")
    print(f"  max |u|:        {float(np.max(np.abs(sr.u))):.6e}")
    mk_len = len(sr.moment_curvature_history)
    pair_len = len(sr.contact_pair_history)
    print(f"  mk_history:     {mk_len} entries")
    print(f"  pair_history:   {pair_len} entries")

    # ── ContactPairAnalysisProcess 実行 ──
    analysis = ContactPairAnalysisProcess().process(
        ContactPairAnalysisInput(
            contact_pair_history=sr.contact_pair_history,
            moment_curvature_history=sr.moment_curvature_history,
        )
    )

    print()
    print("=" * 70)
    print("ContactPairAnalysisProcess 結果")
    print("=" * 70)
    print(f"  n_steps:            {analysis.n_steps}")
    print(f"  n_unique_pairs:     {analysis.n_unique_pairs}")
    print(f"  n_slipped_pairs:    {analysis.n_slipped_pairs}")
    print(f"  total_dissipation:  {analysis.total_dissipation:.6e}")

    if analysis.n_slipped_pairs > 0:
        print(f"  κ_cr mean:          {analysis.kappa_cr_mean:.6e}")
        print(f"  κ_cr std:           {analysis.kappa_cr_std:.6e}")
        print(f"  κ_cr min:           {analysis.kappa_cr_min:.6e}")
        print(f"  κ_cr max:           {analysis.kappa_cr_max:.6e}")
        cv = analysis.kappa_cr_std / analysis.kappa_cr_mean if analysis.kappa_cr_mean > 0 else 0.0
        print(f"  κ_cr CV (std/mean): {cv:.4f}")

    # 活性ペア数推移（代表点）
    n_active_tup = analysis.n_active_per_step
    if n_active_tup:
        print()
        print("  活性ペア数推移（最初/中間/最後）:")
        n = len(n_active_tup)
        for i in (0, n // 4, n // 2, 3 * n // 4, n - 1):
            load_frac = analysis.load_frac_per_step[i]
            print(f"    step[{i}]: load_frac={load_frac:.4f}, n_active={n_active_tup[i]}")
        print(f"  max active: {max(n_active_tup)}")

    # per_pair_dissipation トップ10（7本撚線より多いので 10 に拡張）
    if analysis.per_pair_dissipation:
        sorted_diss = sorted(
            analysis.per_pair_dissipation.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:10]
        print()
        print("  per-pair dissipation top-10:")
        for (a, b), d in sorted_diss:
            kcr = analysis.kappa_cr_per_pair.get((a, b), None)
            kcr_str = f"κ_cr={kcr:.3e}" if kcr is not None else "κ_cr=N/A(未スリップ)"
            print(f"    ({a:3d}, {b:3d}): dissipation={d:.4e}, {kcr_str}")

    # κ_cr ヒストグラム（15 bin に拡張、19本は分布裾が長い想定）
    if analysis.n_slipped_pairs >= 5:
        kcrs = np.array(list(analysis.kappa_cr_per_pair.values()))
        kmin, kmax = kcrs.min(), kcrs.max()
        if kmax > kmin:
            n_bins = 15
            hist, edges = np.histogram(kcrs, bins=n_bins, range=(kmin, kmax))
            print()
            print(f"  κ_cr ヒストグラム（{n_bins} bin）:")
            for i in range(n_bins):
                bar = "#" * int(hist[i])
                print(f"    [{edges[i]:.3e}, {edges[i + 1]:.3e}): {hist[i]:3d} {bar}")

    print()
    print("=" * 70)
    print("完了")
    print("=" * 70)
    return 0 if sr.converged else 1


if __name__ == "__main__":
    sys.exit(main())
