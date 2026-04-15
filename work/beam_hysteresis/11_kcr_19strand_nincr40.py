"""work/beam_hysteresis/11_kcr_19strand_nincr40.py — 19本撚線 κ_cr 実測（n_incr=40）.

[← README](README.md) | [← project README](../../README.md)

status-339 の推奨アクション 2: `n_increments_per_cycle=20→40` リトライ。

status-339 では `n_increments_per_cycle=20` で frac=0.484 Type D stall（未完走）。
曲率プロファイル過粗さが原因の仮説 C を検証するため、増分数を 2 倍化して
1 増分あたりの活性集合変化を緩和し、Newton 修正の追従を可能にする試行。

比較対象（status-339 ベースライン）:
    n_increments_per_cycle=20: frac=0.4839, incr=271, cb=39, elapsed=534.68s
    → n_increments_per_cycle=40 での frac/incr/cb/elapsed を実測

予測:
    - 所要時間 ~1200s（2x + 粗さ緩和効果の補正）
    - 完走（frac=1.0）到達で Type D stall 解消が確認される
    - stall 再発時は Type D の原因が曲率粗さではない（仮説 A/B/D の可能性）

status-340 で新設した ContactPairLayerClassifierProcess も同時に実行し、
完走時は (0,1)/(1,1)/(1,2)/(2,2) の層ペア別 κ_cr 平均を出力する
（バイモーダル仮説の定量検証）。

実行:
    python work/beam_hysteresis/11_kcr_19strand_nincr40.py 2>&1 \
        | tee /tmp/kcr_meas_19_nincr40_$(date +%s).log
"""

from __future__ import annotations

import sys
import time

import numpy as np

from xkep_cae.mesh.process import StrandMeshConfig, StrandMeshProcess
from xkep_cae.numerical_tests.contact_pair_analysis import (
    ContactPairAnalysisInput,
    ContactPairAnalysisProcess,
)
from xkep_cae.numerical_tests.contact_pair_layer_classifier import (
    ContactPairLayerClassifierInput,
    ContactPairLayerClassifierProcess,
)
from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    """19本撚線曲げ（n_incr=40）+ 接触ペア解析 + 層分類."""
    print("=" * 70)
    print("19本撚線 κ_cr 実測 n_incr=40 リトライ（status-339 推奨アクション 2）")
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
        n_increments_per_cycle=40,  # ★ status-339 推奨アクション 2（20→40）
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

    # ── ベースライン比較（status-339） ──
    print()
    print("=" * 70)
    print("status-339 比較（n_incr=20 ベースライン vs n_incr=40）")
    print("=" * 70)
    print(f"  {'項目':<20s} {'n_incr=20':>15s} {'n_incr=40':>15s}")
    print(f"  {'frac_completed':<20s} {'0.4839':>15s} {frac:>15.4f}")
    print(f"  {'n_increments':<20s} {'271':>15s} {sr.n_increments:>15d}")
    print(f"  {'n_cutbacks':<20s} {'39':>15s} {sr.n_cutbacks:>15d}")
    print(f"  {'elapsed [s]':<20s} {'534.68':>15s} {elapsed:>15.2f}")

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

    # κ_cr ヒストグラム
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

    # ── ContactPairLayerClassifierProcess 実行（status-340 バイモーダル仮説検証）──
    mesh_result = StrandMeshProcess().process(
        StrandMeshConfig(
            n_strands=cfg.n_strands,
            wire_radius=cfg.wire_radius,
            pitch_length=cfg.pitch_length,
            gap=cfg.gap,
            n_elements_per_pitch=cfg.n_elements_per_pitch,
            n_pitches=cfg.n_pitches,
        )
    )
    classifier = ContactPairLayerClassifierProcess().process(
        ContactPairLayerClassifierInput(
            kappa_cr_per_pair=analysis.kappa_cr_per_pair,
            per_pair_dissipation=analysis.per_pair_dissipation,
            strand_ids=mesh_result.mesh.strand_ids.tolist(),
            strand_layers=mesh_result.strand_layers,
        )
    )

    print()
    print("=" * 70)
    print("ContactPairLayerClassifier 結果（バイモーダル仮説検証）")
    print("=" * 70)
    print(f"  n_unique_layer_pairs: {classifier.n_unique_layer_pairs}")
    print()
    print(
        f"  {'(l_min,l_max)':<14s} {'n_pairs':>8s} {'n_slip':>7s} "
        f"{'κ_cr mean':>11s} {'κ_cr std':>11s} {'κ_cr CV':>8s} {'diss_sum':>11s}"
    )
    for lp in sorted(classifier.per_layer_pair_stats.keys()):
        s = classifier.per_layer_pair_stats[lp]
        cv = s.kappa_cr_std / s.kappa_cr_mean if s.kappa_cr_mean > 0 else 0.0
        print(
            f"  {str(lp):<14s} {s.n_pairs:>8d} {s.n_slipped:>7d} "
            f"{s.kappa_cr_mean:>11.3e} {s.kappa_cr_std:>11.3e} "
            f"{cv:>8.3f} {s.dissipation_sum:>11.3e}"
        )

    print()
    print("=" * 70)
    print("完了")
    print("=" * 70)
    return 0 if sr.converged else 1


if __name__ == "__main__":
    sys.exit(main())
