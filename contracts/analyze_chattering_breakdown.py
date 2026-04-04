"""チャタリング内訳分析スクリプト（status-287）.

90度曲げ（接触あり Hertz α=1.5）を実行し、NR反復レベルのスナップショットから
チャタリングのタイプ別内訳を分析する。

タイプ分類:
  A: 接触活性集合振動（ペアON↔OFF + 接触DOF残差支配）
  B: 摩擦状態振動（stick↔slide + 接触DOF残差支配）
  C: 構造系収束不良（非接触DOF残差支配）
  D: 接線剛性不整合（収束率 > 0.9）

Usage:
    python contracts/analyze_chattering_breakdown.py 2>&1 | tee /tmp/log-chattering-$(date +%s).log

[← README](../README.md)
"""

import math
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.contact.solver._diagnostics import (  # noqa: E402
    NRIterationSnapshot,
    classify_chattering_type,
)
from xkep_cae.numerical_tests.strand_bending_oscillation import (  # noqa: E402
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def _analyze_increment_diagnostics(incr_diags, sr):
    """インクリメント診断からチャタリング内訳を分析."""
    print("\n" + "=" * 70)
    print("  チャタリング内訳分析 (status-287)")
    print("=" * 70)

    total_incr = len(incr_diags)
    cutback_incr = sum(1 for d in incr_diags if not d.converged)
    print(f"\n  総インクリメント数: {total_incr}")
    print(f"  カットバック数: {sr.n_cutbacks}")
    print(f"  不収束インクリメント数: {cutback_incr}")
    print(f"  最終frac: {sr.load_history[-1]:.4f}" if sr.load_history else "  最終frac: N/A")

    # 不収束インクリメントのfrac分布
    if cutback_incr > 0:
        failed_fracs = [d.load_frac for d in incr_diags if not d.converged]
        print(f"\n  不収束frac範囲: {min(failed_fracs):.4f} - {max(failed_fracs):.4f}")

        # frac区間ごとの不収束率
        bins = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
        print("\n  frac区間別 不収束率:")
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            in_bin = [d for d in incr_diags if lo <= d.load_frac < hi]
            if in_bin:
                failed = sum(1 for d in in_bin if not d.converged)
                rate = failed / len(in_bin) * 100
                print(f"    [{lo:.2f}, {hi:.2f}): {failed}/{len(in_bin)} = {rate:.1f}%")


def _analyze_nr_snapshots(diag):
    """ConvergenceDiagnosticsOutputのNRスナップショットからType分析."""
    snaps: list[NRIterationSnapshot] = diag.nr_iteration_snapshots
    if not snaps:
        print("  NRスナップショットなし（収束したインクリメント）")
        return

    print(f"\n  NR反復数: {len(snaps)}")

    # 全反復のType分類
    type_counter: Counter = Counter()
    for s in snaps:
        t = classify_chattering_type(s)
        type_counter[t] += 1

    print("\n  チャタリングType分布:")
    for t, cnt in type_counter.most_common():
        pct = cnt / len(snaps) * 100
        print(f"    Type {t}: {cnt}/{len(snaps)} = {pct:.1f}%")

    # 残差内訳（接触 vs 構造）
    c_norms = [s.contact_res_norm for s in snaps]
    s_norms = [s.structural_res_norm for s in snaps]
    c_dominant = sum(1 for c, s in zip(c_norms, s_norms, strict=True) if c > s)
    print(f"\n  接触残差支配: {c_dominant}/{len(snaps)} 反復")
    print(f"  構造残差支配: {len(snaps) - c_dominant}/{len(snaps)} 反復")

    # 活性集合変化の統計
    aset_changes = sum(1 for s in snaps if s.active_set_changed)
    fric_changes = sum(1 for s in snaps if s.friction_state_changed)
    print(f"\n  活性集合変化あり: {aset_changes}/{len(snaps)} 反復")
    print(f"  摩擦状態変化あり: {fric_changes}/{len(snaps)} 反復")

    # 収束率統計（att>=2のみ）
    rates = [s.convergence_rate for s in snaps if s.att >= 2]
    if rates:
        import numpy as np

        rates_arr = np.array(rates)
        print("\n  収束率統計 (att>=2):")
        print(f"    平均: {np.mean(rates_arr):.3f}")
        print(f"    中央値: {np.median(rates_arr):.3f}")
        print(f"    最大: {np.max(rates_arr):.3f}")
        print(
            f"    >0.9 (低収束): {np.sum(rates_arr > 0.9)}/{len(rates)} = {np.sum(rates_arr > 0.9) / len(rates) * 100:.1f}%"
        )

    # 詳細: 最終10反復
    print("\n  直近10反復の詳細:")
    print(
        "    att | ||R||/||f|| | R_c      | R_s      | active | slide | "
        "aset_chg | fric_chg | rate  | type"
    )
    print("    " + "-" * 100)
    for s in snaps[-10:]:
        t = classify_chattering_type(s)
        print(
            f"    {s.att:3d} | {s.res_ratio:.3e} | {s.contact_res_norm:.2e} | "
            f"{s.structural_res_norm:.2e} | {s.n_active:6d} | {s.n_sliding:5d} | "
            f"{'Y' if s.active_set_changed else 'N':>8s} | "
            f"{'Y' if s.friction_state_changed else 'N':>8s} | "
            f"{s.convergence_rate:.3f} | {t}"
        )


def main():
    print("=" * 70)
    print("  7本ヘリカル素線 接触あり 90度曲げ チャタリング内訳分析")
    print("  penalty_exponent=1.5 (Hertz), chattering_freeze=True")
    print("=" * 70)

    cfg = StrandBendingOscillationConfig(
        n_strands=7,
        wire_radius=0.5,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=math.pi / 200.0,
        n_cycles=1,
        n_increments_per_cycle=40,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=50,
        tol_force=1e-8,
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=True,
        loading_mode="rotation",
        penalty_exponent=1.5,
    )

    print("\nソルバー実行開始...")
    result = StrandBendingOscillationProcess().process(cfg)
    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0
    print(f"\n完了: frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}")

    # --- インクリメント診断分析 ---
    incr_diags = sr.increment_diagnostics if hasattr(sr, "increment_diagnostics") else []
    if incr_diags:
        _analyze_increment_diagnostics(incr_diags, sr)

    # --- 最終不収束インクリメントのNRスナップショット分析 ---
    last_diag = sr.diagnostics
    if last_diag and hasattr(last_diag, "nr_iteration_snapshots"):
        print("\n" + "=" * 70)
        print("  最終インクリメントのNR反復詳細分析")
        print("=" * 70)
        _analyze_nr_snapshots(last_diag)
    else:
        print("\n  最終診断データなし")

    # --- 全インクリメントのNRスナップショット集計 ---
    # increment_diagnostics にはNRスナップショットはないが、
    # 最終不収束の diag に入っている
    print("\n" + "=" * 70)
    print("  分析完了")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
