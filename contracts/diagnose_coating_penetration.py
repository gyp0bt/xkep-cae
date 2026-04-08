"""被膜貫入量診断スクリプト.

被膜付き7本撚線90度曲げで、被膜圧縮量が被膜厚さに対して
どの程度かを定量評価する。

coat_comp / coat_total の比率:
  0.0 = 被膜未圧縮（ギャップあり）
  0.5 = 被膜半圧縮
  1.0 = 被膜全圧縮（芯線接触）
  >1.0 = 芯線貫入（物理的に不可能）

Usage:
    python contracts/diagnose_coating_penetration.py 2>&1 | tee /tmp/log-coat-diag-$(date +%s).log

[← README](../README.md)
"""

from __future__ import annotations

import math
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from xkep_cae.contact._contact_pair import _evolve_pair  # noqa: E402
from xkep_cae.contact.solver import process as solver_process  # noqa: E402
from xkep_cae.numerical_tests.strand_bending_oscillation import (  # noqa: E402
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)

# --- Monkeypatch: 被膜圧縮量を記録する ---
_coating_log: list[dict] = []
_original_evolve_pair = _evolve_pair


def _patched_evolve_pair(pair, **kwargs):
    """_evolve_pairをフックして被膜圧縮量を記録."""
    result = _original_evolve_pair(pair, **kwargs)
    if "state" in kwargs:
        new_state = kwargs["state"]
        cc_prev = getattr(new_state, "coating_compression_prev", None)
        # coating_compression_prev が変わった場合 = 収束後の保存
        if cc_prev is not None and cc_prev != getattr(pair.state, "coating_compression_prev", 0.0):
            # 収束後のsnapshot
            _coating_log.append(
                {
                    "coating_compression": cc_prev,  # 今ステップの圧縮量
                    "core_radius_a": pair.core_radius_a,
                    "core_radius_b": pair.core_radius_b,
                    "radius_a": pair.radius_a,
                    "radius_b": pair.radius_b,
                }
            )
    return result


# Monkeypatch適用
solver_process._evolve_pair = _patched_evolve_pair  # type: ignore[attr-defined]


def main() -> None:
    """被膜貫入診断を実行."""
    coating_thickness = 0.05  # mm
    wire_radius = 0.5  # mm
    coat_total = 2 * coating_thickness

    print("=" * 70)
    print("被膜貫入量診断")
    print("=" * 70)
    print(f"  wire_radius       = {wire_radius} mm")
    print(f"  coating_thickness = {coating_thickness} mm")
    print(f"  coat_total        = {coat_total} mm (2本分)")
    print(f"  core_radius       = {wire_radius - coating_thickness} mm")
    print("  coating_stiffness = 1e6 N/mm")
    print()

    kappa_90 = math.pi / 2.0 / 100.0

    cfg = StrandBendingOscillationConfig(
        n_strands=7,
        wire_radius=wire_radius,
        pitch_length=100.0,
        n_elements_per_pitch=16,
        n_pitches=1.0,
        E=130.0e3,
        nu=0.3,
        rho=8.96e-9,
        bending_curvature=kappa_90,
        n_cycles=1,
        n_increments_per_cycle=40,
        rho_inf=0.9,
        mu=0.15,
        max_nr_attempts=200,
        tol_force=1e-8,
        max_increments=2000,
        exclude_same_strand=True,
        free_end_mode=True,
        contact_enabled=True,
        penalty_exponent=1.5,
        # 揺動なし（曲げのみ）
        n_oscillation_cycles=0,
        # 被膜パラメータ
        coating_stiffness=1e6,
        coating_damping=1e4,
        coating_mu=0.3,
        coating_thickness=coating_thickness,
    )

    _coating_log.clear()
    print("曲げ解析実行中...")
    result = StrandBendingOscillationProcess().process(cfg)

    sr = result.solver_result
    frac = sr.load_history[-1] if sr.load_history else 0.0
    print(f"\n結果: frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}")

    # --- 被膜圧縮量分析 ---
    print("\n" + "=" * 70)
    print("被膜圧縮量分析")
    print("=" * 70)

    if not _coating_log:
        print("  被膜圧縮データなし")
        return

    cc_vals = np.array([d["coating_compression"] for d in _coating_log])
    # coat_total per pair
    coat_totals = np.array(
        [
            (d["radius_a"] - d["core_radius_a"]) + (d["radius_b"] - d["core_radius_b"])
            for d in _coating_log
        ]
    )

    active = cc_vals > 0.0
    print(f"  記録ペア数: {len(cc_vals)}")
    print(f"  活性ペア数: {active.sum()} ({100 * active.sum() / len(cc_vals):.1f}%)")

    if active.sum() == 0:
        print("  被膜圧縮なし")
        return

    cc_active = cc_vals[active]
    ct_active = coat_totals[active]
    ratios = cc_active / ct_active

    print("\n  被膜圧縮量 [mm]:")
    print(f"    平均     = {cc_active.mean():.6f}")
    print(f"    中央値   = {np.median(cc_active):.6f}")
    print(f"    最大     = {cc_active.max():.6f}")
    print(f"    95%tile  = {np.percentile(cc_active, 95):.6f}")

    print("\n  圧縮率 (coat_comp / coat_total):")
    print(f"    平均     = {ratios.mean():.4f} ({100 * ratios.mean():.2f}%)")
    print(f"    中央値   = {np.median(ratios):.4f} ({100 * np.median(ratios):.2f}%)")
    print(f"    最大     = {ratios.max():.4f} ({100 * ratios.max():.2f}%)")
    print(
        f"    95%tile  = {np.percentile(ratios, 95):.4f} ({100 * np.percentile(ratios, 95):.2f}%)"
    )

    over_50 = (ratios > 0.5).sum()
    over_100 = (ratios > 1.0).sum()
    print(f"\n  >50%圧縮: {over_50} ペア ({100 * over_50 / len(ratios):.1f}%)")
    print(f"  >100%圧縮（芯線貫入）: {over_100} ペア ({100 * over_100 / len(ratios):.1f}%)")

    if over_100 > 0:
        max_over = ratios.max()
        excess = cc_active.max() - ct_active[np.argmax(cc_active)]
        print(f"  最大芯線貫入量: {excess:.6f} mm")
        print(f"  最大圧縮率: {max_over:.2f} ({100 * max_over:.1f}%)")

    # 結論
    print("\n" + "=" * 70)
    print("結論")
    print("=" * 70)
    if ratios.max() < 0.01:
        print("  被膜圧縮は無視できるレベル (< 1%)")
        print("  → 被膜stiffnessが高すぎて実質的にハードコンタクト")
        print("  → 物理的な被膜効果ではなく数値的正則化として機能")
    elif ratios.max() < 0.5:
        print("  被膜圧縮は軽微 (< 50%)")
        print("  → 被膜が適度に圧縮されている")
    elif ratios.max() < 1.0:
        print("  被膜圧縮は大きい (50-100%)")
        print("  → 被膜がほぼ完全に圧縮されるケースあり")
    else:
        print(f"  被膜が物理厚さを超えて圧縮 (max {100 * ratios.max():.0f}%)")
        print("  → 芯線同士が貫入しており物理的に不正確")
        print("  → 非線形硬化（Hertz型）or 圧縮上限の導入が必要")


if __name__ == "__main__":
    main()
