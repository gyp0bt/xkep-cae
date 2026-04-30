"""work/beam_hysteresis/31_render_19strand_explicit.py — 19本撚線 explicit 解 3D レンダリング.

[← README](README.md) | [← project README](../../README.md)

status-380 物理的妥当性検証（status-379 引継ぎ §6.1）。

19 本撚線 90° 曲げを **status-379 採択設定** (`solver_mode="explicit"` +
`mass_scaling_auto=True` + `max_beta=1e3`) で再実行し、frac=1.0 完走解の
3D 形状 / 接触力分布 / 軸応力 / 曲率を `Strand3DContourProcess` で可視化。

implicit は frac=0.5746 で stall するため両解の数値比較は成立しないが、
**explicit 解の物理的妥当性は次の観点で目視評価**:

- ヘリカル撚線構造が 90° 曲げ後も保持されているか
- 接触力分布が外周層で連続的に変化しているか（中心節点での集中などの異常がないか）
- 軸応力分布が曲げによる引張側 / 圧縮側で対称か
- 曲率分布が処方曲率と整合しているか

**追加比較（30_implicit_vs_explicit_7strand.py の知見と対比）**:
7 本撚線 explicit は 90° 曲げで数値発散（max |u| ≈ 1.6×10⁸ mm）したが、
19 本では status-379 で frac=1.0 完走 + E_kin/E_strain=1.15% 達成済。
19 本のみで explicit が機能する理由は本検証スクリプトでは詳細解析しないが、
3D 形状で物理的に妥当であることを担保する。

実行:
    uv run --extra dev python work/beam_hysteresis/31_render_19strand_explicit.py \\
        2>&1 | tee /tmp/render_19strand_explicit_$(date +%s).log

出力: `docs/verification/19strand_explicit/*.png`
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


def main() -> int:
    print("=" * 72)
    print("status-380 19本撚線 explicit 解 3D 可視化（status-379 採択設定再現）")
    print("=" * 72)

    cfg = StrandBendingOscillationConfig(
        n_strands=19,
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
        max_increments=10000,
        exclude_same_strand=True,
        free_end_mode=True,
        penalty_exponent=1.5,
        smoothing_delta=1000.0,
        # status-379 採択 explicit 設定
        solver_mode="explicit",
        explicit_courant_safety=0.9,
        explicit_courant_check_interval=10,
        explicit_mass_scaling_beta=1.0,
        explicit_mass_scaling_auto=True,
        explicit_mass_scaling_max_beta=1.0e3,
        explicit_kinetic_energy_budget_ratio=0.05,
        track_contact_pairs=True,
    )

    t0 = time.perf_counter()
    result = StrandBendingOscillationProcess().process(cfg)
    elapsed = time.perf_counter() - t0
    sr = result.solver_result
    frac = float(sr.load_history[-1]) if sr.load_history else 0.0

    e_hist = sr.energy_history
    e_kin = float(e_hist.entries[-1].kinetic_energy) if e_hist and e_hist.entries else 0.0
    e_str = float(e_hist.entries[-1].strain_energy) if e_hist and e_hist.entries else 0.0
    e_ratio = e_kin / max(e_str, 1e-30)

    import numpy as np

    u_trans = sr.u.reshape(-1, 6)[:, :3]
    max_u = float(np.max(np.linalg.norm(u_trans, axis=1)))

    print()
    print(
        f"frac={frac:.4f}, converged={sr.converged}, "
        f"incr={sr.n_increments}, cb={sr.n_cutbacks}, elapsed={elapsed:.2f}s"
    )
    print(f"max |u_trans| = {max_u:.4e} mm")
    print(f"E_kin={e_kin:.3e}, E_strain={e_str:.3e}, E_kin/E_strain={e_ratio:.3e}")

    # status-379 比較（参照値）
    ref_frac, ref_incr, ref_cb, ref_elapsed = 1.0000, 269, 31, 131.07
    ref_e_ratio = 1.148e-02
    print()
    print("status-379 参照値（同条件）:")
    print(
        f"  baseline: frac={ref_frac:.4f}, incr={ref_incr}, cb={ref_cb}, "
        f"elapsed={ref_elapsed:.2f}s, E_ratio={ref_e_ratio:.3e}"
    )
    print(
        f"  本実行  : frac={frac:.4f}, incr={sr.n_increments}, cb={sr.n_cutbacks}, "
        f"elapsed={elapsed:.2f}s, E_ratio={e_ratio:.3e}"
    )

    # 3D レンダリング
    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = project_root / "docs" / "verification" / "19strand_explicit"
    print()
    print(f"3D レンダリング → {out_dir}")
    try:
        from xkep_cae.output import Strand3DContourConfig, Strand3DContourProcess

        out = Strand3DContourProcess().process(
            Strand3DContourConfig(
                solver_result=sr,
                mesh=result.mesh,
                output_dir=str(out_dir),
                prefix="19strand_explicit",
            )
        )
        for path in out.image_paths:
            print(f"  [render] {path}")
        print(f"  接触要素 {out.n_contact_elements}, チャタリング要素 {out.n_chattering_elements}")
    except ImportError as e:
        print(f"  [render] matplotlib 未導入: {e}")

    # gate
    g_frac = frac >= 0.999
    g_energy = e_ratio < 0.05
    g_max_u = max_u < 1e3  # 100mm 撚線で max disp < 1m なら物理的に妥当
    overall = g_frac and g_energy and g_max_u
    print()
    print(f"  Gate frac=1.0           : {'PASS' if g_frac else 'FAIL'}")
    print(f"  Gate E_kin/E_strain<5%  : {'PASS' if g_energy else 'FAIL'}")
    print(f"  Gate max |u_trans|<1m   : {'PASS' if g_max_u else 'FAIL'}")
    print(f"  Total                   : {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
