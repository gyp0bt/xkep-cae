"""work/beam_hysteresis/30_implicit_vs_explicit_7strand.py — 7本撚線 implicit vs explicit 物理的妥当性比較.

[← README](README.md) | [← project README](../../README.md)

status-380 物理的妥当性検証（status-379 引継ぎ §6.1）。

7 本撚線 90° 曲げを **両ソルバーで完走させ**、両者の解を直接比較する。
19 本撚線は implicit が frac=0.5746 で stall するため両者比較が成立しない。
**7 本撚線は implicit / explicit ともに frac=1.0 完走可能** であり、
両者の数値解一致が陽解法 + mass scaling auto-tune の物理的妥当性を担保する。

**比較項目**:

1. **最終変位**: max |u_trans|, ノード単位 L2 ノルム差 ||u_imp − u_exp|| / ||u_imp||
2. **先端変位**: 自由端中心 (右端 MPC 参照点) の (u_x, u_y, u_z)
3. **接触力分布**: 最終時刻の active pair 群について p_n 統計
   (mean, max, count) を比較。
4. **エネルギー**: E_kin / E_strain 比（陽解法の準静的近似破綻チェック）
5. **3D レンダリング**: implicit / explicit 各々を Strand3DContourProcess で
   `docs/verification/7strand_imp_vs_exp/` 以下に保存し、目視整合確認。

**Gate**:

- 両ソルバーとも frac=1.0 完走
- 最終変位 L2 相対誤差 < 5%（陽解法は時間積分オーダー O(dt) なのでこの程度を許容）
- 先端変位 各成分相対誤差 < 5%
- 接触力 mean p_n の相対誤差 < 20%（pair 数差で揺らぎが大きい）
- explicit の E_kin/E_strain < 5%（mass scaling 準静的近似有効性）

実行:
    uv run --extra dev python work/beam_hysteresis/30_implicit_vs_explicit_7strand.py \\
        2>&1 | tee /tmp/imp_vs_exp_7strand_$(date +%s).log

設計参照:
    docs/status/status-379.md §6.1（物理的妥当性検証 引継ぎ）
    xkep_cae/time_integration/docs/time_integration_explicit.md
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xkep_cae.numerical_tests.strand_bending_oscillation import (
    StrandBendingOscillationConfig,
    StrandBendingOscillationProcess,
)


@dataclass
class RunSummary:
    """1 つの solver_mode 実行のサマリ."""

    label: str
    frac: float
    converged: bool
    n_increments: int
    n_cutbacks: int
    elapsed: float
    u: np.ndarray
    max_u_trans: float
    tip_disp: np.ndarray  # (u_x, u_y, u_z) at right MPC reference point
    p_n_active: np.ndarray  # array of p_n for active pairs (p_n > 0)
    e_kin: float
    e_strain: float
    e_ratio: float
    n_strand_nodes: int
    mesh: object
    solver_result: object


def _extract_tip_disp(
    u: np.ndarray,
    mesh: object,
    *,
    ndof_per_node: int = 6,
) -> np.ndarray:
    """右端撚線ノード群の重心の並進変位 (u_x, u_y, u_z) を返す.

    `free_end_mode=True` では MPC 参照点が存在しないため、各素線右端ノードの
    重心変位を「先端変位」として使用する（処方変位を直接受けるノード群）。
    """
    from xkep_cae.numerical_tests.strand_bending_oscillation import _collect_end_nodes

    _, right_nodes = _collect_end_nodes(
        mesh.connectivity, int(mesh.n_strands), np.asarray(mesh.strand_ids)
    )
    if not right_nodes:
        return np.zeros(3)
    u_mat = u.reshape(-1, ndof_per_node)
    tip = np.zeros(3)
    for n in right_nodes:
        tip += u_mat[n, :3]
    tip /= len(right_nodes)
    return tip


def _extract_active_pn(manager: object) -> np.ndarray:
    """final_contact_manager.pairs から active pair の p_n 配列を返す."""
    if manager is None or not hasattr(manager, "pairs"):
        return np.array([])
    p_n = []
    for p in manager.pairs:
        if hasattr(p, "state") and float(p.state.p_n) > 0.0:
            p_n.append(float(p.state.p_n))
    return np.asarray(p_n)


def _make_base_cfg() -> dict:
    """両 solver_mode で共通の物理 / メッシュ / 接触パラメータ."""
    return dict(
        n_strands=7,
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
        penalty_exponent=1.5,
        smoothing_delta=1000.0,  # status-359 採択（7 本系で frac=1.0 安定）
    )


def _run_one(label: str, **overrides) -> RunSummary:
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

    # 並進 DOF のみ抽出して max |u_trans|
    u = sr.u
    n_total_nodes = u.shape[0] // 6
    u_trans = u.reshape(n_total_nodes, 6)[:, :3]
    max_u_trans = float(np.max(np.linalg.norm(u_trans, axis=1)))

    n_strand_nodes = result.n_strand_nodes
    tip = _extract_tip_disp(u, result.mesh)
    p_n_active = _extract_active_pn(sr.final_contact_manager)

    print(
        f"  frac={frac:.4f}, converged={sr.converged}, "
        f"incr={sr.n_increments}, cb={sr.n_cutbacks}, elapsed={elapsed:.2f}s"
    )
    print(f"  max |u_trans| = {max_u_trans:.4e}")
    print(f"  tip disp = ({tip[0]:.4e}, {tip[1]:.4e}, {tip[2]:.4e})")
    if p_n_active.size > 0:
        print(
            f"  active pairs: n={p_n_active.size}, "
            f"mean p_n={p_n_active.mean():.3e}, max p_n={p_n_active.max():.3e}"
        )
    else:
        print("  active pairs: 0 (no active contact at final state)")
    print(f"  E_kin={e_kin:.3e}, E_strain={e_str:.3e}, ratio={e_ratio:.3e}")

    return RunSummary(
        label=label,
        frac=frac,
        converged=bool(sr.converged),
        n_increments=int(sr.n_increments),
        n_cutbacks=int(sr.n_cutbacks),
        elapsed=elapsed,
        u=u.copy(),
        max_u_trans=max_u_trans,
        tip_disp=tip,
        p_n_active=p_n_active,
        e_kin=e_kin,
        e_strain=e_str,
        e_ratio=e_ratio,
        n_strand_nodes=n_strand_nodes,
        mesh=result.mesh,
        solver_result=sr,
    )


def _compare(imp: RunSummary, exp: RunSummary) -> dict[str, float]:
    """implicit / explicit 解の差分統計を返す."""
    # 撚線ノードの並進 DOF のみ比較（参照点 / 回転 DOF は除外）
    n_strand_nodes = min(imp.n_strand_nodes, exp.n_strand_nodes)

    def trans(sr: RunSummary) -> np.ndarray:
        u_full = sr.u.reshape(-1, 6)[:n_strand_nodes, :3]
        return u_full

    u_imp = trans(imp)
    u_exp = trans(exp)
    diff = u_imp - u_exp
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(u_imp), 1e-30))

    # 先端変位の各成分相対誤差
    tip_diff = imp.tip_disp - exp.tip_disp
    tip_rel = np.abs(tip_diff) / np.maximum(np.abs(imp.tip_disp), 1e-12)

    # 接触力 mean 比較
    if imp.p_n_active.size > 0 and exp.p_n_active.size > 0:
        pn_imp_mean = float(imp.p_n_active.mean())
        pn_exp_mean = float(exp.p_n_active.mean())
        pn_rel = abs(pn_exp_mean - pn_imp_mean) / max(pn_imp_mean, 1e-30)
    else:
        pn_imp_mean = float(imp.p_n_active.mean()) if imp.p_n_active.size else 0.0
        pn_exp_mean = float(exp.p_n_active.mean()) if exp.p_n_active.size else 0.0
        pn_rel = float("nan")

    return {
        "rel_l2_displacement": rel_l2,
        "tip_rel_x": float(tip_rel[0]),
        "tip_rel_y": float(tip_rel[1]),
        "tip_rel_z": float(tip_rel[2]),
        "pn_imp_mean": pn_imp_mean,
        "pn_exp_mean": pn_exp_mean,
        "pn_rel_diff": pn_rel,
        "n_active_imp": float(imp.p_n_active.size),
        "n_active_exp": float(exp.p_n_active.size),
    }


def _render(sr: RunSummary, output_dir: Path, prefix: str) -> None:
    """Strand3DContourProcess で 3D レンダリング."""
    try:
        from xkep_cae.output import Strand3DContourConfig, Strand3DContourProcess
    except ImportError as e:
        print(f"  [render] matplotlib 未インストールのため skip: {e}")
        return

    cfg = Strand3DContourConfig(
        solver_result=sr.solver_result,
        mesh=sr.mesh,
        output_dir=str(output_dir),
        prefix=prefix,
    )
    out = Strand3DContourProcess().process(cfg)
    for path in out.image_paths:
        print(f"  [render] {path}")
    print(f"  接触要素 {out.n_contact_elements}, チャタリング要素 {out.n_chattering_elements}")


def main() -> int:
    print("=" * 72)
    print("status-380 物理的妥当性検証: 7本撚線 implicit vs explicit (90° 曲げ)")
    print("=" * 72)
    print("Gate:")
    print("  - 両 solver_mode で frac=1.0 完走")
    print("  - 並進変位 L2 相対誤差 < 5%")
    print("  - 先端変位 各成分相対誤差 < 5%")
    print("  - 接触力 mean 相対誤差 < 20%")
    print("  - explicit E_kin/E_strain < 5%")

    # implicit ベースライン
    imp = _run_one(
        "implicit",
        solver_mode="implicit",
    )

    # explicit + mass scaling auto-tune
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
    print()
    print("=" * 72)
    print("差分統計（implicit ↔ explicit）")
    print("=" * 72)
    diff = _compare(imp, exp)
    print(f"  並進変位 L2 相対誤差         = {diff['rel_l2_displacement']:.3e}")
    print(
        f"  先端変位 相対誤差           "
        f"= ({diff['tip_rel_x']:.3e}, {diff['tip_rel_y']:.3e}, {diff['tip_rel_z']:.3e})"
    )
    print(
        f"  active pair 数              "
        f"= imp:{int(diff['n_active_imp'])} / exp:{int(diff['n_active_exp'])}"
    )
    print(
        f"  接触力 p_n mean             "
        f"= imp:{diff['pn_imp_mean']:.3e} / exp:{diff['pn_exp_mean']:.3e} "
        f"(rel diff {diff['pn_rel_diff']:.3e})"
    )

    # gate
    L_strand = 100.0  # 撚線長 [mm]
    g_frac = imp.frac >= 0.999 and exp.frac >= 0.999
    g_disp = diff["rel_l2_displacement"] < 0.05
    g_tip = diff["tip_rel_x"] < 0.05 and diff["tip_rel_y"] < 0.05 and diff["tip_rel_z"] < 0.05
    g_pn = np.isnan(diff["pn_rel_diff"]) or diff["pn_rel_diff"] < 0.20
    g_ekin = exp.e_ratio < 0.05
    # 物理的妥当性 gate（status-380 追加）: max |u_trans| < L_strand × 10
    # （撚線長 100mm に対し最大変位 1m を超えたら数値発散とみなす）
    g_phys_imp = imp.max_u_trans < L_strand * 10
    g_phys_exp = exp.max_u_trans < L_strand * 10

    print()
    print("─" * 72)
    print("Gate 判定")
    print("─" * 72)
    print(f"  両 frac=1.0 完走         : {'PASS' if g_frac else 'FAIL'}")
    print(
        f"  imp 物理的妥当性 (|u|<1m): "
        f"{'PASS' if g_phys_imp else 'FAIL'} (max|u|={imp.max_u_trans:.3e})"
    )
    print(
        f"  exp 物理的妥当性 (|u|<1m): "
        f"{'PASS' if g_phys_exp else 'FAIL'} (max|u|={exp.max_u_trans:.3e})"
    )
    print(f"  並進変位 L2 < 5%         : {'PASS' if g_disp else 'FAIL'}")
    print(f"  先端変位 各成分 < 5%      : {'PASS' if g_tip else 'FAIL'}")
    print(f"  接触力 mean < 20%        : {'PASS' if g_pn else 'FAIL'}")
    print(f"  explicit E_kin/E_str<5%  : {'PASS' if g_ekin else 'FAIL'}")
    overall = g_frac and g_phys_imp and g_phys_exp and g_disp and g_tip and g_pn and g_ekin
    print(f"  Total                    : {'PASS' if overall else 'FAIL'}")
    print("=" * 72)

    # 3D レンダリング
    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = project_root / "docs" / "verification" / "7strand_imp_vs_exp"
    print()
    print(f"3D レンダリング → {out_dir}")
    _render(imp, out_dir, "implicit")
    _render(exp, out_dir, "explicit")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
