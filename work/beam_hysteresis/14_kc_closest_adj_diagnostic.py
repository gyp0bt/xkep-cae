"""status-355: KcClosestPointStiffnessProcess 隣接拡張の効果見込み診断.

status-354 で仮説 A（K_hermite_adj への I_nn フル項拡張）が反証され、
Phase C-3 を Phase C-3' へ再々定義。仮説 B は
「KcClosestPointStiffnessProcess を隣接ノード DOF 列にも拡張することで
s-tracking 補償経路 (2) を解析的に組み込む」である。

本スクリプトは `test_kc_component_fd.py::test_helical_3d_hermite`
のシナリオ（n_elems=3, helical_z=True, Hermite+Hertz）で、現行の
K_c_analytical と FD K_c を active-pair DOF / 隣接 DOF に分解して比較し、
「仮説 B が効く余地 = 隣接ブロックに FD 非零が存在するか」を定量評価する。

出力:
- active 12 DOF ブロックの rel_err
- 隣接 12 DOF ブロックの rel_err（現行 K_closest は 0 を返すので、
  FD 絶対値 = analytical との差 = そのまま不整合量）
- comp x/y/z 別シェア

期待結果:
- active ブロック rel_err << 1% → K_c 自体は active 内で正確
- adj ブロックは K_c=0 なので 100% 不整合、FD 絶対値が有意 → 仮説 B 効く
- 全体 rel_err 1.795% が adj 由来か active 由来かの切り分け

背景: docs/status/status-354.md, docs/math/03_huber_contact_penalty.md §7
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# プロジェクトルートを import path に
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from xkep_cae.contact.contact_force.tests.test_kc_component_fd import (  # noqa: E402
    _compute_fd_kc,
    _get_analytical_components,
    _make_two_segment_scenario,
)


def main() -> None:
    # status-354/295 と同一設定
    scenario = _make_two_segment_scenario(
        use_hermite=True,
        penalty_exponent=1.5,
        n_elems=3,
        helical_z=True,
    )

    ndpn = scenario["ndof_per_node"]
    n_nodes = scenario["n_nodes"]

    # 全並進 DOF
    trans_dofs = np.array(
        [node * ndpn + d for node in range(n_nodes) for d in range(3)],
        dtype=int,
    )

    # アクティブペア: elem 1 (nodes 1,2) vs elem 4 (nodes 5,6)
    # → アクティブノード {1, 2, 5, 6}
    active_nodes = np.array([1, 2, 5, 6])
    # 隣接ノード: A-1=0, A+2=3, B-1=4, B+2=7
    adj_nodes = np.array([0, 3, 4, 7])

    active_dofs = np.array([n * ndpn + d for n in active_nodes for d in range(3)], dtype=int)
    adj_dofs = np.array([n * ndpn + d for n in adj_nodes for d in range(3)], dtype=int)

    print(f"active_nodes={active_nodes}, active_dofs={active_dofs}")
    print(f"adj_nodes={adj_nodes},    adj_dofs={adj_dofs}")

    # ── FD K_c ──
    fd_kc = _compute_fd_kc(scenario, eps=1e-7, dof_indices=trans_dofs)

    # ── 解析的 K_c, K_mat, K_geo, K_st ──
    K_mat, K_geo, K_st = _get_analytical_components(scenario)
    K_c_analytical = K_mat - K_geo + K_st

    # dense 差分
    diff = (K_c_analytical - fd_kc).toarray()
    fd_dense = fd_kc.toarray()
    kc_dense = K_c_analytical.toarray()

    total_diff_norm = np.linalg.norm(diff)
    total_fd_norm = np.linalg.norm(fd_dense)
    total_kc_norm = np.linalg.norm(kc_dense)
    print("\n── 全体 ──")
    print(f"  ||K_c||       = {total_kc_norm:.4e}")
    print(f"  ||FD_Kc||     = {total_fd_norm:.4e}")
    print(f"  ||diff||      = {total_diff_norm:.4e}")
    print(f"  rel_err       = {total_diff_norm / max(total_fd_norm, 1e-30):.4e}")

    # ── active × active ブロック ──
    diff_aa = diff[np.ix_(active_dofs, active_dofs)]
    fd_aa = fd_dense[np.ix_(active_dofs, active_dofs)]
    kc_aa = kc_dense[np.ix_(active_dofs, active_dofs)]
    n_aa_diff = np.linalg.norm(diff_aa)
    n_aa_fd = np.linalg.norm(fd_aa)
    n_aa_kc = np.linalg.norm(kc_aa)
    print("\n── active 行 × active 列 (12×12) ──")
    print(f"  ||K_c[aa]||   = {n_aa_kc:.4e}")
    print(f"  ||FD[aa]||    = {n_aa_fd:.4e}")
    print(f"  ||diff[aa]||  = {n_aa_diff:.4e}")
    print(f"  rel_err[aa]   = {n_aa_diff / max(n_aa_fd, 1e-30):.4e}")

    # ── active 行 × 隣接 列 ブロック ──
    diff_ax = diff[np.ix_(active_dofs, adj_dofs)]
    fd_ax = fd_dense[np.ix_(active_dofs, adj_dofs)]
    kc_ax = kc_dense[np.ix_(active_dofs, adj_dofs)]
    n_ax_diff = np.linalg.norm(diff_ax)
    n_ax_fd = np.linalg.norm(fd_ax)
    n_ax_kc = np.linalg.norm(kc_ax)
    print("\n── active 行 × adj 列 (12×12) ──")
    print(f"  ||K_c[ax]||   = {n_ax_kc:.4e}  (現行 K_closest は 0)")
    print(f"  ||FD[ax]||    = {n_ax_fd:.4e}  ← 仮説 B で埋めたい量")
    print(f"  ||diff[ax]||  = {n_ax_diff:.4e}")

    # ── adj 行 × active 列 ブロック ──
    diff_xa = diff[np.ix_(adj_dofs, active_dofs)]
    fd_xa = fd_dense[np.ix_(adj_dofs, active_dofs)]
    kc_xa = kc_dense[np.ix_(adj_dofs, active_dofs)]
    n_xa_diff = np.linalg.norm(diff_xa)
    n_xa_fd = np.linalg.norm(fd_xa)
    n_xa_kc = np.linalg.norm(kc_xa)
    print("\n── adj 行 × active 列 (12×12) ──")
    print(f"  ||K_c[xa]||   = {n_xa_kc:.4e}  (K_hermite_adj が一部埋める)")
    print(f"  ||FD[xa]||    = {n_xa_fd:.4e}")
    print(f"  ||diff[xa]||  = {n_xa_diff:.4e}")

    # ── adj 行 × adj 列 ブロック ──
    diff_xx = diff[np.ix_(adj_dofs, adj_dofs)]
    fd_xx = fd_dense[np.ix_(adj_dofs, adj_dofs)]
    kc_xx = kc_dense[np.ix_(adj_dofs, adj_dofs)]
    n_xx_diff = np.linalg.norm(diff_xx)
    n_xx_fd = np.linalg.norm(fd_xx)
    n_xx_kc = np.linalg.norm(kc_xx)
    print("\n── adj 行 × adj 列 (12×12) ──")
    print(f"  ||K_c[xx]||   = {n_xx_kc:.4e}")
    print(f"  ||FD[xx]||    = {n_xx_fd:.4e}")
    print(f"  ||diff[xx]||  = {n_xx_diff:.4e}")

    # ── シェア集計 ──
    print(f"\n── diff シェア集計（全体ノルム = {total_diff_norm:.4e}）──")
    for name, val in [
        ("aa (active×active)", n_aa_diff),
        ("ax (active×adj)  ", n_ax_diff),
        ("xa (adj×active)  ", n_xa_diff),
        ("xx (adj×adj)     ", n_xx_diff),
    ]:
        pct = val**2 / max(total_diff_norm**2, 1e-30) * 100
        print(f"  {name}: ||diff|| = {val:.4e} ({pct:5.1f}%)")

    # ── comp 別シェア（active 列 vs adj 列） ──
    comp_names = ["x", "y", "z"]
    print("\n── active 列 (行次元全体) comp 別 diff ──")
    for c in range(3):
        rows = np.array([n * ndpn + c for n in range(n_nodes)])
        d = diff[np.ix_(rows, active_dofs)]
        fdn = np.linalg.norm(fd_dense[np.ix_(rows, active_dofs)])
        print(f"  {comp_names[c]}: ||diff||={np.linalg.norm(d):.4e}  ||FD||={fdn:.4e}")

    print("\n── adj 列 (行次元全体) comp 別 diff ──")
    for c in range(3):
        rows = np.array([n * ndpn + c for n in range(n_nodes)])
        d = diff[np.ix_(rows, adj_dofs)]
        fdn = np.linalg.norm(fd_dense[np.ix_(rows, adj_dofs)])
        print(f"  {comp_names[c]}: ||diff||={np.linalg.norm(d):.4e}  ||FD||={fdn:.4e}")


if __name__ == "__main__":
    main()
