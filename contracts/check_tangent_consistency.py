"""接線剛性の有限差分整合性検証.

NR力収束が0件（全disp収束）である根本原因を診断。
K_c (解析的接線) vs FD (有限差分) を比較。

Usage:
    python contracts/check_tangent_consistency.py 2>&1 | tee /tmp/log-$(date +%s).log
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from xkep_cae.contact.contact_force.strategy import (
    ContactForceInput,
    HuberContactForceProcess,
)
from xkep_cae.contact._contact_pair import (
    _ContactConfigInput,
    _ContactPairOutput,
    _ContactStateOutput,
)


def _compute_geometry(node_coords, r_a, r_b, *, use_hermite=False):
    """2梁の最近接点と接触幾何を計算.

    use_hermite=True の場合、線形最近接点を初期値として
    Hermite 曲線上の Newton 精緻化を行う（実ソルバーと同一手順）。
    """
    xA0, xA1 = node_coords[0], node_coords[1]
    xB0, xB1 = node_coords[2], node_coords[3]
    dA = xA1 - xA0
    dB = xB1 - xB0
    a = float(np.dot(dA, dA))
    b = float(np.dot(dA, dB))
    c = float(np.dot(dB, dB))
    w0 = xA0 - xB0
    d = float(np.dot(dA, w0))
    e = float(np.dot(dB, w0))
    denom = a * c - b * b
    if abs(denom) > 1e-30:
        s = float(np.clip((b * e - c * d) / denom, 0.01, 0.99))
        t = float(np.clip((a * e - b * d) / denom, 0.01, 0.99))
    else:
        s, t = 0.5, 0.5

    if use_hermite:
        # Hermite 精緻化（実ソルバーと同一手順: status-242）
        from xkep_cae.contact.geometry._compute import (
            _closest_point_hermite_refine,
            _compute_node_tangents,
        )

        conn = np.array([[0, 1], [2, 3]])
        tangents = _compute_node_tangents(node_coords, conn)
        s_arr, t_arr, pA_arr, pB_arr, dist_arr, normal_arr = _closest_point_hermite_refine(
            np.array([s]),
            np.array([t]),
            node_coords[0:1],
            node_coords[1:2],
            node_coords[2:3],
            node_coords[3:4],
            tangents[0:1],
            tangents[1:2],
            tangents[2:3],
            tangents[3:4],
        )
        s = float(s_arr[0])
        t = float(t_arr[0])
        diff = pA_arr[0] - pB_arr[0]
        dist = float(dist_arr[0])
        normal = normal_arr[0]
        gap = dist - r_a - r_b
        return s, t, gap, normal, dist

    pA = xA0 + s * dA
    pB = xB0 + t * dB
    diff = pA - pB
    dist = float(np.linalg.norm(diff))
    normal = diff / dist if dist > 1e-15 else np.array([0.0, 0.0, 1.0])
    gap = dist - r_a - r_b
    return s, t, gap, normal, dist


def _compute_huber_pn(gap, k_pen, delta):
    """Huber C1 ペナルティ力（コードの _huber と同一実装）."""
    x = k_pen * (-gap)  # x_pen
    delta_h = k_pen / delta if delta > 0 else 0.0
    if delta_h <= 0.0:
        return max(0.0, x)
    if x < -delta_h:
        return 0.0
    if x > delta_h:
        return x
    return (x + delta_h) ** 2 / (4.0 * delta_h)


def _make_manager(u, coords_ref, r_a, r_b, k_pen, delta, config):
    """変位 u から ContactManager 相当のオブジェクトを作る."""
    ndof_per_node = 6
    n_nodes = 4
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords[i] = coords_ref[i] + u[i * ndof_per_node : i * ndof_per_node + 3]

    _use_hermite = hasattr(config, "use_hermite_centerline") and config.use_hermite_centerline
    s, t, gap, normal, dist = _compute_geometry(node_coords, r_a, r_b, use_hermite=_use_hermite)
    p_n = _compute_huber_pn(gap, k_pen, delta)

    state = _ContactStateOutput(
        s=s,
        t=t,
        gap=gap,
        normal=normal,
        p_n=p_n,
    )
    pair = _ContactPairOutput(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array([0, 1]),
        nodes_b=np.array([2, 3]),
        state=state,
        radius_a=r_a,
        radius_b=r_b,
    )

    class _Mgr:
        def __init__(self):
            self.pairs = [pair]
            self.config = config
            self.connectivity = np.array([[0, 1], [2, 3]])

    return _Mgr(), node_coords


def check_tangent_consistency():
    """接触接線剛性を有限差分で検証する."""
    print("=" * 60)
    print("接線剛性有限差分整合性検証")
    print("=" * 60)
    print()

    ndof_per_node = 6
    n_nodes = 4
    ndof = n_nodes * ndof_per_node
    r_a = 0.5
    r_b = 0.5
    k_pen = 1000.0
    delta = 0.02

    # 初期座標: A=(0,0,0)-(10,0,0), B=(5,-5,0)-(5,5,0) 直交交差
    coords_ref = np.zeros((n_nodes, 3))
    coords_ref[0] = [0.0, 0.0, 0.0]
    coords_ref[1] = [10.0, 0.0, 0.0]
    coords_ref[2] = [5.0, -5.0, 0.0]
    coords_ref[3] = [5.0, 5.0, 0.0]

    # 変位: Bをz方向に押し込み（貫入）
    u = np.zeros(ndof)
    u[14] = -0.8  # Node 2 z
    u[20] = -0.8  # Node 3 z

    # 基本情報
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords[i] = coords_ref[i] + u[i * ndof_per_node : i * ndof_per_node + 3]
    s, t, gap, normal, _ = _compute_geometry(node_coords, r_a, r_b)
    print(f"  s={s:.4f}, t={t:.4f}, gap={gap:.6f}, normal={normal}")
    print()

    # --- テスト 1: K_st なし ---
    print("--- テスト 1: K_st なし ---")
    _run_fd_test(
        coords_ref,
        u,
        r_a,
        r_b,
        k_pen,
        delta,
        ndof,
        ndof_per_node,
        consistent_st=False,
        use_hermite=False,
    )

    # --- テスト 2: K_st あり ---
    print("\n--- テスト 2: K_st あり ---")
    _run_fd_test(
        coords_ref,
        u,
        r_a,
        r_b,
        k_pen,
        delta,
        ndof,
        ndof_per_node,
        consistent_st=True,
        use_hermite=False,
    )

    # --- テスト 3: Hermite + K_st ---
    print("\n--- テスト 3: Hermite + K_st ---")
    _run_fd_test(
        coords_ref,
        u,
        r_a,
        r_b,
        k_pen,
        delta,
        ndof,
        ndof_per_node,
        consistent_st=True,
        use_hermite=True,
    )

    print()


def _run_fd_test(
    coords_ref, u, r_a, r_b, k_pen, delta, ndof, ndof_per_node, *, consistent_st, use_hermite
):
    """1構成での FD 検証."""
    config = _ContactConfigInput(
        g_off=2.0,
        mu=0.1,
        use_hermite_centerline=use_hermite,
        consistent_st_tangent=consistent_st,
        freeze_geometry_in_nr=False,
    )

    huber = HuberContactForceProcess(ndof=ndof, smoothing_delta=delta)

    # 解析的接線剛性
    mgr, nc = _make_manager(u, coords_ref, r_a, r_b, k_pen, delta, config)
    K_c = huber.tangent(u, mgr, k_pen, node_coords=nc)

    # FD 接線剛性
    eps = 1e-6
    K_c_fd = np.zeros((ndof, ndof))

    for j in range(ndof):
        # +ε
        u_p = u.copy()
        u_p[j] += eps
        mgr_p, _ = _make_manager(u_p, coords_ref, r_a, r_b, k_pen, delta, config)
        f_p = huber.process(ContactForceInput(u=u_p, manager=mgr_p, k_pen=k_pen)).contact_force

        # -ε
        u_m = u.copy()
        u_m[j] -= eps
        mgr_m, _ = _make_manager(u_m, coords_ref, r_a, r_b, k_pen, delta, config)
        f_m = huber.process(ContactForceInput(u=u_m, manager=mgr_m, k_pen=k_pen)).contact_force

        # K_c[:,j] = -d(f_c_raw)/du_j
        # f_c_raw は HuberContactForceProcess.process() の出力
        # 残差: R = f_int + (-f_c_raw) - f_ext → dR/du_contact = -df_c_raw/du
        K_c_fd[:, j] = -(f_p - f_m) / (2 * eps)

    # 比較
    K_c_dense = K_c.toarray()
    diff_mat = K_c_dense - K_c_fd
    max_err = float(np.max(np.abs(diff_mat)))
    max_K = max(float(np.max(np.abs(K_c_dense))), float(np.max(np.abs(K_c_fd))))
    rel_err = max_err / max_K if max_K > 1e-30 else 0.0

    print(f"  |K_c|_max={np.max(np.abs(K_c_dense)):.4f}, |K_fd|_max={np.max(np.abs(K_c_fd)):.4f}")
    print(f"  最大絶対誤差={max_err:.4e}, 最大相対誤差={rel_err:.4e}")

    # DOF 別
    col_err = np.max(np.abs(diff_mat), axis=0)
    col_K = np.max(np.abs(K_c_dense), axis=0) + np.max(np.abs(K_c_fd), axis=0)
    col_rel = col_err / np.maximum(col_K, 1e-30)
    top5 = np.argsort(col_rel)[::-1][:5]
    for idx in top5:
        node = idx // ndof_per_node
        dof_name = ["x", "y", "z", "rx", "ry", "rz"][idx % ndof_per_node]
        if col_K[idx] > 1e-10:
            print(
                f"    DOF {idx} (N{node}.{dof_name}): abs={col_err[idx]:.3e} rel={col_rel[idx]:.3e}"
            )

    threshold = 0.05
    if rel_err < threshold:
        print(f"  → 整合 (rel_err={rel_err:.2%} < {threshold:.0%})")
    else:
        print(f"  → 不整合 (rel_err={rel_err:.2%} > {threshold:.0%})")


if __name__ == "__main__":
    check_tangent_consistency()
