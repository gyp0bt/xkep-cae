"""ContactForce Strategy 具象実装.

ContactForceStrategy Protocol に従い、接触力を評価する Process。

status-222 で完全一本化:
- HuberContactForceProcess: Huber ペナルティ接触力（唯一の実装）
- SmoothPenalty / NCP / Uzawa は status-222 で削除。復元手順は status-222.md 参照。

status-230: Hermite 幾何対応
- 形状関数係数を Hermite 基底 H00(s)/H01(s) に切替
- ∂n/∂s を Hermite 接線 dpA/ds で計算
- K_st に Hermite 版 StJacobian を使用

status-256 B1: ContactForceStStiffnessProcess
- 接触力 K_st（接触点滑り剛性）を Process 化
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state
from xkep_cae.contact.geometry._st_jacobian import ComputeStJacobianProcess
from xkep_cae.core import ProcessMeta, SolverProcess

# ── Input / Output ─────────────────────────────────────────


@dataclass(frozen=True)
class ContactForceInput:
    """ContactForce Strategy の入力."""

    u: np.ndarray
    manager: object
    k_pen: float


@dataclass(frozen=True)
class ContactForceOutput:
    """ContactForce Strategy の出力."""

    contact_force: np.ndarray


# ── Hermite 形状関数ヘルパー ──────────────────────────────────


def _hermite_shape_coeffs(s: float, t: float) -> list[float]:
    """Hermite 位置基底による形状関数係数.

    H00(s) = 2s³ - 3s² + 1  (始点)
    H01(s) = -2s³ + 3s²     (終点)

    Returns:
        [H00(s), H01(s), -H00(t), -H01(t)]
    """
    s2, s3 = s * s, s * s * s
    t2, t3 = t * t, t * t * t
    return [
        2.0 * s3 - 3.0 * s2 + 1.0,  # A0: H00(s)
        -2.0 * s3 + 3.0 * s2,  # A1: H01(s)
        -(2.0 * t3 - 3.0 * t2 + 1.0),  # B0: -H00(t)
        -(-2.0 * t3 + 3.0 * t2),  # B1: -H01(t)
    ]


def _hermite_dc_ds(s: float) -> list[float]:
    """Hermite 形状関数の s 微分: d[H00(s), H01(s), -, -]/ds.

    H00'(s) = 6s² - 6s
    H01'(s) = -6s² + 6s
    """
    s2 = s * s
    return [6.0 * s2 - 6.0 * s, -6.0 * s2 + 6.0 * s, 0.0, 0.0]


def _hermite_dc_dt(t: float) -> list[float]:
    """Hermite 形状関数の t 微分: d[-, -, -H00(t), -H01(t)]/dt.

    d(-H00(t))/dt = -H00'(t) = -(6t² - 6t)
    d(-H01(t))/dt = -H01'(t) = -(-6t² + 6t)
    """
    t2 = t * t
    return [0.0, 0.0, -(6.0 * t2 - 6.0 * t), -(-6.0 * t2 + 6.0 * t)]


def _hermite_corrected_coeffs(
    s: float,
    t: float,
    dm_A: np.ndarray,
    dm_B: np.ndarray,
) -> tuple[list[float], list[float], list[float]]:
    """∂m/∂u 補正付き Hermite 形状関数係数と微分（status-243）.

    位置感度:
        coeff[Ak] = H_Ak(s) + H10(s)·dm_A[0,k] + H11(s)·dm_A[1,k]

    微分:
        dc_ds[Ak] = H_Ak'(s) + H10'(s)·dm_A[0,k] + H11'(s)·dm_A[1,k]

    Returns:
        (coeffs, dc_ds, dc_dt): 各 4要素のリスト
    """
    s2, s3 = s * s, s * s * s
    t2, t3 = t * t, t * t * t

    # Hermite 基底関数
    h00_s = 2.0 * s3 - 3.0 * s2 + 1.0
    h01_s = -2.0 * s3 + 3.0 * s2
    h10_s = s3 - 2.0 * s2 + s
    h11_s = s3 - s2
    h00_t = 2.0 * t3 - 3.0 * t2 + 1.0
    h01_t = -2.0 * t3 + 3.0 * t2
    h10_t = t3 - 2.0 * t2 + t
    h11_t = t3 - t2

    # 微分
    dh00_s = 6.0 * s2 - 6.0 * s
    dh01_s = -6.0 * s2 + 6.0 * s
    dh10_s = 3.0 * s2 - 4.0 * s + 1.0
    dh11_s = 3.0 * s2 - 2.0 * s
    dh00_t = 6.0 * t2 - 6.0 * t
    dh01_t = -6.0 * t2 + 6.0 * t
    dh10_t = 3.0 * t2 - 4.0 * t + 1.0
    dh11_t = 3.0 * t2 - 2.0 * t

    # 補正付き係数
    coeffs = [
        h00_s + h10_s * dm_A[0, 0] + h11_s * dm_A[1, 0],  # A0
        h01_s + h10_s * dm_A[0, 1] + h11_s * dm_A[1, 1],  # A1
        -(h00_t + h10_t * dm_B[0, 0] + h11_t * dm_B[1, 0]),  # B0
        -(h01_t + h10_t * dm_B[0, 1] + h11_t * dm_B[1, 1]),  # B1
    ]
    dc_ds = [
        dh00_s + dh10_s * dm_A[0, 0] + dh11_s * dm_A[1, 0],  # A0
        dh01_s + dh10_s * dm_A[0, 1] + dh11_s * dm_A[1, 1],  # A1
        0.0,
        0.0,
    ]
    dc_dt = [
        0.0,
        0.0,
        -(dh00_t + dh10_t * dm_B[0, 0] + dh11_t * dm_B[1, 0]),  # B0
        -(dh01_t + dh10_t * dm_B[0, 1] + dh11_t * dm_B[1, 1]),  # B1
    ]
    return coeffs, dc_ds, dc_dt


# ── ヘルパー ───────────────────────────────────────────────


def _contact_shape_vector(
    pair: object,
    *,
    use_hermite: bool = False,
    dm_A: np.ndarray | None = None,
    dm_B: np.ndarray | None = None,
) -> np.ndarray:
    """接触形状ベクトル g_shape (12,) を構築する.

    線形: g_shape = [(1-s)*n, s*n, -(1-t)*n, -t*n]
    Hermite: g_shape = [H00(s)*n, H01(s)*n, -H00(t)*n, -H01(t)*n]
    Hermite+dm: ∂m/∂u 補正付き（status-243）

    Args:
        pair: ContactPair（state.s, state.t, state.normal を持つ）
        use_hermite: True なら Hermite 基底を使用
        dm_A: (2,2) A側 ∂m/∂x 係数（None なら凍結近似）
        dm_B: (2,2) B側 ∂m/∂x 係数

    Returns:
        g_shape: (12,) 形状ベクトル
    """
    s = pair.state.s
    t = pair.state.t
    normal = pair.state.normal
    if use_hermite:
        if dm_A is not None and dm_B is not None:
            coeffs, _, _ = _hermite_corrected_coeffs(s, t, dm_A, dm_B)
        else:
            coeffs = _hermite_shape_coeffs(s, t)
    else:
        coeffs = [(1.0 - s), s, -(1.0 - t), -t]
    g_shape = np.zeros(12)
    for k in range(4):
        g_shape[k * 3 : k * 3 + 3] = coeffs[k] * normal
    return g_shape


def _huber_scalar(x: float, delta: float) -> float:
    """Huber 関数（モジュールレベル版）: max(0,x) の C1 近似."""
    if delta <= 0.0:
        return max(0.0, x)
    if x < -delta:
        return 0.0
    if x > delta:
        return x
    return (x + delta) ** 2 / (4.0 * delta)


def _huber_deriv_scalar(x: float, delta: float) -> float:
    """Huber 導関数（モジュールレベル版）: C0 連続."""
    if delta <= 0.0:
        return 1.0 if x > 0.0 else 0.0
    if x < -delta:
        return 0.0
    if x > delta:
        return 1.0
    return (x + delta) / (2.0 * delta)


# ── B1: _add_kst_contact_to_coo（モジュールレベルヘルパー） ──


def _add_kst_contact_to_coo(
    pair: object,
    p_n: float,
    normal: np.ndarray,
    dofs: np.ndarray,
    st_proc: object,
    StJacobianInput: type,
    rows: list[int],
    cols: list[int],
    vals: list[float],
    node_coords: np.ndarray,
    ndof_per_node: int,
    *,
    use_hermite: bool = False,
    node_tangents: np.ndarray | None = None,
    node_counts: np.ndarray | None = None,
    h_deriv: float = 0.0,
    k_pen: float = 0.0,
    adj_node_map: dict | None = None,
) -> None:
    """接触力の K_st（接触点滑り剛性）を COO に追加.

    f_c_raw = p_n(s,t) * Σ_k c_k(s,t) * n(s,t) の s,t 依存の完全微分。

    ∂f_raw/∂s = (∂p_n/∂s) * Σ c_k * n                  ← status-242 追加
              + p_n * (∂c_k/∂s · n + c_k · ∂n/∂s)       ← 既存

    ∂p_n/∂s = h'(x) * k_pen * (-∂gap/∂s)
    ∂gap/∂s = (delta · dpA/ds) / dist

    線形: ∂n/∂s = (1/dist)(I - n⊗n) · dA
    Hermite: ∂n/∂s = (1/dist)(I - n⊗n) · dpA/ds（status-230）
    """
    st = pair.state
    xA0 = node_coords[pair.nodes_a[0]]
    xA1 = node_coords[pair.nodes_a[1]]
    xB0 = node_coords[pair.nodes_b[0]]
    xB1 = node_coords[pair.nodes_b[1]]

    # dm 係数の計算（frozen-m 解消: status-243）
    _dm_A = None
    _dm_B = None
    _dm_ext_A = None
    _dm_ext_B = None
    if use_hermite and node_counts is not None:
        from xkep_cae.contact.geometry._compute import (
            _compute_dm_coeffs,
        )

        _dm_A = _compute_dm_coeffs(
            node_counts[pair.nodes_a[0]],
            node_counts[pair.nodes_a[1]],
        )
        _dm_B = _compute_dm_coeffs(
            node_counts[pair.nodes_b[0]],
            node_counts[pair.nodes_b[1]],
        )
        # 非局所 dm 係数（status-272: K_st拡張）
        # status-296: K_st_adj再有効化を検証した結果、x,y方向で27倍過大（38.5%に悪化）。
        # K_c_adj geo(I-n⊗n)とK_st_adj接平面成分が物理的に同一寄与であり、
        # どちらを有効にしても二重計上が発生。K_c_adj mat-only(1.8%)が最適解。
        # dm_ext はStJacobianに渡さない（K_st_adjは無効のまま維持）。

    # StJacobian 入力の構築
    st_kw: dict = {
        "xA0": xA0,
        "xA1": xA1,
        "xB0": xB0,
        "xB1": xB1,
        "s": st.s,
        "t": st.t,
    }
    # s_unclamped/t_unclamped を渡す（status-291: smooth_clip_deriv 重み補正）
    if hasattr(st, "s_unclamped") and st.s_unclamped is not None:
        st_kw["s_unclamped"] = st.s_unclamped
    if hasattr(st, "t_unclamped") and st.t_unclamped is not None:
        st_kw["t_unclamped"] = st.t_unclamped
    if use_hermite and node_tangents is not None:
        st_kw["mA0"] = node_tangents[pair.nodes_a[0]]
        st_kw["mA1"] = node_tangents[pair.nodes_a[1]]
        st_kw["mB0"] = node_tangents[pair.nodes_b[0]]
        st_kw["mB1"] = node_tangents[pair.nodes_b[1]]
        st_kw["use_hermite"] = True
        if _dm_A is not None:
            st_kw["dm_A"] = _dm_A
            st_kw["dm_B"] = _dm_B
        if _dm_ext_A is not None:
            st_kw["dm_ext_A"] = _dm_ext_A
            st_kw["dm_ext_B"] = _dm_ext_B

    out = st_proc.process(StJacobianInput(**st_kw))
    if not out.valid:
        return

    s = st.s
    t = st.t
    dist = st.gap + pair.radius_a + pair.radius_b
    if dist < 1e-15:
        return

    # 形状関数係数とその微分
    if use_hermite:
        if _dm_A is not None:
            coeffs, dc_ds, dc_dt = _hermite_corrected_coeffs(s, t, _dm_A, _dm_B)
        else:
            coeffs = _hermite_shape_coeffs(s, t)
            dc_ds = _hermite_dc_ds(s)
            dc_dt = _hermite_dc_dt(t)
    else:
        coeffs = [(1.0 - s), s, -(1.0 - t), -t]
        dc_ds = [-1.0, 1.0, 0.0, 0.0]
        dc_dt = [0.0, 0.0, 1.0, -1.0]

    # ∂n/∂s, ∂n/∂t
    P_perp = np.eye(3) - np.outer(normal, normal)
    if use_hermite and node_tangents is not None:
        from xkep_cae.contact.geometry._st_jacobian import _hermite_deriv_scalar

        dpA = _hermite_deriv_scalar(
            s, xA0, xA1, node_tangents[pair.nodes_a[0]], node_tangents[pair.nodes_a[1]]
        )
        dpB = _hermite_deriv_scalar(
            t, xB0, xB1, node_tangents[pair.nodes_b[0]], node_tangents[pair.nodes_b[1]]
        )
        dn_ds = (1.0 / dist) * P_perp @ dpA
        dn_dt = -(1.0 / dist) * P_perp @ dpB
    else:
        dA = xA1 - xA0
        dB = xB1 - xB0
        dn_ds = (1.0 / dist) * P_perp @ dA
        dn_dt = -(1.0 / dist) * P_perp @ dB

    # ∂p_n/∂s, ∂p_n/∂t（ペナルティ力の滑り微分、status-242）
    dpn_ds = 0.0
    dpn_dt = 0.0
    if h_deriv > 1e-30 and k_pen > 0.0:
        if use_hermite and node_tangents is not None:
            dgap_ds = float(np.dot(normal, dpA))
            dgap_dt = -float(np.dot(normal, dpB))
        else:
            dA = xA1 - xA0
            dB = xB1 - xB0
            dgap_ds = float(np.dot(normal, dA))
            dgap_dt = -float(np.dot(normal, dB))
        dpn_ds = h_deriv * k_pen * (-dgap_ds)
        dpn_dt = h_deriv * k_pen * (-dgap_dt)

    # g_shape = Σ c_k * n (12,)
    g_shape = np.zeros(12)
    for k in range(4):
        for i in range(3):
            g_shape[k * 3 + i] = coeffs[k] * normal[i]

    # ∂f_raw/∂s (12,): (∂p_n/∂s) * g_shape + p_n * (dc_k/ds·n + c_k·∂n/∂s)
    df_ds = np.zeros(12)
    df_dt = np.zeros(12)
    for k in range(4):
        for i in range(3):
            li = k * 3 + i
            df_ds[li] = dpn_ds * g_shape[li] + p_n * (dc_ds[k] * normal[i] + coeffs[k] * dn_ds[i])
            df_dt[li] = dpn_dt * g_shape[li] + p_n * (dc_dt[k] * normal[i] + coeffs[k] * dn_dt[i])

    # K_st = -(df_ds ⊗ ds_du + df_dt ⊗ dt_du)
    K_st_local = -(np.outer(df_ds, out.ds_du) + np.outer(df_dt, out.dt_du))

    for ki in range(4):
        for di in range(3):
            li = ki * 3 + di
            gi = dofs[ki * ndof_per_node + di]
            for kj in range(4):
                for dj in range(3):
                    lj = kj * 3 + dj
                    gj = dofs[kj * ndof_per_node + dj]
                    val = K_st_local[li, lj]
                    if abs(val) > 1e-30:
                        rows.append(gi)
                        cols.append(gj)
                        vals.append(val)

    # K_st 隣接ノードDOF拡張（status-272: Hermite非局所∂g/∂u Step2）
    if out.ds_du_adj is not None and adj_node_map is not None:
        # 隣接ノードのグローバルインデックス取得
        # ds_du_adj レイアウト: [A-1_xyz(3), A+2_xyz(3), B-1_xyz(3), B+2_xyz(3)]
        adj_a = adj_node_map.get(pair.elem_a, (-1, -1))
        adj_b = adj_node_map.get(pair.elem_b, (-1, -1))
        adj_global_nodes = [adj_a[0], adj_a[1], adj_b[0], adj_b[1]]

        K_st_adj = -(np.outer(df_ds, out.ds_du_adj) + np.outer(df_dt, out.dt_du_adj))

        for ki in range(4):
            for di in range(3):
                li = ki * 3 + di
                gi = dofs[ki * ndof_per_node + di]
                for adj_idx in range(4):
                    adj_node = adj_global_nodes[adj_idx]
                    if adj_node < 0:
                        continue
                    for dj in range(3):
                        adj_lj = adj_idx * 3 + dj
                        gj = adj_node * ndof_per_node + dj
                        val = K_st_adj[li, adj_lj]
                        if abs(val) > 1e-30:
                            rows.append(gi)
                            cols.append(gj)
                            vals.append(val)


# ── B1: ContactForceStStiffnessProcess ─────────────────────


@dataclass(frozen=True)
class ContactForceStStiffnessInput:
    """接触力 K_st（接触点滑り剛性）の入力."""

    pairs: list
    node_coords: np.ndarray
    k_pen: float
    delta_h: float
    ndof_total: int
    ndof_per_node: int = 6
    use_hermite: bool = False
    node_tangents: np.ndarray | None = None
    node_counts: np.ndarray | None = None
    adj_node_map: dict | None = None  # status-272: 隣接ノードマップ
    penalty_exponent: float = 1.0  # status-285: Hertz型非線形ペナルティ指数


@dataclass(frozen=True)
class ContactForceStStiffnessOutput:
    """接触力 K_st（接触点滑り剛性）の出力."""

    K_st: sp.csr_matrix


class ContactForceStStiffnessProcess(
    SolverProcess[ContactForceStStiffnessInput, ContactForceStStiffnessOutput],
):
    """接触力の K_st（接触点滑り剛性）を計算する Process.

    status-256 B1: _add_kst_contact を Process 化。
    f_c_raw = p_n(s,t) * Σ_k c_k(s,t) * n(s,t) の s,t 依存の完全微分。
    """

    meta = ProcessMeta(
        name="ContactForceStStiffness",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_force.md",
    )
    uses = [ComputeStJacobianProcess]

    def process(self, inp: ContactForceStStiffnessInput) -> ContactForceStStiffnessOutput:
        return self._process_batch(inp)

    def _process_batch(self, inp: ContactForceStStiffnessInput) -> ContactForceStStiffnessOutput:
        """K_st のバッチ計算（status-309: ベクトル化高速化）.

        全アクティブペアをNumPy配列に抽出し、StJacobian+K_stを一括計算。
        """
        zero = sp.csr_matrix((inp.ndof_total, inp.ndof_total))

        # ── ペアデータ抽出 ──
        n_pairs = len(inp.pairs)
        if n_pairs == 0:
            return ContactForceStStiffnessOutput(K_st=zero)

        has_state = np.zeros(n_pairs, dtype=bool)
        p_n_all = np.zeros(n_pairs)
        gaps = np.zeros(n_pairs)
        s_arr = np.zeros(n_pairs)
        t_arr = np.zeros(n_pairs)
        s_unc_arr = np.zeros(n_pairs)
        t_unc_arr = np.zeros(n_pairs)
        normals = np.zeros((n_pairs, 3))
        nodes = np.zeros((n_pairs, 4), dtype=int)
        radius_a = np.zeros(n_pairs)
        radius_b = np.zeros(n_pairs)

        for i, pair in enumerate(inp.pairs):
            if not hasattr(pair, "state"):
                continue
            has_state[i] = True
            p_n_all[i] = pair.state.p_n
            gaps[i] = pair.state.gap
            s_arr[i] = pair.state.s
            t_arr[i] = pair.state.t
            s_unc_arr[i] = (
                pair.state.s_unclamped
                if hasattr(pair.state, "s_unclamped") and pair.state.s_unclamped is not None
                else pair.state.s
            )
            t_unc_arr[i] = (
                pair.state.t_unclamped
                if hasattr(pair.state, "t_unclamped") and pair.state.t_unclamped is not None
                else pair.state.t
            )
            normals[i] = pair.state.normal
            nodes[i, 0] = pair.nodes_a[0]
            nodes[i, 1] = pair.nodes_a[1]
            nodes[i, 2] = pair.nodes_b[0]
            nodes[i, 3] = pair.nodes_b[1]
            radius_a[i] = pair.radius_a
            radius_b[i] = pair.radius_b

        # アクティブペア: has_state & p_n > 0
        active = has_state & (p_n_all > 1e-30)
        n_act = int(np.sum(active))
        if n_act == 0:
            return ContactForceStStiffnessOutput(K_st=zero)

        # アクティブペアのインデックスとデータ抽出
        act_idx = np.where(active)[0]
        p_n_act = p_n_all[act_idx]
        gaps_act = gaps[act_idx]
        s_act = s_arr[act_idx]
        t_act = t_arr[act_idx]
        s_unc_act = s_unc_arr[act_idx]
        t_unc_act = t_unc_arr[act_idx]
        n_act_v = normals[act_idx]
        nodes_act = nodes[act_idx]
        ra_act = radius_a[act_idx]
        rb_act = radius_b[act_idx]

        # h_deriv バッチ計算
        x_pen = inp.k_pen * (-gaps_act)
        h_deriv = HuberContactForceProcess._huber_deriv_batch(x_pen, inp.delta_h)
        if inp.penalty_exponent != 1.0:
            h_vals = HuberContactForceProcess._huber_batch(x_pen, inp.delta_h)
            pen = h_vals / max(inp.k_pen, 1e-30)
            safe_pen = np.maximum(pen, 0.0)
            h_deriv = np.where(
                safe_pen > 1e-30,
                inp.penalty_exponent * safe_pen ** (inp.penalty_exponent - 1.0) * h_deriv,
                0.0,
            )

        # ── 座標抽出 ──
        nc = inp.node_coords
        xA0 = nc[nodes_act[:, 0]]  # (N, 3)
        xA1 = nc[nodes_act[:, 1]]
        xB0 = nc[nodes_act[:, 2]]
        xB1 = nc[nodes_act[:, 3]]

        # ── バッチ StJacobian ──
        if inp.use_hermite and inp.node_tangents is not None:
            from xkep_cae.contact.geometry._st_jacobian import (
                _batch_st_jacobian_hermite,
            )

            mA0 = inp.node_tangents[nodes_act[:, 0]]
            mA1 = inp.node_tangents[nodes_act[:, 1]]
            mB0 = inp.node_tangents[nodes_act[:, 2]]
            mB1 = inp.node_tangents[nodes_act[:, 3]]
            dm_A_batch = None
            dm_B_batch = None
            if inp.node_counts is not None:
                dm_A_batch, dm_B_batch = HuberContactForceProcess._batch_dm_coeffs(
                    inp.node_counts, nodes_act
                )
            ds_du, dt_du, valid = _batch_st_jacobian_hermite(
                xA0,
                xA1,
                xB0,
                xB1,
                s_act,
                t_act,
                s_unc_act,
                t_unc_act,
                mA0,
                mA1,
                mB0,
                mB1,
                dm_A_batch,
                dm_B_batch,
            )
            # dpA, dpB for dn/ds, dn/dt
            from xkep_cae.contact.geometry._st_jacobian import _hermite_deriv_scalar

            dpA_arr = np.zeros((n_act, 3))
            dpB_arr = np.zeros((n_act, 3))
            for i in range(n_act):
                dpA_arr[i] = _hermite_deriv_scalar(float(s_act[i]), xA0[i], xA1[i], mA0[i], mA1[i])
                dpB_arr[i] = _hermite_deriv_scalar(float(t_act[i]), xB0[i], xB1[i], mB0[i], mB1[i])
        else:
            from xkep_cae.contact.geometry._st_jacobian import (
                _batch_st_jacobian_linear,
            )

            ds_du, dt_du, valid = _batch_st_jacobian_linear(
                xA0,
                xA1,
                xB0,
                xB1,
                s_act,
                t_act,
                s_unc_act,
                t_unc_act,
            )
            dpA_arr = xA1 - xA0  # dA (N, 3)
            dpB_arr = xB1 - xB0  # dB (N, 3)

        # invalid ペアを除外
        if not np.all(valid):
            keep = valid
            act_idx = act_idx[keep]
            p_n_act = p_n_act[keep]
            gaps_act = gaps_act[keep]
            s_act = s_act[keep]
            t_act = t_act[keep]
            n_act_v = n_act_v[keep]
            nodes_act = nodes_act[keep]
            ra_act = ra_act[keep]
            rb_act = rb_act[keep]
            h_deriv = h_deriv[keep]
            ds_du = ds_du[keep]
            dt_du = dt_du[keep]
            dpA_arr = dpA_arr[keep]
            dpB_arr = dpB_arr[keep]
            n_act = int(np.sum(keep))

        if n_act == 0:
            return ContactForceStStiffnessOutput(K_st=zero)

        # ── 形状関数係数・微分（バッチ） ──
        if inp.use_hermite and inp.node_counts is not None:
            dm_A_batch2, dm_B_batch2 = HuberContactForceProcess._batch_dm_coeffs(
                inp.node_counts, nodes_act
            )
            coeffs, dc_ds, dc_dt = HuberContactForceProcess._batch_hermite_corrected_coeffs(
                s_act, t_act, dm_A_batch2, dm_B_batch2
            )
        elif inp.use_hermite:
            coeffs = HuberContactForceProcess._batch_hermite_coeffs(s_act, t_act)
            s2 = s_act * s_act
            t2 = t_act * t_act
            dc_ds = np.column_stack(
                [
                    6.0 * s2 - 6.0 * s_act,
                    -6.0 * s2 + 6.0 * s_act,
                    np.zeros(n_act),
                    np.zeros(n_act),
                ]
            )
            dc_dt = np.column_stack(
                [
                    np.zeros(n_act),
                    np.zeros(n_act),
                    -(6.0 * t2 - 6.0 * t_act),
                    -(-6.0 * t2 + 6.0 * t_act),
                ]
            )
        else:
            coeffs = np.column_stack([1.0 - s_act, s_act, -(1.0 - t_act), -t_act])
            dc_ds = np.tile([-1.0, 1.0, 0.0, 0.0], (n_act, 1))
            dc_dt = np.tile([0.0, 0.0, 1.0, -1.0], (n_act, 1))

        # ── ∂n/∂s, ∂n/∂t のバッチ計算 ──
        dist = gaps_act + ra_act + rb_act  # (N,)
        safe_dist = np.where(dist > 1e-15, dist, 1.0)
        inv_dist = np.where(dist > 1e-15, 1.0 / safe_dist, 0.0)  # (N,)

        # P_perp = I - n⊗n: (N, 3, 3)
        I3 = np.eye(3)[None, :, :]
        nn = n_act_v[:, :, None] * n_act_v[:, None, :]
        P_perp = I3 - nn

        # dn/ds = (1/dist) * P_perp @ dpA: (N, 3)
        dn_ds = inv_dist[:, None] * np.einsum("nij,nj->ni", P_perp, dpA_arr)
        dn_dt = -inv_dist[:, None] * np.einsum("nij,nj->ni", P_perp, dpB_arr)

        # ── ∂p_n/∂s, ∂p_n/∂t のバッチ計算 ──
        # dgap/ds = dot(n, dpA), dgap/dt = -dot(n, dpB)
        dgap_ds = np.sum(n_act_v * dpA_arr, axis=1)  # (N,)
        dgap_dt = -np.sum(n_act_v * dpB_arr, axis=1)
        dpn_ds = h_deriv * inp.k_pen * (-dgap_ds)
        dpn_dt = h_deriv * inp.k_pen * (-dgap_dt)
        # h_deriv が小さい場合はゼロに
        dpn_ds = np.where(h_deriv > 1e-30, dpn_ds, 0.0)
        dpn_dt = np.where(h_deriv > 1e-30, dpn_dt, 0.0)

        # ── g_shape (N, 12) ──
        g_shape = np.zeros((n_act, 12))
        for k in range(4):
            g_shape[:, k * 3 : k * 3 + 3] = coeffs[:, k][:, None] * n_act_v

        # ── df_ds, df_dt (N, 12) ──
        # df_ds = dpn_ds * g_shape + p_n * (dc_ds[k]*n + coeffs[k]*dn_ds)
        df_ds = dpn_ds[:, None] * g_shape
        df_dt = dpn_dt[:, None] * g_shape
        for k in range(4):
            df_ds[:, k * 3 : k * 3 + 3] += p_n_act[:, None] * (
                dc_ds[:, k][:, None] * n_act_v + coeffs[:, k][:, None] * dn_ds
            )
            df_dt[:, k * 3 : k * 3 + 3] += p_n_act[:, None] * (
                dc_dt[:, k][:, None] * n_act_v + coeffs[:, k][:, None] * dn_dt
            )

        # ── K_st_local = -(outer(df_ds, ds_du) + outer(df_dt, dt_du)): (N, 12, 12) ──
        K_st_local = -(
            np.einsum("ni,nj->nij", df_ds, ds_du) + np.einsum("ni,nj->nij", df_dt, dt_du)
        )

        # ── DOF インデックス (N, 12) ──
        ndpn = inp.ndof_per_node
        gdofs = np.zeros((n_act, 12), dtype=int)
        for k in range(4):
            for d in range(3):
                gdofs[:, k * 3 + d] = nodes_act[:, k] * ndpn + d

        # ── COO 構築 ──
        row_idx = np.broadcast_to(gdofs[:, :, None], (n_act, 12, 12)).ravel()
        col_idx = np.broadcast_to(gdofs[:, None, :], (n_act, 12, 12)).ravel()
        val_arr = K_st_local.ravel()
        mask = np.abs(val_arr) > 1e-30
        rows_np = row_idx[mask]
        cols_np = col_idx[mask]
        vals_np = val_arr[mask]

        if len(vals_np) == 0:
            return ContactForceStStiffnessOutput(K_st=zero)
        return ContactForceStStiffnessOutput(
            K_st=sp.coo_matrix(
                (vals_np, (rows_np, cols_np)),
                shape=(inp.ndof_total, inp.ndof_total),
            ).tocsr()
        )


# ── 具象 Process ──────────────────────────────────────────


class HuberContactForceProcess(
    SolverProcess[ContactForceInput, ContactForceOutput],
):
    """Huber ペナルティ接触力評価.

    max(0, x) を Huber 関数で C1 連続化:
        huber(x, δ) =
            0              if x < -δ
            (x+δ)²/(4δ)   if -δ ≤ x ≤ δ
            x              if x > δ

    法線接触力: p_n = huber(k_pen * (-g), δ_huber)
    δ_huber = k_pen / smoothing_delta で自動計算。

    status-222 で NCP (λ/Uzawa) を除去した純粋ペナルティ法。
    """

    meta = ProcessMeta(
        name="HuberContactForce",
        module="solve",
        version="3.0.0",
        document_path="docs/contact_force.md",
    )
    uses = [ContactForceStStiffnessProcess]

    def __init__(
        self,
        ndof: int,
        ndof_per_node: int = 6,
        *,
        smoothing_delta: float = 0.0,
        huber_delta_h: float = 0.0,
        penalty_exponent: float = 1.0,
    ) -> None:
        self._ndof = ndof
        self._ndof_per_node = ndof_per_node
        self._smoothing_delta = smoothing_delta
        self._huber_delta_h = huber_delta_h  # >0: delta_h直接指定（status-261）
        self._delta_h_boost: float = 1.0  # チャタリング時ブースト倍率（status-268）
        # Hertz型非線形ペナルティ（status-285）
        # penalty_exponent=1.0: 線形ペナルティ（従来）
        # penalty_exponent=1.5: Hertz型（p_n ∝ δ^1.5）
        # 接触ON/OFF境界で力が緩やかに立ち上がり、活性集合の離散的切替を平滑化。
        self._penalty_exponent = penalty_exponent

    def set_delta_h_boost(self, factor: float) -> None:
        """Huber遷移幅のブースト倍率を設定（status-268: チャタリング対策）.

        NRソルバーからチャタリング検知時に呼ばれ、delta_hを一時的に拡大する。
        evaluate() と assemble_tangent() 両方に一貫して適用される。
        """
        self._delta_h_boost = max(1.0, factor)

    def _resolve_delta_h(self, k_pen: float) -> float:
        """Huber遷移幅を解決: huber_delta_h直接指定 > smoothing_delta間接指定 > 0."""
        base = 0.0
        if self._huber_delta_h > 0.0:
            base = self._huber_delta_h
        elif self._smoothing_delta > 0.0:
            base = k_pen / self._smoothing_delta
        return base * self._delta_h_boost

    @staticmethod
    def _huber(x: float, delta: float) -> float:
        """Huber 関数: max(0,x) の C1 近似."""
        if delta <= 0.0:
            return max(0.0, x)
        if x < -delta:
            return 0.0
        if x > delta:
            return x
        return (x + delta) ** 2 / (4.0 * delta)

    @staticmethod
    def _huber_deriv(x: float, delta: float) -> float:
        """Huber 導関数: C0 連続."""
        if delta <= 0.0:
            return 1.0 if x > 0.0 else 0.0
        if x < -delta:
            return 0.0
        if x > delta:
            return 1.0
        return (x + delta) / (2.0 * delta)

    @staticmethod
    def _huber_batch(x: np.ndarray, delta: float) -> np.ndarray:
        """ベクトル化 Huber 関数: max(0,x) の C1 近似."""
        if delta <= 0.0:
            return np.maximum(0.0, x)
        result = np.where(
            x < -delta,
            0.0,
            np.where(x > delta, x, (x + delta) ** 2 / (4.0 * delta)),
        )
        return result

    @staticmethod
    def _huber_deriv_batch(x: np.ndarray, delta: float) -> np.ndarray:
        """ベクトル化 Huber 導関数."""
        if delta <= 0.0:
            return np.where(x > 0.0, 1.0, 0.0)
        return np.where(
            x < -delta,
            0.0,
            np.where(x > delta, 1.0, (x + delta) / (2.0 * delta)),
        )

    def _apply_power_law(self, h_vals: np.ndarray, k_pen: float) -> np.ndarray:
        """Hertz型非線形ペナルティ: p_n = h^α / k_pen^{α-1} (status-285).

        h_vals = huber(k_pen * penetration, δ)  →  h/k_pen = penetration (smoothed)
        p_n = k_pen * (h/k_pen)^α = h^α / k_pen^{α-1}

        α=1.0 で線形（元の p_n = h）、α=1.5 でHertz型。
        """
        alpha = self._penalty_exponent
        safe_h = np.maximum(h_vals, 0.0)
        return safe_h**alpha / max(k_pen, 1e-30) ** (alpha - 1.0)

    def _apply_power_law_deriv(
        self, h_vals: np.ndarray, h_deriv: np.ndarray, k_pen: float
    ) -> np.ndarray:
        """Hertz型導関数補正: dp/dx = α * h^{α-1} / k_pen^{α-1} * h'(x) (status-285).

        dp/dg = dp/dx * dx/dg = dp/dx * (-k_pen)
        tangent計算では h_deriv に (-k_pen) が後段で掛かるため、
        h_deriv 自体を α * (h/k_pen)^{α-1} * h'(x) に置換する。
        """
        alpha = self._penalty_exponent
        safe_h = np.maximum(h_vals, 0.0)
        # h/k_pen = smoothed penetration
        pen = safe_h / max(k_pen, 1e-30)
        # (α-1) 乗: penetration=0 付近で 0^{0.5} = 0（Hertz の特徴）
        pen_pow = np.where(pen > 1e-30, pen ** (alpha - 1.0), 0.0)
        return alpha * pen_pow * h_deriv

    def _extract_pair_arrays(
        self,
        pairs: list,
    ) -> tuple[np.ndarray, ...] | None:
        """全ペアの数値データをバッチ配列に抽出.

        Returns:
            (has_state, gaps, s_arr, t_arr, normals, nodes_a0, nodes_a1,
             nodes_b0, nodes_b1, radius_a, radius_b) or None if no pairs.
        """
        n = len(pairs)
        if n == 0:
            return None
        has_state = np.zeros(n, dtype=bool)
        gaps = np.zeros(n)
        s_arr = np.zeros(n)
        t_arr = np.zeros(n)
        normals = np.zeros((n, 3))
        nodes = np.zeros((n, 4), dtype=int)  # A0, A1, B0, B1
        radius_a = np.zeros(n)
        radius_b = np.zeros(n)
        for i, pair in enumerate(pairs):
            if not hasattr(pair, "state"):
                continue
            has_state[i] = True
            gaps[i] = pair.state.gap
            s_arr[i] = pair.state.s
            t_arr[i] = pair.state.t
            normals[i] = pair.state.normal
            nodes[i, 0] = pair.nodes_a[0]
            nodes[i, 1] = pair.nodes_a[1]
            nodes[i, 2] = pair.nodes_b[0]
            nodes[i, 3] = pair.nodes_b[1]
            radius_a[i] = pair.radius_a
            radius_b[i] = pair.radius_b
        return has_state, gaps, s_arr, t_arr, normals, nodes, radius_a, radius_b

    @staticmethod
    def _batch_hermite_coeffs(
        s: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """バッチ Hermite 形状関数係数 (N, 4)."""
        s2, s3 = s * s, s * s * s
        t2, t3 = t * t, t * t * t
        return np.column_stack(
            [
                2.0 * s3 - 3.0 * s2 + 1.0,
                -2.0 * s3 + 3.0 * s2,
                -(2.0 * t3 - 3.0 * t2 + 1.0),
                -(-2.0 * t3 + 3.0 * t2),
            ]
        )

    @staticmethod
    def _batch_hermite_corrected_coeffs(
        s: np.ndarray,
        t: np.ndarray,
        dm_A: np.ndarray,
        dm_B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """バッチ ∂m/∂u 補正付き Hermite 係数 + 微分 (N,4) each.

        dm_A: (N, 2, 2), dm_B: (N, 2, 2)
        """
        s2, s3 = s * s, s * s * s
        t2, t3 = t * t, t * t * t
        h00_s = 2.0 * s3 - 3.0 * s2 + 1.0
        h01_s = -2.0 * s3 + 3.0 * s2
        h10_s = s3 - 2.0 * s2 + s
        h11_s = s3 - s2
        h00_t = 2.0 * t3 - 3.0 * t2 + 1.0
        h01_t = -2.0 * t3 + 3.0 * t2
        h10_t = t3 - 2.0 * t2 + t
        h11_t = t3 - t2
        dh00_s = 6.0 * s2 - 6.0 * s
        dh01_s = -6.0 * s2 + 6.0 * s
        dh10_s = 3.0 * s2 - 4.0 * s + 1.0
        dh11_s = 3.0 * s2 - 2.0 * s
        dh00_t = 6.0 * t2 - 6.0 * t
        dh01_t = -6.0 * t2 + 6.0 * t
        dh10_t = 3.0 * t2 - 4.0 * t + 1.0
        dh11_t = 3.0 * t2 - 2.0 * t
        coeffs = np.column_stack(
            [
                h00_s + h10_s * dm_A[:, 0, 0] + h11_s * dm_A[:, 1, 0],
                h01_s + h10_s * dm_A[:, 0, 1] + h11_s * dm_A[:, 1, 1],
                -(h00_t + h10_t * dm_B[:, 0, 0] + h11_t * dm_B[:, 1, 0]),
                -(h01_t + h10_t * dm_B[:, 0, 1] + h11_t * dm_B[:, 1, 1]),
            ]
        )
        dc_ds = np.column_stack(
            [
                dh00_s + dh10_s * dm_A[:, 0, 0] + dh11_s * dm_A[:, 1, 0],
                dh01_s + dh10_s * dm_A[:, 0, 1] + dh11_s * dm_A[:, 1, 1],
                np.zeros_like(s),
                np.zeros_like(s),
            ]
        )
        dc_dt = np.column_stack(
            [
                np.zeros_like(t),
                np.zeros_like(t),
                -(dh00_t + dh10_t * dm_B[:, 0, 0] + dh11_t * dm_B[:, 1, 0]),
                -(dh01_t + dh10_t * dm_B[:, 0, 1] + dh11_t * dm_B[:, 1, 1]),
            ]
        )
        return coeffs, dc_ds, dc_dt

    @staticmethod
    def _batch_dm_coeffs(
        node_counts: np.ndarray, nodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """バッチ dm 係数計算 (N, 2, 2) for A and B sides.

        nodes: (N, 4) [A0, A1, B0, B1]
        """
        n = len(nodes)
        dm_A = np.zeros((n, 2, 2))
        dm_B = np.zeros((n, 2, 2))
        c_a0 = np.maximum(node_counts[nodes[:, 0]], 1.0)
        c_a1 = np.maximum(node_counts[nodes[:, 1]], 1.0)
        c_b0 = np.maximum(node_counts[nodes[:, 2]], 1.0)
        c_b1 = np.maximum(node_counts[nodes[:, 3]], 1.0)
        # A side
        dm_A[:, 0, 0] = np.where(c_a0 < 1.5, -1.0, 0.0)
        dm_A[:, 0, 1] = 1.0 / c_a0
        dm_A[:, 1, 0] = -1.0 / c_a1
        dm_A[:, 1, 1] = np.where(c_a1 < 1.5, 1.0, 0.0)
        # B side
        dm_B[:, 0, 0] = np.where(c_b0 < 1.5, -1.0, 0.0)
        dm_B[:, 0, 1] = 1.0 / c_b0
        dm_B[:, 1, 0] = -1.0 / c_b1
        dm_B[:, 1, 1] = np.where(c_b1 < 1.5, 1.0, 0.0)
        return dm_A, dm_B

    def _batch_shape_coeffs(
        self,
        s: np.ndarray,
        t: np.ndarray,
        use_hermite: bool,
        node_counts: np.ndarray | None,
        nodes: np.ndarray,
    ) -> np.ndarray:
        """バッチ形状関数係数 (N, 4) を計算."""
        if use_hermite:
            if node_counts is not None:
                dm_A, dm_B = self._batch_dm_coeffs(node_counts, nodes)
                coeffs, _, _ = self._batch_hermite_corrected_coeffs(s, t, dm_A, dm_B)
                return coeffs
            return self._batch_hermite_coeffs(s, t)
        return np.column_stack([1.0 - s, s, -(1.0 - t), -t])

    def evaluate(
        self,
        u: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """接触力を評価（バッチ化版: status-246）.

        Returns:
            (f_c, residuals): f_c は接触力ベクトル、residuals はペア毎の残差
        """
        f_c = np.zeros(self._ndof)
        delta_h = self._resolve_delta_h(k_pen)

        if not hasattr(manager, "pairs") or len(manager.pairs) == 0:
            return f_c, np.zeros(0)

        _use_hermite = (
            hasattr(manager, "config")
            and hasattr(manager.config, "use_hermite_centerline")
            and manager.config.use_hermite_centerline
        )

        # バッチ配列抽出
        extracted = self._extract_pair_arrays(manager.pairs)
        if extracted is None:
            return f_c, np.zeros(0)
        has_state, gaps, s_arr, t_arr, normals, nodes, radius_a, radius_b = extracted

        # Hermite dm 補正用 node_counts
        # status-294: frozen-m 部分解消。evaluate/tangent 両方で dm_A/dm_B 有効化。
        # dm_ext は K_c_adj に委譲（K_st_adj との二重計上を防止）。
        _eval_node_counts = None
        if _use_hermite:
            _conn = getattr(manager, "connectivity", None)
            if _conn is not None:
                from xkep_cae.contact.geometry._compute import _compute_node_counts

                _max_node = int(np.max(_conn)) + 1 if len(_conn) > 0 else 0
                _eval_node_counts = _compute_node_counts(_max_node, _conn)

        # バッチ Huber 計算
        x_pen = k_pen * (-gaps)
        p_n_all = self._huber_batch(x_pen, delta_h)
        # Hertz型非線形ペナルティ（status-285）
        if self._penalty_exponent != 1.0:
            p_n_all = self._apply_power_law(p_n_all, k_pen)
        p_n_all[~has_state] = 0.0

        # 残差計算
        residuals = np.where(
            has_state & (p_n_all > 0.0),
            k_pen * gaps,
            0.0,
        )

        # ペア状態更新（frozen dataclass 制約でループ必須）
        for i in range(len(manager.pairs)):
            if has_state[i]:
                pair = manager.pairs[i]
                manager.pairs[i] = _evolve_pair(
                    pair, state=_evolve_state(pair.state, p_n=float(p_n_all[i]))
                )

        # アクティブペア（p_n > 0）のみで力ベクトル構築
        active = has_state & (p_n_all > 1e-30)
        n_active = int(np.sum(active))
        if n_active == 0:
            return f_c, residuals[has_state]

        p_n_act = p_n_all[active]
        s_act = s_arr[active]
        t_act = t_arr[active]
        n_act = normals[active]
        nodes_act = nodes[active]

        # 形状関数係数 (N, 4)
        coeffs = self._batch_shape_coeffs(
            s_act,
            t_act,
            _use_hermite,
            _eval_node_counts,
            nodes_act,
        )

        # g_shape: (N, 12) = coeffs[:, k] * normal
        g_shape = np.zeros((n_active, 12))
        for k in range(4):
            g_shape[:, k * 3 : k * 3 + 3] = coeffs[:, k : k + 1] * n_act

        # f_local = p_n * g_shape: (N, 12)
        f_local = p_n_act[:, None] * g_shape

        # グローバル DOF インデックス (N, 4) → scatter
        ndpn = self._ndof_per_node
        for k in range(4):
            for d in range(3):
                gdofs = nodes_act[:, k] * ndpn + d
                np.add.at(f_c, gdofs, f_local[:, k * 3 + d])

        return f_c, residuals[has_state]

    def tangent(
        self,
        u: np.ndarray,
        manager: object,
        k_pen: float,
        *,
        node_coords: np.ndarray | None = None,
    ) -> sp.csr_matrix:
        """接触接線剛性行列（バッチ化版: status-246）.

        K_c = K_mat - K_geo + K_st

        材料剛性: K_mat = h'(x) * k_pen * Σ_ij c_i c_j (n ⊗ n)
        幾何剛性: K_geo = p_n / dist * Σ_ij c_i c_j (I₃ - n ⊗ n)
        滑り剛性: K_st = outer(∂f_raw/∂s, ds_du) + outer(∂f_raw/∂t, dt_du)
        """
        delta_h = self._resolve_delta_h(k_pen)

        if not hasattr(manager, "pairs") or len(manager.pairs) == 0:
            return sp.csr_matrix((self._ndof, self._ndof))

        _use_hermite = (
            hasattr(manager, "config")
            and hasattr(manager.config, "use_hermite_centerline")
            and manager.config.use_hermite_centerline
        )

        _use_st = (
            hasattr(manager, "config")
            and hasattr(manager.config, "consistent_st_tangent")
            and manager.config.consistent_st_tangent
        )

        # Hermite 用 node_tangents + node_counts
        # status-294: frozen-m 部分解消。dm_A/dm_B を有効化（z方向DOFカップリング追加）。
        _node_tangents = None
        _node_counts = None
        _conn = None
        if _use_hermite and node_coords is not None:
            _conn = getattr(manager, "connectivity", None)
            if _conn is not None:
                from xkep_cae.contact.geometry._compute import (
                    _compute_node_tangents,
                )

                _node_tangents = _compute_node_tangents(node_coords, _conn)

        # 隣接ノードマップ + node_counts（status-273: K_c/K_st非局所拡張で共用）
        _adj_node_map = None
        _adj_node_counts = None
        if _use_hermite and _conn is not None:
            from xkep_cae.contact.geometry._compute import (
                _compute_adj_node_map,
                _compute_node_counts,
            )

            _adj_node_map = _compute_adj_node_map(_conn)
            _max_node = int(np.max(_conn)) + 1 if len(_conn) > 0 else 0
            _adj_node_counts = _compute_node_counts(_max_node, _conn)
            # status-294: frozen-m 部分解消
            _node_counts = _adj_node_counts

        # バッチ配列抽出
        extracted = self._extract_pair_arrays(manager.pairs)
        if extracted is None:
            return sp.csr_matrix((self._ndof, self._ndof))
        has_state, gaps, s_arr, t_arr, normals, nodes, radius_a, radius_b = extracted

        # バッチ Huber 導関数 + p_n
        x_pen = k_pen * (-gaps)
        h_deriv_all = self._huber_deriv_batch(x_pen, delta_h)
        # Hertz型非線形ペナルティの導関数補正（status-285）
        if self._penalty_exponent != 1.0:
            h_vals = self._huber_batch(x_pen, delta_h)
            h_deriv_all = self._apply_power_law_deriv(h_vals, h_deriv_all, k_pen)
        h_deriv_all[~has_state] = 0.0
        p_n_all = np.array(
            [manager.pairs[i].state.p_n if has_state[i] else 0.0 for i in range(len(manager.pairs))]
        )

        # アクティブ条件: h_deriv > 0 or p_n > 0
        active = has_state & ((h_deriv_all > 1e-30) | (p_n_all > 1e-30))
        n_act = int(np.sum(active))
        if n_act == 0 and not _use_st:
            return sp.csr_matrix((self._ndof, self._ndof))

        # ── K_mat + K_geo のバッチ計算 ──
        if n_act > 0:
            h_deriv_act = h_deriv_all[active]
            p_n_act = p_n_all[active]
            gaps_act = gaps[active]
            s_act = s_arr[active]
            t_act = t_arr[active]
            n_act_v = normals[active]
            nodes_act = nodes[active]
            ra_act = radius_a[active]
            rb_act = radius_b[active]

            # 重み計算
            w_mat = h_deriv_act * k_pen  # (N,)
            dist = gaps_act + ra_act + rb_act
            w_geo = np.where(dist > 1e-15, p_n_act / dist, 0.0)  # (N,)

            # 係数 (N, 4)
            coeffs = self._batch_shape_coeffs(
                s_act,
                t_act,
                _use_hermite,
                _node_counts,
                nodes_act,
            )

            # n⊗n: (N, 3, 3)
            nn = n_act_v[:, :, None] * n_act_v[:, None, :]
            # I3 - n⊗n
            I3 = np.eye(3)[None, :, :]  # (1, 3, 3)
            I_nn = I3 - nn  # (N, 3, 3)

            # K_3x3 = w_mat * (n⊗n) - w_geo * (I - n⊗n) per pair
            K_3x3 = w_mat[:, None, None] * nn - w_geo[:, None, None] * I_nn  # (N, 3, 3)
            # K_3x3_mat: 隣接ノード用。幾何剛性(I-n⊗n)を除外（status-295）。
            # 理由: 隣接ノード変位→s追従により法線変化はほぼ相殺されるが、
            # ギャップ変化(n⊗n項)は維持される。K_c_adj = mat_only で 11%→1.8%。
            K_3x3_mat = w_mat[:, None, None] * nn  # (N, 3, 3)

            # c_i * c_j: (N, 4, 4)
            cc = coeffs[:, :, None] * coeffs[:, None, :]

            # 局所 12×12 行列: K_local[ki*3+di, kj*3+dj] = cc[ki,kj] * K_3x3[di,dj]
            K_local = np.zeros((n_act, 12, 12))
            for ki in range(4):
                for kj in range(4):
                    K_local[:, ki * 3 : (ki + 1) * 3, kj * 3 : (kj + 1) * 3] = (
                        cc[:, ki, kj][:, None, None] * K_3x3
                    )

            # DOF インデックス (N, 12)
            ndpn = self._ndof_per_node
            gdofs = np.zeros((n_act, 12), dtype=int)
            for k in range(4):
                for d in range(3):
                    gdofs[:, k * 3 + d] = nodes_act[:, k] * ndpn + d

            # COO 配列構築
            row_idx = np.broadcast_to(gdofs[:, :, None], (n_act, 12, 12)).ravel()
            col_idx = np.broadcast_to(gdofs[:, None, :], (n_act, 12, 12)).ravel()
            val_arr = K_local.ravel()
            mask = np.abs(val_arr) > 1e-30
            rows_np = row_idx[mask]
            cols_np = col_idx[mask]
            vals_np = val_arr[mask]

            # ── K_c_adj: 隣接ノードへの K_mat+K_geo 拡張（status-273: Step3） ──
            if _adj_node_map is not None and _adj_node_counts is not None:
                # elem_a, elem_b 抽出
                active_idx = np.where(active)[0]
                elem_a_act = np.array([manager.pairs[int(idx)].elem_a for idx in active_idx])
                elem_b_act = np.array([manager.pairs[int(idx)].elem_b for idx in active_idx])

                # dm_ext 係数のバッチ計算
                c_a0 = np.maximum(_adj_node_counts[nodes_act[:, 0]], 1.0)
                c_a1 = np.maximum(_adj_node_counts[nodes_act[:, 1]], 1.0)
                c_b0 = np.maximum(_adj_node_counts[nodes_act[:, 2]], 1.0)
                c_b1 = np.maximum(_adj_node_counts[nodes_act[:, 3]], 1.0)
                dm_ext_a0 = np.where(c_a0 >= 1.5, -1.0 / c_a0, 0.0)
                dm_ext_a1 = np.where(c_a1 >= 1.5, 1.0 / c_a1, 0.0)
                dm_ext_b0 = np.where(c_b0 >= 1.5, -1.0 / c_b0, 0.0)
                dm_ext_b1 = np.where(c_b1 >= 1.5, 1.0 / c_b1, 0.0)

                # Hermite tangent basis H10, H11
                s2_h, s3_h = s_act * s_act, s_act * s_act * s_act
                t2_h, t3_h = t_act * t_act, t_act * t_act * t_act
                h10_s = s3_h - 2.0 * s2_h + s_act
                h11_s = s3_h - s2_h
                h10_t = t3_h - 2.0 * t2_h + t_act
                h11_t = t3_h - t2_h

                # alpha_adj (N, 4): 隣接ノードの有効係数
                alpha_adj = np.column_stack(
                    [
                        h10_s * dm_ext_a0,
                        h11_s * dm_ext_a1,
                        -h10_t * dm_ext_b0,
                        -h11_t * dm_ext_b1,
                    ]
                )

                # adj global node indices (N, 4)
                adj_gnodes = np.full((n_act, 4), -1, dtype=int)
                for i in range(n_act):
                    adj_a = _adj_node_map.get(int(elem_a_act[i]), (-1, -1))
                    adj_b = _adj_node_map.get(int(elem_b_act[i]), (-1, -1))
                    adj_gnodes[i] = [adj_a[0], adj_a[1], adj_b[0], adj_b[1]]

                # c_alpha (N, 4, 4) = coeffs[ki] * alpha_adj[aj]
                c_alpha = coeffs[:, :, None] * alpha_adj[:, None, :]
                K_c_adj = np.zeros((n_act, 12, 12))
                for ki in range(4):
                    for aj in range(4):
                        K_c_adj[:, ki * 3 : (ki + 1) * 3, aj * 3 : (aj + 1) * 3] = (
                            c_alpha[:, ki, aj][:, None, None] * K_3x3_mat
                        )

                # adj DOF indices (N, 12) + validity mask
                ndpn = self._ndof_per_node
                adj_gdofs = np.zeros((n_act, 12), dtype=int)
                adj_valid = np.zeros((n_act, 12), dtype=bool)
                for aj in range(4):
                    valid = adj_gnodes[:, aj] >= 0
                    for d in range(3):
                        adj_gdofs[:, aj * 3 + d] = np.where(valid, adj_gnodes[:, aj] * ndpn + d, 0)
                        adj_valid[:, aj * 3 + d] = valid

                # COO 構築
                row_adj = np.broadcast_to(gdofs[:, :, None], (n_act, 12, 12)).ravel()
                col_adj = np.broadcast_to(adj_gdofs[:, None, :], (n_act, 12, 12)).ravel()
                val_adj = K_c_adj.ravel()
                valid_flat = np.broadcast_to(adj_valid[:, None, :], (n_act, 12, 12)).ravel()
                mask_adj = valid_flat & (np.abs(val_adj) > 1e-30)
                if mask_adj.any():
                    rows_np = np.concatenate([rows_np, row_adj[mask_adj]])
                    cols_np = np.concatenate([cols_np, col_adj[mask_adj]])
                    vals_np = np.concatenate([vals_np, val_adj[mask_adj]])
        else:
            rows_np = np.array([], dtype=int)
            cols_np = np.array([], dtype=int)
            vals_np = np.array([], dtype=float)

        # ── K_st（ContactForceStStiffnessProcess 経由: status-256 B1） ──
        if _use_st and node_coords is not None:
            b1 = ContactForceStStiffnessProcess()
            K_st = b1.process(
                ContactForceStStiffnessInput(
                    pairs=manager.pairs,
                    node_coords=node_coords,
                    k_pen=k_pen,
                    delta_h=delta_h,
                    ndof_total=self._ndof,
                    ndof_per_node=self._ndof_per_node,
                    use_hermite=_use_hermite,
                    node_tangents=_node_tangents,
                    node_counts=_node_counts,
                    adj_node_map=_adj_node_map,
                    penalty_exponent=self._penalty_exponent,
                )
            ).K_st
            # K_st の COO を結合
            K_st_coo = K_st.tocoo()
            if K_st_coo.nnz > 0:
                rows_np = np.concatenate([rows_np, K_st_coo.row])
                cols_np = np.concatenate([cols_np, K_st_coo.col])
                vals_np = np.concatenate([vals_np, K_st_coo.data])

        if len(vals_np) > 0:
            return sp.coo_matrix(
                (vals_np, (rows_np, cols_np)),
                shape=(self._ndof, self._ndof),
            ).tocsr()
        return sp.csr_matrix((self._ndof, self._ndof))

    def tangent_components(
        self,
        u: np.ndarray,
        manager: object,
        k_pen: float,
        *,
        node_coords: np.ndarray | None = None,
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        """K_mat, K_geo, K_st を個別に返す（status-291: 個別FD検証用）.

        K_c = K_mat - K_geo + K_st
        """
        delta_h = self._resolve_delta_h(k_pen)
        ndof = self._ndof
        zero = sp.csr_matrix((ndof, ndof))

        if not hasattr(manager, "pairs") or len(manager.pairs) == 0:
            return zero, zero, zero

        _use_hermite = (
            hasattr(manager, "config")
            and hasattr(manager.config, "use_hermite_centerline")
            and manager.config.use_hermite_centerline
        )
        _use_st = (
            hasattr(manager, "config")
            and hasattr(manager.config, "consistent_st_tangent")
            and manager.config.consistent_st_tangent
        )

        _node_tangents = None
        _node_counts = None
        _conn = None
        if _use_hermite and node_coords is not None:
            _conn = getattr(manager, "connectivity", None)
            if _conn is not None:
                from xkep_cae.contact.geometry._compute import (
                    _compute_node_tangents,
                )

                _node_tangents = _compute_node_tangents(node_coords, _conn)

        _adj_node_map = None
        _adj_node_counts = None
        if _use_hermite and _conn is not None:
            from xkep_cae.contact.geometry._compute import (
                _compute_adj_node_map,
                _compute_node_counts,
            )

            _adj_node_map = _compute_adj_node_map(_conn)
            _max_node = int(np.max(_conn)) + 1 if len(_conn) > 0 else 0
            _adj_node_counts = _compute_node_counts(_max_node, _conn)
            # status-294: frozen-m 部分解消
            _node_counts = _adj_node_counts

        extracted = self._extract_pair_arrays(manager.pairs)
        if extracted is None:
            return zero, zero, zero
        has_state, gaps, s_arr, t_arr, normals, nodes, radius_a, radius_b = extracted

        x_pen = k_pen * (-gaps)
        h_deriv_all = self._huber_deriv_batch(x_pen, delta_h)
        if self._penalty_exponent != 1.0:
            h_vals = self._huber_batch(x_pen, delta_h)
            h_deriv_all = self._apply_power_law_deriv(h_vals, h_deriv_all, k_pen)
        h_deriv_all[~has_state] = 0.0
        p_n_all = np.array(
            [manager.pairs[i].state.p_n if has_state[i] else 0.0 for i in range(len(manager.pairs))]
        )

        active = has_state & ((h_deriv_all > 1e-30) | (p_n_all > 1e-30))
        n_act = int(np.sum(active))

        K_mat = zero
        K_geo = zero
        K_st = zero

        if n_act > 0:
            h_deriv_act = h_deriv_all[active]
            p_n_act = p_n_all[active]
            gaps_act = gaps[active]
            s_act = s_arr[active]
            t_act = t_arr[active]
            n_act_v = normals[active]
            nodes_act = nodes[active]
            ra_act = radius_a[active]
            rb_act = radius_b[active]

            w_mat = h_deriv_act * k_pen
            dist = gaps_act + ra_act + rb_act
            w_geo = np.where(dist > 1e-15, p_n_act / dist, 0.0)

            coeffs = self._batch_shape_coeffs(
                s_act,
                t_act,
                _use_hermite,
                _node_counts,
                nodes_act,
            )

            nn = n_act_v[:, :, None] * n_act_v[:, None, :]
            I3 = np.eye(3)[None, :, :]
            I_nn = I3 - nn

            # K_mat 3x3: w_mat * (n⊗n)
            K_3x3_mat = w_mat[:, None, None] * nn
            # K_geo 3x3: w_geo * (I - n⊗n)
            K_3x3_geo = w_geo[:, None, None] * I_nn

            cc = coeffs[:, :, None] * coeffs[:, None, :]
            ndpn = self._ndof_per_node

            gdofs = np.zeros((n_act, 12), dtype=int)
            for k in range(4):
                for d in range(3):
                    gdofs[:, k * 3 + d] = nodes_act[:, k] * ndpn + d

            def _assemble_12x12(K_3x3_block):
                K_local = np.zeros((n_act, 12, 12))
                for ki in range(4):
                    for kj in range(4):
                        K_local[:, ki * 3 : (ki + 1) * 3, kj * 3 : (kj + 1) * 3] = (
                            cc[:, ki, kj][:, None, None] * K_3x3_block
                        )
                row_idx = np.broadcast_to(gdofs[:, :, None], (n_act, 12, 12)).ravel()
                col_idx = np.broadcast_to(gdofs[:, None, :], (n_act, 12, 12)).ravel()
                val_arr = K_local.ravel()
                mask = np.abs(val_arr) > 1e-30
                return sp.coo_matrix(
                    (val_arr[mask], (row_idx[mask], col_idx[mask])),
                    shape=(ndof, ndof),
                ).tocsr()

            K_mat = _assemble_12x12(K_3x3_mat)
            K_geo = _assemble_12x12(K_3x3_geo)

            # ── K_mat_adj: 隣接ノードへの材料剛性拡張（status-295） ──
            # 幾何剛性(I-n⊗n)は隣接ノードではs追従により相殺されるため除外。
            if _adj_node_map is not None and _adj_node_counts is not None:
                active_idx = np.where(active)[0]
                elem_a_act = np.array([manager.pairs[int(idx)].elem_a for idx in active_idx])
                elem_b_act = np.array([manager.pairs[int(idx)].elem_b for idx in active_idx])

                c_a0 = np.maximum(_adj_node_counts[nodes_act[:, 0]], 1.0)
                c_a1 = np.maximum(_adj_node_counts[nodes_act[:, 1]], 1.0)
                c_b0 = np.maximum(_adj_node_counts[nodes_act[:, 2]], 1.0)
                c_b1 = np.maximum(_adj_node_counts[nodes_act[:, 3]], 1.0)
                dm_ext_a0 = np.where(c_a0 >= 1.5, -1.0 / c_a0, 0.0)
                dm_ext_a1 = np.where(c_a1 >= 1.5, 1.0 / c_a1, 0.0)
                dm_ext_b0 = np.where(c_b0 >= 1.5, -1.0 / c_b0, 0.0)
                dm_ext_b1 = np.where(c_b1 >= 1.5, 1.0 / c_b1, 0.0)

                s2_h, s3_h = s_act * s_act, s_act * s_act * s_act
                t2_h, t3_h = t_act * t_act, t_act * t_act * t_act
                h10_s = s3_h - 2.0 * s2_h + s_act
                h11_s = s3_h - s2_h
                h10_t = t3_h - 2.0 * t2_h + t_act
                h11_t = t3_h - t2_h

                alpha_adj = np.column_stack(
                    [
                        h10_s * dm_ext_a0,
                        h11_s * dm_ext_a1,
                        -h10_t * dm_ext_b0,
                        -h11_t * dm_ext_b1,
                    ]
                )

                adj_gnodes = np.full((n_act, 4), -1, dtype=int)
                for i in range(n_act):
                    adj_a = _adj_node_map.get(int(elem_a_act[i]), (-1, -1))
                    adj_b = _adj_node_map.get(int(elem_b_act[i]), (-1, -1))
                    adj_gnodes[i] = [adj_a[0], adj_a[1], adj_b[0], adj_b[1]]

                c_alpha = coeffs[:, :, None] * alpha_adj[:, None, :]
                K_mat_adj_local = np.zeros((n_act, 12, 12))
                for ki in range(4):
                    for aj in range(4):
                        K_mat_adj_local[:, ki * 3 : (ki + 1) * 3, aj * 3 : (aj + 1) * 3] = (
                            c_alpha[:, ki, aj][:, None, None] * K_3x3_mat
                        )

                adj_gdofs = np.zeros((n_act, 12), dtype=int)
                adj_valid = np.zeros((n_act, 12), dtype=bool)
                for aj in range(4):
                    valid = adj_gnodes[:, aj] >= 0
                    for d in range(3):
                        adj_gdofs[:, aj * 3 + d] = np.where(valid, adj_gnodes[:, aj] * ndpn + d, 0)
                        adj_valid[:, aj * 3 + d] = valid

                row_adj = np.broadcast_to(gdofs[:, :, None], (n_act, 12, 12)).ravel()
                col_adj = np.broadcast_to(adj_gdofs[:, None, :], (n_act, 12, 12)).ravel()
                val_adj = K_mat_adj_local.ravel()
                valid_flat = np.broadcast_to(adj_valid[:, None, :], (n_act, 12, 12)).ravel()
                mask_adj = valid_flat & (np.abs(val_adj) > 1e-30)
                if mask_adj.any():
                    K_mat_adj = sp.coo_matrix(
                        (
                            val_adj[mask_adj],
                            (row_adj[mask_adj], col_adj[mask_adj]),
                        ),
                        shape=(ndof, ndof),
                    ).tocsr()
                    K_mat = K_mat + K_mat_adj

        if _use_st and node_coords is not None:
            b1 = ContactForceStStiffnessProcess()
            K_st = b1.process(
                ContactForceStStiffnessInput(
                    pairs=manager.pairs,
                    node_coords=node_coords,
                    k_pen=k_pen,
                    delta_h=delta_h,
                    ndof_total=ndof,
                    ndof_per_node=self._ndof_per_node,
                    use_hermite=_use_hermite,
                    node_tangents=_node_tangents,
                    node_counts=_node_counts,
                    adj_node_map=_adj_node_map,
                    penalty_exponent=self._penalty_exponent,
                )
            ).K_st

        return K_mat, K_geo, K_st

    def process(self, input_data: ContactForceInput) -> ContactForceOutput:
        f, _ = self.evaluate(
            input_data.u,
            input_data.manager,
            input_data.k_pen,
        )
        return ContactForceOutput(contact_force=f)


# ── ファクトリ ─────────────────────────────────────────────


def _create_contact_force_strategy(
    *,
    ndof: int = 0,
    ndof_per_node: int = 6,
    smoothing_delta: float = 0.0,
    huber_delta_h: float = 0.0,
    penalty_exponent: float = 1.0,
) -> HuberContactForceProcess:
    """接触力 Strategy ファクトリ（status-222 で一本化）."""
    return HuberContactForceProcess(
        ndof=ndof,
        ndof_per_node=ndof_per_node,
        smoothing_delta=smoothing_delta,
        huber_delta_h=huber_delta_h,
        penalty_exponent=penalty_exponent,
    )
