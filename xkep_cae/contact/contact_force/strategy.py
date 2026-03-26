"""ContactForce Strategy 具象実装.

ContactForceStrategy Protocol に従い、接触力を評価する Process。

status-222 で完全一本化:
- HuberContactForceProcess: Huber ペナルティ接触力（唯一の実装）
- SmoothPenalty / NCP / Uzawa は status-222 で削除。復元手順は status-222.md 参照。

status-230: Hermite 幾何対応
- 形状関数係数を Hermite 基底 H00(s)/H01(s) に切替
- ∂n/∂s を Hermite 接線 dpA/ds で計算
- K_st に Hermite 版 StJacobian を使用
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._assembly_utils import _contact_dofs
from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state
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


# ── ヘルパー ───────────────────────────────────────────────


def _contact_shape_vector(pair: object, *, use_hermite: bool = False) -> np.ndarray:
    """接触形状ベクトル g_shape (12,) を構築する.

    線形: g_shape = [(1-s)*n, s*n, -(1-t)*n, -t*n]
    Hermite: g_shape = [H00(s)*n, H01(s)*n, -H00(t)*n, -H01(t)*n]

    Args:
        pair: ContactPair（state.s, state.t, state.normal を持つ）
        use_hermite: True なら Hermite 基底を使用

    Returns:
        g_shape: (12,) 形状ベクトル
    """
    s = pair.state.s
    t = pair.state.t
    normal = pair.state.normal
    if use_hermite:
        coeffs = _hermite_shape_coeffs(s, t)
    else:
        coeffs = [(1.0 - s), s, -(1.0 - t), -t]
    g_shape = np.zeros(12)
    for k in range(4):
        g_shape[k * 3 : k * 3 + 3] = coeffs[k] * normal
    return g_shape


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

    def __init__(
        self,
        ndof: int,
        ndof_per_node: int = 6,
        *,
        smoothing_delta: float = 0.0,
    ) -> None:
        self._ndof = ndof
        self._ndof_per_node = ndof_per_node
        self._smoothing_delta = smoothing_delta

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

    def evaluate(
        self,
        u: np.ndarray,
        manager: object,
        k_pen: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """接触力を評価.

        Returns:
            (f_c, residuals): f_c は接触力ベクトル、residuals はペア毎の残差
        """
        f_c = np.zeros(self._ndof)
        residuals: list[float] = []
        delta_h = k_pen / self._smoothing_delta if self._smoothing_delta > 0.0 else 0.0

        _use_hermite = (
            hasattr(manager, "config")
            and hasattr(manager.config, "use_hermite_centerline")
            and manager.config.use_hermite_centerline
        )

        if hasattr(manager, "pairs"):
            for i, pair in enumerate(manager.pairs):
                if not hasattr(pair, "state"):
                    continue
                # SDI 排除: INACTIVE skip を除去。Huber penalty は gap > 0 で
                # 自然に p_n=0 を返すため、全候補ペアを評価する（status-233）。

                g_i = pair.state.gap
                x_pen = k_pen * (-g_i)
                p_n = self._huber(x_pen, delta_h)

                # pair.state.p_n を更新（摩擦力計算で使用）
                manager.pairs[i] = _evolve_pair(pair, state=_evolve_state(pair.state, p_n=p_n))
                pair = manager.pairs[i]

                residuals.append(k_pen * g_i if p_n > 0.0 else 0.0)

                if p_n <= 1e-30:
                    continue

                g_shape = _contact_shape_vector(pair, use_hermite=_use_hermite)
                dofs = _contact_dofs(pair, self._ndof_per_node)
                for k in range(4):
                    for d in range(3):
                        local_idx = k * 3 + d
                        global_idx = dofs[k * self._ndof_per_node + d]
                        f_c[global_idx] += p_n * g_shape[local_idx]

        residual_arr = np.array(residuals) if residuals else np.zeros(0)
        return f_c, residual_arr

    def tangent(
        self,
        u: np.ndarray,
        manager: object,
        k_pen: float,
        *,
        node_coords: np.ndarray | None = None,
    ) -> sp.csr_matrix:
        """接触接線剛性行列（Huber C1 連続 + 幾何剛性 + K_st）.

        残差 R = f_int + f_c - f_ext において f_c = -f_c_raw（status-221）。
        したがって dR/du の接触寄与は:

        K_c = K_mat - K_geo + K_st

        材料剛性（ペナルティ勾配、正定値）:
            K_mat = h'(x) * k_pen * Σ_ij c_i c_j (n ⊗ n)

        幾何剛性（法線回転、減算）:
            K_geo = p_n / dist * Σ_ij c_i c_j (I₃ - n ⊗ n)

        滑り剛性（接触点移動、status-226 + status-242 ∂p_n/∂s 追加）:
            K_st = outer(∂f_raw/∂s, ds_du) + outer(∂f_raw/∂t, dt_du)
            ∂f_raw/∂s = (∂p_n/∂s)·g_shape + p_n·(∂g_shape/∂s)

        線形: c = [(1-s), s, -(1-t), -t]
        Hermite: c = [H00(s), H01(s), -H00(t), -H01(t)]（status-230）
        """
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        delta_h = k_pen / self._smoothing_delta if self._smoothing_delta > 0.0 else 0.0

        # Hermite フラグ
        _use_hermite = (
            hasattr(manager, "config")
            and hasattr(manager.config, "use_hermite_centerline")
            and manager.config.use_hermite_centerline
        )

        # consistent_st_tangent フラグの取得
        _use_st = (
            hasattr(manager, "config")
            and hasattr(manager.config, "consistent_st_tangent")
            and manager.config.consistent_st_tangent
        )
        if _use_st:
            from xkep_cae.contact.geometry._st_jacobian import (
                ComputeStJacobianProcess,
                StJacobianInput,
            )

            _st_proc = ComputeStJacobianProcess()

        # Hermite 用 node_tangents（K_st, dn/ds で使用）
        _node_tangents = None
        if _use_hermite and node_coords is not None:
            _conn = getattr(manager, "connectivity", None)
            if _conn is not None:
                from xkep_cae.contact.geometry._compute import _compute_node_tangents

                _node_tangents = _compute_node_tangents(node_coords, _conn)

        if hasattr(manager, "pairs"):
            for pair in manager.pairs:
                if not hasattr(pair, "state"):
                    continue
                # SDI 排除: INACTIVE skip を除去（status-233）。

                g_i = pair.state.gap
                x_pen = k_pen * (-g_i)
                h_deriv = self._huber_deriv(x_pen, delta_h)
                p_n = pair.state.p_n

                if h_deriv < 1e-30 and p_n < 1e-30:
                    continue

                # 材料剛性の重み
                w_mat = h_deriv * k_pen

                # 幾何剛性の重み: p_n / dist
                dist = g_i + pair.radius_a + pair.radius_b
                w_geo = p_n / dist if dist > 1e-15 else 0.0

                normal = pair.state.normal
                s = pair.state.s
                t = pair.state.t

                if _use_hermite:
                    coeffs = _hermite_shape_coeffs(s, t)
                else:
                    coeffs = [(1.0 - s), s, -(1.0 - t), -t]

                dofs = _contact_dofs(pair, self._ndof_per_node)

                for ki in range(4):
                    ci = coeffs[ki]
                    if abs(ci) < 1e-30:
                        continue
                    for kj in range(4):
                        cj = coeffs[kj]
                        if abs(cj) < 1e-30:
                            continue
                        cc = ci * cj
                        for di in range(3):
                            gi = dofs[ki * self._ndof_per_node + di]
                            for dj in range(3):
                                gj = dofs[kj * self._ndof_per_node + dj]
                                # K_mat: +w_mat * cc * n_i * n_j（正定値）
                                val = w_mat * cc * normal[di] * normal[dj]
                                # K_geo: -w_geo * cc * (δ_ij - n_i * n_j)（法線回転）
                                delta_ij = 1.0 if di == dj else 0.0
                                val -= w_geo * cc * (delta_ij - normal[di] * normal[dj])
                                if abs(val) > 1e-30:
                                    rows.append(gi)
                                    cols.append(gj)
                                    vals.append(val)

                # K_st: 接触点滑り剛性（∂(s,t)/∂u の連鎖微分項）
                if _use_st and p_n > 1e-30 and node_coords is not None:
                    self._add_kst_contact(
                        pair,
                        p_n,
                        normal,
                        dofs,
                        _st_proc,
                        StJacobianInput,
                        rows,
                        cols,
                        vals,
                        node_coords,
                        use_hermite=_use_hermite,
                        node_tangents=_node_tangents,
                        h_deriv=h_deriv,
                        k_pen=k_pen,
                    )

        if rows:
            return sp.csr_matrix(
                (np.array(vals), (np.array(rows), np.array(cols))),
                shape=(self._ndof, self._ndof),
            )
        return sp.csr_matrix((self._ndof, self._ndof))

    def _add_kst_contact(
        self,
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
        *,
        use_hermite: bool = False,
        node_tangents: np.ndarray | None = None,
        h_deriv: float = 0.0,
        k_pen: float = 0.0,
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

        # StJacobian 入力の構築
        st_kw: dict = {
            "xA0": xA0,
            "xA1": xA1,
            "xB0": xB0,
            "xB1": xB1,
            "s": st.s,
            "t": st.t,
        }
        if use_hermite and node_tangents is not None:
            st_kw["mA0"] = node_tangents[pair.nodes_a[0]]
            st_kw["mA1"] = node_tangents[pair.nodes_a[1]]
            st_kw["mB0"] = node_tangents[pair.nodes_b[0]]
            st_kw["mB1"] = node_tangents[pair.nodes_b[1]]
            st_kw["use_hermite"] = True

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
        # gap(s,t) = dist - R_A - R_B, dist = ||pA(s) - pB(t)||
        # delta = pA(s) - pB(t) (normal 方向に dist を掛けたもの)
        # ∂gap/∂s = (delta · dpA/ds) / dist = normal · dpA/ds
        # ∂gap/∂t = -(delta · dpB/dt) / dist = -normal · dpB/dt
        # ∂p_n/∂s = h'(x) * k_pen * (-∂gap/∂s)  (x = k_pen * (-gap))
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
                df_ds[li] = dpn_ds * g_shape[li] + p_n * (
                    dc_ds[k] * normal[i] + coeffs[k] * dn_ds[i]
                )
                df_dt[li] = dpn_dt * g_shape[li] + p_n * (
                    dc_dt[k] * normal[i] + coeffs[k] * dn_dt[i]
                )

        # K_st = -(df_ds ⊗ ds_du + df_dt ⊗ dt_du)
        K_st_local = -(np.outer(df_ds, out.ds_du) + np.outer(df_dt, out.dt_du))

        ndpn = self._ndof_per_node
        for ki in range(4):
            for di in range(3):
                li = ki * 3 + di
                gi = dofs[ki * ndpn + di]
                for kj in range(4):
                    for dj in range(3):
                        lj = kj * 3 + dj
                        gj = dofs[kj * ndpn + dj]
                        val = K_st_local[li, lj]
                        if abs(val) > 1e-30:
                            rows.append(gi)
                            cols.append(gj)
                            vals.append(val)

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
) -> HuberContactForceProcess:
    """接触力 Strategy ファクトリ（status-222 で一本化）."""
    return HuberContactForceProcess(
        ndof=ndof,
        ndof_per_node=ndof_per_node,
        smoothing_delta=smoothing_delta,
    )
