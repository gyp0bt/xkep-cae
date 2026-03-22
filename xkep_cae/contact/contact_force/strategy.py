"""ContactForce Strategy 具象実装.

ContactForceStrategy Protocol に従い、接触力を評価する Process。

status-222 で完全一本化:
- HuberContactForceProcess: Huber ペナルティ接触力（唯一の実装）
- SmoothPenalty / NCP / Uzawa は status-222 で削除。復元手順は status-222.md 参照。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._assembly_utils import _contact_dofs
from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state
from xkep_cae.contact._types import ContactStatus
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


# ── ヘルパー ───────────────────────────────────────────────


def _contact_shape_vector(pair: object) -> np.ndarray:
    """接触形状ベクトル g_shape (12,) を構築する.

    法線方向の形状関数:
        g_shape = [(1-s)*n, s*n, -(1-t)*n, -t*n]  (4×3 = 12)

    Args:
        pair: ContactPair（state.s, state.t, state.normal を持つ）

    Returns:
        g_shape: (12,) 形状ベクトル
    """
    s = pair.state.s
    t = pair.state.t
    normal = pair.state.normal
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

        if hasattr(manager, "pairs"):
            for i, pair in enumerate(manager.pairs):
                if not hasattr(pair, "state"):
                    continue
                if pair.state.status == ContactStatus.INACTIVE:
                    continue

                g_i = pair.state.gap
                x_pen = k_pen * (-g_i)
                p_n = self._huber(x_pen, delta_h)

                # pair.state.p_n を更新（摩擦力計算で使用）
                manager.pairs[i] = _evolve_pair(
                    pair, state=_evolve_state(pair.state, p_n=p_n)
                )
                pair = manager.pairs[i]

                residuals.append(k_pen * g_i if p_n > 0.0 else 0.0)

                if p_n <= 1e-30:
                    continue

                g_shape = _contact_shape_vector(pair)
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
    ) -> sp.csr_matrix:
        """接触接線剛性行列（Huber C1 連続）.

        残差 R = f_int + f_c - f_ext において f_c = -f_c_raw（status-221）。
        したがって dR/du の接触寄与は:
            K_c = -d(f_c_raw)/du = -dp_n/dg * g_shape ⊗ g_shape

        p_n = huber(k_pen*(-g), δ_h)
        dp_n/dg = huber'(x) * (-k_pen)  （h' > 0, -k_pen < 0 → 負）
        よって K_c = huber'(x) * k_pen * g_shape ⊗ g_shape  （正定値）
        """
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        delta_h = k_pen / self._smoothing_delta if self._smoothing_delta > 0.0 else 0.0

        if hasattr(manager, "pairs"):
            for pair in manager.pairs:
                if not hasattr(pair, "state"):
                    continue
                if pair.state.status == ContactStatus.INACTIVE:
                    continue

                g_i = pair.state.gap
                x_pen = k_pen * (-g_i)
                h_deriv = self._huber_deriv(x_pen, delta_h)

                if h_deriv < 1e-30:
                    continue

                weight = h_deriv * k_pen

                g_shape = _contact_shape_vector(pair)
                dofs = _contact_dofs(pair, self._ndof_per_node)

                for ki in range(4):
                    for di in range(3):
                        li = ki * 3 + di
                        gi = dofs[ki * self._ndof_per_node + di]
                        w_gi = weight * g_shape[li]
                        if abs(w_gi) < 1e-30:
                            continue
                        for kj in range(4):
                            for dj in range(3):
                                lj = kj * 3 + dj
                                gj = dofs[kj * self._ndof_per_node + dj]
                                val = w_gi * g_shape[lj]
                                if abs(val) > 1e-30:
                                    rows.append(gi)
                                    cols.append(gj)
                                    vals.append(val)

        if rows:
            return sp.csr_matrix(
                (np.array(vals), (np.array(rows), np.array(cols))),
                shape=(self._ndof, self._ndof),
            )
        return sp.csr_matrix((self._ndof, self._ndof))

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
