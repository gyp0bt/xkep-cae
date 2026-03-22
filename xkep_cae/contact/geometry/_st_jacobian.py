"""∂(s,t)/∂u Jacobian — 最近接点パラメータの変位微分.

最近接点条件の陰関数微分により、接触パラメータ (s, t) の
節点変位に対する感度を計算する。

理論:
    最近接点条件:
        F₁ = δ · dA = 0
        F₂ = -δ · dB = 0
    ただし δ = pA(s) - pB(t), dA = xA1 - xA0, dB = xB1 - xB0

    陰関数定理: J · [ds, dt]ᵀ = -[∂F₁/∂u, ∂F₂/∂u]ᵀ
    J = [[a, -b], [-b, c]] (a=dA·dA, b=dA·dB, c=dB·dB)

status-078 の旧実装を Process Architecture で再実装。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.core import ProcessMeta, SolverProcess


@dataclass(frozen=True)
class StJacobianInput:
    """∂(s,t)/∂u 計算の入力."""

    xA0: np.ndarray  # (3,) セグメントA始点
    xA1: np.ndarray  # (3,) セグメントA終点
    xB0: np.ndarray  # (3,) セグメントB始点
    xB1: np.ndarray  # (3,) セグメントB終点
    s: float  # 最近接点パラメータ s ∈ [0,1]
    t: float  # 最近接点パラメータ t ∈ [0,1]
    s_unclamped: float | None = None  # クランプ前の s（None なら s を使用）
    t_unclamped: float | None = None  # クランプ前の t（None なら t を使用）
    tol_singular: float = 1e-10  # 特異判定閾値


@dataclass(frozen=True)
class StJacobianOutput:
    """∂(s,t)/∂u 計算の出力."""

    ds_du: np.ndarray  # (12,) ds/du（4ノード × 3次元）
    dt_du: np.ndarray  # (12,) dt/du（4ノード × 3次元）
    valid: bool  # 計算が有効か（平行特異でなければ True）


class ComputeStJacobianProcess(
    SolverProcess[StJacobianInput, StJacobianOutput],
):
    """最近接点パラメータの変位感度 ∂(s,t)/∂u を計算.

    status-078 の compute_st_jacobian を Process Architecture で再実装。
    """

    meta = ProcessMeta(
        name="ComputeStJacobian",
        module="geometry",
        version="2.0.0",
        document_path="docs/contact_geometry.md",
    )

    def process(self, inp: StJacobianInput) -> StJacobianOutput:
        dA = inp.xA1 - inp.xA0
        dB = inp.xB1 - inp.xB0
        s = inp.s
        t = inp.t

        # クランプ状態の検出
        s_unc = inp.s_unclamped if inp.s_unclamped is not None else s
        t_unc = inp.t_unclamped if inp.t_unclamped is not None else t
        s_clamped = (s_unc < 0.0) or (s_unc > 1.0)
        t_clamped = (t_unc < 0.0) or (t_unc > 1.0)

        ds_du = np.zeros(12)
        dt_du = np.zeros(12)

        if s_clamped and t_clamped:
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

        # Gram 行列の要素
        a = float(np.dot(dA, dA))
        b = float(np.dot(dA, dB))
        c = float(np.dot(dB, dB))
        det = a * c - b * b

        # δ = pA(s) - pB(t)
        delta = (1.0 - s) * inp.xA0 + s * inp.xA1 - (1.0 - t) * inp.xB0 - t * inp.xB1

        if s_clamped:
            # ds/du = 0, dt のみ 1×1 系
            if c < inp.tol_singular:
                return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=False)
            dt_du = self._compute_dt_only(delta, dA, dB, s, t, c)
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

        if t_clamped:
            # dt/du = 0, ds のみ 1×1 系
            if a < inp.tol_singular:
                return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=False)
            ds_du = self._compute_ds_only(delta, dA, dB, s, t, a)
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

        # 通常: 2×2 系
        ac_product = max(a * c, 1e-30)
        if abs(det) < inp.tol_singular * ac_product:
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=False)

        inv_det = 1.0 / det
        # J^{-1} = (1/det) * [[c, b], [b, a]]
        J_inv = np.array([[c, b], [b, a]]) * inv_det

        # 各ノード DOF (A0, A1, B0, B1) に対する ∂F/∂u を計算
        for node_idx in range(4):
            rhs = self._compute_rhs(node_idx, delta, dA, dB, s, t)
            # [ds, dt] = -J^{-1} · [rhs1, rhs2]
            st_deriv = -J_inv @ rhs
            ds_du[node_idx * 3 : node_idx * 3 + 3] = st_deriv[0]
            dt_du[node_idx * 3 : node_idx * 3 + 3] = st_deriv[1]

        return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

    @staticmethod
    def _compute_rhs(
        node_idx: int,
        delta: np.ndarray,
        dA: np.ndarray,
        dB: np.ndarray,
        s: float,
        t: float,
    ) -> np.ndarray:
        """ノード node_idx の各方向 d に対する [∂F₁/∂u_d, ∂F₂/∂u_d] を計算.

        Returns:
            (2, 3) array: rhs[eq, dim]
        """
        rhs = np.zeros((2, 3))

        if node_idx == 0:
            # u_A0: ∂δ/∂u_A0 = (1-s)·I, ∂dA/∂u_A0 = -I
            # ∂F₁/∂u_A0 = (1-s)·dA + δ·(-I)  = (1-s)*dA - delta  (per dim)
            # ∂F₂/∂u_A0 = -(1-s)·dB
            rhs[0] = (1.0 - s) * dA - delta
            rhs[1] = -(1.0 - s) * dB
        elif node_idx == 1:
            # u_A1: ∂δ/∂u_A1 = s·I, ∂dA/∂u_A1 = +I
            # ∂F₁/∂u_A1 = s·dA + δ·(+I) = s*dA + delta
            # ∂F₂/∂u_A1 = -s·dB
            rhs[0] = s * dA + delta
            rhs[1] = -s * dB
        elif node_idx == 2:
            # u_B0: ∂δ/∂u_B0 = -(1-t)·I, ∂dB/∂u_B0 = -I
            # ∂F₁/∂u_B0 = -(1-t)·dA
            # ∂F₂/∂u_B0 = (1-t)·dB + δ·(+I) = (1-t)*dB + delta
            rhs[0] = -(1.0 - t) * dA
            rhs[1] = (1.0 - t) * dB + delta
        else:
            # u_B1: ∂δ/∂u_B1 = -t·I, ∂dB/∂u_B1 = +I
            # ∂F₁/∂u_B1 = (-t)·dA  (∂dA/∂u_B1 = 0)
            # ∂F₂/∂u_B1 = t·dB - delta  (∂dB/∂u_B1 = +I → δ·(-I))
            rhs[0] = -t * dA
            rhs[1] = t * dB - delta

        return rhs

    @staticmethod
    def _compute_dt_only(
        delta: np.ndarray,
        dA: np.ndarray,
        dB: np.ndarray,
        s: float,
        t: float,
        c: float,
    ) -> np.ndarray:
        """s クランプ時: dt/du のみ計算（1×1 系）.

        F₂ = -δ · dB = 0 のみ使用。
        ∂F₂/∂t = c, dt/du = -(1/c) · ∂F₂/∂u
        """
        dt_du = np.zeros(12)
        inv_c = 1.0 / c

        # node 0 (A0): ∂F₂/∂u_A0 = -(1-s)·dB
        dt_du[0:3] = inv_c * (1.0 - s) * dB
        # node 1 (A1): ∂F₂/∂u_A1 = -s·dB
        dt_du[3:6] = inv_c * s * dB
        # node 2 (B0): ∂F₂/∂u_B0 = (1-t)·dB + delta
        dt_du[6:9] = -inv_c * ((1.0 - t) * dB + delta)
        # node 3 (B1): ∂F₂/∂u_B1 = t·dB - delta
        dt_du[9:12] = -inv_c * (t * dB - delta)

        return dt_du

    @staticmethod
    def _compute_ds_only(
        delta: np.ndarray,
        dA: np.ndarray,
        dB: np.ndarray,
        s: float,
        t: float,
        a: float,
    ) -> np.ndarray:
        """t クランプ時: ds/du のみ計算（1×1 系）.

        F₁ = δ · dA = 0 のみ使用。
        ∂F₁/∂s = a (> 0), ds/du = -(1/a) · ∂F₁/∂u
        """
        ds_du = np.zeros(12)
        inv_a = 1.0 / a

        # node 0 (A0): ∂F₁/∂u_A0 = (1-s)·dA - delta
        ds_du[0:3] = -inv_a * ((1.0 - s) * dA - delta)
        # node 1 (A1): ∂F₁/∂u_A1 = s·dA + delta
        ds_du[3:6] = -inv_a * (s * dA + delta)
        # node 2 (B0): ∂F₁/∂u_B0 = -(1-t)·dA
        ds_du[6:9] = inv_a * (1.0 - t) * dA
        # node 3 (B1): ∂F₁/∂u_B1 = -t·dA
        ds_du[9:12] = inv_a * t * dA

        return ds_du
