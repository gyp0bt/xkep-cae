"""∂(s,t)/∂u Jacobian — 最近接点パラメータの変位微分.

最近接点条件の陰関数微分により、接触パラメータ (s, t) の
節点変位に対する感度を計算する。

理論（線形セグメント）:
    最近接点条件:
        F₁ = δ · dA = 0
        F₂ = -δ · dB = 0
    ただし δ = pA(s) - pB(t), dA = xA1 - xA0, dB = xB1 - xB0

    陰関数定理: J · [ds, dt]ᵀ = -[∂F₁/∂u, ∂F₂/∂u]ᵀ
    J = [[a, -b], [-b, c]] (a=dA·dA, b=dA·dB, c=dB·dB)

理論（Hermite 曲線 — status-230）:
    最近接点条件:
        F₁ = δ · dpA/ds = 0
        F₂ = -δ · dpB/dt = 0
    ただし δ = pA(s) - pB(t) (Hermite 補間), dpA/ds = Hermite 接線

    Gram 行列: a=dpA·dpA, b=dpA·dpB, c=dpB·dpB
    RHS: Hermite 基底関数の微分を使用（m は凍結近似）

status-078 の旧実装を Process Architecture で再実装。
status-230 で Hermite 幾何対応に拡張。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xkep_cae.core import ProcessMeta, SolverProcess

# ── Hermite スカラーヘルパー ──────────────────────────────────


def _hermite_h00(s: float) -> float:
    """H00(s) = 2s³ - 3s² + 1 — 始点の位置基底."""
    return 2.0 * s * s * s - 3.0 * s * s + 1.0


def _hermite_h01(s: float) -> float:
    """H01(s) = -2s³ + 3s² — 終点の位置基底."""
    return -2.0 * s * s * s + 3.0 * s * s


def _hermite_h10(s: float) -> float:
    """H10(s) = s³ - 2s² + s — 始点の接線基底."""
    return s * s * s - 2.0 * s * s + s


def _hermite_h11(s: float) -> float:
    """H11(s) = s³ - s² — 終点の接線基底."""
    return s * s * s - s * s


def _hermite_dh00(s: float) -> float:
    """H00'(s) = 6s² - 6s."""
    return 6.0 * s * s - 6.0 * s


def _hermite_dh01(s: float) -> float:
    """H01'(s) = -6s² + 6s."""
    return -6.0 * s * s + 6.0 * s


def _hermite_dh10(s: float) -> float:
    """H10'(s) = 3s² - 4s + 1."""
    return 3.0 * s * s - 4.0 * s + 1.0


def _hermite_dh11(s: float) -> float:
    """H11'(s) = 3s² - 2s."""
    return 3.0 * s * s - 2.0 * s


def _hermite_ddh00(s: float) -> float:
    """H00''(s) = 12s - 6."""
    return 12.0 * s - 6.0


def _hermite_ddh01(s: float) -> float:
    """H01''(s) = -12s + 6."""
    return -12.0 * s + 6.0


def _hermite_ddh10(s: float) -> float:
    """H10''(s) = 6s - 4."""
    return 6.0 * s - 4.0


def _hermite_ddh11(s: float) -> float:
    """H11''(s) = 6s - 2."""
    return 6.0 * s - 2.0


def _hermite_second_deriv_scalar(
    s: float,
    x0: np.ndarray,
    x1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
) -> np.ndarray:
    """スカラー版 Hermite 2階微分: d²p/ds² (3,)."""
    return (
        _hermite_ddh00(s) * x0
        + _hermite_ddh10(s) * m0
        + _hermite_ddh01(s) * x1
        + _hermite_ddh11(s) * m1
    )


def _hermite_eval_scalar(
    s: float,
    x0: np.ndarray,
    x1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
) -> np.ndarray:
    """スカラー版 Hermite 補間: s → 位置 (3,)."""
    return _hermite_h00(s) * x0 + _hermite_h10(s) * m0 + _hermite_h01(s) * x1 + _hermite_h11(s) * m1


def _hermite_deriv_scalar(
    s: float,
    x0: np.ndarray,
    x1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
) -> np.ndarray:
    """スカラー版 Hermite 接線: dp/ds (3,)."""
    return (
        _hermite_dh00(s) * x0
        + _hermite_dh10(s) * m0
        + _hermite_dh01(s) * x1
        + _hermite_dh11(s) * m1
    )


# ── Input / Output ───────────────────────────────────────────


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
    # Hermite 用フィールド（status-230）
    mA0: np.ndarray | None = None  # (3,) A始点の接線ベクトル
    mA1: np.ndarray | None = None  # (3,) A終点の接線ベクトル
    mB0: np.ndarray | None = None  # (3,) B始点の接線ベクトル
    mB1: np.ndarray | None = None  # (3,) B終点の接線ベクトル
    use_hermite: bool = False  # True なら Hermite 幾何で計算


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
    status-230 で Hermite 幾何対応を追加。
    """

    meta = ProcessMeta(
        name="ComputeStJacobian",
        module="geometry",
        version="3.0.0",
        document_path="docs/contact_geometry.md",
    )

    # smooth_clip_01 のスカラー微分（C1 遷移重み）
    _SMOOTH_EPS = 0.02

    @staticmethod
    def _smooth_clip_deriv(s_unc: float, epsilon: float = 0.02) -> float:
        """_smooth_clip_01 の s_unc に対する微分.

        ds_smooth/ds_unc:
            s_unc < -ε      → 0
            -ε ≤ s_unc < ε  → (s_unc + ε) / (2ε)
            ε ≤ s_unc ≤ 1-ε → 1
            1-ε < s_unc ≤ 1+ε → (1+ε - s_unc) / (2ε)
            s_unc > 1+ε     → 0
        """
        if s_unc < -epsilon:
            return 0.0
        if s_unc < epsilon:
            return (s_unc + epsilon) / (2.0 * epsilon)
        if s_unc <= 1.0 - epsilon:
            return 1.0
        if s_unc <= 1.0 + epsilon:
            return (1.0 + epsilon - s_unc) / (2.0 * epsilon)
        return 0.0

    def process(self, inp: StJacobianInput) -> StJacobianOutput:
        if inp.use_hermite and inp.mA0 is not None:
            return self._process_hermite(inp)
        return self._process_linear(inp)

    def _process_linear(self, inp: StJacobianInput) -> StJacobianOutput:
        """線形セグメント前提の ∂(s,t)/∂u 計算."""
        dA = inp.xA1 - inp.xA0
        dB = inp.xB1 - inp.xB0
        s = inp.s
        t = inp.t

        # クランプ前の値でスムーズ遷移重みを計算
        s_unc = inp.s_unclamped if inp.s_unclamped is not None else s
        t_unc = inp.t_unclamped if inp.t_unclamped is not None else t
        w_s = self._smooth_clip_deriv(s_unc, self._SMOOTH_EPS)
        w_t = self._smooth_clip_deriv(t_unc, self._SMOOTH_EPS)

        ds_du = np.zeros(12)
        dt_du = np.zeros(12)

        # 両方ゼロ重みなら早期リターン
        if w_s < 1e-30 and w_t < 1e-30:
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

        # Gram 行列の要素
        a = float(np.dot(dA, dA))
        b = float(np.dot(dA, dB))
        c = float(np.dot(dB, dB))
        det = a * c - b * b

        # δ = pA(s) - pB(t)
        delta = (1.0 - s) * inp.xA0 + s * inp.xA1 - (1.0 - t) * inp.xB0 - t * inp.xB1

        # 通常: 2×2 系で計算し、スムーズ重みで減衰
        ac_product = max(a * c, 1e-30)
        if abs(det) < inp.tol_singular * ac_product:
            # 特異の場合は 1×1 フォールバック（重み付き）
            if w_t > 1e-30 and c >= inp.tol_singular:
                dt_du = self._compute_dt_only(delta, dA, dB, s, t, c)
                dt_du *= w_t
            if w_s > 1e-30 and a >= inp.tol_singular:
                ds_du = self._compute_ds_only(delta, dA, dB, s, t, a)
                ds_du *= w_s
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

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

        # スムーズクランプの連鎖律: ds_smooth/du = (ds_smooth/ds_unc) * (ds_unc/du)
        ds_du *= w_s
        dt_du *= w_t

        return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

    def _process_hermite(self, inp: StJacobianInput) -> StJacobianOutput:
        """Hermite 幾何対応の ∂(s,t)/∂u 計算（status-230）.

        最近接点条件（Hermite）:
            F₁ = δ · dpA/ds = 0
            F₂ = -δ · dpB/dt = 0

        ここで dpA/ds は Hermite 接線ベクトル。
        接線ベクトル m は凍結近似（∂m/∂u = 0）。

        RHS 導出:
            ∂F₁/∂u_Ak = H_Ak(s) · dpA + H_Ak'(s) · δ   (A 側ノード)
            ∂F₁/∂u_Bk = -H_Bk(t) · dpA                   (B 側ノード)
            ∂F₂/∂u_Ak = -H_Ak(s) · dpB                    (A 側ノード)
            ∂F₂/∂u_Bk = H_Bk(t) · dpB - H_Bk'(t) · δ    (B 側ノード)

        ここで H_A0=H00(s), H_A1=H01(s), H_B0=H00(t), H_B1=H01(t)。
        """
        s = inp.s
        t = inp.t

        # クランプ前の値でスムーズ遷移重みを計算
        s_unc = inp.s_unclamped if inp.s_unclamped is not None else s
        t_unc = inp.t_unclamped if inp.t_unclamped is not None else t
        w_s = self._smooth_clip_deriv(s_unc, self._SMOOTH_EPS)
        w_t = self._smooth_clip_deriv(t_unc, self._SMOOTH_EPS)

        ds_du = np.zeros(12)
        dt_du = np.zeros(12)

        if w_s < 1e-30 and w_t < 1e-30:
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

        # Hermite 接線ベクトル
        dpA = _hermite_deriv_scalar(s, inp.xA0, inp.xA1, inp.mA0, inp.mA1)
        dpB = _hermite_deriv_scalar(t, inp.xB0, inp.xB1, inp.mB0, inp.mB1)

        # δ = pA(s) - pB(t)（Hermite 補間）
        delta = _hermite_eval_scalar(s, inp.xA0, inp.xA1, inp.mA0, inp.mA1) - _hermite_eval_scalar(
            t, inp.xB0, inp.xB1, inp.mB0, inp.mB1
        )

        # 完全 Jacobian（Gauss-Newton + 2階微分項）
        # ∂F₁/∂s = dpA·dpA + δ·d²pA/ds², ∂F₂/∂t = dpB·dpB - δ·d²pB/dt²
        d2pA = _hermite_second_deriv_scalar(s, inp.xA0, inp.xA1, inp.mA0, inp.mA1)
        d2pB = _hermite_second_deriv_scalar(t, inp.xB0, inp.xB1, inp.mB0, inp.mB1)
        a = float(np.dot(dpA, dpA) + np.dot(delta, d2pA))
        b = float(np.dot(dpA, dpB))
        c = float(np.dot(dpB, dpB) - np.dot(delta, d2pB))
        det = a * c - b * b

        ac_product = max(abs(a * c), 1e-30)
        if abs(det) < inp.tol_singular * ac_product:
            # 特異: 1×1 フォールバック
            if w_t > 1e-30 and c >= inp.tol_singular:
                dt_du = self._compute_dt_only_hermite(delta, dpA, dpB, s, t, c)
                dt_du *= w_t
            if w_s > 1e-30 and a >= inp.tol_singular:
                ds_du = self._compute_ds_only_hermite(delta, dpA, dpB, s, t, a)
                ds_du *= w_s
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

        inv_det = 1.0 / det
        J_inv = np.array([[c, b], [b, a]]) * inv_det

        for node_idx in range(4):
            rhs = self._compute_rhs_hermite(node_idx, delta, dpA, dpB, s, t)
            st_deriv = -J_inv @ rhs
            ds_du[node_idx * 3 : node_idx * 3 + 3] = st_deriv[0]
            dt_du[node_idx * 3 : node_idx * 3 + 3] = st_deriv[1]

        ds_du *= w_s
        dt_du *= w_t

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
        """線形版: ノード node_idx の [∂F₁/∂u_d, ∂F₂/∂u_d] を計算.

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
    def _compute_rhs_hermite(
        node_idx: int,
        delta: np.ndarray,
        dpA: np.ndarray,
        dpB: np.ndarray,
        s: float,
        t: float,
    ) -> np.ndarray:
        """Hermite 版: ノード node_idx の [∂F₁/∂u_d, ∂F₂/∂u_d] を計算.

        m（接線ベクトル）は凍結近似。
        A 側ノードの位置基底: H_A0=H00(s), H_A1=H01(s)
        B 側ノードの位置基底: H_B0=H00(t), H_B1=H01(t)

        ∂F₁/∂u_Ak = H_Ak(s) · dpA + H_Ak'(s) · δ
        ∂F₁/∂u_Bk = -H_Bk(t) · dpA
        ∂F₂/∂u_Ak = -H_Ak(s) · dpB
        ∂F₂/∂u_Bk = H_Bk(t) · dpB - H_Bk'(t) · δ

        Returns:
            (2, 3) array: rhs[eq, dim]
        """
        rhs = np.zeros((2, 3))

        if node_idx == 0:
            # A0: h=H00(s), dh=H00'(s)
            h = _hermite_h00(s)
            dh = _hermite_dh00(s)
            rhs[0] = h * dpA + dh * delta
            rhs[1] = -h * dpB
        elif node_idx == 1:
            # A1: h=H01(s), dh=H01'(s)
            h = _hermite_h01(s)
            dh = _hermite_dh01(s)
            rhs[0] = h * dpA + dh * delta
            rhs[1] = -h * dpB
        elif node_idx == 2:
            # B0: h=H00(t), dh=H00'(t)
            h = _hermite_h00(t)
            dh = _hermite_dh00(t)
            rhs[0] = -h * dpA
            rhs[1] = h * dpB - dh * delta
        else:
            # B1: h=H01(t), dh=H01'(t)
            h = _hermite_h01(t)
            dh = _hermite_dh01(t)
            rhs[0] = -h * dpA
            rhs[1] = h * dpB - dh * delta

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
        """線形版: s クランプ時 dt/du のみ計算（1×1 系）."""
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
        """線形版: t クランプ時 ds/du のみ計算（1×1 系）."""
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

    @staticmethod
    def _compute_dt_only_hermite(
        delta: np.ndarray,
        dpA: np.ndarray,
        dpB: np.ndarray,
        s: float,
        t: float,
        c: float,
    ) -> np.ndarray:
        """Hermite 版: s クランプ時 dt/du のみ計算（1×1 系）."""
        dt_du = np.zeros(12)
        inv_c = 1.0 / c

        # A0: ∂F₂/∂u_A0 = -H00(s)·dpB
        dt_du[0:3] = inv_c * _hermite_h00(s) * dpB
        # A1: ∂F₂/∂u_A1 = -H01(s)·dpB
        dt_du[3:6] = inv_c * _hermite_h01(s) * dpB
        # B0: ∂F₂/∂u_B0 = H00(t)·dpB - H00'(t)·δ
        dt_du[6:9] = -inv_c * (_hermite_h00(t) * dpB - _hermite_dh00(t) * delta)
        # B1: ∂F₂/∂u_B1 = H01(t)·dpB - H01'(t)·δ
        dt_du[9:12] = -inv_c * (_hermite_h01(t) * dpB - _hermite_dh01(t) * delta)

        return dt_du

    @staticmethod
    def _compute_ds_only_hermite(
        delta: np.ndarray,
        dpA: np.ndarray,
        dpB: np.ndarray,
        s: float,
        t: float,
        a: float,
    ) -> np.ndarray:
        """Hermite 版: t クランプ時 ds/du のみ計算（1×1 系）."""
        ds_du = np.zeros(12)
        inv_a = 1.0 / a

        # A0: ∂F₁/∂u_A0 = H00(s)·dpA + H00'(s)·δ
        ds_du[0:3] = -inv_a * (_hermite_h00(s) * dpA + _hermite_dh00(s) * delta)
        # A1: ∂F₁/∂u_A1 = H01(s)·dpA + H01'(s)·δ
        ds_du[3:6] = -inv_a * (_hermite_h01(s) * dpA + _hermite_dh01(s) * delta)
        # B0: ∂F₁/∂u_B0 = -H00(t)·dpA
        ds_du[6:9] = inv_a * _hermite_h00(t) * dpA
        # B1: ∂F₁/∂u_B1 = -H01(t)·dpA
        ds_du[9:12] = inv_a * _hermite_h01(t) * dpA

        return ds_du
