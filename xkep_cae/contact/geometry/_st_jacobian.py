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
    RHS: Hermite 基底関数の微分を使用
    dm_A/dm_B 指定時は ∂m/∂u のローカル寄与を含む（status-243）

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
    # frozen-m 解消用: ∂m/∂x のローカル係数（status-243）
    dm_A: np.ndarray | None = None  # (2,2) [[∂mA0/∂xA0, ∂mA0/∂xA1], [∂mA1/∂xA0, ∂mA1/∂xA1]]
    dm_B: np.ndarray | None = None  # (2,2) [[∂mB0/∂xB0, ∂mB0/∂xB1], [∂mB1/∂xB0, ∂mB1/∂xB1]]
    # Hermite 非局所: 隣接ノードの ∂m/∂x 係数（status-271）
    dm_ext_A: np.ndarray | None = None  # (2,) [∂mA0/∂x_{A-1}, ∂mA1/∂x_{A+2}]
    dm_ext_B: np.ndarray | None = None  # (2,) [∂mB0/∂x_{B-1}, ∂mB1/∂x_{B+2}]


@dataclass(frozen=True)
class StJacobianOutput:
    """∂(s,t)/∂u 計算の出力."""

    ds_du: np.ndarray  # (12,) ds/du（4ノード × 3次元）
    dt_du: np.ndarray  # (12,) dt/du（4ノード × 3次元）
    valid: bool  # 計算が有効か（平行特異でなければ True）
    # Hermite 非局所: 隣接ノードの ∂(s,t)/∂u（status-271）
    # レイアウト: [A-1_xyz(3), A+2_xyz(3), B-1_xyz(3), B+2_xyz(3)]
    ds_du_adj: np.ndarray | None = None  # (12,) or None
    dt_du_adj: np.ndarray | None = None  # (12,) or None


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

        # クランプ前の値でスムーズ遷移重みを計算
        s_unc = inp.s_unclamped if inp.s_unclamped is not None else inp.s
        t_unc = inp.t_unclamped if inp.t_unclamped is not None else inp.t
        w_s = self._smooth_clip_deriv(s_unc, self._SMOOTH_EPS)
        w_t = self._smooth_clip_deriv(t_unc, self._SMOOTH_EPS)

        ds_du = np.zeros(12)
        dt_du = np.zeros(12)

        # 両方ゼロ重みなら早期リターン
        if w_s < 1e-30 and w_t < 1e-30:
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

        # status-293: IFT 幾何を unclamped (s_unc, t_unc) で評価
        # F₁(s_unc, t_unc, u) = 0 の陰関数微分は unclamped 点での幾何を使う
        s = s_unc
        t = t_unc

        # Gram 行列の要素
        a = float(np.dot(dA, dA))
        b = float(np.dot(dA, dB))
        c = float(np.dot(dB, dB))
        det = a * c - b * b

        # δ = pA(s_unc) - pB(t_unc)
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

        # status-293: smooth blending — 1×1系と2×2系をw_t/w_sで連続補間
        # ds_du = w_t * ds_du_2x2 + (1 - w_t) * ds_du_1x1（tクランプ度でblend）
        # dt_du = w_s * dt_du_2x2 + (1 - w_s) * dt_du_1x1（sクランプ度でblend）
        # w_s = w_t = 1（内部接触点）では2×2のみで計算（高速パス）

        if w_s >= 1.0 - 1e-10 and w_t >= 1.0 - 1e-10:
            # 高速パス: 両方完全有効 → 2×2系のみ
            J_inv = np.array([[c, b], [b, a]]) * inv_det
            for node_idx in range(4):
                rhs = self._compute_rhs(node_idx, delta, dA, dB, s, t)
                st_deriv = -J_inv @ rhs
                ds_du[node_idx * 3 : node_idx * 3 + 3] = st_deriv[0]
                dt_du[node_idx * 3 : node_idx * 3 + 3] = st_deriv[1]
            ds_du *= w_s
            dt_du *= w_t
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

        # 1×1系の結果を計算
        ds_du_1x1 = np.zeros(12)
        dt_du_1x1 = np.zeros(12)
        if w_s > 1e-30 and a >= inp.tol_singular:
            ds_du_1x1 = self._compute_ds_only(delta, dA, dB, s, t, a)
            ds_du_1x1 *= w_s
        if w_t > 1e-30 and c >= inp.tol_singular:
            dt_du_1x1 = self._compute_dt_only(delta, dA, dB, s, t, c)
            dt_du_1x1 *= w_t

        # 2×2系の結果を計算
        ds_du_2x2 = np.zeros(12)
        dt_du_2x2 = np.zeros(12)
        J_inv = np.array([[c, b], [b, a]]) * inv_det
        for node_idx in range(4):
            rhs = self._compute_rhs(node_idx, delta, dA, dB, s, t)
            st_deriv = -J_inv @ rhs
            ds_du_2x2[node_idx * 3 : node_idx * 3 + 3] = st_deriv[0]
            dt_du_2x2[node_idx * 3 : node_idx * 3 + 3] = st_deriv[1]
        ds_du_2x2 *= w_s
        dt_du_2x2 *= w_t

        # smooth blend: w_t でds_duを補間、w_s でdt_duを補間
        ds_du = w_t * ds_du_2x2 + (1.0 - w_t) * ds_du_1x1
        dt_du = w_s * dt_du_2x2 + (1.0 - w_s) * dt_du_1x1

        return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

    def _process_hermite(self, inp: StJacobianInput) -> StJacobianOutput:
        """Hermite 幾何対応の ∂(s,t)/∂u 計算（status-230）.

        最近接点条件（Hermite）:
            F₁ = δ · dpA/ds = 0
            F₂ = -δ · dpB/dt = 0

        ここで dpA/ds は Hermite 接線ベクトル。
        接線ベクトル m は凍結近似（∂m/∂u = 0）。

        RHS 導出（dm_A/dm_B 指定時は ∂m/∂u 補正付き — status-243）:
            h_eff = H_Ak(s) + H10(s)·dm_A[0,k] + H11(s)·dm_A[1,k]
            dh_eff = H_Ak'(s) + H10'(s)·dm_A[0,k] + H11'(s)·dm_A[1,k]
            ∂F₁/∂u_Ak = h_eff · dpA + dh_eff · δ   (A 側ノード)
            ∂F₁/∂u_Bk = -h_eff · dpA                 (B 側ノード)

        dm_A=None 時は H_Ak(s), H_Ak'(s) のみ（従来の凍結近似）。
        """
        # クランプ前の値でスムーズ遷移重みを計算
        s_unc = inp.s_unclamped if inp.s_unclamped is not None else inp.s
        t_unc = inp.t_unclamped if inp.t_unclamped is not None else inp.t
        w_s = self._smooth_clip_deriv(s_unc, self._SMOOTH_EPS)
        w_t = self._smooth_clip_deriv(t_unc, self._SMOOTH_EPS)

        ds_du = np.zeros(12)
        dt_du = np.zeros(12)

        if w_s < 1e-30 and w_t < 1e-30:
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

        # status-293: IFT 幾何を unclamped (s_unc, t_unc) で評価
        s = s_unc
        t = t_unc

        # Hermite 接線ベクトル
        dpA = _hermite_deriv_scalar(s, inp.xA0, inp.xA1, inp.mA0, inp.mA1)
        dpB = _hermite_deriv_scalar(t, inp.xB0, inp.xB1, inp.mB0, inp.mB1)

        # δ = pA(s_unc) - pB(t_unc)（Hermite 補間）
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
                dt_du = self._compute_dt_only_hermite(
                    delta,
                    dpA,
                    dpB,
                    s,
                    t,
                    c,
                    dm_A=inp.dm_A,
                    dm_B=inp.dm_B,
                )
                dt_du *= w_t
            if w_s > 1e-30 and a >= inp.tol_singular:
                ds_du = self._compute_ds_only_hermite(
                    delta,
                    dpA,
                    dpB,
                    s,
                    t,
                    a,
                    dm_A=inp.dm_A,
                    dm_B=inp.dm_B,
                )
                ds_du *= w_s
            return StJacobianOutput(ds_du=ds_du, dt_du=dt_du, valid=True)

        inv_det = 1.0 / det

        # status-293: smooth blending — 1×1系と2×2系をw_t/w_sで連続補間
        J_inv = np.array([[c, b], [b, a]]) * inv_det

        if w_s >= 1.0 - 1e-10 and w_t >= 1.0 - 1e-10:
            # 高速パス: 両方完全有効 → 2×2系のみ
            for node_idx in range(4):
                rhs = self._compute_rhs_hermite(
                    node_idx,
                    delta,
                    dpA,
                    dpB,
                    s,
                    t,
                    dm_A=inp.dm_A,
                    dm_B=inp.dm_B,
                )
                st_deriv = -J_inv @ rhs
                ds_du[node_idx * 3 : node_idx * 3 + 3] = st_deriv[0]
                dt_du[node_idx * 3 : node_idx * 3 + 3] = st_deriv[1]
            ds_du *= w_s
            dt_du *= w_t
        else:
            # 1×1系の結果
            ds_du_1x1 = np.zeros(12)
            dt_du_1x1 = np.zeros(12)
            if w_s > 1e-30 and a >= inp.tol_singular:
                ds_du_1x1 = self._compute_ds_only_hermite(
                    delta,
                    dpA,
                    dpB,
                    s,
                    t,
                    a,
                    dm_A=inp.dm_A,
                    dm_B=inp.dm_B,
                )
                ds_du_1x1 *= w_s
            if w_t > 1e-30 and c >= inp.tol_singular:
                dt_du_1x1 = self._compute_dt_only_hermite(
                    delta,
                    dpA,
                    dpB,
                    s,
                    t,
                    c,
                    dm_A=inp.dm_A,
                    dm_B=inp.dm_B,
                )
                dt_du_1x1 *= w_t

            # 2×2系の結果
            ds_du_2x2 = np.zeros(12)
            dt_du_2x2 = np.zeros(12)
            for node_idx in range(4):
                rhs = self._compute_rhs_hermite(
                    node_idx,
                    delta,
                    dpA,
                    dpB,
                    s,
                    t,
                    dm_A=inp.dm_A,
                    dm_B=inp.dm_B,
                )
                st_deriv = -J_inv @ rhs
                ds_du_2x2[node_idx * 3 : node_idx * 3 + 3] = st_deriv[0]
                dt_du_2x2[node_idx * 3 : node_idx * 3 + 3] = st_deriv[1]
            ds_du_2x2 *= w_s
            dt_du_2x2 *= w_t

            # smooth blend
            ds_du = w_t * ds_du_2x2 + (1.0 - w_t) * ds_du_1x1
            dt_du = w_s * dt_du_2x2 + (1.0 - w_s) * dt_du_1x1

        # Hermite 非局所: 隣接ノード微分（status-271）
        ds_du_adj = None
        dt_du_adj = None
        if inp.dm_ext_A is not None and inp.dm_ext_B is not None:
            ds_du_adj = np.zeros(12)
            dt_du_adj = np.zeros(12)
            inv_a_safe = 1.0 / a if a >= inp.tol_singular else 0.0
            inv_c_safe = 1.0 / c if c >= inp.tol_singular else 0.0
            # 4隣接ノード: [A-1, A+2, B-1, B+2]
            for adj_idx, (dm_coeff, side, m_slot) in enumerate(
                [
                    (inp.dm_ext_A[0], "A", 0),  # A-1 → ∂mA0/∂x_{A-1}
                    (inp.dm_ext_A[1], "A", 1),  # A+2 → ∂mA1/∂x_{A+2}
                    (inp.dm_ext_B[0], "B", 0),  # B-1 → ∂mB0/∂x_{B-1}
                    (inp.dm_ext_B[1], "B", 1),  # B+2 → ∂mB1/∂x_{B+2}
                ]
            ):
                if abs(dm_coeff) < 1e-30:
                    continue
                rhs = self._compute_rhs_hermite_neighbor(
                    delta, dpA, dpB, s, t, side, m_slot, dm_coeff
                )
                # 2×2系
                st_2x2 = -J_inv @ rhs
                ds_adj_2x2 = st_2x2[0] * w_s
                dt_adj_2x2 = st_2x2[1] * w_t
                if w_s >= 1.0 - 1e-10 and w_t >= 1.0 - 1e-10:
                    ds_du_adj[adj_idx * 3 : adj_idx * 3 + 3] = ds_adj_2x2
                    dt_du_adj[adj_idx * 3 : adj_idx * 3 + 3] = dt_adj_2x2
                else:
                    # 1×1系: ds = -(1/a)*rhs[0], dt = -(1/c)*rhs[1]
                    ds_adj_1x1 = -inv_a_safe * rhs[0] * w_s
                    dt_adj_1x1 = -inv_c_safe * rhs[1] * w_t
                    ds_du_adj[adj_idx * 3 : adj_idx * 3 + 3] = (
                        w_t * ds_adj_2x2 + (1.0 - w_t) * ds_adj_1x1
                    )
                    dt_du_adj[adj_idx * 3 : adj_idx * 3 + 3] = (
                        w_s * dt_adj_2x2 + (1.0 - w_s) * dt_adj_1x1
                    )

        return StJacobianOutput(
            ds_du=ds_du,
            dt_du=dt_du,
            valid=True,
            ds_du_adj=ds_du_adj,
            dt_du_adj=dt_du_adj,
        )

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
        *,
        dm_A: np.ndarray | None = None,
        dm_B: np.ndarray | None = None,
    ) -> np.ndarray:
        """Hermite 版: ノード node_idx の [∂F₁/∂u_d, ∂F₂/∂u_d] を計算.

        dm_A/dm_B が指定された場合、∂m/∂u のローカル寄与を含む（status-243）。
        None の場合は従来の凍結近似（∂m/∂u = 0）。

        位置感度:
            ∂pA/∂x_{Ak} = H_Ak(s)·I + H10(s)·∂mA0/∂x_{Ak}·I + H11(s)·∂mA1/∂x_{Ak}·I
            → h_eff = H_Ak(s) + H10(s)*dm_A[0,k] + H11(s)*dm_A[1,k]

        接線感度:
            ∂(dpA/ds)/∂x_{Ak} = H_Ak'(s)·I + H10'(s)·∂mA0/∂x_{Ak}·I + H11'(s)·∂mA1/∂x_{Ak}·I
            → dh_eff = H_Ak'(s) + H10'(s)*dm_A[0,k] + H11'(s)*dm_A[1,k]

        Returns:
            (2, 3) array: rhs[eq, dim]
        """
        rhs = np.zeros((2, 3))

        if node_idx == 0:
            # A0: h=H00(s), dh=H00'(s)
            h = _hermite_h00(s)
            dh = _hermite_dh00(s)
            if dm_A is not None:
                h += _hermite_h10(s) * dm_A[0, 0] + _hermite_h11(s) * dm_A[1, 0]
                dh += _hermite_dh10(s) * dm_A[0, 0] + _hermite_dh11(s) * dm_A[1, 0]
            rhs[0] = h * dpA + dh * delta
            rhs[1] = -h * dpB
        elif node_idx == 1:
            # A1: h=H01(s), dh=H01'(s)
            h = _hermite_h01(s)
            dh = _hermite_dh01(s)
            if dm_A is not None:
                h += _hermite_h10(s) * dm_A[0, 1] + _hermite_h11(s) * dm_A[1, 1]
                dh += _hermite_dh10(s) * dm_A[0, 1] + _hermite_dh11(s) * dm_A[1, 1]
            rhs[0] = h * dpA + dh * delta
            rhs[1] = -h * dpB
        elif node_idx == 2:
            # B0: h=H00(t), dh=H00'(t)
            h = _hermite_h00(t)
            dh = _hermite_dh00(t)
            if dm_B is not None:
                h += _hermite_h10(t) * dm_B[0, 0] + _hermite_h11(t) * dm_B[1, 0]
                dh += _hermite_dh10(t) * dm_B[0, 0] + _hermite_dh11(t) * dm_B[1, 0]
            rhs[0] = -h * dpA
            rhs[1] = h * dpB - dh * delta
        else:
            # B1: h=H01(t), dh=H01'(t)
            h = _hermite_h01(t)
            dh = _hermite_dh01(t)
            if dm_B is not None:
                h += _hermite_h10(t) * dm_B[0, 1] + _hermite_h11(t) * dm_B[1, 1]
                dh += _hermite_dh10(t) * dm_B[0, 1] + _hermite_dh11(t) * dm_B[1, 1]
            rhs[0] = -h * dpA
            rhs[1] = h * dpB - dh * delta

        return rhs

    @staticmethod
    def _compute_rhs_hermite_neighbor(
        delta: np.ndarray,
        dpA: np.ndarray,
        dpB: np.ndarray,
        s: float,
        t: float,
        side: str,
        m_slot: int,
        dm_coeff: float,
    ) -> np.ndarray:
        """Hermite 非局所: 隣接ノードの [∂F₁/∂u_d, ∂F₂/∂u_d] を計算.

        隣接ノード x_adj は m_{side}[m_slot] にのみ影響する。
        ∂m/∂x_adj = dm_coeff * I₃ なので:

        A 側 (side="A"):
            ∂pA/∂x_adj = H1{m_slot}(s) * dm_coeff * I
            ∂(dpA/ds)/∂x_adj = H1{m_slot}'(s) * dm_coeff * I
            → h_eff = H1{m_slot}(s) * dm_coeff
            → dh_eff = H1{m_slot}'(s) * dm_coeff
            rhs[0] = h_eff * dpA + dh_eff * delta
            rhs[1] = -h_eff * dpB

        B 側 (side="B"):
            ∂pB/∂x_adj = H1{m_slot}(t) * dm_coeff * I
            rhs[0] = -h_eff * dpA
            rhs[1] = h_eff * dpB - dh_eff * delta

        Parameters:
            side: "A" or "B"
            m_slot: 0 (始点接線 m0) or 1 (終点接線 m1)
            dm_coeff: ∂m/∂x_adj のスカラー係数
        """
        rhs = np.zeros((2, 3))

        if side == "A":
            if m_slot == 0:
                h = _hermite_h10(s) * dm_coeff
                dh = _hermite_dh10(s) * dm_coeff
            else:
                h = _hermite_h11(s) * dm_coeff
                dh = _hermite_dh11(s) * dm_coeff
            rhs[0] = h * dpA + dh * delta
            rhs[1] = -h * dpB
        else:  # side == "B"
            if m_slot == 0:
                h = _hermite_h10(t) * dm_coeff
                dh = _hermite_dh10(t) * dm_coeff
            else:
                h = _hermite_h11(t) * dm_coeff
                dh = _hermite_dh11(t) * dm_coeff
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
        *,
        dm_A: np.ndarray | None = None,
        dm_B: np.ndarray | None = None,
    ) -> np.ndarray:
        """Hermite 版: s クランプ時 dt/du のみ計算（1×1 系）."""
        dt_du = np.zeros(12)
        inv_c = 1.0 / c

        # A0: ∂F₂/∂u_A0 = -h_eff·dpB
        h_a0 = _hermite_h00(s)
        if dm_A is not None:
            h_a0 += _hermite_h10(s) * dm_A[0, 0] + _hermite_h11(s) * dm_A[1, 0]
        dt_du[0:3] = inv_c * h_a0 * dpB
        # A1: ∂F₂/∂u_A1 = -h_eff·dpB
        h_a1 = _hermite_h01(s)
        if dm_A is not None:
            h_a1 += _hermite_h10(s) * dm_A[0, 1] + _hermite_h11(s) * dm_A[1, 1]
        dt_du[3:6] = inv_c * h_a1 * dpB
        # B0: ∂F₂/∂u_B0 = h_eff·dpB - dh_eff·δ
        h_b0 = _hermite_h00(t)
        dh_b0 = _hermite_dh00(t)
        if dm_B is not None:
            h_b0 += _hermite_h10(t) * dm_B[0, 0] + _hermite_h11(t) * dm_B[1, 0]
            dh_b0 += _hermite_dh10(t) * dm_B[0, 0] + _hermite_dh11(t) * dm_B[1, 0]
        dt_du[6:9] = -inv_c * (h_b0 * dpB - dh_b0 * delta)
        # B1: ∂F₂/∂u_B1 = h_eff·dpB - dh_eff·δ
        h_b1 = _hermite_h01(t)
        dh_b1 = _hermite_dh01(t)
        if dm_B is not None:
            h_b1 += _hermite_h10(t) * dm_B[0, 1] + _hermite_h11(t) * dm_B[1, 1]
            dh_b1 += _hermite_dh10(t) * dm_B[0, 1] + _hermite_dh11(t) * dm_B[1, 1]
        dt_du[9:12] = -inv_c * (h_b1 * dpB - dh_b1 * delta)

        return dt_du

    @staticmethod
    def _compute_ds_only_hermite(
        delta: np.ndarray,
        dpA: np.ndarray,
        dpB: np.ndarray,
        s: float,
        t: float,
        a: float,
        *,
        dm_A: np.ndarray | None = None,
        dm_B: np.ndarray | None = None,
    ) -> np.ndarray:
        """Hermite 版: t クランプ時 ds/du のみ計算（1×1 系）."""
        ds_du = np.zeros(12)
        inv_a = 1.0 / a

        # A0: ∂F₁/∂u_A0 = h_eff·dpA + dh_eff·δ
        h_a0 = _hermite_h00(s)
        dh_a0 = _hermite_dh00(s)
        if dm_A is not None:
            h_a0 += _hermite_h10(s) * dm_A[0, 0] + _hermite_h11(s) * dm_A[1, 0]
            dh_a0 += _hermite_dh10(s) * dm_A[0, 0] + _hermite_dh11(s) * dm_A[1, 0]
        ds_du[0:3] = -inv_a * (h_a0 * dpA + dh_a0 * delta)
        # A1: ∂F₁/∂u_A1 = h_eff·dpA + dh_eff·δ
        h_a1 = _hermite_h01(s)
        dh_a1 = _hermite_dh01(s)
        if dm_A is not None:
            h_a1 += _hermite_h10(s) * dm_A[0, 1] + _hermite_h11(s) * dm_A[1, 1]
            dh_a1 += _hermite_dh10(s) * dm_A[0, 1] + _hermite_dh11(s) * dm_A[1, 1]
        ds_du[3:6] = -inv_a * (h_a1 * dpA + dh_a1 * delta)
        # B0: ∂F₁/∂u_B0 = -h_eff·dpA
        h_b0 = _hermite_h00(t)
        if dm_B is not None:
            h_b0 += _hermite_h10(t) * dm_B[0, 0] + _hermite_h11(t) * dm_B[1, 0]
        ds_du[6:9] = inv_a * h_b0 * dpA
        # B1: ∂F₁/∂u_B1 = -h_eff·dpA
        h_b1 = _hermite_h01(t)
        if dm_B is not None:
            h_b1 += _hermite_h10(t) * dm_B[0, 1] + _hermite_h11(t) * dm_B[1, 1]
        ds_du[9:12] = inv_a * h_b1 * dpA

        return ds_du


# ── バッチ版 StJacobian（status-309: ベクトル化高速化） ──────


def _smooth_clip_deriv_batch(s_unc: np.ndarray, epsilon: float = 0.02) -> np.ndarray:
    """_smooth_clip_01 のバッチ微分（ベクトル化版）.

    Returns:
        (N,) 各ペアの smooth clip 重み
    """
    result = np.where(
        s_unc < -epsilon,
        0.0,
        np.where(
            s_unc < epsilon,
            (s_unc + epsilon) / (2.0 * epsilon),
            np.where(
                s_unc <= 1.0 - epsilon,
                1.0,
                np.where(s_unc <= 1.0 + epsilon, (1.0 + epsilon - s_unc) / (2.0 * epsilon), 0.0),
            ),
        ),
    )
    return result


def _batch_st_jacobian_linear(
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    s: np.ndarray,
    t: np.ndarray,
    s_unc: np.ndarray,
    t_unc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """線形セグメントの ∂(s,t)/∂u をバッチ計算.

    Args:
        xA0, xA1, xB0, xB1: (N, 3) セグメント端点座標
        s, t: (N,) クランプ済みパラメータ
        s_unc, t_unc: (N,) クランプ前パラメータ

    Returns:
        ds_du: (N, 12) ds/du
        dt_du: (N, 12) dt/du
        valid: (N,) bool 有効フラグ
    """
    N = len(s)
    ds_du = np.zeros((N, 12))
    dt_du = np.zeros((N, 12))
    valid = np.ones(N, dtype=bool)

    if N == 0:
        return ds_du, dt_du, valid

    # Smooth clip 重み
    w_s = _smooth_clip_deriv_batch(s_unc)
    w_t = _smooth_clip_deriv_batch(t_unc)

    # 両方ゼロ重みは早期リターン済み（valid=True, ds_du=dt_du=0）
    active = (w_s > 1e-30) | (w_t > 1e-30)
    if not np.any(active):
        return ds_du, dt_du, valid

    # IFT 幾何を unclamped で評価（status-293）
    s_eval = s_unc
    t_eval = t_unc

    dA = xA1 - xA0  # (N, 3)
    dB = xB1 - xB0  # (N, 3)

    # Gram 行列
    a = np.sum(dA * dA, axis=1)  # (N,)
    b_gram = np.sum(dA * dB, axis=1)  # (N,)
    c = np.sum(dB * dB, axis=1)  # (N,)
    det = a * c - b_gram * b_gram  # (N,)

    # delta = pA(s_unc) - pB(t_unc)
    delta = (
        (1.0 - s_eval)[:, None] * xA0
        + s_eval[:, None] * xA1
        - (1.0 - t_eval)[:, None] * xB0
        - t_eval[:, None] * xB1
    )  # (N, 3)

    tol = 1e-10
    ac_product = np.maximum(a * c, 1e-30)
    non_singular = np.abs(det) >= tol * ac_product

    # ── 高速パス: w_s≈1, w_t≈1, 非特異 ──
    fast = active & (w_s >= 1.0 - 1e-10) & (w_t >= 1.0 - 1e-10) & non_singular
    n_fast = int(np.sum(fast))

    if n_fast > 0:
        inv_det_f = 1.0 / det[fast]  # (Nf,)
        dA_f = dA[fast]
        dB_f = dB[fast]
        delta_f = delta[fast]
        s_f = s_eval[fast]
        t_f = t_eval[fast]
        a_f = a[fast]
        b_f = b_gram[fast]
        c_f = c[fast]
        w_s_f = w_s[fast]
        w_t_f = w_t[fast]

        # 4ノードの RHS をバッチ計算: rhs[node] = (Nf, 2, 3)
        # Node 0 (A0): rhs0 = (1-s)*dA - delta, -(1-s)*dB
        rhs0_F1 = (1.0 - s_f)[:, None] * dA_f - delta_f  # (Nf, 3)
        rhs0_F2 = -(1.0 - s_f)[:, None] * dB_f

        # Node 1 (A1): s*dA + delta, -s*dB
        rhs1_F1 = s_f[:, None] * dA_f + delta_f
        rhs1_F2 = -s_f[:, None] * dB_f

        # Node 2 (B0): -(1-t)*dA, (1-t)*dB + delta
        rhs2_F1 = -(1.0 - t_f)[:, None] * dA_f
        rhs2_F2 = (1.0 - t_f)[:, None] * dB_f + delta_f

        # Node 3 (B1): -t*dA, t*dB - delta
        rhs3_F1 = -t_f[:, None] * dA_f
        rhs3_F2 = t_f[:, None] * dB_f - delta_f

        # st_deriv = -J_inv @ rhs, J_inv = [[c, b], [b, a]] / det
        # ds = -(c * rhs_F1 + b * rhs_F2) / det
        # dt = -(b * rhs_F1 + a * rhs_F2) / det
        ds_du_f = np.zeros((n_fast, 12))
        dt_du_f = np.zeros((n_fast, 12))

        for node_idx, (rF1, rF2) in enumerate(
            [(rhs0_F1, rhs0_F2), (rhs1_F1, rhs1_F2), (rhs2_F1, rhs2_F2), (rhs3_F1, rhs3_F2)]
        ):
            ds_du_f[:, node_idx * 3 : node_idx * 3 + 3] = (
                -(c_f[:, None] * rF1 + b_f[:, None] * rF2) * inv_det_f[:, None]
            )
            dt_du_f[:, node_idx * 3 : node_idx * 3 + 3] = (
                -(b_f[:, None] * rF1 + a_f[:, None] * rF2) * inv_det_f[:, None]
            )

        ds_du_f *= w_s_f[:, None]
        dt_du_f *= w_t_f[:, None]
        ds_du[fast] = ds_du_f
        dt_du[fast] = dt_du_f

    # ── 低速パス: エッジケース（特異 or クランプ境界） ──
    slow = active & ~fast
    if np.any(slow):
        slow_idx = np.where(slow)[0]
        proc = ComputeStJacobianProcess()
        for idx in slow_idx:
            out = proc.process(
                StJacobianInput(
                    xA0=xA0[idx],
                    xA1=xA1[idx],
                    xB0=xB0[idx],
                    xB1=xB1[idx],
                    s=float(s[idx]),
                    t=float(t[idx]),
                    s_unclamped=float(s_unc[idx]),
                    t_unclamped=float(t_unc[idx]),
                )
            )
            ds_du[idx] = out.ds_du
            dt_du[idx] = out.dt_du
            valid[idx] = out.valid

    return ds_du, dt_du, valid


def _batch_st_jacobian_hermite(
    xA0: np.ndarray,
    xA1: np.ndarray,
    xB0: np.ndarray,
    xB1: np.ndarray,
    s: np.ndarray,
    t: np.ndarray,
    s_unc: np.ndarray,
    t_unc: np.ndarray,
    mA0: np.ndarray,
    mA1: np.ndarray,
    mB0: np.ndarray,
    mB1: np.ndarray,
    dm_A: np.ndarray | None = None,
    dm_B: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hermite 幾何の ∂(s,t)/∂u をバッチ計算.

    Args:
        xA0..xB1: (N, 3) セグメント端点座標
        s, t: (N,) クランプ済みパラメータ
        s_unc, t_unc: (N,) クランプ前パラメータ
        mA0..mB1: (N, 3) ノード接線ベクトル
        dm_A: (N, 2, 2) or None — ∂m/∂x ローカル係数
        dm_B: (N, 2, 2) or None — ∂m/∂x ローカル係数

    Returns:
        ds_du: (N, 12) ds/du
        dt_du: (N, 12) dt/du
        valid: (N,) bool 有効フラグ
    """
    N = len(s)
    ds_du = np.zeros((N, 12))
    dt_du = np.zeros((N, 12))
    valid = np.ones(N, dtype=bool)

    if N == 0:
        return ds_du, dt_du, valid

    w_s = _smooth_clip_deriv_batch(s_unc)
    w_t = _smooth_clip_deriv_batch(t_unc)
    active = (w_s > 1e-30) | (w_t > 1e-30)
    if not np.any(active):
        return ds_du, dt_du, valid

    s_eval = s_unc
    t_eval = t_unc

    # Hermite 接線ベクトル（バッチ）
    # dpA/ds = H00'(s)*xA0 + H10'(s)*mA0 + H01'(s)*xA1 + H11'(s)*mA1
    dh00_s = 6.0 * s_eval * s_eval - 6.0 * s_eval  # (N,)
    dh01_s = -6.0 * s_eval * s_eval + 6.0 * s_eval
    dh10_s = 3.0 * s_eval * s_eval - 4.0 * s_eval + 1.0
    dh11_s = 3.0 * s_eval * s_eval - 2.0 * s_eval

    dh00_t = 6.0 * t_eval * t_eval - 6.0 * t_eval
    dh01_t = -6.0 * t_eval * t_eval + 6.0 * t_eval
    dh10_t = 3.0 * t_eval * t_eval - 4.0 * t_eval + 1.0
    dh11_t = 3.0 * t_eval * t_eval - 2.0 * t_eval

    dpA = (
        dh00_s[:, None] * xA0
        + dh10_s[:, None] * mA0
        + dh01_s[:, None] * xA1
        + dh11_s[:, None] * mA1
    )  # (N, 3)
    dpB = (
        dh00_t[:, None] * xB0
        + dh10_t[:, None] * mB0
        + dh01_t[:, None] * xB1
        + dh11_t[:, None] * mB1
    )  # (N, 3)

    # Hermite 位置
    h00_s = 2.0 * s_eval**3 - 3.0 * s_eval**2 + 1.0
    h01_s = -2.0 * s_eval**3 + 3.0 * s_eval**2
    h10_s = s_eval**3 - 2.0 * s_eval**2 + s_eval
    h11_s = s_eval**3 - s_eval**2

    h00_t = 2.0 * t_eval**3 - 3.0 * t_eval**2 + 1.0
    h01_t = -2.0 * t_eval**3 + 3.0 * t_eval**2
    h10_t = t_eval**3 - 2.0 * t_eval**2 + t_eval
    h11_t = t_eval**3 - t_eval**2

    pA = h00_s[:, None] * xA0 + h10_s[:, None] * mA0 + h01_s[:, None] * xA1 + h11_s[:, None] * mA1
    pB = h00_t[:, None] * xB0 + h10_t[:, None] * mB0 + h01_t[:, None] * xB1 + h11_t[:, None] * mB1
    delta = pA - pB  # (N, 3)

    # 2階微分
    ddh00_s = 12.0 * s_eval - 6.0
    ddh01_s = -12.0 * s_eval + 6.0
    ddh10_s = 6.0 * s_eval - 4.0
    ddh11_s = 6.0 * s_eval - 2.0

    ddh00_t = 12.0 * t_eval - 6.0
    ddh01_t = -12.0 * t_eval + 6.0
    ddh10_t = 6.0 * t_eval - 4.0
    ddh11_t = 6.0 * t_eval - 2.0

    d2pA = (
        ddh00_s[:, None] * xA0
        + ddh10_s[:, None] * mA0
        + ddh01_s[:, None] * xA1
        + ddh11_s[:, None] * mA1
    )
    d2pB = (
        ddh00_t[:, None] * xB0
        + ddh10_t[:, None] * mB0
        + ddh01_t[:, None] * xB1
        + ddh11_t[:, None] * mB1
    )

    # Gram 行列（Hermite: 2階微分項を含む）
    a = np.sum(dpA * dpA, axis=1) + np.sum(delta * d2pA, axis=1)
    b_gram = np.sum(dpA * dpB, axis=1)
    c = np.sum(dpB * dpB, axis=1) - np.sum(delta * d2pB, axis=1)
    det = a * c - b_gram * b_gram

    tol = 1e-10
    ac_product = np.maximum(np.abs(a * c), 1e-30)
    non_singular = np.abs(det) >= tol * ac_product

    # ── 高速パス: w_s≈1, w_t≈1, 非特異 ──
    fast = active & (w_s >= 1.0 - 1e-10) & (w_t >= 1.0 - 1e-10) & non_singular
    n_fast = int(np.sum(fast))

    if n_fast > 0:
        inv_det_f = 1.0 / det[fast]
        dpA_f = dpA[fast]
        dpB_f = dpB[fast]
        delta_f = delta[fast]
        a_f = a[fast]
        b_f = b_gram[fast]
        c_f = c[fast]
        w_s_f = w_s[fast]
        w_t_f = w_t[fast]

        # h_eff, dh_eff per node（Hermite 基底 + dm 補正）
        # Node 0: h=H00(s), dh=H00'(s)
        h_A0 = h00_s[fast].copy()
        dh_A0 = dh00_s[fast].copy()
        h_A1 = h01_s[fast].copy()
        dh_A1 = dh01_s[fast].copy()
        h_B0 = h00_t[fast].copy()
        dh_B0 = dh00_t[fast].copy()
        h_B1 = h01_t[fast].copy()
        dh_B1 = dh01_t[fast].copy()

        if dm_A is not None:
            dm_A_f = dm_A[fast]  # (Nf, 2, 2)
            h10_sf = h10_s[fast]
            h11_sf = h11_s[fast]
            dh10_sf = dh10_s[fast]
            dh11_sf = dh11_s[fast]
            h_A0 += h10_sf * dm_A_f[:, 0, 0] + h11_sf * dm_A_f[:, 1, 0]
            dh_A0 += dh10_sf * dm_A_f[:, 0, 0] + dh11_sf * dm_A_f[:, 1, 0]
            h_A1 += h10_sf * dm_A_f[:, 0, 1] + h11_sf * dm_A_f[:, 1, 1]
            dh_A1 += dh10_sf * dm_A_f[:, 0, 1] + dh11_sf * dm_A_f[:, 1, 1]

        if dm_B is not None:
            dm_B_f = dm_B[fast]
            h10_tf = h10_t[fast]
            h11_tf = h11_t[fast]
            dh10_tf = dh10_t[fast]
            dh11_tf = dh11_t[fast]
            h_B0 += h10_tf * dm_B_f[:, 0, 0] + h11_tf * dm_B_f[:, 1, 0]
            dh_B0 += dh10_tf * dm_B_f[:, 0, 0] + dh11_tf * dm_B_f[:, 1, 0]
            h_B1 += h10_tf * dm_B_f[:, 0, 1] + h11_tf * dm_B_f[:, 1, 1]
            dh_B1 += dh10_tf * dm_B_f[:, 0, 1] + dh11_tf * dm_B_f[:, 1, 1]

        # RHS for 4 nodes (Hermite)
        # A0: rhs[0] = h_A0 * dpA + dh_A0 * delta, rhs[1] = -h_A0 * dpB
        # A1: rhs[0] = h_A1 * dpA + dh_A1 * delta, rhs[1] = -h_A1 * dpB
        # B0: rhs[0] = -h_B0 * dpA, rhs[1] = h_B0 * dpB - dh_B0 * delta
        # B1: rhs[0] = -h_B1 * dpA, rhs[1] = h_B1 * dpB - dh_B1 * delta
        rhs_data = [
            (h_A0[:, None] * dpA_f + dh_A0[:, None] * delta_f, -h_A0[:, None] * dpB_f),
            (h_A1[:, None] * dpA_f + dh_A1[:, None] * delta_f, -h_A1[:, None] * dpB_f),
            (-h_B0[:, None] * dpA_f, h_B0[:, None] * dpB_f - dh_B0[:, None] * delta_f),
            (-h_B1[:, None] * dpA_f, h_B1[:, None] * dpB_f - dh_B1[:, None] * delta_f),
        ]

        ds_du_f = np.zeros((n_fast, 12))
        dt_du_f = np.zeros((n_fast, 12))

        for node_idx, (rF1, rF2) in enumerate(rhs_data):
            ds_du_f[:, node_idx * 3 : node_idx * 3 + 3] = (
                -(c_f[:, None] * rF1 + b_f[:, None] * rF2) * inv_det_f[:, None]
            )
            dt_du_f[:, node_idx * 3 : node_idx * 3 + 3] = (
                -(b_f[:, None] * rF1 + a_f[:, None] * rF2) * inv_det_f[:, None]
            )

        ds_du_f *= w_s_f[:, None]
        dt_du_f *= w_t_f[:, None]
        ds_du[fast] = ds_du_f
        dt_du[fast] = dt_du_f

    # ── 低速パス: エッジケース ──
    slow = active & ~fast
    if np.any(slow):
        slow_idx = np.where(slow)[0]
        proc = ComputeStJacobianProcess()
        for idx in slow_idx:
            kw: dict = {
                "xA0": xA0[idx],
                "xA1": xA1[idx],
                "xB0": xB0[idx],
                "xB1": xB1[idx],
                "s": float(s[idx]),
                "t": float(t[idx]),
                "s_unclamped": float(s_unc[idx]),
                "t_unclamped": float(t_unc[idx]),
                "mA0": mA0[idx],
                "mA1": mA1[idx],
                "mB0": mB0[idx],
                "mB1": mB1[idx],
                "use_hermite": True,
            }
            if dm_A is not None:
                kw["dm_A"] = dm_A[idx]
            if dm_B is not None:
                kw["dm_B"] = dm_B[idx]
            out = proc.process(StJacobianInput(**kw))
            ds_du[idx] = out.ds_du
            dt_du[idx] = out.dt_du
            valid[idx] = out.valid

    return ds_du, dt_du, valid
