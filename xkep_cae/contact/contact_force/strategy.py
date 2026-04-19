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
from typing import ClassVar

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state
from xkep_cae.contact.geometry._st_jacobian import ComputeStJacobianProcess
from xkep_cae.core import ProcessMeta, SolverProcess
from xkep_cae.mathematics import MathematicalContract, TermExpansionContract

# ── K_c 項展開契約（status-350 Phase C-1 → status-351 Phase C-2 → status-353 訂正） ─────
# 数理台帳 03 章の 5 項完全分解 (K_mat_nn / K_closest / K_hermite_adj / K_geo /
# K_st) を独立 Process として確立済み。
#
# status-353 訂正: 当初 status-346〜352 で計画されていた「K_mat_ndir（法線方向
# 感度 -p_n · ∂n̂/∂u）を独立項として追加」は、数理検証により
# K_geo（= -p_n · ∂n̂/∂u のペア局所形、重み p_n/d、I_nn 項）と **同一である**
# ことが判明（docs/math/03_huber_contact_penalty.md §4 参照）。新規追加は
# 二重計上となるため撤回し、5 項で完結とする。
#
# term_names/providers の長さは等しく、`TermExpansionContract.__post_init__` で
# 同一性が検査される。orchestrator `tangent_components()` は 5 Process の出力を
# 組み立て、後方互換のため 3-tuple (K_mat, K_geo, K_st) を返す:
#   K_mat = K_mat_nn + K_hermite_adj (KcNormal + KcHermiteNonlocal)
#   K_st  = K_closest + K_st         (KcClosest + ContactForceSt)
_K_C_TERM_EXPANSION_CONTRACT: TermExpansionContract = TermExpansionContract(
    name="K_c_term_expansion",
    equation_ref="03_huber_contact_penalty.md#eq-kc-full-decomposition",
    total_name="K_c",
    term_names=("K_mat_nn", "K_closest", "K_hermite_adj", "K_geo", "K_st"),
    providers=(
        "KcNormalStiffnessProcess",
        "KcClosestPointStiffnessProcess",
        "KcHermiteNonlocalStiffnessProcess",
        "KcGeoStiffnessProcess",
        "ContactForceStStiffnessProcess",
    ),
    combinator="add_sub",
    tol_rel=5e-3,
    severity="nightly",
    description=(
        "status-351 Phase C-2 + status-353 訂正: tangent_components() の 5 項分解。"
        "K_mat_nn / K_closest / K_hermite_adj / K_geo / K_st が独立 Process。"
        "K_mat_ndir（当初 Phase C-3 計画）は K_geo と同一のため未追加（status-353）。"
    ),
)

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


# ── B1: ContactForceStStiffnessProcess ─────────────────────
# 旧スカラー版 _add_kst_contact_to_coo は status-311 で削除
# （バッチ版で完全置換済み。69-208x高速化: status-310）


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
    """接触力 K_st（接触点滑り剛性）の出力.

    status-321: K_st の型を `sp.csr_matrix | sp.coo_matrix` に緩めた。
    内部 `tocsr()` を skip し COO 形式のまま返すことで、呼び出し側で 1 度だけ
    CSR に変換する fast path を可能にした（往復変換削減）。CSR/COO どちらでも
    `+`, `@`, `.shape`, `.nnz`, `.toarray()`, `sp.linalg.norm` 等は等価動作する。
    """

    K_st: sp.csr_matrix | sp.coo_matrix


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
        version="1.1.0",
        document_path="docs/contact_force.md",
    )
    uses = [ComputeStJacobianProcess]
    # status-350 Phase C-1: K_c 項展開契約の K_st プロバイダとして宣言
    contracts: ClassVar[tuple[MathematicalContract, ...]] = (_K_C_TERM_EXPANSION_CONTRACT,)

    def process(self, inp: ContactForceStStiffnessInput) -> ContactForceStStiffnessOutput:
        return self._process_batch(inp)

    @classmethod
    def _assemble_term_coo(
        cls,
        inp: ContactForceStStiffnessInput,
        *,
        term: str,
    ) -> sp.csr_matrix | sp.coo_matrix:
        """K_st 残差項 / K_closest 項を共通セットアップから組み立てる共通ヘルパ.

        status-351 Phase C-2: ``ContactForceStStiffnessProcess`` と
        ``KcClosestPointStiffnessProcess`` が同じ StJacobian・∂n/∂s・∂p_n/∂s
        セットアップから異なる項を組み立てる。両 Process は本メソッドを呼び出して
        `term="residual"` / `term="closest"` を指定する。

        Args:
            inp: ペア・座標・k_pen 等の入力（既存 ContactForceStStiffnessInput）
            term: "residual"（K_st の形状関数・法線追従項）または
                  "closest"（K_closest の dpn 追従項）

        Returns:
            対象項の COO/CSR 行列（アクティブペアなしの場合は空 CSR）。
        """
        return cls._process_batch_term(inp, term=term)

    def _process_batch(self, inp: ContactForceStStiffnessInput) -> ContactForceStStiffnessOutput:
        """K_st 残差項のバッチ計算（status-351 Phase C-2 以降は残差のみ）.

        status-309 で全アクティブペアを NumPy 配列に抽出、StJacobian+K_st を
        一括計算するバッチ経路を導入。status-321/322 で更に高速化。

        status-351 Phase C-2: 本 Process は K_st 残差項 ``-outer(p_n*(dc_ds*n
        + coeffs*dn_ds), ds_du) + ...`` のみを返す。K_closest 項
        ``-outer(dpn_ds*g_shape, ds_du) + ...`` は
        ``KcClosestPointStiffnessProcess`` が ``_assemble_term_coo(term="closest")``
        経由で独立に組み立てる。
        """
        K_st = self._process_batch_term(inp, term="residual")
        return ContactForceStStiffnessOutput(K_st=K_st)

    @classmethod
    def _process_batch_term(
        cls,
        inp: ContactForceStStiffnessInput,
        *,
        term: str,
    ) -> sp.csr_matrix | sp.coo_matrix:
        """K_st 残差項または K_closest 項のバッチ計算.

        status-351 Phase C-2 で ``_process_batch`` から項選択ヘルパへ抽出。
        共通セットアップ（StJacobian, ∂n/∂s, ∂p_n/∂s）を 1 度実行し、
        ``term`` 引数で K_st 残差 / K_closest のどちらを返すかを切り替える。
        """
        zero = sp.csr_matrix((inp.ndof_total, inp.ndof_total))

        # ── ペアデータ抽出（status-322: pre-bound states パターン） ──
        if not inp.pairs:
            return zero

        # Distance culling threshold (status-324):
        # Huber derivative = 0 when gap > delta_h / k_pen → K_st 寄与ゼロ。
        # この gap を超えるペアを step 1 で除外し、p_n 抽出・StJacobian 計算を
        # スキップする。1e-8 は浮動小数点境界のマージン。
        _gap_cull = float("inf")
        if inp.k_pen > 0:
            _gap_cull = (inp.delta_h / inp.k_pen if inp.delta_h > 0 else 0.0) + 1e-8

        # Step 1: state + gap < _gap_cull pre-filter（status-324 distance culling）
        has_state_pairs = [p for p in inp.pairs if hasattr(p, "state") and p.state.gap < _gap_cull]
        if not has_state_pairs:
            return zero

        # Step 2: p_n > 0 で active filter（state を一度だけ bind）
        states_all = [p.state for p in has_state_pairs]
        p_n_state = np.fromiter(
            (st.p_n for st in states_all),
            dtype=float,
            count=len(states_all),
        )
        active_local = p_n_state > 1e-30
        n_act = int(np.sum(active_local))
        if n_act == 0:
            return zero

        # Step 3: active ペア + state を pre-bind
        act_idx = np.where(active_local)[0]
        act_pairs = [has_state_pairs[k] for k in act_idx]
        states = [states_all[k] for k in act_idx]

        p_n_act = p_n_state[active_local]
        gaps_act = np.fromiter((st.gap for st in states), dtype=float, count=n_act)
        s_act = np.fromiter((st.s for st in states), dtype=float, count=n_act)
        t_act = np.fromiter((st.t for st in states), dtype=float, count=n_act)
        # `getattr(st, 's_unclamped', st.s) or st.s` で None/0.0 どちらも st.s に
        # フォールバック（friction/_assembly.py と同一パターン）
        s_unc_act = np.fromiter(
            (getattr(st, "s_unclamped", st.s) or st.s for st in states),
            dtype=float,
            count=n_act,
        )
        t_unc_act = np.fromiter(
            (getattr(st, "t_unclamped", st.t) or st.t for st in states),
            dtype=float,
            count=n_act,
        )
        n_act_v = np.array([st.normal for st in states], dtype=float)
        nodes_act = np.array(
            [(p.nodes_a[0], p.nodes_a[1], p.nodes_b[0], p.nodes_b[1]) for p in act_pairs],
            dtype=int,
        )
        ra_act = np.fromiter((p.radius_a for p in act_pairs), dtype=float, count=n_act)
        rb_act = np.fromiter((p.radius_b for p in act_pairs), dtype=float, count=n_act)

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
            ds_du, dt_du, valid, _, _ = _batch_st_jacobian_hermite(
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
            # dpA, dpB for dn/ds, dn/dt（バッチ化: status-310）
            from xkep_cae.contact.geometry._st_jacobian import _hermite_deriv_batch

            dpA_arr = _hermite_deriv_batch(s_act, xA0, xA1, mA0, mA1)
            dpB_arr = _hermite_deriv_batch(t_act, xB0, xB1, mB0, mB1)
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
            return zero

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

        # ── ∂n/∂s, ∂n/∂t および ∂p_n/∂s, ∂p_n/∂t のバッチ計算 ──
        # status-322: P_perp の (N,3,3) 明示形成を回避。
        #   P_perp @ v = v - n * (n · v)
        # つまり dgap/ds = n·dpA を先に計算すれば、dn/ds は 1 回の軸ブロードキャスト
        # 差分だけで求まる。einsum("nij,nj->ni") 2 回分のコストが消える。
        dist = gaps_act + ra_act + rb_act  # (N,)
        dist_valid = dist > 1e-15
        inv_dist = np.where(dist_valid, 1.0 / np.where(dist_valid, dist, 1.0), 0.0)

        # dgap/ds = n·dpA, dgap/dt = -(n·dpB)
        n_dot_dpA = np.einsum("ij,ij->i", n_act_v, dpA_arr)  # (N,)
        n_dot_dpB = np.einsum("ij,ij->i", n_act_v, dpB_arr)
        dgap_ds = n_dot_dpA
        dgap_dt = -n_dot_dpB

        # P_perp @ dpA = dpA - n * (n·dpA)  (broadcasting で (N,3) 直接)
        dn_ds = inv_dist[:, None] * (dpA_arr - n_act_v * n_dot_dpA[:, None])
        dn_dt = -inv_dist[:, None] * (dpB_arr - n_act_v * n_dot_dpB[:, None])

        dpn_ds = h_deriv * inp.k_pen * (-dgap_ds)
        dpn_dt = h_deriv * inp.k_pen * (-dgap_dt)
        # h_deriv が小さい場合はゼロに
        h_active = h_deriv > 1e-30
        dpn_ds = np.where(h_active, dpn_ds, 0.0)
        dpn_dt = np.where(h_active, dpn_dt, 0.0)

        # ── g_shape (N, 4, 3) をブロードキャスト一発で構築 ──
        # g_shape[:, k, d] = coeffs[:, k] * n_act_v[:, d]
        g_shape_3d = coeffs[:, :, None] * n_act_v[:, None, :]  # (N, 4, 3)
        g_shape = g_shape_3d.reshape(n_act, 12)

        # ── df_ds, df_dt を status-351 Phase C-2 で 2 成分に分解 ──
        # ∂f_c/∂(s,t) の完全微分:
        #   df_ds = dpn_ds * g_shape                   ... (K_closest 寄与)
        #         + p_n * (dc_ds * n + coeffs * dn_ds) ... (K_st 残差寄与)
        # 前者は "(s,t) 摂動に伴う p_n 追従" = K_closest 項、
        # 後者は "∂(c_k * n̂)/∂(s,t) に伴う形状関数・法線追従" = K_st 残差項。
        # 本 Process は K_st 残差項のみを返す。K_closest は
        # ``KcClosestPointStiffnessProcess`` が同じ分解式から算出する。
        # status-322: for-k ループを broadcasting に置換。
        df_ds_rest_block = p_n_act[:, None, None] * (
            dc_ds[:, :, None] * n_act_v[:, None, :] + coeffs[:, :, None] * dn_ds[:, None, :]
        )  # (N, 4, 3)
        df_dt_rest_block = p_n_act[:, None, None] * (
            dc_dt[:, :, None] * n_act_v[:, None, :] + coeffs[:, :, None] * dn_dt[:, None, :]
        )
        df_ds_rest = df_ds_rest_block.reshape(n_act, 12)
        df_dt_rest = df_dt_rest_block.reshape(n_act, 12)

        # ── 項別 K_local = -(outer(df_ds_*, ds_du) + outer(df_dt_*, dt_du)) ──
        # status-321: einsum → 直接ブロードキャスト（外積は broadcasting の方が高速）。
        if term == "closest":
            df_ds_term = dpn_ds[:, None] * g_shape
            df_dt_term = dpn_dt[:, None] * g_shape
        elif term == "residual":
            df_ds_term = df_ds_rest
            df_dt_term = df_dt_rest
        else:
            raise ValueError(f"Unknown term: {term!r}; expected 'residual' or 'closest'")

        K_local = -(
            df_ds_term[:, :, None] * ds_du[:, None, :] + df_dt_term[:, :, None] * dt_du[:, None, :]
        )

        # ── DOF インデックス (N, 12) ──
        # status-322: 二重 for-k/for-d ループを単一ブロードキャスト式へ
        # gdofs[:, k, d] = nodes_act[:, k] * ndpn + d
        ndpn = inp.ndof_per_node
        gdofs = (nodes_act[:, :, None] * ndpn + np.arange(3, dtype=int)[None, None, :]).reshape(
            n_act, 12
        )

        # ── COO 構築 ──
        # status-321: mask filter を skip。零エントリは CSR 統合時に自動で
        # 集約され、無視できる。マスク作成 + 索引コピーが per-call 1〜2ms 程度。
        row_idx = np.broadcast_to(gdofs[:, :, None], (n_act, 12, 12)).ravel()
        col_idx = np.broadcast_to(gdofs[:, None, :], (n_act, 12, 12)).ravel()
        val_arr = K_local.ravel()

        if len(val_arr) == 0:
            return zero
        # status-321: tocsr() を skip し COO のまま返す（呼び出し側で 1 度だけ
        # 集約 CSR 化することで往復変換 ~11ms/call を削減）
        return sp.coo_matrix(
            (val_arr, (row_idx, col_idx)),
            shape=(inp.ndof_total, inp.ndof_total),
        )


# ── K_c 項別 Process (status-350 Phase C-1) ────────────────


@dataclass(frozen=True)
class KcTermAssemblyInput:
    """K_c 項別 Process の共通入力 (status-350 Phase C-1).

    KcNormalStiffnessProcess / KcGeoStiffnessProcess が共有する入力構造。
    ``pairs`` から各 Process が独立にアクティブ集合を抽出し、対応する項の
    接触接線剛性ブロックを組み立てる。各 Process は自己完結し、他 Process
    の内部状態には依存しない（MCDD 脱法実装 pattern 3 の防止）。

    Attributes:
        pairs: 接触ペアのリスト（ContactManager.pairs と互換）。
        k_pen: ペナルティ係数。
        delta_h: Huber 遷移幅（既に boost 反映済み）。
        ndof_total: 全体 DOF 数。
        ndof_per_node: ノードあたり DOF 数（デフォルト 6、並進 3 + 回転 3）。
        penalty_exponent: Hertz 型指数（1.0 で線形、1.5 で Hertz、status-285）。
        use_hermite: Hermite 形状関数基底を使用するか。
        node_counts: 各ノードの要素参照数（Hermite 補正用、None で線形補間）。
        adj_node_map: elem → (隣接ノード 0, 隣接ノード 1) のマップ
            （K_mat_adj 隣接拡張用、KcGeo では未使用）。
        adj_node_counts: 各ノードの隣接要素数（K_mat_adj 用、KcGeo では未使用）。
    """

    pairs: list
    k_pen: float
    delta_h: float
    ndof_total: int
    ndof_per_node: int = 6
    penalty_exponent: float = 1.0
    use_hermite: bool = False
    node_counts: np.ndarray | None = None
    adj_node_map: dict | None = None
    adj_node_counts: np.ndarray | None = None


@dataclass(frozen=True)
class KcNormalStiffnessOutput:
    """KcNormalStiffnessProcess の出力.

    status-351 Phase C-2 以降、``K_mat`` は **K_mat_nn のみ** を表す
    （K_hermite_adj は ``KcHermiteNonlocalStiffnessOutput.K_hermite_adj`` へ分離）。
    """

    K_mat: sp.csr_matrix


@dataclass(frozen=True)
class KcHermiteNonlocalStiffnessOutput:
    """KcHermiteNonlocalStiffnessProcess の出力 (status-351 Phase C-2).

    隣接ノードの Hermite 接線経由で伝搬する ∂p_A/∂u_adj 寄与を持つ
    材料剛性拡張（status-295 の mat-only K_c_adj に相当）。
    """

    K_hermite_adj: sp.csr_matrix


@dataclass(frozen=True)
class KcGeoStiffnessOutput:
    """KcGeoStiffnessProcess の出力."""

    K_geo: sp.csr_matrix


@dataclass(frozen=True)
class KcClosestPointStiffnessOutput:
    """KcClosestPointStiffnessProcess の出力 (status-351 Phase C-2).

    最近接点 $(s,t)$ の摂動に伴う $\\partial p_n/\\partial \\boldsymbol{u}$
    追従寄与 ``-(dpn_ds * g_shape) \\otimes ds_du - (dpn_dt * g_shape) \\otimes dt_du``
    を担う。
    """

    K_closest: sp.csr_matrix | sp.coo_matrix


def _extract_kc_active_pair_data(inp: KcTermAssemblyInput) -> dict | None:
    """K_c 項別 Process 共通のアクティブペア抽出ヘルパ (status-350).

    ``has_state & ((h_deriv > 0) | (p_n > 0))`` でアクティブ集合をフィルタし、
    w_mat / w_geo / coeffs / nn / I_nn / cc / gdofs などの共有計算を実行する。

    Returns:
        辞書 ``{"n_act", "w_mat", "w_geo", "coeffs", "nn", "I_nn", "cc",
        "gdofs", "s_act", "t_act", "nodes_act"}`` を返す。
        アクティブ 0 の場合は None を返す。
    """
    if not inp.pairs:
        return None
    n = len(inp.pairs)
    has_state = np.zeros(n, dtype=bool)
    gaps = np.zeros(n)
    s_arr = np.zeros(n)
    t_arr = np.zeros(n)
    normals = np.zeros((n, 3))
    nodes = np.zeros((n, 4), dtype=int)
    radius_a = np.zeros(n)
    radius_b = np.zeros(n)
    p_n_all = np.zeros(n)
    for i, pair in enumerate(inp.pairs):
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
        p_n_all[i] = pair.state.p_n

    x_pen = inp.k_pen * (-gaps)
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
    h_deriv[~has_state] = 0.0

    active = has_state & ((h_deriv > 1e-30) | (p_n_all > 1e-30))
    n_act = int(np.sum(active))
    if n_act == 0:
        return None

    h_deriv_act = h_deriv[active]
    p_n_act = p_n_all[active]
    gaps_act = gaps[active]
    s_act = s_arr[active]
    t_act = t_arr[active]
    n_act_v = normals[active]
    nodes_act = nodes[active]
    ra_act = radius_a[active]
    rb_act = radius_b[active]

    w_mat = h_deriv_act * inp.k_pen
    dist = gaps_act + ra_act + rb_act
    w_geo = np.where(dist > 1e-15, p_n_act / dist, 0.0)

    # 形状関数係数（Hermite or 線形）
    if inp.use_hermite:
        if inp.node_counts is not None:
            dm_A, dm_B = HuberContactForceProcess._batch_dm_coeffs(inp.node_counts, nodes_act)
            coeffs, _, _ = HuberContactForceProcess._batch_hermite_corrected_coeffs(
                s_act, t_act, dm_A, dm_B
            )
        else:
            coeffs = HuberContactForceProcess._batch_hermite_coeffs(s_act, t_act)
    else:
        coeffs = np.column_stack([1.0 - s_act, s_act, -(1.0 - t_act), -t_act])

    nn = n_act_v[:, :, None] * n_act_v[:, None, :]
    I3 = np.eye(3)[None, :, :]
    I_nn = I3 - nn
    cc = coeffs[:, :, None] * coeffs[:, None, :]

    ndpn = inp.ndof_per_node
    gdofs = (nodes_act[:, :, None] * ndpn + np.arange(3, dtype=int)[None, None, :]).reshape(
        n_act, 12
    )

    return {
        "active": active,
        "n_act": n_act,
        "p_n_act": p_n_act,
        "s_act": s_act,
        "t_act": t_act,
        "n_act_v": n_act_v,
        "nodes_act": nodes_act,
        "w_mat": w_mat,
        "w_geo": w_geo,
        "coeffs": coeffs,
        "nn": nn,
        "I_nn": I_nn,
        "cc": cc,
        "gdofs": gdofs,
    }


def _assemble_12x12_pair_block(
    K_3x3_block: np.ndarray,
    cc: np.ndarray,
    gdofs: np.ndarray,
    ndof_total: int,
) -> sp.csr_matrix:
    """ペア 3x3 ブロック × 形状係数外積 cc(4,4) → 12x12 COO アセンブリ.

    K_local[n, ki*3:(ki+1)*3, kj*3:(kj+1)*3] = cc[n, ki, kj] * K_3x3_block[n]
    """
    n_act = K_3x3_block.shape[0]
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
        shape=(ndof_total, ndof_total),
    ).tocsr()


class KcNormalStiffnessProcess(
    SolverProcess[KcTermAssemblyInput, KcNormalStiffnessOutput],
):
    """K_c 法線材料剛性項 K_mat_nn を計算する Process (status-350 / status-351 / status-353).

    数理台帳 ``docs/math/03_huber_contact_penalty.md#eq-kc-full-decomposition``
    の第 1 項 $\\boldsymbol{K}_{\\mathrm{mat,nn}}
    = +\\frac{\\partial p_n}{\\partial \\boldsymbol{u}}\\otimes\\hat{\\boldsymbol{n}}$
    （符号は $\\boldsymbol{K}_c = \\partial(-\\boldsymbol{f}_c)/\\partial \\boldsymbol{u}$ の残差系規約、
    $(s,t)$ を凍結したペア局所 $3\\times 3$ ブロック $w_{\\mathrm{mat}}\\,(\\hat{\\boldsymbol{n}}\\otimes\\hat{\\boldsymbol{n}})$）
    を提供する。

    status-351 Phase C-2 で隣接ノード拡張 (K_hermite_adj) を
    ``KcHermiteNonlocalStiffnessProcess`` へ分離。本 Process はペア局所の
    12x12 ブロックのみを扱い、非局所ノードには寄与しない。

    status-353 訂正: 当初計画されていた K_mat_ndir 項（法線方向感度
    $-p_n\\,\\partial\\hat{\\boldsymbol{n}}/\\partial \\boldsymbol{u}$）は、
    ``KcGeoStiffnessProcess`` が既に担う I_nn 項と数理的に同一であることが
    status-353 で確認されたため、追加 Process は不要（5 項で完結）。
    """

    meta = ProcessMeta(
        name="KcNormalStiffness",
        module="solve",
        version="1.1.0",
        document_path="docs/contact_force.md",
    )
    uses: list[type] = []
    contracts: ClassVar[tuple[MathematicalContract, ...]] = (_K_C_TERM_EXPANSION_CONTRACT,)

    def process(self, inp: KcTermAssemblyInput) -> KcNormalStiffnessOutput:
        zero = sp.csr_matrix((inp.ndof_total, inp.ndof_total))
        data = _extract_kc_active_pair_data(inp)
        if data is None:
            return KcNormalStiffnessOutput(K_mat=zero)

        # K_mat_nn ペア局所: w_mat * (n⊗n) * c_i c_j
        K_3x3_mat = data["w_mat"][:, None, None] * data["nn"]
        K_mat = _assemble_12x12_pair_block(K_3x3_mat, data["cc"], data["gdofs"], inp.ndof_total)
        return KcNormalStiffnessOutput(K_mat=K_mat)


class KcHermiteNonlocalStiffnessProcess(
    SolverProcess[KcTermAssemblyInput, KcHermiteNonlocalStiffnessOutput],
):
    """K_c Hermite 非局所材料剛性 K_hermite_adj を計算する Process (status-351).

    Hermite 補間の接線ベクトル $\\boldsymbol{m}_0,\\boldsymbol{m}_1$ が隣接ノード
    座標の有限差分で構成されるため、$\\boldsymbol{p}_A(s)$ は実質的に隣接ノード
    座標にも依存する（status-271〜274 / status-294 frozen-m 部分解消）。
    本項は材料剛性 $w_{\\mathrm{mat}}\\,(\\hat{\\boldsymbol{n}}\\otimes\\hat{\\boldsymbol{n}})$
    を隣接ノード DOF 列に拡張した mat-only 組み立て（status-295 で正当化）。

    数理台帳 ``docs/math/03_huber_contact_penalty.md#eq-hermite-pA`` 参照。
    """

    meta = ProcessMeta(
        name="KcHermiteNonlocalStiffness",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_force.md",
    )
    uses: list[type] = []
    contracts: ClassVar[tuple[MathematicalContract, ...]] = (_K_C_TERM_EXPANSION_CONTRACT,)

    def process(self, inp: KcTermAssemblyInput) -> KcHermiteNonlocalStiffnessOutput:
        zero = sp.csr_matrix((inp.ndof_total, inp.ndof_total))
        if inp.adj_node_map is None or inp.adj_node_counts is None:
            return KcHermiteNonlocalStiffnessOutput(K_hermite_adj=zero)
        data = _extract_kc_active_pair_data(inp)
        if data is None:
            return KcHermiteNonlocalStiffnessOutput(K_hermite_adj=zero)

        n_act = data["n_act"]
        w_mat = data["w_mat"]
        nn = data["nn"]
        gdofs = data["gdofs"]
        coeffs = data["coeffs"]
        s_act = data["s_act"]
        t_act = data["t_act"]
        nodes_act = data["nodes_act"]

        K_3x3_mat = w_mat[:, None, None] * nn

        active_idx = np.where(data["active"])[0]
        pairs_active = [inp.pairs[int(idx)] for idx in active_idx]
        elem_a_act = np.array([p.elem_a for p in pairs_active], dtype=int)
        elem_b_act = np.array([p.elem_b for p in pairs_active], dtype=int)

        c_a0 = np.maximum(inp.adj_node_counts[nodes_act[:, 0]], 1.0)
        c_a1 = np.maximum(inp.adj_node_counts[nodes_act[:, 1]], 1.0)
        c_b0 = np.maximum(inp.adj_node_counts[nodes_act[:, 2]], 1.0)
        c_b1 = np.maximum(inp.adj_node_counts[nodes_act[:, 3]], 1.0)
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
            adj_a = inp.adj_node_map.get(int(elem_a_act[i]), (-1, -1))
            adj_b = inp.adj_node_map.get(int(elem_b_act[i]), (-1, -1))
            adj_gnodes[i] = [adj_a[0], adj_a[1], adj_b[0], adj_b[1]]

        c_alpha = coeffs[:, :, None] * alpha_adj[:, None, :]
        K_mat_adj_local = np.zeros((n_act, 12, 12))
        for ki in range(4):
            for aj in range(4):
                K_mat_adj_local[:, ki * 3 : (ki + 1) * 3, aj * 3 : (aj + 1) * 3] = (
                    c_alpha[:, ki, aj][:, None, None] * K_3x3_mat
                )

        ndpn = inp.ndof_per_node
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
        if not mask_adj.any():
            return KcHermiteNonlocalStiffnessOutput(K_hermite_adj=zero)
        K_hermite_adj = sp.coo_matrix(
            (
                val_adj[mask_adj],
                (row_adj[mask_adj], col_adj[mask_adj]),
            ),
            shape=(inp.ndof_total, inp.ndof_total),
        ).tocsr()
        return KcHermiteNonlocalStiffnessOutput(K_hermite_adj=K_hermite_adj)


class KcGeoStiffnessProcess(
    SolverProcess[KcTermAssemblyInput, KcGeoStiffnessOutput],
):
    """K_c 幾何補正項 K_geo を計算する Process (status-350 Phase C-1 / status-353 訂正).

    K_geo = (p_n / d) * (I - n̂⊗n̂) のペア局所組み立て。数理台帳
    ``docs/math/03_huber_contact_penalty.md#eq-kc-pair-block`` の w_geo 項。
    ``tangent_components()`` の規約では ``K_c = K_mat - K_geo + K_st`` で
    合成されるため、Process は絶対値の幾何剛性を返し、符号は呼び出し側が処理する。

    status-353 訂正: 本項は **法線方向感度 $-p_n \\cdot \\partial\\hat{\\boldsymbol{n}}/\\partial\\boldsymbol{u}$
    のペア局所形そのもの** であり、$1/d$ 因子は
    $\\hat{\\boldsymbol{n}} = \\boldsymbol{r}/d$ の内在項として現れる。
    status-346〜352 で独立項扱いされていた K_mat_ndir は本 Process と同一のため、
    新規 Process 追加は二重計上となる（詳細は数理台帳 §4）。
    """

    meta = ProcessMeta(
        name="KcGeoStiffness",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_force.md",
    )
    uses: list[type] = []
    contracts: ClassVar[tuple[MathematicalContract, ...]] = (_K_C_TERM_EXPANSION_CONTRACT,)

    def process(self, inp: KcTermAssemblyInput) -> KcGeoStiffnessOutput:
        zero = sp.csr_matrix((inp.ndof_total, inp.ndof_total))
        data = _extract_kc_active_pair_data(inp)
        if data is None:
            return KcGeoStiffnessOutput(K_geo=zero)

        # K_geo ペア局所: w_geo * (I - n⊗n)
        K_3x3_geo = data["w_geo"][:, None, None] * data["I_nn"]
        K_geo = _assemble_12x12_pair_block(K_3x3_geo, data["cc"], data["gdofs"], inp.ndof_total)
        return KcGeoStiffnessOutput(K_geo=K_geo)


class KcClosestPointStiffnessProcess(
    SolverProcess[ContactForceStStiffnessInput, KcClosestPointStiffnessOutput],
):
    """K_c 最近接点追従項 K_closest を計算する Process (status-351 Phase C-2).

    数理台帳 ``docs/math/03_huber_contact_penalty.md#eq-kc-full-decomposition``
    の項 $\\boldsymbol{K}_{\\mathrm{closest}}
    = -\\,(\\partial p_n/\\partial(s,t))\\;(\\boldsymbol{c}\\otimes\\hat{\\boldsymbol{n}})
    \\,\\partial(s,t)/\\partial\\boldsymbol{u}$ を担う。現行の K_st 全体のうち
    "(s,t) 摂動に伴う p_n 追従" 成分を分離したもので、残差
    $-\\,p_n\\,(\\partial \\boldsymbol{c}/\\partial(s,t))\\cdot\\hat{\\boldsymbol{n}}
    -\\,p_n\\,\\boldsymbol{c}\\cdot(\\partial\\hat{\\boldsymbol{n}}/\\partial(s,t))$
    は ``ContactForceStStiffnessProcess`` が K_st として返す。

    status-351 Phase C-2: ``ContactForceStStiffnessProcess._assemble_term_coo``
    の共通セットアップを再利用（StJacobian / ∂n/∂s / ∂p_n/∂s を 2 度計算する
    ことを避けるため、共通 classmethod を呼び出す）。
    """

    meta = ProcessMeta(
        name="KcClosestPointStiffness",
        module="solve",
        version="1.0.0",
        document_path="docs/contact_force.md",
    )
    # uses: ContactForceStStiffnessProcess._assemble_term_coo 経由で共通セットアップを
    # 再利用する（status-351 Phase C-2 の設計。重計算の 2 度走りを避ける）。
    uses = [ComputeStJacobianProcess, ContactForceStStiffnessProcess]
    contracts: ClassVar[tuple[MathematicalContract, ...]] = (_K_C_TERM_EXPANSION_CONTRACT,)

    def process(self, inp: ContactForceStStiffnessInput) -> KcClosestPointStiffnessOutput:
        K_closest = ContactForceStStiffnessProcess._assemble_term_coo(inp, term="closest")
        return KcClosestPointStiffnessOutput(K_closest=K_closest)


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

    def compute_gap_cull_threshold(self, k_pen: float) -> float:
        """K_st distance culling の gap 閾値を返す (status-324).

        Huber 遷移幅から gap 閾値を自動計算。この gap を超えるペアは
        K_st 寄与がゼロなのでアセンブリをスキップできる。
        摩擦 K_st にも同一の閾値を適用する。
        """
        if k_pen <= 0:
            return float("inf")
        delta_h = self._resolve_delta_h(k_pen)
        return (delta_h / k_pen if delta_h > 0 else 0.0) + 1e-8

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

                # adj global node indices (N, 4) — dict→配列ルックアップ（status-310）
                _max_elem = max(max(elem_a_act), max(elem_b_act)) + 1
                _adj_arr = np.full((_max_elem, 2), -1, dtype=int)
                for _eidx, (_al, _ar) in _adj_node_map.items():
                    if _eidx < _max_elem:
                        _adj_arr[_eidx] = [_al, _ar]
                adj_gnodes = np.column_stack(
                    [
                        _adj_arr[elem_a_act, 0],
                        _adj_arr[elem_a_act, 1],
                        _adj_arr[elem_b_act, 0],
                        _adj_arr[elem_b_act, 1],
                    ]
                )

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
            # status-321: K_st は既に COO（tocsr() を skip 済み）。tocoo() は
            # no-op だが、CSR 経路でも動作するようガード付きで結合する。
            K_st_coo = K_st if isinstance(K_st, sp.coo_matrix) else K_st.tocoo()
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
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix | sp.coo_matrix]:
        """K_mat, K_geo, K_st を個別に返す（status-291: 個別FD検証用）.

        K_c = K_mat - K_geo + K_st

        status-321: K_st は ContactForceStStiffnessProcess 経由で COO 形式の
        まま返される場合がある（往復変換削減のため）。`+ - @` 等の sparse 演算は
        CSR/COO 互換で動作する。

        status-350 Phase C-1: 3 項を ``KcNormalStiffnessProcess`` /
        ``KcGeoStiffnessProcess`` / ``ContactForceStStiffnessProcess`` に抽出。

        status-351 Phase C-2: 更に ``KcHermiteNonlocalStiffnessProcess`` /
        ``KcClosestPointStiffnessProcess`` を分離し 5 項構成へ。本メソッドは
        後方互換のため 3-tuple を保つが、内部では:
          K_mat = K_mat_nn + K_hermite_adj  (KcNormal + KcHermiteNonlocal)
          K_st  = K_closest + K_st_residual  (KcClosest + ContactForceSt)
        として 5 Process の出力を加算する。``TermExpansionContract
        ("K_c_term_expansion")`` の 5 providers と 1:1 対応する。
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

        # ── K_mat_nn + K_hermite_adj + K_geo (KcTermAssemblyInput 3 Process) ──
        term_input = KcTermAssemblyInput(
            pairs=manager.pairs,
            k_pen=k_pen,
            delta_h=delta_h,
            ndof_total=ndof,
            ndof_per_node=self._ndof_per_node,
            penalty_exponent=self._penalty_exponent,
            use_hermite=_use_hermite,
            node_counts=_node_counts,
            adj_node_map=_adj_node_map,
            adj_node_counts=_adj_node_counts,
        )
        K_mat_nn = KcNormalStiffnessProcess().process(term_input).K_mat
        K_hermite_adj = KcHermiteNonlocalStiffnessProcess().process(term_input).K_hermite_adj
        K_geo = KcGeoStiffnessProcess().process(term_input).K_geo
        # 後方互換: 呼び出し側は K_mat = K_mat_nn + K_hermite_adj を期待
        K_mat = K_mat_nn + K_hermite_adj

        # ── K_closest + K_st 残差 (KcClosestPoint + ContactForceSt) ──
        K_st: sp.csr_matrix | sp.coo_matrix = zero
        if _use_st and node_coords is not None:
            st_input = ContactForceStStiffnessInput(
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
            K_closest = KcClosestPointStiffnessProcess().process(st_input).K_closest
            K_st_residual = ContactForceStStiffnessProcess().process(st_input).K_st
            # 後方互換: 呼び出し側は K_st = K_closest + K_st_residual を期待
            K_st = K_closest + K_st_residual

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
