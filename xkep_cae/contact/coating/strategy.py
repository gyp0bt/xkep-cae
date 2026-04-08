"""Coating Strategy 具象実装.

被膜接触モデル（Kelvin-Voigt弾性+粘性ダッシュポット）を Strategy として実装する。

実装:
- KelvinVoigtCoatingProcess: 弾性+粘性被膜モデル（status-137/140）
- NoCoatingProcess: 被膜なし（ゼロ返却）
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state
from xkep_cae.core import ProcessMeta, SolverProcess

# ── バリア関数ヘルパー（status-303） ────────────────────────

_BARRIER_CLAMP = 1e-3  # min(1 - ratio) で特異点回避


def _coat_total(pair: object) -> float:
    """ペアの被膜総厚さ（A側+B側）."""
    return (pair.radius_a - pair.core_radius_a) + (pair.radius_b - pair.core_radius_b)


def _barrier_p_n(k_coat: float, delta: float, delta_max: float) -> float:
    """バリア関数: f = k * δ / (1 - δ/δ_max).

    δ→δ_max で力→∞。芯線貫入を防止する。
    """
    if delta_max <= 1e-30:
        return k_coat * delta
    ratio = delta / delta_max
    denom = max(1.0 - ratio, _BARRIER_CLAMP)
    return k_coat * delta / denom


def _barrier_k_eff(k_coat: float, delta: float, delta_max: float) -> float:
    """バリア関数の接線剛性: df/dδ = k / (1 - δ/δ_max)²."""
    if delta_max <= 1e-30:
        return k_coat
    ratio = delta / delta_max
    denom = max(1.0 - ratio, _BARRIER_CLAMP)
    return k_coat / (denom**2)


def _barrier_energy(k_coat: float, delta: float, delta_max: float) -> float:
    """バリア関数の弾性エネルギー（解析積分）.

    E = ∫₀^δ k·x/(1-x/δ_max) dx = k·δ_max²·[-ln(1-r) - r]  (r = δ/δ_max)

    δ→δ_max でエネルギー→∞（芯線貫入防止のペナルティ）。
    """
    if delta <= 0.0:
        return 0.0
    if delta_max <= 1e-30:
        # 線形フォールバック
        return 0.5 * k_coat * delta**2
    ratio = delta / delta_max
    ratio_clamped = min(ratio, 1.0 - _BARRIER_CLAMP)
    return k_coat * delta_max**2 * (-math.log(1.0 - ratio_clamped) - ratio_clamped)


class NoCoatingProcess(SolverProcess[None, None]):
    """被膜なし（ゼロ返却）.

    coating_stiffness == 0 の場合に使用。全メソッドがゼロを返す。
    """

    meta = ProcessMeta(
        name="NoCoating",
        module="solve",
        version="1.0.0",
        document_path="docs/coating.md",
    )

    def forces(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        dt: float,
    ) -> np.ndarray:
        """ゼロ力を返す."""
        ndof_per_node = getattr(config, "ndof_per_node", 6)
        return np.zeros(len(node_coords) * ndof_per_node)

    def stiffness(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        ndof_total: int,
        dt: float,
    ) -> sp.csr_matrix:
        """ゼロ剛性行列を返す."""
        return sp.csr_matrix((ndof_total, ndof_total))

    def friction_forces(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        u_cur: np.ndarray,
        u_ref: np.ndarray,
    ) -> np.ndarray:
        """ゼロ摩擦力を返す."""
        return np.zeros_like(u_cur)

    def friction_stiffness(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        ndof_total: int,
    ) -> sp.csr_matrix:
        """ゼロ摩擦剛性行列を返す."""
        return sp.csr_matrix((ndof_total, ndof_total))

    def energy(self, pairs: list, config: object) -> float:
        """被膜弾性エネルギー（常にゼロ）."""
        return 0.0

    def process(self, input_data: None) -> None:
        """Strategy として使用。直接呼出不要."""
        return None


class KelvinVoigtCoatingProcess(SolverProcess[None, None]):
    """Kelvin-Voigt 被膜接触モデル（status-137/140）.

    弾性スプリング + 粘性ダッシュポット:
      f_coat = k * δ + c * δ̇

    被膜圧縮量 δ = max(0, t_coat_total - gap_core) に対して
    法線力 + 被膜摩擦（Coulomb return mapping）を生成する。
    """

    meta = ProcessMeta(
        name="KelvinVoigtCoating",
        module="solve",
        version="1.0.0",
        document_path="docs/coating.md",
    )

    def forces(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        dt: float,
    ) -> np.ndarray:
        """被膜圧縮による接触力ベクトルを計算する."""
        k_coat = config.coating_stiffness
        c_coat = config.coating_damping
        coords = np.asarray(node_coords, dtype=float)
        n_nodes = len(coords)
        ndof_per_node = getattr(config, "ndof_per_node", 6)
        f_coat = np.zeros(n_nodes * ndof_per_node)

        # status-303: バリア関数判定
        use_barrier = (
            getattr(config, "coating_barrier", True)
            and getattr(config, "coating_thickness", 0.0) > 0.0
        )

        for pair in pairs:
            cc = pair.state.coating_compression
            if cc <= 0.0 and pair.state.coating_compression_prev <= 0.0:
                continue

            # 被膜法線力
            if use_barrier:
                ct = _coat_total(pair)
                f_n = _barrier_p_n(k_coat, cc, ct)
            else:
                f_n = k_coat * cc

            # 粘性項（Kelvin-Voigt）
            if c_coat > 0.0 and dt > 0.0:
                delta_dot = (cc - pair.state.coating_compression_prev) / dt
                f_n += c_coat * delta_dot

            if abs(f_n) < 1e-30:
                continue

            n_vec = pair.state.normal
            s = pair.state.s
            t = pair.state.t

            nA0 = int(pair.nodes_a[0])
            nA1 = int(pair.nodes_a[1])
            nB0 = int(pair.nodes_b[0])
            nB1 = int(pair.nodes_b[1])

            fA = -f_n * n_vec
            fB = f_n * n_vec

            dpn = ndof_per_node
            f_coat[nA0 * dpn : nA0 * dpn + 3] += (1.0 - s) * fA
            f_coat[nA1 * dpn : nA1 * dpn + 3] += s * fA
            f_coat[nB0 * dpn : nB0 * dpn + 3] += (1.0 - t) * fB
            f_coat[nB1 * dpn : nB1 * dpn + 3] += t * fB

        return f_coat

    def stiffness(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        ndof_total: int,
        dt: float,
    ) -> sp.csr_matrix:
        """被膜Kelvin-Voigtモデルの接線剛性行列を計算する."""
        k_coat = config.coating_stiffness
        c_coat = config.coating_damping
        ndof_per_node = getattr(config, "ndof_per_node", 6)

        # status-303: バリア関数判定
        use_barrier = (
            getattr(config, "coating_barrier", True)
            and getattr(config, "coating_thickness", 0.0) > 0.0
        )

        # 線形モデル時はグローバル k_eff（バリア時はペア毎に計算）
        k_eff_global = k_coat
        if c_coat > 0.0 and dt > 0.0:
            k_eff_global += c_coat / dt

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for pair in pairs:
            cc = pair.state.coating_compression
            if cc <= 1e-15:
                continue

            # ペア毎の実効接線剛性
            if use_barrier:
                ct = _coat_total(pair)
                k_eff = _barrier_k_eff(k_coat, cc, ct)
                if c_coat > 0.0 and dt > 0.0:
                    k_eff += c_coat / dt
            else:
                k_eff = k_eff_global

            n_vec = pair.state.normal
            s = pair.state.s
            t = pair.state.t

            nA0 = int(pair.nodes_a[0])
            nA1 = int(pair.nodes_a[1])
            nB0 = int(pair.nodes_b[0])
            nB1 = int(pair.nodes_b[1])

            weights = [-(1.0 - s), -s, (1.0 - t), t]
            node_ids = [nA0, nA1, nB0, nB1]

            gdofs = []
            g_n = np.zeros(12)
            for idx, (w, nid) in enumerate(zip(weights, node_ids, strict=True)):
                base = nid * ndof_per_node
                gdofs.extend([base, base + 1, base + 2])
                g_n[3 * idx : 3 * idx + 3] = w * n_vec

            K_local = k_eff * np.outer(g_n, g_n)

            for i_loc in range(12):
                for j_loc in range(12):
                    val = K_local[i_loc, j_loc]
                    if abs(val) > 1e-30:
                        rows.append(gdofs[i_loc])
                        cols.append(gdofs[j_loc])
                        data.append(val)

        if not data:
            return sp.csr_matrix((ndof_total, ndof_total))

        return sp.coo_matrix((data, (rows, cols)), shape=(ndof_total, ndof_total)).tocsr()

    def friction_forces(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        u_cur: np.ndarray,
        u_ref: np.ndarray,
    ) -> np.ndarray:
        """被膜接触面のCoulomb摩擦力を計算する."""
        from xkep_cae.contact.friction.law_friction import _return_mapping_core

        mu = config.coating_mu
        if mu <= 0.0:
            return np.zeros_like(u_cur)

        k_coat = config.coating_stiffness
        k_t_ratio = config.coating_k_t_ratio
        k_t = k_coat * k_t_ratio

        ndof_per_node = getattr(config, "ndof_per_node", 6)
        n_nodes = len(node_coords)
        f_fric = np.zeros(n_nodes * ndof_per_node)
        du = u_cur - u_ref

        # status-303: バリア関数判定
        use_barrier = (
            getattr(config, "coating_barrier", True)
            and getattr(config, "coating_thickness", 0.0) > 0.0
        )

        for pi, pair in enumerate(pairs):
            cc = pair.state.coating_compression
            if cc <= 0.0:
                # 非接触: 摩擦履歴リセット
                pairs[pi] = _evolve_pair(
                    pair,
                    state=_evolve_state(
                        pair.state,
                        coating_z_t=np.zeros(2),
                        coating_stick=True,
                        coating_q_trial_norm=0.0,
                        coating_dissipation=0.0,
                    ),
                )
                continue

            # 被膜法線力（弾性成分のみ — 摩擦のCoulomb条件に使用）
            if use_barrier:
                ct = _coat_total(pair)
                p_n = _barrier_p_n(k_coat, cc, ct)
            else:
                p_n = k_coat * cc

            # 接線相対変位増分
            s = pair.state.s
            t = pair.state.t
            t1 = pair.state.tangent1
            t2 = pair.state.tangent2

            nA0, nA1 = int(pair.nodes_a[0]), int(pair.nodes_a[1])
            nB0, nB1 = int(pair.nodes_b[0]), int(pair.nodes_b[1])
            dpn = ndof_per_node
            du_A0 = du[nA0 * dpn : nA0 * dpn + 3]
            du_A1 = du[nA1 * dpn : nA1 * dpn + 3]
            du_B0 = du[nB0 * dpn : nB0 * dpn + 3]
            du_B1 = du[nB1 * dpn : nB1 * dpn + 3]

            du_A = (1.0 - s) * du_A0 + s * du_A1
            du_B = (1.0 - t) * du_B0 + t * du_B1
            du_rel = du_B - du_A

            delta_ut = np.array([float(np.dot(du_rel, t1)), float(np.dot(du_rel, t2))])

            # Coulomb return mapping（純粋関数）
            q, is_stick, q_trial_norm, dissipation = _return_mapping_core(
                pair.state.coating_z_t.copy(), delta_ut, k_t, p_n, mu
            )

            # 状態更新
            pairs[pi] = _evolve_pair(
                pair,
                state=_evolve_state(
                    pair.state,
                    coating_z_t=q.copy(),
                    coating_stick=is_stick,
                    coating_q_trial_norm=q_trial_norm,
                    coating_dissipation=dissipation,
                ),
            )

            # 接線力を節点力に分配
            f_t_3d = q[0] * t1 + q[1] * t2

            fA = -f_t_3d
            fB = f_t_3d

            f_fric[nA0 * dpn : nA0 * dpn + 3] += (1.0 - s) * fA
            f_fric[nA1 * dpn : nA1 * dpn + 3] += s * fA
            f_fric[nB0 * dpn : nB0 * dpn + 3] += (1.0 - t) * fB
            f_fric[nB1 * dpn : nB1 * dpn + 3] += t * fB

        return f_fric

    def friction_stiffness(
        self,
        pairs: list,
        node_coords: np.ndarray,
        config: object,
        ndof_total: int,
    ) -> sp.csr_matrix:
        """被膜摩擦の接線剛性行列を計算する."""
        from xkep_cae.contact.friction.law_friction import _tangent_2x2_core

        mu = config.coating_mu
        if mu <= 0.0:
            return sp.csr_matrix((ndof_total, ndof_total))

        k_coat = config.coating_stiffness
        k_t = k_coat * config.coating_k_t_ratio
        ndof_per_node = getattr(config, "ndof_per_node", 6)

        # status-303: バリア関数判定
        use_barrier = (
            getattr(config, "coating_barrier", True)
            and getattr(config, "coating_thickness", 0.0) > 0.0
        )

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for pair in pairs:
            cc = pair.state.coating_compression
            if cc <= 0.0:
                continue

            if use_barrier:
                ct = _coat_total(pair)
                p_n = _barrier_p_n(k_coat, cc, ct)
            else:
                p_n = k_coat * cc

            # 2x2接線剛性
            D_t = _tangent_2x2_core(
                k_t=k_t,
                p_n=p_n,
                mu=mu,
                z_t=pair.state.coating_z_t,
                q_trial_norm=pair.state.coating_q_trial_norm,
                is_stick=pair.state.coating_stick,
            )

            # 3D接線方向への展開
            t1 = pair.state.tangent1
            t2 = pair.state.tangent2
            s = pair.state.s
            t = pair.state.t

            nA0 = int(pair.nodes_a[0])
            nA1 = int(pair.nodes_a[1])
            nB0 = int(pair.nodes_b[0])
            nB1 = int(pair.nodes_b[1])

            T_mat = np.column_stack([t1, t2])  # (3, 2)
            K_3x3 = T_mat @ D_t @ T_mat.T  # (3, 3)

            weights_A = [-(1.0 - s), -s]
            weights_B = [(1.0 - t), t]
            node_ids = [nA0, nA1, nB0, nB1]
            weights = weights_A + weights_B

            dpn = ndof_per_node
            gdofs = []
            for nid in node_ids:
                gdofs.append(nid * dpn)

            for i_node in range(4):
                for j_node in range(4):
                    w_ij = weights[i_node] * weights[j_node]
                    K_block = w_ij * K_3x3
                    gi = gdofs[i_node]
                    gj = gdofs[j_node]
                    for ii in range(3):
                        for jj in range(3):
                            val = K_block[ii, jj]
                            if abs(val) > 1e-30:
                                rows.append(gi + ii)
                                cols.append(gj + jj)
                                data.append(val)

        if not data:
            return sp.csr_matrix((ndof_total, ndof_total))

        return sp.coo_matrix((data, (rows, cols)), shape=(ndof_total, ndof_total)).tocsr()

    def energy(self, pairs: list, config: object) -> float:
        """被膜弾性エネルギーの総和.

        バリア関数: E = k·δ_max²·[-ln(1 - δ/δ_max) - δ/δ_max]
        線形モデル: E = 0.5·k·δ²
        """
        k_coat = config.coating_stiffness
        use_barrier = (
            getattr(config, "coating_barrier", True)
            and getattr(config, "coating_thickness", 0.0) > 0.0
        )
        total = 0.0
        for pair in pairs:
            cc = pair.state.coating_compression
            if cc <= 0.0:
                continue
            if use_barrier:
                ct = _coat_total(pair)
                total += _barrier_energy(k_coat, cc, ct)
            else:
                total += 0.5 * k_coat * cc**2
        return total

    def process(self, input_data: None) -> None:
        """Strategy として使用。直接呼出不要."""
        return None


def _create_coating_strategy(
    *,
    coating_stiffness: float = 0.0,
) -> NoCoatingProcess | KelvinVoigtCoatingProcess:
    """CoatingStrategy ファクトリ.

    Args:
        coating_stiffness: 被膜接触剛性。0 の場合 NoCoating を返す。

    Returns:
        CoatingStrategy 実装インスタンス
    """
    if coating_stiffness <= 0.0:
        return NoCoatingProcess()
    return KelvinVoigtCoatingProcess()
