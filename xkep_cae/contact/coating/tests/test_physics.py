"""Kelvin-Voigt 被膜モデル物理検証テスト.

被膜接触モデルの物理的妥当性を検証する:
- 弾性力の線形性
- 粘性減衰の正負
- 作用反作用
- 接線剛性の対称性・正定値性
- 摩擦のCoulomb則準拠
"""

from __future__ import annotations

import numpy as np

from xkep_cae.contact._contact_pair import (
    _ContactPairOutput,
    _ContactStateOutput,
)
from xkep_cae.contact.coating.strategy import KelvinVoigtCoatingProcess


def _make_config(
    *,
    k_coat: float = 1e6,
    c_coat: float = 0.0,
    mu: float = 0.0,
    k_t_ratio: float = 0.5,
    coating_barrier: bool = False,
    coating_thickness: float = 0.0,
) -> object:
    """テスト用の接触設定を作成.

    _ContactConfigInput は ndof_per_node を持たないため、
    SimpleNamespace で代替する。
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        coating_stiffness=k_coat,
        coating_damping=c_coat,
        coating_mu=mu,
        coating_k_t_ratio=k_t_ratio,
        ndof_per_node=6,
        coating_barrier=coating_barrier,
        coating_thickness=coating_thickness,
    )


def _make_pair(
    *,
    coating_compression: float = 0.0,
    coating_compression_prev: float = 0.0,
    normal: np.ndarray | None = None,
    s: float = 0.5,
    t: float = 0.5,
    tangent1: np.ndarray | None = None,
    tangent2: np.ndarray | None = None,
    coating_z_t: np.ndarray | None = None,
    coating_stick: bool = True,
    coating_q_trial_norm: float = 0.0,
    radius_a: float = 0.0,
    radius_b: float = 0.0,
    core_radius_a: float = 0.0,
    core_radius_b: float = 0.0,
) -> _ContactPairOutput:
    """テスト用の接触ペアを作成."""
    if normal is None:
        normal = np.array([0.0, 1.0, 0.0])
    if tangent1 is None:
        tangent1 = np.array([1.0, 0.0, 0.0])
    if tangent2 is None:
        tangent2 = np.array([0.0, 0.0, 1.0])
    if coating_z_t is None:
        coating_z_t = np.zeros(2)
    state = _ContactStateOutput(
        s=s,
        t=t,
        normal=normal,
        tangent1=tangent1,
        tangent2=tangent2,
        coating_compression=coating_compression,
        coating_compression_prev=coating_compression_prev,
        coating_z_t=coating_z_t,
        coating_stick=coating_stick,
        coating_q_trial_norm=coating_q_trial_norm,
    )
    return _ContactPairOutput(
        elem_a=0,
        elem_b=1,
        nodes_a=np.array([0, 1]),
        nodes_b=np.array([2, 3]),
        state=state,
        radius_a=radius_a,
        radius_b=radius_b,
        core_radius_a=core_radius_a,
        core_radius_b=core_radius_b,
    )


# 4ノード × 3座標
_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],  # A0
        [1.0, 0.0, 0.0],  # A1
        [0.0, 0.5, 0.0],  # B0
        [1.0, 0.5, 0.0],  # B1
    ],
    dtype=float,
)
_NDOF = 4 * 6  # 4ノード × 6DOF


class TestKelvinVoigtCoatingPhysics:
    """Kelvin-Voigt 被膜モデルの物理的妥当性テスト."""

    # ── 弾性力 ──────────────────────────────────────

    def test_no_compression_no_force(self) -> None:
        """δ=0 なら力ゼロ."""
        p = KelvinVoigtCoatingProcess()
        pair = _make_pair(coating_compression=0.0)
        config = _make_config()
        f = p.forces([pair], _COORDS, config, 0.01)
        assert np.allclose(f, 0.0)

    def test_force_proportional_to_compression(self) -> None:
        """力が圧縮量に比例する（弾性項）."""
        p = KelvinVoigtCoatingProcess()
        config = _make_config(k_coat=1e6, c_coat=0.0)

        delta1 = 0.01
        delta2 = 0.02
        pair1 = _make_pair(coating_compression=delta1)
        pair2 = _make_pair(coating_compression=delta2)

        f1 = p.forces([pair1], _COORDS, config, 0.01)
        f2 = p.forces([pair2], _COORDS, config, 0.01)

        # 2倍の圧縮量 → 2倍の力（弾性線形）
        nonzero = np.abs(f1) > 1e-30
        assert np.any(nonzero), "力がゼロ"
        ratio = f2[nonzero] / f1[nonzero]
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-10)

    def test_force_direction_action_reaction(self) -> None:
        """A側とB側に逆向き等大の力が作用する（作用反作用）."""
        p = KelvinVoigtCoatingProcess()
        config = _make_config(k_coat=1e6, c_coat=0.0)
        # s=0.5, t=0.5 → 各セグメント中央
        pair = _make_pair(
            coating_compression=0.01,
            s=0.5,
            t=0.5,
            normal=np.array([0.0, 1.0, 0.0]),
        )

        f = p.forces([pair], _COORDS, config, 0.01)
        f_nodes = f.reshape(4, 6)

        # A側合力（ノード0,1）
        f_A = f_nodes[0, :3] + f_nodes[1, :3]
        # B側合力（ノード2,3）
        f_B = f_nodes[2, :3] + f_nodes[3, :3]

        # 作用反作用: f_A + f_B ≈ 0
        np.testing.assert_allclose(f_A + f_B, 0.0, atol=1e-12)

    def test_force_direction_repulsive(self) -> None:
        """被膜力が法線方向反発力（B側は法線方向、A側は逆法線方向）."""
        p = KelvinVoigtCoatingProcess()
        config = _make_config(k_coat=1e6, c_coat=0.0)
        normal = np.array([0.0, 1.0, 0.0])
        pair = _make_pair(coating_compression=0.01, normal=normal)

        f = p.forces([pair], _COORDS, config, 0.01)
        f_nodes = f.reshape(4, 6)

        # B側は法線方向に正の力（反発）
        f_B = f_nodes[2, :3] + f_nodes[3, :3]
        assert np.dot(f_B, normal) > 0, "B側は法線方向に正であるべき"

        # A側は逆法線方向（引き込まれる方向）
        f_A = f_nodes[0, :3] + f_nodes[1, :3]
        assert np.dot(f_A, normal) < 0, "A側は逆法線方向であるべき"

    # ── 粘性減衰 ──────────────────────────────────────

    def test_viscous_force_zero_when_no_damping(self) -> None:
        """c_coat=0 のとき粘性力ゼロ."""
        p = KelvinVoigtCoatingProcess()
        config_no_damp = _make_config(k_coat=1e6, c_coat=0.0)
        config_damp = _make_config(k_coat=1e6, c_coat=1e4)

        # 圧縮が増加中（δ̇ > 0）
        pair = _make_pair(coating_compression=0.02, coating_compression_prev=0.01)
        dt = 0.01

        f_no = p.forces([pair], _COORDS, config_no_damp, dt)
        f_yes = p.forces([pair], _COORDS, config_damp, dt)

        # 粘性項がある方が力が大きい（圧縮増加 → 正の粘性力が加算）
        norm_no = np.linalg.norm(f_no)
        norm_yes = np.linalg.norm(f_yes)
        assert norm_yes > norm_no, "粘性減衰により力が増大すべき"

    def test_viscous_force_proportional_to_rate(self) -> None:
        """粘性力が圧縮速度に比例する."""
        p = KelvinVoigtCoatingProcess()
        k_coat = 1e6
        c_coat = 1e4
        config = _make_config(k_coat=k_coat, c_coat=c_coat)
        dt = 0.01

        # 同じ弾性圧縮量、異なる速度
        pair_fast = _make_pair(coating_compression=0.01, coating_compression_prev=0.0)
        pair_slow = _make_pair(coating_compression=0.01, coating_compression_prev=0.005)

        f_fast = p.forces([pair_fast], _COORDS, config, dt)
        f_slow = p.forces([pair_slow], _COORDS, config, dt)

        # 弾性項を除外すると粘性項のみ
        config_elastic_only = _make_config(k_coat=k_coat, c_coat=0.0)
        f_elastic = p.forces([pair_fast], _COORDS, config_elastic_only, dt)

        viscous_fast = np.linalg.norm(f_fast - f_elastic)
        # pair_slow は同じ弾性圧縮量
        f_elastic_slow = p.forces([pair_slow], _COORDS, config_elastic_only, dt)
        viscous_slow = np.linalg.norm(f_slow - f_elastic_slow)

        # δ̇_fast = 0.01/0.01 = 1.0, δ̇_slow = 0.005/0.01 = 0.5
        assert viscous_fast > 0, "粘性力は正"
        np.testing.assert_allclose(viscous_fast / viscous_slow, 2.0, rtol=0.01)

    def test_stable_at_equilibrium(self) -> None:
        """δ=const → δ̇=0 → 粘性力ゼロ（弾性力のみ）."""
        p = KelvinVoigtCoatingProcess()
        config_elastic = _make_config(k_coat=1e6, c_coat=0.0)
        config_kv = _make_config(k_coat=1e6, c_coat=1e4)

        # 圧縮一定（δ̇=0）
        pair = _make_pair(coating_compression=0.01, coating_compression_prev=0.01)
        dt = 0.01

        f_elastic = p.forces([pair], _COORDS, config_elastic, dt)
        f_kv = p.forces([pair], _COORDS, config_kv, dt)

        np.testing.assert_allclose(f_elastic, f_kv, atol=1e-12)

    # ── 剛性行列 ──────────────────────────────────────

    def test_effective_stiffness_composition(self) -> None:
        """有効剛性 k_eff = k + c/dt."""
        p = KelvinVoigtCoatingProcess()
        k_coat = 1e6
        c_coat = 1e4
        dt = 0.01
        config = _make_config(k_coat=k_coat, c_coat=c_coat)
        config_elastic_only = _make_config(k_coat=k_coat, c_coat=0.0)

        pair = _make_pair(coating_compression=0.01)

        K_elastic = p.stiffness([pair], _COORDS, config_elastic_only, _NDOF, dt)
        K_kv = p.stiffness([pair], _COORDS, config, _NDOF, dt)

        # k_eff / k_coat = (k + c/dt) / k = 1 + c/(k*dt)
        expected_ratio = 1.0 + c_coat / (k_coat * dt)
        # 絶対値合計で比較（符号キャンセルを避ける）
        actual_ratio = np.abs(K_kv.toarray()).sum() / np.abs(K_elastic.toarray()).sum()
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-10)

    def test_stiffness_symmetry(self) -> None:
        """接線剛性行列が対称."""
        p = KelvinVoigtCoatingProcess()
        config = _make_config(k_coat=1e6, c_coat=1e4)
        pair = _make_pair(coating_compression=0.01)
        dt = 0.01

        K = p.stiffness([pair], _COORDS, config, _NDOF, dt).toarray()

        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_stiffness_positive_semi_definite(self) -> None:
        """接線剛性行列が半正定値."""
        p = KelvinVoigtCoatingProcess()
        config = _make_config(k_coat=1e6, c_coat=1e4)
        pair = _make_pair(coating_compression=0.01)
        dt = 0.01

        K = p.stiffness([pair], _COORDS, config, _NDOF, dt).toarray()

        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8), f"負の固有値: {eigenvalues[eigenvalues < -1e-8]}"

    def test_no_compression_no_stiffness(self) -> None:
        """δ=0 なら剛性行列ゼロ."""
        p = KelvinVoigtCoatingProcess()
        config = _make_config(k_coat=1e6)
        pair = _make_pair(coating_compression=0.0)
        dt = 0.01

        K = p.stiffness([pair], _COORDS, config, _NDOF, dt)
        assert K.nnz == 0

    # ── 摩擦 ──────────────────────────────────────

    def test_friction_zero_without_compression(self) -> None:
        """δ=0 では摩擦力ゼロ."""
        p = KelvinVoigtCoatingProcess()
        config = _make_config(k_coat=1e6, mu=0.3)
        pair = _make_pair(coating_compression=0.0)
        u_cur = np.random.RandomState(42).randn(_NDOF) * 0.001
        u_ref = np.zeros(_NDOF)

        pairs = [pair]
        f = p.friction_forces(pairs, _COORDS, config, u_cur, u_ref)
        assert np.allclose(f, 0.0)

    def test_friction_zero_with_zero_mu(self) -> None:
        """μ=0 では摩擦力ゼロ."""
        p = KelvinVoigtCoatingProcess()
        config = _make_config(k_coat=1e6, mu=0.0)
        pair = _make_pair(coating_compression=0.01)
        u_cur = np.random.RandomState(42).randn(_NDOF) * 0.001
        u_ref = np.zeros(_NDOF)

        f = p.friction_forces([pair], _COORDS, config, u_cur, u_ref)
        assert np.allclose(f, 0.0)

    def test_friction_uses_elastic_force_only(self) -> None:
        """摩擦のCoulomb条件に弾性力のみ使用（粘性項含まない）.

        p_n = k * δ であり、c * δ̇ は含まない。
        """
        p = KelvinVoigtCoatingProcess()
        k_coat = 1e6
        mu = 0.3

        # 大変位でスリップさせる
        config = _make_config(k_coat=k_coat, c_coat=0.0, mu=mu)
        config_damp = _make_config(k_coat=k_coat, c_coat=1e6, mu=mu)

        delta = 0.01
        # 接線方向に大きな変位
        u_cur = np.zeros(_NDOF)
        u_cur[0 * 6 + 0] = -0.1  # A0: x方向
        u_cur[1 * 6 + 0] = -0.1  # A1: x方向
        u_ref = np.zeros(_NDOF)

        # スリップ時の摩擦力: |q| = μ * p_n = μ * k * δ
        # c_coat が異なっても摩擦力は同じ
        pairs1 = [_make_pair(coating_compression=delta, coating_compression_prev=0.0)]
        pairs2 = [_make_pair(coating_compression=delta, coating_compression_prev=0.0)]
        f1 = p.friction_forces(pairs1, _COORDS, config, u_cur, u_ref)
        f2 = p.friction_forces(pairs2, _COORDS, config_damp, u_cur, u_ref)

        np.testing.assert_allclose(f1, f2, atol=1e-10)

    def test_coulomb_slip_limit(self) -> None:
        """スリップ時の摩擦力が μ*p_n を超えない."""
        p = KelvinVoigtCoatingProcess()
        k_coat = 1e6
        mu = 0.3
        delta = 0.01
        config = _make_config(k_coat=k_coat, mu=mu, k_t_ratio=0.5)

        # 大きな接線変位 → スリップ
        u_cur = np.zeros(_NDOF)
        u_cur[0 * 6 + 0] = -1.0  # A0: 接線方向に大変位
        u_cur[1 * 6 + 0] = -1.0
        u_ref = np.zeros(_NDOF)

        pair = _make_pair(coating_compression=delta)
        pairs = [pair]
        f = p.friction_forces(pairs, _COORDS, config, u_cur, u_ref)
        f_nodes = f.reshape(4, 6)

        # p_n = k * δ
        p_n = k_coat * delta
        coulomb_limit = mu * p_n

        # 総摩擦力は 0 であるべき（作用反作用でキャンセル）
        # 各セグメント側の力を確認
        f_A = f_nodes[0, :3] + f_nodes[1, :3]
        f_friction_mag = np.linalg.norm(f_A)
        assert f_friction_mag <= coulomb_limit * (1.0 + 1e-8), (
            f"摩擦力 {f_friction_mag:.4e} が Coulomb 限界 {coulomb_limit:.4e} を超過"
        )

    def test_friction_history_reset_on_separation(self) -> None:
        """非接触時に摩擦履歴がリセットされる."""
        p = KelvinVoigtCoatingProcess()
        config = _make_config(k_coat=1e6, mu=0.3)

        u_cur = np.zeros(_NDOF)
        u_ref = np.zeros(_NDOF)

        # 分離（compression=0）
        separated = _make_pair(
            coating_compression=0.0,
            coating_z_t=np.array([100.0, 200.0]),
        )
        pairs = [separated]
        p.friction_forces(pairs, _COORDS, config, u_cur, u_ref)

        # 摩擦履歴がリセットされている
        updated_pair = pairs[0]
        np.testing.assert_allclose(updated_pair.state.coating_z_t, 0.0, atol=1e-15)
        assert updated_pair.state.coating_stick is True

    # ── 摩擦剛性行列 ──────────────────────────────────

    def test_friction_stiffness_symmetry(self) -> None:
        """摩擦接線剛性行列が対称（stick 状態）."""
        p = KelvinVoigtCoatingProcess()
        config = _make_config(k_coat=1e6, mu=0.3, k_t_ratio=0.5)
        pair = _make_pair(
            coating_compression=0.01,
            coating_stick=True,
            coating_z_t=np.array([10.0, 20.0]),
            coating_q_trial_norm=22.36,
        )

        K_fric = p.friction_stiffness([pair], _COORDS, config, _NDOF).toarray()

        np.testing.assert_allclose(K_fric, K_fric.T, atol=1e-10)

    def test_friction_stiffness_zero_when_separated(self) -> None:
        """分離時の摩擦剛性行列がゼロ."""
        p = KelvinVoigtCoatingProcess()
        config = _make_config(k_coat=1e6, mu=0.3)
        pair = _make_pair(coating_compression=0.0)

        K = p.friction_stiffness([pair], _COORDS, config, _NDOF)
        assert K.nnz == 0


# ── バリア関数テスト（status-303）──────────────────────────────

# 被膜厚さ 0.05mm/本 → coat_total = 0.10mm
_COAT_THICK = 0.05
_RADIUS = 0.25
_CORE_RADIUS = _RADIUS - _COAT_THICK  # 0.20


def _make_barrier_config(
    *,
    k_coat: float = 1e6,
    c_coat: float = 0.0,
    mu: float = 0.0,
    k_t_ratio: float = 0.5,
) -> object:
    """バリア関数有効の被膜設定."""
    return _make_config(
        k_coat=k_coat,
        c_coat=c_coat,
        mu=mu,
        k_t_ratio=k_t_ratio,
        coating_barrier=True,
        coating_thickness=_COAT_THICK,
    )


def _make_barrier_pair(
    *,
    coating_compression: float = 0.0,
    coating_compression_prev: float = 0.0,
    **kwargs: object,
) -> _ContactPairOutput:
    """バリア関数用ペア（radius/core_radius設定済み）."""
    return _make_pair(
        coating_compression=coating_compression,
        coating_compression_prev=coating_compression_prev,
        radius_a=_RADIUS,
        radius_b=_RADIUS,
        core_radius_a=_CORE_RADIUS,
        core_radius_b=_CORE_RADIUS,
        **kwargs,
    )


class TestBarrierCoatingPhysics:
    """バリア関数被膜モデルの物理的妥当性テスト（status-303）."""

    # ── 基本特性 ──────────────────────────────────────

    def test_small_compression_matches_linear(self) -> None:
        """小圧縮量ではバリアと線形がほぼ一致する."""
        p = KelvinVoigtCoatingProcess()
        delta = 0.001  # coat_total=0.10 の 1%
        config_barrier = _make_barrier_config(k_coat=1e6)
        config_linear = _make_config(k_coat=1e6)
        pair_barrier = _make_barrier_pair(coating_compression=delta)
        pair_linear = _make_pair(coating_compression=delta)

        f_barrier = p.forces([pair_barrier], _COORDS, config_barrier, 0.01)
        f_linear = p.forces([pair_linear], _COORDS, config_linear, 0.01)

        # 1%圧縮 → ratio=0.01 → 1/(1-0.01)≈1.01 → 1%差以内
        np.testing.assert_allclose(f_barrier, f_linear, rtol=0.02)

    def test_force_increases_nonlinearly(self) -> None:
        """圧縮量↑で力が非線形に増大する（線形より大きい）."""
        p = KelvinVoigtCoatingProcess()
        config = _make_barrier_config(k_coat=1e6)
        delta_small = 0.02  # 20%
        delta_large = 0.08  # 80%

        pair_small = _make_barrier_pair(coating_compression=delta_small)
        pair_large = _make_barrier_pair(coating_compression=delta_large)

        f_small = p.forces([pair_small], _COORDS, config, 0.01)
        f_large = p.forces([pair_large], _COORDS, config, 0.01)

        norm_small = np.linalg.norm(f_small)
        norm_large = np.linalg.norm(f_large)

        # 線形なら 4倍（0.08/0.02=4）。バリアならそれ以上。
        # 80%圧縮: f=k*0.08/(1-0.8)=k*0.08/0.2=k*0.4
        # 20%圧縮: f=k*0.02/(1-0.2)=k*0.02/0.8=k*0.025
        # 比率: 0.4/0.025 = 16 >> 4
        assert norm_large / norm_small > 4.0, "バリア関数は線形より急速に増大すべき"

    def test_no_core_penetration(self) -> None:
        """δ=δ_max（100%圧縮）でも有限の力（クランプ）."""
        p = KelvinVoigtCoatingProcess()
        config = _make_barrier_config(k_coat=1e6)
        coat_total = 2 * _COAT_THICK  # 0.10mm
        pair = _make_barrier_pair(coating_compression=coat_total)

        f = p.forces([pair], _COORDS, config, 0.01)
        # クランプで有限値（NaNやInfにならない）
        assert np.all(np.isfinite(f))
        assert np.linalg.norm(f) > 0

    def test_barrier_force_exact_value(self) -> None:
        """バリア関数の解析値と一致する."""
        p = KelvinVoigtCoatingProcess()
        k_coat = 1e6
        config = _make_barrier_config(k_coat=k_coat)
        delta = 0.05  # 50%圧縮
        coat_total = 2 * _COAT_THICK  # 0.10mm

        pair = _make_barrier_pair(coating_compression=delta)
        f = p.forces([pair], _COORDS, config, 0.01)
        f_nodes = f.reshape(4, 6)
        f_B = f_nodes[2, :3] + f_nodes[3, :3]

        # f = k * δ / (1 - δ/δ_max) = 1e6 * 0.05 / (1 - 0.5) = 1e5
        expected_fn = k_coat * delta / (1.0 - delta / coat_total)
        actual_fn = np.linalg.norm(f_B)
        np.testing.assert_allclose(actual_fn, expected_fn, rtol=1e-10)

    def test_action_reaction_with_barrier(self) -> None:
        """バリア関数でも作用反作用が成立する."""
        p = KelvinVoigtCoatingProcess()
        config = _make_barrier_config(k_coat=1e6)
        pair = _make_barrier_pair(coating_compression=0.08)

        f = p.forces([pair], _COORDS, config, 0.01)
        f_nodes = f.reshape(4, 6)
        f_A = f_nodes[0, :3] + f_nodes[1, :3]
        f_B = f_nodes[2, :3] + f_nodes[3, :3]

        np.testing.assert_allclose(f_A + f_B, 0.0, atol=1e-10)

    # ── 接線剛性 ──────────────────────────────────────

    def test_barrier_stiffness_exact_value(self) -> None:
        """バリア接線剛性の解析値と一致する."""
        p = KelvinVoigtCoatingProcess()
        k_coat = 1e6
        config = _make_barrier_config(k_coat=k_coat)
        delta = 0.05  # 50%圧縮
        coat_total = 2 * _COAT_THICK  # 0.10mm

        pair = _make_barrier_pair(coating_compression=delta)
        K_barrier = p.stiffness([pair], _COORDS, config, _NDOF, 0.01)
        K_linear = p.stiffness(
            [_make_pair(coating_compression=delta)],
            _COORDS,
            _make_config(k_coat=k_coat),
            _NDOF,
            0.01,
        )

        # バリア k_eff = k / (1 - ratio)² = k / 0.25 = 4k
        ratio = delta / coat_total  # 0.5
        expected_ratio = 1.0 / (1.0 - ratio) ** 2  # 4.0
        actual_ratio = np.abs(K_barrier.toarray()).sum() / np.abs(K_linear.toarray()).sum()
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-10)

    def test_barrier_stiffness_symmetry(self) -> None:
        """バリア接線剛性行列が対称."""
        p = KelvinVoigtCoatingProcess()
        config = _make_barrier_config(k_coat=1e6)
        pair = _make_barrier_pair(coating_compression=0.08)

        K = p.stiffness([pair], _COORDS, config, _NDOF, 0.01).toarray()
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_barrier_stiffness_positive_semi_definite(self) -> None:
        """バリア接線剛性行列が半正定値."""
        p = KelvinVoigtCoatingProcess()
        config = _make_barrier_config(k_coat=1e6)
        pair = _make_barrier_pair(coating_compression=0.08)

        K = p.stiffness([pair], _COORDS, config, _NDOF, 0.01).toarray()
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8)

    def test_barrier_stiffness_fd_consistency(self) -> None:
        """バリア接線剛性がFD（有限差分）と整合する."""
        k_coat = 1e6
        delta = 0.05
        coat_total = 2 * _COAT_THICK

        # 解析的接線剛性
        from xkep_cae.contact.coating.strategy import _barrier_k_eff

        k_analytical = _barrier_k_eff(k_coat, delta, coat_total)

        # FD: (f(δ+h) - f(δ-h)) / (2h)
        from xkep_cae.contact.coating.strategy import _barrier_p_n

        h = 1e-7
        f_plus = _barrier_p_n(k_coat, delta + h, coat_total)
        f_minus = _barrier_p_n(k_coat, delta - h, coat_total)
        k_fd = (f_plus - f_minus) / (2 * h)

        np.testing.assert_allclose(k_analytical, k_fd, rtol=1e-5)

    # ── 摩擦 ──────────────────────────────────────

    def test_barrier_friction_slip_limit(self) -> None:
        """バリア関数の法線力に基づくCoulomb限界."""
        p = KelvinVoigtCoatingProcess()
        k_coat = 1e6
        mu = 0.3
        delta = 0.08  # 80%圧縮
        coat_total = 2 * _COAT_THICK
        config = _make_barrier_config(k_coat=k_coat, mu=mu, k_t_ratio=0.5)

        u_cur = np.zeros(_NDOF)
        u_cur[0 * 6 + 0] = -1.0  # 大きな接線変位 → スリップ
        u_cur[1 * 6 + 0] = -1.0
        u_ref = np.zeros(_NDOF)

        pair = _make_barrier_pair(coating_compression=delta)
        pairs = [pair]
        f = p.friction_forces(pairs, _COORDS, config, u_cur, u_ref)
        f_nodes = f.reshape(4, 6)

        # バリア法線力に基づくCoulomb限界
        ratio = delta / coat_total
        p_n_barrier = k_coat * delta / (1.0 - ratio)
        coulomb_limit = mu * p_n_barrier

        f_A = f_nodes[0, :3] + f_nodes[1, :3]
        f_friction_mag = np.linalg.norm(f_A)
        assert f_friction_mag <= coulomb_limit * (1.0 + 1e-8)

    def test_barrier_disabled_when_no_thickness(self) -> None:
        """coating_thickness=0ではバリア無効（線形に戻る）."""
        p = KelvinVoigtCoatingProcess()
        k_coat = 1e6
        delta = 0.05

        # barrier=True だが thickness=0 → 線形
        config_no_thick = _make_config(k_coat=k_coat, coating_barrier=True, coating_thickness=0.0)
        config_linear = _make_config(k_coat=k_coat)

        pair1 = _make_pair(coating_compression=delta)
        pair2 = _make_pair(coating_compression=delta)

        f1 = p.forces([pair1], _COORDS, config_no_thick, 0.01)
        f2 = p.forces([pair2], _COORDS, config_linear, 0.01)

        np.testing.assert_allclose(f1, f2, atol=1e-15)

    def test_barrier_force_stiffness_fd_process_level(self) -> None:
        """Process level FD: 力と剛性行列の整合（幾何固定条件）.

        被膜の力ベクトル f = f_n(δ) * g_n、剛性 K = k_eff(δ) * outer(g_n, g_n)。
        圧縮量δの摂動に対して K*du ≈ Δf を検証する。
        幾何固定（s, t, n 固定）なので材料接線のみの検証。

        注: ソルバー全体系の67%FD誤差（status-301）は被膜の幾何接線剛性
        （∂n/∂u, ∂s/∂u）の欠落が原因であり、この材料接線は正確。
        """
        p = KelvinVoigtCoatingProcess()
        k_coat = 1e6
        config = _make_barrier_config(k_coat=k_coat, c_coat=0.0)
        delta = 0.05
        dt = 0.01

        # 法線方向のギャップ勾配ベクトル g_n_full（DOF空間）
        normal = np.array([0.0, 1.0, 0.0])
        s, t = 0.5, 0.5
        weights = [-(1.0 - s), -s, (1.0 - t), t]
        node_ids = [0, 1, 2, 3]
        g_n_full = np.zeros(_NDOF)
        for w, nid in zip(weights, node_ids, strict=True):
            for comp in range(3):
                g_n_full[nid * 6 + comp] = w * normal[comp]

        # 微小変位 du を g_n 方向に設定
        eps = 1e-7
        du = eps * g_n_full
        d_delta = np.dot(g_n_full, du)  # = eps * ||g_n||²

        # 力の中央差分
        pair_plus = _make_barrier_pair(coating_compression=delta + d_delta)
        pair_minus = _make_barrier_pair(coating_compression=delta - d_delta)
        f_plus = p.forces([pair_plus], _COORDS, config, dt)
        f_minus = p.forces([pair_minus], _COORDS, config, dt)
        df_fd = (f_plus - f_minus) / 2.0  # dδ=d_delta の力変化

        # 剛性行列の予測: K * du
        pair = _make_barrier_pair(coating_compression=delta)
        K = p.stiffness([pair], _COORDS, config, _NDOF, dt)
        df_analytical = K @ du

        # 幾何固定条件では材料接線が完全一致
        nonzero = np.abs(df_analytical) > 1e-20
        np.testing.assert_allclose(df_analytical[nonzero], df_fd[nonzero], rtol=1e-5)

    def test_linear_force_stiffness_fd_process_level(self) -> None:
        """線形モデルでも同じFD整合テスト."""
        p = KelvinVoigtCoatingProcess()
        k_coat = 1e6
        config = _make_config(k_coat=k_coat, c_coat=0.0)
        delta = 0.05
        dt = 0.01

        normal = np.array([0.0, 1.0, 0.0])
        s, t = 0.5, 0.5
        weights = [-(1.0 - s), -s, (1.0 - t), t]
        node_ids = [0, 1, 2, 3]
        g_n_full = np.zeros(_NDOF)
        for w, nid in zip(weights, node_ids, strict=True):
            for comp in range(3):
                g_n_full[nid * 6 + comp] = w * normal[comp]

        eps = 1e-7
        du = eps * g_n_full
        d_delta = np.dot(g_n_full, du)

        pair_plus = _make_pair(coating_compression=delta + d_delta)
        pair_minus = _make_pair(coating_compression=delta - d_delta)
        f_plus = p.forces([pair_plus], _COORDS, config, dt)
        f_minus = p.forces([pair_minus], _COORDS, config, dt)
        df_fd = (f_plus - f_minus) / 2.0

        pair = _make_pair(coating_compression=delta)
        K = p.stiffness([pair], _COORDS, config, _NDOF, dt)
        df_analytical = K @ du

        nonzero = np.abs(df_analytical) > 1e-20
        np.testing.assert_allclose(df_analytical[nonzero], df_fd[nonzero], rtol=1e-5)
