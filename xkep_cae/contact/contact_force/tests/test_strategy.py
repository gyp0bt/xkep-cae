"""ContactForceStrategy 具象実装のテスト.

status-222 で HuberContactForceProcess に一本化。
"""

from __future__ import annotations

import numpy as np
import pytest

from xkep_cae.contact._contact_pair import (
    _ContactPairOutput,
    _ContactStateOutput,
)
from xkep_cae.contact._types import ContactStatus as _ContactStatus
from xkep_cae.contact.contact_force import (
    ContactForceInput,
    ContactForceOutput,
    HuberContactForceProcess,
)
from xkep_cae.contact.contact_force.strategy import (
    _create_contact_force_strategy,
)
from xkep_cae.core.strategies import ContactForceStrategy
from xkep_cae.core.testing import binds_to


def _make_state(*, gap: float = -0.01, status: _ContactStatus = _ContactStatus.ACTIVE, **kw):
    return _ContactStateOutput(
        gap=gap,
        p_n=0.0,
        normal=np.array([0.0, 1.0, 0.0]),
        s=0.5,
        t=0.5,
        status=status,
        **kw,
    )


def _make_pair(*, gap: float = -0.01, **kw):
    return _ContactPairOutput(
        elem_a=0,
        elem_b=1,
        nodes_a=[0, 1],
        nodes_b=[2, 3],
        state=_make_state(gap=gap, **kw),
    )


class _MockManager:
    def __init__(self, pairs):
        self.pairs = list(pairs)


class TestContactForceProtocolConformance:
    """HuberContactForceProcess が Protocol を満たすことを検証."""

    def test_protocol_conformance(self):
        instance = HuberContactForceProcess(ndof=24)
        assert isinstance(instance, ContactForceStrategy)


# ── HuberContactForce ────────────────────────────────────────


@binds_to(HuberContactForceProcess)
class TestHuberContactForceProcess:
    """HuberContactForceProcess の単体テスト."""

    def test_evaluate_no_pairs(self):
        proc = HuberContactForceProcess(ndof=24)
        manager = _MockManager([])
        f, r = proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))
        assert len(r) == 0

    def test_evaluate_with_active_pair(self):
        proc = HuberContactForceProcess(ndof=24)
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        assert np.any(f != 0.0)
        assert len(r) == 1

    def test_evaluate_inactive_pair_with_penetration(self):
        """SDI 排除（status-233）: INACTIVE でも gap<0 なら Huber で力が発生."""
        proc = HuberContactForceProcess(ndof=24)
        from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state

        pair = _make_pair(gap=-0.01)
        pair = _evolve_pair(pair, state=_evolve_state(pair.state, status=_ContactStatus.INACTIVE))
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        # SDI 排除: 全候補ペアを評価するため、gap<0 なら力が発生
        assert np.any(f != 0.0)

    def test_evaluate_positive_gap(self):
        proc = HuberContactForceProcess(ndof=24)
        pair = _make_pair(gap=0.1)
        manager = _MockManager([pair])
        f, r = proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))

    def test_tangent_returns_zero(self):
        proc = HuberContactForceProcess(ndof=24)
        manager = _MockManager([])
        K = proc.tangent(np.zeros(24), manager, k_pen=1e4)
        assert K.shape == (24, 24)
        assert K.nnz == 0

    def test_tangent_with_penetration(self):
        """Huber は正定値接線を返す.

        K_contact = h'(x) * k_pen * g_shape ⊗ g_shape ≥ 0.
        """
        proc = HuberContactForceProcess(ndof=24)
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        K = proc.tangent(np.zeros(24), manager, k_pen=1e4)
        assert K.shape == (24, 24)
        assert K.nnz > 0
        K_dense = K.toarray()
        np.testing.assert_allclose(K_dense, K_dense.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(K_dense)
        assert np.all(eigvals >= -1e-10)

    def test_process_returns_output(self):
        proc = HuberContactForceProcess(ndof=24)
        inp = ContactForceInput(
            u=np.zeros(24),
            manager=_MockManager([]),
            k_pen=1e4,
        )
        out = proc.process(inp)
        assert isinstance(out, ContactForceOutput)

    def test_no_manager_pairs(self):
        proc = HuberContactForceProcess(ndof=24)
        manager = object()
        f, r = proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        np.testing.assert_array_equal(f, np.zeros(24))

    def test_huber_function(self):
        h = HuberContactForceProcess._huber
        assert h(-1.0, 0.5) == 0.0
        assert h(1.0, 0.5) == pytest.approx(1.0)
        assert h(0.0, 0.5) > 0.0

    def test_huber_deriv(self):
        hd = HuberContactForceProcess._huber_deriv
        assert hd(-1.0, 0.5) == 0.0
        assert hd(1.0, 0.5) == pytest.approx(1.0)
        assert 0.0 < hd(0.0, 0.5) < 1.0

    def test_p_n_updated_on_pair(self):
        """evaluate が pair.state.p_n を更新する."""
        proc = HuberContactForceProcess(ndof=24)
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        assert manager.pairs[0].state.p_n > 0.0


# ── ファクトリ ─────────────────────────────────────────────


class TestCreateContactForceStrategy:
    """ファクトリ関数のテスト."""

    def test_returns_huber(self):
        s = _create_contact_force_strategy(ndof=24)
        assert isinstance(s, HuberContactForceProcess)

    def test_with_smoothing_delta(self):
        s = _create_contact_force_strategy(ndof=24, smoothing_delta=10.0)
        assert isinstance(s, HuberContactForceProcess)
        assert s._smoothing_delta == 10.0

    def test_with_huber_delta_h(self):
        """huber_delta_h 直接指定がファクトリ経由で貫通する."""
        s = _create_contact_force_strategy(ndof=24, huber_delta_h=0.01)
        assert isinstance(s, HuberContactForceProcess)
        assert s._huber_delta_h == 0.01

    def test_resolve_delta_h_direct(self):
        """huber_delta_h > 0 なら k_pen に関わらず直接値を返す."""
        s = HuberContactForceProcess(ndof=24, huber_delta_h=0.05)
        assert s._resolve_delta_h(k_pen=100.0) == 0.05

    def test_resolve_delta_h_indirect(self):
        """huber_delta_h=0 なら k_pen/smoothing_delta で間接計算."""
        s = HuberContactForceProcess(ndof=24, smoothing_delta=1000.0)
        assert s._resolve_delta_h(k_pen=100.0) == 0.1  # 100/1000

    def test_resolve_delta_h_none(self):
        """両方 0 なら delta_h=0（max(0,x)相当）."""
        s = HuberContactForceProcess(ndof=24)
        assert s._resolve_delta_h(k_pen=100.0) == 0.0

    def test_resolve_delta_h_priority(self):
        """huber_delta_h が smoothing_delta より優先される."""
        s = HuberContactForceProcess(ndof=24, smoothing_delta=1000.0, huber_delta_h=0.01)
        assert s._resolve_delta_h(k_pen=100.0) == 0.01


class TestHertzPenalty:
    """Hertz型非線形ペナルティ（status-285）のテスト."""

    def test_default_exponent_is_linear(self):
        """デフォルトで線形ペナルティ（α=1.0）."""
        s = HuberContactForceProcess(ndof=24)
        assert s._penalty_exponent == 1.0

    def test_hertz_exponent(self):
        """α=1.5 でHertz型."""
        s = HuberContactForceProcess(ndof=24, penalty_exponent=1.5)
        assert s._penalty_exponent == 1.5

    def test_power_law_identity_for_linear(self):
        """α=1.0 で _apply_power_law は恒等変換."""
        s = HuberContactForceProcess(ndof=24, penalty_exponent=1.0)
        h = np.array([0.0, 1.0, 5.0, 10.0])
        result = s._apply_power_law(h, k_pen=100.0)
        np.testing.assert_allclose(result, h)

    def test_power_law_hertz(self):
        """α=1.5 で p_n = h^1.5 / k_pen^0.5."""
        s = HuberContactForceProcess(ndof=24, penalty_exponent=1.5)
        k_pen = 100.0
        h = np.array([0.0, 1.0, 4.0, 100.0])
        result = s._apply_power_law(h, k_pen)
        expected = h**1.5 / k_pen**0.5
        np.testing.assert_allclose(result, expected)

    def test_power_law_deriv_identity_for_linear(self):
        """α=1.0 で _apply_power_law_deriv は恒等変換."""
        s = HuberContactForceProcess(ndof=24, penalty_exponent=1.0)
        h = np.array([1.0, 5.0, 10.0])
        h_d = np.array([0.5, 0.8, 1.0])
        result = s._apply_power_law_deriv(h, h_d, k_pen=100.0)
        np.testing.assert_allclose(result, h_d)

    def test_power_law_deriv_hertz(self):
        """α=1.5 のHertz導関数が正しい."""
        s = HuberContactForceProcess(ndof=24, penalty_exponent=1.5)
        k_pen = 100.0
        h = np.array([4.0, 100.0])
        h_d = np.array([1.0, 1.0])
        result = s._apply_power_law_deriv(h, h_d, k_pen)
        # dp/dx = α * (h/k_pen)^{α-1} * h'
        pen = h / k_pen
        expected = 1.5 * pen**0.5 * h_d
        np.testing.assert_allclose(result, expected)

    def test_power_law_zero_penetration(self):
        """貫入量ゼロでは力もゼロ."""
        s = HuberContactForceProcess(ndof=24, penalty_exponent=1.5)
        h = np.array([0.0])
        result = s._apply_power_law(h, k_pen=100.0)
        np.testing.assert_allclose(result, [0.0])

    def test_factory_passes_exponent(self):
        """ファクトリ関数が penalty_exponent を渡す."""
        s = _create_contact_force_strategy(ndof=24, penalty_exponent=1.5)
        assert s._penalty_exponent == 1.5


class TestHertzTangentFD:
    """Hertz型ペナルティの接線剛性FD検証（status-289）.

    K_c の解析的接線と有限差分を比較し、∂p/∂g 導関数の整合性を検証する。
    """

    def test_hertz_tangent_w_mat_consistency(self):
        """Hertz型 w_mat = h_deriv_modified * k_pen がスカラーFDと一致するか検証.

        K_matの重み w_mat は dp_n/dg（符号反転済み）に相当する。
        スカラーFDで求めた dp_n/dg と比較し、行列アセンブリ以前の段階で
        不整合がないか確認する。
        """
        k_pen = 1e4
        for alpha in [1.0, 1.5]:
            proc = HuberContactForceProcess(ndof=24, penalty_exponent=alpha)
            delta_h = 0.0
            for gap in [-0.1, -0.05, -0.01]:
                x = k_pen * (-gap)
                h_der = proc._huber_deriv(x, delta_h)
                if alpha != 1.0:
                    h_val = proc._huber(x, delta_h)
                    h_arr = np.array([h_val])
                    h_d_arr = np.array([h_der])
                    h_der = float(proc._apply_power_law_deriv(h_arr, h_d_arr, k_pen)[0])
                w_mat = h_der * k_pen  # = dp_n/d(penetration)

                # FD: dp_n/d(penetration)
                eps = 1e-7
                pen_base = -gap
                pen_pert = pen_base + eps
                x_base = k_pen * pen_base
                x_pert = k_pen * pen_pert
                p_base = proc._huber(x_base, delta_h)
                p_pert = proc._huber(x_pert, delta_h)
                if alpha != 1.0:
                    p_base = p_base**alpha / k_pen ** (alpha - 1.0)
                    p_pert = p_pert**alpha / k_pen ** (alpha - 1.0)
                dpn_dpen_fd = (p_pert - p_base) / eps

                rel_err = abs(w_mat - dpn_dpen_fd) / max(abs(dpn_dpen_fd), 1e-30)
                assert rel_err < 1e-4, (
                    f"w_mat FD不整合: α={alpha}, gap={gap}, "
                    f"w_mat={w_mat:.6e}, FD={dpn_dpen_fd:.6e}, rel_err={rel_err:.3e}"
                )

    def test_hertz_scalar_dpn_dg_fd(self):
        """Hertz型 dp_n/dg のスカラーFD検証.

        p_n(g) = [huber(k_pen * (-g), δ)]^α / k_pen^{α-1}
        dp_n/dg = (p_n(g+eps) - p_n(g)) / eps（FD）
        vs 解析的: -α * (h/k_pen)^{α-1} * h'(x) * k_pen
        """
        k_pen = 1e4
        alpha = 1.5
        delta_h = 0.0
        proc = HuberContactForceProcess(ndof=24, penalty_exponent=alpha)

        for gap in [-0.1, -0.05, -0.01, -0.001]:
            x = k_pen * (-gap)
            h_val = proc._huber(x, delta_h)
            h_der = proc._huber_deriv(x, delta_h)
            # 解析的 dp_n/dg
            pen = h_val / k_pen
            if pen > 1e-30:
                dpn_dx_analytical = alpha * pen ** (alpha - 1.0) * h_der
            else:
                dpn_dx_analytical = 0.0
            dpn_dg_analytical = dpn_dx_analytical * (-k_pen)

            # FD
            eps = 1e-7
            x_pert = k_pen * (-(gap + eps))
            h_val_pert = proc._huber(x_pert, delta_h)
            p_n_base = h_val**alpha / k_pen ** (alpha - 1.0)
            p_n_pert = h_val_pert**alpha / k_pen ** (alpha - 1.0)
            dpn_dg_fd = (p_n_pert - p_n_base) / eps

            rel_err = abs(dpn_dg_analytical - dpn_dg_fd) / max(abs(dpn_dg_fd), 1e-30)
            assert rel_err < 1e-4, (
                f"Hertz dp_n/dg FD不整合 at gap={gap}: "
                f"analytical={dpn_dg_analytical:.6e}, FD={dpn_dg_fd:.6e}, "
                f"rel_err={rel_err:.3e}"
            )

    def test_hertz_scalar_dpn_dg_fd_with_delta_h(self):
        """Hertz型 dp_n/dg のスカラーFD検証（delta_h > 0）."""
        k_pen = 1e4
        alpha = 1.5
        delta_h = 10.0  # Huber遷移幅
        proc = HuberContactForceProcess(ndof=24, penalty_exponent=alpha)

        for gap in [-0.1, -0.05, -0.01, -0.001, -1e-4]:
            x = k_pen * (-gap)
            h_val = proc._huber(x, delta_h)
            h_der = proc._huber_deriv(x, delta_h)
            pen = h_val / k_pen
            if pen > 1e-30:
                dpn_dx_analytical = alpha * pen ** (alpha - 1.0) * h_der
            else:
                dpn_dx_analytical = 0.0
            dpn_dg_analytical = dpn_dx_analytical * (-k_pen)

            eps = 1e-7
            x_pert = k_pen * (-(gap + eps))
            h_val_pert = proc._huber(x_pert, delta_h)
            p_n_base = h_val**alpha / k_pen ** (alpha - 1.0)
            p_n_pert = h_val_pert**alpha / k_pen ** (alpha - 1.0)
            dpn_dg_fd = (p_n_pert - p_n_base) / eps

            rel_err = abs(dpn_dg_analytical - dpn_dg_fd) / max(abs(dpn_dg_fd), 1e-30)
            assert rel_err < 1e-3, (
                f"Hertz dp_n/dg FD不整合(δ_h={delta_h}) at gap={gap}: "
                f"analytical={dpn_dg_analytical:.6e}, FD={dpn_dg_fd:.6e}, "
                f"rel_err={rel_err:.3e}"
            )


class TestActiveEmaSmoothing:
    """active 履歴 EMA 平滑化（status-371: 候補 (g1)）のテスト.

    `p_n_eff = α·p_n_new + (1-α)·p_n_prev` を NR 反復間で適用し、active
    集合振動を抑制する escape hatch。default α=0.0 で完全無効化。
    """

    def test_default_alpha_is_zero(self):
        """デフォルトで EMA 平滑化なし（後方互換）."""
        proc = HuberContactForceProcess(ndof=24)
        assert proc._active_ema_alpha == 0.0
        assert proc._p_n_prev_array is None

    def test_alpha_field_stored(self):
        """active_ema_alpha が float として保存される."""
        proc = HuberContactForceProcess(ndof=24, active_ema_alpha=0.3)
        assert proc._active_ema_alpha == pytest.approx(0.3)

    def test_factory_passes_alpha(self):
        """ファクトリ経由で active_ema_alpha が貫通する."""
        s = _create_contact_force_strategy(ndof=24, active_ema_alpha=0.5)
        assert s._active_ema_alpha == pytest.approx(0.5)

    def test_alpha_zero_no_smoothing(self):
        """α=0.0 では p_n_prev_array を更新しない（完全無効）."""
        proc = HuberContactForceProcess(ndof=24, active_ema_alpha=0.0)
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        # α=0 のとき履歴ストレージは触らない
        assert proc._p_n_prev_array is None

    def test_alpha_first_iter_no_history(self):
        """履歴なし（_p_n_prev_array=None）の初回反復は raw p_n を採用."""
        proc_smooth = HuberContactForceProcess(ndof=24, active_ema_alpha=0.3)
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc_smooth.evaluate(np.zeros(24), manager, k_pen=1e4)
        smoothed_first = manager.pairs[0].state.p_n

        proc_raw = HuberContactForceProcess(ndof=24, active_ema_alpha=0.0)
        pair2 = _make_pair(gap=-0.01)
        manager2 = _MockManager([pair2])
        proc_raw.evaluate(np.zeros(24), manager2, k_pen=1e4)
        raw_first = manager2.pairs[0].state.p_n

        assert smoothed_first == pytest.approx(raw_first), (
            "初回反復は履歴がないため raw p_n と一致する"
        )
        # 1回目で履歴は記録される
        assert proc_smooth._p_n_prev_array is not None

    def test_alpha_second_iter_blends(self):
        """2 回目反復で p_n_eff = α·p_n_new + (1-α)·p_n_prev が適用される."""
        alpha = 0.3
        proc = HuberContactForceProcess(ndof=24, active_ema_alpha=alpha)

        # 1回目: gap=-0.01 → p_n1
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        p_n_prev = manager.pairs[0].state.p_n

        # 2回目: gap=-0.02 でより深く貫入 → raw p_n_new=2*p_n_prev
        from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state

        manager.pairs[0] = _evolve_pair(
            manager.pairs[0], state=_evolve_state(manager.pairs[0].state, gap=-0.02)
        )
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        p_n_smoothed = manager.pairs[0].state.p_n

        # 期待値: α·(k_pen·0.02) + (1-α)·p_n_prev
        p_n_new_expected = 1e4 * 0.02
        expected = alpha * p_n_new_expected + (1.0 - alpha) * p_n_prev
        assert p_n_smoothed == pytest.approx(expected, rel=1e-10)

    def test_reset_ema_state_clears_history(self):
        """reset_ema_state() で履歴がクリアされる."""
        proc = HuberContactForceProcess(ndof=24, active_ema_alpha=0.3)
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        assert proc._p_n_prev_array is not None

        proc.reset_ema_state()
        assert proc._p_n_prev_array is None

    def test_reset_between_increments(self):
        """reset 後の次反復は raw p_n に戻る（インクリメント境界の効果）."""
        proc = HuberContactForceProcess(ndof=24, active_ema_alpha=0.5)

        # 1回目: gap=-0.01
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)

        # reset → 履歴クリア
        proc.reset_ema_state()

        # 2回目: gap=-0.02。reset 後なので履歴なし → raw p_n をそのまま採用
        from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state

        manager.pairs[0] = _evolve_pair(
            manager.pairs[0], state=_evolve_state(manager.pairs[0].state, gap=-0.02)
        )
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        p_n_after_reset = manager.pairs[0].state.p_n
        # raw p_n_new = k_pen * 0.02 = 200.0
        assert p_n_after_reset == pytest.approx(1e4 * 0.02, rel=1e-10)

    def test_alpha_one_recovers_raw_pn(self):
        """α=1.0 は EMA 無効化と同等（前反復の重み 0）."""
        proc = HuberContactForceProcess(ndof=24, active_ema_alpha=1.0)
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state

        manager.pairs[0] = _evolve_pair(
            manager.pairs[0], state=_evolve_state(manager.pairs[0].state, gap=-0.05)
        )
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        # α=1 なので p_n_eff = 1·p_n_new + 0·p_n_prev = raw p_n
        assert manager.pairs[0].state.p_n == pytest.approx(1e4 * 0.05, rel=1e-10)

    def test_pair_count_change_skips_smoothing(self):
        """ペア数が変わったら平滑化をスキップ（shape mismatch 安全策）."""
        proc = HuberContactForceProcess(ndof=24, active_ema_alpha=0.3)

        pair1 = _make_pair(gap=-0.01)
        manager = _MockManager([pair1])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)

        # ペアを追加 → shape が変わる
        pair2 = _make_pair(gap=-0.02)
        manager.pairs.append(pair2)
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)

        # 2 つ目のペアは履歴なし扱い → raw p_n を採用
        # 1 つ目も shape mismatch で平滑化スキップ → raw p_n
        assert manager.pairs[0].state.p_n == pytest.approx(1e4 * 0.01, rel=1e-10)
        assert manager.pairs[1].state.p_n == pytest.approx(1e4 * 0.02, rel=1e-10)
        # 履歴は新 shape で更新される
        assert proc._p_n_prev_array is not None
        assert proc._p_n_prev_array.shape == (2,)


class TestAugmentedLagrangianOffset:
    """Augmented Lagrangian λ オフセット（status-376: 候補 (g2)）のテスト.

    `p_n_eff = max(0, p_n_huber + λ)` を NR 内側で固定値の λ_offset として適用し、
    外側 Uzawa 更新で `λ_new = max(0, p_n_eff_converged)` を反映する escape hatch。
    default は無効化（_lambda_offset_pairs = None）。
    """

    def test_default_lambda_is_none(self):
        """デフォルトで AL λ オフセットなし（後方互換）."""
        proc = HuberContactForceProcess(ndof=24)
        assert proc._lambda_offset_pairs is None
        assert proc._last_p_n_eff is None

    def test_set_al_lambda_offset_stores_copy(self):
        """set_al_lambda_offset が配列を copy 保持し、外部変更の影響を受けない."""
        proc = HuberContactForceProcess(ndof=24)
        lam = np.array([1.0, 2.0, 3.0])
        proc.set_al_lambda_offset(lam)
        assert proc._lambda_offset_pairs is not None
        np.testing.assert_array_equal(proc._lambda_offset_pairs, lam)
        # 外部 lam を破壊しても strategy 内部に影響しない
        lam[0] = 999.0
        assert proc._lambda_offset_pairs[0] == pytest.approx(1.0)

    def test_set_al_lambda_offset_none_clears(self):
        """None で AL を完全無効化する."""
        proc = HuberContactForceProcess(ndof=24)
        proc.set_al_lambda_offset(np.ones(3))
        assert proc._lambda_offset_pairs is not None
        proc.set_al_lambda_offset(None)
        assert proc._lambda_offset_pairs is None

    def test_lambda_zero_no_op(self):
        """λ=0 のとき p_n_eff は p_n_huber と同値（既定動作と等価）."""
        proc_no_al = HuberContactForceProcess(ndof=24)
        pair_no = _make_pair(gap=-0.01)
        manager_no = _MockManager([pair_no])
        proc_no_al.evaluate(np.zeros(24), manager_no, k_pen=1e4)
        p_n_no_al = manager_no.pairs[0].state.p_n

        proc_al = HuberContactForceProcess(ndof=24)
        proc_al.set_al_lambda_offset(np.array([0.0]))
        pair_al = _make_pair(gap=-0.01)
        manager_al = _MockManager([pair_al])
        proc_al.evaluate(np.zeros(24), manager_al, k_pen=1e4)
        p_n_al = manager_al.pairs[0].state.p_n

        assert p_n_al == pytest.approx(p_n_no_al, rel=1e-12)

    def test_lambda_positive_augments_p_n(self):
        """λ > 0 のとき p_n_eff = p_n_huber + λ（active pair で加算）."""
        proc = HuberContactForceProcess(ndof=24)
        lam = 50.0
        proc.set_al_lambda_offset(np.array([lam]))
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        # p_n_huber = k_pen·|gap| = 100, p_n_eff = 100 + 50 = 150
        expected = 1e4 * 0.01 + lam
        assert manager.pairs[0].state.p_n == pytest.approx(expected, rel=1e-10)

    def test_lambda_keeps_inactive_pair_active(self):
        """gap > 0（非接触）でも λ > 0 で p_n_eff = λ > 0 となり active 扱い."""
        proc = HuberContactForceProcess(ndof=24)
        lam = 25.0
        proc.set_al_lambda_offset(np.array([lam]))
        # gap=+0.01 (no penetration), p_n_huber = 0
        pair = _make_pair(gap=+0.01)
        manager = _MockManager([pair])
        f, _ = proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        assert manager.pairs[0].state.p_n == pytest.approx(lam, rel=1e-10)
        # f_c も非ゼロ（λ 駆動の力）
        assert np.any(f != 0.0)

    def test_lambda_clamped_to_nonneg(self):
        """λ + p_n_huber < 0（負の λ で p_n_huber を打ち消す）でも p_n_eff ≥ 0."""
        proc = HuberContactForceProcess(ndof=24)
        # 負の λ は本来 Uzawa 更新で生成されないが、API 安全策として max(0, ...) で保護
        proc.set_al_lambda_offset(np.array([-1e6]))
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        assert manager.pairs[0].state.p_n == pytest.approx(0.0, abs=1e-12)

    def test_get_last_p_n_eff_after_evaluate(self):
        """evaluate() 後に get_last_p_n_eff() が p_n_eff スナップショットを返す."""
        proc = HuberContactForceProcess(ndof=24)
        proc.set_al_lambda_offset(np.array([10.0]))
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        snap = proc.get_last_p_n_eff()
        assert snap is not None
        assert snap.shape == (1,)
        # p_n_eff = 100 + 10 = 110
        assert snap[0] == pytest.approx(110.0, rel=1e-10)

    def test_get_last_p_n_eff_before_evaluate_is_none(self):
        """evaluate() 前は get_last_p_n_eff() が None."""
        proc = HuberContactForceProcess(ndof=24)
        assert proc.get_last_p_n_eff() is None

    def test_lambda_shape_mismatch_skips_offset(self):
        """λ 配列の shape が pair 数と異なる場合は AL を no-op（安全策）."""
        proc = HuberContactForceProcess(ndof=24)
        proc.set_al_lambda_offset(np.array([10.0, 20.0]))  # shape (2,)
        # ペア 1 つだけ → shape mismatch
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        # AL 適用されず raw p_n_huber のみ
        assert manager.pairs[0].state.p_n == pytest.approx(1e4 * 0.01, rel=1e-10)

    def test_uzawa_update_pattern(self):
        """1 サイクル目で λ=0 → 収束 → λ_new=p_n_converged → 2 サイクル目で再評価."""
        proc = HuberContactForceProcess(ndof=24)

        # サイクル 1: λ=0 (no AL)
        proc.set_al_lambda_offset(np.array([0.0]))
        pair = _make_pair(gap=-0.01)
        manager = _MockManager([pair])
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        p_n_cycle1 = manager.pairs[0].state.p_n
        snap1 = proc.get_last_p_n_eff()
        assert p_n_cycle1 == pytest.approx(100.0, rel=1e-10)
        assert snap1[0] == pytest.approx(p_n_cycle1, rel=1e-10)

        # Uzawa 更新: λ_new = max(0, p_n_eff_converged) = 100
        lambda_new = np.maximum(0.0, snap1)
        proc.set_al_lambda_offset(lambda_new)

        # サイクル 2: 同じ gap で再評価 → p_n_eff = 100 + 100 = 200
        from xkep_cae.contact._contact_pair import _evolve_pair, _evolve_state

        manager.pairs[0] = _evolve_pair(
            manager.pairs[0], state=_evolve_state(manager.pairs[0].state, p_n=0.0)
        )
        proc.evaluate(np.zeros(24), manager, k_pen=1e4)
        p_n_cycle2 = manager.pairs[0].state.p_n
        # p_n_huber は同じ (100)、λ=100 で p_n_eff=200
        assert p_n_cycle2 == pytest.approx(200.0, rel=1e-10)
