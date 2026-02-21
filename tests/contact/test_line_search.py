"""Merit function + backtracking line search の単体テスト.

Phase C4: line_search.py のテスト。
"""

import numpy as np

from xkep_cae.contact.line_search import backtracking_line_search, merit_function
from xkep_cae.contact.pair import ContactConfig, ContactManager, ContactStatus

# ====================================================================
# ヘルパー
# ====================================================================


def _make_manager_with_pairs(
    gaps: list[float],
    dissipations: list[float] | None = None,
) -> ContactManager:
    """指定した gap / dissipation を持つ ACTIVE ペアの ContactManager を作成."""
    mgr = ContactManager(config=ContactConfig())
    if dissipations is None:
        dissipations = [0.0] * len(gaps)
    for i, (g, d) in enumerate(zip(gaps, dissipations, strict=True)):
        pair = mgr.add_pair(
            elem_a=i * 2,
            elem_b=i * 2 + 1,
            nodes_a=np.array([i * 4, i * 4 + 1]),
            nodes_b=np.array([i * 4 + 2, i * 4 + 3]),
        )
        pair.state.status = ContactStatus.ACTIVE
        pair.state.gap = g
        pair.state.dissipation = d
    return mgr


# ====================================================================
# TestMeritFunction
# ====================================================================


class TestMeritFunction:
    """merit_function の単体テスト."""

    def test_zero_residual_no_contact(self):
        """残差ゼロ・接触なしで merit = 0."""
        mgr = ContactManager(config=ContactConfig())
        residual = np.zeros(10)
        phi = merit_function(residual, mgr)
        assert phi == 0.0

    def test_residual_only(self):
        """残差のみ（接触なし）の merit = ||R||."""
        mgr = ContactManager(config=ContactConfig())
        residual = np.array([3.0, 4.0])  # norm = 5.0
        phi = merit_function(residual, mgr)
        assert abs(phi - 5.0) < 1e-10

    def test_penetration_penalty(self):
        """貫通がある場合のペナルティ項."""
        mgr = _make_manager_with_pairs(gaps=[-0.1])
        residual = np.zeros(10)
        # Phi = 0 + alpha * 0.1^2 = 0.01
        phi = merit_function(residual, mgr, alpha=1.0, beta=0.0)
        assert abs(phi - 0.01) < 1e-10

    def test_no_penalty_for_positive_gap(self):
        """正の gap にはペナルティなし."""
        mgr = _make_manager_with_pairs(gaps=[0.1])
        residual = np.zeros(10)
        phi = merit_function(residual, mgr, alpha=1.0, beta=0.0)
        assert abs(phi) < 1e-10

    def test_dissipation_penalty(self):
        """散逸項の加算."""
        mgr = _make_manager_with_pairs(gaps=[0.0], dissipations=[0.5])
        residual = np.zeros(10)
        phi = merit_function(residual, mgr, alpha=0.0, beta=1.0)
        assert abs(phi - 0.5) < 1e-10

    def test_negative_dissipation_ignored(self):
        """負の散逸は無視される."""
        mgr = _make_manager_with_pairs(gaps=[0.0], dissipations=[-0.1])
        residual = np.zeros(10)
        phi = merit_function(residual, mgr, alpha=0.0, beta=1.0)
        assert abs(phi) < 1e-10

    def test_combined_merit(self):
        """残差 + 貫通 + 散逸の合計."""
        mgr = _make_manager_with_pairs(
            gaps=[-0.2, 0.1],
            dissipations=[0.3, 0.0],
        )
        residual = np.array([3.0, 4.0])  # norm = 5.0
        # Phi = 5.0 + 1.0 * 0.04 + 1.0 * 0.3 = 5.34
        phi = merit_function(residual, mgr, alpha=1.0, beta=1.0)
        assert abs(phi - 5.34) < 1e-10

    def test_alpha_weight(self):
        """alpha 重みの効果."""
        mgr = _make_manager_with_pairs(gaps=[-0.1])
        residual = np.zeros(10)
        phi = merit_function(residual, mgr, alpha=10.0, beta=0.0)
        # Phi = 10.0 * 0.01 = 0.1
        assert abs(phi - 0.1) < 1e-10

    def test_beta_weight(self):
        """beta 重みの効果."""
        mgr = _make_manager_with_pairs(gaps=[0.0], dissipations=[1.0])
        residual = np.zeros(10)
        phi = merit_function(residual, mgr, alpha=0.0, beta=5.0)
        assert abs(phi - 5.0) < 1e-10

    def test_inactive_pairs_ignored(self):
        """INACTIVE ペアは merit 計算に含まれない."""
        mgr = ContactManager(config=ContactConfig())
        pair = mgr.add_pair(0, 1, np.array([0, 1]), np.array([2, 3]))
        pair.state.status = ContactStatus.INACTIVE
        pair.state.gap = -1.0  # 大きな貫通だが INACTIVE
        pair.state.dissipation = 10.0

        residual = np.zeros(10)
        phi = merit_function(residual, mgr, alpha=1.0, beta=1.0)
        assert abs(phi) < 1e-10

    def test_multiple_penetrations(self):
        """複数ペアの貫通の合算."""
        mgr = _make_manager_with_pairs(gaps=[-0.1, -0.2, -0.3])
        residual = np.zeros(10)
        # sum(g^2) = 0.01 + 0.04 + 0.09 = 0.14
        phi = merit_function(residual, mgr, alpha=1.0, beta=0.0)
        assert abs(phi - 0.14) < 1e-10

    def test_merit_nonnegative(self):
        """merit は常に非負."""
        mgr = _make_manager_with_pairs(gaps=[-0.5], dissipations=[2.0])
        residual = np.array([1.0, -2.0, 3.0])
        phi = merit_function(residual, mgr, alpha=1.0, beta=1.0)
        assert phi >= 0.0


# ====================================================================
# TestBacktrackingLineSearch
# ====================================================================


class TestBacktrackingLineSearch:
    """backtracking_line_search の単体テスト."""

    def test_full_step_accepted(self):
        """merit が十分減少するなら eta=1 を採用."""
        u = np.array([0.0, 0.0])
        du = np.array([1.0, 0.0])
        phi_current = 10.0

        # 全ての試行で merit が大幅減少
        def eval_merit(u_trial):
            return 1.0

        eta, n_steps = backtracking_line_search(
            u,
            du,
            phi_current,
            eval_merit,
        )
        assert abs(eta - 1.0) < 1e-10
        assert n_steps == 1

    def test_backtracking_needed(self):
        """full step では merit 増加、縮小で減少するケース."""
        u = np.zeros(2)
        du = np.array([2.0, 0.0])
        phi_current = 5.0

        call_count = [0]

        def eval_merit(u_trial):
            call_count[0] += 1
            x = u_trial[0]
            # eta=1 (x=2): phi=10 (増加)
            # eta=0.5 (x=1): phi=2 (減少)
            return x**2 + 1.0

        eta, n_steps = backtracking_line_search(
            u,
            du,
            phi_current,
            eval_merit,
            max_steps=5,
        )
        # eta=1 → phi=5 > 5*(1-1e-4) → fail
        # eta=0.5 → phi=2 < 5*(1-0.5e-4) → pass
        assert eta < 1.0
        assert n_steps >= 2

    def test_zero_merit_accepts_full_step(self):
        """現在の merit が 0 なら即座に eta=1."""
        u = np.zeros(2)
        du = np.array([1.0, 0.0])
        phi_current = 0.0

        eta, n_steps = backtracking_line_search(
            u,
            du,
            phi_current,
            lambda _: 1.0,
        )
        assert abs(eta - 1.0) < 1e-10
        assert n_steps == 0

    def test_best_eta_returned_on_failure(self):
        """全ステップで Armijo 未達 → 最良の eta を返す."""
        u = np.zeros(2)
        du = np.array([1.0, 0.0])
        phi_current = 1.0

        # 全試行で merit 増加
        def eval_merit(u_trial):
            return 100.0

        eta, n_steps = backtracking_line_search(
            u,
            du,
            phi_current,
            eval_merit,
            max_steps=3,
        )
        assert n_steps == 3
        # 最良の eta は 1.0（全て同じ merit だから最初の試行）
        assert eta > 0.0

    def test_shrink_factor(self):
        """shrink 係数に従って eta が縮小される."""
        u = np.zeros(2)
        du = np.array([1.0, 0.0])
        phi_current = 10.0

        etas_tried = []

        def eval_merit(u_trial):
            etas_tried.append(u_trial[0])  # eta = u_trial[0] / du[0]
            return 20.0  # 常に merit 増加

        backtracking_line_search(
            u,
            du,
            phi_current,
            eval_merit,
            max_steps=4,
            shrink=0.5,
        )
        # etas_tried should be [1.0, 0.5, 0.25, 0.125]
        expected = [1.0, 0.5, 0.25, 0.125]
        assert len(etas_tried) == 4
        for e, exp in zip(etas_tried, expected, strict=True):
            assert abs(e - exp) < 1e-10

    def test_armijo_condition(self):
        """Armijo 条件 phi_trial <= phi*(1 - c*eta) の検証."""
        u = np.zeros(2)
        du = np.array([1.0, 0.0])
        phi_current = 10.0
        c_armijo = 0.1

        # eta=1: threshold = 10*(1-0.1*1) = 9.0
        # phi=8.9 < 9.0 → accept
        def eval_merit(u_trial):
            return 8.9

        eta, n_steps = backtracking_line_search(
            u,
            du,
            phi_current,
            eval_merit,
            c_armijo=c_armijo,
        )
        assert abs(eta - 1.0) < 1e-10
        assert n_steps == 1

    def test_armijo_condition_marginal_fail(self):
        """Armijo 条件ギリギリ不成立."""
        u = np.zeros(2)
        du = np.array([1.0, 0.0])
        phi_current = 10.0
        c_armijo = 0.1

        # eta=1: threshold = 9.0, phi=9.1 > 9.0 → fail
        # eta=0.5: threshold = 10*(1-0.05) = 9.5, phi=9.1 < 9.5 → accept
        def eval_merit(u_trial):
            return 9.1

        eta, n_steps = backtracking_line_search(
            u,
            du,
            phi_current,
            eval_merit,
            c_armijo=c_armijo,
        )
        assert abs(eta - 0.5) < 1e-10
        assert n_steps == 2

    def test_max_steps_respected(self):
        """max_steps が尊重される."""
        u = np.zeros(2)
        du = np.array([1.0, 0.0])
        phi_current = 1.0

        call_count = [0]

        def eval_merit(u_trial):
            call_count[0] += 1
            return 100.0

        _, n_steps = backtracking_line_search(
            u,
            du,
            phi_current,
            eval_merit,
            max_steps=7,
        )
        assert n_steps == 7
        assert call_count[0] == 7

    def test_best_of_multiple_trials(self):
        """複数試行から最良の eta が選ばれる."""
        u = np.zeros(2)
        du = np.array([1.0, 0.0])
        phi_current = 1.0

        trial_phis = {1.0: 5.0, 0.5: 2.0, 0.25: 3.0}

        def eval_merit(u_trial):
            eta = u_trial[0]
            for k, v in trial_phis.items():
                if abs(eta - k) < 1e-10:
                    return v
            return 100.0

        eta, _ = backtracking_line_search(
            u,
            du,
            phi_current,
            eval_merit,
            max_steps=3,
            shrink=0.5,
        )
        # Best merit is 2.0 at eta=0.5
        assert abs(eta - 0.5) < 1e-10
