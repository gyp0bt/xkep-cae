"""Merit function + backtracking line search for contact NR solver.

Phase C4: Newton step length の適応制御。

設計仕様: docs/contact/beam_beam_contact_spec_v0.1.md §8

Merit function:
    Phi = ||R|| + alpha * sum(max(0, -g)^2) + beta * sum(max(0, D_inc))

Backtracking line search:
    u_trial = u + eta * du  (eta in (0, 1])
    eta を段階的に縮小し、Phi が減少する最大の eta を採用。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from xkep_cae.contact.pair import ContactManager, ContactStatus


def merit_function(
    residual: np.ndarray,
    manager: ContactManager,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Merit function Phi を評価する.

    Phi = ||R|| + alpha * sum(max(0, -g)^2) + beta * sum(max(0, D_inc))

    Args:
        residual: (ndof,) 力残差ベクトル（BC 適用後）
        manager: 接触マネージャ（gap, dissipation を参照）
        alpha: 貫通ペナルティ重み
        beta: 散逸ペナルティ重み

    Returns:
        Phi: merit 値（非負）
    """
    res_norm = float(np.linalg.norm(residual))

    penetration_sum = 0.0
    dissipation_sum = 0.0
    for pair in manager.pairs:
        if pair.state.status == ContactStatus.INACTIVE:
            continue
        # 貫通 = max(0, -gap)^2
        if pair.state.gap < 0.0:
            penetration_sum += pair.state.gap**2
        # 散逸（正成分のみ）
        if pair.state.dissipation > 0.0:
            dissipation_sum += pair.state.dissipation

    return res_norm + alpha * penetration_sum + beta * dissipation_sum


def backtracking_line_search(
    u: np.ndarray,
    du: np.ndarray,
    phi_current: float,
    eval_merit: Callable[[np.ndarray], float],
    *,
    max_steps: int = 5,
    shrink: float = 0.5,
    c_armijo: float = 1e-4,
) -> tuple[float, int]:
    """Backtracking line search で step length eta を決定する.

    Armijo 条件:
        Phi(u + eta*du) <= Phi(u) * (1 - c * eta)

    条件を満たす最初の eta を返す。
    見つからなければ最良の eta を返す（Newton 発散よりまし）。

    Args:
        u: (ndof,) 現在の変位
        du: (ndof,) Newton step
        phi_current: 現在の merit 値
        eval_merit: u_trial → Phi(u_trial) を返すコールバック
        max_steps: 最大縮小ステップ数
        shrink: 縮小率（0 < shrink < 1）
        c_armijo: Armijo パラメータ

    Returns:
        (eta, n_ls_steps): 採用された step length と line search ステップ数
    """
    if phi_current < 1e-30:
        return 1.0, 0

    eta = 1.0
    best_eta = 1.0
    best_phi = float("inf")

    for step in range(max_steps):
        u_trial = u + eta * du
        phi_trial = eval_merit(u_trial)

        if phi_trial < best_phi:
            best_phi = phi_trial
            best_eta = eta

        # Armijo 条件
        if phi_trial <= phi_current * (1.0 - c_armijo * eta):
            return eta, step + 1

        eta *= shrink

    # 条件未達 → 最良の eta を返す
    return best_eta, max_steps
