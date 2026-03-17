"""段階的アクティベーション — 純関数ユーティリティ.

ContactManager.compute_active_layer_for_step / filter_pairs_by_layer を
純関数化したもの。pair.py の責務軽量化（status-170）。
"""

from __future__ import annotations

# -- typing only --
from typing import TYPE_CHECKING

from __xkep_cae_deprecated.contact.pair import ContactStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from __xkep_cae_deprecated.contact.pair import ContactPair


def max_layer(elem_layer_map: dict[int, int] | None) -> int:
    """elem_layer_map の最大層番号を返す（未設定なら 0）."""
    if not elem_layer_map:
        return 0
    return max(elem_layer_map.values())


def compute_active_layer_for_step(
    step: int,
    staged_activation_steps: int,
    elem_layer_map: dict[int, int] | None,
) -> int:
    """現在ステップで許容する最大層番号を計算する.

    staged_activation_steps ステップをかけて層を段階的にオンにする。
    例: 3層構造で staged_activation_steps=6 の場合:
      - step 1-2: layer 0 のみ（中心素線どうし）
      - step 3-4: layer 0-1
      - step 5-6: layer 0-2（全層）
      - step 7+: 全層

    Args:
        step: 現在の荷重ステップ（1-indexed）
        staged_activation_steps: 段階的活性化に使うステップ数（0=無効）
        elem_layer_map: 要素→層番号マップ

    Returns:
        最大許容層番号
    """
    max_lay = max_layer(elem_layer_map)
    if staged_activation_steps <= 0:
        return max_lay
    if max_lay <= 0:
        return 0

    steps_per_layer = max(1, staged_activation_steps // (max_lay + 1))
    return min(max_lay, (step - 1) // steps_per_layer)


def filter_pairs_by_layer(
    pairs: Sequence[ContactPair],
    max_layer_num: int,
    elem_layer_map: dict[int, int] | None,
) -> None:
    """許容層より上の接触ペアを INACTIVE にする.

    Args:
        pairs: 接触ペアリスト
        max_layer_num: 許容する最大層番号
        elem_layer_map: 要素→層番号マップ
    """
    if not elem_layer_map:
        return

    for pair in pairs:
        if pair.state.status == ContactStatus.INACTIVE:
            continue
        layer_a = elem_layer_map.get(pair.elem_a, 0)
        layer_b = elem_layer_map.get(pair.elem_b, 0)
        if layer_a > max_layer_num or layer_b > max_layer_num:
            pair.state.status = ContactStatus.INACTIVE


def count_same_layer_pairs(
    pairs: Sequence[ContactPair],
    elem_layer_map: dict[int, int] | None,
) -> int:
    """同層ペア数を返す（除外効果の事前評価用）.

    Args:
        pairs: 接触ペアリスト
        elem_layer_map: 要素→層番号マップ

    Returns:
        同層ペア数
    """
    if not elem_layer_map:
        return 0
    count = 0
    for pair in pairs:
        if pair.state.status == ContactStatus.INACTIVE:
            continue
        layer_a = elem_layer_map.get(pair.elem_a, -1)
        layer_b = elem_layer_map.get(pair.elem_b, -1)
        if layer_a == layer_b and layer_a >= 0:
            count += 1
    return count
