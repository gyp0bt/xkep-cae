"""接触モジュール共通型定義."""

from __future__ import annotations

from enum import Enum


class ContactStatus(Enum):
    """接触状態."""

    INACTIVE = 0
    ACTIVE = 1
    SLIDING = 2
