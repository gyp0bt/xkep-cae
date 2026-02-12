"""xkep_cae.core - 要素・構成則・断面の抽象インタフェース定義."""

from xkep_cae.core.constitutive import ConstitutiveProtocol
from xkep_cae.core.element import ElementProtocol

__all__ = ["ElementProtocol", "ConstitutiveProtocol"]
