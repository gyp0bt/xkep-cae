"""pycae.core - 要素・構成則・断面の抽象インタフェース定義."""

from pycae.core.element import ElementProtocol
from pycae.core.constitutive import ConstitutiveProtocol

__all__ = ["ElementProtocol", "ConstitutiveProtocol"]
