"""要素モジュール — 梁要素・断面特性・アセンブラ.

公開 API:
  - BeamSection, BeamSection2D: 梁断面特性データクラス（frozen）
  - BeamForces3D: 断面力データクラス（frozen）

ULCRBeamAssembler はプライベートモジュールから直接インポート:
  from xkep_cae.elements._beam_assembler import ULCRBeamAssembler
"""

from xkep_cae.elements._beam_cr import BeamForces3D
from xkep_cae.elements._beam_section import BeamSection, BeamSection2D

__all__ = [
    "BeamForces3D",
    "BeamSection",
    "BeamSection2D",
]
