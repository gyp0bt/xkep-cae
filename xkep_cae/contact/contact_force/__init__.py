"""ContactForce Strategy サブパッケージ.

接触力の評価方法（NCP / SmoothPenalty）。
"""

from xkep_cae.contact.contact_force.strategy import (
    ContactForceInput,
    ContactForceOutput,
    NCPContactForceProcess,
    SmoothPenaltyContactForceProcess,
)

__all__ = [
    "NCPContactForceProcess",
    "SmoothPenaltyContactForceProcess",
    "ContactForceInput",
    "ContactForceOutput",
]
