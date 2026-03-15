"""ContactForce Strategy サブパッケージ.

接触力の評価方法（NCP / SmoothPenalty）。
"""

from xkep_cae.contact.contact_force.strategy import (
    ContactForceInput,
    ContactForceOutput,
    NCPContactForceProcess,
    SmoothPenaltyContactForceProcess,
)
from xkep_cae.contact.contact_force.strategy import (
    _create_contact_force_strategy as create_contact_force_strategy,
)

__all__ = [
    "NCPContactForceProcess",
    "SmoothPenaltyContactForceProcess",
    "ContactForceInput",
    "ContactForceOutput",
    "create_contact_force_strategy",
]
