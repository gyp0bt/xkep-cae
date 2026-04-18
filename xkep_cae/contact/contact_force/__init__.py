"""ContactForce Strategy サブパッケージ.

Huber ペナルティ接触力（status-222 で一本化）。
"""

from xkep_cae.contact.contact_force.strategy import (
    ContactForceInput,
    ContactForceOutput,
    ContactForceStStiffnessInput,
    ContactForceStStiffnessOutput,
    ContactForceStStiffnessProcess,
    HuberContactForceProcess,
    KcClosestPointStiffnessOutput,
    KcClosestPointStiffnessProcess,
    KcGeoStiffnessOutput,
    KcGeoStiffnessProcess,
    KcHermiteNonlocalStiffnessOutput,
    KcHermiteNonlocalStiffnessProcess,
    KcNormalStiffnessOutput,
    KcNormalStiffnessProcess,
    KcTermAssemblyInput,
)

__all__ = [
    "HuberContactForceProcess",
    "ContactForceInput",
    "ContactForceOutput",
    # B1: 接触力 K_st
    "ContactForceStStiffnessProcess",
    "ContactForceStStiffnessInput",
    "ContactForceStStiffnessOutput",
    # Phase C-1: K_c 項別 Process (status-350)
    "KcTermAssemblyInput",
    "KcNormalStiffnessProcess",
    "KcNormalStiffnessOutput",
    "KcGeoStiffnessProcess",
    "KcGeoStiffnessOutput",
    # Phase C-2: K_hermite_adj / K_closest (status-351)
    "KcHermiteNonlocalStiffnessProcess",
    "KcHermiteNonlocalStiffnessOutput",
    "KcClosestPointStiffnessProcess",
    "KcClosestPointStiffnessOutput",
]
