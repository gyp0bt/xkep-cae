"""Protocol ベースの要素・材料・アセンブリのテスト.

既存の関数ベースAPIと同じ結果が得られることを検証する。
"""

from __future__ import annotations

import numpy as np

from xkep_cae.assembly import assemble_global_stiffness, assemble_global_stiffness_mixed
from xkep_cae.bc import apply_dirichlet
from xkep_cae.core.constitutive import ConstitutiveProtocol
from xkep_cae.core.element import ElementProtocol
from xkep_cae.elements.quad4 import Quad4PlaneStrain
from xkep_cae.elements.quad4_bbar import Quad4BBarPlaneStrain
from xkep_cae.elements.tri3 import Tri3PlaneStrain
from xkep_cae.elements.tri6 import Tri6PlaneStrain
from xkep_cae.materials.elastic import PlaneStrainElastic
from xkep_cae.solver import solve_displacement


def test_protocol_isinstance():
    """全要素・材料クラスがProtocolに適合すること"""
    mat = PlaneStrainElastic(200e3, 0.3)
    assert isinstance(mat, ConstitutiveProtocol)

    for cls in [Quad4PlaneStrain, Tri3PlaneStrain, Tri6PlaneStrain, Quad4BBarPlaneStrain]:
        obj = cls()
        assert isinstance(obj, ElementProtocol), f"{cls.__name__} is not ElementProtocol"


def test_generic_assembly_matches_legacy_quad4():
    """汎用アセンブリがQ4レガシー版と同じ結果を出すこと"""
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    mat = PlaneStrainElastic(10.0, 0.25)
    conn = np.array([[0, 1, 2, 3]])

    K_new = assemble_global_stiffness(
        nodes,
        [(Quad4PlaneStrain(), conn)],
        mat,
        thickness=1.0,
        show_progress=False,
    )
    K_old = assemble_global_stiffness_mixed(
        nodes,
        conn,
        None,
        None,
        10.0,
        0.25,
        t=1.0,
        show_progress=False,
    )
    assert np.allclose(K_new.toarray(), K_old.toarray(), atol=1e-10)


def test_generic_assembly_matches_legacy_tri3():
    """汎用アセンブリがTRI3レガシー版と同じ結果を出すこと"""
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    mat = PlaneStrainElastic(5.0, 0.3)
    conn = np.array([[0, 1, 2]])

    K_new = assemble_global_stiffness(
        nodes,
        [(Tri3PlaneStrain(), conn)],
        mat,
        thickness=1.0,
        show_progress=False,
    )
    K_old = assemble_global_stiffness_mixed(
        nodes,
        None,
        conn,
        None,
        5.0,
        0.3,
        t=1.0,
        show_progress=False,
    )
    assert np.allclose(K_new.toarray(), K_old.toarray(), atol=1e-10)


def test_generic_assembly_matches_legacy_tri6():
    """汎用アセンブリがTRI6レガシー版と同じ結果を出すこと"""
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.0],
            [0.75, 0.5],
            [0.25, 0.5],
        ]
    )
    mat = PlaneStrainElastic(200e3, 0.3)
    conn = np.array([[0, 1, 2, 3, 4, 5]])

    K_new = assemble_global_stiffness(
        nodes,
        [(Tri6PlaneStrain(), conn)],
        mat,
        thickness=1.0,
        show_progress=False,
    )
    K_old = assemble_global_stiffness_mixed(
        nodes,
        None,
        None,
        conn,
        200e3,
        0.3,
        t=1.0,
        show_progress=False,
    )
    assert np.allclose(K_new.toarray(), K_old.toarray(), atol=1e-10)


def test_generic_assembly_mixed_solve():
    """汎用アセンブリでQ4+TRI3混在メッシュのソルブが成功すること"""
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 0.5],
        ]
    )
    mat = PlaneStrainElastic(100.0, 0.29)

    conn_q4 = np.array([[0, 1, 2, 3]])
    conn_t3 = np.array([[1, 4, 2]])

    K = assemble_global_stiffness(
        nodes,
        [(Quad4PlaneStrain(), conn_q4), (Tri3PlaneStrain(), conn_t3)],
        mat,
        thickness=1.0,
        show_progress=False,
    )

    ndof = K.shape[0]
    fixed_dofs = np.array([0, 1, 6, 7], dtype=int)

    # 製造解テスト
    rng = np.random.default_rng(42)
    u_true = rng.standard_normal(ndof)
    u_true[fixed_dofs] = 0.0
    f = K @ u_true

    Kbc, fbc = apply_dirichlet(K, f, fixed_dofs, 0.0)
    u, info = solve_displacement(Kbc, fbc, size_threshold=4000)

    free = np.setdiff1d(np.arange(ndof), fixed_dofs)
    assert np.allclose(u[free], u_true[free], rtol=1e-10, atol=1e-10)
