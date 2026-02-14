# 使用例

[← README](../README.md)

## 高レベルAPI（ラベルベース）

```python
from xkep_cae.api import solve_plane_strain

u_map = solve_plane_strain(
    node_coord_array=nodes,
    node_label_df_mapping={1: (False, False), 2: (False, False)},
    node_label_load_mapping={5: (1.0, 0.0)},
    E=200e3,
    nu=0.3,
    thickness=1.0,
    elem_quads=elem_q4,
    elem_tris=elem_t3,
)
```

## Protocol API（低レベル）

```python
from xkep_cae.elements.quad4_eas_bbar import Quad4EASPlaneStrain  # EAS-4 (推奨)
from xkep_cae.elements.tri3 import Tri3PlaneStrain
from xkep_cae.materials.elastic import PlaneStrainElastic
from xkep_cae.assembly import assemble_global_stiffness
from xkep_cae.bc import apply_dirichlet
from xkep_cae.solver import solve_displacement

mat = PlaneStrainElastic(E=200e3, nu=0.3)

K = assemble_global_stiffness(
    nodes_xy,
    [(Quad4EASPlaneStrain(), conn_q4), (Tri3PlaneStrain(), conn_t3)],
    mat,
    thickness=1.0,
)

Kbc, fbc = apply_dirichlet(K, f, fixed_dofs)
u, info = solve_displacement(Kbc, fbc)
```

## 梁要素

```python
from xkep_cae.elements.beam_eb2d import EulerBernoulliBeam2D
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection2D

sec = BeamSection2D.rectangle(b=10.0, h=10.0)
beam = EulerBernoulliBeam2D(section=sec)
mat = BeamElastic1D(E=200e3)

K = assemble_global_stiffness(nodes_xy, [(beam, conn)], mat)
```

## 3D梁要素

```python
from xkep_cae.elements.beam_timo3d import TimoshenkoBeam3D
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection

sec = BeamSection.rectangle(b=10.0, h=20.0)
beam = TimoshenkoBeam3D(section=sec, kappa_y="cowper", kappa_z="cowper")
mat = BeamElastic1D(E=200e3, nu=0.3)

K = assemble_global_stiffness(nodes_xyz, [(beam, conn)], mat)

# 解析後の断面力計算
from xkep_cae.elements.beam_timo3d import beam3d_section_forces
forces_1, forces_2 = beam.section_forces(coords_elem, u_elem, mat)
print(f"軸力: {forces_1.N:.3f}, せん断力: {forces_1.Vy:.3f}, モーメント: {forces_1.Mz:.3f}")
```

## Cosserat rod（四元数ベース幾何学的厳密梁）

```python
from xkep_cae.elements.beam_cosserat import CosseratRod
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection

sec = BeamSection.circle(d=10.0)
beam = CosseratRod(section=sec, kappa_y="cowper", kappa_z="cowper")
mat = BeamElastic1D(E=200e3, nu=0.3)

coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
Ke = beam.local_stiffness(coords, mat)

# 一般化歪みの計算
strains = beam.compute_strains(coords, u_elem)
print(f"軸伸び: {strains.gamma[0]:.6f}, ねじり: {strains.kappa[0]:.6f}")
```

## 非線形解析（Newton-Raphson法 / 弧長法）

```python
import numpy as np
import scipy.sparse as sp
from xkep_cae.elements.beam_cosserat import CosseratRod, assemble_cosserat_beam
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.sections.beam import BeamSection
from xkep_cae.solver import newton_raphson

sec = BeamSection.rectangle(10.0, 10.0)
mat = BeamElastic1D(E=200e3, nu=0.3)
rod = CosseratRod(section=sec, integration_scheme="sri")  # SRI or "uniform"
n_elems, L = 20, 200.0

def tangent(u):
    K, _ = assemble_cosserat_beam(n_elems, L, rod, mat, u, stiffness=True, internal_force=False)
    return sp.csr_matrix(K)

def fint(u):
    _, f = assemble_cosserat_beam(n_elems, L, rod, mat, u, stiffness=False, internal_force=True)
    return f

f_ext = np.zeros((n_elems + 1) * 6)
f_ext[6 * n_elems + 1] = 1000.0  # y方向先端荷重

result = newton_raphson(f_ext, np.arange(6), tangent, fint, n_load_steps=10)
print(f"収束: {result.converged}, 先端変位: {result.u[6*n_elems+1]:.4f}")

# 弧長法（スナップスルー・座屈追跡）
from xkep_cae.solver import arc_length
result_al = arc_length(f_ext, np.arange(6), tangent, fint, n_steps=50, delta_l=0.5)
print(f"荷重パラメータ: {result_al.lam:.4f}, ステップ数: {result_al.n_steps}")
```

### 幾何学的非線形（大変形）

`CosseratRod(nonlinear=True)` で大回転・大変形に対応。tangent / fint のシグネチャは同じ。

```python
rod = CosseratRod(section=sec, nonlinear=True)
result = newton_raphson(f_ext, np.arange(6), tangent, fint, n_load_steps=20, max_iter=50)
```

## 弾塑性解析（材料非線形）

```python
import numpy as np
import scipy.sparse as sp
from xkep_cae.core.state import CosseratPlasticState
from xkep_cae.elements.beam_cosserat import CosseratRod, assemble_cosserat_beam_plastic
from xkep_cae.materials.beam_elastic import BeamElastic1D
from xkep_cae.materials.plasticity_1d import IsotropicHardening, Plasticity1D
from xkep_cae.sections.beam import BeamSection
from xkep_cae.solver import newton_raphson

sec = BeamSection.rectangle(10.0, 20.0)
mat = BeamElastic1D(E=200e3, nu=0.3)
rod = CosseratRod(section=sec, integration_scheme="uniform", n_gauss=1)
plas = Plasticity1D(E=200e3, iso=IsotropicHardening(sigma_y0=250.0, H_iso=1000.0))

n_elems, L = 4, 100.0
states = [CosseratPlasticState() for _ in range(n_elems)]
u = np.zeros((n_elems + 1) * 6)
f_ext = np.zeros_like(u)
f_ext[6 * n_elems] = 100_000.0  # 軸力

for step in range(5):
    lam = (step + 1) / 5
    states_trial = None

    def fint(u_, _st=states):
        nonlocal states_trial
        _, f, states_trial = assemble_cosserat_beam_plastic(
            n_elems, L, rod, mat, u_, _st, plas,
            stiffness=False, internal_force=True,
        )
        return f

    def tangent(u_, _st=states):
        K, _, _ = assemble_cosserat_beam_plastic(
            n_elems, L, rod, mat, u_, _st, plas,
            stiffness=True, internal_force=False,
        )
        return sp.csr_matrix(K)

    result = newton_raphson(
        lam * f_ext, np.arange(6), tangent, fint,
        n_load_steps=1, u0=u, show_progress=False,
    )
    u = result.u
    states = [s.copy() for s in states_trial]
```

## 数値試験フレームワーク

```python
from xkep_cae.numerical_tests import (
    NumericalTestConfig, run_test, run_all_tests,
    export_static_csv, parse_test_input,
)

# 関数引数で指定
cfg = NumericalTestConfig(
    name="bend3p", beam_type="timo2d", E=200e3, nu=0.3,
    length=100.0, n_elems=10, load_value=1000.0,
    section_shape="rectangle", section_params={"b": 10.0, "h": 20.0},
)
result = run_test(cfg)
print(f"FEM: {result.displacement_max:.6f}, 解析解: {result.displacement_analytical:.6f}")

# Abaqusライクテキスト入力
cfg = parse_test_input("""
*TEST, TYPE=BEND3P
*BEAM SECTION, SECTION=RECT
 10.0, 20.0
*ELASTIC
 200000.0, 0.3
*SPECIMEN
 100.0, 10
*LOAD
 1000.0
""", beam_type="timo2d")
result = run_test(cfg)

# CSV出力
export_static_csv(result, output_dir="./results")
```

## 周波数応答試験

```python
from xkep_cae.numerical_tests import (
    FrequencyResponseConfig, run_frequency_response,
    export_frequency_response_csv,
)

cfg = FrequencyResponseConfig(
    beam_type="timo2d", E=200e3, nu=0.3, rho=7.85e-9,
    length=100.0, n_elems=10,
    section_shape="rectangle", section_params={"b": 10.0, "h": 20.0},
    freq_min=10.0, freq_max=5000.0, n_freq=200,
    excitation_type="displacement", excitation_dof="uy",
    damping_alpha=0.0, damping_beta=1e-7,
)
result = run_frequency_response(cfg)
print(f"推定固有振動数: {result.natural_frequencies}")
export_frequency_response_csv(result, output_dir="./results")
```
