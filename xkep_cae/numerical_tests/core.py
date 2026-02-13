"""数値試験フレームワーク — コアデータクラス・解析解・メッシュ生成.

静的試験（3点曲げ・4点曲げ・引張・ねん回）と周波数応答試験の
共通データ構造を定義する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# 試験種別定数
# ---------------------------------------------------------------------------
TEST_TYPES_STATIC = ("bend3p", "bend4p", "tensile", "torsion")
TEST_TYPES_ALL = (*TEST_TYPES_STATIC, "freq_response")

BeamType = Literal["eb2d", "timo2d", "timo3d"]
SupportCondition = Literal["roller", "pin"]


# ---------------------------------------------------------------------------
# 静的試験コンフィグ
# ---------------------------------------------------------------------------
@dataclass
class NumericalTestConfig:
    """数値試験の定義.

    Attributes:
        name: 試験名 ("bend3p", "bend4p", "tensile", "torsion")
        beam_type: 梁タイプ ("eb2d", "timo2d", "timo3d")
        E: ヤング率 [Pa]
        nu: ポアソン比
        length: 試料長さ [m]
        n_elems: 要素分割数
        load_value: 荷重値 P [N] or T [N·m]
        section_shape: 断面形状 ("rectangle", "circle", "pipe")
        section_params: 断面パラメータ dict
            - rectangle: {"b": float, "h": float}
            - circle: {"d": float}
            - pipe: {"d_outer": float, "d_inner": float}
        load_span: 荷重スパン a（4点曲げのみ）[m]
        support_condition: 支持条件 ("roller" or "pin", 曲げ試験用)
    """

    name: str
    beam_type: BeamType
    E: float
    nu: float
    length: float
    n_elems: int
    load_value: float
    section_shape: str = "rectangle"
    section_params: dict = field(default_factory=lambda: {"b": 10.0, "h": 20.0})
    load_span: float | None = None
    support_condition: SupportCondition = "roller"

    def __post_init__(self) -> None:
        if self.name not in TEST_TYPES_STATIC:
            raise ValueError(
                f"試験名は {TEST_TYPES_STATIC} のいずれか: {self.name}"
            )
        if self.beam_type not in ("eb2d", "timo2d", "timo3d"):
            raise ValueError(f"beam_type は eb2d/timo2d/timo3d: {self.beam_type}")
        if self.name == "bend4p" and self.load_span is None:
            raise ValueError("4点曲げ試験には load_span が必要です。")
        if self.name == "torsion" and self.beam_type != "timo3d":
            raise ValueError("ねん回試験は 3D 梁 (timo3d) のみ対応。")

    @property
    def G(self) -> float:
        """せん断弾性率."""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def span_ratio(self) -> float:
        """スパン比 L/h（曲げ試験での摩擦影響の目安）."""
        h = self.section_params.get("h", self.section_params.get("d", 0.0))
        if h <= 0:
            return float("inf")
        return self.length / h


# ---------------------------------------------------------------------------
# 周波数応答試験コンフィグ
# ---------------------------------------------------------------------------
@dataclass
class FrequencyResponseConfig:
    """周波数応答試験の定義.

    片端保持（カンチレバー）のもう片端への変位付加 or 加速度付加。

    Attributes:
        beam_type: 梁タイプ ("eb2d", "timo2d", "timo3d")
        E: ヤング率 [Pa]
        nu: ポアソン比
        rho: 密度 [kg/m³]
        length: 試料長さ [m]
        n_elems: 要素分割数
        section_shape: 断面形状
        section_params: 断面パラメータ
        freq_min: 最小周波数 [Hz]
        freq_max: 最大周波数 [Hz]
        n_freq: 周波数分割数
        excitation_type: 励起タイプ ("displacement" or "acceleration")
        excitation_dof: 励起する自由端DOF方向 ("uy", "uz", "theta_z" 等)
        response_dof: 応答を取得するDOF方向 (None=全DOF)
        damping_alpha: Rayleigh減衰 α（質量比例）
        damping_beta: Rayleigh減衰 β（剛性比例）
    """

    beam_type: BeamType
    E: float
    nu: float
    rho: float
    length: float
    n_elems: int
    section_shape: str = "rectangle"
    section_params: dict = field(default_factory=lambda: {"b": 10.0, "h": 20.0})
    freq_min: float = 1.0
    freq_max: float = 1000.0
    n_freq: int = 200
    excitation_type: str = "displacement"
    excitation_dof: str = "uy"
    response_dof: str | None = None
    damping_alpha: float = 0.0
    damping_beta: float = 0.0

    def __post_init__(self) -> None:
        if self.beam_type not in ("eb2d", "timo2d", "timo3d"):
            raise ValueError(f"beam_type は eb2d/timo2d/timo3d: {self.beam_type}")
        if self.excitation_type not in ("displacement", "acceleration"):
            raise ValueError("excitation_type は displacement/acceleration")
        if self.rho <= 0:
            raise ValueError(f"密度 rho は正値: {self.rho}")

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))


# ---------------------------------------------------------------------------
# 静的試験結果
# ---------------------------------------------------------------------------
@dataclass
class StaticTestResult:
    """静的試験の結果.

    Attributes:
        config: 元の試験コンフィグ
        node_coords: 節点座標 (n_nodes, ndim)
        displacement: 全節点の変位ベクトル (ndof,)
        element_forces: 各要素の断面力リスト [(forces_node1, forces_node2), ...]
        displacement_max: 着目点の最大変位
        displacement_analytical: 解析解（存在する場合）
        relative_error: 解析解との相対誤差（存在する場合）
        max_bending_stress: 最大曲げ応力
        max_shear_stress: 最大せん断応力
        friction_warning: 摩擦に関する注記（曲げ試験）
        solver_info: ソルバー情報 dict
    """

    config: NumericalTestConfig
    node_coords: np.ndarray
    displacement: np.ndarray
    element_forces: list
    displacement_max: float
    displacement_analytical: float | None = None
    relative_error: float | None = None
    max_bending_stress: float = 0.0
    max_shear_stress: float = 0.0
    friction_warning: str = ""
    solver_info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 周波数応答試験結果
# ---------------------------------------------------------------------------
@dataclass
class FrequencyResponseResult:
    """周波数応答試験の結果.

    Attributes:
        config: 元のコンフィグ
        frequencies: 周波数配列 [Hz] (n_freq,)
        transfer_function: 伝達関数（複素数）(n_freq,) or (n_freq, n_response_dof)
        magnitude: 振幅 |H(ω)| (n_freq,) or (n_freq, n_response_dof)
        phase_deg: 位相 [deg] (n_freq,) or (n_freq, n_response_dof)
        natural_frequencies: 推定固有振動数 [Hz]（ピーク検出）
        node_coords: 節点座標
    """

    config: FrequencyResponseConfig
    frequencies: np.ndarray
    transfer_function: np.ndarray
    magnitude: np.ndarray
    phase_deg: np.ndarray
    natural_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    node_coords: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# メッシュ生成
# ---------------------------------------------------------------------------
def generate_beam_mesh_2d(
    n_elems: int,
    total_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """2D梁のメッシュ（x軸方向）を生成する.

    Args:
        n_elems: 要素数
        total_length: 全長

    Returns:
        nodes: (n_nodes, 2) 節点座標
        connectivity: (n_elems, 2) 要素接続
    """
    n_nodes = n_elems + 1
    x = np.linspace(0, total_length, n_nodes)
    nodes = np.column_stack([x, np.zeros(n_nodes)])
    connectivity = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])
    return nodes, connectivity


def generate_beam_mesh_3d(
    n_elems: int,
    total_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """3D梁のメッシュ（x軸方向）を生成する.

    Args:
        n_elems: 要素数
        total_length: 全長

    Returns:
        nodes: (n_nodes, 3) 節点座標
        connectivity: (n_elems, 2) 要素接続
    """
    n_nodes = n_elems + 1
    x = np.linspace(0, total_length, n_nodes)
    nodes = np.column_stack([x, np.zeros(n_nodes), np.zeros(n_nodes)])
    connectivity = np.column_stack([np.arange(n_elems), np.arange(1, n_nodes)])
    return nodes, connectivity


# ---------------------------------------------------------------------------
# 断面プロパティ生成ヘルパー
# ---------------------------------------------------------------------------
def _build_section_props(
    section_shape: str,
    section_params: dict,
    beam_type: BeamType,
    nu: float,
) -> dict:
    """断面パラメータ dict から、解析に必要な A, I, Iy, Iz, J, kappa 等を返す.

    Returns:
        dict with keys: A, I (2D), Iy, Iz, J (3D), kappa, kappa_y, kappa_z,
                         shape, r_max (最外縁距離)
    """
    if section_shape == "rectangle":
        b = section_params["b"]
        h = section_params["h"]
        A = b * h
        Iy = b * h**3 / 12.0
        Iz = h * b**3 / 12.0
        Ixy = Iz  # 2D default: xy面内
        a_long = max(b, h)
        b_short = min(b, h)
        ratio = b_short / a_long
        J = a_long * b_short**3 * (1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio**4 / 12.0))
        kappa = 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)
        r_max_y = h / 2.0  # xz面曲げの最外縁
        r_max_z = b / 2.0  # xy面曲げの最外縁
    elif section_shape == "circle":
        d = section_params["d"]
        r = d / 2.0
        A = math.pi * r**2
        I_val = math.pi * d**4 / 64.0
        Iy = I_val
        Iz = I_val
        Ixy = I_val
        J = math.pi * d**4 / 32.0
        kappa = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
        r_max_y = r
        r_max_z = r
    elif section_shape == "pipe":
        d_o = section_params["d_outer"]
        d_i = section_params["d_inner"]
        r_o, r_i = d_o / 2.0, d_i / 2.0
        A = math.pi * (r_o**2 - r_i**2)
        I_val = math.pi * (d_o**4 - d_i**4) / 64.0
        Iy = I_val
        Iz = I_val
        Ixy = I_val
        J = math.pi * (d_o**4 - d_i**4) / 32.0
        kappa = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
        r_max_y = r_o
        r_max_z = r_o
    else:
        raise ValueError(f"未対応の断面形状: {section_shape}")

    return {
        "A": A,
        "I": Ixy,
        "Iy": Iy,
        "Iz": Iz,
        "J": J,
        "kappa": kappa,
        "kappa_y": kappa,
        "kappa_z": kappa,
        "shape": section_shape,
        "r_max_y": r_max_y,
        "r_max_z": r_max_z,
    }


# ---------------------------------------------------------------------------
# 解析解
# ---------------------------------------------------------------------------
def analytical_bend3p(
    P: float,
    L: float,
    E: float,
    I: float,  # noqa: E741
    kappa: float | None = None,
    G: float | None = None,
    A: float | None = None,
) -> dict:
    """3点曲げ試験の解析解.

    Args:
        P: 荷重（正 = 下向き）
        L: スパン長
        E: ヤング率
        I: 断面二次モーメント
        kappa: せん断補正係数（Timoshenko用）
        G: せん断弾性率（Timoshenko用）
        A: 断面積（Timoshenko用）

    Returns:
        dict: delta_mid, V_max, M_max, delta_eb (EB理論分), delta_shear (せん断分)
    """
    delta_eb = P * L**3 / (48.0 * E * I)
    delta_shear = 0.0
    if kappa is not None and G is not None and A is not None:
        delta_shear = P * L / (4.0 * kappa * G * A)
    return {
        "delta_mid": delta_eb + delta_shear,
        "delta_eb": delta_eb,
        "delta_shear": delta_shear,
        "V_max": abs(P) / 2.0,
        "M_max": abs(P) * L / 4.0,
    }


def analytical_bend4p(
    P: float,
    L: float,
    a: float,
    E: float,
    I: float,  # noqa: E741
    kappa: float | None = None,
    G: float | None = None,
    A: float | None = None,
) -> dict:
    """4点曲げ試験の解析解.

    2点対称荷重（各 P）、両端単純支持。
    反力は各支点 P ずつ。純曲げ区間 (a < x < L-a) で M = Pa。

    Euler-Bernoulli解:
      δ_mid = Pa(3L²-4a²) / (24EI)
      ※ 分母 24EI は、単一集中荷重の式 (48EI) の2倍
        （2つの対称荷重の重ね合わせ）。

    Timoshenko補正:
      δ_shear = Pa / (κGA)

    Args:
        P: 各荷重点の荷重（正 = 下向き）
        L: スパン長
        a: 荷重スパン（支点から荷重点までの距離）
        E: ヤング率
        I: 断面二次モーメント
        kappa, G, A: Timoshenko用

    Returns:
        dict: delta_mid, V_max, M_max (純曲げ区間), delta_eb, delta_shear
    """
    delta_eb = P * a * (3.0 * L**2 - 4.0 * a**2) / (24.0 * E * I)
    delta_shear = 0.0
    if kappa is not None and G is not None and A is not None:
        delta_shear = P * a / (kappa * G * A)
    return {
        "delta_mid": delta_eb + delta_shear,
        "delta_eb": delta_eb,
        "delta_shear": delta_shear,
        "V_max": abs(P),
        "M_max": abs(P) * a,
    }


def analytical_tensile(
    P: float,
    L: float,
    E: float,
    A: float,
) -> dict:
    """引張試験の解析解.

    Returns:
        dict: delta, N (軸力)
    """
    return {
        "delta": P * L / (E * A),
        "N": P,
    }


def analytical_torsion(
    T: float,
    L: float,
    G: float,
    J: float,
    r_max: float,
) -> dict:
    """ねん回試験の解析解.

    Returns:
        dict: theta (ねじり角 [rad]), Mx, tau_max
    """
    return {
        "theta": T * L / (G * J),
        "Mx": T,
        "tau_max": abs(T) * r_max / J,
    }


# ---------------------------------------------------------------------------
# 摩擦滑り影響の評価
# ---------------------------------------------------------------------------
def assess_friction_effect(
    test_name: str,
    span_ratio: float,
    support_condition: SupportCondition,
) -> str:
    """3点/4点曲げ試験における支持治具摩擦の影響を評価する.

    支持治具表面の摩擦滑りの影響を実用的に判定する。

    判定基準:
    - L/h > 10: 摩擦影響は無視可能（<1%誤差）→ roller/pin どちらでも可
    - 4 < L/h ≤ 10: 摩擦影響は軽微（1-5%誤差）→ roller 推奨、注意喚起
    - L/h ≤ 4: 摩擦影響が顕著 → 曲げ試験としての妥当性に警告

    物理的背景:
    梁の撓みに伴い支持点で水平方向の変位が生じる。
    - 摩擦が十分大きい場合 → pin支持（水平拘束あり）に近い
    - 摩擦が小さい場合 → roller支持（水平自由）に近い
    - 線形微小変形解析では、この差は O(δ²/L²) のオーダーで
      通常の試験条件（δ/L << 1）では無視可能。
    - 但し短スパン（太い梁）では水平拘束がアーチ効果を生み、
      見かけの剛性が上昇する。

    Returns:
        str: 判定メッセージ（空文字列 = 問題なし）
    """
    if test_name not in ("bend3p", "bend4p"):
        return ""

    msgs = []
    if support_condition == "roller":
        mode_desc = "ローラー支持（水平自由）"
    else:
        mode_desc = "ピン支持（水平拘束）"

    if span_ratio > 10:
        msgs.append(
            f"L/h={span_ratio:.1f} > 10: 摩擦影響は無視可能。"
            f"現在の支持条件: {mode_desc}"
        )
    elif span_ratio > 4:
        msgs.append(
            f"L/h={span_ratio:.1f} (4-10): 摩擦影響は軽微（1-5%程度）。"
            f"現在の支持条件: {mode_desc}。"
            f"実測との比較時はローラー/ピンの両方で検証することを推奨。"
        )
    else:
        msgs.append(
            f"警告: L/h={span_ratio:.1f} ≤ 4: 短スパン梁のため支持条件の影響が顕著。"
            f"水平拘束によるアーチ効果で見かけ剛性が上昇する可能性あり。"
            f"現在の支持条件: {mode_desc}。"
            f"曲げ試験としての妥当性を再検討してください。"
        )

    return " ".join(msgs)
