"""厚肉弾性リングのコンプライアンス行列（Stage S1）.

シースモデルの力学的挙動を解析的リング理論で表現する。
z=const 断面で厚肉弾性リングとして N 本の最外層素線からの
径方向集中荷重を Fourier 展開で解析的に解く。

== 理論的背景 ==

厚肉円筒（リング）に対する Michell 応力関数解:
  φ_n = (A r^n + B r^{-n} + C r^{n+2} + D r^{-n+2}) cos(nθ)

モード 0: Lamé 解（均等内圧）
モード n≥2: Michell 解（曲げ + 膜変形）
モード 1: 剛体並進（多点荷重では励起されない）

N 本の等間隔径方向荷重に対する励起モード:
  n = 0, N, 2N, 3N, ...  （7本撚り外層6本 → n=0,6,12,...）

== コンプライアンス行列 ==

  δr_i = Σ_j C_ij F_j

ここで C は N×N 対称正定値循環行列。
Green 関数 G(θ) を Fourier モード別コンプライアンスから構成し、
C_ij = G(θ_i - θ_j) として行列を組み立てる。

参考文献:
  - Timoshenko, S.P. "Strength of Materials" Part II
  - Barber, J.R. "Elasticity" 3rd ed. — Michell solution
  - Roark's Formulas for Stress and Strain — リング荷重閉形式解
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import solve


def ring_mode0_compliance(
    a: float,
    b: float,
    E: float,
    nu: float,
    plane: str = "strain",
) -> float:
    """モード 0（均等内圧）コンプライアンス — Lamé 解.

    Parameters
    ----------
    a : float
        内径 [m]
    b : float
        外径 [m]
    E : float
        ヤング率 [Pa]
    nu : float
        ポアソン比
    plane : str
        "strain"（平面ひずみ）or "stress"（平面応力）

    Returns
    -------
    float
        c₀ = u_r(a) / p₀  （単位内圧あたりの内面径方向変位）
    """
    if a <= 0 or b <= a:
        raise ValueError(f"内径 a={a} < 外径 b={b} > 0 が必要")
    if E <= 0:
        raise ValueError(f"ヤング率 E={E} は正値が必要")
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"ポアソン比 nu={nu} は (-1, 0.5) の範囲が必要")

    if plane == "strain":
        # 平面ひずみ: u_r = a(1+ν)/[E(b²-a²)] · [(1-2ν)a² + b²]
        c0 = a * (1.0 + nu) / (E * (b**2 - a**2)) * ((1.0 - 2.0 * nu) * a**2 + b**2)
    elif plane == "stress":
        # 平面応力: u_r = a/[E(b²-a²)] · [(1-ν)a² + (1+ν)b²]
        c0 = a / (E * (b**2 - a**2)) * ((1.0 - nu) * a**2 + (1.0 + nu) * b**2)
    else:
        raise ValueError(f"plane='{plane}' は未対応。'strain' or 'stress'")

    return c0


def _build_michell_system(beta: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Michell 解の 4×4 連立方程式を構築（正規化座標系）.

    正規化: α₁ = A·a^{n-2}, α₂ = B·a^{-n-2}, α₃ = C·a^n, α₄ = D·a^{-n}
    これにより r=a での評価が β⁰=1 となり、大きな n でも数値安定。

    Parameters
    ----------
    beta : float
        外径/内径比 b/a > 1
    n : int
        Fourier モード番号 (n ≥ 2)

    Returns
    -------
    M : (4, 4) ndarray
        係数行列
    rhs : (4,) ndarray
        右辺ベクトル（単位 Fourier 圧力 p_n = 1）
    """
    M = np.zeros((4, 4))
    rhs = np.zeros(4)

    # σ_r 係数 [α₁, α₂, α₃, α₄] at r̃ = r/a
    def sigma_r_row(r_tilde: float) -> np.ndarray:
        rt = r_tilde
        return np.array(
            [
                -n * (n - 1) * rt ** (n - 2),
                -n * (n + 1) * rt ** (-n - 2),
                -(n + 1) * (n - 2) * rt**n,
                -(n + 2) * (n - 1) * rt ** (-n),
            ]
        )

    # τ_rθ 係数（n で割った後の bracket）at r̃
    def tau_row(r_tilde: float) -> np.ndarray:
        rt = r_tilde
        return np.array(
            [
                -(n - 1) * rt ** (n - 2),
                (n + 1) * rt ** (-n - 2),
                -(n + 1) * rt**n,
                (n - 1) * rt ** (-n),
            ]
        )

    # r̃=1 (内面)
    M[0, :] = sigma_r_row(1.0)
    M[2, :] = tau_row(1.0)
    # r̃=β (外面)
    M[1, :] = sigma_r_row(beta)
    M[3, :] = tau_row(beta)

    rhs[0] = -1.0  # σ_r(a) = -p_n with p_n = 1

    return M, rhs


def ring_mode_n_compliance(
    a: float,
    b: float,
    E: float,
    nu: float,
    n: int,
    plane: str = "strain",
) -> float:
    """モード n (n≥2) コンプライアンス — Michell 解.

    Parameters
    ----------
    a : float
        内径 [m]
    b : float
        外径 [m]
    E : float
        ヤング率 [Pa]
    nu : float
        ポアソン比
    n : int
        Fourier モード番号 (n ≥ 2)
    plane : str
        "strain"（平面ひずみ）or "stress"（平面応力）

    Returns
    -------
    float
        cₙ = U_r(a) / p_n  （単位 Fourier 圧力あたりの内面径方向変位）
    """
    if n < 2:
        raise ValueError(f"n={n} は 2 以上が必要（n=0 は ring_mode0_compliance を使用）")
    if a <= 0 or b <= a:
        raise ValueError(f"内径 a={a} < 外径 b={b} > 0 が必要")
    if E <= 0:
        raise ValueError(f"ヤング率 E={E} は正値が必要")

    G = E / (2.0 * (1.0 + nu))
    if plane == "strain":
        kappa = 3.0 - 4.0 * nu
    elif plane == "stress":
        kappa = (3.0 - nu) / (1.0 + nu)
    else:
        raise ValueError(f"plane='{plane}' は未対応。'strain' or 'stress'")

    beta = b / a

    # 4×4 連立方程式を解く
    M, rhs = _build_michell_system(beta, n)
    alpha = solve(M, rhs)  # [α₁, α₂, α₃, α₄]

    # 内面 (r=a) での径方向変位
    # 2G·U_r(a) = a·[-n·α₁ + n·α₂ + (κ-n-1)·α₃ + (κ+n-1)·α₄]
    disp_coeffs = np.array(
        [
            -n,
            n,
            kappa - n - 1.0,
            kappa + n - 1.0,
        ]
    )
    u_r_a = a / (2.0 * G) * np.dot(disp_coeffs, alpha)

    return u_r_a


def build_ring_compliance_matrix(
    N: int,
    a: float,
    b: float,
    E: float,
    nu: float,
    *,
    n_modes: int | None = None,
    plane: str = "strain",
) -> np.ndarray:
    """N×N コンプライアンス行列を構築.

    等間隔 N 点に径方向集中荷重を受ける厚肉リングの応答を表す。
    Green 関数 G(θ) を Fourier モード別コンプライアンスから構成:

      G(θ) = c₀/(2πa) + (1/(πa)) Σ_{n≥2} cₙ cos(nθ)

    C_ij = G(θ_i - θ_j)

    Parameters
    ----------
    N : int
        荷重点数（= 最外層素線本数）
    a : float
        内径 [m]
    b : float
        外径 [m]
    E : float
        ヤング率 [Pa]
    nu : float
        ポアソン比
    n_modes : int | None
        Fourier 級数の打ち切りモード数。None の場合は max(4*N, 24)。
    plane : str
        "strain"（平面ひずみ）or "stress"（平面応力）

    Returns
    -------
    C : (N, N) ndarray
        コンプライアンス行列 [m/N]。対称正定値循環行列。
    """
    if N < 2:
        raise ValueError(f"N={N} は 2 以上が必要")
    if a <= 0 or b <= a:
        raise ValueError(f"内径 a={a} < 外径 b={b} > 0 が必要")

    if n_modes is None:
        n_modes = max(4 * N, 24)

    # モード別コンプライアンス計算
    c0 = ring_mode0_compliance(a, b, E, nu, plane)
    cn = {}
    for mode in range(2, n_modes + 1):
        cn[mode] = ring_mode_n_compliance(a, b, E, nu, mode, plane)

    # Green 関数から C 行列を組み立て
    theta = np.array([2.0 * np.pi * k / N for k in range(N)])

    # 循環行列なので第0行だけ計算すれば十分
    g_row = np.zeros(N)
    for j in range(N):
        dtheta = theta[j]  # theta[0] - theta[j] = -theta[j]
        val = c0 / (2.0 * np.pi * a)
        for mode in range(2, n_modes + 1):
            val += cn[mode] / (np.pi * a) * np.cos(mode * dtheta)
        g_row[j] = val

    # 循環行列を構成
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            idx = (j - i) % N
            C[i, j] = g_row[idx]

    return C


def ring_compliance_summary(
    N: int,
    a: float,
    b: float,
    E: float,
    nu: float,
    *,
    n_modes: int | None = None,
    plane: str = "strain",
) -> dict:
    """コンプライアンス行列と診断情報を返す.

    Parameters
    ----------
    N, a, b, E, nu, n_modes, plane : see build_ring_compliance_matrix

    Returns
    -------
    dict with keys:
        C : (N, N) ndarray — コンプライアンス行列
        eigenvalues : (N,) ndarray — 固有値（昇順）
        condition_number : float — 条件数
        c0 : float — モード 0 コンプライアンス
        mode_compliances : dict[int, float] — モード別コンプライアンス
    """
    if n_modes is None:
        n_modes = max(4 * N, 24)

    C = build_ring_compliance_matrix(N, a, b, E, nu, n_modes=n_modes, plane=plane)

    eigvals = np.linalg.eigvalsh(C)
    cond = eigvals[-1] / eigvals[0] if eigvals[0] > 0 else float("inf")

    c0 = ring_mode0_compliance(a, b, E, nu, plane)
    cn = {}
    for mode in range(2, n_modes + 1):
        cn[mode] = ring_mode_n_compliance(a, b, E, nu, mode, plane)

    return {
        "C": C,
        "eigenvalues": eigvals,
        "condition_number": cond,
        "c0": c0,
        "mode_compliances": cn,
    }


# ====================================================================
# Stage S2: Fourier 分解 + 膜厚分布コンプライアンス
# ====================================================================


def fourier_decompose_profile(
    theta: np.ndarray,
    r_profile: np.ndarray,
    n_modes: int,
) -> dict:
    """円周プロファイルの Fourier 分解（Stage S2）.

    r(θ) ≈ R₀ + Σ_{n=1}^{N} [aₙ cos(nθ) + bₙ sin(nθ)]

    等間隔サンプルに対する DFT ベースの係数抽出。

    Parameters
    ----------
    theta : (M,) ndarray
        角度サンプル [rad]（等間隔推奨）
    r_profile : (M,) ndarray
        各角度での半径値
    n_modes : int
        抽出する最大 Fourier モード番号 (≥ 1)

    Returns
    -------
    dict with keys:
        R0 : float — 平均半径（DC 成分）
        a : (n_modes+1,) ndarray — cos 係数 [a₀=R₀, a₁, ..., aₙ]
        b : (n_modes+1,) ndarray — sin 係数 [b₀=0, b₁, ..., bₙ]
        n_modes : int — モード数
    """
    if n_modes < 1:
        raise ValueError(f"n_modes={n_modes} は 1 以上が必要")
    if len(theta) != len(r_profile):
        raise ValueError("theta と r_profile の長さが一致しません")

    M = len(theta)
    R0 = float(np.mean(r_profile))

    a = np.zeros(n_modes + 1)
    b = np.zeros(n_modes + 1)
    a[0] = R0

    for n in range(1, n_modes + 1):
        a[n] = 2.0 / M * np.sum(r_profile * np.cos(n * theta))
        b[n] = 2.0 / M * np.sum(r_profile * np.sin(n * theta))

    return {
        "R0": R0,
        "a": a,
        "b": b,
        "n_modes": n_modes,
    }


def evaluate_fourier_profile(
    fourier_coeffs: dict,
    theta: np.ndarray,
) -> np.ndarray:
    """Fourier 係数からプロファイルを再構築する.

    Parameters
    ----------
    fourier_coeffs : dict
        ``fourier_decompose_profile`` の返り値
    theta : (M,) ndarray
        評価する角度

    Returns
    -------
    r : (M,) ndarray
        再構築されたプロファイル
    """
    a = fourier_coeffs["a"]
    b = fourier_coeffs["b"]
    n_modes = fourier_coeffs["n_modes"]

    r = np.full_like(theta, float(a[0]))
    for n in range(1, n_modes + 1):
        r = r + a[n] * np.cos(n * theta) + b[n] * np.sin(n * theta)
    return r


def build_variable_thickness_compliance_matrix(
    N: int,
    contact_angles: np.ndarray,
    r_inner_at_contacts: np.ndarray,
    r_outer: float,
    E: float,
    nu: float,
    *,
    n_modes: int | None = None,
    plane: str = "strain",
) -> np.ndarray:
    """膜厚分布を考慮したコンプライアンス行列（Stage S2）.

    各接触点での局所内径（シース内面プロファイルから決定）を
    考慮し、非均一厚みリングとしてのコンプライアンス行列を構築する。

    接触点 i, j 間のコンプライアンスは有効内径
    a_eff = (a_i + a_j) / 2 を用いた Green 関数で評価する:

      C[i,j] = G(θ_i − θ_j; a_eff, b, E, ν)

    有効内径が全接触点で同一の場合、Stage S1 の
    ``build_ring_compliance_matrix`` と一致する。

    Parameters
    ----------
    N : int
        接触点数（最外層素線本数）
    contact_angles : (N,) ndarray
        接触点の角度 [rad]
    r_inner_at_contacts : (N,) ndarray
        各接触点でのシース内面半径 [m]
    r_outer : float
        シース外径 [m]
    E : float
        ヤング率 [Pa]
    nu : float
        ポアソン比
    n_modes : int | None
        Fourier 級数の打ち切りモード数（None → max(4*N, 24)）
    plane : str
        "strain"（平面ひずみ）or "stress"（平面応力）

    Returns
    -------
    C : (N, N) ndarray
        修正コンプライアンス行列 [m/N]。対称正定値。
    """
    if N < 2:
        raise ValueError(f"N={N} は 2 以上が必要")
    if len(contact_angles) != N:
        raise ValueError(f"contact_angles の長さ ({len(contact_angles)}) と N={N} が不一致")
    if len(r_inner_at_contacts) != N:
        raise ValueError(
            f"r_inner_at_contacts の長さ ({len(r_inner_at_contacts)}) と N={N} が不一致"
        )

    for i in range(N):
        a_i = r_inner_at_contacts[i]
        if a_i <= 0 or a_i >= r_outer:
            raise ValueError(f"内径 r_inner[{i}]={a_i} は正かつ外径 {r_outer} 未満が必要")

    if n_modes is None:
        n_modes = max(4 * N, 24)

    # 有効内径ごとのモードコンプライアンスをキャッシュ
    # （同一有効内径の重複計算を回避）
    _cache: dict[float, tuple[float, dict[int, float]]] = {}

    def _get_mode_compliances(
        a_eff: float,
    ) -> tuple[float, dict[int, float]]:
        key = round(a_eff, 15)
        if key not in _cache:
            c0 = ring_mode0_compliance(a_eff, r_outer, E, nu, plane)
            cn = {}
            for mode in range(2, n_modes + 1):
                cn[mode] = ring_mode_n_compliance(a_eff, r_outer, E, nu, mode, plane)
            _cache[key] = (c0, cn)
        return _cache[key]

    C = np.zeros((N, N))

    for i in range(N):
        for j in range(i, N):
            a_eff = (r_inner_at_contacts[i] + r_inner_at_contacts[j]) / 2.0
            dtheta = contact_angles[i] - contact_angles[j]

            c0, cn = _get_mode_compliances(a_eff)

            val = c0 / (2.0 * np.pi * a_eff)
            for mode in range(2, n_modes + 1):
                val += cn[mode] / (np.pi * a_eff) * np.cos(mode * dtheta)

            C[i, j] = val
            C[j, i] = val

    return C
