"""数理契約の型定義.

status-346 で新設。Process が満たすべき数理的事実を frozen dataclass で
宣言可能にする 5 種の契約型を提供する。

契約階層
--------
- `MathematicalContract` (抽象基底)
  - `IdentityContract`     : LHS ≡ RHS（恒等式、TeX 文字列）
  - `InequalityContract`   : LHS ≥/≤ RHS（不等式不変量）
  - `FDConsistencyContract`: K @ du ≈ (f(u+εdu) − f(u))/ε（FD 整合性）
  - `SymmetryContract`     : K = K^T または K = −K^T（対称性）
  - `TermExpansionContract`: K = Σ_k K_term_k（項展開網羅性）★鍵

**項展開網羅性契約の役割**
--------------------------
`TermExpansionContract` は K_mat x/z カップリング問題（status-344）のような
「解析接線の一部の項が実装で抜けている」欠落を型レベルで可視化する。
`term_processes` に列挙された Process が全て実装されていることを静的検査
（C19、Phase E）し、合計が FD と一致することを動的検証（Phase D Dispatcher）
する二段構え。

**型の設計方針**
----------------
1. 式そのものは TeX 文字列として保持（`equation_ref` で台帳参照）。
   シンボリック処理 (sympy 等) は導入しない。
2. 検証可能な部分（FD 整合・対称性・項網羅性）は数値契約として表現。
3. frozen=True で不変性を担保（C17 契約との整合）。
4. `severity` で CI での扱いを制御（hard=必須、soft=警告、nightly=日次）。

設計仕様: docs/mathematics.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# severity のリテラル型
# - "hard"   : CI で違反検出時 fail（必須契約）
# - "soft"   : CI で警告のみ（情報提供）
# - "nightly": 夜間/PR ラベル運用、通常 CI では検査しない（重量 FD 検査用）
Severity = Literal["hard", "soft", "nightly"]

# 対称性種別
# - "symmetric": M == M^T（例: K_mat）
# - "skew"     : M == -M^T（例: 回転生成子）
SymmetryKind = Literal["symmetric", "skew"]

# 不等式方向
InequalityKind = Literal["geq", "leq", "gt", "lt"]

# 項合成パターン
# - "sum"    : K = Σ K_k（全項加算）
# - "add_sub": K = K_a − K_b + K_c 等（符号付き加算）
CombinatorKind = Literal["sum", "add_sub"]


@dataclass(frozen=True)
class MathematicalContract:
    """数理契約の抽象基底クラス.

    全ての具象契約は frozen dataclass として本クラスを継承する。
    C17（private dataclass 衛生）と整合し、再現性・並行実行安全性を担保する。

    Attributes:
        name: 契約の一意名（例 ``"K_c_fd_consistency"``）。同一 Process 内で
            重複不可。`DiagnosticDispatcher` が契約違反を dispatch する際の
            キーにも使用される。
        equation_ref: 対応する数式の参照（例 ``"03_huber_contact_penalty.md#eq-kc"``）。
            `docs/math/` 台帳配下の md + アンカー形式。Phase B で台帳整備。
        severity: CI での扱い。詳細は module docstring の `Severity` 参照。
        description: 人間可読な契約の意味。省略可（デフォルト空文字列）。
    """

    name: str
    equation_ref: str
    severity: Severity = "hard"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__}.name は必須です（空文字列不可）")
        if not self.equation_ref:
            raise ValueError(
                f"{type(self).__name__}({self.name}).equation_ref は必須です。"
                "docs/math/ 台帳のアンカー参照を指定してください"
            )
        if self.severity not in ("hard", "soft", "nightly"):
            raise ValueError(
                f"{type(self).__name__}({self.name}).severity は "
                f"'hard'/'soft'/'nightly' のいずれか必須: got {self.severity!r}"
            )


@dataclass(frozen=True)
class IdentityContract(MathematicalContract):
    """恒等式契約: LHS ≡ RHS.

    人間可読な方程式の表明。数値検証は行わず、設計文書として機能する。
    （数値検証可能な恒等式は `FDConsistencyContract` / `SymmetryContract` 等
    の具象契約で表現する）

    Attributes:
        lhs: 左辺の TeX 文字列（例 ``"K_c"``）。
        rhs: 右辺の TeX 文字列（例 ``"\\partial f_c / \\partial u"``）。

    Example:
        >>> IdentityContract(
        ...     name="K_c_is_jacobian_of_fc",
        ...     equation_ref="03_huber_contact_penalty.md#eq-kc-def",
        ...     lhs="K_c",
        ...     rhs="\\partial f_c / \\partial u",
        ...     severity="soft",
        ... )
    """

    lhs: str = ""
    rhs: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.lhs or not self.rhs:
            raise ValueError(f"IdentityContract({self.name}) は lhs/rhs 両方必須です")


@dataclass(frozen=True)
class InequalityContract(MathematicalContract):
    """不等式契約: LHS {≥,≤,>,<} RHS.

    不変量として保持すべき不等式を宣言する。
    例: 法線ペナルティ力が非負 `p_n ≥ 0`。

    Attributes:
        expr: 検査対象の量を示す TeX 文字列または参照名
            （例 ``"p_n"`` や ``"||R||"``）。
        kind: 不等式方向。``"geq"``/``"leq"``/``"gt"``/``"lt"``。
        bound: 比較対象の数式または値（例 ``"0"`` や ``"tol"``）。

    Example:
        >>> InequalityContract(
        ...     name="p_n_nonneg",
        ...     equation_ref="03_huber_contact_penalty.md#eq-pn",
        ...     expr="p_n",
        ...     kind="geq",
        ...     bound="0",
        ... )
    """

    expr: str = ""
    kind: InequalityKind = "geq"
    bound: str = "0"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.expr:
            raise ValueError(f"InequalityContract({self.name}) は expr 必須です")
        if self.kind not in ("geq", "leq", "gt", "lt"):
            raise ValueError(
                f"InequalityContract({self.name}).kind は "
                f"'geq'/'leq'/'gt'/'lt' のいずれか必須: got {self.kind!r}"
            )


@dataclass(frozen=True)
class FDConsistencyContract(MathematicalContract):
    """FD 整合性契約: K @ du ≈ (f(u+εdu) − f(u)) / ε.

    Process が計算する接線剛性 `K` が、対応する力 `f` の有限差分と整合する
    ことを宣言する。status-342/344 の K_c FD 診断の基盤。

    Attributes:
        vector_name: FD 対象の力ベクトル名（例 ``"f_c"``、``"f_int"``）。
        jacobian_name: 整合検査する行列名（例 ``"K_c"``、``"K_mat"``）。
        tol_rel: 相対誤差の許容値。現状 7本撚線で mat_only 1.8%、
            19本で 44% → 本契約 5e-3 で Phase C status-353 までに達成目標。
        perturbation: FD 摂動幅 ε。デフォルト 1e-7（既存 `ContactKcComponent
            FDDiagnosticProcess` と整合）。
        scope: DOF 成分スコープ（例 ``("x", "z")``）。空タプルは全成分対象。
            x/z カップリング問題の追跡用に成分別許容を提供。

    Example:
        >>> FDConsistencyContract(
        ...     name="K_c_fd_consistency",
        ...     equation_ref="03_huber_contact_penalty.md#eq-kc-fd",
        ...     vector_name="f_c",
        ...     jacobian_name="K_c",
        ...     tol_rel=5e-3,
        ...     scope=("x", "z"),
        ...     severity="nightly",
        ... )
    """

    vector_name: str = ""
    jacobian_name: str = ""
    tol_rel: float = 1e-5
    perturbation: float = 1e-7
    scope: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.vector_name or not self.jacobian_name:
            raise ValueError(
                f"FDConsistencyContract({self.name}) は vector_name/jacobian_name 両方必須です"
            )
        if self.tol_rel <= 0.0:
            raise ValueError(
                f"FDConsistencyContract({self.name}).tol_rel は正の値必須: got {self.tol_rel}"
            )
        if self.perturbation <= 0.0:
            raise ValueError(
                f"FDConsistencyContract({self.name}).perturbation は正の値必須: "
                f"got {self.perturbation}"
            )


@dataclass(frozen=True)
class SymmetryContract(MathematicalContract):
    """対称性契約: K = K^T または K = −K^T.

    行列の対称性・反対称性を宣言する。K_mat は対称、回転生成子等は反対称。

    Attributes:
        matrix_name: 検査対象の行列名（例 ``"K_mat"``）。
        kind: ``"symmetric"`` or ``"skew"``。
        tol: 対称性誤差の絶対許容値 `||K − K^T||_F / ||K||_F`。

    Example:
        >>> SymmetryContract(
        ...     name="K_mat_symmetric",
        ...     equation_ref="03_huber_contact_penalty.md#eq-kmat",
        ...     matrix_name="K_mat",
        ...     kind="symmetric",
        ...     tol=1e-10,
        ... )
    """

    matrix_name: str = ""
    kind: SymmetryKind = "symmetric"
    tol: float = 1e-10

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.matrix_name:
            raise ValueError(f"SymmetryContract({self.name}) は matrix_name 必須です")
        if self.kind not in ("symmetric", "skew"):
            raise ValueError(
                f"SymmetryContract({self.name}).kind は "
                f"'symmetric'/'skew' のいずれか必須: got {self.kind!r}"
            )
        if self.tol < 0.0:
            raise ValueError(f"SymmetryContract({self.name}).tol は非負値必須: got {self.tol}")


@dataclass(frozen=True)
class TermExpansionContract(MathematicalContract):
    """項展開網羅性契約: K = Σ_k K_term_k.

    **本契約が MCDD の核心。** 解析接線を項別 Process に分解し、各項 Process が
    全て実装されていることを静的検査（C19）、合計が FD と一致することを動的
    検証する。K_mat x/z カップリング問題（status-344）のような「一部の項の
    実装が抜けている」欠落を構造的に検出する。

    Attributes:
        total_name: 合計行列の名前（例 ``"K_c"``）。
        term_names: 各項の名前タプル（例 ``("K_mat_nn", "K_mat_ndir",
            "K_closest", "K_hermite_adj", "K_geo", "K_st")``）。
        providers: 各項を実装する Process クラス名のタプル。`term_names` と
            長さ一致必須。C19 で `ProcessRegistry` 経由で実在確認。
            **文字列** で保持するのは循環 import 回避と遅延解決のため。
        combinator: 項合成パターン。``"sum"``（全加算）または ``"add_sub"``
            （符号付き加算、例 K_c = K_mat − K_geo + K_st）。
        tol_rel: 合計と FD の相対誤差許容値。

    Example:
        >>> TermExpansionContract(
        ...     name="K_c_term_expansion",
        ...     equation_ref="03_huber_contact_penalty.md#eq-kc-full-decomposition",
        ...     total_name="K_c",
        ...     term_names=("K_mat_nn", "K_mat_ndir", "K_geo", "K_st"),
        ...     providers=(
        ...         "KcNormalStiffnessProcess",
        ...         "KcNormalDirectionStiffnessProcess",
        ...         "KcGeoStiffnessProcess",
        ...         "KcStStiffnessProcess",
        ...     ),
        ...     combinator="add_sub",
        ...     tol_rel=5e-3,
        ...     severity="nightly",
        ... )
    """

    total_name: str = ""
    term_names: tuple[str, ...] = field(default_factory=tuple)
    providers: tuple[str, ...] = field(default_factory=tuple)
    combinator: CombinatorKind = "sum"
    tol_rel: float = 1e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.total_name:
            raise ValueError(f"TermExpansionContract({self.name}) は total_name 必須です")
        if not self.term_names:
            raise ValueError(
                f"TermExpansionContract({self.name}) は term_names 必須です（最低 1 項）"
            )
        if len(self.term_names) != len(self.providers):
            raise ValueError(
                f"TermExpansionContract({self.name}): "
                f"term_names ({len(self.term_names)}) と "
                f"providers ({len(self.providers)}) の長さが一致していません"
            )
        if self.combinator not in ("sum", "add_sub"):
            raise ValueError(
                f"TermExpansionContract({self.name}).combinator は "
                f"'sum'/'add_sub' のいずれか必須: got {self.combinator!r}"
            )
        if self.tol_rel <= 0.0:
            raise ValueError(
                f"TermExpansionContract({self.name}).tol_rel は正の値必須: got {self.tol_rel}"
            )
        # providers の重複チェック（同一 Process を二重登録しない）
        if len(set(self.providers)) != len(self.providers):
            raise ValueError(
                f"TermExpansionContract({self.name}): providers に重複: {self.providers}"
            )
