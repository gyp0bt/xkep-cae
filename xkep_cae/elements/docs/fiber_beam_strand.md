# 撚線用ファイバー梁要素 設計仕様

[← README](../../../README.md) | [← 設計文書索引](../../../docs/design/README.md) | [← roadmap](../../../docs/roadmap.md)

## 目的

撚線（stranded cable）1本を1本の **ファイバー梁要素** として等価モデル化し、
素線間の内部摩擦による非線形ヒステリシス（ティアドロップ型 M–κ 履歴）を
セクションレベルで再現することを目的とする。

従来の線形 Timoshenko 梁（`xkep_cae/elements/_beam_cr.py` の `EI = E·I` 直接評価）は
素線間ロックとスリップを表現できないため、
1000本×10万節点規模の大域解析では **素線1本1本を陽に離散化する現行アプローチ**
（`xkep_cae/mesh/process.py` の `StrandMeshProcess` + `ContactFrictionProcess`）
と相補的な**下位モデル**として位置付ける。

> 「1000本撚線を6時間以内」の長期ターゲットに対し、
> 陽接触モデル（S3〜S6）とファイバー梁モデル（本仕様）の
> 2系統ソルバー併用で精度／速度トレードオフを選択可能にする。

---

## 背景：`work/beam_hysteresis/` 概念検証の要約

`work/beam_hysteresis/` 下の Stage 01〜08 で、以下を数値的に確認済み。
これを実装の裏付けとする。

### 01. 移動硬化則 ≡ 撚線摩擦（`01_kh_vs_friction_equivalence.py`）

1D Prager 移動硬化モデル

$$\sigma = E(\varepsilon - \varepsilon^p), \quad |\sigma - \alpha| \leq \sigma_y$$

と、1D 撚線摩擦モデル

$$f = k_{\text{strand}}(\varepsilon - u^{\text{slip}}), \quad |f - f_{\text{locked}}| \leq f_y$$

が **数学的に同型** であることを確認。
即ち、撚線間クーロンスリップはセクションレベルでは
「塑性ひずみ $\varepsilon^p$ ＋ 背応力 $\alpha$」を持つ擬塑性材料として扱える。

### 02〜03. 傾き非対称性（`02_slope_asymmetry_degradation.py`, `03_multilayer_degradation.py`）

通常の kinematic hardening は **除荷剛性 = 負荷剛性** で対称な平行四辺形ループになる。
実ケーブルの「負荷 > 除荷」の傾き非対称性を再現するには、
**接触剛性劣化**が必要：

$$k_i^{\text{contact}} = \begin{cases} k_i^{\text{virgin}} & \text{（初回スリップ前）} \\ \beta \cdot k_i^{\text{virgin}} & \text{（スリップ後, }0 < \beta < 1\text{）} \end{cases}$$

$\beta = 0.25$ で U/L = 0.77（実ケーブル計測値に近い）。

### 04〜05. 滑らかなティアドロップ（`05_smooth_teardrop.py`）

$N = 150$ 本の摩擦要素を **対数間隔**で降伏閾値を散らし、
剛性に外層ソフト・内層スティフの重み付け $k_i \propto [0.5, 1.5]$ を与えると、
角のない丸いティアドロップが得られる。
繊維断面（外繊維ほど大きな $\varepsilon$）と組み合わせることで、
右上の膨らみ／左下の細い尾が自然に現れる。

### 06. ジグ摩擦の効果（`06_jig_friction.py`）

三点曲げ試験では支持・荷重ローラー摩擦が
**乗算的**に荷重を変化させる：

$$P_{\text{load}} = \frac{4M}{L - 2\mu(r_s + r_l)}, \quad
P_{\text{unload}} = \frac{4M}{L + 2\mu(r_s + r_l)}$$

内部摩擦は strain-history 依存の**加算的**ヒステリシスを生むのに対し、
ジグ摩擦は形状を変えずに **全体的にループ幅を広げる**。
この分離が実験値同定の前提となる。

### 07〜08. サイクル動作（`07_cyclic_hysteresis.png`, `run_strand_hysteresis.py`）

- **劣化なし** 移動硬化は Cycle 2 以降で安定（シェイクダウン）し、U/L > 1 の逆非対称が残る。
- **劣化あり** は 2 サイクル目以降で定常ループに収束し、正しい U/L < 1 を保つ。
- 7本撚線の `StrandBendingOscillationProcess` 実解析でも
  $\mu = 0.15$ で散逸エネルギーループを確認（`08_strand_hysteresis.png`）。
  これがファイバー梁等価モデルの **キャリブレーション目標** となる。

---

## スコープ

### 対象
- **1D ファイバー離散化**による円形断面セクション応答（N, M_y, M_z）
- **ヒステリシス付き 1D 材料則**（移動硬化＋接触剛性劣化）
- セクション→梁要素への接続（既存 `_beam_cr.py` / `_beam_assembler.py` への入替可能な材料層）
- `StrandBendingOscillationProcess` などの高位プロセスから
  `StrandFiberBeamProcess` を選択肢に追加可能にする

### 非対象
- ねじれ硬化（将来 Phase 4.6+ で扱う）
- 素線個別の3D応力場（そのモードは既存の陽接触モデルで扱う）
- 温度・レート依存

---

## モジュール構成

既存のパッケージ規約（Process + Strategy Protocol、frozen dataclass、
コロケーション設計ドキュメント）に従う。
追加モジュールは `xkep_cae/elements/` 配下に閉じる。

```
xkep_cae/elements/
├── fiber/                       # 新規
│   ├── __init__.py
│   ├── materials.py             # 1D 材料則（Strategy 実装）
│   ├── section.py               # 円形ファイバー断面ジェネレータ
│   ├── integrator.py            # セクション積分 Process
│   ├── state.py                 # frozen dataclass: FiberState / SectionState
│   ├── strand_beam.py           # 梁要素ラッパ（N/M/EI_tangent 提供）
│   ├── docs/
│   │   └── fiber_beam_strand.md  # 本仕様のシンボリックリンク（or 本体）
│   └── tests/
│       ├── test_materials_api.py       # Test〇〇API
│       ├── test_materials_physics.py   # Test〇〇Physics
│       ├── test_section_convergence.py
│       └── test_strand_beam_hysteresis.py
├── _beam_section.py             # 既存（幾何断面）
├── _beam_cr.py                  # 既存（線形 EI 評価）
└── _beam_assembler.py           # 既存（CR+UL 統合）
```

> 本文書は `docs/design/README.md` の索引表にリンクを追加して参照する。

---

## Strategy Protocol 追加

`xkep_cae/core/strategies/protocols.py` に以下を追加する。

```python
@runtime_checkable
class Fiber1DMaterialStrategy(Protocol):
    """ファイバー1材料点の応力則.

    - 入力: 軸ひずみ ε と現在状態
    - 出力: 応力 σ、接線 dσ/dε、更新後の状態
    - 状態は frozen dataclass として返す（C17 準拠: no mutation）
    """

    def evaluate(
        self,
        eps: float,
        state: "Fiber1DState",
    ) -> tuple[float, float, "Fiber1DState"]: ...
```

既存の `ContactForceStrategy` / `FrictionStrategy` と同じ作法で
`runtime_checkable` とし、
`xkep_cae/elements/fiber/materials.py` に具象クラスを置く。

### 具象 Strategy

1. **`Elastic1D`** — 参照用。`σ = Eε`、状態変化なし。
2. **`BilinearKinematicHardening1D`** — Prager 移動硬化。 `(E, σ_y, H)`。
3. **`MultiLayerFrictionDegrading1D`** —
   `work/beam_hysteresis/05_smooth_teardrop.py` の
   `MultiLayerFriction` を frozen dataclass + 戻り値 state で再実装。
   パラメータ:
   - `E_base: float` — 素線単体の曲げ剛性分
   - `k_virgin: np.ndarray[N]` — 初期接触剛性
   - `k_degraded: np.ndarray[N]` — スリップ後剛性 $\beta k^{virgin}$
   - `f_y: np.ndarray[N]` — 各層の滑り閾値（対数分布）
   - `slip: np.ndarray[N]`（状態）
   - `slipped: np.ndarray[N, bool]`（状態）

   > 長期的にはブロック Jacobian 返却のため、`k_virgin` 配列に C17 準拠の
   > 配列ハッシュ化 or 事前確定 frozen セットアップを適用する
   > （`xkep_cae/core/strategies/_arraykey.py` パターン参照）。

### 状態 dataclass

`xkep_cae/elements/fiber/state.py`:

```python
@dataclass(frozen=True)
class Fiber1DState:
    """1ファイバー点の履歴変数."""
    eps_p: float = 0.0          # 塑性ひずみ（KH の場合）
    alpha: float = 0.0          # 背応力
    slip: tuple[float, ...] = ()        # 摩擦層スリップ
    slipped: tuple[bool, ...] = ()      # 摩擦層スリップ履歴フラグ

@dataclass(frozen=True)
class SectionState:
    """断面全ファイバーの履歴変数."""
    fibers: tuple[Fiber1DState, ...]    # len = n_fiber
```

**全ての履歴は frozen dataclass**。
材料の `evaluate()` は `(sigma, dsigma_deps, new_state)` を返し、
呼び出し側が新状態を明示的に保持する（現在の `work/beam_hysteresis/`
スクリプトは mutation だが、本格移植時に不変化する）。

---

## ファイバー断面ジェネレータ

`xkep_cae/elements/fiber/section.py`:

```python
@dataclass(frozen=True)
class CircularFiberSection:
    """円形断面のファイバー離散化.

    デフォルトは y 方向のみの 1D 離散（平面曲げ）。
    3D 二軸曲げ用は y-z 格子バージョンを別途提供する。

    Attributes:
        diameter: 断面直径
        n_fiber: ファイバー数
        y: ファイバー座標 (shape=(n_fiber,))
        z: ファイバー座標 (shape=(n_fiber,))
        area: ファイバー面積 (shape=(n_fiber,))
    """

    diameter: float
    n_fiber: int
    y: tuple[float, ...]
    z: tuple[float, ...]
    area: tuple[float, ...]

    @classmethod
    def strip(cls, diameter: float, n_fiber: int) -> "CircularFiberSection":
        """平面曲げ用の y-方向ストリップ分割.
        既存 `work/beam_hysteresis/05_smooth_teardrop.py::FiberSection._area`
        と同じ面積計算を採用する."""
        ...

    @classmethod
    def polar(cls, diameter: float, n_radial: int, n_theta: int) -> "CircularFiberSection":
        """3D 用極座標格子分割."""
        ...
```

**16要素/ピッチ以上厳守**（CLAUDE.md テスト分類規約）に準じ、
デフォルト `n_fiber = 60` 以上とする。

---

## セクション積分 Process

`xkep_cae/elements/fiber/integrator.py`:

```python
class FiberSectionIntegratorProcess(AbstractProcess):
    """ファイバー断面積分.

    入力: 軸ひずみ ε0 + 曲率 κy, κz、旧 SectionState
    出力: (N, M_y, M_z)、接線行列 C_section = [[EA, -EA·y̅, EA·z̅], ...]、
          新 SectionState

    実装方針:
      for i in range(n_fiber):
          eps_i = eps0 - κy * y_i + κz * z_i
          sigma_i, E_t_i, fiber_state_new = material.evaluate(eps_i, fiber_state)
          N += sigma_i * A_i
          M_y -= sigma_i * y_i * A_i
          M_z += sigma_i * z_i * A_i
          C_section += E_t_i * A_i * [[1, -y, z], [-y, y², -yz], [z, -yz, z²]]

    ベクトル化:
      1D 材料則がステート持ちであるため、NumPy バッチ化は
      Strategy 側で `evaluate_batch(eps_array, state_batch)` を用意して行う。
    """

    def process(self, cfg: FiberIntegratorConfig) -> FiberIntegratorResult: ...
```

### 接線行列

平面曲げ（y 方向のみ）では

$$
\mathbf{C}_{\text{sec}} = \sum_{i=1}^{n_f} E_t^{(i)} A_i
\begin{bmatrix} 1 & -y_i \\ -y_i & y_i^2 \end{bmatrix}
$$

対角項が $EA$（軸）・$EI$（曲げ）、非対角項が **履歴依存の軸–曲げカップリング**。
弾性均質断面では中立軸対称性で非対角がゼロだが、
塑性進行時にはゼロでない。
これをそのまま `_beam_cr.py` の `EI_effective` として流す。

---

## 梁要素ラッパ

`xkep_cae/elements/fiber/strand_beam.py`:

```python
class StrandFiberBeamProcess(AbstractProcess):
    """ファイバー断面を持つ CR Timoshenko 梁要素プロセス.

    既存 `_beam_cr.timo_beam3d_cr_internal_force` との差分:
      - EI / EA を定数ではなく、セクション積分結果から取り出す
      - 要素ごとに SectionState リスト（= 全積分点×全要素）を保持
      - 更新手順:
          1. u_local から (ε0, κy, κz) を抽出
          2. 各積分点で FiberSectionIntegratorProcess を呼ぶ
          3. f_int_local を 2節点 12DOF に分配
          4. K_mat_local は C_section を補間マトリクスに挟んで構築
          5. 新 SectionState をリストに保存（frozen、置換で update）

    積分点:
      Timoshenko: 2点 Gauss（status-282 Hermite 分解と整合）
      EB: 1点 Simpson or 2点 Gauss 選択可

    大回転:
      CR formulation（既存）に委ね、材料層は local frame で動く
    """

    def process(self, cfg: StrandFiberBeamConfig) -> StrandFiberBeamResult: ...
```

### セクション状態の持ち運び

状態は **要素ごとの SectionState のタプル** として
`TimeSteppingState` に同乗させる。
リスタート解析方式（CLAUDE.md 「次の課題」欄）と整合させ、
`(u, v, a, contact_pairs, fiber_states)` を
1ステップ単位で入出力するインタフェースとする。

---

## 既存コードへの組み込みポイント

1. **`_beam_cr.timo_beam3d_cr_internal_force(...)` の呼び出し側**
   → `StrandFiberBeamProcess` に付け替える分岐を
   `_beam_assembler.ULCRBeamAssembler` に追加する。
   既存の「線形 EI モード」を破壊しない：
   - `BeamElementConfig.material_mode: Literal["elastic", "fiber"]`
   - `material_mode == "elastic"` デフォルト（後方互換）

2. **`StrandBendingOscillationProcess`**（`xkep_cae/numerical_tests/`）
   → `StrandBendingOscillationConfig` に
   `use_fiber_beam: bool = False` フラグを追加する。
   `True` のとき素線メッシュを作らず、1本のファイバー梁として解く。
   同一 API で接触モデルとファイバー梁モデルを比較可能にする。

3. **キャリブレーション**
   - `run_strand_hysteresis.py` の 7本接触モデル計算結果
     （θ–M, エネルギー散逸）を**真値**として、
     `MultiLayerFrictionDegrading1D` のパラメータ
     $(E_{\text{base}}, k_i, f_{y,i}, \beta)$ を
     非線形最小二乗（`scipy.optimize.least_squares`）で同定する。
   - 同定プロセスを `xkep_cae/tuning/` の既存チューニング基盤に載せる
     （`BenchmarkRunnerProcess` のマニフェスト記録で再現性を保証）。

---

## テスト計画

CLAUDE.md のテスト分類に従い、API テストと Physics テストを分ける。

### `TestFiber1DMaterialAPI`
- `Fiber1DMaterialStrategy` Protocol 準拠チェック（`isinstance` 検査）
- 引数／戻り値 dtype 検査
- state の frozen 性確認
- 弾性単調加重で `σ = E·ε` 再現

### `TestFiber1DMaterialPhysics`
- **単軸サイクル**: `ε = [0, 0.3, -0.3, 0.3]` で
  閉ループ＋残留ひずみ＋U/L 非対称を確認（`05_smooth_teardrop` と一致）
- **退化**: $\beta = 1.0$ で KH と厳密一致
- **塑性仕事 = ループ面積**: エネルギー整合

### `TestFiberSectionConvergence`
- 弾性極限で解析 $EI = E·π·d^4/64$ 誤差 < 1%（`n_fiber` 収束）
- ファイバー数 $n_f \in \{20, 40, 60, 80\}$ で
  負荷–除荷ループ面積の収束率を報告（status ログに残す）
- 接線行列 $\mathbf{C}_{\text{sec}}$ の FD 検証（`atol = 1e-5`）
  — `TangentFDDiagnosticProcess` を流用

### `TestStrandBeamHysteresisPhysics`
- 三点曲げシナリオ（$D=17, L=100, d_{\max}=30$）で
  `05_smooth_teardrop.py` の基準値と荷重履歴を一致させる（rtol=1%）
- ティアドロップ形状指標:
  - 残留変位 $d_{\text{res}} / d_{\max} \approx 0.06$
  - $U/L$ 傾き比 $\approx 0.77$
- サイクル2以降のシェイクダウン確認（U/L 収束 < 0.5%）
- **ジグ摩擦オプション**: `06_jig_friction.py` の乗算補正を
  境界条件層で再現し、内部摩擦効果と分離して検証

### `TestStrandBeamVsContactModel`（統合）
- 7本撚線ケース（`StrandBendingOscillationProcess`）を
  (a) 陽接触モデル、(b) ファイバー梁モデル（同定後パラメータ）で解き、
  θ–M ループを重ね描き。
- 散逸エネルギー一致度 `|ΔE_fb − ΔE_contact| / ΔE_contact < 10%`
- 計算時間比を記録（ファイバー梁が1〜2桁高速になる見込み）

全ケースで `| tee /tmp/log-$(date +%s).log` を必須。
収束ログには `[f_ref]`, `[CUTBACK:原因]`, `[SPIKE]`, `[収束型統計]` を含める
（CLAUDE.md ソルバー診断ログ規約）。

---

## 実装フェーズ

| Phase | 内容 | 完了判定 |
|-------|------|---------|
| **F1** | `fiber/state.py` + `fiber/materials.py` に `Elastic1D`, `BilinearKinematicHardening1D` 実装 | Physics テスト 6 件合格 |
| **F2** | `MultiLayerFrictionDegrading1D` 実装（frozen 化込み） | `05_smooth_teardrop.py` 再現 rtol 1% |
| **F3** | `CircularFiberSection` + `FiberSectionIntegratorProcess` | FD 接線 atol 1e-5 |
| **F4** | `StrandFiberBeamProcess` + `_beam_assembler` への配線 | 弾性 EI 一致 < 0.1%、線形梁モードとの切替動作 |
| **F5** | `StrandBendingOscillationProcess` に `use_fiber_beam` フラグ | 7本撚線との散逸エネルギー一致 < 10% |
| **F6** | キャリブレーション Process（`tuning/` 配下） + ベンチマーク記録 | `BenchmarkRunnerProcess` マニフェスト出力 |

各フェーズで feature コミット → status 新規作成 → `status-index.md` 更新 →
`roadmap.md` 更新（CLAUDE.md の必須手順）。

---

## 既知のリスク

1. **接線の FD 整合性**
   摩擦層のスイッチ（弾性↔スリップ）で接線が不連続になる。
   既存のチャタリング対策（status-284〜287 の凍結／Hertz 化と同じ発想）で
   **Huber smoothing** を材料則側にも入れる。
   パラメータは既存の `huber_delta_h` と別軸で `material_huber_delta` を持つ。

2. **状態の肥大化**
   1要素 2 積分点 × 60 ファイバー × 150 層 = 18,000 float/要素。
   1000 要素で 18M float ≈ 144 MB。現実的だが、
   層数は適応的に 30〜60 に抑えるプリセットを用意する。

3. **キャリブレーション非一意性**
   内部摩擦パラメータは観測量（θ–M ループ）に対して過パラメータ。
   対策: 降伏閾値を対数分布で固定し、
   自由パラメータは $(E_{\text{base}}, k_{\text{contact, total}}, \beta)$
   の3次元に限定する（`05_smooth_teardrop.py` と同じ）。

4. **動的解析との結合**
   `GeneralizedAlpha` 時間積分で履歴変数をどのタイミングで update するか。
   現行 UL 更新 (`update_reference`) の直前に確定させる方針。
   CR 梁の UL f_int=0 問題（`CLAUDE.md` リスタート解析節）と同一線で整理する。

---

## 参考文献

- Foti, F. & Martinelli, L. (2016) *Hysteretic bending of spiral strands* — 本モデルの学術的基礎
- Costello, G.A. *Theory of Wire Rope* — 撚線内部力学
- de Souza Neto et al. *Computational Methods for Plasticity* — 1D return mapping
- Simo, J.C. & Hughes, T.J.R. *Computational Inelasticity* — kinematic hardening 数値実装

---

## 関連リソース

- 概念検証スクリプト: `work/beam_hysteresis/01_kh_vs_friction_equivalence.py` 〜 `08_strand_hysteresis.png`
- 陽接触ベースライン: `work/beam_hysteresis/run_strand_hysteresis.py`
- 既存梁要素: `xkep_cae/elements/_beam_cr.py`, `_beam_assembler.py`
- 既存高位プロセス: `xkep_cae/numerical_tests/strand_bending_oscillation.py`
- 関連 status: 023（Phase 4.1–4.2 ファイバーモデル凍結）、
  280–312（撚線曲げ揺動収束の現状）
