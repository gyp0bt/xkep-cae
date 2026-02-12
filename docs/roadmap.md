# pycae ロードマップ

[← README](../README.md)

## プロジェクトビジョン

汎用FEMソフトでは解けないニッチドメイン問題を解くための自作有限要素ソルバー基盤。
支配方程式・構成則・要素・更新則・積分スキーマ・接触・非線形をモジュール化し、
組み合わせて問題特化ソルバーを構成できるフレームワークを目指す。

### 第一ターゲット：空間梁モデル

撚線電線・単線・ロッド等の「線」の曲げ・引張・ねじり・揺動試験を再現する。
試験荷重および運動経路を再現するモデルレジストリと、パラメータフィッティングアルゴリズムを備える。

---

## 現在地（Phase 0 完了）

### 実装済み

| カテゴリ | 内容 |
|---------|------|
| **要素** | Q4（双線形四角形）, TRI3（一次三角形）, TRI6（二次三角形）, Q4_BBAR（B̄法） |
| **材料** | 線形弾性（平面ひずみ） |
| **ソルバー** | 直接法（spsolve）, AMG反復法（pyamg） |
| **境界条件** | Dirichlet（行列消去法 / Penalty法） |
| **API** | ラベルベース高レベルAPI, 混在メッシュ対応 |
| **検証** | 製造解テスト, Abaqusベンチマーク, 実メッシュテスト |

### 未実装（現状の制約）

- 3次元要素なし（平面問題限定）
- 梁要素なし
- 非線形解析なし（幾何学/材料）
- 応力・ひずみのポスト処理なし
- 動的解析なし
- pyproject.toml等のプロジェクト標準構成なし

---

## Phase 1: アーキテクチャ再構成

**目的**: 線形平面ソルバーの実装を保持しつつ、多様な要素・構成則・ソルバーを組み合わせ可能なモジュール構成へ移行する。

### 1.1 プロジェクト基盤整備

- [x] `pyproject.toml` 作成（依存関係、ビルド設定）
- [x] テストフレームワーク（pytest）統一
- [ ] 既存テストの `tests/` 配下への整理・命名規約統一
- [x] CI設定（GitHub Actions）

### 1.2 コア抽象レイヤー設計

以下のプロトコル（Protocol / ABC）を導入する。

```
pycae/
├── core/
│   ├── element.py          # ElementProtocol: 要素の抽象インタフェース
│   ├── constitutive.py     # ConstitutiveProtocol: 構成則の抽象インタフェース
│   ├── integrator.py       # IntegratorProtocol: 時間積分スキーマ
│   ├── section.py          # SectionProtocol: 断面特性
│   ├── dof.py              # DOFの管理・番号付け
│   └── state.py            # StateVariable: 履歴変数・内部変数の管理
├── elements/               # 具象要素（既存 + 新規）
├── materials/              # 具象構成則
├── sections/               # 断面モデル
├── solvers/                # 線形・非線形ソルバー
├── assembly.py             # 要素アセンブリ
├── bc.py                   # 境界条件
└── api.py                  # ユーザーAPI
```

#### ElementProtocol の設計方針

```python
class ElementProtocol(Protocol):
    """要素の共通インタフェース"""
    ndof: int                           # 要素あたりのDOF数
    nnodes: int                         # 節点数

    def local_stiffness(self, coords, material, section, state) -> ndarray:
        """局所剛性行列"""
        ...

    def internal_force(self, coords, u_elem, material, section, state) -> ndarray:
        """内力ベクトル（非線形解析用）"""
        ...

    def update_state(self, coords, u_elem, material, section, state) -> StateVariable:
        """状態変数の更新"""
        ...

    def mass_matrix(self, coords, material, section) -> ndarray:
        """質量行列（動的解析用）"""
        ...
```

#### ConstitutiveProtocol の設計方針

```python
class ConstitutiveProtocol(Protocol):
    """構成則の共通インタフェース"""

    def tangent(self, strain, state) -> Tuple[ndarray, ndarray]:
        """接線剛性テンソルと応力を返す
        Returns: (D_tangent, stress)
        """
        ...

    def update(self, strain, state) -> StateVariable:
        """履歴変数の更新"""
        ...
```

### 1.3 既存コードの移行

- [x] 現行の `quad4.py`, `tri3.py`, `tri6.py`, `quad4_bbar.py` を `ElementProtocol` に適合
- [x] `elastic.py` を `ConstitutiveProtocol` に適合
- [x] `assembly.py` のアセンブリロジックを Protocol ベースに汎化
- [x] 既存テストが引き続きパスすることを確認（16 passed, 2 skipped）

---

## Phase 2: 空間梁要素

**目的**: Timoshenko梁理論に基づく空間梁要素を実装する。Phase 1のアーキテクチャ上に構築する。

### 2.1 Euler-Bernoulli梁（2D）

基礎として2D Euler-Bernoulli梁を実装。6 DOF/要素（各節点: ux, uy, θz）。

- [ ] 形状関数（Hermite補間）
- [ ] 局所剛性行列 Ke
- [ ] 座標変換行列 T
- [ ] 分布荷重の等価節点力
- [ ] 単体テスト（片持ち梁解析解比較）

### 2.2 Timoshenko梁（2D）

せん断変形を考慮。同じDOF構成。

- [ ] せん断補正係数 κ の導入
- [ ] せん断ロッキング対策（選択的低減積分 or 仮定ひずみ場）
- [ ] 解析解との比較（太い梁問題）

### 2.3 Timoshenko梁（3D空間）

12 DOF/要素（各節点: ux, uy, uz, θx, θy, θz）。

- [ ] 3D座標変換行列（局所→全体）
- [ ] ねじり剛性 GJ
- [ ] 二軸曲げ（EIy, EIz）
- [ ] せん断変形（κy, κz）
- [ ] ワーピング（オプション、薄肉断面用）
- [ ] テスト：3D片持ち梁、ねじり問題

### 2.4 断面モデル

```python
class BeamSection:
    """梁の断面特性"""
    A: float          # 断面積
    Iy: float         # y軸まわり断面二次モーメント
    Iz: float         # z軸まわり断面二次モーメント
    J: float          # ねじり定数（St.Venant）
    ky: float         # y方向せん断補正係数
    kz: float         # z方向せん断補正係数
```

- [ ] 基本断面（矩形、円形、円管）
- [ ] 一般断面（任意形状、メッシュベース数値積分）
- [ ] ファイバーモデル断面（材料非線形用、Phase 4で拡張）

---

## Phase 3: 幾何学的非線形

**目的**: 大変形・大回転に対応する。梁の大たわみ・座屈・スナップスルーを解ける。

### 3.1 非線形ソルバーフレームワーク

- [ ] Newton-Raphson法
  - 残差ベクトル R = f_ext - f_int
  - 接線剛性行列 K_T
  - 収束判定（力・変位・エネルギーノルム）
- [ ] 荷重増分法（Load stepping）
- [ ] 弧長法（Arc-length / Riks法）：スナップスルー・座屈追跡用
- [ ] ラインサーチ（収束加速）

### 3.2 共回転（Corotational）定式化

空間梁の大回転を扱う主要手法。

- [ ] 要素ごとのローカルフレーム追従
- [ ] 剛体回転の分離と変形成分の抽出
- [ ] 接線剛性の幾何学的剛性項 Kg
- [ ] 回転のパラメタ化（四元数 or 回転ベクトル）
- [ ] テスト：大たわみ片持ち梁（Lee's frame等の標準ベンチマーク）

### 3.3 Updated Lagrangian（オプション）

- [ ] 参照配置の更新
- [ ] Green-Lagrangeひずみ
- [ ] 第二Piola-Kirchhoffストレス
- [ ] 共回転定式化との比較検証

---

## Phase 4: 材料非線形

**目的**: 弾塑性、粘弾性、異方性を梁要素に組み込む。

### 4.1 1次元弾塑性

梁の軸方向に対する1D弾塑性構成則から着手。

- [ ] 1D等方硬化モデル（線形硬化）
- [ ] 1D移動硬化モデル（Armstrong-Frederick）
- [ ] Return mapping アルゴリズム
- [ ] 接線剛性の一貫線形化（consistent tangent）
- [ ] 単軸引張・圧縮テスト

### 4.2 ファイバーモデル

断面をファイバー（微小断面要素）に分割し、各ファイバーに1D構成則を適用。

- [ ] ファイバー断面の離散化
- [ ] 断面力 ← ファイバーの応力積分
- [ ] 断面接線剛性 ← ファイバーの接線剛性積分
- [ ] テスト：弾塑性片持ち梁の荷重変位曲線

### 4.3 構造減衰

- [ ] Rayleigh減衰（C = αM + βK）
- [ ] ヒステリシス減衰（履歴変数ベース）
- [ ] 構成則レベルの粘性項

### 4.4 粘弾性

- [ ] 一般化Maxwell（Prony級数）モデル
- [ ] 再帰的更新アルゴリズム（計算効率）
- [ ] 緩和試験・クリープ試験シミュレーション

### 4.5 異方性

- [ ] 異方性弾性（断面方向依存の剛性）
- [ ] 梁の曲げ-ねじり連成（偏心、非対称断面）

---

## Phase 5: 動的解析

**目的**: 揺動試験等の過渡応答解析を可能にする。

### 5.1 時間積分スキーマ

- [ ] Newmark-β法（暗黙的）
- [ ] α法（HHT法、数値減衰付き）
- [ ] 陽解法（Central Difference、オプション）

### 5.2 質量行列

- [ ] 集中質量行列（lumped mass）
- [ ] 整合質量行列（consistent mass）
- [ ] 梁要素の質量行列実装

### 5.3 減衰行列

- [ ] Rayleigh減衰（Phase 4.3と連携）
- [ ] モーダル減衰

---

## Phase 6: NNサロゲートモデル対応

**目的**: 構成則や物理スキーマの一部をニューラルネットワークで代替可能にする。

### 6.1 ConstitutiveProtocolのNN実装

- [ ] PyTorchバックエンドの `NNConstitutive` クラス
- [ ] 入力: ひずみ（＋履歴変数）→ 出力: 応力（＋接線剛性）
- [ ] 自動微分による接線剛性の算出
- [ ] 訓練データ生成インタフェース（従来構成則からの合成データ）

### 6.2 Physics-Informed制約

- [ ] 熱力学的整合性の制約（散逸不等式）
- [ ] 対称性拘束（接線剛性の主対称性）
- [ ] 物理量の正値性保証

### 6.3 ハイブリッドモデル

- [ ] 既知の物理モデル＋残差補正NN
- [ ] スケール分離（マクロモデル＋ミクロ補正）

---

## Phase 7: モデルレジストリとパラメータフィッティング

**目的**: 実験データから最適なモデルとパラメータを同定する。

### 7.1 モデルレジストリ

- [ ] モデル定義のシリアライゼーション（JSON/YAML）
- [ ] モデルカタログ（梁モデル種類 × 構成則 × 断面）
- [ ] バージョン管理と再現性保証

### 7.2 パラメータフィッティング

- [ ] 目的関数定義（荷重-変位曲線の残差）
- [ ] 最適化バックエンド（scipy.optimize, Optuna等）
- [ ] 感度解析（パラメータの影響度）
- [ ] ベイズ推定（パラメータの不確実性定量化、オプション）

### 7.3 実験データインタフェース

- [ ] 試験データフォーマット定義
- [ ] 前処理（フィルタリング、リサンプリング）
- [ ] 複数試験条件の同時フィッティング

---

## Phase 8: 応用展開（将来計画）

ロードマップの主軸完了後に検討する領域。

### 8.1 接触・多体連成

- 電線束内の芯線間接触
- 加工ロールと被加工材の接触境界条件

### 8.2 ミクロ-マクロ連成

- 微小クラックの変形追跡
- 計算ホモジナイゼーション（FE²）

### 8.3 連続体要素の拡張

- 既存2D要素の平面応力対応
- 3D固体要素（Hex8, Tet4, Tet10）
- シェル要素

---

## 優先度と依存関係

```
Phase 1 (アーキテクチャ)
    ↓
Phase 2 (梁要素)
    ↓
Phase 3 (幾何学非線形) ─────────────┐
    ↓                                │
Phase 4 (材料非線形) ←───────────────┤
    ↓                                │
Phase 5 (動的解析) ←─────────────────┘
    ↓
Phase 6 (NNサロゲート) ← Phase 4の構成則設計に依存
    ↓
Phase 7 (レジストリ/フィッティング) ← Phase 2-6の全要素が必要
    ↓
Phase 8 (応用展開) ← 必要に応じて
```

**クリティカルパス**: Phase 1 → 2 → 3 → 4

**並行開発可能**: Phase 5は Phase 3完了後に並行可、Phase 6は Phase 4の構成則設計と並行可

---

## 設計原則

1. **モジュール合成可能性**: 要素・構成則・ソルバー・積分スキーマを自由に組み合わせ可能
2. **Protocol/ABCベース**: 具象クラスへの依存を避け、インタフェースに依存
3. **状態変数の明示管理**: 履歴変数は `StateVariable` で明示的に管理、暗黙のグローバル状態を排除
4. **テスト駆動**: 各要素・構成則は解析解またはリファレンスソルバーとの比較テスト必須
5. **NN互換設計**: 構成則インタフェースは最初からNN代替を考慮した入出力設計
6. **段階的拡張**: 既存のテストを破壊しないよう、後方互換性を保ちながら拡張

---

## 参考文献（予定）

- Crisfield, M.A. "Non-linear Finite Element Analysis of Solids and Structures" Vol. 1 & 2
- Bathe, K.J. "Finite Element Procedures"
- de Souza Neto et al. "Computational Methods for Plasticity"
- Simo, J.C. & Hughes, T.J.R. "Computational Inelasticity"
- Felippa, C.A. "A unified formulation of small-strain corotational finite elements"
