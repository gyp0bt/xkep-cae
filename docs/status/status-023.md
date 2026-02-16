# status-023: Phase 4.2 ファイバーモデル（曲げの塑性化）完了

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-022](./status-022.md)

**日付**: 2026-02-16
**作業者**: Claude Code
**ブランチ**: `claude/execute-status-todos-6MUvC`

---

## 概要

Phase 4.2（ファイバーモデル）を完了。断面をファイバー（微小断面要素）に分割し、
各ファイバーに既存の 1D 弾塑性構成則（Plasticity1D）を適用することで、
曲げの塑性化を表現する。Phase 4.1 の軸方向のみの弾塑性を、軸+曲げに拡張した。

既存の 435 テストを維持しつつ 36 テスト追加、合計 471 テストパス（2 skipped）。

---

## 実施内容

### Step 1: FiberSection クラス（新規）

**ファイル**: `xkep_cae/sections/fiber.py`

- `FiberSection` — ファイバーモデル断面クラス
  - フィールド: `y`, `z` (座標配列), `areas` (面積配列), `J`, `shape`
  - プロパティ: `A`, `Iy`, `Iz`, `n_fibers`
  - `cowper_kappa_y/z()` — Cowper せん断補正係数
- ファクトリメソッド:
  - `rectangle(b, h, ny, nz)` — 矩形断面（ny × nz ファイバー）
  - `circle(d, nr, nt)` — 円形断面（同心円リング × 等角度）
  - `pipe(d_outer, d_inner, nr, nt)` — パイプ（中空円形）
- `xkep_cae/sections/__init__.py` にエクスポート追加

### Step 2: CosseratFiberPlasticState（新規）

**ファイル**: `xkep_cae/core/state.py`

- `CosseratFiberPlasticState` — ファイバーモデル弾塑性状態
  - `fiber_states: list[PlasticState1D]` — 各ファイバーの独立な塑性状態
  - `create(n_fibers)` — 指定数のファイバーを持つ初期状態を生成
  - `copy()` — 深いコピー
- `xkep_cae/core/__init__.py` にエクスポート追加

### Step 3: ファイバー積分による断面力・接線剛性

**ファイル**: `xkep_cae/elements/beam_cosserat.py`（関数追加、既存変更なし）

#### `_compute_generalized_stress_fiber()`

各ファイバーのひずみ:
```
epsilon_i = Gamma_1 + kappa_2 * z_i - kappa_3 * y_i
```

断面力（ファイバー積分）:
```
N  = Sum(sigma_i * A_i)
My = Sum(sigma_i * z_i * A_i)
Mz = -Sum(sigma_i * y_i * A_i)
```

接線剛性（3×3 サブ行列、indices [0,4,5]×[0,4,5]）:
```
C[0,0] = Sum(D_i * A_i)           # EA_eff
C[0,4] = Sum(D_i * z_i * A_i)     # N-My 連成
C[0,5] = -Sum(D_i * y_i * A_i)    # N-Mz 連成
C[4,4] = Sum(D_i * z_i^2 * A_i)   # EIy_eff
C[4,5] = -Sum(D_i * y_i*z_i * A_i)
C[5,5] = Sum(D_i * y_i^2 * A_i)   # EIz_eff
```

せん断 (Γ₂, Γ₃) とねじり (κ₁) は弾性のまま。

#### `assemble_cosserat_beam_fiber()`

- uniform 積分 + SRI 両方に対応
- Phase 4.1 の `assemble_cosserat_beam_plastic()` と同じインタフェースパターン
- states サイズ: uniform → `n_elems * n_gauss`, SRI → `n_elems * 2`

### Step 4: テスト

**ファイル**: `tests/test_fiber_section.py`（新規、36テスト）

#### FiberSection 単体テスト（16テスト）
- 矩形断面: A, Iy, Iz, ファイバー数, 重心位置
- 円形断面: A, Iy, 重心位置
- パイプ断面: A
- Iy の分割数収束
- 入力検証（空配列、負面積、負J、長さ不一致）
- Cowper 補正係数

#### ファイバー応力積分テスト（10テスト）
- 弾性軸力 N = EA * Gamma_1
- 弾性曲げ My = EIy * kappa_2
- 弾性曲げ Mz = EIz * kappa_3
- 弾性接線の対角成分
- 弾性接線の非対角ゼロ（対称断面）
- 接線の対称性
- 接線の有限差分検証（3成分×3成分）
- 塑性域で EA が低下
- 部分塑性の連成
- 入力状態の不変性

#### アセンブリレベルテスト（4テスト）
- uniform 弾性一致: 内力
- uniform 弾性一致: 剛性
- SRI 弾性一致: 内力
- 全体接線の有限差分検証

#### 弾塑性曲げテスト（6テスト）
- 弾性限界モーメントの検証
- 全塑性モーメント M_p = sigma_y * b*h^2/4
- 形状係数 M_p/M_y = 1.5（矩形断面）
- 片持ち梁の曲げ NR 解析
- NR 二次収束（consistent tangent）
- モーメント-曲率曲線（弾性→弾塑性→全塑性遷移）
- 軸引張の bilinear 解析解一致

### Step 5: 検証図（2枚新規）

**ファイル**: `tests/generate_verification_plots.py`（関数追加）

1. `fiber_moment_curvature.png` — モーメント-曲率曲線（完全弾塑性 vs 等方硬化 vs 弾性）
2. `fiber_cantilever_moment.png` — 片持ち梁先端モーメント荷重-回転曲線

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/sections/fiber.py` | **新規**: FiberSection クラス |
| `xkep_cae/sections/__init__.py` | エクスポート追加 |
| `xkep_cae/core/state.py` | CosseratFiberPlasticState 追加 |
| `xkep_cae/core/__init__.py` | エクスポート追加 |
| `xkep_cae/elements/beam_cosserat.py` | `_compute_generalized_stress_fiber()`, `assemble_cosserat_beam_fiber()` 追加 |
| `tests/test_fiber_section.py` | **新規**: 36テスト |
| `tests/generate_verification_plots.py` | ファイバーモデル検証図追加（2枚） |
| `docs/verification/*.png` | 検証プロット2枚新規 |
| `docs/roadmap.md` | Phase 4.2 チェックリスト完了、テスト数471に更新 |
| `README.md` | 現在の状態・テスト数・ステータスリンク更新 |
| `docs/status/status-023.md` | **新規**: 本ステータス |

---

## テスト結果

```
471 passed, 2 skipped
```

新規追加テスト内訳:
- `test_fiber_section.py` — 36件
- **合計 36件追加**（435 → 471）

---

## 技術的知見（デバッグ記録）

### ny=1 でのIz=0問題

`FiberSection.rectangle(b, h, ny=1, nz=N)` とすると全ファイバーが `y=0` に配置され、
`Iz = Sum(A_i * y_i^2) = 0` となる。これにより z 軸曲げの剛性がゼロとなり、
全体剛性行列が特異になる。

**対策**: ny >= 2 を使用し、全方向に非ゼロ剛性を確保する。
テストではこの点を明示的に文書化した。

### ファイバーひずみの弾性限界

構成レベルの曲げテストでは、断面最外ファイバーのひずみが降伏ひずみを超えないよう
曲率を設定する必要がある:
```
kappa_elastic_limit = sigma_y / (E * z_max)
```
z_max は最外ファイバーの z 座標（e.g., nz=20, h=20 → z_max=9.5）。

---

## 設計判断

- **既存関数は変更なし**: `assemble_cosserat_beam_plastic()` はそのまま残し、新関数 `assemble_cosserat_beam_fiber()` を追加
- **せん断・ねじりは弾性**: ファイバー積分は N, My, Mz のみ。Vy, Vz, Mx は弾性構成行列を使用
- **接線の連成**: 塑性域では N-My, N-Mz, My-Mz の連成が発生（対称断面・純曲げでは消滅）
- **FiberSection のインタフェース**: BeamSection と同じダックタイピングインタフェース（A, Iy, Iz, J, cowper_kappa_*）を提供

---

## 次作業（TODO）

### 優先度A（Phase 4 続き）
- [ ] Phase 4.3: 構造減衰（ヒステリシス減衰、粘性項）
- [ ] Phase 4.4: 粘弾性（一般化Maxwell）
- [ ] Phase 4.5: 異方性弾性

### 優先度B（Phase 5）
- [ ] Phase 5: 動的解析（Newmark-β、lumped mass）

### 優先度C
- [ ] Phase C: 梁–梁接触

---

## 確認事項・懸念

- ファイバー数が大きくなると計算コストが増加する（各ファイバーで return_mapping を呼ぶため）。大規模問題ではベクトル化を検討する価値がある。
- 3D von Mises 降伏関数（status-021 の TODO Phase 4.3）は roadmap の Phase 4 の構成と若干の齟齬がある。roadmap に合わせ、構造減衰 → 粘弾性 → 異方性の順序とした。

---

## 引き継ぎメモ（Codex/Claude 2交代運用）

- **Phase 4.2 完了**。ファイバーモデルでの弾塑性曲げ解析が使える状態。
- `assemble_cosserat_beam_fiber()` の使用パターンは `tests/test_fiber_section.py` の `TestFiberBendingLoadDisplacement` を参照。
- `FiberSection` は BeamSection と同じインタフェースを提供するので、`CosseratRod(section=fs)` として渡せる。ただし `ny >= 2` でないと Iz=0 になるので注意。
- 検証図は `python tests/generate_verification_plots.py` で再生成可能（12枚、Phase 4.2 含む）。
- 降伏判定のトレランス `1e-10 * sigma_y_n`（Plasticity1D）は Phase 4.1 から変更なし。
