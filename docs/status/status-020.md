# status-020: Phase 3 幾何学的非線形 完了

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-019](./status-019.md)

**日付**: 2026-02-14
**作業者**: Claude Code
**ブランチ**: `master`

---

## 実施内容

Phase 3（幾何学的非線形）の主要実装を全て完了した。
SO(3) ヤコビアン → 非線形 Cosserat rod → Euler elastica 検証 → 弧長法の順で実装。
407テスト全パス（2 skipped）、リグレッションなし。

### Step 1: SO(3) 右ヤコビアン

**ファイル**: `xkep_cae/math/quaternion.py`

- `so3_right_jacobian(rotvec)` — J_r(θ) = I - c₁·S + c₂·S²
- `so3_right_jacobian_inverse(rotvec)` — J_r⁻¹(θ)
- 小角度 |θ| < 1e-6 でテイラー展開に切り替え

**テスト**: `tests/test_quaternion.py` に `TestSO3Jacobian` 追加（6テスト）
- J_r(0)=I, 逆行列往復, テイラー分岐一致, 数値微分照合, 既知値(90°z軸)

### Step 2: 非線形 Cosserat rod

**ファイル**: `xkep_cae/elements/beam_cosserat.py`

- `_nonlinear_strains()` — 力歪み Γ = R(θ)ᵀR₀ᵀr' - e₁, モーメント歪み κ = J_r(θ)·θ'
- `_nonlinear_B_matrix()` — 非線形 B 行列 (6×12)
- `_nonlinear_internal_force()` — f_int = L₀·B_nlᵀ·C·[Γ; κ-κ₀]
- `_nonlinear_tangent_stiffness()` — 中心差分ヤコビアンによる接線剛性（対称化）
- `CosseratRod(nonlinear=True)` で非線形モードへディスパッチ

**テスト**: `tests/test_nonlinear_cosserat.py`（13テスト）
- ゼロ変位→ゼロ力, 小変位で線形版と一致, K_T 対称性, **有限差分接線検証（最重要）**

### Step 3: Euler elastica ベンチマーク

**ファイル**: `tests/test_euler_elastica.py`（9テスト）

#### 端モーメント試験（5ケース）
片持ち梁に先端モーメント M → 一様曲率で円弧変形
- θ = π/4, π/2, π, 3π/2, 2π（完全円）でパラメトリックテスト
- 解析解: x = (EI/M)sin(ML/EI), y = (EI/M)(1-cos(ML/EI))
- 全ケース先端位置誤差 < 3%L（20〜40要素）

#### 先端荷重試験（4ケース）
片持ち梁に先端鉛直荷重 P → Mattiasson 型の大変形
- PL²/EI = 1, 2, 5, 10 での先端変位を検証
- 参照値は elastica ODE（EI·θ'' = -P·cos θ）の shooting method 解
  （scipy brentq + solve_ivp, rtol=1e-12 で独自検証）
- 全ケース相対誤差 < 5%

### Step 4: 弧長法（Crisfield）

**ファイル**: `xkep_cae/solver.py`

- `ArcLengthResult` データクラス — u, lam, converged, n_steps, total_iterations, load_history, displacement_history
- `_apply_bc()` — 境界条件適用ヘルパー
- `arc_length()` — 円筒弧長法（Crisfield, 1981）
  - 弧長拘束 ||Δu||² = Δl²
  - 接線予測子 + 円筒修正子（二次方程式）
  - 荷重方向符号の自動追跡
  - ステップサイズ適応（カットバック + 拡大/縮小）
  - delta_l 自動推定

**テスト**: `tests/test_arc_length.py`（5テスト）
- `TestArcLengthResult` — データクラス生成
- `TestArcLengthLinear` — 軸引張/曲げで NR と弧長法の結果が一致
- `TestSnapThroughSpring` — 1-DOF 非線形スプリング（f_int = k₁u + k₃u³）
  - リミットポイント検出（解析値との相対誤差 < 15%）
  - リミット前の荷重-変位曲線が解析解と一致（< 1%）

### Step 5: ロードマップ更新

- `docs/roadmap.md` — Phase 3 チェックリスト完了、テスト数 407、「現在地」更新

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/math/quaternion.py` | `so3_right_jacobian()`, `so3_right_jacobian_inverse()` 追加 |
| `xkep_cae/elements/beam_cosserat.py` | 非線形歪み・内力・接線剛性関数追加、`CosseratRod(nonlinear=True)` |
| `xkep_cae/solver.py` | `ArcLengthResult`, `_apply_bc()`, `arc_length()` 追加 |
| `tests/test_quaternion.py` | `TestSO3Jacobian` クラス追加 |
| `tests/test_nonlinear_cosserat.py` | **新規** — 非線形要素の単体テスト（13件） |
| `tests/test_euler_elastica.py` | **新規** — Euler elastica ベンチマーク（9件） |
| `tests/test_arc_length.py` | **新規** — 弧長法テスト（5件） |
| `docs/roadmap.md` | Phase 3 チェックリスト更新 |
| `README.md` | 現在の状態・テスト数・ステータスリンク更新 |
| `docs/status/status-020.md` | **新規** — 本ステータス |

---

## テスト結果

```
407 passed, 2 skipped (192.70s)
```

新規追加テスト内訳:
- `test_quaternion.py` — SO(3) ヤコビアン 6件
- `test_nonlinear_cosserat.py` — 非線形要素 13件
- `test_euler_elastica.py` — Euler elastica 9件
- `test_arc_length.py` — 弧長法 5件
- **合計 33件追加**（374 → 407）

---

## 技術的知見（デバッグ記録）

### Euler elastica 参照値の誤り

初期実装で文献から引用した「Mattiasson (1981)」参照値が α=5, 10 で誤っていた。
FEM の収束確認（30→50→80要素でメッシュ収束）とelastica ODE の独自解法（shooting method）で
FEM 解が正しいことを確認し、参照値を修正。

### 弧長法のカットバック

初期実装では二次方程式の判別式が負になった場合に δλ=0 とするフォールバックを使用したが、
弧長拘束が崩れて発散する問題が発生。ステップサイズ半減 + 再開始（カットバック）方式に変更して解決。

---

## Phase 3 完了状態サマリ

| 項目 | 状態 |
|------|------|
| Newton-Raphson + 荷重増分 | 完了 |
| 弧長法（Crisfield） | 完了 |
| SO(3) 右ヤコビアン | 完了 |
| 非線形歪み（Γ, κ） | 完了 |
| 非線形内力 f_int | 完了 |
| 非線形接線剛性 K_T | 完了（中心差分） |
| CosseratRod(nonlinear=True) | 完了 |
| Euler elastica 検証 | 完了 |
| **ラインサーチ** | **未実装（オプション）** |
| **Lee's frame 等の追加ベンチマーク** | **未実装（オプション）** |

---

## 次作業（TODO）

### 優先度A（次フェーズ着手時）
- [ ] Phase 4 または Phase 5 の着手判断（ユーザー指示待ち）
- [ ] Phase 4.1〜4.5 の実装順序・完了条件の具体化
- [ ] Phase 5 の最小成立ライン（Newmark + lumped mass）の明文化

### 優先度B（Phase 3 残オプション）
- [ ] ラインサーチ追加（NR 収束加速、必要に応じて）
- [ ] Lee's frame / Williams toggle frame 等の標準ベンチマーク

### 優先度C（ドキュメント）
- [ ] 非線形 Cosserat rod の設計詳細を `docs/cosserat-design.md` に追記

---

## 確認事項・設計上の懸念

- 非線形接線剛性は中心差分ヤコビアン（数値微分）で実装。解析的接線への移行は精度/速度の観点で将来検討。
- 弧長法の delta_l 自動推定は初期接線剛性ベース。問題によっては手動調整が必要。
- Phase 3 の「完了」は主要機能の実装完了を意味し、ラインサーチ・追加ベンチマークはオプション扱い。

---

## 引き継ぎメモ（Codex/Claude 2交代運用）

- **Phase 3 は実質完了**。残りのラインサーチ・Lee's frame はオプション。
- 次に進むべきは Phase 4（材料非線形）, Phase 5（動的解析）, Phase C（接触）のいずれか。ユーザー判断を仰ぐこと。
- `CosseratRod(nonlinear=True)` + `newton_raphson()` or `arc_length()` で大変形問題を解ける状態。
- 弧長法のテストは 1-DOF スプリング問題。梁構造でのスナップスルーテストは追加ベンチマーク（優先度B）。
