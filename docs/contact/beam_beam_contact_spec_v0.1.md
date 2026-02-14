# 梁–梁（素線–素線）接触モジュール 設計仕様書 v0.1

[← README](../../README.md) | [ロードマップ](../roadmap.md)

## 0. 目的と非目的

### 目的
- Cosserat rod（一次SRI要素）をベースに、接触支配系でも破綻しにくい梁–梁接触を実装する。
- 収束性を最優先し、**AL + Active-set + return mapping** を中核に据える。
- 最近接点ジャンプによるNewton破綻を避けるため、**探索（Outer）と解く（Inner）を分離**する。

### 非目的（v0.1）
- mortar法の完全実装
- 高次要素対応
- 自動微分による接線生成

## 1. 用語
- **セグメント**: 一次梁要素の軸線上の線分（節点 i–j）
- **最近接**: 2線分間の最短距離点（パラメタ s, t）
- **ギャップ**: 法線方向距離 g（g >= 0: 離間, g < 0: 貫通）
- **AL**: Augmented Lagrangian（拡張ラグランジュ）
- **Active-set**: 接触有効集合
- **Return mapping**: Coulomb摩擦の投影更新
- **Outer loop**: 接触探索更新（geometry update）
- **Inner loop**: Newton反復（最近接固定）

## 2. 全体アーキテクチャ

### 2.1 モジュール構成（プロジェクトスキーマ準拠）
- `xkep_cae/contact/geometry.py`
  - segment-to-segment 最近接、距離、法線・接線フレーム生成
- `xkep_cae/contact/law_normal.py`
  - 法線接触（AL）、ギャップC¹正則化（オプション）
- `xkep_cae/contact/law_friction.py`
  - 接線摩擦（return mapping、stick/slip）
- `xkep_cae/contact/assembly.py`
  - 接触内力 `f_c` と接触接線 `K_c` の組み込み
- `xkep_cae/contact/solver_hooks.py`
  - `newton_raphson` への接触寄与注入インターフェース
- `tests/contact/`
  - 幾何・法線・摩擦・統合のテスト

### 2.2 データフロー（1荷重増分）
1. Outer: 近接探索（candidate pairs）
2. 各ペアで最近接 `(s,t)` と局所フレームを確定し、Active-set初期化
3. Inner Newton: `(s,t)` 固定で `f_c, K_c` を組み込み、収束判定
4. Outer: 必要時のみ最近接更新して再度Inner

## 3. 接触幾何（必須）

### 3.1 入力
- セグメントA端点 `xA0, xA1`
- セグメントB端点 `xB0, xB1`

### 3.2 出力
- `s ∈ [0,1]`, `t ∈ [0,1]`
- 最近接点 `pA(s)`, `pB(t)`
- 最近接ベクトル `d = pB - pA`
- 距離 `dist = ||d||`
- 法線 `n = d / ||d||`（`dist > eps`）
- 接線基底 `t1, t2`（フレーム履歴を使って連続更新）

### 3.3 安定化ルール
- 平行/準平行ケースは分岐処理で安定化（clamp + branch）
- `dist` 極小で法線不定の場合は前ステップ法線を優先

## 4. 法線接触（AL）

### 4.1 状態量（接触点ごと）
- `lambda_n >= 0`
- `k_pen`
- `g`
- `active`

### 4.2 反力評価
- 候補条件: `g <= 0`
- `p_n = max(0, lambda_n + k_pen * (-g))`
- `f_n = p_n * n`

### 4.3 乗数更新
- `lambda_n <- p_n`（必要時ダンピング）
- `k_pen` は上限クリップ付きで更新可

### 4.4 C¹正則化（オプション）
- `max(0,x)` を滑らか近似へ切替可能（デフォルトOFF）

## 5. 摩擦（Coulomb return mapping）

### 5.1 状態量
- `z_t`（接線履歴）
- `stick`（状態フラグ）
- `t1, t2`（接線フレーム）

### 5.2 入力
- 接線相対変位増分 `Δu_t`
- 法線反力 `p_n`
- 摩擦係数 `μ`
- 接線ペナルティ `k_t`

### 5.3 更新則
1. `q_trial = q_old + k_t * Δu_t`
2. クーロン条件 `||q|| <= μ p_n`
3. 判定
   - stick: `||q_trial|| <= μ p_n` → `q_new = q_trial`
   - slip: それ以外 → `q_new = μ p_n * q_trial / ||q_trial||`
4. 散逸監視: `D_inc = q_new · Δu_t`

### 5.4 μランプ
- 反復荒れ対策として `μ: 0 → μ_target` をOuterごとに漸増

## 6. Active-set と Outer/Inner 分離（必須）

### 6.1 Active-set更新
- 候補: `g <= g_act`
- ヒステリシス:
  - activate: `g <= g_on`
  - deactivate: `g >= g_off`（`g_off > g_on`）

### 6.2 探索と解くの分離
- Inner中は最近接 `(s,t)` を固定
- 収束後のみOuterで更新
- 推奨: `N_outer_max = 3~5`、`|Δs|, |Δt|` が閾値以下で停止

## 7. アセンブリ要件

### 7.1 v0.1最小要件
- 接触内力 `f_c` は必須
- 接線 `K_c` は支配項中心の近似で開始
  - penalty/AL主項を最優先
  - 幾何微分（`n`変化等）はv0.2で拡張

### 7.2 優先順位
1. 法線 penalty/AL 主項
2. stick時接線剛性
3. slip時は近似接線 + merit line search

## 8. 収束戦略（merit + line search）

### 8.1 推奨merit（v0.1簡略版）
- `Φ = ||R|| + α Σ max(0,-g)^2 + β Σ D_inc_pos`
- `Φ` が減少する step length を採用

### 8.2 目的
- stick/slip切替や接触状態反転時のNewton暴走を抑制

## 9. パラメタ初期値案
- 近接探索: `r_search`（断面半径の数倍）、broadphaseはAABB格子から開始
- AL: `k_pen` 自動初期化（`EA/L`スケール）、上限クリップ、`lambda_n=0`
- 摩擦: `k_t = γ k_pen`（`γ=0.1~1.0`）、`μ`ランプ有効
- Active-set: `g_on=0`, `g_off=0.1*g_tol`
- 反復: `N_outer_max=5`, `N_newton_max=30`

## 10. テスト計画（必須）

### 10.1 幾何
- 解析解が分かる2線分最近接
- 平行、交差近傍、端点-端点

### 10.2 法線
- 1自由度縮約系で貫通→反力の単調性
- AL更新で貫通が減ること

### 10.3 摩擦
- stick→slip遷移の再現
- `D_inc >= 0`（数値誤差許容）

### 10.4 統合
- 梁–梁押し付け（摩擦なし）収束
- 摩擦ありで過大チャタリング抑制
- 「探索固定」vs「毎Newton更新」の比較で前者の優位性確認

## 11. 実装ロードマップ（接触モジュール）

### Phase C0（1–2日）
- データ構造（`ContactPair`, `ContactState`）
- モジュール分割とAPI固定

**完了条件**: Contact APIを `solver.py` から呼び出せる。

### Phase C1（2–4日）
- segment-to-segment最近接（robust）
- フレーム更新（法線履歴必須）
- broadphase（AABB格子）

**完了条件**: 多数セグメントで候補抽出が破綻しない。

### Phase C2（2–4日）
- 法線AL更新（`g`, `p_n`, `lambda_n`）
- Active-setヒステリシス
- 内力組み込み（`K_c`主項）

**完了条件**: 摩擦なし梁–梁が安定収束する。

### Phase C3（3–6日）
- 摩擦return mapping
- `μ`ランプ
- 散逸監視

**完了条件**: 滑り発生時の発散が顕著に減る。

### Phase C4（3–6日）
- Outer/Inner分離
- 最近接固定Newton
- merit line search

**完了条件**: 最近接点ジャンプで解が崩れない。

### Phase C5（v0.2）
- 幾何微分込みの一貫接線
- semi-smooth Newton / PDAS検討
- 断面フレーム連続輸送の強化

## 12. Claude Code向け実装指示（最重要）
- Newton反復中は最近接 `(s,t)` を更新しない。
- 法線は penalty単独でなく AL を使う。
- 摩擦は return mapping（投影）で更新する。
- Active-setはヒステリシスを必ず入れる。
- 接線は主項優先、収束難時は merit line search を優先投入。
- テストは **幾何 → 法線 → 摩擦 → 統合** の順で先行作成する。

---

## 引き継ぎメモ（Codex/Claude 2交代運用）
- v0.1は「動く・落ちにくい」収束重視の段階実装を優先し、数理的な完全一貫性（厳密接線）はv0.2以降で強化する。
- 実装時は `docs/status/status-xxx.md` に、Outer/Innerの反復挙動・active-set反転回数・line searchの発動頻度まで記録する。
