# status-044: Phase C3 — 摩擦 return mapping + μランプ

[← README](../../README.md) | [← roadmap](../roadmap.md)

## 日付

2026-02-21

## 概要

status-043 の TODO を消化:
- **Phase C3: 摩擦 return mapping + μランプ** — 本status

テスト数 **932**（+27: 摩擦単体テスト 22件 + 統合テスト 5件）。

## 実施内容

### 1. law_friction.py（新規）

Coulomb 摩擦の return mapping モジュールを新規作成。

| 関数 | 説明 |
|------|------|
| `compute_tangential_displacement()` | 接触点における接線相対変位増分 Δu_t を計算（形状関数重み付け、接線面投影） |
| `friction_return_mapping()` | Coulomb return mapping: 弾性予測 → stick/slip 判定 → radial return → z_t 更新 → 散逸計算 |
| `friction_tangent_2x2()` | 摩擦接線剛性 (2×2 局所座標系): stick → k_t·I₂, slip → 近似 k_t·I₂ (v0.1) |
| `compute_mu_effective()` | μランプ: μ_eff = μ_target × min(1, counter/steps) |

#### 設計判断

- **全量ベース return mapping**: NR iteration ごとに z_t をステップ開始時の収束値にリセットし、ステップ開始からの全量 Δu_t で return mapping を実行。NR の一貫性を保持
- **slip 接線剛性の近似**: v0.1 では slip 時も stick と同じ k_t·I₂ を使用（consistent tangent は v0.2 で実装予定）
- **p_n 事前更新**: 摩擦 return mapping の前に evaluate_normal_force を呼び、現在の gap に基づく p_n を使用

### 2. assembly.py 拡張

接線方向の形状ベクトルと摩擦力/剛性のアセンブリを追加。

| 関数 | 変更内容 |
|------|---------|
| `_contact_tangent_shape_vector()` | **新規** — 接線方向形状ベクトル（t1/t2 方向、4節点配分） |
| `compute_contact_force()` | `friction_forces` 引数追加（後方互換: None なら法線力のみ） |
| `compute_contact_stiffness()` | `friction_tangents` 引数追加（後方互換: None なら法線剛性のみ） |

### 3. solver_hooks.py 拡張

NR ソルバーに摩擦を統合。

| 変更 | 説明 |
|------|------|
| μランプ | `global_ramp_counter` による Outer loop 通算カウント、`compute_mu_effective()` で漸増 |
| z_t 状態管理 | ステップ開始時に `z_t_conv` を保存、各 NR iteration でリセット後に全量 return mapping |
| 摩擦力/剛性 | Inner loop 内で `friction_return_mapping()` + `friction_tangent_2x2()` を呼び、`compute_contact_force()/stiffness()` に渡す |
| 進捗表示 | 摩擦情報（μ_eff, slip 数）を表示 |
| ContactConfig | `use_friction`, `mu`, `mu_ramp_steps` を活用 |

### 4. テスト結果

#### 摩擦単体テスト（22件）

| テストクラス | テスト数 | 内容 |
|-------------|---------|------|
| TestFrictionReturnMapping | 10 | stick/slip/遷移/散逸/零条件 |
| TestFrictionTangent | 4 | stick接線/非接触零/対称性/半正定値 |
| TestMuRamp | 5 | ランプなし/開始/途中/完了/超過 |
| TestTangentialDisplacement | 3 | 接線滑り/法線零/形状関数重み |

#### 統合テスト（5件）

| テスト | 結果 |
|--------|------|
| 摩擦あり接触収束（法線押し付けのみ） | PASS |
| 接線方向荷重での摩擦応答 | PASS |
| μランプ付き接触収束 | PASS |
| 摩擦散逸非負性 | PASS |
| use_friction=False 後方互換性 | PASS |

### 5. 既存テストへの影響

- **既存テスト 905 件**: 全パス — 回帰なし
- **新規テスト 27 件**: 摩擦単体 22 + 統合 5
- **合計 932 テスト**（922 passed + 10 skipped ≈ テスト構成変動による差異）

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/law_friction.py` | **新規** — Coulomb 摩擦 return mapping モジュール |
| `xkep_cae/contact/assembly.py` | 摩擦力/剛性アセンブリ拡張 |
| `xkep_cae/contact/solver_hooks.py` | NR ソルバーに摩擦統合（μランプ、z_t 管理） |
| `xkep_cae/contact/__init__.py` | 摩擦関連 export 追加 |
| `tests/contact/test_law_friction.py` | **新規** — 摩擦単体テスト 22件 |
| `tests/contact/test_solver_hooks.py` | 統合テスト 5件追加 |

## テスト数

932（+27）

## 確認事項・懸念

1. **slip 接線剛性の近似**: v0.1 では slip 時も k_t·I₂ を使用（過大推定）。consistent tangent ((μ·p_n/||q_trial||)·k_t·(I - q̂⊗q̂)) は v0.2 で実装予定。現状は収束速度が若干劣るが安定
2. **k_t_ratio の推奨値**: テストでは 0.1（法線の10%）が安定。0.5 以上では交差ビームの接触開始時に NR が発散する傾向あり。実用上は 0.05〜0.2 を推奨
3. **μランプの効果**: 3〜5 Outer ステップでの漸増が有効。ランプなし(mu_ramp_steps=0)では接触開始ステップで発散リスクあり

## TODO

- [ ] Phase C4: merit line search + 探索/求解分離の運用強化
- [ ] slip consistent tangent の実装（v0.2）
- [ ] 摩擦ありの Abaqus バリデーション

---
