# status-277: NR収束壁の根本原因特定 — evaluate/tangent dm不整合 + NR制御改善

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-31
- **ブランチ**: `claude/improve-7wire-bending-6vnxR`
- **テスト数**: 600 passed, 0 failed
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 調査概要

7本撚線曲げ揺動ベンチマークの frac=0.41 壁（status-276 で特定）の根本原因を追求。
status-260（frac=0.59）との差分を系統的に分析。

---

## 発見事項

### 1. evaluate() と tangent() の dm 不整合

**status-260（frac=0.59）**: evaluate()、tangent() 両方で dm 補正付き Hermite 係数を使用。残差とヤコビアンが整合。

**現HEAD（frac=0.41）**: evaluate() は dm 補正付き、tangent() は dm 補正なし（status-266 で「修正NR法」として導入）。残差とヤコビアンが不整合。

| 設定 | frac | 評価 |
|------|------|------|
| ベースライン（dm eval ON, tangent OFF） | 0.413 | 現状 |
| tangent dm ON | 0.388 | 悪化 |
| tangent dm ON + adj OFF | 0.369 | 悪化 |
| adj OFF のみ | 0.361 | 悪化 |
| _cur_ratio を res_u_norm に戻す | 0.350 | 悪化 |
| NR min restore OFF | 0.413 | 変化なし |

**重要**: 個別のリバートでは回復しない。status-260 のフル復元（NR+接触）でのみ 0.59 に到達。

### 2. 複合的回帰（NR + 接触コードの相互作用）

| 組み合わせ | frac |
|-----------|------|
| status-260 NR + status-260 接触 | **0.59** |
| status-260 NR + 現接触 | 0.37 |
| 現NR + status-260 接触 | 0.35 |
| 現NR + 現接触 | 0.41 |

→ **接触コードとNRソルバー両方の変更が相互作用して回帰**。現コードの接触改善（非局所DOF）とNR改善（min restore等）が個別には有効だが、組み合わせでは status-260 の整合したアプローチに劣る。

### 3. FD 接線診断結果

- **全体系 K@du FD vs 解析**: 相対誤差 13-37%（接触活性化時に最大）
- **方向微分（スカラー）**: 1-2% で整合 — du 方向の投影は正確
- **ベクトル誤差**: K@du ベクトル全体は大幅にずれている
- **原因**: MPC slave DOF の残差がゼロ化されないため、全系指標が悪化

### 4. NR 挙動の根本差異

**status-260**: frac=0.35 で壁 → cutback → 非常に小さい dt → 接触力リラクゼーション (ω=0.5 + tangent scaling) → エネルギー収束で 195 インクリメントかけて frac=0.59 に到達（226秒）

**現HEAD**: frac=0.41 まで進行（非局所DOF改善のおかげ）→ 壁 → cutback 後も残差 ~103% で停滞 → 50反復でも収束せず → 停止

---

## 実装した変更

| 変更 | ファイル | 理由 |
|------|----------|------|
| `nr_min_restore` デフォルト False | `core/data.py`, `_newton_dynamic.py`, `process.py` | 8.7%残差の不正確な状態を次incrに持ち越す問題を防止 |
| `_diverged = True` relax timeout時 | `_newton_dynamic.py` | 積極的dt縮小で小dtチャタリング回復を促進 |
| `contact_tangent_scale = ω` relax時 | `_newton_dynamic.py` | 接触力リラクゼーションと接線剛性の整合性を回復 |
| `du_norm_cap` パイプライン貫通 | `strand_bending_oscillation.py` | NR更新キャップの実験基盤 |

**ベンチマーク結果**: frac=0.40（min restore OFF により微低下だが、次incrの破綻は防止）

---

## 次セッションへの推奨アクション

### 最優先: evaluate/tangent dm 整合性の回復

**問題**: status-266 で tangent() の dm を OFF にした「修正NR法」が、非局所DOF拡張との組み合わせでNR収束を阻害。

**アプローチ案**:
1. **tangent dm ON + 非局所DOF OFF**: status-260 相当に戻す（0.59回復見込み）→ ただし NRソルバーも status-260 に戻す必要あり（複合回帰）
2. **evaluate dm OFF**: 逆のアプローチ—evaluate も dm なしにして一貫させる
3. **dm のヤコビアン寄与を実装**: ∂(dm_corrected_coeffs)/∂u を tangent に追加。最も正しいが実装コスト大

### 高優先: NR 制御の根本改善

- チャタリング帯域での **接触力リラクゼーション戦略の再設計**
  - status-260 の ω blending + tangent scaling が有効だったことが証明された
  - delta_h boost との共存方法を検討
- **活性集合安定化**: NR 反復初期に活性集合を凍結し、収束後に解放するスキーム

### 中期: MPC + 接触の構造的改善

- MPC slave DOF が全系残差に寄与する問題の解消
- 縮退系での接触接線剛性品質の改善

---

## 再現手順

```bash
git checkout claude/improve-7wire-bending-6vnxR
pip install -e .

# 全テスト
python -m pytest xkep_cae/ tests/ -q --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# 撚線ベンチマーク
python -c "
from xkep_cae.numerical_tests.strand_bending_oscillation import *
cfg = StrandBendingOscillationConfig(
    n_strands=7, wire_radius=0.5, pitch_length=100.0,
    n_elements_per_pitch=16, n_pitches=1.0,
    E=130.0e3, nu=0.3, rho=8.96e-9,
    bending_curvature=0.001, n_cycles=1,
    n_increments_per_cycle=40, rho_inf=0.9, mu=0.15,
    max_nr_attempts=50, tol_force=1e-8, max_increments=10000,
    exclude_same_strand=True,
)
r = StrandBendingOscillationProcess().process(cfg)
sr = r.solver_result
print(f'frac={sr.load_history[-1]:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}, {sr.elapsed_seconds:.1f}s')
"
# 期待値: frac≈0.40

# 契約検証
python contracts/validate_process_contracts.py
```

---

## STA2 準拠チェック

- [x] **tee ログ保存**: 全ベンチマーク結果を /tmp/log-bench-*.log に保存
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: frac=0.40を正直に報告
- [x] **ベースライン先行取得**: status-260 worktree で frac=0.5914 確認

---
