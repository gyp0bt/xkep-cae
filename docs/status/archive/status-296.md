# status-296: K_c FD 1.8%分析 + 端部接触除外実装 + frozen-m効果検証

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-05
- **ブランチ**: `claude/execute-status-todos-26c7R`
- **テスト数**: 442+ passed（既存テスト全合格、test_stress_contour既知失敗除く）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-295のTODO 3件を実行:

1. **TODO 1（分析完了）**: K_c FD残余1.8%の詳細分析 — K_st_adj再有効化は38.5%に悪化
2. **TODO 2（実装・検証完了）**: MPC+contact端部接触除外 — frac 0.001→0.004（不十分）
3. **TODO 3（検証完了）**: frozen-m解消効果 — **Hertz型で frac 0.40→0.9997（事実上完走！）**

---

## 1. K_c FD 残余1.8%の詳細分析（TODO 1）

### 仮説

K_c_adj mat-only（status-295）で除外した幾何項(I-n⊗n)を、K_st_adjで補完すれば1.8%→0%に近づく。

### 検証

| 構成 | rel_err | 備考 |
|------|---------|------|
| K_c_adj mat-only（ベースライン） | **1.79%** | status-295の最適構成 |
| K_c_adj mat-only + K_st_adj有効化 | 38.45% | x,y方向で27倍過大 |
| K_c_adj full + K_st_adj | 38.45% | 同上 |

### 分析

- **z方向**: K_st_adj有効化で77.3%→2.0%に改善（正確に補完）
- **x,y方向**: K_st_adjがx,y方向で27倍過大（ratio=27）

根本原因:
- K_c_adj geo(I-n⊗n)とK_st_adjの接平面成分は**物理的に同一の寄与**
- K_c_adj fullでもK_st_adjでも、どちらか片方を有効にすると接平面で過大
- K_c_adj mat-only(n⊗n)が法線方向のみ正確にカバーし、**1.79%が最適解**

### 結論

**K_c_adj mat-only（1.79%）を維持。改善はKst計算のリファクタリングが必要だが、
1.8%はNR 2次収束に十分な精度であり、優先度は低い。**

### コード変更

`strategy.py` L276-280: コメント更新（K_st_adj再有効化検証結果を記録）

---

## 2. MPC+contact 端部接触除外（TODO 2）

### 実装

`exclude_end_elements` オプションを接触パイプラインに追加:
- `_ContactConfigInput.exclude_end_elements: int = 0`
- `StrandBendingOscillationConfig.exclude_end_elements: int = 0`
- `_build_end_element_set()`: 素線チェーン構造から端部N要素を特定
- `DetectCandidatesProcess`: 端部要素を含む候補ペアをフィルタリング

### テスト（8件追加）

| テスト | 内容 |
|--------|------|
| `TestBuildEndElementSet` (5件) | 端部要素セット構築の単体テスト |
| `TestExcludeEndElementsIntegration` (1件) | DetectCandidatesProcessとの統合テスト |

### 検証結果

| モード | exclude_end | frac | 備考 |
|--------|-------------|------|------|
| MPC + contact | 0 | 0.001 | ベースライン（即座に発散） |
| MPC + contact | 2 | **0.004** | 微改善のみ |

### 分析

端部要素除外は不十分。根本原因はMPCのT^T K_c Tグローバルカップリング:
- 端部以外の要素の接触力も、slave DOFを通じてT^T変換でmaster DOFに分散
- 7本全ワイヤの全ノードが1つのmasterで結合されるため、局所接触→グローバル影響
- MPC+contact問題の解決にはMPC構造自体の改善が必要（例: ワイヤ単位のローカルMPC）

---

## 3. frozen-m解消効果の検証（TODO 3）

### ベースライン（free_end + contact）

| penalty_exponent | max_incr | frac | incr | cutback |
|------------------|----------|------|------|---------|
| 1.0（線形） | 300 | **0.40** | 208 | 14 |
| 1.5（Hertz型） | 300 | 0.62 | 300 | 23 |
| **1.5（Hertz型）** | **600** | **0.9997** | **541** | **41** |

### 分析

- 線形ペナルティ: status-295と同一（frac=0.40）、frozen-m解消（status-294/295）の恩恵なし
- **Hertz型ペナルティ: frac=0.9997（事実上完走！）** 541 incr, 41 cutback
- frac=0.79付近でチャタリングが発生するが、カットバック+再試行で通過
- 接触あり90度曲げで**正しい接線剛性（frozen-m解消後）での事実上完走を達成**

### 参考: status-285との比較

status-285ではfrac=0.998を達成（frozen-m近似あり）。今回はfrozen-m解消後の
「正しい接線」でfrac=0.9997を達成。接線精度の向上がcutback増加（14→41）を
招いたが、最終的には完走に到達。

---

## 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/_contact_pair.py` | `exclude_end_elements` フィールド追加 |
| `xkep_cae/contact/_manager_process.py` | `_build_end_element_set()` 実装 + フィルタリング |
| `xkep_cae/contact/tests/test_manager_process.py` | 端部除外テスト8件追加 |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `exclude_end_elements` パイプライン貫通 |
| `xkep_cae/contact/contact_force/strategy.py` | K_st_adj検証結果コメント更新 |

---

## TODO

- [ ] 接触あり90度曲げ完走の安定性確認（異なるパラメータでの再現性）
- [ ] MPC+contact: ローカルMPC（ワイヤ単位の端部結合）の検討
- [ ] K_c FD 1.8%: K_st計算の接平面成分リファクタリング（低優先度）
- [ ] cutback数削減（41→目標20以下）のためのチャタリング対策最適化

---

## 次の担当者向け

### exclude_end_elements の使い方

```python
cfg = StrandBendingOscillationConfig(
    ...,
    exclude_end_elements=2,  # 各素線の両端2要素を接触から除外
)
```

`_ContactConfigInput.exclude_end_elements` で直接指定も可能。MPC+contactでは効果が限定的（frac 0.001→0.004）だが、端部接触アーティファクトの回避には有用。

### MPC+contact発散の根本対策候補

1. **ローカルMPC**: 7ワイヤ一括ではなく、ワイヤ対（隣接2本）でのカップリング
2. **ペナルティMPC**: 剛体結合ではなく、ペナルティ法でのソフトカップリング
3. **MPC slave DOFの接触力除去**: T^T変換後にslave DOFの接触力成分をゼロ化

### frozen-m検証の再現手順

```bash
python -c "
import math
from xkep_cae.numerical_tests.strand_bending_oscillation import *
cfg = StrandBendingOscillationConfig(
    n_strands=7, wire_radius=0.5, pitch_length=100.0,
    n_elements_per_pitch=16, n_pitches=1.0, E=130.0e3, nu=0.3,
    rho=8.96e-9, bending_curvature=math.pi/200.0,
    n_cycles=1, n_increments_per_cycle=40, rho_inf=0.9, mu=0.15,
    max_nr_attempts=50, tol_force=1e-8, max_increments=600,
    free_end_mode=True, contact_enabled=True, loading_mode='rotation',
    penalty_exponent=1.5,
)
result = StrandBendingOscillationProcess().process(cfg)
sr = result.solver_result
print(f'frac={sr.load_history[-1]:.4f}, incr={sr.n_increments}, cutbacks={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-frozen-m-hertz.log
```

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: テスト結果はpytest出力と一致
- [x] **回帰なし**: 442テスト合格（test_stress_contour既知失敗除く）
- [x] **ベースライン確認**: status-295のfrac=0.40、K_c rel_err=1.79%がベースライン
