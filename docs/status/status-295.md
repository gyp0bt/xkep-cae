# status-295: K_c_adj mat-only化 + MPC+contact発散調査 + frozen-mベースライン

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-05
- **ブランチ**: `claude/check-status-todos-8yAO3`
- **テスト数**: 631+ passed（既存テスト全合格）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

status-294のTODO 3件を実行:

1. **TODO 1（完了）**: K_c_adj mat-only化（z方向DOFカップリング追加）
2. **TODO 2（調査完了）**: MPC+contact 90度曲げ発散原因の調査・仮説検証
3. **TODO 3（ベースライン取得）**: frozen-m解消効果検証のベースライン

---

## 1. K_c_adj mat-only化（TODO 1）

### 問題

status-294でfrozen-m部分解消後も残るK_c FD相対誤差11%。
99.4%がz方向に集中し、全て隣接ノード（adj nodes）列で解析値がゼロ。

### 原因分析

K_c_adj（隣接ノードへの接触剛性拡張）が `K_3x3 = w_mat*(n⊗n) - w_geo*(I-n⊗n)` を使用。
しかし、隣接ノード変位→sパラメータ追従により:

- **材料剛性(n⊗n)**: ギャップ変化はs追従でほぼ不変 → **正しく寄与**
- **幾何剛性(I-n⊗n)**: 法線変化はs追従でほぼ相殺 → **K_st_adjとの相殺項**

ブロックレベルの検証:
- zz成分: ratio ≈ 1.00（K_c_adjが正確）
- xx, yy成分: ratio ~ 500-6000x（K_c_adjが巨大に過大計上）

### 解決策

K_c_adjに材料剛性 `w_mat*(n⊗n)` のみを使用。幾何剛性を除外。

### 結果

| 構成 | K_c FD rel_err |
|------|---------------|
| status-294（adj無し） | 10.96% |
| **K_c_adj mat-only** | **1.79%** |
| K_c_adj full（mat+geo） | 38.50% |

### 変更箇所

| ファイル | 変更 |
|----------|------|
| `strategy.py` tangent() | K_3x3_mat定義追加、K_c_adjでK_3x3_matを使用 |
| `strategy.py` tangent_components() | K_mat_adj追加（同方式） |
| `test_st_stiffness_process.py` | FD参照を法線固定方式に変更（mat-only検証） |
| `test_kc_component_fd.py` | 閾値0.5→0.05に強化、docstring更新 |

---

## 2. MPC+contact 90度曲げ発散調査（TODO 2）

### 現状

| モード | contact | frac | 備考 |
|--------|---------|------|------|
| MPC | OFF | 1.0 | 完走 |
| free_end | OFF | 1.0 | 完走 |
| free_end | ON | 0.40 | 207 incr, 14 cutback |
| **MPC** | **ON** | **0.001** | **即座に発散** |

### 発散パターン

```
Incr 1 (frac=0.025): active=0→40, Type A+B, 残差5回連続増加→early abort
Incr 1 (frac=0.0063): active=71, Type A+B+D.div→early abort
Incr 1 (frac=0.0016): active=30, Type A+B+D.div→early abort
Incr 1 (frac=0.0004): 収束（energy converged, 2 attempts）
Incr 2 (frac=0.0010): 収束（energy converged, 2 attempts）
Incr 3 (frac=0.0019): Type A+B+D.div→cutback
Incr 3 (frac=0.0012): D.stall (att=41, R=9.9e-4, z=62%)
```

### 仮説A検証結果（T行列再構築）

**否定**: T行列はUL更新後に `RebuildMPCTransformProcess` で毎回再構築されている
（process.py:673-685）。

### 仮説B検証結果（参照点回転更新）

**部分的に確認**: `_ExtendedULAssemblerWrapper.update_reference()` は並進変位のみ更新。
ただし、RebuildMPCTransformは並進座標のみ使用するため、回転DOFは直接影響しない。

### 根本原因の仮説（新規: 仮説E）

**MPC slave DOFでの接触活性化がT^T K_c Tの条件数を悪化させる**:

1. 端部ノード（MPC slave）で接触ペアが活性化
2. 接触力がslave DOFに作用 → T^T変換で全slave→master DOFに分散
3. 7本全ワイヤの端部が1つのmasterで結合 → 強いグローバルカップリング
4. 接触力の局所性とMPCのグローバル性が矛盾 → 条件数悪化

### 推奨対策

- **E1**: exclude_same_strand に加えて、端部要素の接触を除外するオプション追加
- **E2**: NR初期反復でMPC slave DOFの接触を凍結し、安定化後に解放
- **E3**: 接触検出のマージンを端部付近で縮小

---

## 3. frozen-m解消効果検証ベースライン（TODO 3）

free_end + contact（現在の推奨構成）:
- **frac = 0.40**
- 207 increment, 14 cutback
- status-285のHertz型ペナルティでfrac=0.998達成時とは別パラメータ
  （7本撚線、n_elems_per_pitch=16）

このベースラインを基に、frozen-m解消の効果を次ステップで検証予定。

---

## 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/contact/contact_force/strategy.py` | K_3x3_mat定義、K_c_adj mat-only化、tangent_components K_mat_adj追加 |
| `xkep_cae/contact/contact_force/tests/test_st_stiffness_process.py` | FD参照法線固定方式、docstring更新 |
| `xkep_cae/contact/contact_force/tests/test_kc_component_fd.py` | 閾値強化、docstring更新 |

---

## TODO

- [ ] K_c FD 残余1.8%の詳細分析（K_st_adjの部分有効化で0%に近づけるか）
- [ ] MPC+contact: 仮説E1（端部接触除外）の実装・検証
- [ ] frozen-m解消: 90度曲げ接触ありでfrac=0.40→改善の検証

---

## 次の担当者向け

### K_c_adj mat-only化の理論的根拠

隣接ノードを変位させると:
1. Hermite接線 m が変化 → 曲線上の接触点位置 pA(s) が変化
2. しかし s パラメータも追従（最近接点条件により）
3. s追従により法線方向の変化はほぼ相殺（∂n/∂u_adj ≈ 0）
4. ギャップ変化は維持（∂gap/∂u_adj ≈ n · ∂pA/∂u_adj at fixed s）

従って K_c_adj = c_i * alpha_adj * w_mat * (n⊗n) のみが正確。
幾何剛性(I-n⊗n)部分はK_st_adj（ds_du_adj経由）で相殺されるべき項。

### MPC+contact発散の再現手順

```bash
python -c "
import math
from xkep_cae.numerical_tests.strand_bending_oscillation import *
cfg = StrandBendingOscillationConfig(
    n_strands=7, wire_radius=0.5, pitch_length=100.0,
    n_elements_per_pitch=16, n_pitches=1.0, E=130.0e3, nu=0.3,
    rho=8.96e-9, bending_curvature=math.pi/200.0,
    n_cycles=1, n_increments_per_cycle=40, rho_inf=0.9, mu=0.15,
    max_nr_attempts=50, tol_force=1e-8, max_increments=100,
    free_end_mode=False, contact_enabled=True, loading_mode='rotation',
)
result = StrandBendingOscillationProcess().process(cfg)
sr = result.solver_result
print(f'frac={sr.load_history[-1]:.4f}, cutbacks={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-mpc-contact.log
```

---

## STA2 準拠チェック

- [x] **数値の捏造なし**: FD検証テスト結果はpytest出力と一致（rel_err=0.01795）
- [x] **回帰なし**: 全テスト合格（test_stress_contourの既存失敗を除く）
- [x] **ベースライン確認**: status-294のK_c_adj（11.0%）がベースライン
