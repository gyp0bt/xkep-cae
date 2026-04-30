[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

# status-380: 物理的妥当性検証 — 7本/19本 explicit 解は数値発散、status-379 凍結解除判定を**撤回**

**日付**: 2026-04-30
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12+11+28+10+11 passed（status-379 と同一、実装本体無変更）

## 概要

status-379 引継ぎ §6.1（**最優先**: 数値解の物理的妥当性検証）を実施。
7 本撚線 90° 曲げで implicit / explicit を比較する `30_implicit_vs_explicit_7strand.py`、
および 19 本 explicit 解の 3D 可視化 `31_render_19strand_explicit.py` を新設し、
解の物理的妥当性を `Strand3DContourProcess` で目視確認。

**重大な発見**: **7 本 / 19 本ともに explicit + mass scaling auto-tune 解は数値発散**。
両ケースで `max |u_trans| ≈ 1.59 × 10⁸ mm`（≈ 159 km）に達し、
3D 形状はヘリカル構造を完全に喪失（撚線が空間を直線的に飛散）。
status-379 は frac=1.0 / E_kin/E_strain<5% という形式 gate 2 件のみで PASS と判定したが、
**変位の物理的妥当性 gate が欠落しており、実際には解そのものが破綻していた**。

→ **status-379 の MCDD 凍結解除条件「19本 frac=1.0 完走」達成判定を撤回**。
mass scaling auto-tune は数学的構造（M_lump → β²·M_lump → Δt_c → β·Δt_c、
E_kin/E_strain 同比率維持）が「frac=1.0 + E_ratio<5% を形式的に満たすが
変位は意味を持たない」という gate 設計の盲点を露出させた。

## 1. 検証実施内容

### 1.1 7 本撚線 implicit vs explicit 比較

`work/beam_hysteresis/30_implicit_vs_explicit_7strand.py`（新規 +330 行）:

7 本撚線 90° 曲げ（`bending_curvature=0.015`、`free_end_mode=True`、
`smoothing_delta=1000`）を **両 solver_mode で完走させ**、両解を比較。

| 項目 | implicit | explicit (mass_scaling auto, max_β=1e3) |
|------|----------|---------------------------------------|
| frac | 1.0000 | 1.0000 |
| 収束 | True | True |
| n_increments | 99 | 269 |
| n_cutbacks | 22 | 31 |
| elapsed [s] | 50.4 | 23.2 |
| **max \|u_trans\| [mm]** | **1.594 × 10²** | **1.584 × 10⁸** |
| tip 並進変位 (x,y,z) [mm] | (157, 1.6, -5.5e-5) | (1.57e8, 1.58, -5.5e-5) |
| active pair 数 | 13 | **0** |
| E_kin / E_strain | 1.6 × 10⁻³ | 1.1 × 10⁻² |

implicit は max disp ≈ 159 mm（撚線長 100 mm の 90° 曲げで物理的に妥当）。
explicit は max disp ≈ 159 km、active pair=0 で接触状態すら成立していない。

3D 可視化:
- `docs/verification/7strand_imp_vs_exp/implicit_*.png` — 物理的に妥当な
  90° 弧状変形、ヘリカル構造保持、24 接触要素検出
- `docs/verification/7strand_imp_vs_exp/explicit_*.png` — 軸スケール 10⁸ mm、
  撚線が空間を直線で飛散、接触要素 0、明確な数値発散

### 1.2 19 本撚線 explicit 解の 3D 可視化

`work/beam_hysteresis/31_render_19strand_explicit.py`（新規 +155 行）:

status-379 採択設定（`solver_mode="explicit"`、`mass_scaling_auto=True`、
`max_beta=1e3`）を **そのまま再現** し、frac=1.0 完走解の 3D 形状を確認。

| 項目 | status-379 報告値 | 本検証実測 |
|------|------------------|-----------|
| frac | 1.0000 | 1.0000 |
| n_increments | 269 | 269 |
| n_cutbacks | 31 | 31 |
| elapsed [s] | 131.07 | 112.76 |
| E_kin/E_strain | 1.148 × 10⁻² | 1.148 × 10⁻² |
| **max \|u_trans\| [mm]** | （未計測） | **1.588 × 10⁸** |

→ status-379 の数値再現性は完全（incr/cb/E_ratio バイト一致）。
**しかし max \|u_trans\| = 1.59 × 10⁸ mm で 7 本撚線 explicit と同じく完全に発散**。

3D 可視化（`docs/verification/19strand_explicit/19strand_explicit_*.png`）:
- contact / contact_force: 接触要素 0、撚線形状を保たず空間に分散
- curvature / stress: 軸スケール 10⁸ mm、19 本が放射状に飛散
- chatter_binary / chatter_score: 19 要素にチャタリング検出

→ **status-379 の MCDD 凍結解除条件達成判定は、変位の物理的妥当性を
全く満たしていない解に対する誤判定**。

## 2. なぜ frac=1.0 + E_kin/E_strain<5% でも発散するのか

### 2.1 mass scaling の数学的盲点

`mass_scaling_beta` は集中質量を β² 倍化する:
$$ M_\text{scaled} = \beta^2 \cdot M_\text{lump}, \quad \Delta t_c = \beta \cdot \Delta t_{c,\text{raw}} $$

中央差分法では:
$$ \ddot{u}_n = M_\text{scaled}^{-1} (f_\text{ext} - f_\text{int} - f_c) $$

質量を β² 倍化すると加速度は 1/β² にスケールするが、**外力 / 内力 / 接触力は
β に依存しない**。1 ステップあたりの変位増分は $\Delta u \propto \dot{u} \cdot \Delta t$
で保たれるが、剛性が小さい / 接触ペナルティが弱い領域で慣性が支配的になり、
撚線が大きく飛散する可能性がある（剛体運動的解）。

### 2.2 E_kin / E_strain 比の幻惑

mass scaling は kinetic energy も β² 倍化するが、status-379 で報告された
1.15% は **両方が β² 倍化された後の比**である。比率は β に独立で、
「準静的近似が成立している」という意味にはならない。

実際の 19 本 explicit 解で:
- `初期 SE: 2.32×10¹⁹` （β² ≈ 10⁶ 倍化された SE）
- `最終 SE: 1.61×10¹²`（β² 倍化前のスケール）

→ 解析中に撚線が圧倒的に変形し、SE スケールが 7 桁減少。
これは「動的緩和」ではなく、「変位が物理的範囲を逸脱した結果として
内力が再評価されただけ」と解釈すべき。

### 2.3 frac=1.0 完走の意味

`load_history[-1]=1.0` は **処方変位 BC が完了した** ことだけを意味する。
explicit ソルバーでは時間積分の各ステップで処方 DOF を BC として固定しており、
これは「BC 値が達成された」ことを保証するだけで、内部 DOF が物理的に
妥当な解に到達したかは別問題。

## 3. 対応

### 3.1 status-379 の判定撤回

CLAUDE.md 「やってはいけないこと」「STA2 防止ルール」「MCDD 脱法 pattern 1」
に照らし、status-379 の **MCDD 凍結解除条件達成判定を撤回**:

- **形式 gate**: frac=1.0 + E_kin/E_strain<5% は **両方とも数学的構造由来で
  発散時にも PASS する** ため、物理的妥当性 gate として不十分
- **必須 gate（追加）**: `max |u_trans| < L_strand × C`（例: C=10、撚線長
  100mm に対し最大変位 1m を許容）

### 3.2 検証スクリプトに変位 gate 追加

両スクリプトに `max |u_trans| < 1m` gate を追加（`30_*.py` `g_phys_imp/exp`、
`31_*.py` `g_max_u`）。再実行すると:

- 7 本 implicit: PASS（max u=159mm、L_strand=100mm の物理的妥当範囲）
- 7 本 explicit: **FAIL**（max u=1.58×10⁸ mm）
- 19 本 explicit: **FAIL**（max u=1.59×10⁸ mm）

### 3.3 MCDD 凍結解除条件の見直し提案

CLAUDE.md `凍結解除条件` を以下に**訂正**:

- **旧**: Phase E 完了 + 19 本 frac=1.0 完走 + `KcNormalDirectionStiffness` FD rel_err < 1e-2
- **新**: Phase E 完了 + 19 本 frac=1.0 完走 **かつ max \|u_trans\| < L_strand × 10**
  + `KcNormalDirectionStiffness` FD rel_err < 1e-2

19 本 implicit + AL n=2（status-376）の frac=0.5746 が現実的な「既知最良」値。

## 4. 次の課題（次 status へ）

### 4.1 mass scaling auto-tune の再検討（最優先）

candidate (h1) が gate 設計の盲点で「達成」と誤判定されただけで、
**実質的には未達**と確定。次の選択肢:

| 候補 | 概要 | 期待 |
|------|------|------|
| (h1') β cap 強化 | `max_beta` 大幅縮小（例: 100）+ 失敗時 cutback | 大変位を強制抑止 |
| (h2) dt subcycling | β 拡大せず dt 細分化のみ | 物理的妥当性維持 |
| (h3) selective explicit | 接触ペアのみ explicit、その他 implicit | 19 本接触剛性を吸収 |
| **(h4) implicit + AL n>2** | status-376 AL n=2 frac=0.5746 を n=3,4 で延伸 | 既存 implicit 系で 1.0 |
| (h5) bending_curvature 段階処方 | 0.005 → 0.010 → 0.015 と段階増加 | NR 良条件再開 |

### 4.2 19 本 implicit AL n=2 完走解の妥当性確認

status-376 の AL n=2 frac=0.5746 解は max |u_trans| が物理的範囲内であるはず
（implicit は 7 本で max u=159mm）。19 本 AL n=2 解の max |u_trans| を測定し、
真の「凍結解除条件」を再定義する基礎データとする。

### 4.3 mass scaling 動作領域の調査

mass scaling は status-378 7 本 smoke（`bending_curvature=0.0005`、線形領域）
では数値完全（β=4.7×10⁴ 1 増分収束）。**90° 大変形 + 接触ペナルティ非線形性**
で破綻する。線形 / 弱非線形領域での妥当性を別途検証し、
適用領域を明文化する必要がある。

## 5. 実装変更

実装本体（`xkep_cae/`、`tests/`、`contracts/`）は **無変更**:
- 検証スクリプト 2 本新設のみ（`work/beam_hysteresis/30_*.py`、`31_*.py`）
- 3D 可視化 PNG 出力（`docs/verification/7strand_imp_vs_exp/` 8 枚 +
  `docs/verification/19strand_explicit/` 6 枚）
- 回帰: 691 passed 5 skipped（status-379 と同一）/ 全 24 契約検査 OK / ruff pass

## 6. MCDD 脱法 pattern 回避

- pattern 1（tol 緩和）: 本 status は **gate 強化** であり緩和ではない
- pattern 6（骨格 status）: 比較スクリプト 2 本 + 3D 可視化 14 枚 + status-379
  訂正提案で完結
- pattern 8（根拠なき主張）: max |u_trans| 数値 + 3D 可視化で解の発散を実証
- pattern 10（TODO 先送り）: 候補 (h1') / (h2) / (h3) / (h4) / (h5) を §4.1 に
  明記、次 status で着手

## 7. 引継ぎコマンド

```bash
# 7 本撚線 implicit vs explicit 比較
uv run --extra dev python work/beam_hysteresis/30_implicit_vs_explicit_7strand.py \
    2>&1 | tee /tmp/imp_vs_exp_7strand_$(date +%s).log

# 19 本 explicit 3D 可視化
uv run --extra dev python work/beam_hysteresis/31_render_19strand_explicit.py \
    2>&1 | tee /tmp/render_19strand_explicit_$(date +%s).log

# 回帰（status-379 と同一）
pytest xkep_cae/contact/ xkep_cae/mathematics/ xkep_cae/time_integration/ \
       xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py
python contracts/validate_process_contracts.py
ruff check xkep_cae/ tests/ && ruff format --check xkep_cae/ tests/
```

## 8. 観察 — 開発運用上の効果的 / 非効果的な点

### 8.1 効果的だった点

- **3D 可視化の即時実行**: 数値だけでは「frac=1.0 + E_ratio<5%」で済まされた
  発散が、3D 画像 1 枚で即座に判別できた。`Strand3DContourProcess` の整備
  （status-362）が status-379 の見落としを救う形になった。
- **比較スクリプトの両ソルバー実行**: 同じ問題を両方で解いて差分を取る
  シンプルな手法が、形式 gate の盲点を最も直接的に露出させた。

### 8.2 非効果的だった点（status-379 で発見した運用の弱点）

- **gate 設計が数学的構造に依存**: frac=1.0 / E_kin/E_strain は両方とも
  「BC 達成 + 比率」で発散時にも数値が PASS する。本当に物理的妥当性を
  問うには、変位の絶対スケール gate（例: max disp < L_strand × C）が必要。
- **MCDD 凍結解除条件が形式 gate のみ**: `KcNormalDirectionStiffness` FD
  rel_err は機械精度を維持しているが、これは数学的整合性であって解の物理性
  ではない。**Phase F として「物理的妥当性 gate」を MCDD に追加** する
  ことを提案する（次 status での議論項目）。
