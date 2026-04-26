# status-375: 候補 (g3) pair-wise relaxation Phase 2 NR 配線 + 19 本実機検証で却下

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-25
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12 passed（status-374 比 維持、Phase 2 配線本体は既存 freeze 12 テストで回帰検証）

## 概要

status-374 で Phase 1 単体実装を完了した `PairwiseFreezingProcess` を、Phase 2
として **NR ループ (`_newton_dynamic.py`) に配線**し、`StrandBendingOscillationConfig`
から 3 経路 plumb-through を整備した。

**判定: 候補 (g3) pair-wise relaxation 却下**。

19 本撚線 90° 曲げで `pairwise_freeze_flip_threshold ∈ {2, 3, 5}` 掃引、
**全 3 ケースで Gate `frac ≥ 0.6` 未達**:

| flip_threshold | frac | incr | cb | elapsed | baseline 比 | Gate |
|----------------|------|------|----|---------|-------------|------|
| 2 (aggressive) | 0.1975 | 87 | 19 | 229.35s | -47.2% | FAIL |
| 3 (default) | 0.3482 | 145 | 18 | 254.25s | -6.9% | FAIL |
| 5 (conservative) | 0.1963 | 90 | 19 | 235.54s | -47.5% | FAIL |

baseline (status-357 19 本 K_c FD 再計測): `frac=0.3739`。

7 本撚線回帰: **frac=1.0 維持**（default OFF でバイト一致、`pairwise_freeze_enabled=False` 時は
従来コードパスがそのまま走る）。

**結論**: pair-wise DOF block 上書きアプローチは、19 本 Type D stall が
**K_c x/z カップリング不整合**（status-344 mat_only rel_err 44%）に支配されている
領域では、active flip を per-pair で凍結しても根本症状（接線剛性誤差）を解消できない。
status-362 候補 (c) line search、status-367 候補 (e) 接触減衰、status-368 候補 (d)
接触凍結モードと**同パターンの却下**。

## 1. Phase 2 NR 配線実装

### 1.1 新ディレクトリは追加なし — 既存 freeze パッケージ流用

status-374 で作成済の `xkep_cae/contact/freeze/` を利用、新規ファイル追加なし。
本 Phase 2 は配線のみ。

### 1.2 変更箇所

| ファイル | 変更内容 |
|----------|----------|
| `xkep_cae/core/data.py` | `ContactFrictionInputData` に `pairwise_freeze_enabled` / `pairwise_freeze_flip_threshold` / `pairwise_freeze_skip_type_d` 3 field 追加（default OFF） |
| `xkep_cae/contact/solver/_newton_dynamic.py` | (1) `NewtonDynamicInput` に同 3 field 追加、(2) `PairwiseFreezingProcess` import + `uses` 追加、(3) NR 反復先頭で is_active_now 構築 → flip_counts 更新 → process 呼び出し → freeze=True ペアの DOF block を snapshot に上書き、(4) `pairwise_freeze_enabled=True` 時の既存全体凍結 (`chattering_freeze_*`) 排他抑止 2 箇所 |
| `xkep_cae/contact/solver/process.py` | (1) `freeze_slot = StrategySlot(default_types=(PairwiseFreezingProcess,))` 追加、(2) `nr_config_dyn` で 3 field plumb-through |
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | (1) `StrandBendingOscillationConfig` に同 3 field 追加（docstring 付き）、(2) 曲げ / 揺動 / free_end 3 経路で `ContactFrictionInputData` 構築箇所に plumb-through 追加 |
| `work/beam_hysteresis/27_pairwise_freeze_19strand.py` | 新規 19 本撚線 90° 曲げ検証スクリプト（92 行） |

合計実装: `_newton_dynamic.py` +88 行（NR 反復内のロジック）/ `process.py` +18 行 /
`data.py` +9 行 / `strand_bending_oscillation.py` +12 行 + 9 経路（plumb-through）.

### 1.3 NR 配線アルゴリズム（DOF block 上書き近似版）

```python
# NR 反復先頭（force assembly 後、effective_residual 前）
if pairwise_enabled:
    is_active_now = [pair.state.p_n > 0 for pair in pairs]   # bool[n_pairs]
    if active_prev is not None:
        flip_counts += (is_active_now != active_prev).astype(int)
    chatter_type = classify_chattering_type(prev_nr_snapshot)

    out = PairwiseFreezingProcess().process(PairwiseFreezingInput(...))
    active_prev = is_active_now

    if not out.skip_freeze_global and out.n_frozen > 0:
        if freeze_f_c_snapshot is None:
            freeze_f_c_snapshot = f_c.copy()
        # 凍結ペアの DOF を構築
        dof_mask = np.zeros(ndof, dtype=bool)
        for k in np.where(out.pair_freeze_flags)[0]:
            dof_mask[_contact_dofs(pairs[k], ndof_per_node)] = True
        # f_c / R_u を snapshot 値に固定
        f_c_orig = f_c
        f_c = f_c.copy()
        f_c[dof_mask] = freeze_f_c_snapshot[dof_mask]
        R_u = R_u - f_c_orig + f_c
```

**設計選択**: status-374 §3 で挙げた 2 案（per-pair 力ベクトル組立 vs DOF block 近似）
のうち、後者を採用。理由:
- `HuberContactForceProcess` の出力構造を破壊せず、最小変更で実装可能
- 既存全体凍結 (`_freeze_active`/`_freeze_f_c`、status-284) と同パターンで保守容易
- DOF 重複（pair k と pair k' が同じノードを共有）は最後の freeze pair の値で上書きされる近似だが、
  Phase 2 の探索的検証としては十分

**排他制御**: `pairwise_freeze_enabled=True` のとき:
- 既存全体凍結 (`chattering_freeze_*`) の発動側 2 箇所（高残差ストール / 低残差チャタリング）で
  `not _pairwise_enabled` ガードを追加して排他化
- pair-wise 経路と全体凍結経路の二重発動を防止

## 2. 検証

### 2.1 Default OFF 回帰（gate 必達）

| 項目 | 結果 |
|------|------|
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK |
| `pytest xkep_cae/contact/` | **468 passed, 5 skipped**（status-374 維持） |
| `pytest xkep_cae/mathematics/` | 109 passed（status-374 維持） |
| `test_helical_3d_hermite` | rel_err=2.18e-07 維持（status-356 機械精度継続） |
| 7 本撚線 90° 曲げ (`TestHelical90DegBendConvergence::test_7strand_90deg_dynamic_completes`) | **PASS, 12.63s**（frac=1.0 完走、回帰なし） |
| `ruff check xkep_cae/ tests/` | OK（201 files） |
| `ruff format --check xkep_cae/ tests/` | OK |

### 2.2 19 本撚線実機検証（gate 未達）

`work/beam_hysteresis/27_pairwise_freeze_19strand.py` で flip_threshold ∈ {2, 3, 5} を実測:

| ケース | flip_threshold | frac | incr | cb | elapsed [s] | converged |
|--------|---------------|------|------|----|-------------|-----------|
| baseline (status-357) | — | 0.3739 | — | — | — | False |
| **threshold=2** | aggressive | 0.1975 | 87 | 19 | 229.35 | False |
| **threshold=3** | default | 0.3482 | 145 | 18 | 254.25 | False |
| **threshold=5** | conservative | 0.1963 | 90 | 19 | 235.54 | False |

**観察**:

- threshold=3 (default) でも baseline 比 **-6.9% 退化**。pair-wise freeze が活性化すると
  発散検知（残差 5 回連続増加）が頻発し early abort → 細かい cutback で進捗。
- threshold=2/5 は **-47% 退化**。閾値の感度は強い non-monotonic だが、
  どの値も baseline を上回らない。
- chattering_type ログは `A+B+D` 中心で、Type D 単独支配でないため `skip_freeze_global` は
  発動しない。Type D dominant (`D` のみ) になる前に `A+B+D` の混合パターンで freeze 発動 →
  発散加速の悪循環。
- 7 本撚線では baseline (frac=1.0) を一切壊さず（`pairwise_freeze_enabled=False`）
  完全互換。

### 2.3 物理的解釈

19 本 Type D stall の主因は **K_c x/z カップリング不整合**（status-342/344 で mat_only
rel_err mean=44%、x 成分 max=98% を確定）。pair-wise freeze は active 集合振動を
per-pair で凍結する数値的 escape hatch だが、**接線剛性自体の誤差を補正しない**ため、
接線方向の悪化（Type D）には無力。むしろ凍結された pair の DOF block で R_u 残差が
人為的に固定されることで、隣接 pair に正のフィードバックが波及して divergence を誘発した
可能性が高い（NR Type 分布が `A+B+D.div:71%` に集中）。

これは status-362 候補 (c) line search、status-367 候補 (e) 接触減衰、status-368 候補 (d)
接触凍結モードと**同パターン**の却下: いずれも 19 本 Type D stall の K_c 構造的不整合に
対しては局所的対処では追いつかない。

## 3. MCDD 凍結解除条件への影響

**Phase E gate 基準** (`docs/roadmap.md`):
- 19 本 frac=1.0 完走 ← **未達**（候補 (g3) で 0.3482 止まり）
- `KcNormalDirectionStiffness` FD rel_err < 1e-2 ← active 集合固定下で機械精度（status-356、status-370）

候補 (g) 3 サブライン (status-370 plan doc) のうち:
- (g1) active EMA 平滑化: status-371/372 で既に **却下**（19 本 elapsed +131%）
- (g3) pair-wise relaxation: 本 status-375 で **却下**（19 本 -6.9〜-47.5%）
- (g2) AL 再導入: **未着手**（次候補、最後の (g) サブライン）

**MCDD 脱法 pattern 6 / 8 / 10 回避**:
- 数値結果は実測値のまま記録、目標緩和なし（pattern 1）
- Phase 2 配線は単体テストではなく実機 19 本撚線で検証済（pattern 6）
- baseline 0.3739 を実測再現できない場合に「ベースラインが誤っていた」と主張せず（pattern 8）
- 退化を「次回掃引で改善する」と先送りせず即却下判定（pattern 10）

## 4. 引継ぎ（status-376 へ）

### 4.1 最優先: 候補 (g2) AL 再導入

status-221 で凍結した Uzawa 外側ループの 1〜2 回限定再導入:
- 設計参照: status-221 / `docs/math/03_huber_contact_penalty.md` §5 "AL motivation"
- 実装方針: `_newton_dynamic.py` の NR 内側 + AL 外側（max 2 cycle）の二重ループ化
- gate: 19 本 frac ≥ 0.6（候補 (g3) と同基準）
- 副次: 摩擦接線剛性符号問題（status-147 で凍結）の 19 本撚線への波及を再検証

### 4.2 候補 (g2) も却下時: explicit 時間積分への移行

`solver_mode = "explicit"` 拡張（status-373 §3）:
- 陰解法 default / リスタート opt-in / explicit opt-in の 3 値拡張
- 19 本以上の K_c 構造的不整合を陽解法（時間積分自体で安定化）で escape

### 4.3 副次: solver_mode フラグ実装

status-373 §3 設計に従い、`StrandBendingOscillationConfig.solver_mode:
Literal["implicit","restart"]` 新設。本 status-375 では未実装、次 status へ繰越。

### 4.4 検証スクリプトの取扱い

`work/beam_hysteresis/27_pairwise_freeze_19strand.py`: **失敗実験記録**として残置
（status-358/360/372 と同方針）。gate 未達 3 ケースのログは `/tmp/pairwise_freeze_19strand_*.log`
に保存（一時領域）。

## 5. 運用所見

### 5.1 Phase 1/2 分割の有効性

status-374 Phase 1 で純計算 Process を完結させ、Phase 2 で NR 配線のみに集中できた
（status-365/366 ContactNormalDamping と同パターン）。Phase 2 配線の不具合（divergence
誘発）が判明しても、Phase 1 ロジック自体は単体テスト 12 件で独立に検証済のため、
原因切り分けが容易。

### 5.2 DOF block 近似の限界

per-pair 力ベクトル組立を導入せず DOF block 上書きで済ませた選択は、実装コストを
最小化したが、隣接 pair への正のフィードバックを誘発した可能性がある。次サブライン
((g2) AL 再導入) では同じ近似は採用せず、AL ループで全 pair の力を一斉に再評価する
**全体的アプローチ**が必要。

### 5.3 候補 (g) サブライン 2/3 が却下された意味

候補 (g1)（status-371/372、active EMA）+ (g3)（本 status）の 2/3 が却下された結果、
**19 本 Type D stall は active 集合振動 / pair 間相互作用ではなく、K_c x/z カップリング
不整合の構造的問題**であることが、status-370 結果 B（active 境界 FD 機械精度）と
合わせて確証された。MCDD 数理側の追加調査（K_mat の x/z 成分二次補正項）が次の
本命課題となる可能性が高い。
