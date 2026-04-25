# Pairwise Freezing 設計仕様 (status-374 Phase 1)

[← README](../../../../README.md)

## 概要

候補 (g3) pair-wise relaxation の **Phase 1 インフラ**。status-284 で導入された
全体凍結モード（NR 全反復で接触力ベクトル全体を snapshot 値に固定）を
**pair granularity** に拡張する `PairwiseFreezingProcess` を新設する。本
status-374 では単体 Process + ユニットテストのみ、solver（`_newton_dynamic.py`）
への配線は Phase 2（status-375 以降予定）に分離する。

## 背景

status-368 で `chattering_freeze_*` 3 パラメータ × 6 ケース感度掃引を実施し、
Case B `chattering_freeze_nr_max=30` のみ frac=0.5642（+50.9%、status-339
baseline 比 +16.6%）の改善を得たが、MCDD 凍結解除条件 `frac=1.0` 未達で
**全体凍結モードはクローズ**。一方 disabled ケースは `D+E:98%` で 200 反復
ハマるため、freeze 機構自体は **D+E ロック回避の支柱**として残存。

status-370 Phase C-3' Step 3.1 active 境界 FD 診断で全 20 測定点 rel_err
2.18e-07〜2.20e-07 の機械精度を維持し、**結果 B 確定**（K_c 項欠落ではなく
NR alg 側動力学）。`phase_c3prime_19strand_plan.md` §3.2 で候補 (g) を 3
サブラインに再配分し、本 (g3) は status-372 で却下された (g1) active 履歴
EMA 平滑化（19 本 frac=0.5133 / elapsed +131%）の代替として、19 本 Type D
stall の **多 pair 相互作用**領域に対する pair 単位介入を試みる。

## 数理

### 全体凍結 → pair-wise 凍結への拡張動機

status-284 全体凍結は

    R_u ← R_u - f_c + f_c_snapshot
    f_c ← f_c_snapshot   （NR 反復内で固定）

によって**全 active 接触ペアの寄与**を NR から実質排除する。これは active
集合振動を完全停止させるため Type A/B/E（active flip / 摩擦 stick-slip flip）
には強力だが、安定 pair まで凍結すると有効自由度が過剰に削られ NR 収束能
そのものが低下する（status-368 Case B `nr_max=30` で 19 本 +16.6% に頭打ち
した一因と推定される）。

pair 単位凍結では

    freeze[k] := (flip_counts[k] ≥ threshold) ∧ is_active_now[k]
    R_u ← R_u - Σ_k freeze[k] · (f_c[k]_current - f_c[k]_snapshot)

で振動 pair k のみ snapshot 値に固定し、安定 pair は active のまま NR が
通常通り更新できる。これにより 19 本撚線で支配的な多 pair 相互作用領域に
おいて、NR の有効自由度を最大限残しつつチャタリングだけ抑制できる。

### Phase 1 判定アルゴリズム（純計算）

入力（per-pair）:
- `pair_active_flip_counts[k]`: ペア k の累積 active 切替回数（NR 反復間で
  Phase 2 がインクリメント）
- `is_active_now[k]`: ペア k の現反復 active 状態
- `chattering_type`: グローバル分類文字列（`classify_chattering_type` 出力）

判定:

    skip_global := skip_when_type_d_dominant ∧ _is_type_d_dominant(chattering_type)

    freeze[k] := False                           （skip_global == True）
              := False                           （is_active_now[k] == False）
              := True                            （flip_counts[k] ≥ threshold）
              := False                           （otherwise）

`_is_type_d_dominant("D+E")` = False、`_is_type_d_dominant("D")` = True。Type D
単独支配時に freeze をスキップする理由は status-288 の Type D 対策方針と同じ
（active 集合安定 + 接線不整合 → 凍結は意味がなく NR 反復拡張で対応）。

### Phase 2 配線（次 status 以降）

```
NR 反復先頭で:
    flip_counts = _update_pair_active_flips(flip_counts_prev,
                                           is_active_now, is_active_prev)
    out = PairwiseFreezingProcess().process(PairwiseFreezingInput(
        n_pairs=len(manager.pairs),
        pair_active_flip_counts=flip_counts,
        is_active_now=is_active_now,
        flip_threshold=cfg.pairwise_freeze_flip_threshold,
        chattering_type=classify_chattering_type(_nr_snap),
        skip_when_type_d_dominant=cfg.pairwise_freeze_skip_type_d,
    ))
    if not out.skip_freeze_global and out.n_frozen > 0:
        # 凍結ペア k の接触力寄与を snapshot に差し替え（per-pair 組立）
        for k in np.where(out.pair_freeze_flags)[0]:
            f_c -= per_pair_force_current[k]
            f_c += per_pair_force_snapshot[k]
        # K_c も同様に凍結ペアの寄与をマスク
```

per-pair 組立は `HuberContactForceProcess` 出力を pair 単位で保持する経路の
追加が必要（Phase 2 設計）。または既存全体組立後、凍結ペアの DOF ブロックを
snapshot で上書きする近似版から開始する。

## API 仕様

### PairwiseFreezingInput

| field | type | 説明 |
|-------|------|------|
| `n_pairs` | int | 接触ペア総数（INACTIVE 含む） |
| `pair_active_flip_counts` | `np.ndarray[int]` (n_pairs,) | 各ペアの累積 active flip 回数 |
| `is_active_now` | `np.ndarray[bool]` (n_pairs,) | 現反復 active/inactive 状態 |
| `flip_threshold` | int | 凍結発動閾値（既定 3） |
| `chattering_type` | str | グローバル分類文字列（"" / "D" / "D+E" / "A+B+E" 等） |
| `skip_when_type_d_dominant` | bool | Type D 単独時 skip（既定 True） |

### PairwiseFreezingOutput

| field | type | 説明 |
|-------|------|------|
| `pair_freeze_flags` | `np.ndarray[bool]` (n_pairs,) | 各ペアの凍結フラグ |
| `n_frozen` | int | freeze=True のペア数 |
| `n_active_pairs` | int | is_active_now=True のペア数 |
| `skip_freeze_global` | bool | Type D 単独支配 skip 判定結果 |
| `freeze_reasons` | `tuple[str, ...]` (n_pairs,) | per-pair 判定理由文字列 |

### no-op 経路

- `n_pairs == 0`: 全フィールド空（freeze_flags shape (0,) bool）
- `chattering_type` 単独 "D" + `skip_when_type_d_dominant=True`: 全フラグ
  False、skip_freeze_global=True、reasons 全要素 "skip_type_d"

### shape ガード

`pair_active_flip_counts.shape != (n_pairs,)` または `is_active_now.shape !=
(n_pairs,)` で `ValueError` を送出。NR ループ側のバグを Phase 2 結合時に早期
検知する。

## MCDD 整合性

`@verified_by` は **Phase 2 の solver 配線時に検討**。Phase 1 は
`TermExpansionContract` の K_c 5 項に所属しない（凍結は NR 制御ロジックで
力法則ではないため、ContactNormalDamping と同じく独立系統）。本 Process は
全契約検査（C3-C24）と直交。

## テスト

`tests/test_strategy.py` で以下を検証（12 テスト、`@binds_to(PairwiseFreezingProcess)` メタ整合は API 契約テストクラスに付与）:

| カテゴリ | テスト | 検証内容 |
|----------|--------|----------|
| API 契約 | 3 | `n_pairs=0` no-op / shape mismatch raises ValueError / Output 型整合 |
| 判定ロジック | 6 | `flip_counts < threshold` no-freeze / `flip_counts >= threshold` freeze / inactive はスキップ / Type D 単独 skip / Type D+E は freeze 実行 / threshold 境界値 |
| ヘルパ純関数 | 3 | `_update_pair_active_flips` 整合 / `_is_type_d_dominant` 真理値表 / shape mismatch safety |

## 参照

- status-284: 全体凍結モード初出（`_newton_dynamic.py` `_freeze_active`/`_freeze_f_c`）
- status-288: Type D 対策方針（active 集合安定 + 接線不整合は freeze 意味なし）
- status-368: `chattering_freeze_*` 19 本感度掃引、Case B `nr_max=30` で
  +16.6%、frac=1.0 未達で全体凍結クローズ
- status-370: Phase C-3' Step 3.1、結果 B 確定（K_c 項欠落ではなく NR alg 側）
- status-372: 候補 (g1) active 履歴 EMA 平滑化 19 本却下（frac=0.5133、
  elapsed +131%）
- 設計上位文書: [`phase_c3prime_19strand_plan.md`](../../../mathematics/docs/phase_c3prime_19strand_plan.md) §3.2
