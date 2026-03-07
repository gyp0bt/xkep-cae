# status-131: UL+NCP統合 — adaptive_timesteppingとUpdated Lagrangianの一体化

[← README](../../README.md) | [← status-index](status-index.md) | [← status-130](status-130.md)

**日付**: 2026-03-07

## 概要

status-130 で実装した UL CR梁アセンブラを NCP ソルバー内部に統合。
従来は外側ループで手動ステップ分割していたが、NCP 内部の adaptive_timestepping と UL 参照更新を一体化。
不収束時に**角度増分の自動縮小**、楽に収束したら**自動拡大**が可能に。

## 設計

### 問題（status-130 の課題）

status-130 では `wire_bending_benchmark.py` が NCP ソルバーを `n_load_steps=1` で N 回独立に呼び出していた:

```python
for bend_step in range(n_bending_steps):
    result = newton_raphson_contact_ncp(..., n_load_steps=1)
    ul_asm.update_reference(result.u)
```

この設計の問題:
1. NCP 内部の adaptive_timestepping が 1 ステップ完結で実質無意味
2. `n_bending_steps` のハードコード（45°→15, 90°→30）が必要
3. 不収束時に break するだけで自動リカバリーなし
4. 揺動や他の撚線構成で全く通用しないアドホック設計

### 解決策: NCP 内部に UL コールバック統合

`newton_raphson_contact_ncp()` に `ul_assembler` パラメータを追加:

```python
result = newton_raphson_contact_ncp(
    ...,
    n_load_steps=n_bending_steps,
    prescribed_values=total_angle,  # 全角度
    adaptive_timestepping=True,
    ul_assembler=ul_asm,            # NEW
)
```

NCP 内部で各ステップ収束後に:
1. `ul_assembler.update_reference(u)` — 参照配置更新
2. `node_coords_ref = ul_assembler.coords_ref` — 座標切替
3. `u = 0, u_ref = 0` — 変位リセット
4. `_ul_frac_base = load_frac` — 処方変位のオフセット更新

処方変位は `u[dofs] = (load_frac - _ul_frac_base) * prescribed_values` で計算。
UL リセット後は `load_frac - _ul_frac_base` が次のステップ増分になるため、
**adaptive_timestepping が自然に角度増分を制御**する。

### ULCRBeamAssembler 拡張

| メソッド | 内容 |
|---------|------|
| `checkpoint()` | coords_ref, R_ref, u_total_accum を保存 |
| `rollback()` | チェックポイントから復元（adaptive Δt ロールバック用） |
| `u_total_accum` | 初期配置からの累積変位（出力用プロパティ） |
| `get_total_displacement(u_incr)` | 累積 + 現在増分 |

### adaptive_timestepping との連動

| イベント | UL 動作 | adaptive_dt 動作 |
|---------|---------|-----------------|
| ステップ収束 | 参照更新 + u リセット | 次ステップ幅を反復数で拡大/縮小 |
| ステップ不収束 | rollback で参照復元 | Δt 縮小してリトライ |
| チェックポイント | checkpoint() で状態保存 | u_ckpt 等を保存 |

### 旧ターゲットスキップ修正

adaptive_dt の成長で追い越された旧 bisection ターゲットをスキップするロジックを追加:
```python
if load_frac <= load_frac_prev + 1e-15:
    step_queue.popleft()
    continue
```
UL なしの非 UL ケースでも正しい動作（既存テスト影響なし）。

### n_bending_steps 自動推定

`wire_bending_benchmark.py` の `n_bending_steps` を `int | None` に変更:
- `None`（デフォルト）: `ceil(bend_angle_deg / max_angle_per_step_deg)` で自動推定
- `max_angle_per_step_deg=3.0` がデフォルト
- 明示指定も可能（後方互換）

## 収束結果

### 7本撚線 45° 曲げ（UL+NCP 統合, adaptive=True）

```
Step 1/15, iter 4 (converged, 0 active)
Adaptive dt: delta 0.0667 → 0.1000  # 3° → 4.5°
Step 2/15, iter 4 (converged, 0 active)
Adaptive dt: delta 0.1000 → 0.1500  # 4.5° → 6.75°
Step 3/15, 30 iters → FAILED           # 6.75° は大きすぎ
Adaptive dt retry: delta 0.1500 → 0.0750  # → 3.375°
Step 3/15, iter 4 (converged)           # 3.375° で収束
...（以降 ~5° で安定）
Phase 1 完了: converged=True, NR=86, 10 steps
```

- 旧（固定 15 steps）: 75 NR 反復, 44.5s
- 新（adaptive 10 steps）: 86 NR 反復, 53.5s
- adaptive が ~5° の安定角度を自動発見

### 7本撚線 90° 曲げ（UL+NCP 統合, adaptive=True）

- converged=True
- adaptive_timestepping が自動で適切な角度増分を選択

## 互換ヒストリー

| 旧機能 | 新機能 | 移行status | 備考 |
|--------|--------|-----------|------|
| UL 外部手動ループ | NCP 内部 UL 統合 | status-131 | `ul_assembler` パラメータ追加 |
| `n_bending_steps: int = 45` | `n_bending_steps: int \| None = None` | status-131 | 自動推定 `ceil(angle/3°)` |

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/elements/beam_timo3d.py` | ULCRBeamAssembler に checkpoint/rollback/u_total_accum 追加 |
| `xkep_cae/contact/solver_ncp.py` | `ul_assembler` パラメータ、UL 参照更新/リセット/ロールバック統合、旧ターゲットスキップ修正 |
| `xkep_cae/numerical_tests/wire_bending_benchmark.py` | 手動 UL ループ → 単一 NCP 呼出し、n_bending_steps 自動推定 |
| `tests/contact/test_ncp_bending_oscillation.py` | n_bending_steps ハードコード除去（自動推定に依存） |

## TODO

### 次の優先: ボトルネック分析・高速化・収束安定化（S3復帰前に完了すべき）

- [ ] **ボトルネック調査**: NCP ソルバーの計算プロファイリング（7本/19本で hotspot 特定）
- [ ] **高速化対策**: プロファイリング結果に基づく最適化実装
- [ ] **収束安定化**: adaptive_timestepping + UL の安定性向上
  - adaptive_timestepping の角度上限パラメータ化（現在は adaptive が自動発見、~5°）
  - UL Phase 2（揺動）の特異行列問題調査・修正（status-130 から継続）

### S3 復帰後

- [ ] 19本撚線の90° UL曲げ収束テスト追加
- [ ] 解析的接線剛性のB行列修正（Rodrigues drill spin成分の反映）

### 運用改善

- [ ] CLAUDE.md にログ tee 必須化ルール追加済み
