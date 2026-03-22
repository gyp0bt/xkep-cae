# 接触力符号規約

[← contact_friction](contact_friction.md) | [← README](../../../README.md)

## 結論

**残差式内で `f_c = -f_c` とし、接触力を外力方向に反転する。**

```python
# _nuzawa_steps.py ContactForceAssemblyProcess.process()
f_c = -f_c  # アセンブリ後に反転
R_u = f_int + f_c - f_ext
```

## 符号の流れ

| ステージ | 変数 | 符号 | 理由 |
|---------|------|------|------|
| gap | `g = dist - (r_A + r_B)` | 貫入時 **負** | 標準定義 |
| softplus | `p_n = (1/δ)log(1+exp(-δg))` | g<0 → **正** | 押し返す力の大きさ |
| normal | `(point_A - point_B) / dist` | A=ワイヤ, B=ジグ → **上向き** | ジグが下にあるため |
| g_shape | ワイヤ側: `+(1-s)*n`, ジグ側: `-(1-t)*n` | ワイヤ上向き, ジグ下向き | 接触形状関数の規約 |
| f_c (生) | `Σ p_n * g_shape` | ワイヤ側 **上向き** | p_n>0, normal上向き |
| f_c (反転後) | `-f_c` | ワイヤ側 **下向き** | 外力として残差に加算 |
| R_u | `f_int + f_c - f_ext` | f_c がワイヤを下に押す | NR補正で上に戻す=正しい |

## 接線剛性

K_c は反転**しない**。理由:
- `K_c = dfc_raw/du` は softplus 導関数 `dp/dg * dg/du` に基づく
- f_c を反転すると `d(-fc)/du = -K_c` だが、`K_T = K + K_c` のまま維持
- K_c は exact_tangent=True 時に softplus の sigmoid で正定値寄与
- K_c を反転すると負定値になり、K_T の正定値性が壊れる
- 近似接線（修正ニュートン）として K_c は正の寄与で十分

## 経緯（何度も再発した理由）

1. **status-147**: NCP鞍点系の摩擦接線剛性符号問題を発見。smooth_penalty に移行。
2. **status-219**: k_pen 適正化で 0.5 周期収束達成（自由振動バウンス方式）。
3. **status-220**: 変位制御押し下げに変更 → ワイヤが**上に**動く符号問題を特定。
   - 「f_c は内力と同じ符号で加算される（= 内力として扱い）」
   - 「しかし softplus の接触力計算は f_c を負で返す（= 外力的な符号）」
   - 根本原因を特定したが修正は次セッションへ。
4. **status-221**: `R_u = f_int + f_c - f_ext` → `R_u = f_int - f_c - f_ext` を試すも
   K_c の符号不整合で発散。最終的に `f_c = -f_c` をアセンブリ後に行い、
   K_c は変更せず近似接線として運用。δ=5000 で収束達成。

**なぜ再発するか**: g_shape の符号規約（ワイヤ→上向きが正）が
残差式の符号規約（内力→正、外力→負）と一致しない。
この文書を参照すれば再発を防止できる。

## 用語

| 用語 | 意味 |
|------|------|
| increment | 荷重増分（load_frac が 0→1 へ進む 1 ステップ） |
| attempt | 1 increment 内の NR 反復 1 回 |
| step | increment と同義（診断出力での表示名） |
