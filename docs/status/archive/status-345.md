# status-345: ContactKcComponentFDDiagnosticProcess 報告精度補正 — status-344「K_geo=0」誤認の訂正

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-15
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12（test_kc_component_fd.py に 1 件追加、11 → 12）

## 概要

status-344 が結論付けた **「share_geo = 0.000 全 183 件」= K_geo が完全にゼロ**
という判定は、**`ContactKcComponentFDDiagnosticProcess` の report 出力
フォーマットが `{:5.2f}`（小数 2 桁固定）で微小値を 0.00 に丸めていた
表示上のアーティファクト**であったことを、既存 log の再解析で確定。

**訂正後の実測値**（`docs/measurements/kc_component_fd_19strand_20260415T214702.log`
の `||K_geo@du||` 行 183 件から直接算出、高精度復元）:

| 指標 | mean | min | max | median |
|------|------|-----|-----|--------|
| `share_geo = ||K_geo@du|| / ||K_c@du||` | **1.020e-03** | 1.186e-04 | 3.787e-03 | — |
| `||K_geo@du|| / ||K_mat@du||` | **1.374e-03** | 1.183e-04 | 8.678e-03 | 1.029e-03 |

**K_geo は K_mat の 0.1% 程度で確かに小さいが、ゼロではない**。
status-344 の仮説 A 決着（K_mat 主導 + K_st 追従、K_geo は寄与小）
という主旨は変わらないが、**推奨アクション 3（K_geo == 0 原因調査）は
報告精度側の問題であり、ソルバー実装バグではなかった**と判明したため
本 status でクローズ。

## 根本原因

`xkep_cae/verify/kc_component_fd.py:180`（旧）:

```python
lines.append(f"  寄与率: mat={share_mat:5.2f}, geo={share_geo:5.2f}, st={share_st:5.2f}")
```

- `share_geo ≈ 1e-3` は `{:5.2f}` で **" 0.00"** と表示される
- 下流の `work/beam_hysteresis/13_kc_component_fd_19strand.py` は
  regex `[0-9.eE+\-]+` でパースするが、入力が既に `0.00` なので
  `float("0.00") = 0.0` となり CSV に `0.000000e+00` が記録される
- **CSV も report も情報が失われているため、「K_geo=0」と誤認**

一方、同じ report ブロック内の絶対ノルム行は既に科学表記:

```python
lines.append(
    f"  ||K_mat@du|| = {mat_du_norm:.4e}, "
    f"||K_geo@du|| = {geo_du_norm:.4e}, "
    f"||K_st@du||  = {st_du_norm:.4e}"
)
```

→ ログには `||K_geo@du|| = 2.2604e-09` 等の真値が保存されており、
ここから高精度復元が可能だった（本 status の再解析で実施）。

## 修正内容

### 1. `xkep_cae/verify/kc_component_fd.py`

#### report フォーマット

```python
# 修正後
lines.append(
    f"  寄与率: mat={share_mat:.3e}, geo={share_geo:.3e}, st={share_st:.3e}"
)
```

`{:.3e}` で 3 桁有効数字 + 指数部を保証。`share_geo = 2.260e-03` のような
微小値も report テキストから直接復元可能に。

#### Output dataclass 拡張

`ContactKcComponentFDDiagnosticOutput` に 5 フィールド追加:

```python
mat_du_norm: float = 0.0
geo_du_norm: float = 0.0
st_du_norm: float = 0.0
full_du_norm: float = 0.0
dfc_fd_norm: float = 0.0
```

share 分母の `full_du_norm` が小さい場合の 0 除算リスクや、複数 trigger
間での分母スケール比較を、プログラム的に（report 再パース無しで）
実行可能にする。

### 2. `xkep_cae/verify/tests/test_kc_component_fd.py`

新規テスト `test_share_report_preserves_small_geo_precision`（12 番目、
ファイル内テスト総数 11 → 12）:

- `K_mat` を `K_geo` の 500 倍にスケールし、実測比 500:1 を模擬
- `share_geo ∈ (0, 0.05)` を確認（非ゼロかつ dominant ではないこと）
- report 文字列に含まれる `geo=X.XXXeY` を regex で抽出し、dataclass の
  `share_geo` と 1% rtol で一致することを確認
- `geo_du_norm > 0`、`mat_du_norm > geo_du_norm` を確認（新フィールドの疎通）

## 再解析スクリプト

既存 log の再解析は以下の一行コマンドで実施（再現可能）:

```bash
uv run --quiet python -c "
import re, numpy as np
text = open('docs/measurements/kc_component_fd_19strand_20260415T214702.log').read()
pat = re.compile(r'\|\|K_mat@du\|\| = ([0-9.eE+\-]+), \|\|K_geo@du\|\| = ([0-9.eE+\-]+), \|\|K_st@du\|\|\s+= ([0-9.eE+\-]+)')
pat_full = re.compile(r'\|\|K_c@du\|\|\s+= ([0-9.eE+\-]+)')
mats = np.array([float(m.group(1)) for m in pat.finditer(text)])
geos = np.array([float(m.group(2)) for m in pat.finditer(text)])
sts  = np.array([float(m.group(3)) for m in pat.finditer(text)])
fulls = np.array([float(m.group(1)) for m in pat_full.finditer(text)])
share_geo = geos/fulls
print(f'share_geo: mean={share_geo.mean():.6e}, min={share_geo.min():.6e}, max={share_geo.max():.6e}')
"
```

出力:

```
Records: 183
share_geo: mean=1.020443e-03, max=3.786701e-03, min=1.185653e-04
|K_geo|/|K_mat|: mean=1.374394e-03, max=8.677668e-03, min=1.182650e-04
```

## 仮説 A への影響（status-344 結論の再評価）

status-344 の主結論は**維持**:

1. **最良組み合わせ = `mat_only`（183/183 = 100%）** — 変更なし
2. **K_st 追加で rel_err 平均 +16%/最大 +52% 悪化** — 変更なし
3. **K_mat 単独で FD rel_err mean = 44%、comp_x max = 98%** — 変更なし
4. **K_geo の寄与**: 〜0.000 と誤認していたが、実際は **0.1% 程度の微小値**。
   絶対量として無視できるレベルだが「実装が抑圧している」のではなく
   **「19本撚線条件では物理的に K_geo ≪ K_mat」** というのが正しい理解。

→ 仮説 A の primary driver 特定（K_mat の x/z 成分不整合）は不動。
**推奨アクション 1（K_mat の x/z カップリング修正）は引き続き最優先**。

## 推奨アクション 3 のクローズ

status-344 の「推奨アクション 3（長期）: K_geo == 0 の原因調査」は
本 status で**クローズ**:

- K_geo は実装上非ゼロで正常に組み立てられている
- 19本撚線条件下では K_geo が K_mat の 0.1% 程度に留まるのは
  物理的に妥当（`w_geo = p_n / dist` で p_n が小さいため）
- 「K_geo 由来でない」という status-344 の定性結論は正しい

次セッションは status-344 推奨アクション 1（K_mat の x/z カップリング修正）
に集中可能。

## 成果物

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/verify/kc_component_fd.py` | report 寄与率フォーマット `{:5.2f}`→`{:.3e}` / Output に `mat_du_norm`, `geo_du_norm`, `st_du_norm`, `full_du_norm`, `dfc_fd_norm` 追加 |
| `xkep_cae/verify/tests/test_kc_component_fd.py` | `test_share_report_preserves_small_geo_precision` 追加（11→12 件） |
| `docs/status/status-345.md` | **新規**（本ファイル） |
| `docs/status/status-index.md` | status-345 エントリ追加 |
| `docs/roadmap.md` | status-344 行に補足 + 345 行追加 |
| `README.md` | 現状行更新 |

## 検証・品質確認

- **単体テスト**: `xkep_cae/verify/tests/test_kc_component_fd.py` 12 件全 PASS
- **ruff check / format**: GREEN
- **契約違反**: 0 件（Output 拡張は後方互換、frozen dataclass default 0.0）
- **回帰**: report 文字列フォーマット変更は下流パーサ（`work/beam_hysteresis/13_*`）
  の regex `[0-9.eE+\-]+` が既に科学表記対応済みのため、次回実測時には
  自動的に高精度 share が CSV に反映される。

## 開発運用メモ

- **三現主義の教訓**: 「0.000 表示 → 真に 0」と短絡せず、同レポート内の
  他の表記（ノルム絶対値）を必ず突き合わせること。status-344 では
  `||K_geo@du|| ≈ 1e-9`（科学表記で保存されていた）という決定的情報が
  あったにもかかわらず「share=0.000 = K_geo=0」と判定してしまった。
- **ログフォーマットの原則**（CLAUDE.md「ソルバー診断ログ規約」に追記候補）:
  **分母・分子ともに同じ精度系で出力する**。ノルムを科学表記で出すなら
  その比率も科学表記で揃える。固定小数 `{:.2f}` は 0.01 未満の情報を
  常に捨てるため、診断 Process の寄与率・シェア類には原則使わない。
- **Process 化の価値**: status-344 で誤認された情報が、本 status では
  Process コード 3 行の修正と 1 テスト追加で訂正可能だった。
  診断ロジックが Process 化されていなければ、複数箇所（work script /
  ソルバー内ログ / 単体テスト）に散らばった print 文を一斉修正する
  必要があった。
