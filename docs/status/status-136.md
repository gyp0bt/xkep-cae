# status-136: 初期貫入オフセット手法の物理的妥当性問題（要やり直し）

[← README](../../README.md) | [← status-index](status-index.md)

**日付**: 2026-03-08
**テスト数**: 2271（変更なし）

---

## 概要

現在の初期貫入補正（`gap_offset`）は **LS-DYNA IGNORE=1 相当のオフセット手法** であり、
被膜が丸ごと貫入しているケースでは **物理的に全く妥当でない** ことを確認。
次回TODOとして、この仕組みの根本的なやり直しを記録する。

---

## 問題の詳細

### 現在の実装

`xkep_cae/contact/pair.py`:

```python
# L482-483: update_geometry() 内
offsets = np.array([p.gap_offset for p in self.pairs])
gap_all = dist_all - (radii_a + radii_b) - offsets

# L668-669: store_initial_offsets() 内
if gap_raw < 0.0:
    pair.gap_offset = gap_raw  # 初期貫入量をそのままオフセットとして保存
```

### 何が起きているか

1. 被膜付き梁の `radius = strand_radius + coating_thickness`
2. メッシュ生成時に隣接素線が被膜分だけ重なる（物理的に正しい初期配置）
3. `store_initial_offsets()` が初期貫入を検出し、`gap_offset = gap_raw` として保存
4. 以後の全ステップで `g_effective = g_raw - gap_offset` → **見かけ上ギャップ = 0**
5. **結果**: 被膜による接触力が完全にゼロ → 被膜の力学的効果が消失

### なぜ物理的に不正か

- 被膜は弾性体であり、圧縮されれば反力を発生する
- オフセットで貫入をゼロにすると「被膜が存在しない」のと同等
- 収束は達成できるが、得られた解に物理的意味がない
- 被膜剛性のパラメータスタディ（status-098, 120）の結果も疑問

### 影響範囲

| ファイル | 行 | 内容 |
|---------|---|------|
| `pair.py` | 124 | `gap_offset: float = 0.0` 変数定義 |
| `pair.py` | 259-261 | `adjust_initial_penetration` 設定 |
| `pair.py` | 482-483 | オフセット適用（コア計算） |
| `pair.py` | 642-674 | `store_initial_offsets()` |
| `solver_ncp.py` | 1724-1763 | オフセット制御フロー |

---

## 正しいアプローチ（次回TODO）

### 案1: 被膜を接触層として陽にモデル化

- ギャップ定義: `g = dist - (r_core_a + r_core_b)` （芯線半径のみ）
- 被膜層: `0 < g < t_coating_a + t_coating_b` の範囲で被膜の弾性構成則を適用
- 接触力: 被膜の弾性変形に基づく反力 `f_n = k_coating * penetration_into_coating`
- 完全貫入: `g < 0` のとき芯線同士の剛体接触（高いペナルティ）

### 案2: 等価接触剛性モデル

- 被膜厚を考慮した等価接触ペナルティ `k_eq = f(E_coating, t_coating, R_strand)`
- Hertz接触理論の被膜層拡張
- Johnson (1985) Contact Mechanics の層状体接触を参照

### 案3: 被膜を別要素として離散化

- 被膜をシェル要素またはソリッド要素でメッシュ化
- 梁要素との結合拘束
- 最も正確だが計算コスト大

### 推奨

**案1** が実装コストと物理的妥当性のバランスが最も良い。
NCP枠組みとの整合性も保てる。

---

## TODO

- [ ] gap_offset手法の完全廃止（LS-DYNA IGNORE=1相当の撤去）
- [ ] 被膜層を陽に考慮した接触ギャップ定義の実装
- [ ] 被膜の弾性構成則に基づく接触力計算
- [ ] 被膜付き撚線テストの物理的妥当性再検証
- [ ] status-098, 120の被膜パラメータスタディの再実施

---

## 前回status

- [status-135](status-135.md): 19本NCP曲げ揺動収束達成 + mortar rollbackバグ修正

---
