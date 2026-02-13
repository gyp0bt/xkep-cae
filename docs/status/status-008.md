# status-008: ロードマップ拡張 — Cosserat rod & 撚線モデル（拡張ファイバー理論）

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-007](./status-007.md)

**日付**: 2026-02-13
**作業者**: Claude Code
**ブランチ**: `claude/execute-status-todos-FbFC8`

---

## 実施内容

### 1. ロードマップに Phase 2.5: Cosserat rod（幾何学的厳密梁）を追加

Timoshenko梁を超える一般的な梁定式化として、Cosserat rod を Phase 2.5 に追加。

#### 定式化の概要

| 変数 | 物理的意味 | 空間 |
|------|-----------|------|
| r(s) | 中心線 | R³ |
| R(s) | 断面回転 | SO(3) |
| ν = Rᵀr' | せん断＋軸伸び | R³ |
| κ = axial(RᵀR') | 曲率＋ねじり | R³ |

#### 位置づけ

- Phase 2（梁要素）の最終ステップとして配置
- 本質的に幾何学的非線形定式化であり、Phase 3 と密接に連携
- Phase 3.2（共回転定式化）の代替としても機能
- Phase 4.6（撚線モデル）の「外側の梁」および「個別素線」の幾何記述の土台

### 2. ロードマップに Phase 4.6: 撚線モデル（拡張ファイバー理論）を追加

ファイバー理論を拡張し、「離散素線モデルの縮約（homogenization/ROM）」として
撚線電線の1Dモデルを定式化する。

#### 通常のファイバー理論との決定的な差異

| 観点 | 通常のファイバー梁 | 撚線の拡張ファイバー |
|------|------------------|---------------------|
| 繊維の走り方 | 軸方向に並行 | ヘリカルに走る（3D） |
| 断面積分 | ε₀ + z·κ で平面仮定 | ヘリックス集合の縮約 |
| 断面自由度 | なし（拘束） | θ_i（撚り位相）が未知量 |
| 接触・摩擦 | なし | 素線間接触＋摩擦散逸 |

#### 内部変数の設計

- `θ_i(s)`: 素線ヘリックス位相（撚り戻り・撚り解き）
- `δ_ij(s)`: 素線間接線方向相対すべり（摩擦散逸の主役）
- `g_ij(s)`: ギャップ（接触開閉）
- stick/slip状態: 離散フラグ or 連続正則化変数
- `α_i(s)`: 素線材料の塑性履歴
- `γ_c(s)`: 被膜せん断変形（最小限）

#### 3段階の実装計画

| レベル | 概要 | 目的 |
|--------|------|------|
| Level 0 | 軸方向のみ＋penalty接触＋正則化Coulomb | 基本ヒステリシスの確認 |
| Level 1 | θ_i 未知量化＋被膜の平均化ばね | 撚り戻り＋摩擦散逸の再現 |
| Level 2 | 素線Cosserat rod化＋接触ペア最適化 | 局所座屈モードの再現 |

### 3. 依存関係図の更新

撚線モデルのクリティカルパスを明確化:

```
Phase 2.5 (Cosserat rod) → Phase 3 → Phase 4.1-4.2 → Phase 4.6 (撚線モデル)
```

### 4. 参考文献の追加

Cosserat rod および撚線力学に関する文献をロードマップに追加:

- Simo (1985): 有限ひずみ梁定式化
- Antman: 非線形弾性理論（Cosserat rod の理論的基盤）
- Costello: ワイヤロープ理論
- Cardou & Jolicoeur (1997): ヘリカルストランドの力学モデル
- Foti & Martinelli (2016): 撚線の曲げヒステリシス解析

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `docs/roadmap.md` | 大幅拡張 — Phase 2.5, Phase 4.6 追加、依存関係図更新、参考文献追加 |
| `docs/status/status-008.md` | **新規** — 本ステータス |
| `README.md` | 更新 — ドキュメントリンク・現在の状態を更新 |

---

## 設計上の考慮事項

### Cosserat rod の実装戦略

1. **Phase 3 との関係**: Cosserat rod は本質的に幾何学的非線形定式化。
   Phase 2.5 で定式化を導入し、Phase 3 で非線形ソルバーと結合するのが自然な流れ。
   ただし、小変形近似版を先に作り、線形テストを通してから非線形に進む戦略も有効。

2. **SO(3) のパラメトライゼーション**: 四元数 vs 回転ベクトル。
   - 四元数: 特異点なし、正規化制約が必要
   - 回転ベクトル: 直感的、π 近傍で特異（Rodrigues公式）
   - 実装の簡便さを考慮し、まず回転ベクトルで始め、必要に応じて四元数に移行。

### 撚線モデルの設計判断

1. **状態変数の選定がモデルの生死を決める**: 曲げ主目的なら θ_i をどう縮約するか、
   ねじり疲労なら δ_ij をどう統計化するか。用途を決めてから最小モデルを設計する。

2. **接触ペアの爆発回避**: Level 0 では隣接ペアのみ。Level 1 以降で代表接触
   または連続平均化を導入。全ペアを持つのは現実的でない。

3. **摩擦の数値手法**: 増分ポテンシャル（incremental potential）を前提とし、
   rate-independent + 非滑らか最適化 or 正則化（tanh）で滑らかにして Newton で回す。
   Coulomb をそのまま入れると非滑らかで収束しない。

4. **被膜モデルの範囲**: 剛性寄与＋摩擦制御目的で弾性ばねとする。
   温度依存性・粘弾性・損傷を後から入れる可能性を考慮し、インタフェースを設計。

---

## 設計判断（本セッションで確定）

ユーザーの回答に基づき、以下の設計判断をロードマップに反映済み。

### 1. 第一ターゲット用途：曲げ（＋ねじり連成曲げ・疲労）

- **曲げ**が最優先。ねじりが加わるような曲げ（ねじり-曲げ連成）と、その時の疲労も対象。
- **θ_i の縮約**が最重要設計課題。曲げ時の素線滑り挙動を正確に捉える必要がある。
- **δ_ij のサイクルカウント**（雨流計数法等）による疲労評価基盤を Level 0 で作る。

### 2. 被膜モデル：剛性寄与＋摩擦制御（理想化弾性体）

- 被膜は弾性ばね（せん断剛性 G_c, 圧縮剛性 K_c）でモデル化。
- 素線間の摩擦係数 μ を実効的に変化させる機構として機能。
- 温度・損傷は当面スコープ外だが、インタフェースは拡張可能に設計。

### 3. 接触ペア戦略：隣接のみ＋Nイテレーション更新（実験的）

- 初期ペア: 撚り構造から幾何的に隣接する素線ペアを自動生成。
- N_update イテレーションごとにペアリストを再評価。
- g_ij < g_threshold の素線ペアを活性ペアとして保持。
- **実験的アプローチ**: N_update, g_threshold のチューニングが必要。
  大変形時のペア更新遅延→力の不連続リスクあり。

---

## TODO（次回以降の作業）

### 短期（Phase 2 残り）

- [ ] Phase 2.3: Timoshenko梁（3D空間）の実装
- [ ] Phase 2.4: 断面モデルの拡張（一般断面）
- [ ] SCF（スレンダネス補償係数）のオプション実装検討
- [ ] Abaqus .inp パーサーへの `*TRANSVERSE SHEAR STIFFNESS` サポート追加

### 中期（Phase 2.5 / Phase 3）

- [ ] Phase 2.5: Cosserat rod の設計仕様書作成
- [ ] SO(3) 回転パラメトライゼーションの選定と実装
- [ ] Cosserat rod の線形化バージョン（テスト用）
- [ ] Phase 3: Newton-Raphson ソルバーフレームワーク

### 長期（Phase 4.6 撚線モデル）

- [ ] θ_i 縮約の具体的手法検討（曲げ主目的に最適化）
- [ ] δ_ij 追跡＋サイクルカウント基盤の設計
- [ ] Level 0: 軸方向素線＋penalty接触＋正則化Coulomb＋δ_ij追跡
- [ ] Level 1: θ_i 未知量化＋被膜弾性ばね（G_c, K_c）
- [ ] Level 2: 素線Cosserat rod化＋接触ペア動的更新

### 残存する不確定事項

- [ ] 接触ペア更新頻度 N_update の適切な値（数値実験で決定）
- [ ] 接触活性化閾値 g_threshold の設定方針
- [ ] 大変形時のペア更新に伴う力の不連続の許容度
- [ ] 疲労評価のサイクルカウント手法（雨流計数法？他？）

---

## 参考文献

### Cosserat rod

- Simo, J.C. (1985) "A finite strain beam formulation — Part I", CMAME
- Crisfield, M.A. & Jelenić, G. (1999) "Objectivity of strain measures in the geometrically exact beam", IJNME
- Antman, S.S. "Nonlinear Problems of Elasticity", Springer

### 撚線力学

- Costello, G.A. "Theory of Wire Rope", Springer
- Cardou, A. & Jolicoeur, C. (1997) "Mechanical Models of Helical Strands", Applied Mechanics Reviews
- Jiang, W.G. et al. (2006) "Statically indeterminate contacts in axially loaded wire strand", IJNME
- Foti, F. & Martinelli, L. (2016) "An analytical approach to model the hysteretic bending behavior of spiral strands", J. Mech. Phys. Solids
