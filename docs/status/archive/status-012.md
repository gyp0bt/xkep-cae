# status-012: 数値試験フレームワーク（Phase 2.6）実装

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-011](./status-011.md)

**日付**: 2026-02-13
**作業者**: Claude Code
**ブランチ**: `claude/add-frequency-response-testing-115AB`

---

## 実施内容

status-011 の短期TODOリスト（Phase 2.6 数値試験フレームワーク）を一括実装。
追加指示として周波数応答試験、CSV出力、Abaqusライクテキスト入力を実装。

### 1. 数値試験フレームワーク（`xkep_cae/numerical_tests/`）

新規パッケージとして以下の5モジュールを作成:

| ファイル | 概要 |
|---------|------|
| `__init__.py` | 公開API一覧 |
| `core.py` | データクラス（`NumericalTestConfig`, `StaticTestResult`, `FrequencyResponseConfig`, `FrequencyResponseResult`）、メッシュ生成、解析解、摩擦影響評価 |
| `runner.py` | 静的試験ランナー（3点曲げ・4点曲げ・引張・ねん回）、一括/部分実行API |
| `frequency.py` | 周波数応答試験（整合質量行列、動的剛性行列、FRF計算） |
| `csv_export.py` | CSV出力（静的試験・周波数応答試験） |
| `inp_input.py` | Abaqusライクテキスト入力パーサー |

### 2. 静的試験（4種）

| 試験種別 | 略称 | 荷重条件 | 解析解 | 梁タイプ |
|---------|------|---------|--------|---------|
| **3点曲げ試験** | `bend3p` | 中央集中荷重、両端単純支持 | δ = PL³/(48EI) + PL/(4κGA) | eb2d, timo2d, timo3d |
| **4点曲げ試験** | `bend4p` | 2点対称荷重、両端単純支持 | δ = Pa(3L²-4a²)/(24EI) + Pa/(κGA) | eb2d, timo2d, timo3d |
| **引張試験** | `tensile` | 一端固定、他端軸方向荷重 | δ = PL/(EA) | eb2d, timo2d, timo3d |
| **ねん回試験** | `torsion` | 一端固定、他端ねじりモーメント | θ = TL/(GJ) | timo3d のみ |

### 3. 3点/4点曲げ試験における摩擦滑りの実用的判定

支持治具表面の摩擦滑りの影響を `assess_friction_effect()` 関数で評価。

**物理的背景:**
梁の撓みに伴い支持点で水平方向の変位が生じる。摩擦が大きい場合は
pin支持（水平拘束）、小さい場合はroller支持（水平自由）に近い。
線形微小変形解析では、この差は O(δ²/L²) のオーダー。

**判定基準:**

| スパン比 L/h | 摩擦影響 | 判定 |
|-------------|---------|------|
| > 10 | 無視可能（<1%） | roller/pin どちらでも可 |
| 4 〜 10 | 軽微（1-5%） | roller推奨、両方で検証を推奨 |
| ≤ 4 | 顕著 | 曲げ試験としての妥当性に警告 |

**実用的結論:** 通常の試験条件（L/h > 4）ではroller支持モデルで十分実用的。
短スパン梁では水平拘束によるアーチ効果で見かけ剛性が上昇するが、
`support_condition="pin"` オプションで比較可能。

### 4. 周波数応答試験（新規追加）

片端保持（カンチレバー）のもう片端への変位付加/加速度付加による FRF 計算。

#### 整合質量行列

2D/3D梁の整合質量行列（consistent mass matrix）を実装:
- `_beam2d_consistent_mass_local()`: 6×6 局所質量行列
- `_beam3d_consistent_mass_local()`: 12×12 局所質量行列
- 座標変換 → 全体質量行列アセンブリ

#### 周波数応答計算

```
動的剛性行列: K_dyn = K - ω²M + iωC  (C = αM + βK: Rayleigh減衰)
```

**変位励起モード:**
- 自由端の特定DOFに単位変位を付加
- 縮約系 K_rr * u_r = -K_rp * u_p を周波数ごとに解く
- 伝達関数 H(ω) = u_response / u_excitation

**加速度励起モード:**
- 自由端に単位加速度相当の慣性力を付加
- F = M列 × 1.0（単位加速度）
- 伝達関数 H(ω) = u_response / a_input

#### 出力
- 伝達関数（複素数）、振幅、位相
- ピーク検出による推定固有振動数

### 5. CSV出力

各試験結果を3種のCSVファイルに出力:

| 試験 | CSV種別 | 内容 |
|------|---------|------|
| 静的 | `*_summary.csv` | パラメータ、最大変位、解析解、相対誤差 |
| 静的 | `*_nodal_disp.csv` | 全節点の座標・変位 |
| 静的 | `*_element_forces.csv` | 各要素両端の断面力 |
| FRF | `*_summary.csv` | パラメータ、推定固有振動数 |
| FRF | `*_frf.csv` | 周波数、伝達関数（実部/虚部/振幅/位相） |

`output_dir=None` で文字列として返却、パス指定でファイル出力。

### 6. Abaqusライクテキスト入力

```
*TEST, TYPE=BEND3P
*BEAM SECTION, SECTION=RECT
 10.0, 20.0
*MATERIAL
*ELASTIC
 200000.0, 0.3
*SPECIMEN
 100.0, 10
*LOAD
 1000.0
*SUPPORT, TYPE=ROLLER
```

サポートキーワード:
`*TEST`, `*BEAM SECTION`, `*MATERIAL`, `*ELASTIC`, `*DENSITY`,
`*SPECIMEN`, `*LOAD`, `*SUPPORT`, `*FREQUENCY`, `*EXCITATION`, `*DAMPING`, `*BEAM TYPE`

---

## テスト結果

**241 passed, 2 deselected (external)**（前回193 → 48テスト増加）

### テスト増加の内訳

| テストクラス | テスト数 | 内容 |
|------------|---------|------|
| `TestMeshGeneration` | 4 | メッシュ生成の正当性検証 |
| `TestAnalyticalSolutions` | 5 | 解析解の計算精度検証 |
| `TestFrictionAssessment` | 4 | 摩擦影響判定ロジック |
| `TestBend3p` | 5 | 3点曲げ: eb2d, timo2d, timo3d + 断面力・摩擦 |
| `TestBend4p` | 3 | 4点曲げ: eb2d, timo2d, timo3d |
| `TestTensile` | 3 | 引張: eb2d, timo2d, timo3d |
| `TestTorsion` | 2 | ねん回: 解析解一致 + 2Dバリデーション |
| `TestRunAPI` | 2 | 一括/部分実行API |
| `TestFrequencyResponse` | 5 | 周波数応答: 変位/加速度励起 × 2D/3D + EB |
| `TestCSVExport` | 4 | CSV出力（文字列/ファイル × 静的/FRF） |
| `TestInpInput` | 6 | Abaqusライク入力パーサー |
| `TestCircularSection` | 2 | 円形断面での試験 |
| `TestValidation` | 3 | バリデーション（エラーケース） |

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/numerical_tests/__init__.py` | **新規** — パッケージ初期化・公開API |
| `xkep_cae/numerical_tests/core.py` | **新規** — データクラス、メッシュ生成、解析解、摩擦評価 |
| `xkep_cae/numerical_tests/runner.py` | **新規** — 静的試験ランナー（4種） |
| `xkep_cae/numerical_tests/frequency.py` | **新規** — 周波数応答試験（質量行列・FRF） |
| `xkep_cae/numerical_tests/csv_export.py` | **新規** — CSV出力 |
| `xkep_cae/numerical_tests/inp_input.py` | **新規** — Abaqusライクテキスト入力 |
| `tests/test_numerical_tests.py` | **新規** — 48テスト |
| `docs/status/status-012.md` | **新規** — 本ステータス |
| `docs/roadmap.md` | 更新 — Phase 2.6 実装完了マーク |
| `README.md` | 更新 — 現在の状態・リンク・使用例更新 |

---

## TODO（次回以降の作業）

### 短期（Phase 2 残り）

- [ ] Phase 2.5: Cosserat rod の設計仕様書作成
- [ ] 数値試験の pytest マーカー対応（`-m bend3p`, `-m freq_response` 等）
- [ ] 周波数応答試験の固有振動数の解析解との比較検証
- [ ] 非一様メッシュ（荷重点周辺の細分割）サポート

### 中期（Phase 2.5 / Phase 3）

- [ ] SO(3) 回転パラメトライゼーションの選定と実装
- [ ] Cosserat rod の線形化バージョン（テスト用）
- [ ] Phase 3: Newton-Raphson ソルバーフレームワーク
- [ ] 数値試験フレームワークの非線形対応（荷重増分）

### 長期（Phase 4.6 撚線モデル）

- [ ] θ_i 縮約の具体的手法検討（曲げ主目的に最適化）
- [ ] δ_ij 追跡＋サイクルカウント基盤の設計
- [ ] Level 0: 軸方向素線＋penalty接触＋正則化Coulomb＋δ_ij追跡

### 残存する不確定事項

- [ ] 接触ペア更新頻度 N_update の適切な値（数値実験で決定）
- [ ] 接触活性化閾値 g_threshold の設定方針
- [ ] 大変形時のペア更新に伴う力の不連続の許容度
- [ ] 疲労評価のサイクルカウント手法（雨流計数法？他？）
