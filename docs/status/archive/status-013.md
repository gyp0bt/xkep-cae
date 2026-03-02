# status-013: Cosserat rod 四元数回転実装（Phase 2.5 前半）

[← README](../../README.md) | [ロードマップ](../roadmap.md) | [前回: status-012](./status-012.md)

**日付**: 2026-02-13
**作業者**: Claude Code
**ブランチ**: `claude/cosserat-quaternion-rotation-WveGe`

---

## 実施内容

status-012 の TODO に基づき、Phase 2.5 の核心部分を実装:
1. 四元数演算モジュール（`xkep_cae/math/quaternion.py`）
2. Cosserat rod 要素の線形化版（`xkep_cae/elements/beam_cosserat.py`）
3. 設計仕様書（`docs/cosserat-design.md`）

### 設計判断: なぜ四元数か

ユーザー指示に従い、Cosserat rod の回転表現に**四元数**を採用。

| 比較 | 回転ベクトル | 四元数 |
|------|------------|--------|
| 特異点 | π近傍で特異（Rodrigues） | なし |
| 正規化 | 不要 | ||q||=1 の制約 |
| 補間 | Lie群上の補間が必要 | SLERPで自然に球面補間 |
| 増分更新 | exp map の計算 | Hamilton積 1回 |
| Phase 4.6 接触フレーム | body/spatialの切り替えが煩雑 | q ⊗ (0,v) ⊗ q* で統一 |

### 1. 四元数演算モジュール（15関数）

新規パッケージ `xkep_cae/math/` を作成し、四元数の全基本演算を実装。

| 関数 | 説明 |
|------|------|
| `quat_identity()` | 恒等四元数 [1,0,0,0] |
| `quat_multiply(p, q)` | Hamilton積（非可換） |
| `quat_conjugate(q)` | 共役（= 単位四元数の逆） |
| `quat_norm(q)` | ノルム |
| `quat_normalize(q)` | 正規化 |
| `quat_rotate_vector(q, v)` | ベクトル回転（効率的実装） |
| `quat_to_rotation_matrix(q)` | q → R(3×3) |
| `rotation_matrix_to_quat(R)` | R → q（Shepperd法、全4分岐テスト済み） |
| `quat_from_axis_angle(axis, angle)` | 軸-角 → q |
| `quat_from_rotvec(rotvec)` | 回転ベクトル → q（指数写像、テイラー展開付き） |
| `quat_to_rotvec(q)` | q → 回転ベクトル（対数写像） |
| `quat_slerp(q0, q1, t)` | 球面線形補間 |
| `quat_angular_velocity(q, q_dot)` | 角速度（body frame） |
| `quat_material_curvature(q, q_prime)` | 物質曲率 κ = 2·Im(q* ⊗ q') |
| `skew(v)` / `axial(S)` | hat/vee 写像 |

### 2. Cosserat rod 要素（線形化版）

B行列ベースの定式化。1点ガウス求積でせん断ロッキングを回避。

#### 定式化の概要

```
配位:       (r(s), q(s))  — 中心線 + 断面回転（四元数）
一般化歪み: Γ = Rᵀr' - e₁  （力歪み: 軸伸び + せん断）
            κ = 2·Im(q*⊗q') （モーメント歪み: 曲率 + ねじり）
構成則:     C = diag(EA, κy·GA, κz·GA, GJ, EIy, EIz)
剛性:       Ke = ∫₀ᴸ Bᵀ · C · B ds （1点ガウス求積）
```

#### 要素クラス構成

| クラス/関数 | 説明 |
|------------|------|
| `CosseratStrains` | 一般化歪み (Γ, κ) |
| `CosseratForces` | 断面力 (N, Vy, Vz, Mx, My, Mz) |
| `CosseratRod` | ElementProtocol 適合クラス（6DOF/node×2node=12DOF） |
| `cosserat_ke_local()` | 局所剛性行列 |
| `cosserat_ke_global()` | 全体座標系剛性行列 |
| `cosserat_section_forces()` | 断面力計算 |

#### Timoshenko 3D との差異

| 項目 | Timoshenko 3D | Cosserat rod |
|------|--------------|--------------|
| 定式化 | 解析的（Φ補正） | B行列 + ガウス求積 |
| 回転表現 | 回転行列 | 四元数（内部状態） |
| 1要素曲げ精度 | 高い | 低い（要メッシュ細分割） |
| 非線形拡張 | 大幅再設計 | 歪み計算の変更のみ |

### 3. 設計仕様書

`docs/cosserat-design.md` に以下を文書化:
- 数学的定式化（四元数規約、一般化歪み、構成則）
- 有限要素離散化（B行列、ガウス求積）
- 検証結果（収束性）
- Phase 3 / Phase 4.6 への拡張方針

---

## テスト結果

**314 passed, 2 skipped**（前回 241 → 73テスト増加）

### テスト増加の内訳

| テストファイル | テスト数 | 内容 |
|-------------|---------|------|
| `test_quaternion.py` | **37** | 四元数演算の全関数 |
| `test_beam_cosserat.py` | **36** | Cosserat rod 要素 |

### 四元数テスト詳細

| テストクラス | テスト数 | 内容 |
|------------|---------|------|
| `TestQuatBasics` | 5 | 恒等・共役・ノルム・正規化 |
| `TestQuatMultiply` | 5 | Hamilton積（単位元・逆・非可換・結合則） |
| `TestQuatRotation` | 4 | ベクトル回転（恒等・90°・180°・任意） |
| `TestQuatRotationMatrix` | 6 | q↔R変換（ラウンドトリップ・直交性・Shepperd全分岐） |
| `TestQuatAxisAngle` | 3 | 軸-角変換 |
| `TestQuatRotvec` | 6 | 回転ベクトル（exp/log map・ラウンドトリップ・π回転） |
| `TestQuatSlerp` | 3 | 球面線形補間（端点・中点・微小差分） |
| `TestQuatCurvature` | 2 | 物質曲率（ゼロ・一様ねじり） |
| `TestSkewAxial` | 3 | hat/vee写像（外積・逆変換・歪対称性） |

### Cosserat rod テスト詳細

| テストクラス | テスト数 | 内容 |
|------------|---------|------|
| `TestConstitutiveMatrix` | 3 | 構成行列の形状・値・対角性 |
| `TestBMatrix` | 4 | B行列の形状・値・純軸伸び・純ねじり |
| `TestStiffnessMatrix` | 7 | 対称性・正半定値性・軸/ねじり剛性・変換・2GP |
| `TestCantileverAxial` | 2 | 軸引張（1要素厳密 + 10要素） |
| `TestCantileverTorsion` | 2 | ねじり（1要素厳密 + 10要素） |
| `TestCantileverBending` | 2 | 曲げ収束（y方向・z方向、2〜32要素） |
| `TestCantileverCombined` | 1 | 軸+ねじり複合荷重 |
| `TestCoordinateTransform` | 1 | 座標変換（固有値不変性） |
| `TestSectionForces` | 2 | 断面力（軸力・ねじり） |
| `TestCosseratRodClass` | 9 | クラスAPI（属性・DOF・Cowper・歪み・バリデーション） |
| `TestSimplySupportedBeam` | 1 | 3点曲げ収束（4〜64要素） |
| `TestCircularSection` | 2 | 円形断面（軸・ねじり） |

### 検証結果サマリ

**軸力・ねじり**: 1要素で厳密解と一致（相対誤差 < 1e-12）

**曲げ（片持ち梁）メッシュ収束**:

| 要素数 | 相対誤差 |
|--------|---------|
| 2 | ~25% |
| 4 | ~7% |
| 8 | ~1.7% |
| 16 | ~0.4% |
| 32 | < 0.1% |

→ 2次収束。32要素以上で工学的に十分。

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `xkep_cae/math/__init__.py` | **新規** — 数学パッケージ初期化 |
| `xkep_cae/math/quaternion.py` | **新規** — 四元数演算（15関数） |
| `xkep_cae/elements/beam_cosserat.py` | **新規** — Cosserat rod 要素 |
| `tests/test_quaternion.py` | **新規** — 四元数テスト（37テスト） |
| `tests/test_beam_cosserat.py` | **新規** — Cosserat rod テスト（36テスト） |
| `docs/cosserat-design.md` | **新規** — 設計仕様書 |
| `docs/status/status-013.md` | **新規** — 本ステータス |
| `docs/roadmap.md` | 更新 — Phase 2.5 チェックボックス更新 |
| `README.md` | 更新 — 状態・リンク更新 |

---

## TODO（次回以降の作業）

### 短期（Phase 2.5 残り）

- [ ] Cosserat rod の内力ベクトル `internal_force()` 実装（Phase 3 準備）
- [ ] 幾何剛性行列 `geometric_stiffness()` の導出と実装
- [ ] 初期曲率を持つ要素のサポート（ヘリカル構造の基盤）
- [ ] 数値試験フレームワークへの Cosserat rod 統合

### 短期（Phase 2 他）

- [ ] 数値試験の pytest マーカー対応（`-m bend3p`, `-m freq_response` 等）
- [ ] 周波数応答試験の固有振動数の解析解との比較検証
- [ ] 非一様メッシュ（荷重点周辺の細分割）サポート

### 中期（Phase 3: 幾何学的非線形）

- [ ] Newton-Raphson ソルバーフレームワーク
- [ ] 四元数の増分更新ロジック: q_{n+1} = quat_from_rotvec(Δθ) ⊗ q_n
- [ ] Cosserat rod の非線形歪み計算
- [ ] テスト: 大変形片持ち梁（Euler elastica）

### 長期（Phase 4.6 撚線モデル）

- [ ] θ_i 縮約の具体的手法検討（曲げ主目的に最適化）
- [ ] δ_ij 追跡＋サイクルカウント基盤の設計
- [ ] Level 0: 軸方向素線＋penalty接触＋正則化Coulomb＋δ_ij追跡

### 確認事項（ユーザーへの質問）

- [ ] Cosserat rod の非線形版実装は Phase 3 と同時に進めるか、先に線形版のベンチマークを充実させるか
- [ ] 撚線モデルの素線本数（2〜3本ベンチマーク → 7本実用）の優先順位
