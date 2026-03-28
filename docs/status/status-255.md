# status-255: MPC縮退系残差判定 + C2-C3幾何計算Process化

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-03-28
- **ブランチ**: `claude/execute-status-todos-8wu8d`
- **テスト数**: 200+10s+16+3+23+1+6（新規6件: C2×2 + C3×4）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実施内容

### 1. MPC縮退系での収束判定（候補C）

**問題**: slave DOFの制約反力がR_uに含まれ、力収束判定が不可能だった。

**修正**: `ConvergenceCheckProcess`に`mpc_transform`を追加。R_red = T^T R_uで収束判定し、slave DOFの反力を自然に消去。独立DOFの並進/回転分離もMPC対応。

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/_newton_steps.py` | `ConvergenceCheckInput`に`mpc_transform`追加、`process()`でR_red判定 |
| `xkep_cae/contact/solver/_newton_dynamic.py` | 2箇所の`ConvergenceCheckInput`にmpc_transform伝搬 |

### 2. Time integrator MPC対応（候補B）

**問題**: `predict()`で保存される`_u_pred`がMPC非整合のため、`correct()`の`acc = c0*(u - u_pred)`でslave DOFの加速度が不正になる。

**修正**: `process.py`のMPC射影箇所で`_time_strategy._u_pred`にも同一射影を適用。

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver/process.py` | MPC射影後にu_predもMPC射影（predict整合性） |

### 3. ストール検知拡張

**発見**: active setが安定（振動なし）でもNR残差が停滞するパターンを発見。従来のチャタリング検知は`active_changed`を要求していたため、このパターンを検知できなかった。

**修正**: 残差停滞 < 5%の条件のみでstall検知。active set振動/安定の区別はログメッセージに反映。

### 4. 接触力のMPC整合性（候補A）

**調査結果**: `LinearSolveProcess._solve_with_mpc()`内のT^T K T / T^T R_uで自然に実現済み。追加実装不要。

### 5. C2-C3 幾何計算Process化

| ID | 変換前 | 変換後 | テスト |
|----|--------|--------|--------|
| C3 | `_build_contact_frame_batch` (108行) | `ContactFrameProcess` | 4件（empty, single, batch, transport） |
| C2 | `_batch_update_geometry` (175行) | `BatchUpdateGeometryProcess` | 2件（empty, returns_updated） |

- `_batch_update_geometry`内部のContactFrame呼び出しをProcess経由に変更
- 3つの具象Process（PtP, L2L, Mortar）に`uses=[BatchUpdateGeometryProcess]`追加
- `__init__.py`に6シンボル追加export

---

## 収束テスト結果

### MPC改善後の結果
- **frac到達**: 0.35（status-254と同等）
- **改善点**: ストール検知が早くなり実行時間短縮（24s→14s）
- **壁の本質**: ||R_t||/||f|| ≈ 0.85-1.05で完全停滞。NR更新方向が残差を減少させない
- active set安定（6ペア一定）でも停滞 → 接線剛性とMPCの不整合が原因の可能性

### 診断
- frac=0.35付近で1ペアのみ真にactive（pair 112: gap=-0.2e-3, p_n=6.3e-3）
- 接触ペアの一方の要素（elem 102）が strand 6 の端部要素 → slave DOF関与
- T^T K_c T は正しく接触剛性を縮退系に変換しているはず
- 50反復で残差が全く減少しない → du ≈ 0（NR方向が無効）

---

## テスト結果

- 新規テスト: 6件（C2×2 + C3×4）
- 既存テスト: 534 passed, 20 skipped, 1 xfailed（回帰なし）
- 契約違反: 0件
- 条例違反: 0件
- lint: 全合格

---

## 再現手順

```bash
git checkout claude/execute-status-todos-8wu8d
pip install -e .
# 高速テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"
# 収束テスト（slow、約14秒）
python -m pytest tests/numerical_tests/test_strand_bending_convergence.py -x -s -v --timeout=600 2>&1 | tee /tmp/log-strand-bending.log
# 契約検証
python contracts/validate_process_contracts.py
# lint
ruff check xkep_cae/ tests/
ruff format --check xkep_cae/ tests/
```

---

## 次セッションへの引き継ぎ

### 優先TODO
1. **MPC+接触の接線剛性不整合の診断** — frac=0.35の壁
   - FD接線チェック: 不収束ステップでK_Tの整合性をFDで検証
   - 接触ペアがslave DOF（端部要素）を含む場合のK_cの挙動を確認
   - 候補: K_c のslave DOF成分をmaster DOFに事前集約してからK_Tに加算
2. **B1-B4 摩擦アセンブリProcess化** — C2-C3完了に続くPhase D後半

### STA2 tolerance 厳格化（status-252引き継ぎ）
3. **T1 Hermite atol → 1e-5** → frozen-m完全解消後
4. **T2 beam oscillation rtol → 0.02** → 要素数≥40時

---

## 懸念・設計メモ

1. **NR方向無効化の根本原因**: 50反復で残差が全く減少しないのは、K_T du = -R_u のduが残差を全く改善しないことを意味する。T^T K T の縮退系で解いた du_red がT展開後に残差を減少させない。接触ペアがslave DOFを含む場合、T変換後の接線が不整合の可能性がある。
2. **FD診断の提案**: 不収束ステップでR(u + eps*du)/R(u)を数値的にチェックし、接線方向の有効性を検証すべき。
3. **C2-C3 Process化の効果**: 幾何計算がProcess管理下に入り、プロファイリングとトレーサビリティが向上。機能変更なし。
