# status-279: チェックポイント途中再開 + NR収束改善トライ

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-02
- **ブランチ**: `claude/load-bending-oscillation-convergence-rajA9`
- **テスト数**: 600 passed, 0 failed
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 実装内容

### 1. チェックポイント途中再開パイプライン

status-278で保存のみ実装されていたチェックポイント復元を完全実装。

| 変更 | ファイル | 内容 |
|------|----------|------|
| `load_frac_start` フィールド追加 | `xkep_cae/core/data.py` | ContactFrictionInputData に load_frac_start: float = 0.0 追加 |
| ソルバー途中再開 | `xkep_cae/contact/solver/process.py` | state.load_frac_prev と stepping キューを frac_start から初期化 |
| 復元パス修正 | `xkep_cae/numerical_tests/strand_bending_oscillation.py` | load_frac_start設定 + ULアセンブラ累積変位復元 |

**使用方法**:
```python
cfg = StrandBendingOscillationConfig(
    ...,
    resume_checkpoint='/tmp/ckpt_7wire_frac05.pkl',
)
```

### 2. NR収束改善トライ（結果: 改善なし）

#### 試行A: N-サイクルリミットサイクル検知
- 2サイクル検知（status-278）を周期2〜6のN-サイクル検知に拡張
- **結果**: 周期3以上の振動を「収束」判定すると不正確な状態が蓄積し悪化
- **結論**: 微小接触フィルタは周期2のみに限定（status-278互換）

#### 試行B: 後期リミットサイクル判定（||R_t||/||f|| < 2.0〜3.0）
- att>=15で残差が制限内なら任意周期の振動を収束判定
- **結果**: 残差の大きい状態を受け入れると次インクリメントで発散
- **結論**: 逆効果で無効化

#### 試行C: ステートリスタート（自動チェックポイント復元）
- NR不収束時にvel/acc/pairsをリセットして同じfracから再試行
- **結果**: vel/accリセットが不十分、接触ペアクリアだけでは不十分
- **結論**: 逆効果で削除

### 3. ul_frac_base処方変位バグ発見・修正

**重大バグ**: load_frac_start設定時に `ul_frac_base = frac_start` とすると、
処方変位が `(load_frac - frac_start) * prescribed_values` に変わり、
曲げがリセットされる。

- frac=0.8復元が「完走」していたのはこのバグが原因（曲げがリセットされactive=0で進行）
- 修正: ul_frac_base=0.0のまま維持（動的解析ではUL更新しない）

---

## ベンチマーク結果

| テスト | frac | incr | cutback | elapsed |
|--------|------|------|---------|---------|
| ゼロスタート（修正後） | **0.5543** | 485 | 3 | 1038s |
| チェックポイント復元（frac=0.5→） | 0.5160 | 18 | 3 | 118s |
| status-278ベースライン | ~0.55 | - | - | - |

**結論**: ゼロスタートのベースラインfrac≈0.55は維持。チェックポイント復元は動作するが、NR停滞の壁を突破する効果はない。

---

## 発見事項

### N-サイクル検知のインデックスバグ

`_u_history_ring[-1] - _u_history_ring[-period]` で period=2 のとき、
`_u_history_ring[-2]` = u[att-1]（1反復前）を参照していた。
周期2の検知には `_u_history_ring[-3]` = u[att-2] が必要。

修正: `_u_history_ring[-(period+1)]` で正しいインデックスを参照。

### リスタート機能が逆効果な理由

手動チェックポイント復元は:
1. StrandBendingOscillationProcess全体を再構築（メッシュ、MPC、アセンブラ）
2. ContactFrictionProcessの新インスタンスで実行
3. ContactManagerが完全に再構築

内部リスタートでは:
1. strategies/managerのインスタンスが共有
2. vel/accの復元が不完全（NR停滞中の値が残る）
3. ULアセンブラの参照配置が完全にはリセットされない

---

## 再現手順

```bash
git checkout claude/load-bending-oscillation-convergence-rajA9
pip install -e .

# 全テスト
python -m pytest xkep_cae/ tests/ -q --timeout=120 --ignore=tests/contact/test_st_jacobian.py -k "not slow and not stress_contour"

# ゼロスタートベンチマーク（~17分）
python -c "
from xkep_cae.numerical_tests.strand_bending_oscillation import *
cfg = StrandBendingOscillationConfig(
    n_strands=7, wire_radius=0.5, pitch_length=100.0,
    n_elements_per_pitch=16, n_pitches=1.0,
    E=130.0e3, nu=0.3, rho=8.96e-9,
    bending_curvature=0.001, n_cycles=1,
    n_increments_per_cycle=40, rho_inf=0.9, mu=0.15,
    max_nr_attempts=50, tol_force=1e-8, max_increments=10000,
    exclude_same_strand=True,
)
proc = StrandBendingOscillationProcess()
result = proc.process(cfg)
sr = result.solver_result
frac = sr.load_history[-1] if sr.load_history else 0.0
print(f'frac={frac:.4f}')
"
# 期待値: frac≈0.55

# 契約検証
python contracts/validate_process_contracts.py
```

---

## STA2 準拠チェック

- [x] **tee ログ保存**: 全ベンチマーク結果を /tmp/log-*.log に保存
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: ul_frac_baseバグによる虚偽完走を正直に報告
- [x] **ベースライン先行取得**: ゼロスタートfrac=0.5543で回帰なし確認

---

## TODO

- [ ] **NR停滞の根本対策**: K_c/K_struct=10^-4のペナルティ法限界（status-278）
  - evaluate/tangent dm整合性回復（status-277 推奨手順）
  - 接触DOF方向のNR更新増幅スキーム
  - 陽解法スイッチ（リミットサイクル検知時に陽的時間積分で数ステップ進める）— 物理的妥当性の保証が課題
- [ ] 回転残差θ_z単調増加の原因調査（status-278 TODO継続）
- [ ] consistent質量行列への切替検証（status-278 TODO継続）
- [ ] smoothing_deltaの自動推定改善（status-278 TODO継続）

---
