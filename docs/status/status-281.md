# status-281: ヘリカル素線接触なし90度曲げ完走 — 静的NRソルバー実装

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

- **日時**: 2026-04-02
- **ブランチ**: `claude/helical-wire-90-bend-aljSm`
- **テスト数**: 602 + 6 = 608 passed（+6: _collect_adjacent_nodes + static_solver config テスト）
- **契約違反**: **0件**
- **条例違反**: 0件

---

## 概要

7本ヘリカル素線の接触なし90度曲げ（θ_y=π/2処方）を**frac=1.0完走**した。

**根本原因の特定と解決**:
- 動的ソルバー（Generalized-α）では、NR残差が `||R_t||/||f|| ~ 1e-5` で停滞し、dt_min到達で停止
- 原因: ContactFrictionProcessの既存NRソルバーが**全累積変位**をULアセンブラに渡していた（TL的な使い方）
- 解決: 接触なし問題用の**静的NRソルバー（UL増分形）**を実装し、各収束後にupdate_referenceで参照配置を更新

**結果**: 7本ヘリカル × 90度曲げ = frac=1.0、40インクリメント、カットバック0回、各ステップ11NR反復

---

## 実装内容

### 1. 静的NRソルバー `_static_nr_solve()`

| 項目 | 動的ソルバー（従来） | 静的NRソルバー（新規） |
|------|---------------------|----------------------|
| 時間積分 | Generalized-α | なし（純静的） |
| 慣性項 | M*a + C*v あり | なし |
| 参照配置更新 | ソルバー内部で管理 | 明示的にupdate_reference() |
| 変位引数 | 全累積変位 | **UL増分変位** |
| 収束判定 | 並進残差のみ | 並進残差（初回残差基準） |
| 接触 | あり | なし（接触なし専用） |
| カットバック | 適応的dt | 適応的dt_frac |

### 2. Config追加

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `static_solver` | `False` | 静的NRソルバー使用フラグ |
| `loading_mode` | `"rotation"` | `"rotation"`:θ_y処方、`"moment"`:力カップルM_y荷重 |

### 3. 力カップル方式（loading_mode='moment'）

動的ソルバーの収束判定は並進残差のみ。純モーメント荷重では偽収束する。
対策として端部2ノードに逆向き力(F,-F)で力カップルを構成（M_y = F_x * Δz）。
※ 動的ソルバーでは力カップルでも停止（frac=0.063）。静的ソルバーとの組み合わせで使用。

### 4. 隣接ノード収集 `_collect_adjacent_nodes()`

力カップル方式で使用。端部ノードの1要素内側のノードを返す。

### 変更ファイル

| ファイル | 変更 |
|----------|------|
| `xkep_cae/numerical_tests/strand_bending_oscillation.py` | `_static_nr_solve()`, `_collect_adjacent_nodes()`, Config拡張, `_process_free_end`静的ソルバー分岐 |
| `xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py` | +6テスト（隣接ノード, static_solver, loading_mode） |
| `docs/verification/7strand_90deg_bending_nocontact.png` | 2D投影スナップショット |

---

## ベンチマーク結果

### 7本ヘリカル接触なし90度曲げ（κ=π/200, θ=π/2）

| 構成 | frac | incr | cutback | NR/step | 備考 |
|------|------|------|---------|---------|------|
| 動的ソルバー rotation (status-280) | 0.065 | 65 | 2 | 11-50 | ||R_t|| stall |
| 動的ソルバー moment | 0.063 | 27 | 1 | 10-45 | 同上 |
| **静的NR rotation** | **1.000** | **40** | **0** | **11** | **完走** |

### 先端変位（エラスティカ理論比較）

| 素線 | u_x [mm] | u_z [mm] | θ_y [deg] | 理論u_x | 理論u_z |
|------|----------|----------|-----------|---------|---------|
| center (s0) | 63.69 | -36.31 | 90.0 | 63.66 | -36.34 |
| outer (s1) | 62.59 | -37.41 | 90.0 | - | - |
| outer (s4) | 64.79 | -35.21 | 90.0 | - | - |

中心素線の先端変位が理論値と0.04%一致。外層素線はヘリカルオフセットにより少し異なる。

---

## 物理的考察

### なぜ静的NRソルバーが動的ソルバーより優れるか

1. **UL増分形の正しい使用**: 各収束後にupdate_reference()で参照配置を更新。NR中の増分変位は小さい(θ≈2.25°/step)ため、接線剛性が正確
2. **慣性項の排除**: M*a項がないため、回転DOFの条件数が改善。動的ソルバーではlumped質量の回転慣性~10^-7が回転残差の収束を阻害
3. **収束判定の適切化**: 初回残差を基準とした相対収束。外力がゼロ（変位処方）でも適切に動作

### 動的ソルバー停滞の根本原因

動的ソルバーでは `assemble_internal_force(u)` に**累積変位**を渡していた。ULアセンブラでは増分変位が期待されるため:
- 大変形（累積θ > 5°）で接線剛性の精度が低下
- NRが二次収束でなく線形収束（rate 0.88-0.99）に劣化
- `||R_t||/||f|| ~ 1e-5` で停滞し、2サイクル検知 → dt縮小 → dt_min到達

---

## 再現手順

```bash
git checkout claude/helical-wire-90-bend-aljSm
pip install -e .

# 7本ヘリカル接触なし90度曲げ（~30秒）
python -c "
from xkep_cae.numerical_tests.strand_bending_oscillation import *
import math
cfg = StrandBendingOscillationConfig(
    n_strands=7, wire_radius=0.5, pitch_length=100.0,
    n_elements_per_pitch=16, n_pitches=1.0,
    E=130.0e3, nu=0.3, rho=8.96e-9,
    bending_curvature=math.pi/200.0, n_cycles=1,
    n_increments_per_cycle=40,
    max_nr_attempts=50, tol_force=1e-8,
    free_end_mode=True, contact_enabled=False,
    loading_mode='rotation', static_solver=True,
)
result = StrandBendingOscillationProcess().process(cfg)
sr = result.solver_result
frac = sr.load_history[-1] if sr.load_history else 0.0
print(f'frac={frac:.4f}, incr={sr.n_increments}, cutback={sr.n_cutbacks}')
" 2>&1 | tee /tmp/log-7strand-90deg.log
# 期待値: frac=1.0000, incr=40, cutback=0

# テスト
python -m pytest xkep_cae/numerical_tests/tests/test_strand_bending_oscillation.py -v
# 期待値: 17 passed

# 契約検証
python contracts/validate_process_contracts.py
```

---

## STA2 準拠チェック

- [x] **tee ログ保存**: `/tmp/log-7strand-*` 各テスト実行時
- [x] **再現手順記載**: 全コマンドをstatusに記載
- [x] **数値の捏造なし**: ベースラインfrac=0.065（status-280）と比較
- [x] **物理検証**: 2D投影スナップショットでエラスティカ理論と目視確認

---

## TODO

- [x] 7本接触なし90度曲げ完走（static_solver=True）
- [x] 2D投影スナップショット作成
- [ ] 接触あり90度曲げの試行（静的NRソルバーに接触力組み込み）
- [ ] 動的ソルバーのUL増分変位受け渡し修正（ContactFrictionProcess）
- [ ] evaluate/tangent dm整合性回復（status-277 推奨手順）

---
