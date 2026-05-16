# status-374: 候補 (g3) pair-wise relaxation Phase 1 — `PairwiseFreezingProcess` 単体実装

[← status-index](status-index.md) | [← roadmap](../roadmap.md) | [← README](../../README.md)

**日付**: 2026-04-25
**テスト数**: 459+13+22+5+8+12+12+25+26+10+15+10+9+8+12+33+33+21+8+25+6+12+12+7+10+12 passed（status-373 比 +12、新 freeze テスト群）

## 概要

候補 (g3) pair-wise relaxation の **Phase 1 インフラ**を実装。status-284 で導入された
**全体凍結モード**（NR 全反復で接触力ベクトル全体を snapshot 値に固定）を
**pair granularity** に拡張する `PairwiseFreezingProcess` を新設。
`xkep_cae/contact/freeze/` サブパッケージを新規作成し、status-365
ContactNormalDamping と同じ Phase 1/2 分割パターン（単体 Process + 単体テスト
→ NR 配線 + 実機検証）を踏襲。

**判定**: Phase 1 単体実装は完成。次セッション status-375 で Phase 2 NR 配線
+ 7 本撚線 frac=1.0 維持回帰 + 19 本撚線 90° 曲げで gate `frac ≥ 0.6` 検証へ進む。

## 1. 実装

### 1.1 新ディレクトリ `xkep_cae/contact/freeze/`

| ファイル | 行数 | 役割 |
|----------|------|------|
| `__init__.py` | 47 | サブパッケージ docstring + 公開 API export |
| `strategy.py` | 261 | `PairwiseFreezingProcess` + Input/Output + private ヘルパ純関数 |
| `docs/pairwise_freezing.md` | 159 | 設計仕様書（背景 / 数理 / API / Phase 2 配線案） |
| `tests/__init__.py` | 0 | パッケージマーカ |
| `tests/test_strategy.py` | 197 | 12 単体テスト |

合計 **664 行追加**（実装本体 261 + テスト 197 + docs 159 + パッケージ marker 47）。

### 1.2 `PairwiseFreezingProcess` 公開 API

```python
@dataclass(frozen=True)
class PairwiseFreezingInput:
    n_pairs: int
    pair_active_flip_counts: np.ndarray   # (n_pairs,) int
    is_active_now: np.ndarray              # (n_pairs,) bool
    flip_threshold: int = 3
    chattering_type: str = ""
    skip_when_type_d_dominant: bool = True


@dataclass(frozen=True)
class PairwiseFreezingOutput:
    pair_freeze_flags: np.ndarray   # (n_pairs,) bool
    n_frozen: int
    n_active_pairs: int
    skip_freeze_global: bool
    freeze_reasons: tuple[str, ...]


class PairwiseFreezingProcess(SolverProcess[...]):
    meta = ProcessMeta(name="PairwiseFreezing", module="solve", version="1.0.0",
                       document_path="docs/pairwise_freezing.md")
    def process(self, input_data) -> PairwiseFreezingOutput: ...
```

判定アルゴリズム:

```
skip_global := skip_when_type_d_dominant ∧ _is_type_d_dominant(chattering_type)
freeze[k]   := False   if skip_global or not is_active_now[k]
            := True    if pair_active_flip_counts[k] >= flip_threshold
            := False   otherwise
```

### 1.3 ヘルパ純関数（private、C16 滅菌）

CLAUDE.md「機能は可能な限りprocessクラスとして実装すること」+ 契約 C16
（純粋関数の公開 export 禁止）に従い、ヘルパ 2 本を private 化:

- `_update_pair_active_flips(prev, is_now, is_prev) -> np.ndarray`:
  Phase 2 NR ループ側が各反復先頭で flip カウントをインクリメントする
  ためのヘルパ（shape 不整合時は prev をそのまま返す safety）
- `_is_type_d_dominant(chattering_type) -> bool`:
  `classify_chattering_type` の返り値が "D" 単独支配か判定（"D" 含む∧A/B/E
  全部含まない場合のみ True）

両関数とも `xkep_cae.contact.freeze.strategy` モジュールから直接 import 可能で、
Phase 2 NR ループ配線時にアクセスできるが、`__init__.py` には export しない
ことで C16 違反を回避。

### 1.4 テスト構成（12 テスト）

`xkep_cae/contact/freeze/tests/test_strategy.py`:

| カテゴリ | クラス | テスト | 検証内容 |
|----------|--------|--------|----------|
| API 契約 | `TestPairwiseFreezingProcessAPI` (`@binds_to`) | 3 | n_pairs=0 no-op / shape mismatch raises / Output dtype 整合 |
| 判定ロジック | `TestPairwiseFreezingProcessLogic` | 6 | below_threshold / above_threshold / inactive スキップ / 境界値 / Type D skip / Type D+E は freeze |
| ヘルパ純関数 | `TestPairwiseFreezingHelpers` | 3 | `_update_pair_active_flips` 整合 / shape mismatch safety / `_is_type_d_dominant` 真理値表 |

`@binds_to(PairwiseFreezingProcess)` は API クラスに 1 つだけ付与（C3 1:1
対応制約に準拠）。Logic / Helpers クラスは独立 fixture として実行。

## 2. MCDD 整合性

| 契約 | 状態 |
|------|------|
| C3-C24 全 24 検査 | OK（契約違反 0 件、条例違反 0 件） |
| C16 純パッケージ滅菌 | OK（純関数を private 化、`__init__.py` export なし） |
| `@verified_by` | Phase 2 で検討（凍結は NR 制御で力法則ではないため、`TermExpansionContract` の K_c 5 項に所属しない、ContactNormalDamping と同方針） |

## 3. Phase 2 配線設計（status-375 以降）

```
NR 反復先頭で:
    flip_counts = _update_pair_active_flips(flip_counts_prev,
                                           is_active_now, is_active_prev)
    out = PairwiseFreezingProcess().process(PairwiseFreezingInput(
        n_pairs=len(manager.pairs),
        pair_active_flip_counts=flip_counts,
        is_active_now=is_active_now,
        flip_threshold=cfg.pairwise_freeze_flip_threshold,
        chattering_type=classify_chattering_type(_nr_snap),
        skip_when_type_d_dominant=cfg.pairwise_freeze_skip_type_d,
    ))
    if not out.skip_freeze_global and out.n_frozen > 0:
        # 凍結ペア k の接触力寄与を snapshot 値に差し替え（per-pair 組立）
        for k in np.where(out.pair_freeze_flags)[0]:
            f_c -= per_pair_force_current[k]
            f_c += per_pair_force_snapshot[k]
        # K_c も同様にマスク
```

per-pair 力ベクトル組立は、`HuberContactForceProcess` 出力を pair 単位で
保持する経路の追加が必要（Phase 2 設計）。または既存全体組立後の DOF ブロック
上書き（近似版）から開始する選択肢もあり。

`StrandBendingOscillationConfig` に追加する 3 field（Phase 2）:

- `pairwise_freeze_enabled: bool = False`（既定 OFF、opt-in）
- `pairwise_freeze_flip_threshold: int = 3`
- `pairwise_freeze_skip_type_d: bool = True`

## 4. Gate

| 項目 | 結果 |
|------|------|
| `python contracts/validate_process_contracts.py` | 全 24 検査 OK |
| `pytest xkep_cae/contact/` | **468 passed, 5 skipped**（status-373 比 +12、新 freeze テスト） |
| `pytest xkep_cae/mathematics/` | 109 passed（status-373 維持） |
| `test_helical_3d_hermite` | rel_err=2.18e-07 維持（status-356 機械精度継続） |
| `ruff check xkep_cae/ tests/` | OK（201 files） |
| `ruff format --check xkep_cae/ tests/` | OK |
| 既存実装本体（`_newton_dynamic.py`、`StrandBendingOscillationConfig`、`ContactFrictionProcess`） | **無変更**（Phase 1 は単体実装のみ、Phase 2 で配線） |

## 5. 引継ぎ（status-375 へ）

1. **最優先**: Phase 2 NR 配線実装
   - `ContactFrictionProcess.freeze_slot` 追加（`StrategySlot(PairwiseFreezingProcess,)`、optional）
   - `_newton_dynamic.py` 既存全体凍結ロジック (`_freeze_active`/`_freeze_f_c`) を pair 単位に拡張
   - `StrandBendingOscillationConfig` に 3 field 追加 + 3 経路 plumb-through
   - per-pair 力ベクトル組立: `HuberContactForceProcess` の per-pair 出力経路 or DOF ブロック上書き近似
2. **実機検証**:
   - 7 本撚線 90° 曲げで frac=1.0 維持回帰（既定 OFF でバイト一致が gate）
   - 19 本撚線 90° 曲げで `pairwise_freeze_enabled=True` + flip_threshold ∈ {2,3,5}
     掃引、gate `frac ≥ 0.6`（status-357 baseline 0.3739 の 60% 改善）
3. **却下時**: gate 未達なら**候補 (g2) AL 再導入**（status-221 で凍結した
   Uzawa 外側ループの 1〜2 回限定再導入）に進む
4. **副次**: `solver_mode` フラグ実装（陰解法 default / リスタート opt-in）—
   status-373 §3 設計に従い `StrandBendingOscillationConfig.solver_mode:
   Literal["implicit","restart"]` を追加。リスタート側実装は更に別 status

## 6. 運用所見

- **Phase 1/2 分割の意義**: Phase 1 を solver 配線なしの純計算 Process として
  完結させることで、判定アルゴリズム単体での回帰防止が可能。Phase 2 で配線
  バグが発生しても、Phase 1 のロジックは独立に検証済（status-365/366 と同手順）
- **C16 純関数滅菌の優先順位**: `_is_type_d_dominant` のような小さな
  ユーティリティでも、public export は禁止。private 化 + モジュール直接 import
  でテストアクセスする運用で十分機能する
- **status-284 全体凍結との関係**: 既存全体凍結ロジック (`_freeze_active`/
  `_freeze_f_c`) は Phase 2 配線時に pair 単位拡張する形で **置換**するか、
  あるいは `pairwise_freeze_enabled=False` 時は従来動作を保持して両立させるか、
  Phase 2 で要決定。既定 OFF なら従来動作維持の互換性が高い
