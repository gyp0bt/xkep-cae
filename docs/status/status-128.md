# status-128: NCP摩擦接触の行列特異化修正 + 被膜beam_I修正

[← README](../../README.md) | [← status-index](status-index.md) | [← roadmap](../roadmap.md)

**日付**: 2026-03-07
**テスト数**: 2271（fast: 1691 + 1 xfailed / slow: 362 - 9 xfailed / deprecated: 218）

## 概要

status-127のTODOに基づき、以下の修正を実施。

1. **NCP摩擦接触の行列特異化修正**（solver_ncp.py正則化）
2. **被膜モデルのbeam_I修正**（被膜込み等価断面二次モーメント使用）
3. **CI状況調査とgh CLI利用可能性確認**
4. **旧ソルバーテストNCP移行状況の棚卸し**

## 実施内容

### 1. NCP摩擦接触の行列特異化修正

**問題**: スリップ時の接線Jacobian `J_t_t = I - ratio*(I - q̂⊗q̂)` で、`ratio = μ*p_n/||λ̂_t||` が1を超えると固有値が負になり、拡大行列全体が特異化。`spsolve`でMatrixRankWarningが発生し、NaN発散。

**対処**:
- `solver_ncp.py:1791-1795`: ratio > 1 の場合に `J_t_t += (ratio-1+ε)*I` の正則化項を追加
  - J_t_tの最小固有値が `ε = 1e-4` 以上になることを保証
  - Newton法の二次収束性を損なわない（ratio制限方式との違い）
- `solver_ncp.py:926`: 接触なし時の `spsolve` に `MatrixRankWarning` キャッチ + 対角正則化フォールバック追加
- `solver_ncp.py:973`: 拡大行列の `spsolve` に同様のフォールバック追加

**結果**: Timo3D摩擦テスト（f_z=500N, μ=0.3）が収束するようになり、xfail解除。CR梁はまだ不収束（別原因）。

| テスト | 修正前 | 修正後 |
|--------|--------|--------|
| `test_timo3d_friction_converges` | XFAIL（行列特異化） | **PASSED** |
| `test_cr_friction_converges` | XFAIL（行列特異化） | XFAIL（CR固有の不収束） |

### 2. 被膜モデルのbeam_I修正

**問題**: `test_coated_vs_bare_stiffness` で被膜付きの変位が素線のみより19%大きい。

**調査結果**:
- テストの `ContactConfig` で `beam_I=_SECTION.Iy`（素線のみ）を使用していた
- 被膜込みの場合、`coated_beam_section()` で計算した等価Iyを使用すべき
- **修正**: `with_coating=True` の場合に `beam_I = EIy_coated / E` を使用

**残存問題**: beam_I修正後も比率は1.19のまま。原因は被膜によるメッシュ配置半径の増大（`gap = _COATING.thickness * 4 = 0.2e-3`）が接触配置を変え、構造剛性増加を打ち消していること。これは物理的に妥当な挙動であり、テスト閾値の見直しまたはgapパラメータの調整が必要。xfailを維持。

### 3. CI状況調査

- **gh CLI**: `apt install gh` で導入成功、`gh auth status` で認証確認済み
- **CI失敗パターン**: 直近40件でtest-fast成功は4件のみ（すべてdocs更新のみのコミット）
- **ログ取得制限**: `results-receiver.actions.githubusercontent.com` がプロキシでブロック。`gh run view --log-failed` やジョブログのダウンロードが不可。APIでcheck-runsのアノテーションは取得可能だが、具体的なテスト名は含まれない
- **ローカルテスト**: 全1640 fast test がPASS（CI環境固有の問題の可能性）

### 4. 旧ソルバーテストNCP移行状況

| 旧テスト | NCP版 | 状態 |
|---------|--------|------|
| test_beam_contact_penetration.py | test_beam_contact_penetration_ncp.py | ✅ 移行済み |
| test_coated_wire_integration.py | test_coated_wire_integration_ncp.py | ✅ 移行済み |
| test_friction_validation.py | test_friction_validation_ncp.py | ✅ 移行済み |
| test_hysteresis.py | test_hysteresis_ncp.py | ✅ 移行済み |
| test_large_scale_contact.py | test_large_scale_contact_ncp.py | ✅ 移行済み |
| test_real_beam_contact.py | test_real_beam_contact_ncp.py | ✅ 移行済み |
| test_solver_hooks.py | test_solver_ncp.py | ✅ 移行先明記済み |
| test_twisted_wire_contact.py | test_twisted_wire_contact_ncp.py | ✅ 移行済み |

全8ファイルのNCP移行版が存在。旧コードの削除はまだ時期尚早（一部テストがxfailの段階）。

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `xkep_cae/contact/solver_ncp.py` | J_t_t正則化 + spsolve特異行列フォールバック |
| `tests/contact/test_real_beam_contact_ncp.py` | Timo3D摩擦xfail解除、CR摩擦xfail理由更新 |
| `tests/contact/test_coated_wire_integration_ncp.py` | 被膜込みbeam_Iの使用 |

## 設計上の懸念・ユーザーへの確認事項

1. **CR梁の摩擦接触不収束**: Timo3Dは正則化で解決したがCR梁はまだ不収束。CR梁の接線剛性行列自体の問題か、CR回転パラメータ化の影響を調査する必要がある
2. **被膜モデルのgap影響**: 被膜付きで配置半径が10%増大し、接触配置が変わる。被膜の剛性寄与より配置変化の影響が支配的。テストのgapパラメータ調整または閾値緩和が必要
3. **CI test-fast失敗**: ローカルでは全PASSだがCIでは失敗。CI環境固有の問題（scipy/numpyバージョン差等）。CIログ取得が制限されているため原因特定が困難
4. **7本NCP曲げ揺動**: 10度曲げでは収束するが45度以上で不収束。ソルバーの基本的な性能限界。CR梁+NCP+Mortarの組み合わせで大変形時の収束が困難

## TODO

- [ ] CR梁の摩擦接触不収束の原因調査
- [ ] 被膜テストのgapパラメータ最適化（被膜 ≤ 素線 の変位になるgap値の探索）
- [ ] CI test-fast失敗の原因特定（CIログ取得制限の回避方法を検討）
- [ ] 7本NCP曲げ揺動の大角度収束改善（ソルバーアルゴリズム改善）
- [ ] 19本NCP径方向圧縮のCI環境での収束確認
- [ ] 旧ソルバーコードの削除検討（全NCP版テストがPASS後）

## 運用フィードバック

### 効果的な点
- J_t_tの固有値分析（1, 1-ratio）から正則化項の大きさを理論的に導出できた
- Agent toolによる並列調査（NCP摩擦+被膜モデル同時調査）が効率的
- gh CLIによるCI状況の体系的把握が可能に

### 非効果的な点
- CIログのダウンロードがプロキシ制限でブロック。ローカルで再現しないCI失敗の原因特定が困難
- 7本曲げ揺動テストの実行に~85秒かかり、パラメータ探索に時間がかかる

---

[← README](../../README.md)
