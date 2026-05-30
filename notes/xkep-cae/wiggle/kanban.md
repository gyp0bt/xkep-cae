[[ようこそ]] / [[ポエム]] / [[../../../README|README]]

# wiggle カンバン

撚線工程 3D CG プロジェクトの作業ボード。

## 🔥 Next（次セッションでやる）

### 0. ボビン端点 pin + 巻取り/繰り出し kinematics（最優先）
- [ ] **接触アルゴリズム（衝突判定 / penalty / SDF）ではない**。kinematics 側
- [ ] 素線端点がボビン面（level-wind 螺旋上）に pin され、ボビン自転 Ω で
      巻取り（s_consumed +）/ 繰り出し（−）される
- [ ] 現状: `strandPinEndpoints(cfg, i, t)` が bobbin 中心軸の 1 点を返すだけ
- [ ] 目標: `buildLevelWindHelix(...)` の螺旋上の `helix(s_consumed(t))` 点を返す
- [ ] s_consumed は `strandState(cfg, t).L_consumed` 由来で時刻に応じ進行
- [ ] PBD chain の bobbin 側 pin は、回転に追従して胴体表面を滑る点になる
- [ ] 同様に巻取りスプール側でも実装（chain ではなく helix の終端だが対称）

### 1. トラバース巻き（左右往復）
- [ ] 巻取りスプール上で、現在巻かれている位置が左右に往復する動きを実装
- [ ] 「アクティブ巻き位置」を示す小さな torus / ring を巻線表面に追加
- [ ] traverse_pos = triangleWave(t × traverse_speed) × (TAKEUP_H − margin)
- [ ] traverse_speed は v_feed と pitch_count から決める（実機の比率調査）

### 2. クレードルの支持アーム
- [ ] cradle ring から各ボビンへの放射状の腕（CylinderGeometry 細棒 6 本）
- [ ] ボビンを支持枠ごと回転させる group 構造に再編
- [ ] cradle 全体が ω で回ると、ボビンが宙に浮いて見える違和感が消える
- [ ] cradle 自身も装飾的に回転（torus は対称なので外見上は変わらないが構造として）

### 3. 巻取り入口のガイド（フライヤー/カピストン風）
- [ ] strand 端から巻取りへ「ガイドプーリー」を 1 つ挟む
- [ ] 小さな円盤か torus を z=z1+15 あたりに配置、ケーブルがそこで曲がって
      スプールに入る描写
- [ ] リードチューブを 2 区間に分割（直線 → ガイド → 接線方向で接続）

### 4. lay plate（撚りダイスの手前ガイド板）
- [ ] ダイスの少し上流に薄い円盤（穴 6 つ）を置く
- [ ] 各穴の中心が outerStrand の z_lay 直前の位置と一致するよう配置
- [ ] 穴は ExtrudeGeometry か CSG（複雑なら省略可）
- [ ] これでボビン→lay plate→ダイス の流れがちゃんと「装置っぽく」なる

## 🟡 Backlog（次々セッション以降の候補）

- [ ] HDRI 差し替え（polyhaven の `studio_small_*` あたり、`.hdr` 1 枚）
- [ ] 導糸経路をベジエに（今は bobbin → lay plate が直線）
- [ ] ボビン自体の自転（cradle 内で水平軸まわりに回る — 素線が解けていく感）
- [ ] 1+6+12=19 本構成（2 層）への拡張
- [ ] S/Z lay 切替 GUI
- [ ] 巻線の helical texture（ボビン上に巻きの溝が見える）
- [ ] postprocessing の SSR で床反射をピカピカに
- [ ] glTF export → Blender / Three.js Editor で素材詰め
- [ ] FEM 連成（micro-macro）: helix の曲率から既存梁ソルバで局所応力、色マップ
- [ ] 巻取り満タンで自動的に simT をリセット → 連続稼働ループ
- [ ] 速度ゲージ HUD（v_feed / ω_takeup / 推定生産量 m/h を画面隅に表示）

## ⏸ 凍結（やらない）

- [ ] カラフル虹色パレット（次世代に戻したくなったら復活）
- [ ] PyVista 側の積極開発（sanity check で十分）
- [ ] CLAUDE.md の MCDD 規約系の wiggle への適用
- [ ] FEM のような厳密な接触解析（視覚的接触で十分、≠FEM のやばい接触）

## ✅ Done（直近セッション分）

### 第三航海（2026-05-28 〜 05-30）— パチもん物理三本柱

- [x] **有限長 supply→takeup 巻き替え**: `L_supply + L_transit + L_takeup = L_total`
      の物質保存、phase 1/2/3、s_trail で後端後退、`rewindReset` ボタン
- [x] **多層 level-wind 螺旋**（`buildLevelWindHelix`）でサプライ / 巻取りボビン胴体に
      実際の巻線見える化（軸往復 + 半径 step）
- [x] **PBD 鎖物理 `physics.js` 新設**（`StrandChain` = Verlet + 距離拘束 Gauss-Seidel）
- [x] 外周 6 本 + bundle + core すべて PBD 鎖化（kinematic helix 廃止、両端 pin で
      重力下にだらしなく垂れる）
- [x] **Bishop frame**（parallel-transport）で下流 helix wrap の安定化
- [x] **駆動張力 = rest 長短縮**: `rest_per_segment < chord/N` で boundary tension
      代用。chain type 別 gain（bundle 1.5×）
- [x] **遠心 wrap pretension** `wrapPretensionSlack(...)` + ω² scaling `OMEGA_TAUT`
      で回転に応じた taut 化
- [x] **ω-ramp damping**（`DAMPING_REST=0.992 → DAMPING_RUN=0.94` 補間）で
      高速回転時のチャタリング抑制
- [x] **パチもん EI = Laplacian 引き寄せ** `applyBending(k)` を距離拘束 Gauss-Seidel
      反復に共存。Cosserat / Timoshenko 不要で「真っ直ぐにしたがる」性質を獲得
- [x] cache buster `?v=N` を index.html + import 文に付与（python http.server で
      module キャッシュ強くなる問題の対策）
- [x] GUI: `駆動張力` / `wrap pretension k_ω` / `パチもん EI` / `素線 PBD 物理` トグル
- [x] memory に `wiggle-next-session` / `wiggle-pbd-patterns` 保存

### 第二航海（2026-05-28）

- [x] サプライボビンに**素線の巻線**（copper cylinder + flanges + body）
- [x] **撚りダイス**（LatheGeometry のファンネル形状、メタリック仕上げ）
- [x] **巻取りスプール**（core + flanges + 巻線、Z軸 90° 回転でスプール軸を X 方向に）
- [x] 回転速度倍速（omega 2π/5 → 2π/2、GUI で最大 ±8 rad/s まで）
- [x] **巻取りスプールの回転物理**（超簡易: v_feed/R_wound、解析的時間積分）
- [x] **サプライボビン距離拡大**（L_tail_in 250→450mm）
- [x] **ダイス-素線視覚接触**（DIE_BORE_R を素線束外径 + 0.5mm に絞る）
- [x] strand 端→巻取りの接続チューブ追加
- [x] カメラ FOV 32°→38°、長くなった機械に合わせ画角再調整
- [x] cradle ring（torus）追加で「ボビン群がドーナツ状に配置」を視覚化
- [x] 巻取りが見えなかった問題のデバッグ（背景壁が物理的オクルーダーだった）

### 第一航海（2026-05-22〜27）

- [x] xkep-cae 方向転換の合意形成（FEM 厳密 → 撚線 3D CG）
- [x] `wiggle/` パッケージ作成（kinematics + render + demo）
- [x] PyVista 1+6 撚線の MP4 / PNG 出力（headless）
- [x] 撚りほどき → 撚り工程の符号バグ修正（imprint 位相の物理導出 docstring 付き）
- [x] three.js 版立ち上げ（importmap で CDN、ビルド不要）
- [x] `MeshPhysicalMaterial` + bloom + RoomEnvironment + 3-spot lighting + shadow
- [x] lil-gui で L_pitch / ω / 本数 / R_layer / 速度 / bloom / 露出 を runtime 調整
- [x] chromium headless での自動スクショパス確立（`--enable-unsafe-swiftshader`）
- [x] http.server (port 8780) でローカル配信
- [x] memory に方向転換と「厳格規約は外す」を保存
- [x] 単位を mm 固定に（StranderConfig 全パラメータ）
- [x] 銅色単色化（COPPER_BASE = 0xb87333、素線ごとに ±5% lightness）
- [x] OrbitControls + autoRotate + 視点プリセット（俯瞰/側面/ダイス近景/先端ドリーイン）
- [x] 時間スライダー（GUI で 0〜120s スクラブ可能）
- [x] GUI フォルダ構造化（機械パラメータ / 再生 / ビジュアル / カメラ）

## 環境メモ

- 開発機: Linux, Python 3.10+, uv で管理
- ブラウザ確認: `python -m http.server 8780` を `/home/nishioka/work/xkep-cae` から起動、
  `http://localhost:8780/wiggle/web/index.html`
- スクショ: `chromium --headless=new --enable-unsafe-swiftshader --use-gl=angle --use-angle=swiftshader`
  でないと WebGL コンテキストが取れない
- 依存: `pyvista 0.48`, `imageio[ffmpeg]`, three.js 0.170 (CDN), lil-gui 0.19 (CDN)

## ファイルマップ

```
wiggle/
  __init__.py
  kinematics.py     # Python 版運動学（L_tail_in=450、符号バグ修正済み）
  render.py         # PyVista レンダラ
  demo.py           # 1+6 PNG/MP4 出力デモ
  web/
    index.html      # importmap + HUD
    main.js         # シーン構築 + 毎フレーム TubeGeometry 再構築 + 巻取り回転物理
    kinematics.js   # Python 版と完全対応
```

```
results/wiggle/
  strander_1plus6_t0.png    # PyVista t=0
  strander_1plus6_t3.png    # PyVista t=3
  strander_1plus6.mp4       # PyVista 6 秒アニメ
  web_t0.png  web_t3.png  web3.png  # three.js 初期版
  v3_wall_fix.png            # 巻取り visibility 修正版（背景壁を後退）
  v4_physics.png             # 回転物理 + ダイス接触 + L_tail_in=450 版
  v4_reframe.png             # FOV 38° + カメラ再フレーミング
  v4_takeup_big.png          # 巻取り拡大 + 初期 simT=10
```

## デバッグ Tips（第二航海で得た）

- **「見えないオブジェクト」の特定順序**: ① シーンに add してるか →
  ② カメラ frustum 内か → ③ **手前に何か遮蔽してないか** ← 今回ここで詰まった →
  ④ マテリアルが暗背景に溶け込んでないか → ⑤ scale や opacity が 0 になってないか
- **wireframe デバッグ**: `mat.wireframe = true` を一時的に GUI トグルにすると
  オクルーダー特定が一瞬で済む。次セッションで実装したい
