// wiggle — capstan probe（実験A のビューワ / Coulomb 版）
// 問い: particle 0 の「理想ヘリックス瞬間移動」を殺し、ドラム固定の継ぎ目だけ残したとき、
//   残りの素線は **摩擦 + 張力 + 非貫入** だけで巻き付くか？ → YES（capstan_probe.mjs で確定）。
//
// 物理は capstan_probe.mjs と同一コア（winding_core.js / physics.js を共有）:
//   - 重力 = 0、有限長の伸びない電線（per-segment 固定 rest）。pre-coil をゆるく置く。
//   - particle 0   = lay 点 = world 固定の継ぎ目（feed 側 +z 子午線, 角度 0）。traverse で y のみ移動。
//                    焼き込みが drum-local −θ に広がってヘリックスを成す（co-rotation だと子午線に縮退）。
//   - particle N-1 = feed 点（世界固定）。
//   - 摩擦 = **本物の Coulomb**（滑り補正を μ·dN でクランプ）を preStep で substep に 1 回。
//     → 距離拘束を壊さず inextensibility を保ち、capstan e^{μθ} が emergent に出る。
//   - 非貫入 = 距離ループ内（点 + 線分接触 segmentContact + 自己接触）。
//
// 検証済み（headless）: 貫通=0・トンネル線分=0・カクつき≈6°（みみず解消）。
//   既知の限界: 摩擦の creep で電線が ~1.5 倍に伸びる（連続給線=conveyor で解消予定）。
// 座標系: +y 上、ボビン軸 = +y、bobbin 中心 = 原点。単位 mm / rad / s。

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import GUI from "lil-gui";

const _ver = new URL(import.meta.url).search;
const { StrandChain } = await import("./physics.js" + _ver);
const {
  defaultCfg, estimateCapacity, wrappedArcLength, bobbinContact,
} = await import("./winding_core.js" + _ver);

// ---------- 寸法 ----------
const COPPER = 0xb87333;
const cfg = defaultCfg();
cfg.L_capacity = estimateCapacity(cfg);

const R_lay = cfg.BODY_R + cfg.wire_r;          // 41.8 mm
const circumference = 2 * Math.PI * R_lay;
const innerY = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
const bandR_wrap = cfg.BODY_R + cfg.wire_r + cfg.wire_r * 3;  // 47.2
const minR_body = cfg.BODY_R + cfg.wire_r;
const FEED_POS = new THREE.Vector3(0, 0, 200);
const pitch = 2 * cfg.wire_r * 1.05;
const TRAV_LO = -(innerY - cfg.wire_r - 2);
const TRAV_HI = (innerY - cfg.wire_r - 2);
const TRAV_SPAN = TRAV_HI - TRAV_LO;       // 両フランジ間の総トラベル

// ---------- Renderer / Scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
scene.fog = new THREE.Fog(0x0b0d10, 500, 1800);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.035).texture;

// 水平軸化: 物理は「軸=y」フレームのまま（bobbinContact 非改変）、sim 物体を simRoot に入れ
// sim(x,y,z) → render(y,z,x) の巡回置換回転で水平軸に見せる。重力は sim-z（軸垂直）に与える
// ので render では下向きになる。これで軸滑落（山積み）が物理的に消える。
const simRoot = new THREE.Group();
{
  const m = new THREE.Matrix4().set(
    0, 1, 0, 0,
    0, 0, 1, 0,
    1, 0, 0, 0,
    0, 0, 0, 1);
  simRoot.quaternion.setFromRotationMatrix(m);
}
scene.add(simRoot);

// ---------- Camera ----------
const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 1, 5000);
camera.position.set(220, 110, 240);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

// ---------- Lights ----------
scene.add(new THREE.AmbientLight(0xffffff, 0.18));
const keyL = new THREE.SpotLight(0xfff2dd, 12000, 1000, Math.PI / 5, 0.35, 1.4);
keyL.position.set(-180, 320, 200); keyL.target.position.set(0, 0, 0);
keyL.castShadow = true; keyL.shadow.mapSize.set(1024, 1024); keyL.shadow.bias = -0.002;
scene.add(keyL, keyL.target);
const fillL = new THREE.SpotLight(0x88aacc, 4500, 1000, Math.PI / 4, 0.6, 1.2);
fillL.position.set(280, 160, -120); fillL.target.position.set(0, 0, 0);
scene.add(fillL, fillL.target);
const rimL = new THREE.DirectionalLight(0xffe7cf, 1.2);
rimL.position.set(50, 80, -240);
scene.add(rimL);

// ---------- Floor ----------
const floor = new THREE.Mesh(new THREE.PlaneGeometry(1200, 1200),
  new THREE.MeshPhysicalMaterial({ color: 0x12141a, metalness: 0.25, roughness: 0.42, clearcoat: 0.3, clearcoatRoughness: 0.6 }));
floor.rotation.x = -Math.PI / 2;
floor.position.y = -(cfg.FLANGE_R + 24);
floor.receiveShadow = true;
scene.add(floor);

// ---------- Materials ----------
const copperMat = new THREE.MeshPhysicalMaterial({ color: COPPER, metalness: 0.95, roughness: 0.18 });
const bobbinBodyMat = new THREE.MeshPhysicalMaterial({ color: 0x3b2a1d, metalness: 0.15, roughness: 0.65 });
const bobbinFlangeMat = new THREE.MeshPhysicalMaterial({ color: 0x1f1610, metalness: 0.55, roughness: 0.4 });
const frameMat = new THREE.MeshPhysicalMaterial({ color: 0x5a6068, metalness: 0.8, roughness: 0.28 });

// ---------- Bobbin (spin around sim +y; render では水平軸) ----------
const bobbinSpin = new THREE.Group();
simRoot.add(bobbinSpin);
const body = new THREE.Mesh(new THREE.CylinderGeometry(cfg.BODY_R, cfg.BODY_R, cfg.BODY_H, 36), bobbinBodyMat);
body.castShadow = true; body.receiveShadow = true;
bobbinSpin.add(body);
for (const sgn of [-1, +1]) {
  const fl = new THREE.Mesh(new THREE.CylinderGeometry(cfg.FLANGE_R, cfg.FLANGE_R, cfg.FLANGE_H, 40), bobbinFlangeMat);
  fl.position.y = sgn * (cfg.BODY_H - cfg.FLANGE_H) / 2;
  fl.castShadow = true; fl.receiveShadow = true;
  bobbinSpin.add(fl);
}
const seamMarker = new THREE.Mesh(new THREE.SphereGeometry(cfg.wire_r * 1.6, 12, 12),
  new THREE.MeshBasicMaterial({ color: 0xff5544 }));
simRoot.add(seamMarker);

// ---------- Support frame ----------
const spindle = new THREE.Mesh(new THREE.CylinderGeometry(8, 8, cfg.BODY_H * 1.4, 16), frameMat);
spindle.castShadow = true;
simRoot.add(spindle);
for (const sgn of [-1, +1]) {
  const bearing = new THREE.Mesh(new THREE.TorusGeometry(14, 4, 10, 24), frameMat);
  bearing.rotation.x = Math.PI / 2; bearing.position.y = sgn * cfg.BODY_H * 0.65;
  bearing.castShadow = true;
  simRoot.add(bearing);
}

// ---------- Feed reel ----------
const feedReel = new THREE.Group();
feedReel.position.copy(FEED_POS);
simRoot.add(feedReel);
const feedBody = new THREE.Mesh(new THREE.CylinderGeometry(20, 20, 26, 24), bobbinBodyMat);
feedBody.rotation.z = Math.PI / 2; feedBody.castShadow = true;
feedReel.add(feedBody);
for (const dx of [-15, +15]) {
  const fl = new THREE.Mesh(new THREE.CylinderGeometry(28, 28, 3, 24),
    new THREE.MeshPhysicalMaterial({ color: 0x252028, metalness: 0.55, roughness: 0.4 }));
  fl.rotation.z = Math.PI / 2; fl.position.x = dx; fl.castShadow = true;
  feedReel.add(fl);
}

// ---------- 素線 (有限電線 PBD chain) ----------
const sourcePin = [FEED_POS.x, FEED_POS.y, FEED_POS.z];
const params = {
  omega: cfg.omega,
  mu: 0.8,
  seatGrip: 0.3,
  bendK: 0.04,
  gravity: 1.0,        // 1 = 9810 mm/s²（sim-z = 軸垂直 = render 下向き）
  segmentContact: true,
  traverse: true,
  backTension: 0.2,     // 供給リール制動: リードインを直線化（余長ループ防止）
  paused: false,
  speed: 1.0,
  reset() { buildChain(); },
  viewSide() { controls.target.set(0, 0, 0); camera.position.set(0, 20, 320); controls.update(); },
  viewTop() { controls.target.set(0, 0, 0); camera.position.set(0, 360, 0.01); controls.update(); },
  viewReset() { controls.target.set(0, 0, 0); camera.position.set(220, 110, 240); controls.update(); },
};

const state = { spinTheta: 0, turns: 0 };
let chain = null, N = 0, L_wire = 0;

// ---- 焼き込みコンベア状態 ----
const STANDOFF = 180;           // feed までのリードイン弧長
const SEG_LEN = 3.0;            // 材料セグメント長 ≈ baked 間隔
const bandR_wrap2 = bandR_wrap * bandR_wrap;
let baked = [];                 // 巻き済みパック: drum-local 座標（bobbinSpin で +θ 回す）
let payAccum = 0;               // take-up 累積長。R_lay·Δθ が SEG_LEN 貯まるごとに 1 粒子 pay-out+bake
let bakedSinceRebuild = 0;      // baked チューブ再構築のスロットル

// レベルワインド: 走行距離 dist を両フランジ間で折り返す三角波。
// 旧実装は単調ランプ（登りっぱなし）で 31 周目に上フランジを越えて場外へ → 電線が
// ボビン上端から walk-off して「巻かれたそばから消える」主因だった。三角波なら絶対に場外へ出ない。
function seamY(theta) {
  if (!params.traverse) return 0;
  const dist = (Math.abs(theta) / (2 * Math.PI)) * pitch;
  const period = 2 * TRAV_SPAN;
  let m = dist % period;
  if (m < 0) m += period;
  return m <= TRAV_SPAN ? TRAV_LO + m : TRAV_LO + (period - m);
}
function seam(theta) { return [R_lay * Math.sin(theta), seamY(theta), R_lay * Math.cos(theta)]; }

// アクティブ鎖 = リードインだけ（index 0 = lay 点＝seam, index N-1 = feed）。
// 巻き済みは焼き込みパック（baked）側に蓄積。総弧長が保存量でなくなり、巻き締めの余長が
// パックへ蓄積されて自然解消する（連続 pay-out）。capstan_probe→conveyor_probe で検証済み。
function buildChain() {
  state.spinTheta = 0; state.turns = 0;
  payAccum = 0; resetBaked();
  L_wire = STANDOFF;
  N = Math.max(60, Math.round(STANDOFF / SEG_LEN) + 1);
  const segRest = STANDOFF / (N - 1);
  const lay0 = seam(0);                     // lay 点（index 0）
  chain = new StrandChain(N, segRest, {
    gravity: [0, 0, -9810 * params.gravity],   // sim-z = 軸垂直（水平軸）→ render 下向き
    damping: 0.99, iters: 12,
    restLengths: new Float64Array(N - 1).fill(segRest),
  });
  chain.bendK = params.bendK;
  for (let i = 0; i < N; i++) {              // lay 点 → feed への直線リードイン
    const f = i / (N - 1);
    const k = 3 * i;
    chain.pos[k]     = lay0[0] + (FEED_POS.x - lay0[0]) * f;
    chain.pos[k + 1] = lay0[1] + (FEED_POS.y - lay0[1]) * f;
    chain.pos[k + 2] = lay0[2] + (FEED_POS.z - lay0[2]) * f;
    chain.prev[k] = chain.pos[k]; chain.prev[k + 1] = chain.pos[k + 1]; chain.prev[k + 2] = chain.pos[k + 2];
  }
  buildTubeGeometry();          // N 確定 → リードインチューブのトポロジを 1 回だけ確保
}

const chainMesh = new THREE.Mesh(new THREE.BufferGeometry(), copperMat);
chainMesh.castShadow = true; chainMesh.receiveShadow = true;
simRoot.add(chainMesh);

// 巻き済みパック: drum-local 座標で bobbinSpin の子にして θ で一緒に回す。
// チャンク分割: 焼き込み点を CHUNK_MAX ごとに区切り、満杯チャンクは凍結（二度と再構築しない）。
// → 再構築コストは常に ≤CHUNK_MAX 頂点で頭打ち。ボビン満杯まで巻いても重くならない。
const CHUNK_MAX = 256;
const bakedGroup = new THREE.Group();
bobbinSpin.add(bakedGroup);
let bakedChunks = [];        // [{pts:[{x,y,z}], mesh}]  最後が現在のチャンク
function newChunk(seed) {
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), copperMat);
  mesh.castShadow = true; mesh.receiveShadow = true; mesh.visible = false;
  bakedGroup.add(mesh);
  const pts = [];
  if (seed) pts.push(seed);  // 前チャンク末尾を引き継ぎ → チューブ連続
  bakedChunks.push({ pts, mesh });
}
function resetBaked() {
  for (const c of bakedChunks) { c.mesh.geometry.dispose(); bakedGroup.remove(c.mesh); }
  bakedChunks = []; baked = []; bakedSinceRebuild = 0;
  newChunk(null);
}
function rebuildCurrentChunk() {
  const ch = bakedChunks[bakedChunks.length - 1];
  if (!ch || ch.pts.length < 4) { if (ch) ch.mesh.visible = false; return; }
  const vec = ch.pts.map(p => new THREE.Vector3(p.x, p.y, p.z));
  const curve = new THREE.CatmullRomCurve3(vec, false, "centripetal", 0.5);
  ch.mesh.geometry.dispose();
  ch.mesh.geometry = new THREE.TubeGeometry(curve, Math.max(8, ch.pts.length), cfg.wire_r, 8, false);
  ch.mesh.visible = true;
}
// pay-out + bake: lay 点（index 0, 常に r=R_lay に座る）を drum-local で焼き込み、
// リードインを 1 粒子前進、feed に新粒子を繰り出す（N 一定）。
function bakeAndEmit() {
  const c = Math.cos(-state.spinTheta), s = Math.sin(-state.spinTheta);   // world → drum-local
  const wx = chain.pos[0], wy = chain.pos[1], wz = chain.pos[2];
  const p = { x: c * wx + s * wz, y: wy, z: -s * wx + c * wz };
  baked.push(p);
  let ch = bakedChunks[bakedChunks.length - 1];
  ch.pts.push(p);
  bakedSinceRebuild++;
  if (ch.pts.length >= CHUNK_MAX) {     // チャンク満杯 → 確定して凍結、新チャンク開始
    rebuildCurrentChunk();
    newChunk(p);                        // 末尾点を seed に連続化
    bakedSinceRebuild = 0;
  }
  for (let i = 0; i < N - 1; i++) {
    const d = 3 * i, src = 3 * (i + 1);
    chain.pos[d] = chain.pos[src]; chain.pos[d + 1] = chain.pos[src + 1]; chain.pos[d + 2] = chain.pos[src + 2];
    chain.prev[d] = chain.prev[src]; chain.prev[d + 1] = chain.prev[src + 1]; chain.prev[d + 2] = chain.prev[src + 2];
  }
  const kf = 3 * (N - 1);
  chain.pos[kf] = FEED_POS.x; chain.pos[kf + 1] = FEED_POS.y; chain.pos[kf + 2] = FEED_POS.z;
  chain.prev[kf] = FEED_POS.x; chain.prev[kf + 1] = FEED_POS.y; chain.prev[kf + 2] = FEED_POS.z;
}
// バックテンション: リードイン（接線点 B→feed）を弦に向け直線化（余長ループ防止）
function applyBackTension(pos, n, fixed, gain) {
  if (gain <= 0) return;
  let B = -1;
  for (let i = n - 2; i >= 0; i--) {
    const k = 3 * i;
    if (pos[k] * pos[k] + pos[k + 2] * pos[k + 2] <= bandR_wrap2 && Math.abs(pos[k + 1]) < innerY) { B = i; break; }
  }
  if (B < 0) B = 0;
  const bk = 3 * B;
  const Bx = pos[bk], By = pos[bk + 1], Bz = pos[bk + 2];
  const kf = 3 * (n - 1);
  const dx = pos[kf] - Bx, dy = pos[kf + 1] - By, dz = pos[kf + 2] - Bz;
  const L2 = dx * dx + dy * dy + dz * dz;
  if (L2 < 1e-9) return;
  for (let i = B + 1; i < n - 1; i++) {
    if (fixed[i]) continue;
    const k = 3 * i;
    const t = Math.max(0, Math.min(1, ((pos[k] - Bx) * dx + (pos[k + 1] - By) * dy + (pos[k + 2] - Bz) * dz) / L2));
    pos[k]     += (Bx + dx * t - pos[k])     * gain;
    pos[k + 1] += (By + dy * t - pos[k + 1]) * gain;
    pos[k + 2] += (Bz + dz * t - pos[k + 2]) * gain;
  }
}

// ---------- 接触（Coulomb 摩擦 preStep + 非貫入ループ内） ----------
const contactCtx = {
  cfg, legacyStick: 0, mu: 0.8, seatGrip: 0.3,
  frictionDtheta: 0, segmentContact: true, skipFriction: false,
  eye: null, guide: null,
};
const frictionStep = (pos, n, fixed, prev) => { contactCtx.skipFriction = false; bobbinContact(pos, n, fixed, prev, contactCtx); };
const contactProject = (pos, n, fixed, prev) => {
  contactCtx.skipFriction = true; bobbinContact(pos, n, fixed, prev, contactCtx);
  applyBackTension(pos, n, fixed, params.backTension);
};

// ---------- Helpers ----------
// ---- 軽量チューブ描画 ----
// 旧: 毎フレ mesh.geometry.dispose() + new TubeGeometry(...)。940 分割 × 10 放射 ≈ 9400 頂点を
//   60回/秒 で全再生成 + index/uv を毎回 GPU 再アップロード + GC churn → これが Mac の主負荷。
// 新: トポロジ（index/uv）は buildChain で 1 回だけ確定。毎フレは position/normal だけ書き換え。
//   TubeGeometry と同じ Frenet フレーム式を用いるので見た目は等価。
const _tube = { T: 0, R: 8, pos: null, nor: null };
let _pts = [];
function chainPoints() {                       // chain.pos → 再利用 Vector3 配列（毎フレ alloc 回避）
  if (_pts.length !== N) _pts = Array.from({ length: N }, () => new THREE.Vector3());
  for (let i = 0; i < N; i++) { const k = 3 * i; _pts[i].set(chain.pos[k], chain.pos[k + 1], chain.pos[k + 2]); }
  return _pts;
}
function buildTubeGeometry() {                  // N 確定時に 1 回。index/uv/buffer を確保。
  const T = Math.min(360, Math.max(180, N));    // 軸方向分割（旧 940 → 上限 360 で十分滑らか）
  const R = 8;                                  // 放射分割
  const nv = (T + 1) * (R + 1);
  const positions = new Float32Array(nv * 3);
  const normals = new Float32Array(nv * 3);
  const uvs = new Float32Array(nv * 2);
  const indices = [];
  for (let j = 1; j <= T; j++) for (let i = 1; i <= R; i++) {
    const a = (R + 1) * (j - 1) + (i - 1), b = (R + 1) * j + (i - 1);
    const c = (R + 1) * j + i, d = (R + 1) * (j - 1) + i;
    indices.push(a, b, d, b, c, d);
  }
  for (let j = 0; j <= T; j++) for (let i = 0; i <= R; i++) {
    const k = 2 * ((R + 1) * j + i); uvs[k] = j / T; uvs[k + 1] = i / R;
  }
  const geo = new THREE.BufferGeometry();
  geo.setIndex(indices);
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  chainMesh.geometry.dispose();
  chainMesh.geometry = geo;
  _tube.T = T; _tube.R = R; _tube.pos = positions; _tube.nor = normals;
}
function updateTube(radius) {                   // 毎フレ: 中心線から position/normal のみ更新
  if (N < 4 || !_tube.pos) { chainMesh.visible = false; return; }
  chainMesh.visible = true;
  const T = _tube.T, R = _tube.R, pos = _tube.pos, nor = _tube.nor;
  const curve = new THREE.CatmullRomCurve3(chainPoints(), false, "centripetal", 0.5);
  const frames = curve.computeFrenetFrames(T, false);
  const P = new THREE.Vector3();
  for (let j = 0; j <= T; j++) {
    curve.getPointAt(j / T, P);
    const Nrm = frames.normals[j], Bnm = frames.binormals[j];
    for (let i = 0; i <= R; i++) {
      const v = i / R * Math.PI * 2, s = Math.sin(v), c = -Math.cos(v);
      let nx = c * Nrm.x + s * Bnm.x, ny = c * Nrm.y + s * Bnm.y, nz = c * Nrm.z + s * Bnm.z;
      const inv = 1 / (Math.hypot(nx, ny, nz) || 1); nx *= inv; ny *= inv; nz *= inv;
      const idx = 3 * ((R + 1) * j + i);
      nor[idx] = nx; nor[idx + 1] = ny; nor[idx + 2] = nz;
      pos[idx] = P.x + radius * nx; pos[idx + 1] = P.y + radius * ny; pos[idx + 2] = P.z + radius * nz;
    }
  }
  chainMesh.geometry.attributes.position.needsUpdate = true;
  chainMesh.geometry.attributes.normal.needsUpdate = true;
  chainMesh.geometry.computeBoundingSphere();
}
function wrapStats() {
  let n = 0, sum = 0;
  for (let i = 0; i < N; i++) {
    const k = 3 * i;
    const r = Math.hypot(chain.pos[k], chain.pos[k + 2]);
    if (r <= bandR_wrap && Math.abs(chain.pos[k + 1]) < innerY) { n++; sum += r; }
  }
  return { n, mean: n ? sum / n : 0 };
}

// ---------- 動的更新 ----------
function updateScene(dt) {
  if (params.paused || dt <= 0 || !chain) return;
  state.spinTheta += params.omega * dt;
  bobbinSpin.rotation.y = state.spinTheta;
  state.turns = Math.abs(state.spinTheta) / (2 * Math.PI);

  // lay 点は **world 固定**（feed 側 +z 子午線 = 角度 0）。ドラムがその下を回り、
  // 焼き込みが drum-local で −θ に広がって初めてヘリックスになる。traverse は y のみ駆動。
  // 旧 co-rotating seam(θ) は bake の −θ 回転で角度が相殺され、全巻き点が drum-local
  // 角度 0 の 1 子午線に潰れていた（→ 直線化で TubeGeometry が NaN → 消える電線の主因）。
  const pin0 = [0, seamY(state.spinTheta), R_lay];
  seamMarker.position.set(pin0[0], pin0[1], pin0[2]);

  contactCtx.mu = params.mu;
  contactCtx.seatGrip = params.seatGrip;
  contactCtx.segmentContact = params.segmentContact;
  chain.bendK = params.bendK;

  const SUBSTEPS = 4, subDt = dt / SUBSTEPS;
  contactCtx.frictionDtheta = params.omega * subDt;
  for (let s = 0; s < SUBSTEPS; s++) {
    chain.step(subDt, pin0, sourcePin, { preStep: frictionStep, contactProject, bendIters: 3, maxStep: 8 });
  }
  // 連続 pay-out: take-up 長 R_lay·Δθ が SEG_LEN 貯まるごとに 1 粒子焼き込み + 給線
  payAccum += R_lay * Math.abs(params.omega * dt);
  let safety = 0;
  while (payAccum >= SEG_LEN && safety++ < 50) { bakeAndEmit(); payAccum -= SEG_LEN; }
  if (bakedSinceRebuild >= 12) { rebuildCurrentChunk(); bakedSinceRebuild = 0; }  // 現チャンクのみ再構築

  updateTube(cfg.wire_r);   // リードインチューブ（短い）
}

// ---------- HUD ----------
const statusEl = document.getElementById("status");
function updateHUD(t) {
  if (!chain) return;
  const woundTurns = baked.length * SEG_LEN / circumference;   // 焼き込みパックの巻数
  const stretch = chain.totalArcLength() / L_wire;             // リードインの伸び
  statusEl.innerHTML =
    `<span class="phase">焼き込みコンベア</span> · 自転 <b>${state.turns.toFixed(2)}</b> 周 · ` +
    `巻付 <b>${woundTurns.toFixed(2)}</b> 周（焼込 ${baked.length}点）<br>` +
    `<span class="count">リードイン ${N}粒子 · 伸び ${stretch.toFixed(2)}×</span> ` +
    `(R_lay=${R_lay.toFixed(1)}) · μ=${params.mu.toFixed(2)} · BT=${params.backTension.toFixed(2)} · t=${t.toFixed(1)}s`;
}

// ---------- GUI ----------
buildChain();
const gui = new GUI({ title: "capstan probe (Coulomb)" });
gui.add(params, "reset").name("⟲ リセット");
const fPhys = gui.addFolder("物理（Coulomb 摩擦 + 張力）");
fPhys.add(params, "omega", -4, 6, 0.05).name("ω [rad/s]");
fPhys.add(params, "mu", 0, 1.5, 0.02).name("摩擦 μ");
fPhys.add(params, "seatGrip", 0, 1, 0.05).name("seating");
fPhys.add(params, "bendK", 0, 0.3, 0.01).name("EI (パチもん)");
fPhys.add(params, "gravity", 0, 2, 0.05).name("重力 (×g)")
  .onChange(v => { if (chain) chain.gravity[2] = -9810 * v; });
fPhys.add(params, "segmentContact").name("線分接触 (貫通防止)");
fPhys.add(params, "traverse").name("軸送り traverse").onChange(() => buildChain());
fPhys.add(params, "backTension", 0, 0.6, 0.02).name("バックテンション");
fPhys.open();
const fPlay = gui.addFolder("再生");
fPlay.add(params, "paused").name("一時停止");
fPlay.add(params, "speed", 0, 5, 0.1).name("速度");
const fCam = gui.addFolder("カメラ");
fCam.add(params, "viewReset").name("初期位置");
fCam.add(params, "viewSide").name("側面");
fCam.add(params, "viewTop").name("俯瞰");

// ---------- Main loop ----------
let last = performance.now();
let simT = 0;
const _warmup = parseInt(new URL(location.href).searchParams.get("warmup") || "0", 10);
if (_warmup > 0) { for (let i = 0; i < _warmup; i++) { updateScene(1 / 60); simT += 1 / 60; } }

const FIXED_DT = 1 / 60;
let acc = 0;
function tick() {
  const now = performance.now();
  const wallDt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (!params.paused) {
    acc += wallDt * params.speed;
    let n = 0;
    while (acc >= FIXED_DT && n < 8) { updateScene(FIXED_DT); simT += FIXED_DT; acc -= FIXED_DT; n++; }
  }
  updateHUD(simT);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
