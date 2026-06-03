// wiggle — three.js studio renderer for the strander
// 座標系: 機械軸 = +z（kinematics.js と同じ）、単位 mm

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import GUI from "lil-gui";

// 親 HTML が ?v=Date.now() を付けて main.js を呼んでくれる前提で、
// import.meta.url の search を local module 動的 import に引き継ぎ全ファイル毎回 fresh
const _ver = new URL(import.meta.url).search;
const {
  StranderConfig,
  outerStrandCenterline,
  coreStrandCenterline,
  bobbinTransform,
  strandState,
  strandPathArcLength,
  trimCenterlineByArc,
  buildLevelWindHelix,
  outerStrandHelixDownstream,
  strandPinEndpoints,
  computeBishopFrames,
  cumulativeArcLength,
  wrapPretensionSlack,
} = await import("./kinematics.js" + _ver);
const { StrandChain } = await import("./physics.js" + _ver);

// ---------- Copper palette ----------
const COPPER_BASE = 0xb87333;

function copperVariant(index, total) {
  const c = new THREE.Color(COPPER_BASE);
  const hsl = {};
  c.getHSL(hsl);
  const shift = ((index / Math.max(total, 1)) - 0.5) * 0.10;
  c.setHSL(hsl.h, hsl.s, Math.max(0, Math.min(1, hsl.l + shift)));
  return c;
}

const TUBE_PATH_SEGMENTS = 300;
const TUBE_RADIAL_SEGMENTS = 18;

// ---------- Config ----------
const cfg = new StranderConfig();
const machineLen = cfg.L_tail_in + cfg.L_tail_out;
const zMid = (cfg.z0 + cfg.z1) / 2;

// ---------- Renderer ----------
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

// ---------- Scene ----------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
scene.fog = new THREE.Fog(0x0b0d10, machineLen * 1.8, machineLen * 5);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.035).texture;

// ---------- Camera ----------
const camera = new THREE.PerspectiveCamera(
  38, window.innerWidth / window.innerHeight, 1, 20000,
);
const zCenter = (cfg.z0 - 40 + cfg.z1 + 30) / 2;
camera.position.set(-machineLen * 1.35, cfg.R_bobbin * 1.3, zCenter);
camera.up.set(0, 1, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, zCenter);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.autoRotate = false;
controls.autoRotateSpeed = 1.0;
controls.update();

// ---------- Lights ----------
const ambient = new THREE.AmbientLight(0xffffff, 0.15);
scene.add(ambient);

const key = new THREE.SpotLight(0xfff2dd, 35000, machineLen * 3, Math.PI / 5, 0.35, 1.4);
key.position.set(-machineLen * 0.45, machineLen * 0.5, zMid - machineLen * 0.1);
key.target.position.set(0, 0, zMid);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.bias = -0.002;
key.shadow.radius = 4;
scene.add(key, key.target);

const fill = new THREE.SpotLight(0x88aacc, 12000, machineLen * 3, Math.PI / 4, 0.55, 1.2);
fill.position.set(machineLen * 0.55, machineLen * 0.3, zMid + machineLen * 0.25);
fill.target.position.set(0, 0, zMid);
scene.add(fill, fill.target);

const rim = new THREE.DirectionalLight(0xffe7cf, 2.0);
rim.position.set(50, -80, cfg.z1 + 200);
scene.add(rim);

// ---------- Floor + Wall ----------
const floorGeo = new THREE.PlaneGeometry(machineLen * 2.5, machineLen * 2);
const floorMat = new THREE.MeshPhysicalMaterial({
  color: 0x12141a, metalness: 0.25, roughness: 0.42,
  clearcoat: 0.3, clearcoatRoughness: 0.6,
});
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI / 2;
floor.position.set(0, -cfg.R_bobbin * 1.55, zMid);
floor.receiveShadow = true;
scene.add(floor);

const wallGeo = new THREE.PlaneGeometry(machineLen * 2.5, machineLen);
const wallMat = new THREE.MeshPhysicalMaterial({ color: 0x0c0e12, roughness: 0.85 });
const wall = new THREE.Mesh(wallGeo, wallMat);
wall.position.set(0, machineLen * 0.2, cfg.z1 + 200);
wall.rotation.y = Math.PI;
scene.add(wall);

// ---------- Materials ----------
const copperMat = new THREE.MeshPhysicalMaterial({
  color: COPPER_BASE, metalness: 0.95, roughness: 0.18,
});

let outerMats = [];
function rebuildOuterMats(n) {
  outerMats.forEach((m) => m.dispose());
  outerMats = [];
  for (let i = 0; i < n; i++) {
    outerMats.push(new THREE.MeshPhysicalMaterial({
      color: copperVariant(i, n), metalness: 0.95, roughness: 0.18,
    }));
  }
}
rebuildOuterMats(cfg.n_outer);

const bobbinBodyMat = new THREE.MeshPhysicalMaterial({
  color: 0x3b2a1d, metalness: 0.15, roughness: 0.65,
});
const bobbinFlangeMat = new THREE.MeshPhysicalMaterial({
  color: 0x1f1610, metalness: 0.55, roughness: 0.4,
});
const dieMat = new THREE.MeshPhysicalMaterial({
  color: 0x606870, metalness: 0.85, roughness: 0.25, side: THREE.DoubleSide,
});
const takeupBodyMat = new THREE.MeshPhysicalMaterial({
  color: 0x5a6270, metalness: 0.75, roughness: 0.3,
});
const cradleMat = new THREE.MeshPhysicalMaterial({
  color: 0x404850, metalness: 0.8, roughness: 0.3,
});
const woundCableMat = new THREE.MeshPhysicalMaterial({
  color: 0xa06828, metalness: 0.88, roughness: 0.28,
});

// ---------- Strand meshes ----------
const strandGroup = new THREE.Group();
scene.add(strandGroup);

const coreMesh = new THREE.Mesh(new THREE.BufferGeometry(), copperMat);
coreMesh.castShadow = true;
coreMesh.receiveShadow = true;
strandGroup.add(coreMesh);

let outerMeshes = [];

function ensureOuterMeshes(n) {
  while (outerMeshes.length < n) {
    const idx = outerMeshes.length;
    const m = new THREE.Mesh(new THREE.BufferGeometry(), outerMats[idx % outerMats.length]);
    m.castShadow = true;
    m.receiveShadow = true;
    strandGroup.add(m);
    outerMeshes.push(m);
  }
  while (outerMeshes.length > n) {
    const m = outerMeshes.pop();
    strandGroup.remove(m);
    m.geometry.dispose();
  }
  for (let i = 0; i < outerMeshes.length; i++) {
    outerMeshes[i].material = outerMats[i % outerMats.length];
  }
}
ensureOuterMeshes(cfg.n_outer);

// ---------- 素線 PBD chains（upstream only）----------
const PARTICLES_PER_STRAND = 32;
const CHAIN_SLACK = 1.02;   // rest_length = 弦の 1.02 倍 → 軽くたるむ
// 駆動張力: rest < 弦長 で chain を強制 taut 化（ボビン牽引 + 摩擦 brake の物理代用）。
// driveTension=1 で chord を DRIVE_STRAIN_MAX (= 7%) 短く取る → ε ≈ 7% の引張
const DRIVE_STRAIN_MAX = 0.07;
const BUNDLE_DRIVE_GAIN = 1.5;   // bundle は質量 ~7 倍（6 外周 + 1 芯）なので boost
const SLACK_FLOOR = 0.93;   // 反復爆発防止: rest/chord をこれ以上下げない
// 回転 centrifugal pretension: ω² に比例して taut 化。回転 strand の遠心張力は
// ε ~ ρω²R²/E_eff だが、PBD 風には strain として吸収する
const OMEGA_TAUT = 0.012;   // strain per (rad/s)²。ω=π で約 0.12 strain
// damping ramp-up: 回転 → ボビン軸 brake/摩擦が効く → chain dissipation 増
const DAMPING_REST = 0.992;
const DAMPING_RUN = 0.94;
let chains = [];

function rebuildChains() {
  chains = [];
  for (let i = 0; i < cfg.n_outer; i++) {
    const { start, end } = strandPinEndpoints(cfg, i, 0);
    const dist = Math.hypot(end[0] - start[0], end[1] - start[1], end[2] - start[2]);
    const rest = dist / (PARTICLES_PER_STRAND - 1) * CHAIN_SLACK;
    const c = new StrandChain(PARTICLES_PER_STRAND, rest);
    c.initLine(start, end);
    chains.push(c);
  }
}
rebuildChains();

// ---------- cable bundle PBD chain（downstream を駆動）----------
const CABLE_PARTICLES = 60;
const CABLE_SLACK = 1.008;   // 軽い slack → 自然なたわみ
let cableBundle;
function rebuildBundle() {
  const start = [0, 0, cfg.z_lay];
  const end = [0, 0, cfg.z1];
  const dist = cfg.z1 - cfg.z_lay;
  const rest = dist / (CABLE_PARTICLES - 1) * CABLE_SLACK;
  cableBundle = new StrandChain(CABLE_PARTICLES, rest, { damping: 0.992, iters: 16 });
  cableBundle.initLine(start, end);
}
rebuildBundle();

// ---------- core 素線 supply 側 chain（z0 → z_lay、軸上で静かに sag）----------
const CORE_PARTICLES = 28;
const CORE_SLACK = 1.012;
let coreSupplyChain;
function rebuildCoreChain() {
  const start = [0, 0, cfg.z0];
  const end = [0, 0, cfg.z_lay];
  const dist = cfg.z_lay - cfg.z0;
  const rest = dist / (CORE_PARTICLES - 1) * CORE_SLACK;
  coreSupplyChain = new StrandChain(CORE_PARTICLES, rest, { damping: 0.99, iters: 8 });
  coreSupplyChain.initLine(start, end);
}
rebuildCoreChain();

function toVec3Array(pts) {
  return pts.map((p) => new THREE.Vector3(p.x, p.y, p.z));
}

function updateTube(mesh, pts, radius) {
  const curve = new THREE.CatmullRomCurve3(pts, false, "catmullrom", 0.0);
  const geo = new THREE.TubeGeometry(curve, TUBE_PATH_SEGMENTS, radius, TUBE_RADIAL_SEGMENTS, false);
  mesh.geometry.dispose();
  mesh.geometry = geo;
}

// ---------- Supply bobbins ----------
const bobbinGroup = new THREE.Group();
scene.add(bobbinGroup);
let bobbinAssemblies = [];

const BOBBIN_BODY_R = 25;
const BOBBIN_BODY_H = 70;
const BOBBIN_FLANGE_R = 38;
const BOBBIN_FLANGE_H = 6;
const BOBBIN_WIND_R_MAX = 36;   // flange (38) のわずか内側
const BOBBIN_WIND_H = BOBBIN_BODY_H - BOBBIN_FLANGE_H * 2 - 4;

function makeBobbinAssembly() {
  const group = new THREE.Group();
  const spin = new THREE.Group();  // bobbin 自身の自転（+y_local 周り）
  group.add(spin);
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(BOBBIN_BODY_R, BOBBIN_BODY_R, BOBBIN_BODY_H, 28),
    bobbinBodyMat,
  );
  const flA = new THREE.Mesh(
    new THREE.CylinderGeometry(BOBBIN_FLANGE_R, BOBBIN_FLANGE_R, BOBBIN_FLANGE_H, 28),
    bobbinFlangeMat,
  );
  const flB = new THREE.Mesh(
    new THREE.CylinderGeometry(BOBBIN_FLANGE_R, BOBBIN_FLANGE_R, BOBBIN_FLANGE_H, 28),
    bobbinFlangeMat,
  );
  const wound = new THREE.Mesh(new THREE.BufferGeometry(), copperMat);
  body.castShadow = true;
  flA.castShadow = true;
  flB.castShadow = true;
  wound.castShadow = true;
  body.receiveShadow = true;
  flA.receiveShadow = true;
  flB.receiveShadow = true;
  wound.receiveShadow = true;
  spin.add(body, flA, flB, wound);
  bobbinGroup.add(group);
  return { group, spin, body, flA, flB, wound, spinTheta: 0 };
}

function ensureBobbins(n) {
  while (bobbinAssemblies.length < n) bobbinAssemblies.push(makeBobbinAssembly());
  while (bobbinAssemblies.length > n) {
    const a = bobbinAssemblies.pop();
    bobbinGroup.remove(a.group);
    a.body.geometry.dispose();
    a.flA.geometry.dispose();
    a.flB.geometry.dispose();
    a.wound.geometry.dispose();
  }
}
ensureBobbins(cfg.n_outer);

const yHat = new THREE.Vector3(0, 1, 0);
function placeBobbin(asm, transform) {
  const axis = new THREE.Vector3(transform.axis.x, transform.axis.y, transform.axis.z).normalize();
  const pos = new THREE.Vector3(transform.position.x, transform.position.y, transform.position.z);
  asm.group.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(yHat, axis));
  asm.group.position.copy(pos);
  // spin subgroup 内部に body / flange / wound を中心配置
  asm.body.position.set(0, BOBBIN_BODY_H / 2, 0);
  asm.flA.position.set(0, BOBBIN_FLANGE_H / 2, 0);
  asm.flB.position.set(0, BOBBIN_BODY_H - BOBBIN_FLANGE_H / 2, 0);
  // wound helix は cylinder 中心 (y=BOBBIN_BODY_H/2) に巻線中央が一致するようオフセット
  asm.wound.position.set(0, BOBBIN_BODY_H / 2, 0);
}

// ---------- Cradle ring ----------
const cradleGeo = new THREE.TorusGeometry(cfg.R_bobbin, 3, 12, 64);
const cradle = new THREE.Mesh(cradleGeo, cradleMat);
cradle.position.set(0, 0, cfg.z0 - 40);
cradle.castShadow = true;
scene.add(cradle);

// ---------- Stranding die (funnel) ----------
const DIE_BORE_R = cfg.R_layer + cfg.outer_radius + 0.5;
const DIE_OUTER_R = 35;
const DIE_H = 28;
const dh = DIE_H / 2;
const DIE_ENTRY_R = DIE_BORE_R * 1.6;

const dieProfile = [
  new THREE.Vector2(DIE_OUTER_R, -dh),
  new THREE.Vector2(DIE_ENTRY_R, -dh + 3),
  new THREE.Vector2(DIE_BORE_R, -dh * 0.1),
  new THREE.Vector2(DIE_BORE_R, dh * 0.65),
  new THREE.Vector2(DIE_BORE_R * 1.2, dh),
  new THREE.Vector2(DIE_OUTER_R, dh),
  new THREE.Vector2(DIE_OUTER_R, -dh),
];
const dieGeo = new THREE.LatheGeometry(dieProfile, 64);
const die = new THREE.Mesh(dieGeo, dieMat);
die.rotation.x = Math.PI / 2;
die.position.set(0, 0, cfg.z_lay);
die.castShadow = true;
scene.add(die);

// ---------- Takeup spool ----------
const TAKEUP_CORE_R = 25;
const TAKEUP_H = 90;
const TAKEUP_FLANGE_R = 100;
const TAKEUP_FLANGE_H = 6;
const TAKEUP_WOUND_MAX_R = 85;

const takeupGroup = new THREE.Group();
takeupGroup.position.set(0, 0, cfg.z1 + 30);
takeupGroup.rotation.z = Math.PI / 2;

const takeupSpin = new THREE.Group();
takeupGroup.add(takeupSpin);

const takeupCore = new THREE.Mesh(
  new THREE.CylinderGeometry(TAKEUP_CORE_R, TAKEUP_CORE_R, TAKEUP_H, 28),
  takeupBodyMat,
);
takeupCore.castShadow = true;

const takeupFlangeMat = new THREE.MeshPhysicalMaterial({
  color: 0x5a5248, metalness: 0.65, roughness: 0.3,
});
const takeupFlA = new THREE.Mesh(
  new THREE.CylinderGeometry(TAKEUP_FLANGE_R, TAKEUP_FLANGE_R, TAKEUP_FLANGE_H, 28),
  takeupFlangeMat,
);
takeupFlA.position.y = -(TAKEUP_H - TAKEUP_FLANGE_H) / 2;
takeupFlA.castShadow = true;

const takeupFlB = new THREE.Mesh(
  new THREE.CylinderGeometry(TAKEUP_FLANGE_R, TAKEUP_FLANGE_R, TAKEUP_FLANGE_H, 28),
  takeupFlangeMat,
);
takeupFlB.position.y = (TAKEUP_H - TAKEUP_FLANGE_H) / 2;
takeupFlB.castShadow = true;

const takeupWound = new THREE.Mesh(new THREE.BufferGeometry(), woundCableMat);
takeupWound.castShadow = true;
takeupWound.receiveShadow = true;

takeupSpin.add(takeupCore, takeupFlA, takeupFlB, takeupWound);
scene.add(takeupGroup);

// ---------- Lead-in cable to takeup ----------
const leadInCurve = new THREE.LineCurve3(
  new THREE.Vector3(0, 0, cfg.z1),
  new THREE.Vector3(0, 0, cfg.z1 + 30),
);
const leadInGeo = new THREE.TubeGeometry(leadInCurve, 8, cfg.R_layer + cfg.outer_radius, 18, false);
const leadIn = new THREE.Mesh(leadInGeo, woundCableMat);
leadIn.castShadow = true;
scene.add(leadIn);

// ---------- Update loop ----------
// wrap tube: 多層レベルワインドのヘリックス点列から TubeGeometry を再構築。
// curveType="catmullrom" の張力でなめらかに繋ぐが、ヘリックスは元々滑らかなので殆ど誤差ゼロ。
function updateWoundTube(mesh, pts, tube_r, radial_segs) {
  if (pts.length < 4) {
    mesh.visible = false;
    return;
  }
  mesh.visible = true;
  const vs = pts.map(p => new THREE.Vector3(p.x, p.y, p.z));
  const curve = new THREE.CatmullRomCurve3(vs, false, "catmullrom", 0.0);
  const segs = Math.max(64, Math.min(2000, pts.length * 2));
  const geo = new THREE.TubeGeometry(curve, segs, tube_r, radial_segs, false);
  mesh.geometry.dispose();
  mesh.geometry = geo;
}

// bobbin の巻線可用範囲（軸方向 H_wind、内径 R_body、外径 R_max）
const BOBBIN_WIND_R_INNER = BOBBIN_BODY_R + 0.4;
const BOBBIN_WIND_R_OUTER = BOBBIN_WIND_R_MAX;

// takeup の巻線可用範囲（cable bundle 用、ピッチも大きい）
const TAKEUP_WIND_R_INNER = TAKEUP_CORE_R + 1.0;
const TAKEUP_WIND_R_OUTER = TAKEUP_WOUND_MAX_R;
const TAKEUP_WIND_H = TAKEUP_H - 2 * TAKEUP_FLANGE_H - 4;

let takeupTheta = 0;
let prevT = 0;
function resetTakeupAccum() {
  takeupTheta = 0;
  prevT = 0;
  for (const a of bobbinAssemblies) a.spinTheta = 0;
}

// 現在の巻線最外層半径を推定（spin 速度 Ω = v_strand / R_w に使う）
function estimateOuterWindRadius(L, R_core, R_outer, H_wind, wire_d) {
  // 解析的: L_cap(n_layers) = Σ_{k=0..n-1} turns_per_layer * 2π * (R_core + (k+0.5)*step)
  const pitch = wire_d * 1.05;
  const step = wire_d * 0.96;
  const tpl = Math.max(2, Math.floor((H_wind - pitch) / pitch));
  let acc = 0;
  let layer = 0;
  while (true) {
    const r = R_core + (layer + 0.5) * step;
    if (r > R_outer) return Math.min(R_outer, R_core + (layer - 0.5) * step + 0.5);
    const layer_cap = tpl * 2 * Math.PI * r;
    if (acc + layer_cap >= L) return r;
    acc += layer_cap;
    layer += 1;
    if (layer > 200) return R_outer;
  }
}

const PHYSICS_SUBSTEPS = 4;

// 払い出し端 (bobbin local) を spin → orbit で world 座標へ持ち上げる。
// spinTheta は current frame の値を全 substep に適用（spin 速度は orbital ω より遥かに遅い）。
// tau は orbital 位相のみ補間：bobbinTransform(cfg, i, tau)。
const _tmpAxis = new THREE.Vector3();
const _tmpQuat = new THREE.Quaternion();
const _tmpVec = new THREE.Vector3();
function bobbinHelixEndWorld(i, spinTheta, localEnd, tau) {
  const c = Math.cos(spinTheta);
  const s = Math.sin(spinTheta);
  // spin around local +y
  const sx = localEnd.x * c + localEnd.z * s;
  const sy = localEnd.y;
  const sz = -localEnd.x * s + localEnd.z * c;
  // orbit quaternion + position（bobbinTransform.axis は -cos φ, -sin φ, 0）
  const tf = bobbinTransform(cfg, i, tau);
  _tmpAxis.set(tf.axis.x, tf.axis.y, tf.axis.z).normalize();
  _tmpQuat.setFromUnitVectors(yHat, _tmpAxis);
  _tmpVec.set(sx, sy, sz).applyQuaternion(_tmpQuat);
  return [
    _tmpVec.x + tf.position.x,
    _tmpVec.y + tf.position.y,
    _tmpVec.z + tf.position.z,
  ];
}

// 局所接線（方向のみ）を spin → orbit で world に持ち上げる（並進なし）。
// helix 末端での払い出し接線 → 鎖の起点接線 pin に使う。
function bobbinDirWorld(i, spinTheta, localDir, tau) {
  const c = Math.cos(spinTheta);
  const s = Math.sin(spinTheta);
  const sx = localDir.x * c + localDir.z * s;
  const sy = localDir.y;
  const sz = -localDir.x * s + localDir.z * c;
  const tf = bobbinTransform(cfg, i, tau);
  _tmpAxis.set(tf.axis.x, tf.axis.y, tf.axis.z).normalize();
  _tmpQuat.setFromUnitVectors(yHat, _tmpAxis);
  _tmpVec.set(sx, sy, sz).applyQuaternion(_tmpQuat);
  return [_tmpVec.x, _tmpVec.y, _tmpVec.z];
}

function updateScene(t, frameDt) {
  const state = strandState(cfg, t);
  const dt = Math.max(0, t - prevT);
  prevT = t;

  // ---- supply 側 multi-layer wind helix を物理より先に構築 ----
  // すべての supply bobbin は L_supply を共有（material conservation の単純化）。
  const wire_d_supply = 2 * cfg.outer_radius;
  const R_w_outer = estimateOuterWindRadius(
    state.L_supply, BOBBIN_WIND_R_INNER, BOBBIN_WIND_R_OUTER,
    BOBBIN_WIND_H, wire_d_supply,
  );
  const omega_bobbin = (state.phase < 3 && R_w_outer > 1e-3)
    ? state.v_strand / R_w_outer : 0;
  const supplyPts = buildLevelWindHelix(
    state.L_supply, BOBBIN_WIND_R_INNER, BOBBIN_WIND_R_OUTER,
    BOBBIN_WIND_H, wire_d_supply,
  );
  // 全 bobbin の self-spin を frame-incremental に進める
  for (const asm of bobbinAssemblies) {
    asm.spinTheta += dt * omega_bobbin;
  }
  // 払い出し端の bobbin local 座標（wound mesh の y=BOBBIN_BODY_H/2 オフセット込み）
  // 払い切り (supplyPts 空) 時は body 中央軸上に fallback
  const supplyEndLocal = (supplyPts.length > 0)
    ? {
        x: supplyPts[supplyPts.length - 1].x,
        y: supplyPts[supplyPts.length - 1].y + BOBBIN_BODY_H / 2,
        z: supplyPts[supplyPts.length - 1].z,
      }
    : { x: 0, y: BOBBIN_BODY_H / 2, z: 0 };
  // 払い出し接線（方向のみ。並進なし）: 末端と 1 つ手前の点から差分で近似。
  // 長さ 2 未満なら接線情報なし → 起点 pin は 1 点のみ
  let supplyTangentLocal = null;
  if (supplyPts.length >= 2) {
    const p_last = supplyPts[supplyPts.length - 1];
    const p_prev = supplyPts[supplyPts.length - 2];
    const tx = p_last.x - p_prev.x;
    const ty = p_last.y - p_prev.y;
    const tz = p_last.z - p_prev.z;
    const tm = Math.hypot(tx, ty, tz);
    if (tm > 1e-6) {
      supplyTangentLocal = { x: tx / tm, y: ty / tm, z: tz / tm };
    }
  }

  if (params.physicsOn) {
    const subDt = Math.max(0, Math.min(1 / 60, frameDt)) / PHYSICS_SUBSTEPS;

    // ---- 駆動張力 + 遠心 wrap 由来 pretension で実効 slack を縮める ----
    const drive = params.driveTension * DRIVE_STRAIN_MAX;   // ε_drive
    const omega2 = cfg.omega * cfg.omega;
    const eps_omega = OMEGA_TAUT * omega2;   // 遠心 strain
    // ω→運転時に damping を上げて慣性振動を抑える（ボビン軸 brake/摩擦）
    const omegaNorm = Math.min(1, Math.abs(cfg.omega) / 4);
    const dampRun = DAMPING_REST + (DAMPING_RUN - DAMPING_REST) * omegaNorm;
    cableBundle.damping = dampRun;
    coreSupplyChain.damping = dampRun;
    for (const c of chains) if (c) c.damping = dampRun;
    // パチもん EI を chain type 別に: bundle 1.5×, core 0.6×, 外周 1.0×
    cableBundle.bendK = params.bendStiffness * 1.5;
    coreSupplyChain.bendK = params.bendStiffness * 0.6;
    for (const c of chains) if (c) c.bendK = params.bendStiffness;

    const slackBundle = Math.max(
      SLACK_FLOOR,
      wrapPretensionSlack(CABLE_SLACK, cfg, params.k_omega) - drive * BUNDLE_DRIVE_GAIN - eps_omega,
    );
    cableBundle.restLength =
      (cfg.z1 - cfg.z_lay) / (CABLE_PARTICLES - 1) * slackBundle;
    const slackCore = Math.max(
      SLACK_FLOOR,
      wrapPretensionSlack(CORE_SLACK, cfg, params.k_omega) - drive - eps_omega,
    );
    coreSupplyChain.restLength =
      (cfg.z_lay - cfg.z0) / (CORE_PARTICLES - 1) * slackCore;

    // ---- cable bundle: 端点固定でだらしなく揺らす ----
    const cableStart = [0, 0, cfg.z_lay];
    const cableEnd = [0, 0, cfg.z1];
    for (let s = 0; s < PHYSICS_SUBSTEPS; s++) {
      cableBundle.step(subDt, cableStart, cableEnd);
    }
    const bundlePts = cableBundle.toPoints();
    const frames = computeBishopFrames(bundlePts);
    const arclens = cumulativeArcLength(bundlePts);
    const e1_0 = frames.e1s[0];
    const e2_0 = frames.e2s[0];

    // ---- 外周ストランド: upstream chain + bundle 周りの撚り offset ----
    const slackOuter = Math.max(SLACK_FLOOR, CHAIN_SLACK - drive - eps_omega);
    for (let i = 0; i < cfg.n_outer; i++) {
      const chain = chains[i];
      if (!chain) continue;
      const phi0 = (2 * Math.PI * i) / cfg.n_outer;
      const theta_0 = phi0 - cfg.omega * t;
      const ce0 = Math.cos(theta_0), se0 = Math.sin(theta_0);
      // bundle 起点での実際の helix offset = chain 終端ピン（一筆書き連続性）
      const layEntry = [
        bundlePts[0].x + cfg.R_layer * (ce0 * e1_0[0] + se0 * e2_0[0]),
        bundlePts[0].y + cfg.R_layer * (ce0 * e1_0[1] + se0 * e2_0[1]),
        bundlePts[0].z + cfg.R_layer * (ce0 * e1_0[2] + se0 * e2_0[2]),
      ];
      // chord 長（bobbin orbit + level-wind 払い出し端で時々刻々変わる）から rest 再計算
      const asm = bobbinAssemblies[i];
      const bobbinAnchorNow = bobbinHelixEndWorld(i, asm.spinTheta, supplyEndLocal, t);
      const chord = Math.hypot(
        layEntry[0] - bobbinAnchorNow[0],
        layEntry[1] - bobbinAnchorNow[1],
        layEntry[2] - bobbinAnchorNow[2],
      );
      chain.restLength = chord / (PARTICLES_PER_STRAND - 1) * slackOuter;
      const tangentRest = chain.restLength;
      for (let s = 0; s < PHYSICS_SUBSTEPS; s++) {
        const tau = t - (PHYSICS_SUBSTEPS - 1 - s) * subDt;
        const bobbinAnchor = bobbinHelixEndWorld(i, asm.spinTheta, supplyEndLocal, tau);
        let pinStart1 = null;
        if (supplyTangentLocal !== null) {
          const tan = bobbinDirWorld(i, asm.spinTheta, supplyTangentLocal, tau);
          pinStart1 = [
            bobbinAnchor[0] + tan[0] * tangentRest,
            bobbinAnchor[1] + tan[1] * tangentRest,
            bobbinAnchor[2] + tan[2] * tangentRest,
          ];
        }
        chain.step(subDt, bobbinAnchor, layEntry, { pinStart1 });
      }
      const chainPts = chain.toPoints();
      // downstream helix（bundle に沿った撚り offset、t_pass 整合）
      const N = bundlePts.length;
      const downstream = new Array(N);
      for (let k = 0; k < N; k++) {
        const theta = phi0 - cfg.omega * t + (2 * Math.PI * arclens[k]) / cfg.L_pitch;
        const ck = Math.cos(theta), sk = Math.sin(theta);
        const e1k = frames.e1s[k], e2k = frames.e2s[k];
        downstream[k] = {
          x: bundlePts[k].x + cfg.R_layer * (ck * e1k[0] + sk * e2k[0]),
          y: bundlePts[k].y + cfg.R_layer * (ck * e1k[1] + sk * e2k[1]),
          z: bundlePts[k].z + cfg.R_layer * (ck * e1k[2] + sk * e2k[2]),
        };
      }
      chainPts.pop();   // chain 終端 = downstream[0] と一致しているので間引く
      const combined = chainPts.concat(downstream);
      // 抜けた鎖は弧長が path に沿わないので trim 無効化（whip 形をそのまま描画）
      const trimmed = chain.detached ? combined : trimCenterlineByArc(combined, state.s_trail);
      if (trimmed.length < 2) {
        outerMeshes[i].visible = false;
      } else {
        outerMeshes[i].visible = true;
        updateTube(outerMeshes[i], toVec3Array(trimmed), cfg.outer_radius);
      }
    }

    // ---- core: upstream PBD chain（軸上だらり）+ downstream は bundle そのもの ----
    const coreUpStart = [0, 0, cfg.z0];
    const coreUpEnd = [0, 0, cfg.z_lay];
    for (let s = 0; s < PHYSICS_SUBSTEPS; s++) {
      coreSupplyChain.step(subDt, coreUpStart, coreUpEnd);
    }
    const coreUpPts = coreSupplyChain.toPoints();
    coreUpPts.pop();   // 終端 = bundle[0] と一致
    updateTube(coreMesh, toVec3Array(coreUpPts.concat(bundlePts)), cfg.core_radius);
  } else {
    updateTube(coreMesh, toVec3Array(coreStrandCenterline(cfg)), cfg.core_radius);
    for (let i = 0; i < cfg.n_outer; i++) {
      const full = outerStrandCenterline(cfg, i, t);
      const trimmed = trimCenterlineByArc(full, state.s_trail);
      if (trimmed.length < 2) {
        outerMeshes[i].visible = false;
      } else {
        outerMeshes[i].visible = true;
        updateTube(outerMeshes[i], toVec3Array(trimmed), cfg.outer_radius);
      }
    }
  }

  // ---- 供給ボビン描画: helix / spinTheta は updateScene 冒頭で計算済み ----
  for (let i = 0; i < cfg.n_outer; i++) {
    const asm = bobbinAssemblies[i];
    placeBobbin(asm, bobbinTransform(cfg, i, t));
    asm.spin.rotation.y = asm.spinTheta;
    updateWoundTube(asm.wound, supplyPts, cfg.outer_radius, 8);
  }

  // ---- 巻取りボビン: bundle ケーブルの level-wind wrap ----
  // bundle 長 = L_consumed / α_helix（各 strand の長さ：cable axial = strand_arc / α）
  const alpha_helix = Math.hypot(1, 2 * Math.PI * cfg.R_layer / cfg.L_pitch);
  const cable_outer_r = cfg.R_layer + cfg.outer_radius;
  const bundle_d = 2 * cable_outer_r;
  const L_bundle = state.L_takeup / alpha_helix;
  const takeupPts = buildLevelWindHelix(
    L_bundle, TAKEUP_WIND_R_INNER, TAKEUP_WIND_R_OUTER,
    TAKEUP_WIND_H, bundle_d, { segs_per_turn: 36 },
  );
  updateWoundTube(takeupWound, takeupPts, cable_outer_r, 12);

  // ---- 巻取りスプール自転: dθ = v_feed · dt / R_t ----
  const R_t = estimateOuterWindRadius(
    L_bundle, TAKEUP_WIND_R_INNER, TAKEUP_WIND_R_OUTER,
    TAKEUP_WIND_H, bundle_d,
  );
  const v_feed = state.v_strand / alpha_helix;
  if (state.phase < 3 && R_t > 1e-3) {
    takeupTheta += dt * v_feed / R_t;
  }
  takeupSpin.rotation.y = takeupTheta;
}

// ---------- Post-processing ----------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.0, 0.55, 0.92,
);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// ---------- GUI ----------
const params = {
  L_pitch: cfg.L_pitch,
  omega: cfg.omega,
  n_outer: cfg.n_outer,
  R_layer: cfg.R_layer,
  L_total: cfg.L_total,
  physicsOn: true,
  k_omega: 3e-4,   // wrap 由来 pretension の強さ。物理的には 1/T_base 単位
  driveTension: 0.5,   // ボビン牽引 + 摩擦 brake を boundary tension として近似（0..1）
  bendStiffness: 0.15,   // パチもん EI: Laplacian 引き寄せ。0=純鎖, 0.3 で十分剛い
  paused: false,
  speed: 1.0,
  time: 0,
  bloom: 0.0,
  exposure: renderer.toneMappingExposure,
  autoRotate: false,
  rotateSpeed: 1.0,
  rewindReset() { simT = 0; resetTakeupAccum(); },
  resetCamera() {
    controls.target.set(0, 0, zMid);
    camera.position.set(-machineLen * 0.95, cfg.R_bobbin * 1.4, zMid);
    controls.update();
  },
  viewTop() {
    controls.target.set(0, 0, zMid);
    camera.position.set(0, machineLen * 1.2, zMid);
    controls.update();
  },
  viewSide() {
    controls.target.set(0, 0, zMid);
    camera.position.set(-machineLen * 1.2, 0, zMid);
    controls.update();
  },
  viewLayPlate() {
    controls.target.set(0, 0, cfg.z_lay);
    camera.position.set(-cfg.R_bobbin * 0.8, cfg.R_bobbin * 0.3, cfg.z_lay + cfg.R_bobbin * 0.2);
    controls.update();
  },
  viewTip() {
    controls.target.set(0, 0, cfg.z1 - 50);
    camera.position.set(-40, 20, cfg.z1 + 30);
    controls.update();
  },
};

const gui = new GUI({ title: "strander" });

// 最上位に常時見える巻替リセット
gui.add(params, "rewindReset").name("▶ 巻替リセット (t=0)");

const fMachine = gui.addFolder("機械パラメータ");
fMachine.add(params, "L_pitch", 15, 150, 1).name("ピッチ [mm]").onChange((v) => { cfg.L_pitch = v; resetTakeupAccum(); });
fMachine.add(params, "omega", -8, 8, 0.1).name("ω [rad/s]").onChange((v) => { cfg.omega = v; resetTakeupAccum(); });
fMachine.add(params, "n_outer", [3, 4, 6, 8, 12]).name("外周本数").onChange((v) => {
  cfg.n_outer = parseInt(v);
  rebuildOuterMats(cfg.n_outer);
  ensureOuterMeshes(cfg.n_outer);
  ensureBobbins(cfg.n_outer);
  rebuildChains();
});
fMachine.add(params, "R_layer", 3, 25, 0.5).name("layer R [mm]").onChange((v) => { cfg.R_layer = v; resetTakeupAccum(); });
fMachine.add(params, "L_total", 500, 16000, 100).name("素線長 L_total [mm]").onChange((v) => { cfg.L_total = v; });

const fPlay = gui.addFolder("再生");
fPlay.add(params, "speed", 0, 5, 0.1).name("速度");
fPlay.add(params, "paused").name("一時停止");
fPlay.add(params, "time", 0, 240, 0.1).name("時間 [s]").listen().onChange((v) => { simT = v; resetTakeupAccum(); });
fPlay.add(params, "physicsOn").name("素線 PBD 物理").onChange((v) => { if (v) { rebuildChains(); rebuildBundle(); rebuildCoreChain(); } });
fPlay.add(params, "k_omega", 0, 2e-3, 1e-5).name("wrap pretension k_ω");
fPlay.add(params, "driveTension", 0, 1, 0.01).name("駆動張力 (ボビン牽引)");
fPlay.add(params, "bendStiffness", 0, 1.0, 0.01).name("パチもん EI");
fPlay.open();

const fVisual = gui.addFolder("ビジュアル");
fVisual.add(params, "bloom", 0, 1.5, 0.01).name("bloom").onChange((v) => { bloom.strength = v; });
fVisual.add(params, "exposure", 0.3, 2.5, 0.01).name("露出").onChange((v) => { renderer.toneMappingExposure = v; });

const fCam = gui.addFolder("カメラ");
fCam.add(params, "autoRotate").name("自動回転").onChange((v) => { controls.autoRotate = v; });
fCam.add(params, "rotateSpeed", 0.2, 5, 0.1).name("回転速度").onChange((v) => { controls.autoRotateSpeed = v; });
fCam.add(params, "resetCamera").name("初期位置");
fCam.add(params, "viewTop").name("俯瞰");
fCam.add(params, "viewSide").name("側面");
fCam.add(params, "viewLayPlate").name("ダイス近景");
fCam.add(params, "viewTip").name("先端ドリーイン");

// ---------- Main loop ----------
let last = performance.now();
let simT = 10;
const statusEl = document.getElementById("status");

function updateStatusHUD(t) {
  const s = strandState(cfg, t);
  const phaseTxt = s.phase === 1 ? "1: 通常稼働"
                  : s.phase === 2 ? "2: サプライ枯渇・後端後退"
                  : "3: 全量回収・停止";
  const pct = (100 * s.L_consumed / cfg.L_total).toFixed(1);
  statusEl.innerHTML =
    `<span class="phase">phase ${phaseTxt}</span> &middot; t=${t.toFixed(1)}s / t_total=${s.t_total.toFixed(1)}s<br>` +
    `素線消費 ${pct}% &middot; supply ${s.L_supply.toFixed(0)}mm &middot; transit ${s.L_transit.toFixed(0)}mm &middot; takeup ${s.L_takeup.toFixed(0)}mm<br>` +
    `L_path=${s.L_path.toFixed(0)}mm &middot; v_strand=${s.v_strand.toFixed(2)}mm/s &middot; 巻取り角=${(takeupTheta % (2*Math.PI) * 180/Math.PI).toFixed(0)}°`;
}

function tick() {
  const now = performance.now();
  const dt = Math.min(1 / 30, (now - last) / 1000);   // 長フレーム時の発散ガード
  last = now;
  if (!params.paused) simT += dt * params.speed;
  params.time = simT;
  updateScene(simT, dt * params.speed);
  updateStatusHUD(simT);
  controls.update();
  composer.render();
  requestAnimationFrame(tick);
}
tick();

// ---------- Resize ----------
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
  bloom.setSize(window.innerWidth, window.innerHeight);
});
