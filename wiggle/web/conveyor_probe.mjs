// 焼き込みコンベア probe: 連続給線（continuous pay-out）の物理検証（headless）。
//
// 動機（capstan_probe.mjs での確定事項）:
//   両端ピン + 有限長 = 総弧長が保存量 → 巻き締めの余長が閉じた長さ予算の中で
//   r≈386mm の常駐ループに化ける。供給端をどうピン留めし直しても消えない。
//   実機は供給リールが連続的に線材を *繰り出す* → 余長が消費されて解消する。
//
// 本モデル（焼き込みコンベア, N 一定）:
//   - index 0   = 公転継ぎ目（seam(θ)）。take-up の prime mover（capstan_probe で実証済み）。
//   - index N-1 = feed（世界固定）。supply からの繰り出し口。
//   - デマンド給線: リードインが張る（feed 隣接セグメントが rest 超で伸びる＝巻側が引いている）
//     と 1 粒子 pay-out。同時に最古の巻き済み粒子（index 1）を静的メッシュへ焼き込んで
//     アクティブ鎖から外す（emit と bake を対にして N 一定）。
//   - 焼き込み点は drum-local（−θ 回転）で保存 → 描画時に drum と一緒に回って巻きヘリックスになる。
//
// 検証ゲート:
//   (1) 消費: baked 数が単調増加・tail 弧長が有界（ループ化しない）・rmax 有界
//   (2) 座り: baked 点が全て r≈R_lay（41.8±tol）かつ |y|<innerY（フランジ内）
//   (3) 健全: アクティブ鎖 minRpt≥drum, カクつき<15°, 伸び≈1
//
// 実行: node wiggle/web/conveyor_probe.mjs 2>&1 | tee /tmp/conveyor-$(date +%s).log

import { StrandChain } from "./physics.js";
import { defaultCfg, estimateCapacity, wrappedArcLength, bobbinContact } from "./winding_core.js";

const cfg = defaultCfg();
cfg.L_capacity = estimateCapacity(cfg);

const R_lay = cfg.BODY_R + cfg.wire_r;           // 41.8
const circ = 2 * Math.PI * R_lay;                 // ≈262.6 mm/周
const innerY = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
const minR_body = R_lay;
const bandR_wrap = cfg.BODY_R + cfg.wire_r + cfg.wire_r * 3;   // 47.2
const bandR_wrap2 = bandR_wrap * bandR_wrap;
const FEED = [0, 0, 200];
const pitch = 2 * cfg.wire_r * 1.05;              // 3.78
const TRAV_LO = -(innerY - cfg.wire_r - 2);       // -58.2
const TRAV_HI = (innerY - cfg.wire_r - 2);        // +58.2
const TRAV_SPAN = TRAV_HI - TRAV_LO;

function seamY(theta) {
  const dist = (Math.abs(theta) / (2 * Math.PI)) * pitch;
  const period = 2 * TRAV_SPAN;
  let m = dist % period;
  if (m < 0) m += period;
  return m <= TRAV_SPAN ? TRAV_LO + m : TRAV_LO + (period - m);
}
function seam(theta) { return [R_lay * Math.sin(theta), seamY(theta), R_lay * Math.cos(theta)]; }

// ---- アクティブ鎖サイズ: 巻側バッファ ~1.5 周 + リードイン ----
const segLen = 3.0;
const woundTurns = parseFloat(process.env.WTURN ?? "0.3");   // lay 点近傍の座り曲線ぶん（巻きは baked 側）
const standoff = 180;
const N = Math.round((woundTurns * circ + standoff) / segLen);
const segRest = segLen;
const leadInParticles = Math.round(standoff / segLen);

const chain = new StrandChain(N, segRest, {
  gravity: [0, 0, -9810 * parseFloat(process.env.GRAV ?? "1")],
  damping: 0.99,
  iters: 12,
  restLengths: new Float64Array(N - 1).fill(segRest),
});
chain.bendK = parseFloat(process.env.BENDK ?? "0.04");

// 初期化: index 0=seam(θ=0) から巻側へ R_lay の tight ヘリックスを woundTurns 周ぶん（θ を負側へ）、
// 続けて lead-in を feed まで直線。初期からタイトに座らせて初期 slack ループを作らない。
const woundParticles = N - leadInParticles;
const wEnd = (() => {
  // 巻側末端（index woundParticles-1）の位置: θ = -(woundTurns*2π) ぶん巻いた点
  const phi = -(woundTurns * 2 * Math.PI);
  return [R_lay * Math.sin(phi), seamY(phi), R_lay * Math.cos(phi)];
})();
for (let i = 0; i < N; i++) {
  let p;
  if (i < woundParticles) {
    const f = i / Math.max(1, woundParticles - 1);   // 0..1
    const phi = -(f * woundTurns * 2 * Math.PI);       // seam(0) から負側へ
    p = [R_lay * Math.sin(phi), seamY(phi), R_lay * Math.cos(phi)];
  } else {
    const f = (i - (woundParticles - 1)) / leadInParticles;  // 0..1
    p = [wEnd[0] + (FEED[0] - wEnd[0]) * f, wEnd[1] + (FEED[1] - wEnd[1]) * f, wEnd[2] + (FEED[2] - wEnd[2]) * f];
  }
  const k = 3 * i;
  chain.pos[k] = p[0]; chain.pos[k + 1] = p[1]; chain.pos[k + 2] = p[2];
  chain.prev[k] = p[0]; chain.prev[k + 1] = p[1]; chain.prev[k + 2] = p[2];
}

const ctx = {
  cfg, legacyStick: 0,
  mu: parseFloat(process.env.MU ?? "0.8"),
  seatGrip: parseFloat(process.env.SEATGRIP ?? "0.5"),
  frictionDtheta: 0, segmentContact: true, eye: null, guide: null,
};

// バックテンション: リードイン（接線点 B→feed）を弦に向け直線化（capstan_probe で検証）
const BT = parseFloat(process.env.BT ?? "0.2");
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
  const dx = pos[3 * (n - 1)] - Bx, dy = pos[3 * (n - 1) + 1] - By, dz = pos[3 * (n - 1) + 2] - Bz;
  const L2 = dx * dx + dy * dy + dz * dz;
  if (L2 < 1e-9) return;
  for (let i = B + 1; i < n - 1; i++) {
    if (fixed[i]) continue;
    const k = 3 * i;
    const t = Math.max(0, Math.min(1, ((pos[k] - Bx) * dx + (pos[k + 1] - By) * dy + (pos[k + 2] - Bz) * dz) / L2));
    pos[k] += (Bx + dx * t - pos[k]) * gain;
    pos[k + 1] += (By + dy * t - pos[k + 1]) * gain;
    pos[k + 2] += (Bz + dz * t - pos[k + 2]) * gain;
  }
}
const frictionStep = (pos, n, fixed, prev) => { ctx.skipFriction = false; bobbinContact(pos, n, fixed, prev, ctx); };
const contactProject = (pos, n, fixed, prev) => {
  ctx.skipFriction = true; bobbinContact(pos, n, fixed, prev, ctx);
  applyBackTension(pos, n, fixed, BT);
};

// ---- 焼き込みコンベア ----
const baked = [];   // drum-local 座標 [{x,y,z}]（描画時 +θ で回す）
let theta = 0;
let payAccum = 0;   // take-up 累積長 [mm]。R_lay·Δθ ぶん貯まるごとに 1 粒子 pay-out + bake。
function bakeAndEmit() {
  // (1) lay 点 = index 0（常に seam 上＝r=R_lay に座る）を drum-local で焼き込み
  const c = Math.cos(-theta), s = Math.sin(-theta);   // world → drum-local（−θ 回転 about y）
  const wx = chain.pos[0], wy = chain.pos[1], wz = chain.pos[2];
  baked.push({ x: c * wx + s * wz, y: wy, z: -s * wx + c * wz });
  // (2) index 1..N-1 を 0..N-2 へシフト（lay 点を消費 → リードインが 1 粒子前進）
  for (let i = 0; i < N - 1; i++) {
    const d = 3 * i, src = 3 * (i + 1);
    chain.pos[d] = chain.pos[src]; chain.pos[d + 1] = chain.pos[src + 1]; chain.pos[d + 2] = chain.pos[src + 2];
    chain.prev[d] = chain.prev[src]; chain.prev[d + 1] = chain.prev[src + 1]; chain.prev[d + 2] = chain.prev[src + 2];
  }
  // (3) feed に新粒子を繰り出し（index N-1 = FEED）
  const kf = 3 * (N - 1);
  chain.pos[kf] = FEED[0]; chain.pos[kf + 1] = FEED[1]; chain.pos[kf + 2] = FEED[2];
  chain.prev[kf] = FEED[0]; chain.prev[kf + 1] = FEED[1]; chain.prev[kf + 2] = FEED[2];
}

const dt = 1 / 60, SUBSTEPS = 4, subDt = dt / SUBSTEPS;
const STEPS = parseInt(process.env.STEPS ?? "3000", 10);

function activeStats() {
  let minRpt = Infinity, kinkSum = 0, kinkN = 0, arc = 0;
  for (let i = 0; i < N; i++) {
    if (Math.abs(chain.pos[3 * i + 1]) >= innerY) continue;
    const r = Math.hypot(chain.pos[3 * i], chain.pos[3 * i + 2]);
    if (r < minRpt) minRpt = r;
  }
  for (let i = 1; i < N - 1; i++) {
    const a = 3 * (i - 1), b = 3 * i, cc = 3 * (i + 1);
    const ux = chain.pos[b] - chain.pos[a], uy = chain.pos[b + 1] - chain.pos[a + 1], uz = chain.pos[b + 2] - chain.pos[a + 2];
    const vx = chain.pos[cc] - chain.pos[b], vy = chain.pos[cc + 1] - chain.pos[b + 1], vz = chain.pos[cc + 2] - chain.pos[b + 2];
    const lu = Math.hypot(ux, uy, uz), lv = Math.hypot(vx, vy, vz);
    if (lu < 1e-9 || lv < 1e-9) continue;
    let ca = (ux * vx + uy * vy + uz * vz) / (lu * lv); ca = Math.max(-1, Math.min(1, ca));
    kinkSum += Math.acos(ca); kinkN++;
  }
  for (let i = 0; i < N - 1; i++) { const a = 3 * i, b = 3 * (i + 1); arc += Math.hypot(chain.pos[b] - chain.pos[a], chain.pos[b + 1] - chain.pos[a + 1], chain.pos[b + 2] - chain.pos[a + 2]); }
  return { minRpt, kinkDeg: kinkN ? (kinkSum / kinkN) * 180 / Math.PI : 0, stretch: arc / ((N - 1) * segRest) };
}
function leadInLen() {
  let B = -1;
  for (let i = N - 2; i >= 0; i--) { const k = 3 * i; if (chain.pos[k] * chain.pos[k] + chain.pos[k + 2] * chain.pos[k + 2] <= bandR_wrap2 && Math.abs(chain.pos[k + 1]) < innerY) { B = i; break; } }
  if (B < 0) B = 0;
  let len = 0; for (let i = B; i < N - 1; i++) { const a = 3 * i, b = 3 * (i + 1); len += Math.hypot(chain.pos[b] - chain.pos[a], chain.pos[b + 1] - chain.pos[a + 1], chain.pos[b + 2] - chain.pos[a + 2]); }
  let rmax = 0; for (let i = 0; i < N - 1; i++) { const k = 3 * i; const r = Math.hypot(chain.pos[k], chain.pos[k + 2]); if (r > rmax) rmax = r; }
  return { len, rmax };
}
function bakedStats() {
  if (!baked.length) return { n: 0, rMean: 0, rMin: 0, rMax: 0, yMin: 0, yMax: 0, nBadR: 0, nBadY: 0 };
  let rSum = 0, rMin = Infinity, rMax = 0, yMin = Infinity, yMax = -Infinity, nBadR = 0, nBadY = 0;
  for (const p of baked) {
    const r = Math.hypot(p.x, p.z);
    rSum += r; if (r < rMin) rMin = r; if (r > rMax) rMax = r;
    if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y;
    if (Math.abs(r - R_lay) > 2.0) nBadR++;
    if (Math.abs(p.y) >= innerY) nBadY++;
  }
  return { n: baked.length, rMean: rSum / baked.length, rMin, rMax, yMin, yMax, nBadR, nBadY };
}
// 巻きが本当にヘリックスとして「巻き付いている」か = drum-local 角度の累積走行量。
// 退行バグ（co-rotating seam を −θ で焼き込み）では全点が角度 0 に潰れて travel≈0。
// 健全なら baked 周数 × 360° に比例して増える。これが r/y 統計では捕捉できない欠落軸。
function bakedAngleStats() {
  if (baked.length < 2) return { travelDeg: 0, spanDeg: 0 };
  let prev = Math.atan2(baked[0].x, baked[0].z), travel = 0, aMin = prev, aMax = prev;
  for (let i = 1; i < baked.length; i++) {
    const a = Math.atan2(baked[i].x, baked[i].z);
    let d = a - prev; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
    travel += Math.abs(d); prev = a;
    if (a < aMin) aMin = a; if (a > aMax) aMax = a;
  }
  return { travelDeg: travel * 180 / Math.PI, spanDeg: (aMax - aMin) * 180 / Math.PI };
}

console.log(`# 焼き込みコンベア probe  N=${N}（巻${woundParticles}+lead${leadInParticles}） segRest=${segRest} woundTurns=${woundTurns}`);
console.log(`# μ=${ctx.mu} seatGrip=${ctx.seatGrip} BT=${BT} grav=z`);
console.log(`# step | 自転 | baked数 | baked rMean | baked y幅 | lead弧 | rmax | minRpt | カクつき | 伸び`);

// lay 点モデル: 既定 = world 固定（feed 側 +z 子午線, 角度 0）= ビューワ(capstan.js)と同一。
// COROT=1 で旧 co-rotating seam(θ) を再現（焼き込みが drum-local 角度 0 に潰れる退行バグ）。
const COROT = process.env.COROT === "1";
for (let s = 1; s <= STEPS; s++) {
  theta += cfg.omega * dt;
  const pin0 = COROT ? seam(theta) : [0, seamY(theta), R_lay];
  ctx.frictionDtheta = cfg.omega * subDt;
  for (let sub = 0; sub < SUBSTEPS; sub++) {
    chain.step(subDt, pin0, FEED, { preStep: frictionStep, contactProject, bendIters: 3, maxStep: 8 });
  }
  // 給線: take-up 長 R_lay·Δθ が segRest 貯まるごとに 1 粒子 pay-out + bake（材料間隔 = segRest）
  payAccum += R_lay * Math.abs(cfg.omega * dt);
  while (payAccum >= segRest) { bakeAndEmit(); payAccum -= segRest; }

  if (s % 200 === 0) {
    const a = activeStats(), li = leadInLen(), bk = bakedStats();
    console.log(
      `# ${String(s).padStart(4)} | ${(Math.abs(theta) / (2 * Math.PI)).toFixed(1).padStart(4)} | ` +
      `${String(bk.n).padStart(6)} | ${bk.rMean.toFixed(1).padStart(10)} | ` +
      `${(bk.yMax - bk.yMin).toFixed(0).padStart(8)} | ${li.len.toFixed(0).padStart(5)} | ${li.rmax.toFixed(0).padStart(4)} | ` +
      `${a.minRpt.toFixed(1).padStart(6)} | ${a.kinkDeg.toFixed(1).padStart(7)} | ${a.stretch.toFixed(2)}`,
    );
  }
}

const a = activeStats(), li = leadInLen(), bk = bakedStats();
console.log(`#`);
console.log(`# === 結果 ===`);
console.log(`# baked: ${bk.n}点  rMean=${bk.rMean.toFixed(1)}mm (drum=${R_lay})  r範囲[${bk.rMin.toFixed(1)},${bk.rMax.toFixed(1)}]  y範囲[${bk.yMin.toFixed(1)},${bk.yMax.toFixed(1)}]`);
console.log(`# baked 不良: r逸脱=${bk.nBadR}  フランジ場外=${bk.nBadY}`);
console.log(`# lead弧=${li.len.toFixed(0)}mm  rmax=${li.rmax.toFixed(0)}mm  active minRpt=${a.minRpt.toFixed(1)}  カクつき=${a.kinkDeg.toFixed(1)}°  伸び=${a.stretch.toFixed(2)}×`);
const ang = bakedAngleStats();
const bakedTurns = bk.n * segLen / circ;
const expectTravel = bakedTurns * 360;     // 健全なら焼込周数 × 360° ぶん巻き付く
console.log(`# baked 角度走行=${ang.travelDeg.toFixed(0)}°  span=${ang.spanDeg.toFixed(0)}°  (期待≈${expectTravel.toFixed(0)}° = ${bakedTurns.toFixed(1)}周)`);
const gate1 = bk.n > 50 && li.rmax < 260;                             // 消費進行 + ループ無し（feed=r200）
const gate2 = bk.nBadR === 0 && bk.nBadY === 0;                       // 座り良好
const gate3 = a.minRpt >= minR_body - 0.5 && a.kinkDeg < 15 && a.stretch < 1.8;  // 健全
const gate4 = ang.travelDeg >= 0.5 * expectTravel && ang.travelDeg > 90;  // ヘリックスに巻き付く（子午線縮退でない）
console.log(`# ゲート1 消費&ループ無し: ${gate1 ? "PASS ✅" : "FAIL ❌"}  ゲート2 座り: ${gate2 ? "PASS ✅" : "FAIL ❌"}  ゲート3 健全: ${gate3 ? "PASS ✅" : "FAIL ❌"}  ゲート4 巻付角度: ${gate4 ? "PASS ✅" : "FAIL ❌"}`);
