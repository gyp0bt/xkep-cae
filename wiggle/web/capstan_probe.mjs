// 実験A v2: emergent capstan winding のヘッドレス証明（重力ゼロ・**有限長の伸びない電線**）。
//
// v1 の教訓: 「rest を伸ばして給線」は非物理（セグメント=材料点に材料を注入＝伸び）。
//   巻き済み線に slack が注入され、みみず（カクつき 45°）と drum 貫通（トンネル線分）を生んだ。
// v2: 有限長の伸びない電線をゆるい pre-coil で置き、回る継ぎ目（結び目）が巻き締める。
//   rest 成長ゼロ → slack 注入ゼロ。per-segment rest を固定値で持ち inextensible を厳守。
//
// 駆動:
//   - particle 0   = ドラム固定の継ぎ目。ω で回るだけ（高さ y0 固定）。take-up の prime mover。
//   - particle N-1 = feed 点（世界固定）。
//   - 重力 = 0。摩擦 = co-rotation（legacyStick）。SEGFIX で線分接触+再クランプ。
//
// env: BENDK（パチもん EI）, SEGFIX（線分接触 1/0）, NTURNS（pre-coil 周数）
// 実行: node wiggle/web/capstan_probe.mjs 2>&1 | tee /tmp/capstan-$(date +%s).log

import { StrandChain } from "./physics.js";
import {
  defaultCfg, estimateCapacity, wrappedArcLength, bobbinContact,
} from "./winding_core.js";

const cfg = defaultCfg();
cfg.L_capacity = estimateCapacity(cfg);

const R_lay = cfg.BODY_R + cfg.wire_r;            // 41.8 mm
const circumference = 2 * Math.PI * R_lay;        // ≈ 262.6 mm / 周
const innerY = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
const bandR_wrap = cfg.BODY_R + cfg.wire_r + cfg.wire_r * 3;   // 47.2
const minR_body = cfg.BODY_R + cfg.wire_r;        // 41.8（これ未満 = 貫通）
const y0 = 0;
const FEED = [0, 0, 200];

const BENDK = parseFloat(process.env.BENDK ?? "0.04");
const SEGFIX = (process.env.SEGFIX ?? "1") === "1";
const NTURNS = parseFloat(process.env.NTURNS ?? "4");
const TRAVERSE = (process.env.TRAVERSE ?? "1") === "1";   // 軸送り（helix）

// ---- 有限電線: 標準距離 + NTURNS 周ぶんの pre-coil（helix で軸方向に広げる）----
const pitch = 2 * cfg.wire_r * 1.05;              // 隣接巻きの軸ピッチ ≈ 3.78mm
const TRAV_LO = -(innerY - cfg.wire_r - 2);       // 下フランジ際
const TRAV_HI = (innerY - cfg.wire_r - 2);        // 上フランジ際
const TRAV_SPAN = TRAV_HI - TRAV_LO;
const R_loose = R_lay + parseFloat(process.env.RLOOSE ?? "7");   // ゆるく置く初期半径
// seam 高さ: レベルワインド三角波（両フランジで折り返す）。旧単調ランプは場外 walk-off を起こした。
function seamY(theta) {
  if (!TRAVERSE) return y0;
  const dist = (Math.abs(theta) / (2 * Math.PI)) * pitch;
  const period = 2 * TRAV_SPAN;
  let m = dist % period;
  if (m < 0) m += period;
  return m <= TRAV_SPAN ? TRAV_LO + m : TRAV_LO + (period - m);
}
// SEAMORBIT=1: 旧モデル（継ぎ目が θ で公転＝結んだ端が回る）。有限線を周回ドラッグ＝余長が
//   消費されずループに化ける。SEAMORBIT=0: 接線点を空間固定（実機の lay point）。ドラムは
//   その下で回り、摩擦 take-up が feed から新線を引き込む＝余長が消費されて解消する。
const SEAMORBIT = (process.env.SEAMORBIT ?? "1") === "1";
const LAY_ANGLE = 0;   // 固定 lay 角（feed が居る +z 側に正対）
function seam(theta) {
  const a = SEAMORBIT ? theta : LAY_ANGLE;
  return [R_lay * Math.sin(a), seamY(theta), R_lay * Math.cos(a)];
}

const standoff = 180;                             // テール弧長（feed までの遊び）
const L_coil = NTURNS * 2 * Math.PI * R_loose;
const L_wire = L_coil + standoff;
const segLen = 3.0;
const N = Math.max(120, Math.round(L_wire / segLen) + 1);
const segRest = L_wire / (N - 1);

// pre-coil ポリライン（弧長等分割）: helix（軸方向に pitch で広がる）→ feed への直線テール
const coilEndPhi = L_coil / R_loose;
const coilEndY = TRAVERSE ? TRAV_LO + (coilEndPhi / (2 * Math.PI)) * pitch : y0;
const cEnd = [R_loose * Math.sin(coilEndPhi), coilEndY, R_loose * Math.cos(coilEndPhi)];
function initPoint(s) {                            // s = 弧長 [0, L_wire]
  if (s <= L_coil) {
    const phi = s / R_loose;
    const y = TRAVERSE ? TRAV_LO + (phi / (2 * Math.PI)) * pitch : y0;
    return [R_loose * Math.sin(phi), y, R_loose * Math.cos(phi)];
  }
  const f = (s - L_coil) / standoff;
  return [
    cEnd[0] + (FEED[0] - cEnd[0]) * f,
    cEnd[1] + (FEED[1] - cEnd[1]) * f,
    cEnd[2] + (FEED[2] - cEnd[2]) * f,
  ];
}

const ITERS = parseInt(process.env.ITERS ?? "12", 10);
const GRAV = parseFloat(process.env.GRAV ?? "0");        // 1 = 9810 mm/s²
// GRAVDIR: "y"=軸方向（鉛直軸・滑落する） "z"=軸垂直（水平軸・ドラムへ押し付け）
const GRAVDIR = process.env.GRAVDIR ?? "z";
const gravVec = GRAVDIR === "y" ? [0, -9810 * GRAV, 0] : [0, 0, -9810 * GRAV];
const chain = new StrandChain(N, segRest, {
  gravity: gravVec,
  damping: 0.99,
  iters: ITERS,
  restLengths: new Float64Array(N - 1).fill(segRest),   // per-segment 固定 rest（伸びない）
});
chain.bendK = BENDK;
for (let i = 0; i < N; i++) {
  const p = initPoint((i / (N - 1)) * L_wire);
  const k = 3 * i;
  chain.pos[k] = p[0]; chain.pos[k + 1] = p[1]; chain.pos[k + 2] = p[2];
  chain.prev[k] = p[0]; chain.prev[k + 1] = p[1]; chain.prev[k + 2] = p[2];
}

// 摩擦モデル: STICK>0 なら旧 co-rotation（ハード再配置）、それ以外は本物の Coulomb
// （滑り補正を μ·dN でクランプ → 距離拘束を壊さない＝伸びない）。
const STICK = parseFloat(process.env.STICK ?? "0");
const MU = parseFloat(process.env.MU ?? "0.8");
const SEATGRIP = parseFloat(process.env.SEATGRIP ?? "0.5");
const ctx = {
  cfg,
  legacyStick: STICK,
  mu: MU,
  seatGrip: SEATGRIP,
  frictionDtheta: 0,
  segmentContact: SEGFIX,
  eye: null,
  guide: null,
};
// バックテンション（供給リールの制動）: リードイン（接線点→feed）を弦に向け直線化。
// 旧 BC = feed 固定ピン + 有限長 → 巻き締めの余長が行き場を失い、回る巻き済み端に
// 引きずられて振り回される（先端 r≈433mm）。実機は供給側 T_in で常時張る → 余長が吸われ直線。
const bandR_wrap2 = bandR_wrap * bandR_wrap;
const BT = parseFloat(process.env.BT ?? "0.2");        // バックテンション利得（PBD iter あたり）
function applyBackTension(pos, n, fixed, gain) {
  if (gain <= 0) return;
  let B = -1;                                          // feed 側から内側へ走査 → 最初の巻付帯粒子＝接線点
  for (let i = n - 2; i >= 0; i--) {
    const k = 3 * i;
    if (pos[k] * pos[k] + pos[k + 2] * pos[k + 2] <= bandR_wrap2 && Math.abs(pos[k + 1]) < innerY) { B = i; break; }
  }
  if (B < 0) B = 0;
  const bk = 3 * B, fk = 3 * (n - 1);
  const Bx = pos[bk], By = pos[bk + 1], Bz = pos[bk + 2];
  const dx = pos[fk] - Bx, dy = pos[fk + 1] - By, dz = pos[fk + 2] - Bz;
  const L2 = dx * dx + dy * dy + dz * dz;
  if (L2 < 1e-9) return;
  for (let i = B + 1; i < n - 1; i++) {                // 自由スパンを弦 BF へ引き寄せ（張力直線化）
    if (fixed[i]) continue;
    const k = 3 * i;
    const t = Math.max(0, Math.min(1, ((pos[k] - Bx) * dx + (pos[k + 1] - By) * dy + (pos[k + 2] - Bz) * dz) / L2));
    pos[k]     += (Bx + dx * t - pos[k])     * gain;
    pos[k + 1] += (By + dy * t - pos[k + 1]) * gain;
    pos[k + 2] += (Bz + dz * t - pos[k + 2]) * gain;
  }
}
// 摩擦は preStep で substep に 1 回だけ、距離ループ内は非貫入＋バックテンション（inextensibility 厳守）
const frictionStep = (pos, n, fixed, prev) => { ctx.skipFriction = false; bobbinContact(pos, n, fixed, prev, ctx); };
const contactProject = (pos, n, fixed, prev) => {
  ctx.skipFriction = true; bobbinContact(pos, n, fixed, prev, ctx);
  applyBackTension(pos, n, fixed, BT);
};

const dt = 1 / 60;
const SUBSTEPS = 4;
const subDt = dt / SUBSTEPS;
const STEPS = parseInt(process.env.STEPS ?? "1500", 10);

function wrapStats() {
  let n = 0, sum = 0, yMin = Infinity, yMax = -Infinity;
  for (let i = 0; i < N; i++) {
    const k = 3 * i;
    const r = Math.hypot(chain.pos[k], chain.pos[k + 2]);
    if (r <= bandR_wrap && Math.abs(chain.pos[k + 1]) < innerY) {
      n++; sum += r;
      const y = chain.pos[k + 1];
      if (y < yMin) yMin = y; if (y > yMax) yMax = y;
    }
  }
  return { n, mean: n ? sum / n : 0, yMin: n ? yMin : 0, yMax: n ? yMax : 0 };
}
function segAxisDist2D(ax, az, bx, bz) {
  const dx = bx - ax, dz = bz - az;
  const L2 = dx * dx + dz * dz;
  let t = L2 > 1e-12 ? -(ax * dx + az * dz) / L2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(ax + t * dx, az + t * dz);
}
function penetrationStats() {
  let minRpt = Infinity, nTunnel = 0;
  for (let i = 0; i < N; i++) {
    const k = 3 * i;
    if (Math.abs(chain.pos[k + 1]) >= innerY) continue;
    const r = Math.hypot(chain.pos[k], chain.pos[k + 2]);
    if (r < minRpt) minRpt = r;
  }
  for (let i = 0; i < N - 1; i++) {
    const ka = 3 * i, kb = 3 * (i + 1);
    if (Math.abs(chain.pos[ka + 1]) >= innerY || Math.abs(chain.pos[kb + 1]) >= innerY) continue;
    const ra = Math.hypot(chain.pos[ka], chain.pos[ka + 2]);
    const rb = Math.hypot(chain.pos[kb], chain.pos[kb + 2]);
    if (ra >= minR_body && rb >= minR_body) {
      if (segAxisDist2D(chain.pos[ka], chain.pos[ka + 2], chain.pos[kb], chain.pos[kb + 2]) < minR_body - 0.5) nTunnel++;
    }
  }
  let kinkSum = 0, kinkN = 0;
  for (let i = 1; i < N - 1; i++) {
    const a = 3 * (i - 1), b = 3 * i, c = 3 * (i + 1);
    const ux = chain.pos[b] - chain.pos[a], uy = chain.pos[b + 1] - chain.pos[a + 1], uz = chain.pos[b + 2] - chain.pos[a + 2];
    const vx = chain.pos[c] - chain.pos[b], vy = chain.pos[c + 1] - chain.pos[b + 1], vz = chain.pos[c + 2] - chain.pos[b + 2];
    const lu = Math.hypot(ux, uy, uz), lv = Math.hypot(vx, vy, vz);
    if (lu < 1e-9 || lv < 1e-9) continue;
    let cosA = (ux * vx + uy * vy + uz * vz) / (lu * lv);
    cosA = Math.max(-1, Math.min(1, cosA));
    kinkSum += Math.acos(cosA); kinkN++;
  }
  return { minRpt, nTunnel, kinkDeg: kinkN ? (kinkSum / kinkN) * 180 / Math.PI : 0 };
}

let theta = 0;
let turns = 0;
console.log(`# capstan_probe v2 (有限電線)  N=${N}  全長=${L_wire.toFixed(0)}mm  segLen≈${segRest.toFixed(1)}mm`);
console.log(`# 重力=0  bendK=${BENDK}  SEGFIX=${SEGFIX}  pre-coil=${NTURNS}周  traverse=${TRAVERSE}  ω=${cfg.omega}`);
console.log(`# 摩擦: ${STICK > 0 ? `legacyStick=${STICK}（旧ハード再配置）` : `Coulomb μ=${MU} seatGrip=${SEATGRIP}`}`);
console.log(`# step | 自転[周] | 巻付[周] | 帯内粒子 | meanR | minRpt | トンネル | カクつき[deg]`);

for (let s = 1; s <= STEPS; s++) {
  theta += cfg.omega * dt;
  turns = Math.abs(theta) / (2 * Math.PI);
  const pin0 = seam(theta);
  ctx.frictionDtheta = cfg.omega * subDt;
  for (let sub = 0; sub < SUBSTEPS; sub++) {
    chain.step(subDt, pin0, FEED, { preStep: frictionStep, contactProject, bendIters: 3, maxStep: 8 });
  }
  if (s % 100 === 0) {
    const wrapped = wrappedArcLength(chain.pos, N, cfg);
    const ws = wrapStats();
    const ps = penetrationStats();
    // 自由スパン（接線点→feed）の弧長 = 未消費の余長。消費されれば減るはず。
    let B = -1;
    for (let i = N - 2; i >= 0; i--) { const k = 3 * i; if (chain.pos[k] * chain.pos[k] + chain.pos[k + 2] * chain.pos[k + 2] <= bandR_wrap2 && Math.abs(chain.pos[k + 1]) < innerY) { B = i; break; } }
    let tailLen = 0; for (let i = Math.max(0, B); i < N - 1; i++) { const a = 3 * i, b = 3 * (i + 1); tailLen += Math.hypot(chain.pos[b] - chain.pos[a], chain.pos[b + 1] - chain.pos[a + 1], chain.pos[b + 2] - chain.pos[a + 2]); }
    let rmax = 0; for (let i = 0; i < N - 1; i++) { const k = 3 * i; const r = Math.hypot(chain.pos[k], chain.pos[k + 2]); if (r > rmax) rmax = r; }
    console.log(
      `# ${String(s).padStart(4)} | ${turns.toFixed(2).padStart(7)} | ` +
      `${(wrapped / circumference).toFixed(2).padStart(7)} | ${String(ws.n).padStart(7)} | ` +
      `${ws.mean.toFixed(1).padStart(5)} | ${ps.minRpt.toFixed(1).padStart(6)} | ` +
      `${String(ps.nTunnel).padStart(7)} | ${ps.kinkDeg.toFixed(1).padStart(6)} | ` +
      `tail弧 ${tailLen.toFixed(0).padStart(4)} | rmax ${rmax.toFixed(0).padStart(4)}`,
    );
  }
}

// 場外 walk-off ゲート: in-band フィルタの外で全粒子の y/r を見る（旧バグの再発防止）。
const flangeLimit = innerY;            // |y| がこれを超えた粒子 = フランジ場外
let nEscaped = 0, yAbsMax = 0, rMax = 0;
for (let i = 0; i < N - 1; i++) {       // feed 端(N-1)は世界固定なので除外
  const k = 3 * i;
  const ay = Math.abs(chain.pos[k + 1]);
  const r = Math.hypot(chain.pos[k], chain.pos[k + 2]);
  if (ay > yAbsMax) yAbsMax = ay;
  if (r > rMax) rMax = r;
  if (ay > flangeLimit) nEscaped++;
}

const wrappedFinal = wrappedArcLength(chain.pos, N, cfg);
const psF = penetrationStats();
const totalLen = chain.totalArcLength();
const stretch = totalLen / L_wire;
console.log(`#`);
console.log(`# === 結果  (bendK=${BENDK}  SEGFIX=${SEGFIX}  pre-coil=${NTURNS}周) ===`);
console.log(`# 電線全長: ${totalLen.toFixed(0)}mm / rest ${L_wire.toFixed(0)}mm = 伸び ${stretch.toFixed(2)}倍（1.0=inextensible 厳守）`);
const wsF = wrapStats();
console.log(`# 自転: ${turns.toFixed(2)} 周 / 巻付: ${(wrappedFinal / circumference).toFixed(2)} 周分`);
console.log(`# 帯内 y 範囲: ${wsF.yMin.toFixed(1)}〜${wsF.yMax.toFixed(1)}mm (下フランジ際=${TRAV_LO.toFixed(0)}) ＝ 軸方向の広がり ${(wsF.yMax - wsF.yMin).toFixed(1)}mm`);
console.log(`# 貫通: minRpt=${psF.minRpt.toFixed(1)}mm (drum=${minR_body}) / トンネル線分=${psF.nTunnel} / カクつき=${psF.kinkDeg.toFixed(1)}°`);
const penOK = psF.minRpt >= minR_body - 0.5 && psF.nTunnel === 0;
const smoothOK = psF.kinkDeg < 15;
const escapeOK = nEscaped === 0;
console.log(`# 場外: |y|max=${yAbsMax.toFixed(1)}mm (フランジ=${flangeLimit}) / rmax=${rMax.toFixed(1)}mm / 場外粒子=${nEscaped}`);
console.log(`# 貫通ゲート(minRpt≥${minR_body - 0.5}&トンネル0): ${penOK ? "PASS ✅" : "FAIL ❌"}  ` +
  `平滑ゲート(カクつき<15°): ${smoothOK ? "PASS ✅" : "FAIL ❌"}  ` +
  `場外ゲート(粒子0): ${escapeOK ? "PASS ✅" : "FAIL ❌"}`);
