// wiggle — winding studio headless 物理検証。
//   winding_core.js + physics.js を import して node で鎖を回し、winding.js と
//   同一の数式で「暴れない・巻き取れる・滑りが capstan 的」を数値判定する。
//   実行: node wiggle/web/winding_core.test.mjs
//
// 巻取りモデル: particle 0 = **世界固定の着地点**（traverse_y に追従、回らない）。
//   ボビンがその下で自転し、Coulomb 摩擦が敷き済み素線を引きずって巻き上げる
//   （= 実際の winding。回転 hard-pin だと摩擦の滑りと両立せず素線が裂ける）。
//
// 判定:
//   A. 安定性     — 終盤(t>34s) mean 粒子速度が表面速度 ω·R の 4 倍以内（暴走しない）
//   B. 起動鎮静   — 初期落下後(t∈[3,6]s) mean 粒子速度が ω·R の 3 倍以内
//                   （初期スラックの塊が持続的に暴れない。落下中の瞬間 max は対象外）
//   C. 巻取り     — wrapMax が一定以上（take-up 機能）
//   D. capstan    — 摩擦が巻取りを成立させる: μ>0 の wrapMax が μ=0 の 3 倍超。
//                   （「μ で単調増」は偽: 過大摩擦は素線を bunch させ wrap を減らす。
//                    成立する物理は「摩擦なし=巻けない / 摩擦あり=巻ける」の差。）

import { StrandChain } from "./physics.js";
import {
  defaultCfg, estimateCapacity, wrappedArcLength,
  bobbinContact, supplyStep, traverseStep,
} from "./winding_core.js";

const cfg = defaultCfg();
cfg.L_capacity = estimateCapacity(cfg);

const CHAIN_N = 200;
const FEED = [0, 250 - 18, 120];
const R_pin = cfg.BODY_R + cfg.wire_r;
const EYE_Z = cfg.FLANGE_R + 6;
const GUIDE = { x: 0, y: 150, z: 100, r: 10, hx: 20 };

const innerY0 = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
const Y0 = -(innerY0 - cfg.wire_r);   // 巻き始め着地点 = 下フランジ際
const pin0Init = [0, Y0, R_pin];       // 世界固定の着地点（z+ 側、eye 直下、底）
const initChord = Math.hypot(FEED[0] - pin0Init[0], FEED[1] - pin0Init[1], FEED[2] - pin0Init[2]);

function run(mu, seatGrip, T = 40) {
  cfg.omega = 1.5;
  const chain = new StrandChain(CHAIN_N, initChord / (CHAIN_N - 1), { damping: 0.985, iters: 8 });
  chain.bendK = 0.18;
  chain.initLine(pin0Init, FEED);

  const state = {
    restGrow: 0, L_supplied: initChord, wrappedPrev: 0,
    turns: 0, traverse_y: Y0, traverse_dir: +1, initChord,
  };

  const dt = 1 / 60;
  const steps = Math.round(T / dt);
  const SUBSTEPS = 4;

  const ctx = {
    cfg, mu, seatGrip, frictionDtheta: 0,
    eye: { y: 0, z: EYE_Z, rInner: 3.6 - 0.8 },
    guide: GUIDE,
  };
  const contact = (pos, N, fixed, prev) => bobbinContact(pos, N, fixed, prev, ctx);

  const prevSnapshot = new Float32Array(chain.pos.length);
  let meanSettle = 0, settleCount = 0;   // t∈[3,6]
  let meanLate = 0, lateCount = 0;        // t>T-6
  let wrapMax = 0, wrapFinal = 0;
  const innerY = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
  const bandR = R_pin + cfg.wire_r * 3;

  for (let s = 0; s < steps; s++) {
    const t = s * dt;
    state.turns = cfg.omega * t / (2 * Math.PI);
    // 着地点 = 世界固定（traverse_y に横送り追従、回転しない）
    const pin0World = [0, state.traverse_y, R_pin];
    chain.restLength = state.L_supplied / (CHAIN_N - 1);

    prevSnapshot.set(chain.pos);
    const subDt = dt / SUBSTEPS;
    ctx.frictionDtheta = cfg.omega * subDt;
    ctx.eye.y = state.traverse_y;
    for (let sub = 0; sub < SUBSTEPS; sub++) {
      chain.step(subDt, pin0World, FEED, { contactProject: contact, bendIters: 3, maxStep: 8 });
    }

    const wrapped = wrappedArcLength(chain.pos, CHAIN_N, cfg);
    supplyStep(state, wrapped, dt, cfg, { feedScale: 0.7 });
    traverseStep(state, dt, cfg);

    let sumSp = 0, cnt = 0;
    for (let i = 1; i < CHAIN_N - 1; i++) {
      const k = 3 * i;
      sumSp += Math.hypot(
        chain.pos[k] - prevSnapshot[k],
        chain.pos[k + 1] - prevSnapshot[k + 1],
        chain.pos[k + 2] - prevSnapshot[k + 2],
      ) / dt;
      cnt++;
    }
    const meanSp = sumSp / cnt;
    if (t >= 3 && t < 6) { meanSettle += meanSp; settleCount++; }
    if (t > T - 6) { meanLate += meanSp; lateCount++; }
    if (wrapped > wrapMax) wrapMax = wrapped;
    wrapFinal = wrapped;
  }

  // 終状態の半径分布: 巻線は胴体(r≈R_pin の band)に巻かれるべきで、フランジ面
  // (r>band, |y|≈フランジ際)に山積みするのは欠陥（縦軸重力スランプ）。両者を数える。
  let bodyN = 0, flangeN = 0, yMin = Infinity, yMax = -Infinity;
  const flangeR = cfg.FLANGE_R;
  for (let i = 0; i < CHAIN_N; i++) {
    const k = 3 * i;
    const r = Math.hypot(chain.pos[k], chain.pos[k + 2]);
    const y = chain.pos[k + 1];
    if (Math.abs(y) >= innerY + cfg.wire_r) continue;   // free span は除外
    if (r <= bandR) { bodyN++; if (y < yMin) yMin = y; if (y > yMax) yMax = y; }
    else if (r < flangeR && Math.abs(y) > innerY * 0.6) flangeN++;
  }

  return {
    surfSpeed: Math.abs(cfg.omega) * R_pin,
    meanSettle: meanSettle / Math.max(1, settleCount),
    meanLate: meanLate / Math.max(1, lateCount),
    wrapMax, wrapFinal,
    L_supplied: state.L_supplied,
    wrapPct: wrapFinal / cfg.L_capacity * 100,
    bodyN, flangeN, yMin: bodyN ? yMin : 0, yMax: bodyN ? yMax : 0,
  };
}

console.log("=== winding_core headless 検証（固定着地点モデル）===");
const surf = 1.5 * R_pin;
console.log(`表面速度 ω·R = ${surf.toFixed(0)} mm/s\n`);

console.log("μ    seat | settle late  wrapEnd wrapMax  L_sup | bodyN flangeN yMin yMax");
const rows = [];
for (const [mu, seat] of [[0, 1.0], [0.4, 1.0], [0.8, 1.0], [1.2, 1.0]]) {
  const r = run(mu, seat);
  rows.push([mu, r]);
  console.log(
    `${mu.toFixed(2)} ${seat.toFixed(2)} | ${r.meanSettle.toFixed(0).padStart(6)} ` +
    `${r.meanLate.toFixed(0).padStart(4)} ${r.wrapFinal.toFixed(0).padStart(8)} ` +
    `${r.wrapMax.toFixed(0).padStart(7)} ${r.L_supplied.toFixed(0).padStart(6)} | ` +
    `${String(r.bodyN).padStart(5)} ${String(r.flangeN).padStart(7)} ${r.yMin.toFixed(0).padStart(4)} ${r.yMax.toFixed(0).padStart(4)}`,
  );
}

const base = run(0.8, 1.0);
const A = base.meanLate < surf * 4;
const B = base.meanSettle < surf * 3;
const C = base.wrapMax > 200;
const wNoFric = rows.find(([m]) => m === 0)[1].wrapMax;
const D = base.wrapMax > wNoFric * 3;
// E. 半径分布: 巻線の大半が胴体(band)に巻かれ、フランジ面への山積みが少ない。
//    軸方向 Coulomb 摩擦が laid wrap を胴体に保持すれば bodyN ≫ flangeN。
const E = base.bodyN > base.flangeN * 2;

console.log("\n--- 判定 ---");
console.log(`A 安定性    late=${base.meanLate.toFixed(0)} < ${(surf * 4).toFixed(0)}     : ${A ? "PASS" : "FAIL"}`);
console.log(`B 起動鎮静  settle=${base.meanSettle.toFixed(0)} < ${(surf * 3).toFixed(0)} : ${B ? "PASS" : "FAIL"}`);
console.log(`C 巻取り    wrapMax=${base.wrapMax.toFixed(0)} > 200      : ${C ? "PASS" : "FAIL"}`);
console.log(`D capstan   wrapMax(μ.8)=${base.wrapMax.toFixed(0)} > 3×wrap(μ0)=${(wNoFric * 3).toFixed(0)} : ${D ? "PASS" : "FAIL"}`);
console.log(`E 半径分布  bodyN=${base.bodyN} > 2×flangeN=${base.flangeN * 2}（胴体巻>フランジ山積）: ${E ? "PASS" : "FAIL"}`);

const ok = A && B && C && D && E;
console.log(`\n総合: ${ok ? "✅ ALL PASS" : "❌ FAIL"}`);
process.exit(ok ? 0 : 1);
