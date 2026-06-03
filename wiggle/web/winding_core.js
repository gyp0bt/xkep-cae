// wiggle — winding studio の純物理コア（THREE 非依存）。
// winding.js（描画 + GUI）と headless テスト（winding_core.test.mjs）が
// 同一の真実を import する。接触数式を /tmp にコピペして乖離させない。
//
// 内容:
//   defaultCfg()        — 寸法既定
//   estimateCapacity(c) — 巻線容量（全層埋まる弧長）の解析推定
//   wrappedArcLength()  — ボビン胴体に巻き付いている弧長 [mm]
//   bobbinContact()     — ボビン接触 + Coulomb 摩擦 + flange + 自己接触 + eye + guide
//   supplyStep()        — demand-driven 供給（起動ランプつき）
//   traverseStep()      — traverse jig 駆動
//
// 座標系: +y 上、ボビン軸 = +y、bobbin 中心 = 原点。単位 mm / rad / s。

export function defaultCfg() {
  return {
    BODY_R: 40,      // mm — 胴体半径
    BODY_H: 140,     // mm — 胴体長
    FLANGE_R: 78,    // mm — フランジ外径
    FLANGE_H: 8,     // mm — フランジ厚
    WIND_R_IN: 41,   // mm — 巻線内径
    WIND_R_OUT: 73,  // mm — 巻線最外径
    WIND_H: 120,     // mm — 巻線可用軸長
    wire_r: 1.8,     // mm — 素線半径
    omega: 1.5,      // rad/s — ボビン自転
    L_capacity: 0,   // mm — estimateCapacity 後に設定
  };
}

export function estimateCapacity(c) {
  const pitch = (2 * c.wire_r) * 1.05;
  const step = (2 * c.wire_r) * 0.96;
  const tpl = Math.max(2, Math.floor((c.WIND_H - pitch) / pitch));
  let acc = 0;
  let layer = 0;
  while (true) {
    const r = c.WIND_R_IN + (layer + 0.5) * step;
    if (r > c.WIND_R_OUT) break;
    acc += tpl * 2 * Math.PI * r;
    layer += 1;
    if (layer > 200) break;
  }
  return acc;
}

// ボビン胴体に「巻き付いている」弧長 [mm]。接触帯（軸距離 r ≤ 胴体面 + 3·wire_r、
// かつ flange 内）に両端が入っている seg の長さ合計。demand-driven 供給の巻取量測定用。
export function wrappedArcLength(pos, N, cfg) {
  const bandR = cfg.BODY_R + cfg.wire_r + cfg.wire_r * 3;
  const innerY = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
  let w = 0;
  for (let i = 0; i < N - 1; i++) {
    const ka = 3 * i, kb = 3 * (i + 1);
    const ra = Math.hypot(pos[ka], pos[ka + 2]);
    const rb = Math.hypot(pos[kb], pos[kb + 2]);
    if (ra <= bandR && Math.abs(pos[ka + 1]) < innerY &&
        rb <= bandR && Math.abs(pos[kb + 1]) < innerY) {
      w += Math.hypot(pos[kb] - pos[ka], pos[kb + 1] - pos[ka + 1], pos[kb + 2] - pos[ka + 2]);
    }
  }
  return w;
}

// PBD iter ごとに呼ばれる接触投影。距離拘束の後に走り衝突を解消する。
//   ctx = {
//     cfg, mu, seatGrip, frictionDtheta,   // ボビン胴体 + Coulomb 摩擦
//     eye:  {y, z, rInner} | null,         // traverse 通し穴
//     guide:{x, y, z, r, hx} | null,       // 入口ガイドローラ（軸 = x）
//   }
//
// (1) 胴体円柱 + **Coulomb 摩擦**:
//   法線投影で r=R+wire_r へ押し出す。その押し出し量 dN（貫入深さ）+ 接触帯 seating を
//   法線力の proxy とし、接線方向の表面追従ずれ slipArc を maxFric = μ·dN でクランプ。
//   → 滑り限界が生まれ「巻き付け角だけで張力が指数的に保持される」capstan 式
//     T_out = T_in·e^(μθ) が emergent に出る。深く座る内層ほど grip、外層は滑る。
// (2) フランジ面（円盤）: r < FLANGE_R+wire_r で |y| > flange 内面なら y を clip
// (3) 自己接触: 非隣接粒子間 距離 < 2·wire_r なら半分ずつ押し戻し（skipAdj=4）
// (4) traverse eye: 鎖が eye.y を貫通する seg を eye 穴半径内へ引き込む
// (5) guide roller: feed→eye の経路に置いた水平ローラ（軸 x）を貫通させない
export function bobbinContact(pos, N, fixed, prev, ctx) {
  const cfg = ctx.cfg;
  const wr = cfg.wire_r;
  const minR_body = cfg.BODY_R + wr;
  const minR2_body = minR_body * minR_body;
  const innerY = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2;
  const innerY_clip = innerY - wr;

  // (1) 胴体円柱 + Coulomb 摩擦
  const mu = ctx.mu ?? 0;
  const seatGrip = ctx.seatGrip ?? 0;
  const frictionDtheta = ctx.frictionDtheta ?? 0;
  const bandR = minR_body + wr * (ctx.seatBand ?? 4);   // seating を効かせる接触帯幅
  // 着地点（固定 pin0）直近の素線への摩擦適用。0=全粒子に適用（既定。adjacent を
  // 摩擦で握る方が安定）。>0 にすると pin 直近を除外できるが綱引きで逆に snap する。
  const fricSkipLo = ctx.fricSkipEnds ?? 0;
  for (let i = 0; i < N; i++) {
    if (fixed[i]) continue;
    const k = 3 * i;
    const y = pos[k + 1];
    if (Math.abs(y) >= innerY) continue;
    let x = pos[k], z = pos[k + 2];
    const r2 = x * x + z * z;
    // 法線投影 + 貫入量 dN（= 法線力の proxy）
    let dN = 0;
    if (r2 < minR2_body && r2 > 1e-12) {
      const r = Math.sqrt(r2);
      dN = minR_body - r;
      const scale = minR_body / r;
      x *= scale; z *= scale;
      pos[k] = x; pos[k + 2] = z;
    }
    // 接線摩擦（着地点直近 fricSkipLo 粒子は除外）。
    // ctx.skipFriction=true（距離ループ内の非貫入専用呼び出し）では摩擦を当てない。
    // 摩擦は preStep で substep に 1 回だけ当て inextensibility を守る（physics.js 参照）。
    if (!ctx.skipFriction && (mu > 0 || ctx.legacyStick > 0) && prev && i >= fricSkipLo) {
      const rr2 = x * x + z * z;
      if (rr2 > 1e-9) {
        const rr = Math.sqrt(rr2);
        if (ctx.legacyStick > 0) {
          // 旧 co-rotation（no-slip 食いつき）: **接触帯内**（rr < bandR）の粒子だけ
          // ボビン表面と一緒に回す。バンド外のフリースパンは掴まない（掴むと入線を
          // 引き回して暴れる）。回転 pin0 と同方向に敷き済み素線を運び綺麗なコイルを build。
          if (rr < bandR) {
            // 円周 no-slip: 表面の回転（frictionDtheta）へ stickK 分だけ追従
            const aCur = Math.atan2(z, x);
            const aPrev = Math.atan2(prev[k + 2], prev[k]);
            let dA = (aPrev + frictionDtheta) - aCur;
            while (dA > Math.PI) dA -= 2 * Math.PI;
            while (dA < -Math.PI) dA += 2 * Math.PI;
            const aNew = aCur + ctx.legacyStick * dA;
            pos[k] = rr * Math.cos(aNew);
            pos[k + 2] = rr * Math.sin(aNew);
            // 軸 no-slip: 掴んだ素線は前フレームの高さ y に貼り付く（stickK 分）。
            // 縦軸ボビンでは重力が軸方向に働くので、これが無いと敷いた巻きが
            // 底フランジへ滑落して溜まる。traverse が定める高さに各巻きが積まれる。
            pos[k + 1] = prev[k + 1] + (pos[k + 1] - prev[k + 1]) * (1 - ctx.legacyStick);
          }
        } else {
          // Coulomb 摩擦（滑り限界 μ·dN）。固定着地点モデル用（現在は未使用、比較保持）。
          if (rr < bandR) dN += (bandR - rr) * seatGrip;
          if (dN > 0) {
            const aCur = Math.atan2(z, x);
            const aPrev = Math.atan2(prev[k + 2], prev[k]);
            let dA = (aPrev + frictionDtheta) - aCur;
            while (dA > Math.PI) dA -= 2 * Math.PI;
            while (dA < -Math.PI) dA += 2 * Math.PI;
            const slipArc = rr * dA;
            const maxFric = mu * dN;
            const corrArc = slipArc > maxFric ? maxFric : (slipArc < -maxFric ? -maxFric : slipArc);
            const aNew = aCur + corrArc / rr;
            pos[k] = rr * Math.cos(aNew);
            pos[k + 2] = rr * Math.sin(aNew);
            const dy = pos[k + 1] - prev[k + 1];
            const maxAx = mu * dN;
            if (dy > maxAx) pos[k + 1] = prev[k + 1] + maxAx;
            else if (dy < -maxAx) pos[k + 1] = prev[k + 1] - maxAx;
          }
        }
      }
    }
  }

  // (2) フランジ面
  const minR_flange = cfg.FLANGE_R + wr;
  const minR2_flange = minR_flange * minR_flange;
  for (let i = 0; i < N; i++) {
    if (fixed[i]) continue;
    const k = 3 * i;
    const x = pos[k], y = pos[k + 1], z = pos[k + 2];
    // フランジは薄い円盤。面の近傍（|y| < innerY + wr）だけブロックする。
    // これを外すと、ボビン上方を昇るフリースパンが r < FLANGE_R に入った瞬間
    // |y| を innerY_clip へ瞬間移動させ大暴れする（起動バーストの真因）。
    if (Math.abs(y) >= innerY + wr) continue;
    const r2 = x * x + z * z;
    if (r2 >= minR2_flange) continue;
    if (y > innerY_clip) pos[k + 1] = innerY_clip;
    else if (y < -innerY_clip) pos[k + 1] = -innerY_clip;
  }

  // (3) 自己接触（素線-素線、skipAdj=4）
  const minD = 2 * wr;
  const minD2 = minD * minD;
  for (let i = 0; i < N; i++) {
    const ki = 3 * i;
    const xi = pos[ki], yi = pos[ki + 1], zi = pos[ki + 2];
    for (let j = i + 4; j < N; j++) {
      const kj = 3 * j;
      const dx = xi - pos[kj];
      const dy = yi - pos[kj + 1];
      const dz = zi - pos[kj + 2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 >= minD2 || d2 < 1e-12) continue;
      const d = Math.sqrt(d2);
      const corr = (minD - d) / d * 0.5;
      const cx = dx * corr, cy = dy * corr, cz = dz * corr;
      const fi = fixed[i], fj = fixed[j];
      if (!fi && !fj) {
        pos[ki] += cx; pos[ki + 1] += cy; pos[ki + 2] += cz;
        pos[kj] -= cx; pos[kj + 1] -= cy; pos[kj + 2] -= cz;
      } else if (!fi) {
        pos[ki] += 2 * cx; pos[ki + 1] += 2 * cy; pos[ki + 2] += 2 * cz;
      } else if (!fj) {
        pos[kj] -= 2 * cx; pos[kj + 1] -= 2 * cy; pos[kj + 2] -= 2 * cz;
      }
    }
  }

  // (6) 線分レベルのシリンダ接触 + 半径再クランプ（ctx.segmentContact で gate）。
  //   点接触（block 1）は粒子しか見ないので、両端が drum の外でも **線分が drum を貫く**
  //   トンネリングを素通りさせる（capstan probe で 20〜32 本/フレーム観測）。さらに
  //   自己接触（block 3）が外層粒子を drum 面の内側へ押し込み点貫通も起こす。両方を直す。
  //   既定 OFF。winding.js（laid-pin 駆動）は ctx.segmentContact を立てないので不変。
  if (ctx.segmentContact) {
    // (6a) drum 内へ押し込まれた点を半径再クランプ（自己接触の貫入を打ち消す。最終手番）
    for (let i = 0; i < N; i++) {
      if (fixed[i]) continue;
      const k = 3 * i;
      if (Math.abs(pos[k + 1]) >= innerY) continue;
      const r2 = pos[k] * pos[k] + pos[k + 2] * pos[k + 2];
      if (r2 < minR2_body && r2 > 1e-12) {
        const s = minR_body / Math.sqrt(r2);
        pos[k] *= s; pos[k + 2] *= s;
      }
    }
    // (6b) 両端は外でも線分が drum を貫くトンネリングを解消（xz 平面で最近接点を外へ）
    for (let i = 0; i < N - 1; i++) {
      const ka = 3 * i, kb = 3 * (i + 1);
      if (Math.abs(pos[ka + 1]) >= innerY || Math.abs(pos[kb + 1]) >= innerY) continue;
      const ax = pos[ka], az = pos[ka + 2], bx = pos[kb], bz = pos[kb + 2];
      const dx = bx - ax, dz = bz - az;
      const L2 = dx * dx + dz * dz;
      let t = L2 > 1e-12 ? -(ax * dx + az * dz) / L2 : 0;
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const cx = ax + t * dx, cz = az + t * dz;
      const d2 = cx * cx + cz * cz;
      if (d2 >= minR2_body || d2 < 1e-12) continue;
      const d = Math.sqrt(d2);
      const push = minR_body - d;
      const nx = cx / d, nz = cz / d;
      const wA = 1 - t, wB = t;
      const fa = fixed[i], fb = fixed[i + 1];
      const inv = 1 / (wA * wA + wB * wB);
      if (!fa && !fb) {
        pos[ka] += nx * push * wA * inv; pos[ka + 2] += nz * push * wA * inv;
        pos[kb] += nx * push * wB * inv; pos[kb + 2] += nz * push * wB * inv;
      } else if (!fa && wA > 0.05) {
        pos[ka] += nx * push / wA; pos[ka + 2] += nz * push / wA;
      } else if (!fb && wB > 0.05) {
        pos[kb] += nx * push / wB; pos[kb + 2] += nz * push / wB;
      }
    }
  }

  // (4) traverse eye 通し穴
  const eye = ctx.eye;
  if (eye) {
    const eyeY = eye.y, eyeX = 0, eyeZ = eye.z;
    const eyeR2 = eye.rInner * eye.rInner;
    for (let i = 0; i < N - 1; i++) {
      const ka = 3 * i, kb = 3 * (i + 1);
      const ya = pos[ka + 1], yb = pos[kb + 1];
      const sa = ya - eyeY, sb = yb - eyeY;
      if (sa * sb >= 0) continue;
      const denom_t = (sa - sb);
      if (Math.abs(denom_t) < 1e-9) continue;
      const t = sa / denom_t;
      const xi = pos[ka] + t * (pos[kb] - pos[ka]);
      const zi = pos[ka + 2] + t * (pos[kb + 2] - pos[ka + 2]);
      // フリースパン（入ってくる素線、r > FLANGE_R）だけ案内する。
      // 既に巻かれた素線（r < FLANGE_R）を掴むと穴へ瞬間移動して暴れる。
      if (Math.hypot(xi, zi) < cfg.FLANGE_R) continue;
      const dx = xi - eyeX;
      const dz = zi - eyeZ;
      const d2 = dx * dx + dz * dz;
      if (d2 <= eyeR2 || d2 < 1e-12) continue;
      const d = Math.sqrt(d2);
      const push = (d - eye.rInner) / d;
      const deltaX = -dx * push;
      const deltaZ = -dz * push;
      const wA = 1 - t, wB = t;
      const fi = fixed[i], fj = fixed[i + 1];
      if (!fi && !fj) {
        const inv = 1 / (wA * wA + wB * wB);
        pos[ka] += wA * deltaX * inv;
        pos[ka + 2] += wA * deltaZ * inv;
        pos[kb] += wB * deltaX * inv;
        pos[kb + 2] += wB * deltaZ * inv;
      } else if (!fi) {
        pos[ka] += deltaX / wA;
        pos[ka + 2] += deltaZ / wA;
      } else if (!fj) {
        pos[kb] += deltaX / wB;
        pos[kb + 2] += deltaZ / wB;
      }
    }
  }

  // (5) 入口ガイドローラ（軸 = x）: feed→eye 経路で素線が乗り越える水平プーリー。
  //   軸方向 |x−gx| < hx の範囲で yz 平面の円柱を貫通させない（外側へ押し出す）。
  const guide = ctx.guide;
  if (guide) {
    const minRg = guide.r + wr;
    const minRg2 = minRg * minRg;
    for (let i = 0; i < N; i++) {
      if (fixed[i]) continue;
      const k = 3 * i;
      if (Math.abs(pos[k] - guide.x) > guide.hx) continue;
      const dy = pos[k + 1] - guide.y;
      const dz = pos[k + 2] - guide.z;
      const d2 = dy * dy + dz * dz;
      if (d2 < minRg2 && d2 > 1e-12) {
        const s = minRg / Math.sqrt(d2);
        pos[k + 1] = guide.y + dy * s;
        pos[k + 2] = guide.z + dz * s;
      }
    }
  }
}

// demand-driven 供給: ボビンに巻き付いた弧長の増分だけ rest を成長させる。
//   供給速度を独立に与えない（巻取速度 = 供給速度）。
//   起動バースト対策:
//     ① 上限 maxFeed = |ω|·R·dt（表面速度ちょうど。旧 1.5× を撤廃）
//     ② 最初の RAMP_TURNS 巻きは smoothstep ランプで絞り、初期鎖がボビンへ
//        崩れ落ちる瞬間の供給バーストを断つ
//     ③ wrappedPrev は常に消費し幾何スパイクの backlog を残さない
//   state: { restGrow, L_supplied, wrappedPrev, turns, initChord } を mutate。
//   inc（今フレーム供給長）を返す。
export function supplyStep(state, wrapped, dt, cfg, opts = {}) {
  const R_pin = cfg.BODY_R + cfg.wire_r;
  const RAMP_TURNS = opts.rampTurns ?? 1.0;
  const tn = Math.min(1, Math.abs(state.turns) / RAMP_TURNS);
  const ramp = tn * tn * (3 - 2 * tn);   // smoothstep
  let inc = wrapped - state.wrappedPrev;
  state.wrappedPrev = wrapped;
  if (inc < 0) inc = 0;
  const maxFeed = Math.abs(cfg.omega) * R_pin * dt;
  if (inc > maxFeed) inc = maxFeed;
  // feedScale < 1: 巻取量より僅かに少なく供給 → 鎖が taut になり胴体に張り付く
  // （負帰還で安定。tension-servo の正帰還発散とは逆向き）。
  inc *= ramp * (opts.feedScale ?? 1);
  state.restGrow += inc;
  state.L_supplied = state.initChord + state.restGrow;
  return inc;
}

// traverse jig 駆動: 1 巻きあたり素線 1 本径だけ軸方向へ進み、flange 内面で折り返す。
//   state: { traverse_y, traverse_dir } を mutate。
export function traverseStep(state, dt, cfg) {
  const v_traverse = (Math.abs(cfg.omega) * 2 * cfg.wire_r * 1.05) / (2 * Math.PI);
  state.traverse_y += v_traverse * state.traverse_dir * dt;
  const inner_y = (cfg.BODY_H - 2 * cfg.FLANGE_H) / 2 - cfg.wire_r;
  if (state.traverse_y >= inner_y) { state.traverse_y = inner_y; state.traverse_dir = -1; }
  if (state.traverse_y <= -inner_y) { state.traverse_y = -inner_y; state.traverse_dir = +1; }
}
