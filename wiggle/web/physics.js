// 簡易素線モデル: 質点+軸方向ばねのみ（PBD = Verlet + 距離拘束）。
// 単位: mm / s。曲げ剛性ゼロ → ぐにゃぐにゃ。自己接触なし。
// 端点 2 つは外部からピンで与える。

export class StrandChain {
  constructor(N, restLength, opts = {}) {
    this.N = N;
    this.restLength = restLength;
    // optional: セグメントごとの rest 長（length N-1）。null なら scalar restLength を使う。
    // capstan 巻取で「巻き済み線は laid 長で凍結、供給は自由スパンだけ伸ばす」用。
    this.restLengths = opts.restLengths ?? null;
    this.pos = new Float32Array(3 * N);
    this.prev = new Float32Array(3 * N);
    this.gravity = opts.gravity ?? [0, -9810, 0];   // mm/s²
    this.damping = opts.damping ?? 0.985;
    this.iters = opts.iters ?? 8;
    // パチもん EI: 中間粒子を両隣の中点へ向けて Laplacian 引き寄せ。
    // 0 = 効果なし（純鎖）、1 = 即直線化（数値発散）。実用 0〜0.3。
    this.bendK = opts.bendK ?? 0;
  }

  applyBending(k, fixed) {
    for (let i = 1; i < this.N - 1; i++) {
      if (fixed && fixed[i]) continue;
      const a = 3 * (i - 1), b = 3 * i, c = 3 * (i + 1);
      const mx = 0.5 * (this.pos[a]     + this.pos[c]);
      const my = 0.5 * (this.pos[a + 1] + this.pos[c + 1]);
      const mz = 0.5 * (this.pos[a + 2] + this.pos[c + 2]);
      this.pos[b]     += k * (mx - this.pos[b]);
      this.pos[b + 1] += k * (my - this.pos[b + 1]);
      this.pos[b + 2] += k * (mz - this.pos[b + 2]);
    }
  }

  initLine(start, end) {
    for (let i = 0; i < this.N; i++) {
      const f = i / (this.N - 1);
      const x = start[0] + (end[0] - start[0]) * f;
      const y = start[1] + (end[1] - start[1]) * f;
      const z = start[2] + (end[2] - start[2]) * f;
      const k = 3 * i;
      this.pos[k] = x; this.pos[k + 1] = y; this.pos[k + 2] = z;
      this.prev[k] = x; this.prev[k + 1] = y; this.prev[k + 2] = z;
    }
  }

  setEndpoints(pinStart, pinEnd) {
    this.pos[0] = pinStart[0]; this.pos[1] = pinStart[1]; this.pos[2] = pinStart[2];
    this.prev[0] = pinStart[0]; this.prev[1] = pinStart[1]; this.prev[2] = pinStart[2];
    const L = 3 * (this.N - 1);
    this.pos[L] = pinEnd[0]; this.pos[L + 1] = pinEnd[1]; this.pos[L + 2] = pinEnd[2];
    this.prev[L] = pinEnd[0]; this.prev[L + 1] = pinEnd[1]; this.prev[L + 2] = pinEnd[2];
  }

  // pinStart / pinEnd: 3-vec or null（null = 自由端＝線ハネ事故モード）
  // opts.pinStart1: 3-vec or null（粒子 1 も pin → 起点接線が決まる）
  //   pinStart1 は pinStart が non-null の時だけ有効
  // opts.midPins: [{index, pos}] or null（中間粒子を追加 pin、ガイドアイ等）
  // opts.contactProject: (pos, N, fixed, prev) => void or null（各 PBD iter で接触処理）
  //   既存の距離拘束の後で呼ばれる。prev は摩擦（co-rotation drag）等で前位置が要る用
  // opts.bendIters: 曲げ拘束を距離拘束ループに入る前に何回まとめてかけるか
  //   既定 = this.iters。Strang split: bend-dist 交互適用は拡散方程式の
  //   CFL 超過と等価で発散するため、bend をブロックで先に終わらせる
  // opts.maxStep: 1 step あたりの最大変位 [mm]。既定 = max(restLength*3, 4)
  //   速度クランプ。前 step で blow up した粒子が次 step でさらに加速するのを抑制
  step(dt, pinStart, pinEnd, opts = {}) {
    const pinStart1 = opts.pinStart1 ?? null;
    const midPins = opts.midPins ?? null;
    const contactProject = opts.contactProject ?? null;
    const N = this.N;
    const gxdt2 = this.gravity[0] * dt * dt;
    const gydt2 = this.gravity[1] * dt * dt;
    const gzdt2 = this.gravity[2] * dt * dt;
    const d = this.damping;

    // どの粒子が固定か（pin がある所だけ true）
    const fixed = new Uint8Array(N);
    if (pinStart) fixed[0] = 1;
    if (pinStart && pinStart1) fixed[1] = 1;
    if (pinEnd) fixed[N - 1] = 1;
    if (midPins) {
      for (let m = 0; m < midPins.length; m++) {
        fixed[midPins[m].index] = 1;
      }
    }

    // Verlet 積分（固定粒子は skip）
    for (let i = 0; i < N; i++) {
      if (fixed[i]) continue;
      const k = 3 * i;
      const px = this.pos[k], py = this.pos[k + 1], pz = this.pos[k + 2];
      const vx = (px - this.prev[k]) * d;
      const vy = (py - this.prev[k + 1]) * d;
      const vz = (pz - this.prev[k + 2]) * d;
      this.prev[k] = px;
      this.prev[k + 1] = py;
      this.prev[k + 2] = pz;
      this.pos[k] = px + vx + gxdt2;
      this.pos[k + 1] = py + vy + gydt2;
      this.pos[k + 2] = pz + vz + gzdt2;
    }

    // pin 適用（pos と prev を強制一致 → 速度ゼロ）
    if (pinStart) {
      this.pos[0] = pinStart[0]; this.pos[1] = pinStart[1]; this.pos[2] = pinStart[2];
      this.prev[0] = pinStart[0]; this.prev[1] = pinStart[1]; this.prev[2] = pinStart[2];
    }
    if (pinStart && pinStart1) {
      this.pos[3] = pinStart1[0]; this.pos[4] = pinStart1[1]; this.pos[5] = pinStart1[2];
      this.prev[3] = pinStart1[0]; this.prev[4] = pinStart1[1]; this.prev[5] = pinStart1[2];
    }
    if (pinEnd) {
      const L = 3 * (N - 1);
      this.pos[L] = pinEnd[0]; this.pos[L + 1] = pinEnd[1]; this.pos[L + 2] = pinEnd[2];
      this.prev[L] = pinEnd[0]; this.prev[L + 1] = pinEnd[1]; this.prev[L + 2] = pinEnd[2];
    }
    if (midPins) {
      for (let m = 0; m < midPins.length; m++) {
        const mp = midPins[m];
        const k = 3 * mp.index;
        this.pos[k] = mp.pos[0]; this.pos[k + 1] = mp.pos[1]; this.pos[k + 2] = mp.pos[2];
        this.prev[k] = mp.pos[0]; this.prev[k + 1] = mp.pos[1]; this.prev[k + 2] = mp.pos[2];
      }
    }

    // 速度クランプ: 1 step あたり最大変位 maxStep を超える粒子は引き戻す。
    // 前回 step で発散しかけた粒子（pos - prev が暴れている）の暴走連鎖を抑制。
    const maxStep = opts.maxStep ?? Math.max(this.restLength * 3, 4);
    const maxStep2 = maxStep * maxStep;
    for (let i = 0; i < N; i++) {
      if (fixed[i]) continue;
      const k = 3 * i;
      const dxv = this.pos[k]     - this.prev[k];
      const dyv = this.pos[k + 1] - this.prev[k + 1];
      const dzv = this.pos[k + 2] - this.prev[k + 2];
      const dv2 = dxv * dxv + dyv * dyv + dzv * dzv;
      if (dv2 > maxStep2) {
        const sv = maxStep / Math.sqrt(dv2);
        this.pos[k]     = this.prev[k]     + dxv * sv;
        this.pos[k + 1] = this.prev[k + 1] + dyv * sv;
        this.pos[k + 2] = this.prev[k + 2] + dzv * sv;
      }
    }

    // preStep: 距離拘束ループに入る前に 1 回だけ呼ぶフック（摩擦用）。
    // 摩擦を毎反復の最後に当てると friction が「最後の一手」で再ストレッチし距離拘束が
    // 回復しきれない（capstan 巻取で電線が 2〜6 倍に伸びる主因）。摩擦は接線方向の
    // 一度の位置補正として先に当て、以降の距離+非貫入反復で締める＝距離が最後の一手。
    if (opts.preStep) opts.preStep(this.pos, N, fixed, this.prev);

    // パチもん EI（Strang split）: bend を距離拘束ループの外でブロック適用。
    // 旧: 各 PBD iter 内で bend → dist → bend → dist → ... と交互適用 → これは
    //     拡散方程式 ∂u/∂t = α∇²u の forward Euler を kα·6回 連打する非定常
    //     ガウスザイデルそのもので、CFL 超過時に高周波振動 → 発散する。
    // 新: bend を bendIters 回まとめてから dist+contact ループへ入る。各拘束は
    //     互いに干渉しない（bend は線形 Laplacian relaxation、dist は等長投影）。
    const kBend = this.bendK;
    if (kBend > 0) {
      const bendIters = opts.bendIters ?? this.iters;
      for (let bi = 0; bi < bendIters; bi++) this.applyBending(kBend, fixed);
    }

    // 距離拘束反復（Gauss-Seidel）+ 接触
    const rest = this.restLength;
    const restArr = this.restLengths;
    for (let it = 0; it < this.iters; it++) {
      for (let i = 0; i < N - 1; i++) {
        const a = 3 * i, b = 3 * (i + 1);
        const dx = this.pos[b] - this.pos[a];
        const dy = this.pos[b + 1] - this.pos[a + 1];
        const dz = this.pos[b + 2] - this.pos[a + 2];
        const len = Math.hypot(dx, dy, dz);
        if (len < 1e-9) continue;
        const ri = restArr ? restArr[i] : rest;
        const diff = (len - ri) / len;
        const aFixed = fixed[i];
        const bFixed = fixed[i + 1];
        if (aFixed && bFixed) continue;
        if (aFixed) {
          this.pos[b] -= dx * diff;
          this.pos[b + 1] -= dy * diff;
          this.pos[b + 2] -= dz * diff;
        } else if (bFixed) {
          this.pos[a] += dx * diff;
          this.pos[a + 1] += dy * diff;
          this.pos[a + 2] += dz * diff;
        } else {
          const cx = dx * diff * 0.5;
          const cy = dy * diff * 0.5;
          const cz = dz * diff * 0.5;
          this.pos[a] += cx;
          this.pos[a + 1] += cy;
          this.pos[a + 2] += cz;
          this.pos[b] -= cx;
          this.pos[b + 1] -= cy;
          this.pos[b + 2] -= cz;
        }
      }
      // 接触投影（cylinder / flange / self-contact 等）— ユーザ提供
      if (contactProject) contactProject(this.pos, N, fixed, this.prev);
    }
  }

  toPoints() {
    const out = new Array(this.N);
    for (let i = 0; i < this.N; i++) {
      const k = 3 * i;
      out[i] = { x: this.pos[k], y: this.pos[k + 1], z: this.pos[k + 2] };
    }
    return out;
  }

  totalArcLength() {
    let acc = 0;
    for (let i = 0; i < this.N - 1; i++) {
      const a = 3 * i, b = 3 * (i + 1);
      acc += Math.hypot(
        this.pos[b] - this.pos[a],
        this.pos[b + 1] - this.pos[a + 1],
        this.pos[b + 2] - this.pos[a + 2],
      );
    }
    return acc;
  }
}
