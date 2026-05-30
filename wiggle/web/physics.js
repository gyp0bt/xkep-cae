// 簡易素線モデル: 質点+軸方向ばねのみ（PBD = Verlet + 距離拘束）。
// 単位: mm / s。曲げ剛性ゼロ → ぐにゃぐにゃ。自己接触なし。
// 端点 2 つは外部からピンで与える。

export class StrandChain {
  constructor(N, restLength, opts = {}) {
    this.N = N;
    this.restLength = restLength;
    this.pos = new Float32Array(3 * N);
    this.prev = new Float32Array(3 * N);
    this.gravity = opts.gravity ?? [0, -9810, 0];   // mm/s²
    this.damping = opts.damping ?? 0.985;
    this.iters = opts.iters ?? 8;
    // パチもん EI: 中間粒子を両隣の中点へ向けて Laplacian 引き寄せ。
    // 0 = 効果なし（純鎖）、1 = 即直線化（数値発散）。実用 0〜0.3。
    this.bendK = opts.bendK ?? 0;
  }

  applyBending(k) {
    for (let i = 1; i < this.N - 1; i++) {
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

  step(dt, pinStart, pinEnd) {
    const N = this.N;
    const gxdt2 = this.gravity[0] * dt * dt;
    const gydt2 = this.gravity[1] * dt * dt;
    const gzdt2 = this.gravity[2] * dt * dt;
    const d = this.damping;

    // Verlet 積分（端点を除く）
    for (let i = 1; i < N - 1; i++) {
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

    this.setEndpoints(pinStart, pinEnd);

    // 距離拘束反復（Gauss-Seidel）+ パチもん bend（クランプなし、弾けとんでも OK）
    const rest = this.restLength;
    const kBend = this.bendK;
    for (let it = 0; it < this.iters; it++) {
      if (kBend > 0) this.applyBending(kBend);
      for (let i = 0; i < N - 1; i++) {
        const a = 3 * i, b = 3 * (i + 1);
        const dx = this.pos[b] - this.pos[a];
        const dy = this.pos[b + 1] - this.pos[a + 1];
        const dz = this.pos[b + 2] - this.pos[a + 2];
        const len = Math.hypot(dx, dy, dz);
        if (len < 1e-9) continue;
        const diff = (len - rest) / len;
        const aFixed = (i === 0);
        const bFixed = (i === N - 2);
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
