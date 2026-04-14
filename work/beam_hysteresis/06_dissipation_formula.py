"""
散逸エネルギーの解析的近似式導出
==================================

物理モデル:
  撚線ケーブルの曲げヒステリシスループ面積 W_diss を
  撚線本数 n、ピッチ p、曲率 κ の関数として近似する。

理論的根拠:
  1. M-κ ループ幅 ≈ ΔEI × κ = (EI_max - EI_min) × κ
     - EI_min = n × I_wire（全滑り）
     - EI_max = n × I_wire + n_outer × A × R_helix²（全ロック、Steiner）
     - ΔEI = n_outer × A × R_helix²（Steiner 寄与のみ）

  2. 漸進的スリップの効果:
     - 低 κ: 全層弾性（散逸=0）
     - 高 κ: 全層スリップ（飽和）
     - 遷移域: S 字カーブ → べき乗 κ^α で近似（α ≈ 1.5-2.0）

  3. ピッチ依存性:
     - ヘリックス角 α_h = atan(2πR_helix/p)
     - 接触力 ∝ sin(α_h) × cos(α_h)（軸力の法線・接線分解）
     - 滑り量 ∝ sin(α_h) × κ × R_helix
     - 散逸 ∝ 接触力 × 滑り量 ∝ sin²(α_h) × cos(α_h)

  提案する解析形:
    W_diss(n, p, κ) = C × ΔEI(n) × κ^α × g(p)

  ここで:
    ΔEI(n) = n_outer × π × R² × (2R)² = 4π × n_outer × R⁴
    g(p)   = sin²(α_h) × cos(α_h)  （α_h = atan(2πR_helix/p)）
    C, α   = 回帰パラメータ

パラメータ掃引:
  n_strands ∈ {2, 3, 5, 7, 12, 19}
  pitch_length ∈ {30, 50, 75, 100, 150, 200, 300} mm
  bending_curvature ∈ {0.0005, 0.001, 0.002, 0.003, 0.005} 1/mm

[← README](../../README.md)
"""

from __future__ import annotations

import math
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from xkep_cae.numerical_tests.cable_dissipation import (
    CableDissipationConfig,
    CableDissipationProcess,
    _cable_geometry,
)


# ====================================================================
# 解析モデル関数
# ====================================================================


def delta_EI(n_strands: int, wire_radius: float = 0.5) -> float:
    """Steiner 寄与 ΔEI = EI_max - EI_min [mm⁴]."""
    geom = _cable_geometry(n_strands, wire_radius, pitch_length=100.0)
    return geom["EI_max"] - geom["EI_min"]


def helix_factor(pitch_length: float, R_helix: float = 1.0) -> float:
    """ピッチ依存因子 g(p) = sin²(α_h) × cos(α_h)."""
    alpha_h = math.atan(2.0 * math.pi * R_helix / pitch_length)
    return math.sin(alpha_h) ** 2 * math.cos(alpha_h)


def analytical_model(X, C, alpha, beta):
    """W_diss = C × ΔEI × κ^alpha × g(p)^beta.

    X = (dEI, kappa, gp) の配列。
    """
    dEI, kappa, gp = X
    return C * dEI * kappa**alpha * gp**beta


def analytical_model_simple(X, C, alpha):
    """簡易モデル: W_diss = C × ΔEI × κ^alpha（ピッチ依存なし）."""
    dEI, kappa = X
    return C * dEI * kappa**alpha


# ====================================================================
# パラメータ掃引
# ====================================================================


def run_sweep():
    """パラメータ掃引実行."""
    # --- 掃引パラメータ ---
    n_strands_list = [2, 3, 5, 7, 12, 19]
    pitch_list = [30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0]
    kappa_list = [0.0005, 0.001, 0.002, 0.003, 0.005]

    # 固定パラメータ（軽量設定）
    common = dict(
        n_increments_per_half=15,
        n_half_cycles=2,
        n_elements_per_pitch=8,
        fiber_n_fiber=20,
        n_layers=15,
        show_progress=False,
    )

    results = []
    total = len(n_strands_list) * len(pitch_list) * len(kappa_list)
    count = 0
    t0 = time.time()

    # --- Phase 1: κ 掃引（n, p 固定） ---
    print("=" * 60)
    print("Phase 1: κ 掃引（n_strands × pitch × kappa）")
    print(f"  合計ケース: {total}")
    print("=" * 60)

    proc = CableDissipationProcess()

    for n in n_strands_list:
        for p in pitch_list:
            for kappa in kappa_list:
                count += 1
                try:
                    cfg = CableDissipationConfig(
                        n_strands=n,
                        pitch_length=p,
                        bending_curvature=kappa,
                        **common,
                    )
                    result = proc.process(cfg)

                    dEI = delta_EI(n)
                    geom = _cable_geometry(n, 0.5, p)
                    gp = helix_factor(p, geom["helix_radius"])

                    entry = {
                        "n_strands": n,
                        "pitch_length": p,
                        "bending_curvature": kappa,
                        "dissipation": result.dissipation_energy,
                        "converged": result.solver_result.converged,
                        "dEI": dEI,
                        "EI_ratio": geom["EI_ratio"],
                        "helix_factor": gp,
                        "helix_angle_deg": math.degrees(geom["helix_angle"]),
                        "dissipation_ratio": result.mk_metrics["dissipation_ratio"],
                        "EI_secant": result.mk_metrics["EI_secant"],
                    }
                    results.append(entry)

                    elapsed = time.time() - t0
                    rate = count / elapsed if elapsed > 0 else 0
                    eta = (total - count) / rate if rate > 0 else 0
                    status = "OK" if result.solver_result.converged else "FAIL"
                    print(
                        f"  [{count:3d}/{total}] n={n:2d} p={p:5.0f} κ={kappa:.4f}"
                        f"  W={result.dissipation_energy:.4e}"
                        f"  ratio={result.mk_metrics['dissipation_ratio']:.3f}"
                        f"  [{status}] ({elapsed:.0f}s, ETA {eta:.0f}s)"
                    )
                except Exception as e:
                    print(f"  [{count:3d}/{total}] n={n:2d} p={p:5.0f} κ={kappa:.4f}  ERROR: {e}")

    print(f"\n掃引完了: {len(results)}/{total} ケース成功, {time.time()-t0:.0f}s")
    return results


# ====================================================================
# フィッティング + 可視化
# ====================================================================


def fit_and_plot(results: list[dict]):
    """回帰フィッティングと診断プロット."""
    # 収束ケースのみ使用
    data = [r for r in results if r["converged"] and r["dissipation"] > 1e-15]
    print(f"\nフィッティング対象: {len(data)}/{len(results)} ケース")

    if len(data) < 5:
        print("ERROR: データ不足でフィッティング不可")
        return

    dEI = np.array([d["dEI"] for d in data])
    kappa = np.array([d["bending_curvature"] for d in data])
    gp = np.array([d["helix_factor"] for d in data])
    W = np.array([d["dissipation"] for d in data])
    n_arr = np.array([d["n_strands"] for d in data])
    p_arr = np.array([d["pitch_length"] for d in data])

    # --- Fit 1: 簡易モデル W = C × ΔEI × κ^α ---
    print("\n--- Fit 1: W = C × ΔEI × κ^α ---")
    try:
        popt1, pcov1 = curve_fit(
            analytical_model_simple,
            (dEI, kappa),
            W,
            p0=[1.0, 2.0],
            maxfev=10000,
        )
        C1, alpha1 = popt1
        W_pred1 = analytical_model_simple((dEI, kappa), *popt1)
        r2_1 = 1.0 - np.sum((W - W_pred1) ** 2) / np.sum((W - W.mean()) ** 2)
        rmse1 = np.sqrt(np.mean((W - W_pred1) ** 2))
        print(f"  C = {C1:.6f}")
        print(f"  α = {alpha1:.4f}")
        print(f"  R² = {r2_1:.6f}")
        print(f"  RMSE = {rmse1:.4e}")
    except Exception as e:
        print(f"  Fit 1 失敗: {e}")
        popt1 = None

    # --- Fit 2: 完全モデル W = C × ΔEI × κ^α × g(p)^β ---
    print("\n--- Fit 2: W = C × ΔEI × κ^α × g(p)^β ---")
    try:
        popt2, pcov2 = curve_fit(
            analytical_model,
            (dEI, kappa, gp),
            W,
            p0=[1.0, 2.0, 1.0],
            maxfev=10000,
        )
        C2, alpha2, beta2 = popt2
        W_pred2 = analytical_model((dEI, kappa, gp), *popt2)
        r2_2 = 1.0 - np.sum((W - W_pred2) ** 2) / np.sum((W - W.mean()) ** 2)
        rmse2 = np.sqrt(np.mean((W - W_pred2) ** 2))
        print(f"  C = {C2:.6f}")
        print(f"  α = {alpha2:.4f}")
        print(f"  β = {beta2:.4f}")
        print(f"  R² = {r2_2:.6f}")
        print(f"  RMSE = {rmse2:.4e}")
    except Exception as e:
        print(f"  Fit 2 失敗: {e}")
        popt2 = None

    # --- Fit 3: 対数線形モデル log(W) = a + b×log(ΔEI) + c×log(κ) + d×log(g) ---
    print("\n--- Fit 3: log(W) = a + b×log(ΔEI) + c×log(κ) + d×log(g) ---")
    mask = (W > 1e-15) & (gp > 1e-15)
    if mask.sum() > 4:
        log_W = np.log(W[mask])
        log_dEI = np.log(dEI[mask])
        log_kappa = np.log(kappa[mask])
        log_gp = np.log(gp[mask])

        A = np.column_stack([np.ones(mask.sum()), log_dEI, log_kappa, log_gp])
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, log_W, rcond=None)
        a, b, c, d = coeffs

        log_W_pred = A @ coeffs
        ss_res = np.sum((log_W - log_W_pred) ** 2)
        ss_tot = np.sum((log_W - log_W.mean()) ** 2)
        r2_3 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"  log(W) = {a:.4f} + {b:.4f}×log(ΔEI) + {c:.4f}×log(κ) + {d:.4f}×log(g)")
        print(f"  → W ≈ {math.exp(a):.6f} × ΔEI^{b:.3f} × κ^{c:.3f} × g(p)^{d:.3f}")
        print(f"  R²(log) = {r2_3:.6f}")

        # 各項の寄与度
        var_total = np.var(log_W)
        for name, col in [("ΔEI", log_dEI), ("κ", log_kappa), ("g(p)", log_gp)]:
            corr = np.corrcoef(col, log_W)[0, 1]
            print(f"  相関(log {name}, log W) = {corr:.4f}")
    else:
        print("  有効データ不足")
        coeffs = None

    # ============================================================
    # 診断プロット
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) W vs κ（n別）
    ax = axes[0, 0]
    for n in sorted(set(n_arr)):
        mask_n = n_arr == n
        # p=100 のケースのみ
        mask_p = p_arr == 100.0
        m = mask_n & mask_p
        if m.sum() > 0:
            ax.loglog(kappa[m], W[m], "o-", label=f"n={n}", markersize=6)
    # べき乗参考線
    kref = np.logspace(-3.5, -2.0, 50)
    if popt1 is not None:
        ax.loglog(kref, kref**alpha1 * W[kappa == 0.001].mean() / 0.001**alpha1,
                  "k--", alpha=0.3, label=f"κ^{alpha1:.2f}")
    ax.set_xlabel("κ [1/mm]")
    ax.set_ylabel("W_diss [N·mm²]")
    ax.set_title("(a) 散逸 vs 曲率（p=100mm）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) W vs n_strands（κ別）
    ax = axes[0, 1]
    for k in sorted(set(kappa)):
        mask_k = kappa == k
        mask_p = p_arr == 100.0
        m = mask_k & mask_p
        if m.sum() > 1:
            idx = np.argsort(n_arr[m])
            ax.semilogy(n_arr[m][idx], W[m][idx], "o-", label=f"κ={k}", markersize=6)
    ax.set_xlabel("n_strands")
    ax.set_ylabel("W_diss [N·mm²]")
    ax.set_title("(b) 散逸 vs 撚線本数（p=100mm）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) W vs pitch（n, κ 別）
    ax = axes[0, 2]
    for n in [7, 19]:
        for k in [0.001, 0.002]:
            mask_n = n_arr == n
            mask_k = kappa == k
            m = mask_n & mask_k
            if m.sum() > 1:
                idx = np.argsort(p_arr[m])
                ax.plot(p_arr[m][idx], W[m][idx], "o-",
                        label=f"n={n}, κ={k}", markersize=6)
    ax.set_xlabel("pitch [mm]")
    ax.set_ylabel("W_diss [N·mm²]")
    ax.set_title("(c) 散逸 vs ピッチ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Parity plot（Fit 2）
    ax = axes[1, 0]
    if popt2 is not None:
        ax.loglog(W, W_pred2, "o", markersize=4, alpha=0.6)
        wmin, wmax = W.min() * 0.5, W.max() * 2.0
        ax.loglog([wmin, wmax], [wmin, wmax], "k-", alpha=0.5)
        ax.loglog([wmin, wmax], [wmin * 2, wmax * 2], "k:", alpha=0.3)
        ax.loglog([wmin, wmax], [wmin * 0.5, wmax * 0.5], "k:", alpha=0.3)
        ax.set_xlabel("W_diss (numerical)")
        ax.set_ylabel("W_diss (fit)")
        ax.set_title(f"(d) Parity plot (Fit 2, R²={r2_2:.4f})")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Fit 2 failed", ha="center", va="center",
                transform=ax.transAxes)

    # (e) 残差分布
    ax = axes[1, 1]
    if popt2 is not None:
        rel_err = (W_pred2 - W) / W * 100
        ax.hist(rel_err, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="k", linestyle="--")
        ax.set_xlabel("相対誤差 [%]")
        ax.set_ylabel("頻度")
        ax.set_title(f"(e) 残差分布 (mean={rel_err.mean():.1f}%, std={rel_err.std():.1f}%)")
        ax.grid(True, alpha=0.3)

    # (f) 解析式サマリ
    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        "散逸エネルギー解析的近似式",
        "=" * 50,
        "",
        "■ 物理モデル:",
        "  W_diss(n, p, κ) = C × ΔEI(n) × κ^α × g(p)^β",
        "",
        "  ΔEI(n) = EI_max - EI_min",
        "         = n_outer × A_wire × R_helix²  [Steiner]",
        f"  g(p)   = sin²(α_h) × cos(α_h)",
        f"  α_h    = atan(2π R_helix / p)",
        "",
    ]
    if popt2 is not None:
        lines.extend([
            f"■ Fit 2 (完全モデル):",
            f"  C = {C2:.6f}",
            f"  α = {alpha2:.4f}  (κ べき指数)",
            f"  β = {beta2:.4f}  (ピッチ factor べき)",
            f"  R² = {r2_2:.6f}",
            f"  RMSE = {rmse2:.4e}",
            "",
        ])
    if popt1 is not None:
        lines.extend([
            f"■ Fit 1 (簡易, ピッチ無視):",
            f"  C = {C1:.6f}, α = {alpha1:.4f}",
            f"  R² = {r2_1:.6f}",
            "",
        ])
    if coeffs is not None:
        lines.extend([
            f"■ Fit 3 (対数線形):",
            f"  W ≈ {math.exp(a):.4e} × ΔEI^{b:.3f} × κ^{c:.3f} × g^{d:.3f}",
            f"  R²(log) = {r2_3:.6f}",
            "",
        ])
    lines.extend([
        f"■ データ: {len(data)} ケース",
        f"  n ∈ {sorted(set(n_arr))}",
        f"  p ∈ {sorted(set(p_arr))} mm",
        f"  κ ∈ {sorted(set(kappa))} 1/mm",
    ])

    ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    out = "/home/user/xkep-cae/work/beam_hysteresis/06_dissipation_formula.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nPlot saved: {out}")

    return popt1, popt2, coeffs


# ====================================================================
# メイン
# ====================================================================


def main():
    results = run_sweep()
    if results:
        fit_and_plot(results)


if __name__ == "__main__":
    main()
