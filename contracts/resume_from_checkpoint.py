"""チェックポイントからのソルバー再開スクリプト（status-278）.

使い方:
    # 1. チェックポイント保存（7本撚線 frac=0.50）
    XKEP_CHECKPOINT_FRAC=0.50 XKEP_CHECKPOINT_PATH=/tmp/7wire_ckpt_050.pkl \
    python -c "
    from xkep_cae.numerical_tests.strand_bending_oscillation import *
    cfg = StrandBendingOscillationConfig(n_strands=7, ...)
    StrandBendingOscillationProcess().process(cfg)
    "

    # 2. チェックポイントから再開（10インクリメントだけ）
    python contracts/resume_from_checkpoint.py /tmp/7wire_ckpt_050.pkl --max-incr 10

[← README](../README.md)
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time

import numpy as np


def resume(ckpt_path: str, max_incr: int = 10) -> None:
    """チェックポイントからソルバーを再開し、数インクリメント実行する."""
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    state = ckpt["state"]
    load_frac = ckpt["load_frac"]
    k_pen = ckpt["k_pen"]
    dt_sub = ckpt["dt_sub"]
    incr_start = ckpt["incr_count"]

    print(f"=== チェックポイント復元 ===")
    print(f"  frac={load_frac:.4f}, incr={incr_start}, k_pen={k_pen:.4e}")
    print(f"  ||u||={np.linalg.norm(state.u):.6e}")
    print(f"  dt_sub={dt_sub:.6e}")
    print(f"  max_incr={max_incr}")

    # 時間積分状態の復元
    from xkep_cae.time_integration.strategy import GeneralizedAlphaProcess

    time_strategy = GeneralizedAlphaProcess.__new__(GeneralizedAlphaProcess)
    # チェックポイントの時間積分状態を直接設定
    time_strategy.vel = ckpt["time_vel"]
    time_strategy.acc = ckpt["time_acc"]
    if ckpt["time_vel_old"] is not None:
        time_strategy._vel_old = ckpt["time_vel_old"]
    if ckpt["time_acc_old"] is not None:
        time_strategy._acc_old = ckpt["time_acc_old"]
    if ckpt["time_u_pred"] is not None:
        time_strategy._u_pred = ckpt["time_u_pred"]

    print(f"  ||vel||={np.linalg.norm(time_strategy.vel):.6e}")
    print(f"  ||acc||={np.linalg.norm(time_strategy.acc):.6e}")

    # manager状態
    n_pairs = len(ckpt["manager_pairs"])
    n_active = sum(
        1
        for p in ckpt["manager_pairs"]
        if hasattr(p, "state") and p.state.p_n > 0
    )
    print(f"  pairs={n_pairs}, active={n_active}")

    print(f"\n=== 再開: {max_incr} インクリメント ===")
    # ここからソルバーのメインループを模擬する
    # （実際のソルバー再開にはContactFrictionProcessの内部状態が必要なため、
    #   完全な再開は困難。代わりにチェックポイントの状態を表示して
    #   対策効果の比較基盤を提供する。）

    print("\n[INFO] 完全なソルバー再開にはContactFrictionProcess内部のリファクタリングが必要。")
    print("[INFO] 現在はチェックポイントの状態表示のみ。")
    print("[INFO] 対策効果の検証には、チェックポイント前後のfrac比較を推奨。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="チェックポイントからのソルバー再開")
    parser.add_argument("checkpoint", help="チェックポイントファイルパス")
    parser.add_argument("--max-incr", type=int, default=10, help="最大インクリメント数")
    args = parser.parse_args()
    resume(args.checkpoint, args.max_incr)
"""
