#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from utils.config_v2 import build_config_from_files
from utils.env_wrapper_v2 import PVRPDEnv          
from model.models_v1 import WADRLPolicy

import random

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def infer_feature_sizes(env, device):
    with torch.no_grad():
        obs = env.reset()
        # add batch if needed
        def _ensure_batch(x):
            return x.unsqueeze(0) if (torch.is_tensor(x) and x.dim()==2) else x
        obs = {k: _ensure_batch(v) if torch.is_tensor(v) else v for k,v in obs.items()}
        Fg = obs["graph_ctx"].shape[-1]
        Ft = obs["trucks_ctx"].shape[-1]
        Fd = obs["drones_ctx"].shape[-1]
    return Fg, Ft, Fd

def _ensure_batch(obs, device):
    keys = ["graph_ctx", "trucks_ctx", "drones_ctx", "mask_trk", "mask_dr", "weights"]
    out = {}
    for k in keys:
        x = obs[k]
        if torch.is_tensor(x):
            x = x.to(device)
            if k == "weights":
                # weights: [2] -> [1,2]; if already [B,2], leave as-is
                if x.dim() == 1:
                    x = x.unsqueeze(0)

            else:
                # others: [V,F] / [K,F] / [D,F] / [K,V] / [D,V] -> add batch
                if x.dim() == 2:
                    x = x.unsqueeze(0)
        out[k] = x
    return out

@torch.no_grad()
def eval_once(env, policy, device, w1, w2):
    env.set_fixed_weights(w1, w2)
    obs = env.reset()
    obs = _ensure_batch(obs, device=device)
    done = False
    info_last = None
    h = policy.init_hidden(1, device=device)
    while not done:
        action, _, _, _, h = policy.act(obs, h_prev=h, deterministic=True)
        step = env.step(tuple(int(x) for x in action))
        info_last = step.info
        done = bool(step.done.item()) if torch.is_tensor(step.done) else bool(step.done)
        obs = _ensure_batch(step.obs, device=device)
    return float(info_last["F1"]), float(info_last["F2"]), done

import numpy as np
import csv

def sweep_weights(env, policy, device, step=0.02):
    """Evaluate for (w1, w2=1-w1) with w1 in [0,1] stepping by 'step'."""
    w1_list = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)  # 0.00, 0.02, ..., 1.00
    results = []  # list of dicts
    for w1 in w1_list:
        w2 = float(np.round(1.0 - w1, 2))
        F1, F2, done = eval_once(env, policy, device, w1, w2)
        results.append({"w1": float(w1), "w2": float(w2), "F1": F1, "F2": F2, "F1+F2": F1+F2})
        print(f"  w=({w1:.2f},{w2:.2f}): Done={done}   F1={F1:.3f}  F2={F2:.3f}  F1+F2={F1+F2:.3f}")
    return results

def save_csv(rows, out_path):
    header = ["w1", "w2", "F1", "F2", "F1+F2"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved CSV: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truck_json", required=True)
    ap.add_argument("--drone_json", required=True)
    ap.add_argument("--customers_txt", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="One or more .pth/.pt files")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grid_step", type=float, default=0.02)
    ap.add_argument("--csv", type=str, default=None, help="Path to save CSV")
    ap.add_argument("--strict", action="store_true", help="Use strict state_dict loading")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg, _ = build_config_from_files(args.truck_json, args.drone_json, args.customers_txt)  
    # env = PVRPDEnv(cfg, keep_trip_rounds=True, parallel=False, add_idle=False,
    #                max_trips_per_drone=10, normalize_objectives=False,
    #                weight_mode="fixed", device=device, show_moves=True, max_steps=64)

    set_seed(281107)

    env = PVRPDEnv(
        cfg,
        keep_trip_rounds=True,
        parallel=False,
        add_idle=False,
        max_trips_per_drone=200,
        normalize_objectives=False,
        weight_mode="fixed",
        dirichlet_alpha=(1.0, 1.0),
        max_steps=2048,
        device=device,
    )

    Fg, Ft, Fd = infer_feature_sizes(env, device)
    # policy = WADRLPolicy(Fg=Fg, Ft=Ft, Fd=Fd, d_model=256, nhead=8, depth=16).to(device)
    policy = WADRLPolicy(
        Fg=Fg, Ft=Ft, Fd=Fd,
        d_model=128, nhead=8, depth=1, depth_t = 1, depth_d = 1, depth_c=1
    ).to(device)

    for ck in args.ckpt:
        ck = Path(ck)
        print(f"\n== Evaluating {ck.name} ==")
        obj = torch.load(ck, map_location=device)
        state_dict = obj["model"] if "model" in obj else obj
        if args.strict:
            policy.load_state_dict(state_dict, strict=True)
        else:
            model_state = policy.state_dict()
            filtered = {}
            skipped = []
            for k, v in state_dict.items():
                if k not in model_state:
                    skipped.append(k)
                    continue
                if tuple(v.shape) != tuple(model_state[k].shape):
                    skipped.append(k)
                    continue
                filtered[k] = v
            result = policy.load_state_dict(filtered, strict=False)
            if skipped:
                print(f"[WARN] Skipped mismatched keys: {skipped}")
            if result.missing_keys or result.unexpected_keys:
                print(f"[WARN] Non-strict load: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
        policy.eval()

        rows = sweep_weights(env, policy, device, step=float(getattr(args, "grid_step", 0.02)))

        # Optionally save CSV per checkpoint
        if getattr(args, "csv", None):
            # If user passes a directory, save per-ckpt file inside it
            out_path = Path(args.csv)
            out_path.mkdir(parents=True, exist_ok=True) if not (out_path.suffix in [".csv"]) else None
            if out_path.suffix == ".csv":
                # Single CSV for the last checkpoint only (simple behavior)
                save_csv(rows, out_path)
            else:
                save_csv(rows, str(out_path / f"{ck.stem}_grid.csv"))

if __name__ == "__main__":
    main()
