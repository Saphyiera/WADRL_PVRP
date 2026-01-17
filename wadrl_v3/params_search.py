#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Iterable

from tqdm import tqdm


DEFAULT_SPACE: Dict[str, List[Any]] = {
    "lr": [3e-5, 1e-4, 3e-4],
    "clip_ratio": [0.1, 0.2, 0.3],
    "vf_coef": [0.2, 0.3, 0.5],
    "ent_coef_start": [0.01, 0.02, 0.03],
    "ent_coef_end": [0.003, 0.005, 0.01],
    "horizon": [64, 128, 256],
    "mem_window": [16, 32, 64],
    "d_model": [128, 256],
    "nhead": [8, 16],
}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["random", "grid"], default="random")
    ap.add_argument("--out_root", type=str, default="../results/params_search")
    ap.add_argument("--train_py", type=str, default=None)
    ap.add_argument("--total_iters", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--customers_txt", type=str, default=None)
    ap.add_argument("--space_json", type=str, default=None)
    ap.add_argument("--base_args_json", type=str, default=None)
    ap.add_argument("--dry_run", action="store_true")
    return ap


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_space(args) -> Dict[str, List[Any]]:
    if args.space_json:
        return load_json(args.space_json)
    return DEFAULT_SPACE


def grid_configs(space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(space.keys())
    if not keys:
        yield {}
        return
    pools = [space[k] for k in keys]
    def rec(i: int, cur: Dict[str, Any]):
        if i == len(keys):
            yield dict(cur)
            return
        for v in pools[i]:
            cur[keys[i]] = v
            yield from rec(i + 1, cur)
    yield from rec(0, {})


def random_config(space: Dict[str, List[Any]], rng: random.Random) -> Dict[str, Any]:
    return {k: rng.choice(v) for k, v in space.items()}


def valid_config(cfg: Dict[str, Any]) -> bool:
    d_model = cfg.get("d_model")
    nhead = cfg.get("nhead")
    if d_model is not None and nhead is not None:
        return (d_model % nhead) == 0
    return True


def build_cmd(train_py: Path, params: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, "-u", str(train_py)]
    for k, v in params.items():
        key = f"--{k}"
        if isinstance(v, (list, tuple)):
            cmd.append(key)
            cmd.extend([str(x) for x in v])
        else:
            cmd.extend([key, str(v)])
    return cmd


def parse_eval_metrics(log_path: Path) -> Dict[str, float] | None:
    if not log_path.exists():
        return None
    pattern = re.compile(
        r"F1_best=([0-9.]+)\s+F2_best=([0-9.]+)\s+\(F1\+F2\)_best=([0-9.]+)\s+Mean_best=([0-9.]+)"
    )
    last = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if m:
            last = {
                "best_f1": float(m.group(1)),
                "best_f2": float(m.group(2)),
                "best_sum": float(m.group(3)),
                "best_mean": float(m.group(4)),
            }
    return last


def main() -> int:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)
    space = get_space(args)

    base_args: Dict[str, Any] = {
        "total_iters": args.total_iters,
        "log_every": args.log_every,
        "save_every": args.save_every,
    }
    if args.customers_txt:
        base_args["customers_txt"] = args.customers_txt
    if args.base_args_json:
        base_args.update(load_json(args.base_args_json))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    train_py = Path(args.train_py) if args.train_py else (Path(__file__).resolve().parent / "ppo" / "train.py")

    summary_path = out_root / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial", "run_dir", "status", "best_f1", "best_f2", "best_sum", "best_mean", "params"],
        )
        writer.writeheader()

        if args.mode == "grid":
            configs = list(grid_configs(space))
        else:
            configs = [random_config(space, rng) for _ in range(args.trials)]

        trial = 0
        for cfg in tqdm(configs, desc="trials"):
            if args.mode == "random" and trial >= args.trials:
                break
            if not valid_config(cfg):
                continue
            run_dir = out_root / f"trial_{trial:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            params = dict(base_args)
            params.update(cfg)
            params["run_dir"] = str(run_dir)

            cmd = build_cmd(train_py, params)
            log_path = run_dir / "log.txt"

            if args.dry_run:
                print(" ".join(cmd))
                trial += 1
                continue

            with open(log_path, "w", encoding="utf-8") as logf:
                result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

            metrics = parse_eval_metrics(log_path) or {}
            writer.writerow({
                "trial": trial,
                "run_dir": str(run_dir),
                "status": result.returncode,
                "best_f1": metrics.get("best_f1", ""),
                "best_f2": metrics.get("best_f2", ""),
                "best_sum": metrics.get("best_sum", ""),
                "best_mean": metrics.get("best_mean", ""),
                "params": json.dumps(params),
            })
            f.flush()
            trial += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
