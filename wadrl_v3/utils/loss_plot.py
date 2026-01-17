#!/usr/bin/env python3
# save as: plot_wadrl_logs.py
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

def parse_log(path: Path):
    eval_steps, f1_best, f2_best, fsum_best = [], [], [], []
    iter_steps, losses = [], []

    # Regex patterns
    # Example: [Eval @ 6900] F1_best=3227.052  F2_best=16343.301  (F1+F2)_best=19855.992
    re_eval = re.compile(
        r"\[Eval @\s*(\d+)\]\s*F1_best=([-\d.eE]+)\s*F2_best=([-\d.eE]+)\s*\(F1\+F2\)_best=([-\d.eE]+)"
    )
    # Example: [Iter 07000] loss=560299.8750 pg=-0.0086 v=1120599.7500 ent=0.3750 advμ=-26.575 advσ=1078.016
    re_iter = re.compile(
        r"\[Iter\s+(\d+)\]\s*loss=([-\d.eE]+)\s*pg=([-\d.eE]+)\s*v=([-\d.eE]+)\s*ent=([-\d.eE]+)\s*advμ=([-\d.eE]+)\s*advσ=([-\d.eE]+)"
    )

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_eval = re_eval.search(line)
            if m_eval:
                step = int(m_eval.group(1))
                f1 = float(m_eval.group(2))
                f2 = float(m_eval.group(3))
                fsum = float(m_eval.group(4))
                eval_steps.append(step)
                f1_best.append(f1)
                f2_best.append(f2)
                fsum_best.append(fsum)
                continue

            m_iter = re_iter.search(line)
            if m_iter:
                step = int(m_iter.group(1))
                loss = float(m_iter.group(2))
                iter_steps.append(step)
                losses.append(loss)
                continue

    return {
        "eval_steps": eval_steps,
        "f1_best": f1_best,
        "f2_best": f2_best,
        "fsum_best": fsum_best,
        "iter_steps": iter_steps,
        "losses": losses,
    }

def plot_series(x, y, title, ylabel, outpath: Path):
    if not x or not y:
        print(f"[warn] No data for '{title}'. Skipping plot.")
        return
    plt.figure()
    plt.plot(x, y, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[ok] Wrote {outpath}")

def main():
    ap = argparse.ArgumentParser(description="Parse WADRL PPO logs and plot metrics.")
    ap.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to the training log file (e.g., results/v2/log.txt)",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("plots"),
        help="Directory to write plots",
    )
    ap.add_argument(
        "--dashboard",
        action="store_true",
        help="Also create a combined 2x2 dashboard image",
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    data = parse_log(args.log)

    # Individual plots
    plot_series(data["eval_steps"], data["f1_best"],
                "Best F1 vs Step", "F1_best",
                args.outdir / "best_f1.png")

    plot_series(data["eval_steps"], data["f2_best"],
                "Best F2 vs Step", "F2_best",
                args.outdir / "best_f2.png")

    plot_series(data["eval_steps"], data["fsum_best"],
                "Best (F1+F2) vs Step", "(F1+F2)_best",
                args.outdir / "best_fsum.png")

    plot_series(data["iter_steps"], data["losses"],
                "Loss vs Iteration", "Loss",
                args.outdir / "loss.png")

    # Optional 2x2 dashboard
    if args.dashboard:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

        # Top-left: F1
        if data["eval_steps"]:
            axes[0, 0].plot(data["eval_steps"], data["f1_best"], marker="o", linewidth=1)
            axes[0, 0].set_title("Best F1")
            axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("F1_best")
            axes[0, 0].grid(True, linestyle="--", alpha=0.4)

        # Top-right: F2
        if data["eval_steps"]:
            axes[0, 1].plot(data["eval_steps"], data["f2_best"], marker="o", linewidth=1)
            axes[0, 1].set_title("Best F2")
            axes[0, 1].set_xlabel("Step"); axes[0, 1].set_ylabel("F2_best")
            axes[0, 1].grid(True, linestyle="--", alpha=0.4)

        # Bottom-left: F1+F2
        if data["eval_steps"]:
            axes[1, 0].plot(data["eval_steps"], data["fsum_best"], marker="o", linewidth=1)
            axes[1, 0].set_title("Best (F1+F2)")
            axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("(F1+F2)_best")
            axes[1, 0].grid(True, linestyle="--", alpha=0.4)

        # Bottom-right: loss
        if data["iter_steps"]:
            axes[1, 1].plot(data["iter_steps"], data["losses"], linewidth=1)
            axes[1, 1].set_title("Loss")
            axes[1, 1].set_xlabel("Iteration"); axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(True, linestyle="--", alpha=0.4)

        dash_out = args.outdir / "dashboard_2x2.png"
        fig.savefig(dash_out, dpi=150)
        plt.close(fig)
        print(f"[ok] Wrote {dash_out}")

if __name__ == "__main__":
    main()
