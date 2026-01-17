import argparse
import ast
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_BASELINE = "[(2288.85, 18243.1), (2321, 18198.8), (2331.07, 17323), (2348.04, 16621.2), (2442.63, 16537.2), (2458.85, 16151), (2475.3, 15833.5), (2494.84, 15018.6), (2542.71, 14780.8), (2552.49, 14734.1), (2636.24, 14485.7), (2728.2, 14061.9), (2998.16, 13934.2), (3055.08, 13728.9), (3066.92, 13661.4), (3157.39, 13610.2), (3254.45, 13526.5), (3263.7, 13280.9), (3264.22, 13262.5), (3405.85, 13104.4), (3615.77, 13053.2), (3815.36, 13034), (3919.51, 13030.2), (3922.72, 13003), (3977.06, 12982.9), (4205.4, 12953.4), (4271.5, 12933.2), (4286.26, 12932.7), (4569.97, 12883)]"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to input CSV with F1/F2 columns.")
    ap.add_argument("--out", required=True, help="Path to output PNG.")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE, help="Baseline pairs as a Python list string.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Detect F1/F2 columns
    cols_lower = {c.lower(): c for c in df.columns}

    def pick_col(name_hint: str):
        if name_hint.lower() in cols_lower:
            return cols_lower[name_hint.lower()]
        for k, v in cols_lower.items():
            if name_hint.lower() in k:
                return v
        return None

    col_f1 = pick_col("F1")
    col_f2 = pick_col("F2")
    if col_f1 is None or col_f2 is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            col_f1, col_f2 = num_cols[0], num_cols[1]
        else:
            raise ValueError("Could not detect F1/F2 columns and not enough numeric columns in CSV.")

    csv_f1 = pd.to_numeric(df[col_f1], errors="coerce").to_numpy() * 3600
    csv_f2 = pd.to_numeric(df[col_f2], errors="coerce").to_numpy() * 3600

    # Filter: drop rows where F2 > 1e7 (and drop NaNs)
    mask_valid = ~np.isnan(csv_f1) & ~np.isnan(csv_f2) & (csv_f2 <= 4e4)
    csv_f1 = csv_f1[mask_valid]
    csv_f2 = csv_f2[mask_valid]

    # Parse text pairs (unchanged; all are << 1e7 anyway)
    pairs = ast.literal_eval(args.baseline)
    text_f1 = np.array([p[0] for p in pairs], dtype=float)
    text_f2 = np.array([p[1] for p in pairs], dtype=float)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(csv_f1, csv_f2, marker="^", linestyle="None", label="WADRL (F1,F2)", color="blue")
    plt.plot(text_f1, text_f2, marker="o", label="Baseline (F1,F2)", color="yellow")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.title("F1: max vehicle return time; F2: sum customers wait time")
    plt.legend()
    plt.grid(True)

    # # Force axes to start at (0,0)
    # ax = plt.gca()
    # # # Option A: just pin to zero
    # # ax.set_xlim(left=0)
    # # ax.set_ylim(bottom=0)

    # # Option B (nicer padding) — uncomment if you prefer:
    # x_max = float(max(csv_f1.max(initial=0), text_f1.max(initial=0)))
    # y_max = float(max(csv_f2.max(initial=0), text_f2.max(initial=0)))
    # ax.set_xlim(0, x_max * 1.05 if x_max > 0 else 1.0)
    # ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(out_path)


if __name__ == "__main__":
    main()
