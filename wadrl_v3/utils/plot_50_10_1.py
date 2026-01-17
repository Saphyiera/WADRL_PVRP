#!/usr/bin/env python3
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1. Load WADRL CSV for instance 50.10.1 ===
csv_path = "/home/saphyiera/HUST/WADRL_PVRP/results/50.10.1/v0/best_f1.csv"
df = pd.read_csv(csv_path)

# Detect F1/F2 columns by name (case-insensitive), or fall back
cols_lower = {c.lower(): c for c in df.columns}

def pick_col(name_hint: str) -> str | None:
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

# WADRL results are in hours -> convert to seconds to match NSGA-II scale
csv_f1 = pd.to_numeric(df[col_f1], errors="coerce").to_numpy() * 3600.0
csv_f2 = pd.to_numeric(df[col_f2], errors="coerce").to_numpy() * 3600.0

# Filter invalid rows (NaNs, extreme outliers)
mask_valid = ~np.isnan(csv_f1) & ~np.isnan(csv_f2) & (csv_f2 <= 1e7)
csv_f1 = csv_f1[mask_valid]
csv_f2 = csv_f2[mask_valid]

# === 2. NSGA-II Pareto for 50.10.1 (in seconds) ===
text_pairs_str = "[(3321.65, 48017.5); (3331.07, 47956.2); (3333.89, 47439.1); (3465.98, 47232); (3582.59, 47178.4); (3706.72, 47101.7); (3714.58, 46596.9); (3747.56, 46432); (3766.9, 46020.6); (3889.51, 45801.5); (3960.74, 45548.2); (4192.41, 45446.2); (4247.31, 44596.3); (4367.36, 44494.1); (4516.36, 44445.9); (4591.17, 44360.3); (4595.54, 44282); (4603.07, 44207.1); (4605.38, 44193.5); (4616.49, 44188.2); (4629.82, 44070.8); (4696.89, 44068.5); (4697.64, 43984.4); (4724.8, 43979.8); (4829.79, 43803.7); (4858.85, 43738.2); (4898.28, 43732.5); (5029.21, 43688.5); (5062.47, 43666); (5071.14, 43580.7); (5123.4, 43515.3); (5134.26, 43412.5); (5137.76, 43315); (5212.74, 43295.6); (5220.24, 43198); (5281.7, 43195.9); (5308.87, 43186); (5331.18, 43181.2); (5424.13, 43099.6); (5428.44, 43087); (5462.25, 43085.9); (5571.08, 43062); (5574.87, 43052.7); (5593.67, 42970.7); (5635.09, 42935.4); (5647.55, 42904.6); (5657.71, 42901.7); (5839.62, 42869.2); (5875.32, 42838.4); (6027.51, 42835.6); (6086.87, 42786.3); (6224.21, 42786); (6267.69, 42783.1); (6307.05, 42781.1); (6362.08, 42746.5); (6388.2, 42745.4); (6406.67, 42734.8); (6468.17, 42726.1); (6498.09, 42700.8); (6512.45, 42695.9); (6626.1, 42694.2); (6632.37, 42693.1); (6698.55, 42683.2); (6725.74, 42652.5); (6728.64, 42643.8); (6885.46, 42641.5); (6928.68, 42625.3); (6953.83, 42597.1); (6976.08, 42592.2); (7318.04, 42543)]"
# Turn the semicolon-separated list into a valid Python list
pairs = ast.literal_eval(text_pairs_str.replace(";", ","))
text_f1 = np.array([p[0] for p in pairs], dtype=float)
text_f2 = np.array([p[1] for p in pairs], dtype=float)

# === 3. Plot ===
plt.figure(figsize=(8, 6))
plt.plot(csv_f1,  csv_f2,  marker="^", linestyle="None", label="WADRL (F1,F2)")
plt.plot(text_f1, text_f2, marker="o", linestyle="None", label="NSGA-II Pareto")

plt.xlabel("F1 (max vehicle return time, seconds)")
plt.ylabel("F2 (sum customer waiting time, seconds)")
plt.title("Instance 50.10.1 – WADRL vs NSGA-II Pareto")

plt.legend()
plt.grid(True)

# Nice axis padding starting from 0
# ax = plt.gca()
# x_max = float(max(csv_f1.max(initial=0), text_f1.max(initial=0)))
# y_max = float(max(csv_f2.max(initial=0), text_f2.max(initial=0)))
# ax.set_xlim(0, x_max * 1.05 if x_max > 0 else 1.0)
# ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)

plt.tight_layout()

out_path = "/home/saphyiera/HUST/WADRL_PVRP/results/50.10.1/v0/best_f1.png"
plt.savefig(out_path, dpi=150)
print("Saved plot to:", out_path)
