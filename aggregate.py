#!/usr/bin/env python3
"""mean ± std over seeds of
balanced accuracy per skin-type group, and the cost of privacy
Delta_g(eps) = BalAcc_g(inf) - BalAcc_g(eps), computed per seed and then averaged.

"""
import argparse
import glob
import json
import math
import os

import pandas as pd

GROUPS = ["overall", "I-II", "III-IV", "V-VI"]

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--split", default="test", choices=["val", "test"])
args = ap.parse_args()

rows = []
for path in sorted(glob.glob(os.path.join(args.folder, "*_eps*_seed*.json"))):
    with open(path) as f:
        r = json.load(f)  
    rows.append({"method": r["method"], "model": r["model"], "epsilon": r["epsilon"],
                 "seed": r["seed"], "sigma": r["noise_multiplier"],
                 **{g: r[args.split][g] for g in GROUPS}})
if not rows:
    raise SystemExit(f"no result files found in {args.folder}")
df = pd.DataFrame(rows)

# Delta per seed against the same seed's non-private run (same split, so a paired comparison)
base = df[df["epsilon"].map(math.isinf)].set_index(["method", "model", "seed"])[GROUPS]
for g in GROUPS:
    df[f"delta_{g}"] = [base.loc[(m, mo, s), g] - v if (m, mo, s) in base.index else float("nan")
                        for m, mo, s, v in zip(df["method"], df["model"], df["seed"], df[g])]

cols = GROUPS + [f"delta_{g}" for g in GROUPS]
agg = df.groupby(["method", "model", "epsilon"])[cols].agg(["mean", "std"])
agg[("n_seeds", "")] = df.groupby(["method", "model", "epsilon"]).size()
agg[("sigma", "")] = df.groupby(["method", "model", "epsilon"])["sigma"].mean()

pd.set_option("display.width", 200)
print(f"Balanced accuracy on {args.split} (mean ± std over seeds)\n")
fmt = lambda m, s: f"{m:.3f} ± {0 if pd.isna(s) else s:.3f}"
for prefix, title in [("", "Balanced accuracy"), ("delta_", "Cost of privacy Δ_g(ε) = BalAcc(∞) − BalAcc(ε)")]:
    print(title)
    table = pd.DataFrame({g: [fmt(m, s) for m, s in zip(agg[(prefix + g, "mean")], agg[(prefix + g, "std")])]
                          for g in GROUPS}, index=agg.index)
    table["seeds"] = agg[("n_seeds", "")].values
    table["sigma"] = agg[("sigma", "")].round(3).values
    print(table.to_string(), "\n")

out = os.path.join(args.folder, f"summary_{args.split}.csv")
agg.columns = ["_".join(c).strip("_") for c in agg.columns]
agg.to_csv(out)
print(f"saved {out}")
