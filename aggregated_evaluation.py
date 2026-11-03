"""
Aggregate evaluation results across multiple runs (e.g. random seeds or CV folds).

Expects one folder per run, each written by evaluation.py and containing
  overall_metrics.csv      (Micro / Macro Average rows)
  per_entity_metrics.csv   (one row per entity)

Reports mean, standard deviation (sample SD, n-1) and a 95% confidence
interval (t-distribution) for every precision / recall / F1 metric.

Seeds used in the experiments: 1337, 42, 7, 123, 2024

Usage:
  python aggregated_evaluation.py --eval_dir eval --prefix seed_ --out eval/aggregate

  --eval_dir  folder containing the per-run folders (eval/seed_1337, eval/seed_42, ...)
  --prefix    only use subfolders whose names start with this (default: seed_)
  --out       where to write the aggregate CSVs (default: <eval_dir>/aggregate)
  --decimals  rounding for printed / formatted values (default: 3)
"""
import os
import math
import argparse
import pandas as pd

METRICS = [
    "Strict Precision", "Strict Recall", "Strict F1-score",
    "Lenient Precision", "Lenient Recall", "Lenient F1-score",
]

# two-sided 95% t critical values for df = 1..30 (used if scipy is not installed)
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
       2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
       2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]

def t_crit(df):
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except ImportError:
        return T95[df - 1] if 1 <= df <= 30 else 1.96

def summarize(values):
    """mean, sd, ci_low, ci_high for a list of numbers (NaNs ignored)."""
    v = pd.Series(values, dtype=float).dropna()
    n = len(v)
    if n == 0:
        return math.nan, math.nan, math.nan, math.nan, 0
    mean = v.mean()
    if n == 1:
        return mean, math.nan, math.nan, math.nan, 1
    sd = v.std(ddof=1)
    half = t_crit(n - 1) * sd / math.sqrt(n)
    return mean, sd, mean - half, mean + half, n

def find_runs(eval_dir, prefix):
    runs = []
    for name in sorted(os.listdir(eval_dir)):
        path = os.path.join(eval_dir, name)
        if not (os.path.isdir(path) and name.startswith(prefix)):
            continue
        overall = os.path.join(path, "overall_metrics.csv")
        entity = os.path.join(path, "per_entity_metrics.csv")
        if os.path.exists(overall) and os.path.exists(entity):
            runs.append((name, overall, entity))
        else:
            print(f"[skip] {name}: missing overall_metrics.csv or per_entity_metrics.csv")
    return runs

def aggregate(frames, key_col, run_names, decimals):
    """frames: list of DataFrames (one per run), each with key_col + METRICS."""
    keys = list(dict.fromkeys(k for f in frames for k in f[key_col]))
    stats_rows, fmt_rows = [], []
    for key in keys:
        srow = {key_col: key}
        frow = {key_col: key}
        for m in METRICS:
            vals = []
            for f in frames:
                hit = f.loc[f[key_col] == key, m] if m in f.columns else pd.Series(dtype=float)
                vals.append(float(hit.iloc[0]) if len(hit) else math.nan)
            mean, sd, lo, hi, n = summarize(vals)
            srow[f"{m} mean"] = mean
            srow[f"{m} sd"] = sd
            srow[f"{m} ci95_low"] = lo
            srow[f"{m} ci95_high"] = hi
            srow["n_runs"] = n
            frow[m] = (f"{mean:.{decimals}f} ± {sd:.{decimals}f}"
                       if not math.isnan(sd) else f"{mean:.{decimals}f}")
        stats_rows.append(srow)
        fmt_rows.append(frow)
    return pd.DataFrame(stats_rows), pd.DataFrame(fmt_rows)

def main():
    ap = argparse.ArgumentParser(description="Aggregate evaluation.py results across runs.")
    ap.add_argument("--eval_dir", required=True, help="folder with one subfolder per run")
    ap.add_argument("--prefix", default="seed_", help="run folder name prefix (default: seed_)")
    ap.add_argument("--out", default=None, help="output folder (default: <eval_dir>/aggregate)")
    ap.add_argument("--decimals", type=int, default=3)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.eval_dir, "aggregate")
    os.makedirs(out_dir, exist_ok=True)

    runs = find_runs(args.eval_dir, args.prefix)
    if not runs:
        raise SystemExit(f"No run folders starting with '{args.prefix}' found in {args.eval_dir}")
    names = [r[0] for r in runs]
    print(f"Aggregating {len(runs)} runs: {', '.join(names)}")

    overall_frames = [pd.read_csv(r[1]) for r in runs]
    entity_frames = [pd.read_csv(r[2]) for r in runs]

    # per-run overall table (useful for checking individual seeds)
    per_run = []
    for name, f in zip(names, overall_frames):
        for _, row in f.iterrows():
            per_run.append({"run": name, "Metric": row["Metric"],
                            **{m: row[m] for m in METRICS if m in f.columns}})
    per_run_df = pd.DataFrame(per_run)

    overall_stats, overall_fmt = aggregate(overall_frames, "Metric", names, args.decimals)
    entity_stats, entity_fmt = aggregate(entity_frames, "Entity", names, args.decimals)

    per_run_df.to_csv(os.path.join(out_dir, "per_run_overall.csv"), index=False)
    overall_stats.to_csv(os.path.join(out_dir, "overall_mean_sd_ci.csv"), index=False)
    entity_stats.to_csv(os.path.join(out_dir, "per_entity_mean_sd_ci.csv"), index=False)
    overall_fmt.to_csv(os.path.join(out_dir, "overall_mean_sd_formatted.csv"), index=False)
    entity_fmt.to_csv(os.path.join(out_dir, "per_entity_mean_sd_formatted.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"\nPer-entity (mean ± SD over {len(runs)} runs):")
    print(entity_fmt.to_string(index=False))
    print(f"\nOverall (mean ± SD over {len(runs)} runs):")
    print(overall_fmt.to_string(index=False))

    print("\n95% confidence intervals (overall):")
    for _, r in overall_stats.iterrows():
        for m in ["Strict F1-score", "Lenient F1-score"]:
            lo, hi = r[f"{m} ci95_low"], r[f"{m} ci95_high"]
            if not math.isnan(lo):
                print(f"  {r['Metric']:<14} {m:<17} {r[f'{m} mean']:.{args.decimals}f}  "
                      f"[{lo:.{args.decimals}f}, {hi:.{args.decimals}f}]")

    print(f"\nSaved to {out_dir}/")
    for f in ["per_run_overall.csv", "overall_mean_sd_ci.csv", "per_entity_mean_sd_ci.csv",
              "overall_mean_sd_formatted.csv", "per_entity_mean_sd_formatted.csv"]:
        print(f"  {f}")

if __name__ == "__main__":
    main()
