from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_metric_csv(run_root: str | Path, metric_filename: str) -> Path | None:
    run_root = Path(run_root)
    matches = list(run_root.rglob(metric_filename))
    if not matches:
        return None
    matches = sorted(matches, key=lambda p: (len(p.parts), str(p)))
    return matches[-1]


def read_metric_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    step_col = None
    value_col = None

    for c in df.columns:
        lc = str(c).lower()
        if step_col is None and lc in ["step", "global_step", "wall_step"]:
            step_col = c
        if value_col is None and lc in ["value", "scalar", "episode_reward_mean", "reward_mean"]:
            value_col = c

    if step_col is None:
        step_col = df.columns[0]
    if value_col is None:
        value_col = df.columns[-1]

    out = df[[step_col, value_col]].copy()
    out.columns = ["step", "value"]
    out = out.dropna()
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().sort_values("step").reset_index(drop=True)
    return out


def summarize_curve(df: pd.DataFrame) -> dict:
    x = df["step"].to_numpy(dtype=float)
    y = df["value"].to_numpy(dtype=float)

    if len(df) == 0:
        return {
            "n_points": 0,
            "final": np.nan,
            "best": np.nan,
            "best_step": np.nan,
            "last3_mean": np.nan,
            "auc": np.nan,
            "first_positive_step": None,
            "early_min": np.nan,
            "volatility": np.nan,
        }

    auc = np.trapz(y, x) if len(df) >= 2 else 0.0
    pos_idx = np.where(y > 0)[0]
    first_positive_step = int(x[pos_idx[0]]) if len(pos_idx) > 0 else None
    early_part = y[: max(1, len(y) // 3)]
    dy = np.diff(y) if len(y) >= 2 else np.array([0.0])
    volatility = float(np.std(dy))

    return {
        "n_points": int(len(df)),
        "final": float(y[-1]),
        "best": float(np.max(y)),
        "best_step": int(x[np.argmax(y)]),
        "last3_mean": float(np.mean(y[-3:])),
        "auc": float(auc),
        "first_positive_step": first_positive_step,
        "early_min": float(np.min(early_part)),
        "volatility": volatility,
    }


def compare_metric(metric_name: str, base_csv: str | Path, shape_csv: str | Path, out_dir: str | Path) -> dict:
    base_df = read_metric_csv(base_csv)
    shape_df = read_metric_csv(shape_csv)

    base_sum = summarize_curve(base_df)
    shape_sum = summarize_curve(shape_df)

    print("=" * 90)
    print(f"[Metric] {metric_name}")
    print(f"baseline csv : {base_csv}")
    print(f"shaping  csv : {shape_csv}")
    print("-" * 90)
    print("{:<18} {:>14} {:>14}".format("stat", "baseline", "shaping"))
    for k in [
        "n_points",
        "final",
        "best",
        "best_step",
        "last3_mean",
        "auc",
        "first_positive_step",
        "early_min",
        "volatility",
    ]:
        print("{:<18} {:>14} {:>14}".format(k, str(base_sum[k]), str(shape_sum[k])))

    plt.figure(figsize=(8, 5))
    plt.plot(base_df["step"], base_df["value"], marker="o", label="baseline")
    plt.plot(shape_df["step"], shape_df["value"], marker="o", label="reward shaping")
    plt.xlabel("step")
    plt.ylabel(metric_name.replace(".csv", ""))
    plt.title(metric_name.replace(".csv", ""))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = Path(out_dir) / f"{metric_name.replace('.csv', '')}_compare.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {
        "metric": metric_name,
        "baseline_final": base_sum["final"],
        "shaping_final": shape_sum["final"],
        "baseline_best": base_sum["best"],
        "shaping_best": shape_sum["best"],
        "baseline_last3_mean": base_sum["last3_mean"],
        "shaping_last3_mean": shape_sum["last3_mean"],
        "baseline_auc": base_sum["auc"],
        "shaping_auc": shape_sum["auc"],
        "baseline_first_positive_step": base_sum["first_positive_step"],
        "shaping_first_positive_step": shape_sum["first_positive_step"],
        "baseline_early_min": base_sum["early_min"],
        "shaping_early_min": shape_sum["early_min"],
        "baseline_volatility": base_sum["volatility"],
        "shaping_volatility": shape_sum["volatility"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two BenchMARL run directories")
    parser.add_argument("baseline_run_root", type=str, help="Path to baseline run output directory")
    parser.add_argument("shaping_run_root", type=str, help="Path to reward shaping run output directory")
    parser.add_argument("out_dir", type=str, help="Directory to save comparison summary and plots")
    args = parser.parse_args()

    baseline_root = args.baseline_run_root
    shaping_root = args.shaping_run_root
    out_dir = args.out_dir

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    candidate_metrics = [
        "collection_reward_episode_reward_mean.csv",
        "collection_reward_reward_mean.csv",
    ]

    extra_keywords = ["collision", "success", "smooth", "path", "length", "distance"]
    for root in [baseline_root, shaping_root]:
        for p in Path(root).rglob("*.csv"):
            name = p.name.lower()
            if any(k in name for k in extra_keywords):
                if p.name not in candidate_metrics:
                    candidate_metrics.append(p.name)

    rows = []
    for metric in dict.fromkeys(candidate_metrics):
        base_csv = find_metric_csv(baseline_root, metric)
        shape_csv = find_metric_csv(shaping_root, metric)

        if base_csv is None or shape_csv is None:
            print(f"[Skip] {metric}: one side missing")
            continue

        row = compare_metric(metric, base_csv, shape_csv, out_dir)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = Path(out_dir) / "comparison_summary.csv"
        df.to_csv(csv_path, index=False)
        print("=" * 90)
        print(f"summary saved to: {csv_path}")
    else:
        print("No comparable metric csv found.")


if __name__ == "__main__":
    main()
