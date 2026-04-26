from __future__ import annotations

"""Compare baseline, fixed reward shaping, and APF-AW-MAPPO runs.

Usage:
    python src/compare_apf_aw_runs.py \
        --run baseline=/path/to/baseline/output \
        --run fixed=/path/to/fixed/output \
        --run apf_aw=/path/to/apf/output \
        --out results/apf_aw_compare
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "collection_reward_episode_reward_mean.csv",
    "collection_reward_reward_mean.csv",
    "train_agents_loss_critic.csv",
    "train_agents_loss_objective.csv",
]

EXTRA_KEYWORDS = [
    "collision",
    "success",
    "smooth",
    "path",
    "length",
    "distance",
    "min_dist",
]


def parse_run_item(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"Run must be formatted as name=/path/to/run, got: {item}")
    name, path = item.split("=", 1)
    name = name.strip()
    path = Path(path.strip())
    if not name:
        raise ValueError(f"Empty run name in: {item}")
    return name, path


def find_metric_csv(run_root: Path, metric_filename: str) -> Path | None:
    matches = sorted(run_root.rglob(metric_filename), key=lambda p: (len(p.parts), str(p)))
    return matches[-1] if matches else None


def read_metric_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if len(df.columns) == 1:
        df = pd.read_csv(csv_path, header=None)

    step_col = None
    value_col = None
    for col in df.columns:
        lc = str(col).lower()
        if step_col is None and lc in {"step", "global_step", "wall_step"}:
            step_col = col
        if value_col is None and lc in {"value", "scalar", "episode_reward_mean", "reward_mean"}:
            value_col = col

    if step_col is None:
        step_col = df.columns[0]
    if value_col is None:
        value_col = df.columns[-1]

    out = df[[step_col, value_col]].copy()
    out.columns = ["step", "value"]
    out["step"] = pd.to_numeric(out["step"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().sort_values("step").reset_index(drop=True)
    return out


def summarize_curve(df: pd.DataFrame) -> dict[str, float | int | None]:
    if df.empty:
        return {
            "n_points": 0,
            "final": np.nan,
            "best": np.nan,
            "best_step": None,
            "last3_mean": np.nan,
            "auc": np.nan,
            "first_positive_step": None,
            "early_min": np.nan,
            "volatility": np.nan,
        }

    x = df["step"].to_numpy(dtype=float)
    y = df["value"].to_numpy(dtype=float)
    pos_idx = np.where(y > 0)[0]
    dy = np.diff(y) if len(y) >= 2 else np.array([0.0])

    return {
        "n_points": int(len(df)),
        "final": float(y[-1]),
        "best": float(np.max(y)),
        "best_step": int(x[np.argmax(y)]),
        "last3_mean": float(np.mean(y[-3:])),
        "auc": float(np.trapz(y, x)) if len(y) >= 2 else 0.0,
        "first_positive_step": int(x[pos_idx[0]]) if len(pos_idx) else None,
        "early_min": float(np.min(y[: max(1, len(y) // 3)])),
        "volatility": float(np.std(dy)),
    }


def discover_metrics(runs: dict[str, Path]) -> list[str]:
    metrics = list(DEFAULT_METRICS)
    for root in runs.values():
        for path in root.rglob("*.csv"):
            lower = path.name.lower()
            if any(k in lower for k in EXTRA_KEYWORDS) and path.name not in metrics:
                metrics.append(path.name)
    return list(dict.fromkeys(metrics))


def plot_metric(metric: str, curves: dict[str, pd.DataFrame], out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, df in curves.items():
        plt.plot(df["step"], df["value"], marker="o", label=name)
    plt.xlabel("step")
    plt.ylabel(metric.replace(".csv", ""))
    plt.title(metric.replace(".csv", ""))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric.replace('.csv', '')}_multi_compare.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple BenchMARL/APF-AW runs")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run mapping formatted as name=/path/to/run. Repeat for multiple runs.",
    )
    parser.add_argument("--out", required=True, help="Output directory for tables and plots")
    args = parser.parse_args()

    runs = dict(parse_run_item(item) for item in args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metric in discover_metrics(runs):
        curves = {}
        for name, root in runs.items():
            csv_path = find_metric_csv(root, metric)
            if csv_path is None:
                continue
            df = read_metric_csv(csv_path)
            curves[name] = df
            row = {"metric": metric, "run": name, "csv": str(csv_path)}
            row.update(summarize_curve(df))
            rows.append(row)

        if len(curves) >= 2:
            plot_metric(metric, curves, out_dir)
            print(f"[Plot] {metric}: {', '.join(curves.keys())}")
        else:
            print(f"[Skip] {metric}: fewer than two comparable runs")

    if not rows:
        raise SystemExit("No comparable CSV metrics found.")

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "apf_aw_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[Saved] {summary_path}")

    pivot_cols = ["final", "best", "last3_mean", "auc", "volatility"]
    for col in pivot_cols:
        pivot = summary.pivot_table(index="metric", columns="run", values=col, aggfunc="first")
        pivot.to_csv(out_dir / f"pivot_{col}.csv")


if __name__ == "__main__":
    main()
