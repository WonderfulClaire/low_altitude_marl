from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def load_scalar_csv(run_root: str | Path, filename: str) -> pd.DataFrame:
    run_root = Path(run_root)
    for path in run_root.rglob(filename):
        return pd.read_csv(path, header=None, names=["step", "value"])
    raise FileNotFoundError(f"{filename} not found under {run_root}")


def summarize_scalar(run_root: str | Path, filename: str) -> None:
    df = load_scalar_csv(run_root, filename)
    print(f"\n=== {filename} ===")
    print(df.head())
    print("...")
    print(df.tail())
    print("last value:", df["value"].iloc[-1])


def summarize_run(run_root: str | Path) -> None:
    targets = [
        "collection_reward_episode_reward_mean.csv",
        "collection_reward_reward_mean.csv",
        "train_agents_loss_critic.csv",
        "train_agents_loss_objective.csv",
    ]

    print(f"Run root: {run_root}")
    for fname in targets:
        try:
            summarize_scalar(run_root, fname)
        except FileNotFoundError:
            print(f"\n=== {fname} ===")
            print("not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a BenchMARL run directory")
    parser.add_argument(
        "run_root",
        type=str,
        help="Path to the run output directory, e.g. /content/low_altitude_marl/outputs/2026-03-28/19-42-44",
    )
    args = parser.parse_args()
    summarize_run(args.run_root)


if __name__ == "__main__":
    main()
