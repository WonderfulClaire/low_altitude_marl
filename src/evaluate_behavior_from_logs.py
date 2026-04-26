from __future__ import annotations

"""Unified behavior-metric evaluator for MAPPO / reward-shaping runs.

This script compares multiple BenchMARL/W&B output directories using a shared
set of path-planning metrics. It is designed for the thesis workflow where the
training reward may be modified by reward shaping, so raw `episode_reward_mean`
should not be the only comparison criterion.

Example:
    python src/evaluate_behavior_from_logs.py \
        --run baseline=/content/low_altitude_marl/outputs/2026-xx-xx/BASELINE \
        --run fixed=/content/low_altitude_marl/outputs/2026-xx-xx/FIXED \
        --run apf_aw=/content/low_altitude_marl/outputs/2026-xx-xx/APF_AW \
        --out results/unified_behavior_eval

Outputs:
    - behavior_metric_summary.csv
    - behavior_metric_pivot_final.csv
    - behavior_metric_pivot_last3_mean.csv
    - behavior_metric_report.md
    - one PNG curve per comparable metric
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    key: str
    display_name: str
    higher_is_better: bool
    aliases: tuple[str, ...]
    thesis_meaning: str


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="episode_reward_mean",
        display_name="Episode reward mean",
        higher_is_better=True,
        aliases=(
            "collection/reward/episode_reward_mean",
            "collection/agents/reward/episode_reward_mean",
            "collection_reward_episode_reward_mean",
            "collection_agents_reward_episode_reward_mean",
            "episode_reward_mean",
        ),
        thesis_meaning="任务整体回报，反映训练目标的优化效果。",
    ),
    MetricSpec(
        key="reward_mean",
        display_name="Step reward mean",
        higher_is_better=True,
        aliases=(
            "collection/reward/reward_mean",
            "collection/agents/reward/reward_mean",
            "collection_reward_reward_mean",
            "collection_agents_reward_reward_mean",
            "reward_mean",
        ),
        thesis_meaning="单步平均回报，反映策略在采样过程中的即时反馈水平。",
    ),
    MetricSpec(
        key="agent_collisions",
        display_name="Agent collisions",
        higher_is_better=False,
        aliases=(
            "collection/agents/info/agent_collisions",
            "collection_agents_info_agent_collisions",
            "agent_collisions",
            "collisions",
            "collision",
        ),
        thesis_meaning="智能体碰撞或碰撞惩罚代理指标，数值越低通常表示安全性越好。",
    ),
    MetricSpec(
        key="pos_rew",
        display_name="Position/progress reward",
        higher_is_better=True,
        aliases=(
            "collection/agents/info/pos_rew",
            "collection_agents_info_pos_rew",
            "pos_rew",
            "position_reward",
        ),
        thesis_meaning="位置接近/进度相关奖励，反映无人机向目标点推进的程度。",
    ),
    MetricSpec(
        key="final_rew",
        display_name="Final reward",
        higher_is_better=True,
        aliases=(
            "collection/agents/info/final_rew",
            "collection_agents_info_final_rew",
            "final_rew",
            "success",
        ),
        thesis_meaning="终点相关奖励，可作为是否成功到达目标的代理指标。",
    ),
    MetricSpec(
        key="objective_loss",
        display_name="Actor/objective loss",
        higher_is_better=False,
        aliases=(
            "train/agents/loss_objective",
            "train_agents_loss_objective",
            "loss_objective",
            "objective_loss",
        ),
        thesis_meaning="策略目标损失，仅作为训练稳定性参考，不直接代表路径质量。",
    ),
    MetricSpec(
        key="critic_loss",
        display_name="Critic loss",
        higher_is_better=False,
        aliases=(
            "train/agents/loss_critic",
            "train_agents_loss_critic",
            "loss_critic",
            "critic_loss",
        ),
        thesis_meaning="价值函数损失，仅作为训练稳定性参考。",
    ),
)


def normalize_name(name: str) -> str:
    name = name.strip().lower()
    name = name.replace("\\", "/")
    name = re.sub(r"\.csv$|\.json$", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def parse_run_item(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"Run must be formatted as name=/path/to/run, got: {item}")
    name, path = item.split("=", 1)
    name = name.strip()
    path = Path(path.strip())
    if not name:
        raise ValueError(f"Empty run name in: {item}")
    return name, path


def read_csv_curve(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            df = pd.read_csv(path, header=None)
    except Exception:
        return None

    if df.empty:
        return None

    step_col = None
    value_col = None
    for col in df.columns:
        lc = str(col).lower()
        if step_col is None and lc in {"step", "global_step", "wall_step", "_step"}:
            step_col = col
        if value_col is None and lc in {"value", "scalar", "reward", "mean"}:
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
    if out.empty:
        return None
    return out


def read_wandb_summary(path: Path) -> dict[str, float]:
    """Read wandb-summary.json-like files if they exist."""

    values: dict[str, float] = {}
    for json_path in path.rglob("*summary*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[key] = float(value)
    return values


def discover_csv_files(run_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in run_root.rglob("*.csv"):
        candidates = {
            normalize_name(path.name),
            normalize_name(str(path.relative_to(run_root))),
        }
        for cand in candidates:
            # Prefer deeper/later matches because BenchMARL may create nested logger dirs.
            found[cand] = path
    return found


def find_metric_curve(run_root: Path, spec: MetricSpec) -> tuple[pd.DataFrame | None, str | None]:
    csv_files = discover_csv_files(run_root)
    alias_norms = {normalize_name(alias) for alias in spec.aliases}

    # Exact normalized match.
    for alias in alias_norms:
        if alias in csv_files:
            df = read_csv_curve(csv_files[alias])
            if df is not None:
                return df, str(csv_files[alias])

    # Loose contains match.
    for key, path in csv_files.items():
        if any(alias in key or key in alias for alias in alias_norms):
            df = read_csv_curve(path)
            if df is not None:
                return df, str(path)

    # Summary fallback: create one-point curve.
    summary = read_wandb_summary(run_root)
    for raw_key, value in summary.items():
        raw_norm = normalize_name(raw_key)
        if any(alias == raw_norm or alias in raw_norm or raw_norm in alias for alias in alias_norms):
            return pd.DataFrame({"step": [0], "value": [value]}), f"summary:{raw_key}"

    return None, None


def summarize_curve(df: pd.DataFrame, higher_is_better: bool) -> dict[str, float | int | None]:
    x = df["step"].to_numpy(dtype=float)
    y = df["value"].to_numpy(dtype=float)
    dy = np.diff(y) if len(y) >= 2 else np.array([0.0])

    if higher_is_better:
        best_idx = int(np.argmax(y))
        best = float(np.max(y))
    else:
        best_idx = int(np.argmin(y))
        best = float(np.min(y))

    return {
        "n_points": int(len(y)),
        "final": float(y[-1]),
        "best": best,
        "best_step": int(x[best_idx]) if len(x) else None,
        "last3_mean": float(np.mean(y[-3:])),
        "auc": float(np.trapz(y, x)) if len(y) >= 2 else 0.0,
        "volatility": float(np.std(dy)),
    }


def plot_metric(spec: MetricSpec, curves: dict[str, pd.DataFrame], out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run_name, df in curves.items():
        plt.plot(df["step"], df["value"], marker="o", label=run_name)
    plt.xlabel("step")
    plt.ylabel(spec.display_name)
    plt.title(spec.display_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{spec.key}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_markdown_report(summary: pd.DataFrame, out_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# Unified behavior evaluation report")
    lines.append("")
    lines.append("本报告使用统一日志解析脚本比较不同训练方法，避免只根据 shaped training reward 判断方法优劣。")
    lines.append("")

    for spec in METRICS:
        part = summary[summary["metric_key"] == spec.key].copy()
        if part.empty:
            continue
        lines.append(f"## {spec.display_name}")
        lines.append("")
        lines.append(spec.thesis_meaning)
        lines.append("")
        cols = ["run", "final", "best", "last3_mean", "volatility", "source"]
        lines.append(part[cols].to_markdown(index=False))
        lines.append("")

        sort_ascending = not spec.higher_is_better
        ranked = part.sort_values("last3_mean", ascending=sort_ascending)
        best_run = ranked.iloc[0]["run"]
        direction = "最高" if spec.higher_is_better else "最低"
        lines.append(f"按末期三点均值比较，`{best_run}` 在该指标上表现{direction}。")
        lines.append("")

    lines.append("## Thesis writing note")
    lines.append("")
    lines.append(
        "如果某个 reward-shaping 方法的 raw episode reward 低于 baseline，不能直接等价为方法失败。"
        "因为 reward-shaping 修改了训练目标，logged reward 中包含额外安全/平滑惩罚。"
        "更合理的写法是同时报告任务回报、碰撞/安全代理指标、位置推进指标和训练稳定性。"
    )
    lines.append("")
    report = "\n".join(lines)
    (out_dir / "behavior_metric_report.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified behavior evaluation from BenchMARL/W&B logs")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run mapping formatted as name=/path/to/run. Repeat for multiple runs.",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    runs = dict(parse_run_item(item) for item in args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for spec in METRICS:
        curves: dict[str, pd.DataFrame] = {}
        for run_name, run_root in runs.items():
            df, source = find_metric_curve(run_root, spec)
            if df is None or source is None:
                continue
            curves[run_name] = df
            row: dict[str, object] = {
                "metric_key": spec.key,
                "metric": spec.display_name,
                "higher_is_better": spec.higher_is_better,
                "run": run_name,
                "source": source,
            }
            row.update(summarize_curve(df, spec.higher_is_better))
            rows.append(row)

        if len(curves) >= 2:
            plot_metric(spec, curves, out_dir)
            print(f"[plot] {spec.key}: {', '.join(curves.keys())}")
        elif len(curves) == 1:
            print(f"[partial] {spec.key}: only found {next(iter(curves))}")
        else:
            print(f"[missing] {spec.key}")

    if not rows:
        raise SystemExit("No metrics found. Please check the run directories.")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "behavior_metric_summary.csv", index=False)

    for value_col in ["final", "best", "last3_mean", "volatility"]:
        pivot = summary.pivot_table(index="metric", columns="run", values=value_col, aggfunc="first")
        pivot.to_csv(out_dir / f"behavior_metric_pivot_{value_col}.csv")

    report = build_markdown_report(summary, out_dir)
    print(f"[saved] {out_dir / 'behavior_metric_summary.csv'}")
    print(f"[saved] {out_dir / 'behavior_metric_report.md'}")
    print("\n" + report[:1200] + ("..." if len(report) > 1200 else ""))


if __name__ == "__main__":
    main()
