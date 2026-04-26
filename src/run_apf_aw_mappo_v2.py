from __future__ import annotations

"""Gentler APF-AW-MAPPO v2 launcher.

The first APF-AW version used relatively strong late-stage safety and
inter-agent spacing penalties. In VMAS/navigation this can reduce the original
BenchMARL episode reward even when the learned behavior becomes safer. This v2
launcher keeps the same reward-patching mechanism but uses much weaker shaping
coefficients and a longer warm-up period, so it is more suitable when the main
comparison metric is still the original episode reward.

Recommended first run:
    python src/run_apf_aw_mappo_v2.py \
        --apf-profile gentle \
        --apf-total-frames 300000 \
        algorithm=mappo task=vmas/navigation \
        experiment.render=false experiment.evaluation=false \
        experiment.max_n_frames=300000 seed=0
"""

import argparse
import re
import runpy
import sys

from run_apf_aw_mappo import APFAWConfig, patch_navigation_reward


def _infer_total_frames(remaining_args: list[str]) -> int:
    for arg in remaining_args:
        m = re.match(r"(?:\+\+)?experiment\.max_n_frames=(\d+)", arg)
        if m:
            return int(m.group(1))
    return 300_000


def _make_config(profile: str, total_frames: int) -> APFAWConfig:
    """Create a tuned APF-AW config.

    Profiles:
      - gentle: recommended; very weak constraints, long warm-up.
      - progress_only: only distance-progress shaping, safest for reward score.
      - balanced: slightly stronger than gentle, still weaker than v1.
    """

    if profile == "progress_only":
        return APFAWConfig(
            total_frames=total_frames,
            variant="full",
            progress_start=0.050,
            progress_end=0.030,
            safety_start=0.000,
            safety_end=0.000,
            smooth_start=0.000,
            smooth_end=0.000,
            coop_start=0.000,
            coop_end=0.000,
            safe_distance=0.10,
            coop_distance=0.12,
            warmup_ratio=0.70,
        )

    if profile == "balanced":
        return APFAWConfig(
            total_frames=total_frames,
            variant="full",
            progress_start=0.060,
            progress_end=0.035,
            safety_start=0.000,
            safety_end=0.025,
            smooth_start=0.000,
            smooth_end=0.0015,
            coop_start=0.000,
            coop_end=0.010,
            safe_distance=0.12,
            coop_distance=0.15,
            warmup_ratio=0.55,
        )

    if profile == "gentle":
        return APFAWConfig(
            total_frames=total_frames,
            variant="full",
            progress_start=0.050,
            progress_end=0.030,
            safety_start=0.000,
            safety_end=0.015,
            smooth_start=0.000,
            smooth_end=0.0010,
            coop_start=0.000,
            coop_end=0.005,
            safe_distance=0.10,
            coop_distance=0.12,
            warmup_ratio=0.65,
        )

    raise ValueError(f"Unknown profile: {profile}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--apf-total-frames", type=int, default=None)
    parser.add_argument(
        "--apf-profile",
        type=str,
        default="gentle",
        choices=["gentle", "progress_only", "balanced"],
    )
    known, remaining = parser.parse_known_args(sys.argv[1:])

    total_frames = known.apf_total_frames or _infer_total_frames(remaining)
    cfg = _make_config(profile=known.apf_profile, total_frames=total_frames)

    print(
        "[apf_aw_v2] profile=", known.apf_profile,
        " total_frames=", total_frames,
        " config=", cfg,
        sep="",
    )
    patch_navigation_reward(cfg)
    sys.argv = ["benchmarl.run"] + remaining
    runpy.run_module("benchmarl.run", run_name="__main__")


if __name__ == "__main__":
    main()
