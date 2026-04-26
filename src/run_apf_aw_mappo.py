from __future__ import annotations

"""APF-AW-MAPPO launcher for VMAS/navigation + BenchMARL.

This script patches the installed VMAS ``navigation.py`` scenario in the current
Python process and then launches ``benchmarl.run``. It is intentionally similar
in usage to ``src/run_reward_shaping_v1.py`` but replaces fixed reward shaping
with an adaptive potential-field weighted reward design.

Example:
    python src/run_apf_aw_mappo.py \
        algorithm=mappo \
        task=vmas/navigation \
        experiment.render=false \
        experiment.evaluation=false \
        experiment.max_n_frames=300000 \
        seed=0

Optional launcher-only arguments can be placed before the Hydra arguments:
    python src/run_apf_aw_mappo.py --apf-total-frames 300000 --apf-variant full \
        algorithm=mappo task=vmas/navigation experiment.max_n_frames=300000 seed=0
"""

import argparse
import pathlib
import re
import runpy
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class APFAWConfig:
    """Configuration injected into the VMAS navigation reward patch."""

    total_frames: int = 300_000
    variant: str = "full"  # full | no_coop | no_smooth | fixed
    progress_start: float = 0.080
    progress_end: float = 0.045
    safety_start: float = 0.000
    safety_end: float = 0.070
    smooth_start: float = 0.000
    smooth_end: float = 0.006
    coop_start: float = 0.000
    coop_end: float = 0.045
    safe_distance: float = 0.15
    coop_distance: float = 0.20
    warmup_ratio: float = 0.20


def _find_navigation_py() -> pathlib.Path:
    """Locate VMAS navigation.py under common Colab/site-packages locations."""

    candidates = [
        pathlib.Path("/usr/local/lib/python3.12/dist-packages/vmas/scenarios/navigation.py"),
        pathlib.Path("/usr/local/lib/python3.11/dist-packages/vmas/scenarios/navigation.py"),
        pathlib.Path("/usr/local/lib/python3.10/dist-packages/vmas/scenarios/navigation.py"),
        pathlib.Path("/usr/local/lib/python3.9/dist-packages/vmas/scenarios/navigation.py"),
        pathlib.Path("/usr/local/lib/python3.8/dist-packages/vmas/scenarios/navigation.py"),
    ]
    for path in candidates:
        if path.exists():
            return path

    # Fallback: try to import vmas and infer its installed path.
    try:
        import vmas  # type: ignore

        root = pathlib.Path(vmas.__file__).resolve().parent
        candidate = root / "scenarios" / "navigation.py"
        if candidate.exists():
            return candidate
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not locate VMAS navigation.py. Please verify that vmas is installed "
        "in the current Python environment."
    )


def _infer_total_frames(remaining_args: list[str]) -> int:
    """Read experiment.max_n_frames from Hydra-style args when present."""

    for arg in remaining_args:
        m = re.match(r"(?:\+\+)?experiment\.max_n_frames=(\d+)", arg)
        if m:
            return int(m.group(1))
    return APFAWConfig.total_frames


def _parse_launcher_args(argv: list[str]) -> tuple[APFAWConfig, list[str]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--apf-total-frames", type=int, default=None)
    parser.add_argument(
        "--apf-variant",
        type=str,
        default="full",
        choices=["full", "no_coop", "no_smooth", "fixed"],
        help="Ablation variant for the adaptive potential-field reward.",
    )
    parser.add_argument("--apf-safe-distance", type=float, default=0.15)
    parser.add_argument("--apf-coop-distance", type=float, default=0.20)
    parser.add_argument("--apf-warmup-ratio", type=float, default=0.20)

    known, remaining = parser.parse_known_args(argv)
    total_frames = known.apf_total_frames or _infer_total_frames(remaining)

    cfg = APFAWConfig(
        total_frames=total_frames,
        variant=known.apf_variant,
        safe_distance=known.apf_safe_distance,
        coop_distance=known.apf_coop_distance,
        warmup_ratio=known.apf_warmup_ratio,
    )
    return cfg, remaining


def patch_navigation_reward(cfg: APFAWConfig) -> pathlib.Path:
    """Patch VMAS navigation Scenario.reward with APF-AW shaping."""

    nav_path = _find_navigation_py()
    src = nav_path.read_text(encoding="utf-8")

    # Remove older patches from this project so that only one reward override is active.
    for marker in ["# [apf_aw_patch]", "# [rs_patch]"]:
        if marker in src:
            src = src[: src.index(marker)].rstrip()
            print(f"[apf_aw_mappo] removed old patch marker: {marker}")

    patch = f'''
# [apf_aw_patch]
import torch as _torch

_apf_aw_orig_reward = Scenario.reward
_apf_aw_orig_reset = Scenario.reset_world_at

_APF_TOTAL_FRAMES = int({cfg.total_frames})
_APF_VARIANT = {cfg.variant!r}
_APF_PROGRESS_START = float({cfg.progress_start})
_APF_PROGRESS_END = float({cfg.progress_end})
_APF_SAFETY_START = float({cfg.safety_start})
_APF_SAFETY_END = float({cfg.safety_end})
_APF_SMOOTH_START = float({cfg.smooth_start})
_APF_SMOOTH_END = float({cfg.smooth_end})
_APF_COOP_START = float({cfg.coop_start})
_APF_COOP_END = float({cfg.coop_end})
_APF_SAFE_DISTANCE = float({cfg.safe_distance})
_APF_COOP_DISTANCE = float({cfg.coop_distance})
_APF_WARMUP_RATIO = float({cfg.warmup_ratio})


def _apf_aw_lerp(a, b, x):
    return a + (b - a) * x


def _apf_aw_phase(self):
    if not hasattr(self, "_apf_aw_calls"):
        self._apf_aw_calls = 0
    self._apf_aw_calls += 1
    # Reward is called once per agent per environment step. We use a call-based
    # curriculum proxy rather than depending on BenchMARL internals.
    denom = max(1, _APF_TOTAL_FRAMES)
    raw = min(1.0, float(self._apf_aw_calls) / float(denom))
    if _APF_VARIANT == "fixed":
        return 1.0
    return raw


def _apf_aw_weights(self):
    phase = _apf_aw_phase(self)
    warmup = max(0.0, min(0.95, _APF_WARMUP_RATIO))
    if phase <= warmup:
        ramp = 0.0
    else:
        ramp = (phase - warmup) / max(1e-6, 1.0 - warmup)
    ramp = max(0.0, min(1.0, ramp))

    if _APF_VARIANT == "fixed":
        wp = _APF_PROGRESS_START
        ws = _APF_SAFETY_END * 0.70
        wm = _APF_SMOOTH_END * 0.70
        wc = _APF_COOP_END * 0.70
    else:
        wp = _apf_aw_lerp(_APF_PROGRESS_START, _APF_PROGRESS_END, phase)
        ws = _apf_aw_lerp(_APF_SAFETY_START, _APF_SAFETY_END, ramp)
        wm = _apf_aw_lerp(_APF_SMOOTH_START, _APF_SMOOTH_END, ramp)
        wc = _apf_aw_lerp(_APF_COOP_START, _APF_COOP_END, ramp)

    if _APF_VARIANT == "no_smooth":
        wm = 0.0
    if _APF_VARIANT == "no_coop":
        wc = 0.0
    return wp, ws, wm, wc


def _apf_aw_reset(self, env_index=None):
    out = _apf_aw_orig_reset(self, env_index)
    if not hasattr(self, "_apf_aw_prev_dist"):
        self._apf_aw_prev_dist = {{}}
        self._apf_aw_prev_action = {{}}
    if env_index is None:
        self._apf_aw_prev_dist.clear()
        self._apf_aw_prev_action.clear()
        self._apf_aw_calls = 0
    else:
        for d in (self._apf_aw_prev_dist, self._apf_aw_prev_action):
            for k, v in list(d.items()):
                if isinstance(v, _torch.Tensor) and v.ndim > 0 and env_index < v.shape[0]:
                    v = v.clone()
                    v[env_index] = 0.0
                    d[k] = v
    return out


def _apf_aw_entity_radius(entity):
    try:
        return float(getattr(getattr(entity, "shape", None), "radius", 0.0))
    except Exception:
        return 0.0


def _apf_aw_min_landmark_clearance(self, agent, ap):
    world = getattr(self, "world", None)
    landmarks = getattr(world, "landmarks", []) if world else []
    agent_radius = _apf_aw_entity_radius(agent)
    min_clearance = None
    for lm in landmarks:
        if lm is getattr(agent, "goal", None):
            continue
        lp = getattr(getattr(lm, "state", None), "pos", None)
        if lp is None or ap is None:
            continue
        lm_radius = _apf_aw_entity_radius(lm)
        clearance = _torch.linalg.vector_norm(ap - lp, dim=-1) - agent_radius - lm_radius
        min_clearance = clearance if min_clearance is None else _torch.minimum(min_clearance, clearance)
    return min_clearance


def _apf_aw_min_agent_clearance(self, agent, ap):
    world = getattr(self, "world", None)
    agents = getattr(world, "agents", []) if world else []
    agent_radius = _apf_aw_entity_radius(agent)
    min_clearance = None
    for other in agents:
        if other is agent:
            continue
        op = getattr(getattr(other, "state", None), "pos", None)
        if op is None or ap is None:
            continue
        other_radius = _apf_aw_entity_radius(other)
        clearance = _torch.linalg.vector_norm(ap - op, dim=-1) - agent_radius - other_radius
        min_clearance = clearance if min_clearance is None else _torch.minimum(min_clearance, clearance)
    return min_clearance


def _apf_aw_reward(self, agent):
    base = _apf_aw_orig_reward(self, agent)
    if not isinstance(base, _torch.Tensor):
        return base

    if not hasattr(self, "_apf_aw_prev_dist"):
        self._apf_aw_prev_dist = {{}}
        self._apf_aw_prev_action = {{}}

    reward = base.clone()
    zero = _torch.zeros_like(reward)
    wp, ws, wm, wc = _apf_aw_weights(self)

    key = getattr(agent, "name", str(id(agent)))
    ap = getattr(getattr(agent, "state", None), "pos", None)
    gp = getattr(getattr(getattr(agent, "goal", None), "state", None), "pos", None)

    # 1) Goal-attraction / progress potential field.
    progress = zero
    if ap is not None and gp is not None:
        dist = _torch.linalg.vector_norm(ap - gp, dim=-1)
        prev_dist = self._apf_aw_prev_dist.get(key)
        if prev_dist is not None and prev_dist.shape == dist.shape:
            progress = prev_dist - dist
        self._apf_aw_prev_dist[key] = dist.detach().clone()

    # 2) Static obstacle safety field.
    safety_penalty = zero
    min_landmark_clearance = _apf_aw_min_landmark_clearance(self, agent, ap)
    if min_landmark_clearance is not None:
        thr = _torch.as_tensor(_APF_SAFE_DISTANCE, dtype=reward.dtype, device=reward.device)
        safety_penalty = _torch.relu(thr - min_landmark_clearance)

    # 3) Inter-agent cooperative spacing field.
    coop_penalty = zero
    min_agent_clearance = _apf_aw_min_agent_clearance(self, agent, ap)
    if min_agent_clearance is not None:
        thr = _torch.as_tensor(_APF_COOP_DISTANCE, dtype=reward.dtype, device=reward.device)
        coop_penalty = _torch.relu(thr - min_agent_clearance)

    # 4) Action smoothness field.
    smooth_penalty = zero
    act = getattr(getattr(agent, "action", None), "u", None)
    if act is not None:
        prev_action = self._apf_aw_prev_action.get(key)
        if prev_action is not None and prev_action.shape == act.shape:
            smooth_penalty = _torch.linalg.vector_norm(act - prev_action, dim=-1)
        self._apf_aw_prev_action[key] = act.detach().clone()

    shaped = reward + wp * progress - ws * safety_penalty - wm * smooth_penalty - wc * coop_penalty
    return shaped


Scenario.reward = _apf_aw_reward
Scenario.reset_world_at = _apf_aw_reset
print(
    "[apf_aw_patch] applied: "
    f"variant={{_APF_VARIANT}}, total_frames={{_APF_TOTAL_FRAMES}}, "
    f"safe_distance={{_APF_SAFE_DISTANCE}}, coop_distance={{_APF_COOP_DISTANCE}}, "
    f"warmup_ratio={{_APF_WARMUP_RATIO}}"
)
'''

    src = src + "\n\n" + patch + "\n"
    nav_path.write_text(src, encoding="utf-8")

    pycache_dir = nav_path.parent / "__pycache__"
    if pycache_dir.exists():
        for pyc in pycache_dir.glob("navigation.cpython-*.pyc"):
            try:
                pyc.unlink()
            except Exception:
                pass

    print(f"[apf_aw_mappo] patched {nav_path}")
    return nav_path


def main() -> None:
    cfg, benchmarl_args = _parse_launcher_args(sys.argv[1:])
    patch_navigation_reward(cfg)
    sys.argv = ["benchmarl.run"] + benchmarl_args
    runpy.run_module("benchmarl.run", run_name="__main__")


if __name__ == "__main__":
    main()
