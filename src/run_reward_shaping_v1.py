from __future__ import annotations

import runpy
import sys
from typing import Dict

import torch


def _safe_tensor(x, like: torch.Tensor | None = None) -> torch.Tensor | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if like is not None:
        return torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return torch.as_tensor(x)


def patch_navigation_reward() -> None:
    import vmas.scenarios.navigation as navigation

    Scenario = navigation.Scenario
    original_reward = Scenario.reward
    original_reset_world_at = Scenario.reset_world_at

    def _get_agent_key(agent) -> str:
        return getattr(agent, "name", str(id(agent)))

    def _distance_to_goal(agent):
        goal = getattr(agent, "goal", None)
        agent_pos = getattr(getattr(agent, "state", None), "pos", None)
        goal_pos = getattr(getattr(goal, "state", None), "pos", None)
        if agent_pos is None or goal_pos is None:
            return None
        return torch.linalg.vector_norm(agent_pos - goal_pos, dim=-1)

    def _action_tensor(agent):
        action = getattr(agent, "action", None)
        u = getattr(action, "u", None)
        if u is None:
            return None
        return u

    def _min_obstacle_distance(self, agent):
        agent_pos = getattr(getattr(agent, "state", None), "pos", None)
        if agent_pos is None:
            return None

        agent_radius = 0.0
        try:
            agent_radius = float(getattr(getattr(agent, "shape", None), "radius", 0.0))
        except Exception:
            agent_radius = 0.0

        world = getattr(self, "world", None)
        landmarks = getattr(world, "landmarks", []) if world is not None else []

        min_clearance = None
        for lm in landmarks:
            if lm is getattr(agent, "goal", None):
                continue
            lm_pos = getattr(getattr(lm, "state", None), "pos", None)
            if lm_pos is None:
                continue
            lm_radius = 0.0
            try:
                lm_radius = float(getattr(getattr(lm, "shape", None), "radius", 0.0))
            except Exception:
                lm_radius = 0.0
            center_dist = torch.linalg.vector_norm(agent_pos - lm_pos, dim=-1)
            clearance = center_dist - agent_radius - lm_radius
            min_clearance = clearance if min_clearance is None else torch.minimum(min_clearance, clearance)
        return min_clearance

    def patched_reset_world_at(self, env_index=None):
        out = original_reset_world_at(self, env_index)
        if not hasattr(self, "_reward_shaping_prev_dist"):
            self._reward_shaping_prev_dist: Dict[str, torch.Tensor] = {}
        if not hasattr(self, "_reward_shaping_prev_action"):
            self._reward_shaping_prev_action: Dict[str, torch.Tensor] = {}

        if env_index is None:
            self._reward_shaping_prev_dist.clear()
            self._reward_shaping_prev_action.clear()
        else:
            for key, value in list(self._reward_shaping_prev_dist.items()):
                if isinstance(value, torch.Tensor) and value.ndim > 0 and env_index < value.shape[0]:
                    value = value.clone()
                    value[env_index] = 0.0
                    self._reward_shaping_prev_dist[key] = value
            for key, value in list(self._reward_shaping_prev_action.items()):
                if isinstance(value, torch.Tensor) and value.ndim > 0 and env_index < value.shape[0]:
                    value = value.clone()
                    value[env_index] = 0.0
                    self._reward_shaping_prev_action[key] = value
        return out

    def patched_reward(self, agent):
        base_reward = original_reward(self, agent)
        reward = base_reward.clone() if isinstance(base_reward, torch.Tensor) else base_reward

        if not hasattr(self, "_reward_shaping_prev_dist"):
            self._reward_shaping_prev_dist = {}
        if not hasattr(self, "_reward_shaping_prev_action"):
            self._reward_shaping_prev_action = {}

        key = _get_agent_key(agent)
        dist = _distance_to_goal(agent)
        action_u = _action_tensor(agent)
        min_clearance = _min_obstacle_distance(self, agent)

        if isinstance(reward, torch.Tensor):
            zero = torch.zeros_like(reward)
        else:
            return base_reward

        # progress reward: closer to goal than previous step
        progress_bonus = zero
        if dist is not None:
            prev_dist = self._reward_shaping_prev_dist.get(key)
            if prev_dist is not None and isinstance(prev_dist, torch.Tensor) and prev_dist.shape == dist.shape:
                progress = prev_dist - dist
                progress_bonus = 0.3 * progress
            self._reward_shaping_prev_dist[key] = dist.detach().clone()

        # smoothness penalty: discourage abrupt action changes
        smooth_penalty = zero
        if action_u is not None:
            prev_action = self._reward_shaping_prev_action.get(key)
            if prev_action is not None and isinstance(prev_action, torch.Tensor) and prev_action.shape == action_u.shape:
                action_delta = torch.linalg.vector_norm(action_u - prev_action, dim=-1)
                smooth_penalty = 0.05 * action_delta
            self._reward_shaping_prev_action[key] = action_u.detach().clone()

        # risk penalty: penalize being too close to obstacles
        risk_penalty = zero
        if min_clearance is not None:
            threshold = _safe_tensor(0.15, like=min_clearance)
            risk_penalty = 0.2 * torch.relu(threshold - min_clearance)

        shaped_reward = reward + progress_bonus - smooth_penalty - risk_penalty
        return shaped_reward

    Scenario.reset_world_at = patched_reset_world_at
    Scenario.reward = patched_reward
    print("[reward_shaping_v1] Patched vmas.scenarios.navigation.Scenario.reward")


def main() -> None:
    patch_navigation_reward()
    sys.argv = ["benchmarl.run"] + sys.argv[1:]
    runpy.run_module("benchmarl.run", run_name="__main__")


if __name__ == "__main__":
    main()
