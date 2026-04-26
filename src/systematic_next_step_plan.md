# Systematic next-step plan: stop blind reward-shaping trials

## Why the current APF-AW direction is risky

The current APF-AW launcher modifies the environment reward itself. Therefore the logged `episode_reward_mean` is no longer strictly comparable with the baseline MAPPO result obtained under the original VMAS/navigation reward.

This creates two problems:

1. A lower shaped reward does not necessarily mean the learned policy is worse.
2. If the thesis uses `episode_reward_mean` as the main metric, adding negative safety/smoothness penalties can mechanically lower the reward even when behavior becomes safer.

So the next step should not be another one-by-one weight trial.

## Better strategy

The project should switch from "try another reward weight" to a two-layer evaluation pipeline:

### Layer 1: Training objective

Keep training variants simple:

- MAPPO baseline
- fixed reward shaping v1
- optional progress-only shaping

Avoid strong negative penalties in the training reward unless there is a separate original-reward evaluation.

### Layer 2: Behavior evaluation

Evaluate all trained policies with additional path-planning metrics:

- original episode reward
- success/final reward proxy
- collision indicator
- path progress / position reward
- collision rate
- trajectory smoothness if actions are available

This lets the thesis argue that even if default reward is similar, the method improves safety/smoothness.

## Recommended immediate action

Run a clean baseline again in the current Colab environment using the same seed and 300k frames. This checks whether the old reported baseline 0.963 is reproducible under the current package versions and runtime.

If the current baseline is also around 0.7, then the APF-AW run is not actually bad; the old 0.963 may come from a different setup.

If the current baseline is still around 0.96, then abandon strong APF-AW penalties for the final thesis and use the fixed reward-shaping result as a negative/limited result, while focusing on behavior metrics and a conservative progress-only method.

## Thesis-safe conclusion pattern

If reward shaping does not beat baseline in raw reward, write the experiment honestly:

- baseline optimizes the original VMAS/navigation reward most directly;
- fixed shaping and APF-AW introduce safety/smoothness preferences, changing the optimization target;
- raw reward may decrease, but this reveals the trade-off between task completion reward and behavior constraints;
- future work should separate training reward from evaluation metrics and perform multi-objective evaluation.

This is safer and more academically credible than repeatedly tuning coefficients until one seed looks better.
