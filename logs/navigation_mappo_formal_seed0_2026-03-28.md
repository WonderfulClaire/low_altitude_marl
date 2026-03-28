# Navigation MAPPO formal baseline v1 — 2026-03-28

## Goal
在 `vmas/navigation` 任务上完成一版更正式的 MAPPO baseline 运行，并作为后续多 seed 与算法对比的起点。

## Effective command
```bash
python -m benchmarl.run \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=300000
```

## Key config snapshot
- algorithm: `mappo`
- task: `vmas/navigation`
- seed: `0`
- max_n_frames: `300000`
- render: `false`
- evaluation: `false`
- n_agents: `3`
- collisions: `true`
- lidar_range: `0.35`
- agent_radius: `0.1`

## Result
- 训练顺利完成
- 进度：`100% 50/50`
- 日志末尾：`mean return = 0.9631877541542053`

## Interpretation
这说明 `vmas/navigation` 上的 MAPPO baseline-v1 已可作为后续对比实验的固定起点。

## Next step
- 跑 `seed=1`
- 跑 `seed=2`
- 固定 MAPPO 三个 seed 的结果
- 再补 `ippo` 与 `maddpg`
