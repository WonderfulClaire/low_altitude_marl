# Navigation baseline short run — 2026-03-28

## Goal
在 Colab 上验证 `vmas/navigation` 任务下的 MAPPO baseline 是否能正常启动并完成短版训练。

## Effective command
```bash
python -m benchmarl.run \
  algorithm=mappo \
  task=vmas/navigation \
  experiment.render=false \
  experiment.evaluation=false \
  experiment.max_n_frames=60000
```

## Key config snapshot
- task: `vmas/navigation`
- n_agents: `3`
- collisions: `true`
- lidar_range: `0.35`
- agent_radius: `0.1`
- max_n_frames: `60000`
- render: `false`
- evaluation: `false`

## Result
- 训练正常启动
- 短版训练顺利完成
- 日志末尾显示：`mean return = 0.8521748185157776`
- 进度：`100% 10/10`

## Interpretation
这说明正式任务 `navigation` 的 baseline 平台已经搭建成功，不再只是 `balance` 的 smoke test。

## Next step
- 固定一版 baseline 配置
- 记录输出目录和 csv 日志
- 后续补 IPPO / MADDPG 对比
- 再加入 reward shaping
